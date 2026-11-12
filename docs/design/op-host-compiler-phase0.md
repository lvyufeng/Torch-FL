# Phase 0 record: contract and capture probing

Companion to `docs/design/op-host-compiler.md` (proposed in #823). This records what
Phase 0 has actually established, with the commands and versions that produced it. Nothing
here is an implementation result; gates 1, 3, 4, 6 have not been run.

## Gate 0 - frozen experiment contract

Recorded before any measurement, so later numbers are comparable.

| item | value |
| --- | --- |
| Device | NVIDIA H100 80GB HBM3, driver 580.126.20 |
| Host | 8x H100, container `lvyufeng-torch-fl` |
| Python | 3.12.14 |
| torch | 2.10.0+cu130 (CUDA torch present) |
| CUDA runtime | 13.0 |
| FlagGems | 5.4.0 |
| Triton | 3.6.0 |
| FlagTree | 3.6.0 |
| torch_fl build | FlagGems-primary config, `backends_cuda.conf`, boxing available |
| Measurement method | CPU-submission timing with an explicit synchronise outside the timed region; synchronised end-to-end reported separately. p50 over repeated processes, tolerance fixed at the start of each measurement. |
| Workload | The four categories from the design: pointwise, inner and non-inner softmax, a reduction verified to use several launches, and a multi-output operator. Concrete sizes are pinned per gate. |

**Gap that has to be stated up front:** this host has a CUDA torch build. The design's gate
3 requires CPU-torch + PrivateUse1, which is not what was measured here. Every capture
result below is therefore valid for the CUDA-torch configuration and must be re-run on the
supported configuration before gate 3 can pass. No CPU-torch environment was available on
this host at the time of writing.

Commands used, verbatim:

```bash
docker exec -e TORCH_FL_ENV=cosmos-torchfl-h100 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  lvyufeng-torch-fl bash -c 'export PATH=<conda>/bin:$PATH; export LD_LIBRARY_PATH=<conda>/lib:$LD_LIBRARY_PATH; python /tmp/capture_probe.py'
```

## What the routed callables actually are

Read from the installed wheel (`torch_fl/configs/backends_cuda.conf`) and from FlagGems
5.4.0 on this host:

| operator | route | Python callable | shape |
| --- | --- | --- | --- |
| `mul.Tensor` | `flaggems` | `flag_gems.ops.mul.mul(A, B, *, out=None)` | plain function; branches on complex dtype, then calls a `pointwise_dynamic`-generated `mul_broadcast_func` |
| `softmax` | `flaggems` (default overload) | `flag_gems.ops.softmax.softmax(self, dim, half_to_float=False)` | plain function; has a `numel() == 0` branch, dtype selection and a dim assertion |
| `sum` | `flaggems` | `flag_gems.ops.sum.sum` | not probed yet |
| `max` / `max.dim` | `flaggems` | `flag_gems.ops.max.max` | not probed yet |

Two things worth recording. First, the entry points are ordinary Python functions, so there
is host logic to compile that is not hidden inside a decorator. Second, `softmax`'s
`numel() == 0` branch is a metadata-dependent branch of exactly the kind the design says
must become a guard rather than a frozen constant.

## Gate 2 - capture probing

Method: wrap the real FlagGems callable in `torch.compile` with a capturing backend that
records the FX graph and returns `gm.forward` unchanged, so a successful capture still
produces correct results. Then repeat with `fullgraph=True` to see whether a complete
capture is possible at all. `torch._dynamo.utils.counters["graph_break"]` is read before and
after each case.

Eager execution of all three cases works on `flagos:0`; that is the baseline, and it means
the failures below are capture failures and not operator failures.

| case | eager | captured | fullgraph |
| --- | --- | --- | --- |
| `mul` 32x64 | ok, `flagos:0` | **no** | no |
| `softmax` (2,128,4096) `dim=-1` | ok | **no** | no |
| `softmax` (2,128,4096) `dim=0` | ok | **no** | no |

### Why capture stopped

Three causes, in the order Dynamo hits them.

1. **`logging.Logger` methods are not traceable** (`gb0291`). FlagGems logs at entry -
   `logger.debug("GEMS MUL")`, `logger.debug("GEMS SOFTMAX")`. Dynamo suggests
   `torch._dynamo.config.ignore_logger_methods`. This is a configuration-level fix and does
   not need an upstream change.
2. **`torch.*` op returning a non-Tensor** (`gb0208`). `torch.result_type` returns a dtype,
   which cannot be an FX graph output. Seen three times in the `mul` path. An adapter can
   accept that result as a host-computed value rather than a graph value.
3. **`ValueError: Expected a cuda device, but got: flagos:0`** - the blocker. It is raised
   from FlagGems' own launch path (`flag_gems/ops/mul.py:625` in
   `_launch_contiguous_tensor_tensor`, and `flag_gems/ops/softmax.py:324` in
   `softmax_out`), through `torch_device_fn.device(self.device)`.

   Inspected directly on this host:

   ```text
   flag_gems.runtime.torch_device_fn = <module 'torch.cuda'>
   flag_gems.device                   = 'flagos'
   ```

   FlagGems already knows the device is `flagos`, but its device-context helper is
   `torch.cuda`. Entering `torch.cuda.device(torch.device('flagos:0'))` raises, and this
   happens even with logging ignored, so it is not a consequence of cause 1.

### What this means for the design

The frontend is **not** blocked by FlagGems' decorator machinery, which was the design's
largest stated risk. `mul` traced far enough to reach the launch call, and the host
structure above the kernel (dtype branching, `out=` handling, shape arithmetic) was
traversable. The three blockers are all adapter-shaped:

- cause 1: a Dynamo configuration, owned by us;
- cause 2: handling a host-only value in the IR, owned by us;
- cause 3: a device-context shim around `torch_device_fn` during capture and execution,
  owned by us. `torch.cuda.device` is only used as a context manager here; a device-agnostic
  replacement is enough to unblock the path, and it is the same class of adaptation that
  #820 needed for the stream.

Cause 3 is also the reason the design's "capture must submit no real GPU work" requirement
needs our own launch interception rather than relying on FlagGems' path: the interception
point is exactly where FlagGems enters its device context.

## The CPU-torch configuration: how it runs, and what blocks gate 3

Gate 3 requires CPU-torch + PrivateUse1, which the first probe reported as unavailable. It
exists on this host, but not as a conda environment: it is a prepared venv plus an unpacked
CPU torch tree, and it only works when three details are right.

| piece | location |
| --- | --- |
| CPU torch 2.10.0+cpu | `/public-nvme/lvyufeng/sandbox/pwm/cpu-torch` (unpacked, no interpreter) |
| interpreter + torch_fl + FlagGems | `/public-nvme/lvyufeng/verify-rc2-cuda/venv` (python 3.12.6, torch_fl 2.10.0rc2+cuda13.3, FlagGems 5.4.0) |
| container | `lvyufeng-torch-fl`, glibc 2.39 |

Three details decide whether it works:

1. The container is required. The host's glibc is older than 2.38, and Triton's
   `libtriton.so` needs 2.38, so the venv cannot import FlagGems outside the container.
2. `LD_LIBRARY_PATH` must point at `cpu-torch/torch/lib` and nothing else. The container
   ships its own torch under `/usr/local/lib/python3.12/dist-packages/torch/lib` on the
   default `LD_LIBRARY_PATH`; resolving `libtorch_python.so` from there while the Python
   files come from the CPU wheel makes `import torch` fail with
   `AttributeError: module 'torch._C' has no attribute '_dlpack_exchange_api'`. That
   failure looks like a broken wheel and is only a library-resolution order problem.
3. `PYTHONPATH` must put `cpu-torch` before the venv's site-packages, which is what
   overrides the venv's own CUDA torch.

Working invocation:

```bash
docker exec -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e PYTHONPATH=/public-nvme/lvyufeng/sandbox/pwm/cpu-torch \
  -e LD_LIBRARY_PATH=/public-nvme/lvyufeng/sandbox/pwm/cpu-torch/torch/lib \
  lvyufeng-torch-fl bash -c '$VENV/bin/python /path/to/probe.py'
```

The host already carries environment scripts for the CUDA configurations, and they encode
the same two lessons: `/public-nvme/lvyufeng/env.sh` is the workspace entry point, and
`work/env-torch-fl.sh` strips the container's own torch out of `LD_LIBRARY_PATH` with a
comment that names the `_dlpack_exchange_api` failure and its cause. `work/with-flagos.sh`
is a one-line wrapper around it, so a CUDA run needs no hand-assembled environment:

```bash
docker exec -e TORCH_FL_ENV=cosmos-torchfl-h100 lvyufeng-torch-fl \
  /public-nvme/lvyufeng/work/with-flagos.sh python your_script.py
```

Two gaps remain there rather than in this document: `env-torch-fl.sh` defaults to
`TORCH_FL_ENV=torch-fl-210`, a conda environment that no longer exists, so `with-flagos.sh`
needs the variable overridden; and there is no equivalent script for the CPU-torch
configuration above, which is the one the gates need.

### What it establishes

| check | result |
| --- | --- |
| `torch.__version__` | `2.10.0+cpu`, `torch.version.cuda` is `None` |
| `import torch_fl` | ok, `privateuse1` backend name is `flagos` |
| FlagGems | 5.4.0, `flag_gems.device` is `flagos` |
| `torch.cuda.is_available()` | **`True`** |
| `torch_device_fn.device(flagos:0)` | **ok** |

The last two are the important ones. With CPU torch, `torch.cuda.is_available()` still
reports `True` because torch_fl aliases it, and entering `torch.cuda.device()` with a
`flagos:0` device does not raise here - unlike the CUDA-torch build, where it raised
`Expected a cuda device, but got: flagos:0`. So gate 2's blocker 3 is configuration
dependent, and the configuration gate 3 requires is the one where it does not fire. That
strengthens the case for making the CPU-torch configuration the primary experiment
environment rather than a final check.

### What blocks gate 3 today

Allocation on the device fails immediately:

```text
torch.randn(32, 64, device="flagos:0")
RuntimeError: Cannot get CUDA generator without ATen_cuda library. ...
```

The installed wheel is a **boxing build** (`_build_config.py` records
`KERNELS = ("boxing", "flaggems", "tileops", "vendor")`), so its PrivateUse1 kernels
re-dispatch to CUDA, and CPU torch has no `ATen_cuda` to re-dispatch to. Nothing about this
is a FlagGems or capture problem: the operator never starts.

Gate 3 therefore needs a target built with `FLAGOS_BUILD_BOXING=0`, so the device path has
no CUDA library to want in the first place. The next section narrows this: the operator path
itself already works, and only the RNG and `empty_like` paths need that build. That is the same property #820's stream work
established for the replay (`GetCurrentStreamForDevice` in the plugin's own registry), and
it is the reason that work matters here: under CPU torch there is no `c10::cuda` stream to
borrow, so the FlagGems launch path needs a stream from the plugin.

## Gate 3 precondition met: an operator runs end to end under CPU torch

The previous section recorded that allocation on the device fails under CPU torch with
`Cannot get CUDA generator without ATen_cuda library` and concluded that gate 3 needed a
`FLAGOS_BUILD_BOXING=0` wheel. That conclusion was too broad: the failure is specific to
paths that ask CUDA for something, not to the device itself. Separating those paths:

| operation | result under `torch 2.10.0+cpu` |
| --- | --- |
| `torch.empty(32, 64, device="flagos:0")` | ok |
| `torch.zeros(...)` | ok |
| `torch.ones(...)` | ok |
| CPU tensor `.to("flagos:0")` | ok, and `.cpu()` round-trips equal |
| `torch.empty_like(x)` | `Cannot initialize CUDA without ATen_cuda library` |
| `torch.randn(..., device="flagos:0")` | `Cannot get CUDA generator without ATen_cuda library` |

So the device allocates, accepts copies and returns correct data with no CUDA torch at all;
`randn` needs a CUDA generator and `empty_like` takes a CUDA-initialising path, and both are
narrow. With tensors built by copy instead of by `randn`:

| call | result |
| --- | --- |
| `flag_gems.mul(a, b)` on `flagos:0` | ok, equals the CPU reference |
| `a * b` through the ATen dispatcher | ok, equals the CPU reference |

**This satisfies gate 3's core requirement** - at least one operator runs end to end under
CPU-torch + PrivateUse1 - and it does so with the installed wheel, because
`backends_cuda.conf` routes `mul.Tensor` to `flaggems` rather than to boxing, so nothing in
that call wants `ATen_cuda`. The earlier `Cannot get CUDA generator` result was the RNG
path, not the operator path. A boxing-disabled wheel is still wanted for the RNG and
`empty_like` paths and for a clean contract, but it is no longer what stands between this
project and gate 3.

The one thing to keep in mind is that these tensors were produced by copy, so the measurement
is of an operator that reads device memory, not of one that also generates its own inputs.

## Gate 2 under CPU torch: the blocker changes shape

Re-running the capture probe in this configuration gives a different and more useful
failure, because the same device-context call behaves differently depending on whether
Dynamo is tracing:

| context | `torch_device_fn.device(flagos:0)` |
| --- | --- |
| eager | ok |
| inside a Dynamo-traced call | `ValueError: Expected a cuda device, but got: flagos:0` |

`mul` therefore still fails to capture, at the same place in FlagGems' launch path, but now
the difference is caused by tracing rather than by the build. `softmax` fails slightly
earlier and differently, with `Cannot initialize CUDA without ATen_cuda library`, because
tracing its dtype/branch logic asks CUDA for something the eager path does not.

That distinction matters for the shim: it is not enough to make the device context work
under eager execution, which this build already does. The shim has to hold while tracing,
and `softmax` shows that a second, unrelated CUDA-initialising call also has to be
intercepted. Both are still adapter work, but there are now two call sites to intercept
rather than one, and the capture-time behaviour is the one that has to be proven.

## Gate 2: the capture shim works, and the Triton launch is still outside the graph

The three blockers identified earlier were addressed with a capture-time device shim
(`benchmarks/host_compiler/flagos_capture_shim.py`) and re-probed with
`benchmarks/host_compiler/capture_probe.py --shim`.

The naive patch - replacing `torch.cuda.device` on the module - is not sufficient, and the
reasons are the useful part of this result:

1. **Dynamo special-cases the call.** Tracing routes `with torch_device_fn.device(x)` to
   `CUDADeviceVariable`, which calls `torch.cuda._get_device_index` directly and raises
   `ValueError: Expected a cuda device, but got: flagos:0` before any attribute lookup
   matters. Patching the module attribute changes nothing.
2. **`torch.cuda.device` must stay a class.** torch_fl's accelerator compat layer does
   `isinstance(device, torch.cuda.device)` while resolving device properties; substituting a
   plain function raises `TypeError: isinstance() arg 2 must be a type, a tuple of types, or
   a union`.
3. **FlagGems binds its handle at import time.** Every module that did
   `from flag_gems.runtime import torch_device_fn` holds its own reference, so the shim
   rebinds all of them - 297 modules on the pinned stack - rather than the single runtime
   module.

With all three in place the result changes qualitatively:

| case | before the shim | with the shim |
| --- | --- | --- |
| `mul` 32x64 | failed at FlagGems' `torch_device_fn.device(...)`, no graph | **captured, 1 graph, result equals the CPU reference** |
| `softmax` (2,128,4096) `dim=-1` | failed at the same place | captured, then failed executing the captured graph |

`mul`'s captured graph contains only `torch.empty`, `broadcast_shapes` and the output. **The
Triton launch is not in it** - it became an opaque call that Dynamo could not take into the
graph. So gate 2 is *partially* met: the host logic above the launch (allocation, shape
arithmetic, broadcasting) is captured, and the launch itself is not. By this gate's own
criterion - "a concrete launch snapshot or an opaque Python call fails this gate" - `mul`
has not passed yet. The missing piece is a `wrap_triton`-equivalent so the Triton call enters
the traced path instead of becoming a graph break.

`softmax` is a separate story. Its capture succeeds as far as Dynamo is concerned, and the
failure happens in the *generated* graph:

```text
File "<eval_with_key>.4", line 6, in forward
    out = torch.empty_like(l_self_, dtype = torch.float32)
RuntimeError: Cannot initialize CUDA without ATen_cuda library. ...
```

That is the CPU-torch `empty_like` gap already recorded above, reached through the graph's
allocation rather than through the operator's own code. So `softmax`'s host logic -
including the dtype upcast to float32 and the allocation - is captured, and what blocks it is
an allocation path that needs a CUDA-initialising call. The two operators fail for
independent reasons: `mul` needs the launch exposed, `softmax` needs `empty_like` to work
under CPU torch.

## Correction: the CPU-torch configuration is a supported, CI-tested shape

The sections above describe CPU-torch results obtained with the wheel that happened to be
installed in the venv used for probing, which is an **rc2 boxing build**
(`KERNELS = ("boxing", "flaggems", "tileops", "vendor")`). That wheel is not the shape this
configuration is meant to run, and two conclusions drawn from it have to be narrowed.

The repository's NVIDIA lane already runs CPU torch as its default. From
`.github/scripts/hooks/set_env_nvidia.sh`:

- the vendor lane installs `torch==$CPU_TORCH_VERSION` from the `whl/cpu` index, so the
  interpreter under test has no CUDA torch;
- the wheel that lane builds is the FlagGems-primary one, with boxing off;
- and its own note states that "a boxing wheel runs on the stock +cpu torch exactly as the
  vendor lane does", which is why the two lanes share the CPU-torch interpreter.

So "CPU torch cannot run an operator" was never the right reading of the earlier failure.
What the earlier probes actually measured is: an rc2 **boxing** wheel on CPU torch reports
`Cannot initialize CUDA` for `randn` and `empty_like`, and works for `empty`/`zeros`/copies
and for `flag_gems.mul`. That is a statement about a boxing wheel, not about the supported
configuration.

There is one real constraint worth recording, because it is not obvious from the wheel names:

| wheel on disk | `KERNELS` | loads on CPU torch alone |
| --- | --- | --- |
| `sandbox/wheels/...` (the probed one) | boxing, flaggems, tileops, vendor | yes, via the alias |
| `torch-fl-pr537/dist/...` | flaggems, tileops, vendor | **no** - needs `libtorch_cuda.so` |
| `torch-fl-ci-unify/worktree-05992be2/dist/...` | flaggems, tileops, vendor | **no** - needs `libtorch_cuda.so` |

Neither boxing-off wheel imports without `libtorch_cuda.so`, and the CI satisfies that by
staging the library out of the vendor torch into `FLAGOS_CUDA_ASSETS_DIR` - the same variable
`work/env-torch-fl.sh` exports on this host. So the supported CPU-torch shape is *CPU torch
wheel plus staged CUDA assets plus the boxing-off wheel*, not "no CUDA library anywhere".
Installing a boxing-off wheel into a bare CPU-torch venv fails at
`import torch_fl._C` with `libtorch_cuda.so: cannot open shared object file`.

Consequence for the gates: gate 3 has to be run in the CI-shaped environment, with the
staging step done, rather than in the ad-hoc venv used here. The capture results above are
unaffected in kind - they were obtained through the ATen dispatcher on a real `flagos`
device - but `softmax`'s `empty_like` failure should be re-checked against the boxing-off
wheel before it is treated as a property of the supported configuration.

## Why a boxing-off build here still needed libtorch_cuda

Building the current tree with `FLAGOS_BUILD_BOXING=0` on this host produced
`KERNELS = ("flaggems", "tileops", "vendor")` and still linked
`libtorch_cuda.so` and `libc10_cuda.so` into `libtorch_fl.so`:

```text
$ ldd torch_fl/lib/libtorch_fl.so | grep -E 'torch_cuda|c10_cuda'
    libtorch_cuda.so => .../torch_fl/lib/libtorch_cuda.so
    libc10_cuda.so   => .../torch_fl/lib/libc10_cuda.so
```

The cause is not the kernel set but which torch the build used. `csrc/CMakeLists.txt`
probes the torch it was configured against rather than assuming:

```cmake
set(FLAGOS_TORCH_LINKS_LIBTORCH_CUDA OFF)
get_target_property(_flagos_torch_cpu_location torch_cpu_library IMPORTED_LOCATION)
...
      set(FLAGOS_TORCH_LINKS_LIBTORCH_CUDA ON)
```

That build ran under the CUDA-torch conda environment (`cosmos-torchfl-h100`), whose
`torch_cpu_library` links CUDA, so the probe turned the dependency on even with boxing off.
CI's vendor lane builds with CPU torch as the interpreter, which is what turns it off - and
that is the whole point of that lane, since its wheel "carries no libtorch_cuda.so to
preload".

So the supported build is reproducible on this host, and the missing ingredient was the build
interpreter, not a flag:

```bash
PYTHONPATH=/public-nvme/lvyufeng/sandbox/pwm/cpu-torch \
LD_LIBRARY_PATH=/public-nvme/lvyufeng/sandbox/pwm/cpu-torch/torch/lib \
  <cpu-torch-venv>/bin/python setup.py bdist_wheel    # FLAGOS_BUILD_BOXING unset (off)
```

This also explains the wheels already on disk: the two boxing-off wheels that failed to
import are builds whose torch was a CUDA torch, so they carry the dependency this lane is
supposed to avoid.

## The working environment, and gate 2 passing for softmax

Everything above that used `verify-rc2-cuda/venv` was measured in an environment that is not
the configured one. The configured environment on this host is the conda environment whose
name carries it, `torch-fl-h100-py312`, driven through the workspace wrapper:

```bash
docker exec -e TORCH_FL_ENV=torch-fl-h100-py312 -e FLAGOS_DISABLE_CUDA_ASSETS=1 \
  lvyufeng-torch-fl /public-nvme/lvyufeng/work/with-flagos.sh python probe.py
```

`FLAGOS_DISABLE_CUDA_ASSETS=1` is required, and the reason is worth recording because the
failure does not look like a configuration problem:

```text
terminate called after throwing an instance of 'c10::ValueError'
  what(): Duplicated key 'graph_capture_record_stream_reuse' found in device-specific
  configuration parser hook registration
Exception raised from registerDeviceConfigParserHook at c10/core/AllocatorConfig.h:266
```

That environment has a CUDA torch, so it already provides `libc10_cuda.so`; the wrapper's
`FLAGOS_CUDA_ASSETS_DIR` makes the plugin preload a second staged copy, and the second
registration of the allocator-config hook aborts the process. With the preload disabled the
environment is clean:

| check | result |
| --- | --- |
| `torch.__version__` | `2.10.0+cu130`, `torch.version.cuda` is `13.0` |
| `import torch_fl` | ok, `privateuse1` is `flagos` |
| `torch.empty` / `torch.randn` on `flagos:0` | ok |
| `a * b` through the dispatcher | ok, `flagos:0`, correct value |

### Gate 2, with and without the shim

The same probe, in this environment, run both ways. The comparison is the useful part: it
settles whether the capture shim is a workaround for a broken environment or a real
requirement.

| case | without shim | with shim |
| --- | --- | --- |
| `mul` 32x64 | fails at `torch_device_fn.device(output.device)`, no graph | captured, 1 graph, result equals CPU reference, **launch not in the graph** |
| `softmax` (2,128,4096) `dim=-1` | fails at `torch_device_fn.device(self.device)`, no graph | **captured, 2 graphs, `triton_kernel_wrapper_mutation` present, result equals CPU reference** |

So the shim is required in the configured environment too - the device-context blocker is a
property of FlagGems' launch path on the `flagos` device, not of the earlier venv - and with
it **gate 2 passes for `softmax`**: a real FlagGems operator's host logic was captured with
its Triton kernel call inside the graph, and the captured path produces the same result as
the reference.

`mul` still does not pass: its graph holds `torch.empty`, `broadcast_shapes` and the output,
while the launch becomes an opaque call. The two operators differ in how the launch is
reached, which is why one lands inside the graph and the other does not - `softmax` goes
through the Triton wrapper that Dynamo can take into the graph, while `mul`'s
`pointwise_dynamic` callee launches in a way Dynamo leaves as a call.

## First performance numbers, and one path ruled out

Measured in the configured environment (`torch-fl-h100-py312`, `FLAGOS_DISABLE_CUDA_ASSETS=1`),
`softmax` on `(2, 128, 4096)` float32, median of 200 calls with the CPU clock around the
call and synchronisation outside the timed region, plus 20 synchronised round trips. Vendor
is `torch.softmax` on `cuda:0` in the same process, same shape.

| arm | CPU submission (us) | synchronised end to end (us) |
| --- | ---: | ---: |
| FlagGems Python wrapper on `flagos` | 26.12 | 32.43 |
| captured graph executed through Dynamo | **119.15** | **124.05** |
| vendor `torch.softmax` on `cuda:0` | 8.95 | 16.71 |
| `torch.empty` on `flagos` | 4.41 | 7.13 |

Two things follow, and the first one changes the design.

**Executing the captured graph through `torch.compile` is not a viable host path.** It is
4.5x slower than the Python wrapper it was supposed to replace. Every call re-enters
Dynamo's guard machinery, and the `triton_kernel_wrapper_mutation` higher-order operator
carries its own dispatch. So a captured FX graph is a good *description* of the host work and
a bad *execution mechanism*; the host program has to be a native artifact, which is what the
design already said, now with a number behind it.

**The budget in this environment.** The wrapper is 26.12 us against the vendor's 8.95, so the
host work to remove is about 17 us, of which allocation is 4.41 us. Those two together
account for almost all of the gap, which is consistent with the earlier decomposition and
keeps the arithmetic from the design intact: removing the Python wrapper and keeping a
4.4 us allocation lands near 9-10 us, and bringing the allocation down is what moves it
toward the vendor's 8.95.

## The first host program, and what its decomposition changes

A host program for `softmax` was written by hand as the shape a compiler must produce:
metadata expressions only, no baked dimensions, no baked grid, no pooled output. One program
served six shapes - `(2,128,4096)`, `(3,7,1024)`, `(64,512)`, `(1,1000)`, `(5,3,257)`,
`(8,64,64,96)` - with the grid recomputed each time (256, 21, 64, 1, 15, 32768), every result
matching the reference, all on `flagos:0`. That is the property the replay plan lacks: one
artifact, many sizes.

The guard written for it was `rank == 3`, and the program then ran correctly on rank 2, 3 and
4. The guard was over-specialised: what the program actually requires is "contiguous, dtype,
and the reduction dim is the last one", not a rank. Recording it because guard design is
where a compiler can silently destroy reuse.

The decomposition is the more important result:

| arm | CPU submission (us) |
| --- | ---: |
| FlagGems Python wrapper | 24.07 |
| parameterised host program, still in Python | 20.67 |
| **the Triton kernel launch alone, no allocation** | **15.43** |
| `torch.empty_like` on `flagos` | 3.34 |
| vendor `torch.softmax` on `cuda:0` | 8.95 |

Removing the wrapper's own work - the debug logging, the assertions, the repeated dim
normalisation, `contiguous()`, the `resize_` checks and the device context manager - is worth
only about 3.4 us. **The Triton Python launcher costs 15.43 us**, which dominates everything
else and is larger than the entire vendor operator. `20.67` is consistent with
`15.43 + 3.34` plus a little glue.

So the strategy that follows from the numbers is not "compile the host logic first". Against
the vendor's 8.95 us, the removable costs rank:

1. **the Python Triton launcher: about 11 us.** #820 already measured a direct C++ launch at
   4.33 us, so a native launch is worth roughly 11 us here.
2. **allocation: about 2 us**, from 3.34 down to a fast path's 1.0-1.5.
3. **the remaining host logic: about 3 us**, which is what host compilation addresses.

That reorders the work relative to a plain reading of the design: the kernel ABI adapter and
the allocation fast path carry most of the win, and both are reusable across every operator
and every shape class, whereas the host-program lowering is the per-operator, incremental
part. It also means the first end-to-end native prototype should be
"native launch + fast allocation + parameterised metadata", not "full IR first".

## The kernel ABI adapter: a direct launch works

The decomposition says the Python Triton launcher costs 15.43 us against the 4.33 us a
direct C++ launch costs, so the ABI path was worth prototyping first. For `softmax`'s inner
kernel the whole ABI turns out to be small:

```text
arg_names:  output_ptr, input_ptr, M, N, TILE_N: constexpr, ONE_TILE_PER_CTA: constexpr
num_warps:  16      shared: 64      block: 512
```

So two pointers and two int32s are passed per launch and the two tile parameters are baked
into the compiled kernel. The compiled kernel is reachable from FlagGems' handle without
recompiling: `softmax_kernel_inner` is a `LibEntry`, whose `kernel_cache` is a tuple holding
the Triton `CompiledKernel`, which carries `metadata` (num_warps, shared, signature) and
`function`, the `CUfunction` handle.

Launching that handle through `cuLaunchKernel` with a hand-packed parameter array works:

```text
warm ok, matches: True
```

The result is correct against the reference and stays correct over twenty consecutive
launches, so the adapter's functional half is established: the parameter mapping can be
recovered from what Triton produced, and the kernel can be submitted without Triton's Python
launcher.

**No timing number is claimed.** The measurement loop segfaults after the warm-up passes -
twenty launches are fine, and the process dies once the same call is repeated a few hundred
times in a tight loop. That points at the submission path rather than at the ABI: the
prototype takes its stream from `torch.cuda.current_stream()` and keeps the parameter array
in Python-owned memory reused across iterations, neither of which a real adapter would do.
The next step is to submit on the stream the plugin's own registry reports, with the
parameter block owned for the lifetime of the launch, and only then measure.

### Why the direct-launch timing is still not measured

Two attempts to stabilise the measurement both failed for concrete, recorded reasons.

`torch.cuda.current_stream().cuda_stream` is not usable here: in this environment `torch.cuda`
is the plugin's alias, and the current-stream object it returns has no CUDA handle to read
(`None`). That is the correct behaviour for an aliased device, not a bug.

The stream therefore has to come from the plugin's own registry - which is exactly what
`GetCurrentStreamForDevice` is for, the entry point added for the CPU-torch contract. But the
wheels installed in this environment predate it:

```text
AttributeError: .../torch_fl/lib/libflagos.so: undefined symbol: GetCurrentStreamForDevice
```

`register.cc:563` in that library is an older tree than the merged `6f765952`, so the symbol
is absent. Getting the number therefore requires installing a build of the merged tree into
this environment, or running the measurement where such a build already exists. Neither was
done here, so no timing is reported.

This also gives the ABI adapter its next two steps in order: submit on the registry's stream,
and own the parameter block for the launch's lifetime - then measure, and separately confirm
the twenty-launch success is stable over hundreds.

## Support table so far

Gate 5 asks for a support table with each failure attributed to expressions, control flow,
wrappers, ABI or codegen. Partial, three cases:

| operator | expressions | control flow | wrappers | ABI | verdict so far |
| --- | --- | --- | --- | --- | --- |
| `mul` | captured: `broadcast_shapes` is in the graph; sizes are not yet symbolic | complex/broadcast branch resolved to one path | `pointwise_dynamic` callee traced through | not reached | host logic captured; **Triton launch left the graph as an opaque call** |
| `softmax` inner | captured, including the float32 upcast | `numel() == 0` branch not exercised | hand-written wrapper traced through | not reached | **captured with the Triton call in the graph; gate 2 passes** |
| `softmax` non-inner | as above | as above | as above | not reached | not re-probed with the shim |
| reduction (multi-launch) | not probed | not probed | not probed | not reached | not probed |
| multi-output (`max.dim`) | not probed | not probed | not probed | not reached | not probed |

No case has reached the ABI layer yet, so nothing is known about the kernel parameter
mapping in this configuration. The two probed operators fail for different reasons, which is
the useful shape of this table: one needs the launch exposed to tracing, the other needs an
allocation path that does not initialise CUDA.

## Next

1. Add the device-context shim and re-run gate 2 on `mul`, then `softmax` inner and
   non-inner. Success is a captured graph containing the Triton call rather than a run that
   merely completes.
2. Add the reduction verified to use several launches, and the multi-output operator, to
   the table.
3. Find or build the CPU-torch + PrivateUse1 configuration, without which gate 3 cannot
   pass.
4. Gate 6's allocation measurement is unchanged by any of the above and can run in parallel.