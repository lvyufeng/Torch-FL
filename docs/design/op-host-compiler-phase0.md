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

## Support table so far

Gate 5 asks for a support table with each failure attributed to expressions, control flow,
wrappers, ABI or codegen. Partial, three cases:

| operator | expressions | control flow | wrappers | ABI | verdict so far |
| --- | --- | --- | --- | --- | --- |
| `mul` | not yet reached | complex/broadcast branch visible in source | `pointwise_dynamic` generated callee | not yet reached | blocked by device-context shim; host logic traversable |
| `softmax` inner | not yet reached | `numel() == 0` branch, dtype selection | hand-written wrapper | not yet reached | same blocker |
| `softmax` non-inner | as above | as above | as above | not yet reached | same blocker |

No case has reached the ABI layer yet, so nothing is known about the kernel parameter
mapping in this configuration.

## Next

1. Add the device-context shim and re-run gate 2 on `mul`, then `softmax` inner and
   non-inner. Success is a captured graph containing the Triton call rather than a run that
   merely completes.
2. Add the reduction verified to use several launches, and the multi-output operator, to
   the table.
3. Find or build the CPU-torch + PrivateUse1 configuration, without which gate 3 cannot
   pass.
4. Gate 6's allocation measurement is unchanged by any of the above and can run in parallel.