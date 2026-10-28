# Real streams for boxing-enabled DCU builds

## Scope and current status

This is an **opt-in boxing-build provider**, not a native HIP implementation for
FlagGems-primary/boxing-OFF builds. It requires a matching vendor torch HIP
runtime alongside the official torch core. It is intended to address the missing
DCU Stream/Event plumbing discussed in issue #472, not to certify all FSDP/HSDP
or third-party native-view use cases.

The provider is compiled only when both `FLAGOS_ACCELERATOR=dcu` and
`FLAGOS_BUILD_BOXING=1` are selected. Other builds retain their existing stream
implementation. Python selects the provider by the compiled `_dcu_stream_info`
capability, not by a runtime environment override. The default remains boxing
OFF; this change does not claim to add multistream support to that default.

The original #627 snapshot remained draft while standard ON/OFF wheel and
device gates were outstanding. Its separate `_flagos_nccl` build blocker was
subsequently addressed by #654. The contributor-side current-main follow-up below
supersedes that old build-blocked result; it does not substitute for maintainer
review, all-platform CI or the official published-dependency cohort.

## Execution and lifetime contract

```text
flagos Stream / Event and PrivateUse1 DeviceGuard
  -> vendor CUDA-tag DeviceGuard backed by HIP
  -> one vendor per-thread/per-device stream registry and stream pool
  -> actual hipStream_t for eager, Triton and Inductor launches
```

Opaque `c10::StreamId` values are not raw HIP pointers. Conversion to/from raw
handles uses the matching `c10::hip` interfaces in a dedicated translation unit.
Only that unit consumes HIP headers; it must not include the DTK CUDA facade
headers in the same translation unit. CMake stages the selected vendor's
`c10/hip` subtree and optional generated `ATen/dtk_macros.h` without replacing
official ATen/c10 core headers. The selected headers and vendor libraries must
match; arbitrary vendor releases are not ABI-certified by this change.

The generic PyTorch Stream/Event types retain the public flagos device tag.
Contexts restore both device and per-device stream state on normal exit or an
exception. External streams are borrowed: their owner must keep them alive.
Timing is delegated to vendor events; unsupported IPC/graph-external event
options fail explicitly.

Allocations retain the vendor DataPtr's context and deleter so
`Tensor.record_stream()` can reach the allocator's actual lifetime tracking.
Event/wait orders producer and consumer execution; it does not replace
`record_stream()` when an allocation is consumed on another stream.

Clone, blocking device copies and Storage growth use the selected current
streams. They **remain blocking operations**; this is not an asynchronous-copy
performance claim. Exact TensorImpl self-copy avoids a redundant transfer, but
ordinary Storage aliases still follow normal copy semantics.

## Build and acceptance

Use separate source/build directories and isolated installations for:

1. Unmodified upstream, boxing OFF.
2. This revision, boxing OFF, without vendor torch on the loader search path.
3. This revision, boxing ON, with matching vendor device libraries.

For ON, set `FLAGOS_ACCELERATOR=dcu`, `FLAGOS_BUILD_BOXING=1`, `ROCM_PATH` to
the DTK root and `FLAGOS_VENDOR_TORCH_LIB` to the matching vendor `torch/lib`.
A direct CMake build can explicitly set `FLAGOS_DCU_HIP_INCLUDE_DIR` to that
vendor torch's include directory. OFF must not need that directory. Follow the
existing DCU installation instructions for the official core, runtime loader
and compiler dependencies. Do not replace official `libtorch_python.so` with a
vendor copy to make an import pass.

CPU regressions:

```bash
python -m pytest -q tests/unit/test_dcu_stream_contract.py \
  tests/unit/test_dcu_boxing_build_gate.py tests/unit/test_dcu_cuda_shim.py
ruff check
ruff format --check
```

Device regression (at least two DCUs, both boxing and mixed runtime routes):

```bash
python -c 'import torch_fl; assert hasattr(torch_fl._C, "_dcu_stream_info")'
python -c 'import torch_fl,pytest,sys; sys.exit(pytest.main(sys.argv[1:]))' \
  -q tests/integration/runtime/test_dcu_boxing_streams.py
```

Run against the installed wheel, not an unbuilt source package. Require zero
skips in this targeted ON gate. It checks independent pending queues, event
ordering, TLS, exception restoration, borrowed streams, allocator reuse,
current-stream clone, native derived-view device tags, compile launch,
self-copy and Storage resize. A process timeout is required because a broken
stream/query implementation can block on the deliberately held queue.

For OFF, inspect `compile_commands.json` and the dynamic symbol table: the new
provider must be absent, and the new Python `_dcu_*` methods must not be exported.
Then compare import and existing runtime tests with the unmodified OFF baseline.
Do not turn a failing baseline into a claim that the OFF runtime is validated.

## Current-main follow-up, 2026-10-26

The follow-up merges #627 head `d2b4f90` with upstream `2fa7631`, including #654
and #655. Runtime source `e999531` produced both standard wheels, including the
normal native communication extension; no target was excluded. Subsequent
changes are test isolation/import order and this documentation, not runtime
implementation changes.

The provider imports now follow the Hygon directory rename, and compile-interface
repair matches the actual `hygon` profile token. The MUSA dispatch test loads an
isolated compatibility facade against its mocked extension rather than reusing
an already-bound host DCU class. Its transitive imports see that same facade;
the installed host provider objects are not modified.

Contributor-side evidence on gfx936, DTK 26.04, glibc 2.28, Python 3.10 and
official Torch 2.10.0+cpu:

| Gate | Observed result |
|---|---|
| Standard ON and OFF wheel builds | Both complete, native communication extension included |
| OFF capability and dynamic provider-symbol isolation | Pass |
| Corrected MUSA dispatch file on installed ON / OFF wheels | 9 / 9 passed, zero skips |
| Targeted DCU contract/build/shim files | 84 passed, zero skips |
| Two-device real-stream suite, fresh compile caches, eager fallback disabled | 13 passed, zero skips |
| Four-rank 2x2 HYBRID_SHARD, root-only inference under inference_mode | 3 iterations, all ranks match CPU |
| Four-rank 2x2 HYBRID_SHARD, nested FSDP inference | 3 iterations, all ranks match CPU |
| Four-rank HYBRID_SHARD training | 3 forward/backward/SGD steps and full-parameter CPU comparison, all ranks pass |

Those HSDP probes use Linear/ReLU/Linear and `use_orig_params=True`. Loaded
libraries and communicator logs confirm native RCCL, with staged Gloo disabled.
They are not a production-model, long-run, AMP, checkpoint, multi-node or general
ProcessGroup/Work lifetime certification.

This host uses the compatible portable FlagTree
`0.7.0rc2+hcu3.6.glibc228.compat1.git76201fd9` and FlagGems
`5.5.0.dev0+g1285f5d6a`; the result is not certification of the official release
resolver/dependency pins. Shared Base/Release packages and activation entries were
not changed.

For current-main ON validation, explicitly select the official
`backends_hygon_boxing.conf`, as the #655 boxing registration contract requires.
OFF uses the default `backends_hygon.conf`. Do not confuse a default-conf negative
control with the boxing lane, edit generated conf files, or silently repin ops.
Preload `torch_fl` before Torch/pytest plugins cache the CPU wheel's CUDA hooks.
Expose the real DTK CUDA tool directory and supported GCC toolchain on PATH;
Inductor's CUDA system-info probe invokes `nvcc` even for this HIP-backed runtime.
The two initially failing compile gates passed after restoring those documented
toolchain paths, without a fake compiler, core monkeypatch or eager fallback.

## Not included

- No global shared-Storage bridge replacement or cross-tag differentiable API.
- No ProcessGroup/Work ownership rewrite and no HSDP certification.
- No new boxing-OFF allocator or stream provider.
- No profiler fixes, default-route changes, or Apex/TileOPs consumer migration.
- No throughput claim or guarantee that all runtime host waits are eliminated.

Native communication streams and any native borrowed Tensor views still need
separate end-to-end ownership and alias validation before using this change as
a complete FSDP/HSDP fix. Scope acceptance from maintainers is also outstanding.
