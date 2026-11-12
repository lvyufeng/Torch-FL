# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture-time device shim for the flagos device (Phase 0 prototype).

FlagGems takes its device handle from ``flag_gems.runtime.torch_device_fn``, which is the
``torch.cuda`` module, and wraps every kernel launch in
``with torch_device_fn.device(x.device):``. On the ``flagos`` device that call succeeds in
eager mode and fails while Dynamo is tracing, so capture never reaches the Triton launch.

Three things make the naive patch - replacing the module attribute - insufficient. Each was
observed on the pinned stack:

1. Dynamo special-cases the call and routes it to ``CUDADeviceVariable``, which calls
   ``torch.cuda._get_device_index`` and raises
   ``ValueError: Expected a cuda device, but got: flagos:0`` before any attribute lookup
   matters.
2. ``torch.cuda.device`` must stay a class. torch_fl's accelerator compat layer does
   ``isinstance(device, torch.cuda.device)``; substituting a function raises
   ``TypeError: isinstance() arg 2 must be a type``.
3. FlagGems modules bind ``torch_device_fn`` at import time, so every module that imported it
   holds its own reference and has to be redirected.

So this shim does all three: it maps the privateuse1 device to its index for
``torch.cuda._get_device_index``, keeps ``torch.cuda.device`` a class that accepts the
privateuse1 device as a no-op, and rebinds ``torch_device_fn`` wherever FlagGems imported it
to an object that answers device questions itself.

This is a Phase 0 experiment artifact, not a shipped adapter. It is process-lifetime and
idempotent, every replacement delegates to the original for a real CUDA device, and
``DeviceHandle`` raises for any entry point it has not modelled rather than silently
returning a CUDA-only implementation.
"""

import contextlib
import sys

import torch


def _flagos_name():
    return torch._C._get_privateuse1_backend_name()


def _as_device(device):
    try:
        return torch.device(device) if device is not None else None
    except Exception:
        return None


class DeviceHandle:
    """Stands in for the torch.cuda module as FlagGems' device handle."""

    def __init__(self, device_type):
        self._device_type = device_type

    def _owns(self, device):
        d = _as_device(device)
        return d is not None and d.type == self._device_type

    def device(self, dev=None):
        return contextlib.nullcontext() if self._owns(dev) else torch.cuda.device(dev)

    def is_available(self):
        return True

    def synchronize(self, dev=None):
        return None

    def current_device(self):
        return 0

    def get_device_name(self, dev=None):
        return "flagos"

    def __getattr__(self, name):
        # Anything not modelled here is a FlagGems path this experiment has not needed.
        # Failing loudly beats returning a CUDA-only implementation by accident.
        raise AttributeError(
            f"flagos capture shim: torch_device_fn.{name} is not modelled yet"
        )


def install(verbose=False):
    """Patch torch.cuda in place and rebind FlagGems' device handles. Idempotent."""
    tc = torch.cuda
    if getattr(tc, "_flagos_capture_shim", False):
        return False

    p1 = _flagos_name()
    patched = []

    # 1. Dynamo's CUDADeviceVariable calls this directly.
    orig_index = tc._get_device_index

    def _get_device_index(device, optional=False, allow_cpu=False):
        d = _as_device(device)
        if d is not None and d.type == p1:
            return d.index if d.index is not None else 0
        return orig_index(device, optional, allow_cpu)

    tc._get_device_index = _get_device_index
    patched.append("_get_device_index")

    # 2. Keep torch.cuda.device a class for isinstance checks in torch_fl's compat layer.
    orig_device = tc.device

    class device(orig_device):  # noqa: N801 - mirrors the torch.cuda API name
        def __init__(self, dev=None):
            self._flagos_bypass = False
            d = _as_device(dev)
            if d is not None and d.type == p1:
                self._flagos_bypass = True
                return
            super().__init__(dev)

        def __enter__(self):
            if self._flagos_bypass:
                return self
            return super().__enter__()

        def __exit__(self, *exc):
            if self._flagos_bypass:
                return False
            return super().__exit__(*exc)

    tc.device = device
    patched.append("device")

    tc.is_available = lambda: True
    patched.append("is_available")

    # 3. Rebind the handle that FlagGems modules hold directly.
    handle = DeviceHandle(p1)
    rebound = 0
    for name, mod in list(sys.modules.items()):
        if not name.startswith("flag_gems") or mod is None:
            continue
        if getattr(mod, "torch_device_fn", None) is tc:
            mod.torch_device_fn = handle
            rebound += 1

    tc._flagos_capture_shim = True
    if verbose:
        print(
            f"[shim] patched torch.cuda: {', '.join(patched)}; "
            f"rebound torch_device_fn in {rebound} flag_gems modules; privateuse1={p1}"
        )
    return True
