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

"""Phase 0 gate 2 probe: can a real FlagGems operator's host logic be captured?

Runs the operator's Python implementation under ``torch.compile`` with a backend that records
the FX graph and returns ``gm.forward`` unchanged, so a successful capture still produces
correct results. Reports, per case, whether a graph was produced, which nodes it contains and
crucially whether the Triton launch is part of it: a graph that stops at allocation and shape
arithmetic while the kernel call falls back to Python has not passed this gate.

Inputs are built by copy, not by ``torch.randn``, because the CPU-torch configuration cannot
provide a CUDA generator; the operator path itself does not need one.
"""

import argparse
import collections
import json
import traceback

import torch
import torch._dynamo as dynamo

import torch_fl  # noqa: F401  registers the flagos device
import flag_gems

from flagos_capture_shim import install as install_shim


def capture(name, fn, args, kwargs, expect_cpu):
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    dynamo.reset()
    out = {"case": name}
    try:
        compiled = torch.compile(fn, backend=backend, fullgraph=False)
        with torch.inference_mode():
            result = compiled(*args, **kwargs)
    except Exception:
        lines = traceback.format_exc().strip().splitlines()
        out["captured"] = False
        out["error"] = lines[-1][:300]
        out["flag_gems_frames"] = [
            ln.strip()[:180] for ln in lines if "flag_gems" in ln
        ][-2:]
        return out

    kinds = collections.Counter()
    for graph in graphs:
        for node in graph.graph.nodes:
            kinds[str(getattr(node, "target", node.op))] += 1
    out["captured"] = True
    out["graphs"] = len(graphs)
    out["has_triton_launch"] = any("triton" in k for k in kinds)
    out["nodes"] = sorted(kinds)
    out["matches_cpu_reference"] = bool(torch.allclose(result.cpu(), expect_cpu))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shim", action="store_true", help="install the device shim first"
    )
    args = parser.parse_args()
    if args.shim:
        install_shim(verbose=True)

    device = torch.device("flagos:0")
    results = {"torch": torch.__version__, "torch_version_cuda": torch.version.cuda}

    a_cpu = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64) / 1000.0
    b_cpu = (torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64) % 7) + 1.0
    results["mul"] = capture(
        "mul", flag_gems.mul, (a_cpu.to(device), b_cpu.to(device)), {}, a_cpu * b_cpu
    )

    x_cpu = (
        torch.arange(2 * 128 * 4096, dtype=torch.float32).reshape(2, 128, 4096) / 1e5
    )
    results["softmax_inner"] = capture(
        "softmax_inner",
        flag_gems.softmax,
        (x_cpu.to(device),),
        {"dim": -1},
        torch.softmax(x_cpu, dim=-1),
    )

    print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
