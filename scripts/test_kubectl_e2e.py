#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
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

"""End-to-end test for kubectl platform: kernel generation + optimization.

This mirrors the standard e2e_test.py and run_opt_manager.py workflows
but uses ``platform="kubectl"`` so that GPU operations run on a remote
Kubernetes pod while LLM calls happen locally.

Requirements:
    - Python >= 3.10
    - pip install -e .  (KernelAgent package)
    - kubectl access to a GPU pod
    - ANTHROPIC_API_KEY set

Usage:
    # 1. Switch to the right kube context
    kubectl config use-context prod-lva1-k8s-2

    # 2. Set env vars
    export KUBECTL_POD_NAME=<pod-name>
    export KUBECTL_NAMESPACE=<namespace>
    export ANTHROPIC_API_KEY=<key>

    # 3a. Test kernel generation (TritonKernelAgent)
    python scripts/test_kubectl_e2e.py generate

    # 3b. Test kernel optimization (OptimizationManager)
    python scripts/test_kubectl_e2e.py optimize

    # 3c. Run both
    python scripts/test_kubectl_e2e.py all
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# =========================================================================
# Test 1: Kernel Generation via TritonKernelAgent
# =========================================================================

def test_generate():
    """Generate a Triton kernel using TritonKernelAgent with kubectl platform.

    This is the same workflow as e2e_test.py but the agent's verification
    workers run tests on the GPU pod via kubectl.
    """
    from triton_kernel_agent import TritonKernelAgent
    from triton_kernel_agent.platform.kubectl import KubectlConfig

    print("=" * 80)
    print("KERNEL GENERATION (TritonKernelAgent + kubectl)")
    print("=" * 80)

    agent = TritonKernelAgent(
        num_workers=1,
        max_rounds=1,
        model_name="claude-sonnet-4-20250514",
        kubectl_config=KubectlConfig.from_env(),
    )

    problem_description = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

batch_size = 16
dim = 16384

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []
"""

    print("\nProblem: element-wise sigmoid (16 x 16384)")
    print("Generating kernel...\n")

    start = time.time()
    result = agent.generate_kernel(problem_description, test_code=None)
    elapsed = time.time() - start

    print(f"\nGeneration completed in {elapsed:.1f}s")

    if result["success"]:
        print(f"\nSUCCESS: Worker {result['worker_id']} solved in {result['rounds']} rounds")
        print(f"Session: {result['session_dir']}")
        print("\n--- Generated kernel (first 30 lines) ---")
        for i, line in enumerate(result["kernel_code"].splitlines()[:30], 1):
            print(f"  {i:3d} | {line}")
        print("--- end ---")
    else:
        print(f"\nFAILED: {result['message']}")
        print(f"Session: {result['session_dir']}")

    agent.cleanup()
    return result["success"]


# =========================================================================
# Test 2: Kernel Optimization via OptimizationManager
# =========================================================================

# A simple initial kernel to optimize (correct but unoptimized sigmoid)
_INITIAL_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)

def kernel_function(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _sigmoid_kernel[grid](x, out, n, BLOCK_SIZE=1024)
    return out
'''

_PROBLEM_PY = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

batch_size = 16
dim = 16384

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []
'''

_TEST_PY = '''
import torch
import sys
sys.path.insert(0, ".")
from kernel import kernel_function
from problem import Model, get_inputs, get_init_inputs

def test_kernel():
    device = "cuda"
    dtype = torch.bfloat16

    model = Model(*get_init_inputs()).to(device).to(dtype)
    inputs = [
        x.to(device).to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x.to(device) if isinstance(x, torch.Tensor)
        else x
        for x in get_inputs()
    ]

    with torch.no_grad():
        ref = model(*inputs)
    out = kernel_function(*inputs)

    if torch.allclose(ref, out, rtol=1e-2, atol=1e-2):
        print("PASS")
        return True
    else:
        diff = (ref - out).abs().max().item()
        print(f"FAIL: max diff = {diff}")
        return False

if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
'''


def test_optimize():
    """Optimize a Triton kernel using OptimizationManager with kubectl platform.

    This is the same workflow as run_opt_manager.py but with
    ``platform="kubectl"`` so benchmarking and verification happen on
    the GPU pod.
    """
    import tempfile

    from triton_kernel_agent.opt_manager import OptimizationManager
    from triton_kernel_agent.platform.kubectl import KubectlConfig

    print("=" * 80)
    print("KERNEL OPTIMIZATION (OptimizationManager + kubectl)")
    print("=" * 80)

    kubectl_config = KubectlConfig.from_env()
    print(f"  Pod: {kubectl_config.pod_name}")
    print(f"  Namespace: {kubectl_config.namespace}")

    with tempfile.TemporaryDirectory(prefix="kubectl_opt_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Write problem + test files
        problem_file = tmpdir / "problem.py"
        problem_file.write_text(_PROBLEM_PY)
        (tmpdir / "test.py").write_text(_TEST_PY)

        log_dir = tmpdir / "logs"

        print(f"  Problem: element-wise sigmoid (16 x 16384)")
        print(f"  Strategy: beam_search (2 workers, 2 rounds)")
        print(f"  Log dir: {log_dir}")
        print()

        manager = OptimizationManager(
            strategy="beam_search",
            num_workers=2,
            max_rounds=2,
            log_dir=log_dir,
            openai_model="claude-sonnet-4-20250514",
            high_reasoning_effort=False,
            platform="kubectl",
            kubectl_config=kubectl_config,
            strategy_config={"num_top_kernels": 1, "num_bottlenecks": 2},
        )

        print("Running optimization...\n")
        start = time.time()

        result = manager.run_optimization(
            initial_kernel=_INITIAL_KERNEL,
            problem_file=problem_file,
            test_code=_TEST_PY,
            max_rounds=2,
        )

        elapsed = time.time() - start
        print(f"\nOptimization completed in {elapsed:.1f}s")

        if result["success"]:
            print(f"\nSUCCESS")
            print(f"  Best time:       {result['best_time_ms']:.4f} ms")
            print(f"  PyTorch eager:   {result.get('pytorch_baseline_ms', 'N/A')} ms")
            print(f"  Initial kernel:  {result.get('initial_kernel_time_ms', 'N/A')} ms")
            print(f"  Total rounds:    {result['total_rounds']}")
            print(f"  Top kernels:     {len(result['top_kernels'])}")

            if result["top_kernels"]:
                print("\n  Top kernels:")
                for i, k in enumerate(result["top_kernels"][:3], 1):
                    print(f"    {i}. {k['time_ms']:.4f} ms (gen {k['generation']})")
        else:
            print(f"\nFAILED")
            if result.get("error"):
                print(f"  Error: {result['error']}")

    return result["success"]


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test for kubectl platform"
    )
    parser.add_argument(
        "mode",
        choices=["generate", "optimize", "all"],
        help="Which test to run",
    )
    args = parser.parse_args()

    # Validate env
    if not os.environ.get("KUBECTL_POD_NAME") and not os.environ.get(
        "KUBECTL_LABEL_SELECTOR"
    ):
        print("ERROR: Set KUBECTL_POD_NAME or KUBECTL_LABEL_SELECTOR")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    results = {}

    if args.mode in ("generate", "all"):
        results["generate"] = test_generate()

    if args.mode in ("optimize", "all"):
        results["optimize"] = test_optimize()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
