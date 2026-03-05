#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

"""Run kernel optimization with kubectl platform.

Only requires problem.py and input_kernel.py in the kernel directory.
Test code is generated automatically from the problem definition.

Usage:
    export KUBECTL_POD_NAME=<pod>
    export KUBECTL_NAMESPACE=<ns>
    export KUBECTL_CONTAINER=primary
    export ANTHROPIC_API_KEY=<key>

    python scripts/run_kubectl_optimize.py \
        --kernel-dir sample_kernels/04_Matrix_vector_multiplication \
        --max-rounds 3
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Auto-generated test code template
# ---------------------------------------------------------------------------

_TEST_TEMPLATE = '''
import inspect
import torch
import sys
sys.path.insert(0, ".")
from kernel import kernel_function
from problem import Model, get_inputs, get_init_inputs

def test_kernel():
    device = "cuda"
    dtype = torch.bfloat16

    # Build reference model
    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (tuple, list)):
        init_inputs = [init_inputs]
    model = Model(*init_inputs).to(device).to(dtype)

    # Prepare inputs
    raw_inputs = get_inputs()
    if not isinstance(raw_inputs, (tuple, list)):
        raw_inputs = [raw_inputs]
    inputs = [
        x.to(device).to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x.to(device) if isinstance(x, torch.Tensor)
        else x
        for x in raw_inputs
    ]

    # Reference output
    with torch.no_grad():
        ref = model(*inputs)

    # Detect if kernel_function needs model parameters (weight, bias, etc.)
    _MODEL_PARAM_NAMES = {"weight", "w", "bias", "conv_bias", "eps",
                          "kernel_size", "stride", "padding", "dilation",
                          "groups", "num_groups", "normalized_shape"}
    needs_model = False
    kernel_params = []
    has_var_positional = False
    try:
        sig = inspect.signature(kernel_function)
        kernel_params = [
            name for name, p in sig.parameters.items()
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD)
        ]
        has_var_positional = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in sig.parameters.values()
        )
        if _MODEL_PARAM_NAMES.intersection(kernel_params):
            needs_model = True
        if not needs_model and has_var_positional:
            try:
                src = inspect.getsource(kernel_function)
                needs_model = any(kw in src for kw in ("weight", "w.shape", "kernel_size"))
            except (OSError, TypeError):
                pass
    except Exception:
        pass

    if needs_model:
        # Extract model parameters and build kernel args
        model_params = {}
        all_weights = []
        _CONV = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                 torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)
        _NORM = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                 torch.nn.LayerNorm, torch.nn.GroupNorm,
                 torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d)
        for _, m in model.named_modules():
            if isinstance(m, (*_CONV, torch.nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None:
                    all_weights.append(m.weight)
                    model_params.setdefault("weight", m.weight)
                    model_params.setdefault("w", m.weight)
                    if getattr(m, "bias", None) is not None:
                        model_params.setdefault("conv_bias", m.bias)
                        model_params.setdefault("bias", m.bias)
                    for attr in ("stride", "padding", "dilation", "output_padding"):
                        val = getattr(m, attr, None)
                        if val is not None:
                            model_params.setdefault(attr, val)
                    if hasattr(m, "groups"):
                        model_params.setdefault("groups", m.groups)
            elif isinstance(m, _NORM):
                if getattr(m, "weight", None) is not None:
                    model_params.setdefault("weight", m.weight)
                if getattr(m, "bias", None) is not None:
                    model_params.setdefault("bias", m.bias)
                if hasattr(m, "eps"):
                    model_params["eps"] = m.eps

        if has_var_positional and all_weights:
            # *args style: pass inputs + weights positionally, config as kwargs
            pos_args = list(inputs) + list(all_weights)
            config_kwargs = {}
            for k, v in model_params.items():
                if k not in ("weight", "w", "bias", "conv_bias"):
                    if isinstance(v, (tuple, list)) and len(v) >= 1 and all(e == v[0] for e in v):
                        v = v[0]
                    config_kwargs[k] = v
            out = kernel_function(*pos_args, **config_kwargs)
        else:
            # Named-parameter style
            bound = {}
            pos_idx = 0
            for pname in kernel_params:
                if pname in model_params:
                    v = model_params[pname]
                    if isinstance(v, (tuple, list)) and len(v) >= 1 and all(e == v[0] for e in v):
                        v = v[0]
                    bound[pname] = v
                elif pos_idx < len(inputs):
                    bound[pname] = inputs[pos_idx]
                    pos_idx += 1
            out = kernel_function(**bound)
    else:
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


def main():
    parser = argparse.ArgumentParser(
        description="Optimize Triton kernels via kubectl GPU pod"
    )
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        required=True,
        help="Directory with input_kernel.py (or input.py) and problem.py",
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="LLM model for optimization",
    )
    parser.add_argument(
        "--strategy",
        default="beam_search",
        choices=["beam_search", "greedy"],
    )
    args = parser.parse_args()

    # Validate env
    if not os.environ.get("KUBECTL_POD_NAME") and not os.environ.get(
        "KUBECTL_LABEL_SELECTOR"
    ):
        print("ERROR: Set KUBECTL_POD_NAME or KUBECTL_LABEL_SELECTOR")
        sys.exit(1)

    # Find kernel files
    kernel_dir = args.kernel_dir.resolve()
    problem_file = kernel_dir / "problem.py"

    # Try input_kernel.py first, then input.py
    kernel_file = kernel_dir / "input_kernel.py"
    if not kernel_file.exists():
        kernel_file = kernel_dir / "input.py"

    for f, name in [
        (problem_file, "problem.py"),
        (kernel_file, "input_kernel.py / input.py"),
    ]:
        if not f.exists():
            print(f"ERROR: {name} not found in {kernel_dir}")
            sys.exit(1)

    kernel_code = kernel_file.read_text()
    test_code = _TEST_TEMPLATE

    # Log dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (
        Path.cwd()
        / "triton_kernel_logs"
        / f"kubectl_opt_{kernel_dir.name}_{timestamp}"
    )

    print("=" * 80)
    print("Kernel Optimization (kubectl platform)")
    print("=" * 80)
    print(f"  Kernel dir:  {kernel_dir}")
    print(f"  Kernel file: {kernel_file.name}")
    print(f"  Problem:     {problem_file}")
    print(f"  Strategy:    {args.strategy}")
    print(f"  Workers:     {args.num_workers}")
    print(f"  Max rounds:  {args.max_rounds}")
    print(f"  Model:       {args.model}")
    print(f"  Log dir:     {log_dir}")
    print()

    from triton_kernel_agent.opt_manager import OptimizationManager
    from triton_kernel_agent.platform.kubectl import KubectlConfig

    kubectl_config = KubectlConfig.from_env()
    print(f"  Pod:         {kubectl_config.pod_name}")
    print(f"  Namespace:   {kubectl_config.namespace}")
    print()

    # Strategy config
    if args.strategy == "beam_search":
        strategy_config = {
            "num_top_kernels": 1,
            "num_bottlenecks": args.num_workers,
        }
    else:
        strategy_config = {}

    manager = OptimizationManager(
        strategy=args.strategy,
        num_workers=args.num_workers,
        max_rounds=args.max_rounds,
        log_dir=log_dir,
        openai_model=args.model,
        high_reasoning_effort=True,
        platform="kubectl",
        kubectl_config=kubectl_config,
        strategy_config=strategy_config,
    )

    result = manager.run_optimization(
        initial_kernel=kernel_code,
        problem_file=problem_file,
        test_code=test_code,
        max_rounds=args.max_rounds,
    )

    # Print results
    print()
    print("=" * 80)
    if result["success"]:
        print("OPTIMIZATION SUCCESSFUL")
        print("=" * 80)
        print(f"  Best time:       {result['best_time_ms']:.4f} ms")
        print(f"  PyTorch eager:   {result.get('pytorch_baseline_ms', 'N/A')} ms")
        print(f"  PyTorch compile: {result.get('pytorch_compile_ms', 'N/A')} ms")
        print(f"  Initial kernel:  {result.get('initial_kernel_time_ms', 'N/A')} ms")
        print(f"  Total rounds:    {result['total_rounds']}")

        if result.get("top_kernels"):
            print("\n  Top kernels:")
            for i, k in enumerate(result["top_kernels"][:5], 1):
                print(f"    {i}. {k['time_ms']:.4f} ms (gen {k['generation']})")

        # Save best kernel
        if result.get("kernel_code"):
            out_file = kernel_dir / f"optimized_{args.strategy}.py"
            out_file.write_text(result["kernel_code"])
            print(f"\n  Saved to: {out_file}")
    else:
        print("OPTIMIZATION FAILED")
        print("=" * 80)
        if result.get("error"):
            print(f"  Error: {result['error']}")

    print(f"\n  Logs: {log_dir}")


if __name__ == "__main__":
    main()
