#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

"""GPU operations CLI for the kernel-optimizer Claude Code plugin.

Thin CLI wrapper around existing kubectl platform classes. This is the bridge
between Claude Code's Bash tool and the remote GPU pod.

All subcommands read KubectlConfig from environment variables and output JSON
to stdout.

Environment variables:
    KUBECTL_POD_NAME        — explicit pod name
    KUBECTL_NAMESPACE       — k8s namespace (default: "default")
    KUBECTL_LABEL_SELECTOR  — label selector for auto-discovery
    KUBECTL_CONTAINER       — container name in pod (optional)
    KUBECTL_PYTHON_PATH     — python binary on pod (default: "python3")
    KUBECTL_NCU_BIN_PATH    — NCU binary on pod (default: "ncu")

Usage:
    python gpu_ops.py detect-gpu
    python gpu_ops.py verify --kernel-file <path> --problem-file <path> --test-file <path>
    python gpu_ops.py benchmark --kernel-file <path> --problem-file <path>
    python gpu_ops.py benchmark-reference --problem-file <path> [--compile]
    python gpu_ops.py profile --kernel-file <path> --problem-file <path> --round <N> --artifacts-dir <path>
    python gpu_ops.py roofline --ncu-metrics-file <path>
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from triton_kernel_agent.platform.kubectl import (
    KubectlAcceleratorSpecsProvider,
    KubectlBenchmarker,
    KubectlConfig,
    KubectlKernelProfiler,
    KubectlVerifier,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("gpu_ops")


def _get_config() -> KubectlConfig:
    """Create KubectlConfig from environment variables."""
    return KubectlConfig.from_env()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_detect_gpu(args: argparse.Namespace) -> None:
    """Detect GPU on the pod and return specs."""
    config = _get_config()
    provider = KubectlAcceleratorSpecsProvider(
        kubectl_config=config, logger=logger
    )
    specs = provider.get_specs()
    print(json.dumps(specs, indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify kernel correctness on the GPU pod."""
    config = _get_config()
    verifier = KubectlVerifier(
        log_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
        logger=logger,
        kubectl_config=config,
    )

    kernel_code = Path(args.kernel_file).read_text(encoding="utf-8")
    problem_file = Path(args.problem_file)
    test_code = Path(args.test_file).read_text(encoding="utf-8")

    passed = verifier.verify(kernel_code, problem_file, test_code)

    result = {"passed": passed}
    print(json.dumps(result, indent=2))
    sys.exit(0 if passed else 1)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark a kernel on the GPU pod."""
    config = _get_config()
    benchmarker = KubectlBenchmarker(
        log_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
        logger=logger,
        kubectl_config=config,
    )

    kernel_code = Path(args.kernel_file).read_text(encoding="utf-8")
    problem_file = Path(args.problem_file)

    time_ms = benchmarker.benchmark_kernel(kernel_code, problem_file)

    result = {"time_ms": time_ms}
    print(json.dumps(result, indent=2))

    if time_ms == float("inf"):
        sys.exit(1)


def cmd_benchmark_reference(args: argparse.Namespace) -> None:
    """Benchmark PyTorch reference on the GPU pod."""
    config = _get_config()
    benchmarker = KubectlBenchmarker(
        log_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
        logger=logger,
        kubectl_config=config,
    )

    problem_file = Path(args.problem_file)

    if args.compile:
        time_ms = benchmarker.benchmark_reference_compiled(problem_file)
    else:
        time_ms = benchmarker.benchmark_reference(problem_file)

    result = {"time_ms": time_ms, "mode": "compile" if args.compile else "eager"}
    print(json.dumps(result, indent=2))

    if time_ms == float("inf"):
        sys.exit(1)


def cmd_profile(args: argparse.Namespace) -> None:
    """Profile a kernel with NCU on the GPU pod."""
    config = _get_config()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    profiler = KubectlKernelProfiler(
        kubectl_config=config,
        logger=logger,
        artifacts_dir=artifacts_dir,
        logs_dir=artifacts_dir,
    )

    kernel_file = Path(args.kernel_file)
    problem_file = Path(args.problem_file)
    round_num = args.round

    profiler_result = profiler.profile_kernel(
        kernel_file, problem_file, round_num
    )

    if profiler_result is None:
        print(json.dumps({"error": "Profiling failed", "metrics": {}}))
        sys.exit(1)

    # Extract metrics and CSV path
    result = {
        "metrics": profiler_result.ncu_metrics if hasattr(profiler_result, "ncu_metrics") else {},
        "csv_path": str(profiler_result.csv_path) if hasattr(profiler_result, "csv_path") else "",
        "metrics_prompt": profiler_result.ncu_metrics_prompt if hasattr(profiler_result, "ncu_metrics_prompt") else "",
    }

    # Also save metrics to a JSON file for subagents to read
    metrics_file = artifacts_dir / f"round{round_num:03d}_ncu_metrics.json"
    metrics_file.write_text(
        json.dumps(result["metrics"], indent=2, default=str), encoding="utf-8"
    )

    print(json.dumps(result, indent=2, default=str))


def cmd_roofline(args: argparse.Namespace) -> None:
    """Run roofline analysis on NCU metrics (local compute, no GPU needed)."""
    from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import RooflineAnalyzer

    metrics_file = Path(args.ncu_metrics_file)
    ncu_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

    analyzer = RooflineAnalyzer(logger=logger)

    # If metrics are keyed by kernel name, get the Triton kernel's metrics
    if ncu_metrics and isinstance(next(iter(ncu_metrics.values()), None), dict):
        triton_kernels = {
            name: metrics
            for name, metrics in ncu_metrics.items()
            if not name.startswith("at::") and not name.startswith("void at::")
        }
        flat_metrics = (
            next(iter(triton_kernels.values()))
            if triton_kernels
            else next(iter(ncu_metrics.values()), {})
        )
    else:
        flat_metrics = ncu_metrics

    roofline_result = analyzer.analyze(flat_metrics)

    result = {
        "bottleneck": roofline_result.bottleneck,
        "compute_sol_pct": roofline_result.compute_sol_pct,
        "memory_sol_pct": roofline_result.memory_sol_pct,
        "efficiency_pct": roofline_result.efficiency_pct,
        "headroom_pct": roofline_result.headroom_pct,
        "at_roofline": roofline_result.at_roofline,
        "uses_tensor_cores": roofline_result.uses_tensor_cores,
        "warnings": roofline_result.warnings,
    }

    # Save to file next to the metrics
    output_file = metrics_file.parent / metrics_file.name.replace(
        "ncu_metrics", "roofline"
    )
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU operations CLI for kernel-optimizer plugin"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # detect-gpu
    subparsers.add_parser("detect-gpu", help="Detect GPU on pod and return specs")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify kernel correctness")
    p_verify.add_argument("--kernel-file", required=True)
    p_verify.add_argument("--problem-file", required=True)
    p_verify.add_argument("--test-file", required=True)
    p_verify.add_argument("--artifacts-dir", default=None)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark kernel")
    p_bench.add_argument("--kernel-file", required=True)
    p_bench.add_argument("--problem-file", required=True)
    p_bench.add_argument("--artifacts-dir", default=None)

    # benchmark-reference
    p_ref = subparsers.add_parser(
        "benchmark-reference", help="Benchmark PyTorch reference"
    )
    p_ref.add_argument("--problem-file", required=True)
    p_ref.add_argument("--compile", action="store_true")
    p_ref.add_argument("--artifacts-dir", default=None)

    # profile
    p_profile = subparsers.add_parser("profile", help="Profile kernel with NCU")
    p_profile.add_argument("--kernel-file", required=True)
    p_profile.add_argument("--problem-file", required=True)
    p_profile.add_argument("--round", type=int, required=True)
    p_profile.add_argument("--artifacts-dir", required=True)

    # roofline
    p_roofline = subparsers.add_parser(
        "roofline", help="Run roofline analysis on NCU metrics"
    )
    p_roofline.add_argument("--ncu-metrics-file", required=True)

    args = parser.parse_args()

    dispatch = {
        "detect-gpu": cmd_detect_gpu,
        "verify": cmd_verify,
        "benchmark": cmd_benchmark,
        "benchmark-reference": cmd_benchmark_reference,
        "profile": cmd_profile,
        "roofline": cmd_roofline,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
