#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0

"""Staged integration test for the kubectl platform.

Run this script locally to validate kubectl connectivity, file transfer,
GPU execution, and the full optimization pipeline against a remote pod.

Works on Python 3.9+ (bypasses package __init__.py which needs 3.10+).

Usage:
    # Set required env vars
    export KUBECTL_POD_NAME=<your-pod-name>
    export KUBECTL_NAMESPACE=<your-namespace>

    # If your pod is in a non-default kube context, switch first:
    # kubectl config use-context <your-context>

    python scripts/test_kubectl_platform.py
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ── Bootstrap imports (Python 3.9 compatible) ────────────────────────────
# The main package __init__.py uses 3.10+ syntax (str | None).
# We stub the package namespaces and import only the files we need directly.

def _import_file(module_name, file_path):
    """Import a single .py file by path, registering it in sys.modules."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap():
    """Set up minimal sys.modules stubs so kubectl.py can be imported."""
    pkg_root = _PROJECT_ROOT / "triton_kernel_agent"
    platform_dir = pkg_root / "platform"

    # Stub packages (avoid triggering real __init__.py)
    for pkg_name, pkg_path in [
        ("triton_kernel_agent", pkg_root),
        ("triton_kernel_agent.platform", platform_dir),
    ]:
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_path)]
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg

    # Import interfaces (no heavy deps)
    _import_file(
        "triton_kernel_agent.platform.interfaces",
        platform_dir / "interfaces.py",
    )
    # Import kubectl
    return _import_file(
        "triton_kernel_agent.platform.kubectl",
        platform_dir / "kubectl.py",
    )


_kubectl = _bootstrap()
KubectlConfig = _kubectl.KubectlConfig
KubectlExecutor = _kubectl.KubectlExecutor


# ── Helpers ──────────────────────────────────────────────────────────────

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _pass(msg):
    print(f"  {Colors.GREEN}PASS{Colors.RESET} {msg}")


def _fail(msg):
    print(f"  {Colors.RED}FAIL{Colors.RESET} {msg}")


def _skip(msg):
    print(f"  {Colors.YELLOW}SKIP{Colors.RESET} {msg}")


def _header(stage, title):
    print(f"\n{Colors.BOLD}Stage {stage}: {title}{Colors.RESET}")
    print("-" * 60)


# ── Stage 1: kubectl connectivity ────────────────────────────────────────

def stage_1_connectivity(executor):
    _header(1, "kubectl connectivity")

    try:
        pod = executor.get_pod_name()
        _pass(f"Pod discovered: {pod}")
    except Exception as e:
        _fail(f"Pod discovery failed: {e}")
        return False

    try:
        rc, stdout, stderr = executor.exec("echo hello", timeout=15)
        if rc == 0 and "hello" in stdout:
            _pass("kubectl exec works")
        else:
            _fail(f"kubectl exec returned rc={rc}, stderr={stderr}")
            return False
    except Exception as e:
        _fail(f"kubectl exec failed: {e}")
        return False

    return True


# ── Stage 2: File transfer ───────────────────────────────────────────────

def stage_2_file_transfer(executor):
    _header(2, "File transfer (kubectl cp)")

    try:
        remote_dir = executor.mkdtemp()
        _pass(f"Created remote temp dir: {remote_dir}")
    except Exception as e:
        _fail(f"mkdtemp failed: {e}")
        return False

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test_content_12345\n")
            local_path = f.name

        executor.copy_to(local_path, f"{remote_dir}/test.txt")
        _pass("copy_to succeeded")

        rc, stdout, _ = executor.exec(f"cat {remote_dir}/test.txt", timeout=15)
        if "test_content_12345" in stdout:
            _pass("File content verified on pod")
        else:
            _fail(f"File content mismatch: {stdout!r}")
            return False

        local_back = Path(tempfile.mktemp(suffix=".txt"))
        executor.copy_from(f"{remote_dir}/test.txt", local_back)
        content = local_back.read_text()
        if "test_content_12345" in content:
            _pass("copy_from succeeded and content matches")
        else:
            _fail(f"Round-trip content mismatch: {content!r}")
            return False

        local_back.unlink(missing_ok=True)
        os.unlink(local_path)

    except Exception as e:
        _fail(f"File transfer failed: {e}")
        return False
    finally:
        executor.rm(remote_dir)

    return True


# ── Stage 3: GPU execution ───────────────────────────────────────────────

def stage_3_gpu(executor):
    _header(3, "GPU execution on pod")

    python = executor._config.python_path

    # Check torch
    rc, stdout, stderr = executor.exec(
        f'{python} -c "import torch; print(torch.cuda.is_available()); '
        f'print(torch.cuda.get_device_name(0))"',
        timeout=60,
    )
    lines = stdout.strip().splitlines()
    if rc != 0 or not lines:
        _fail(f"torch import failed: {stderr[:300]}")
        return False

    if lines[0].strip() == "True":
        gpu_name = lines[1].strip() if len(lines) > 1 else "unknown"
        _pass(f"CUDA available, GPU: {gpu_name}")
    else:
        _fail("CUDA not available on pod")
        return False

    # Check triton
    rc, stdout, stderr = executor.exec(
        f'{python} -c "import triton; print(triton.__version__)"',
        timeout=30,
    )
    if rc == 0:
        _pass(f"Triton available: v{stdout.strip()}")
    else:
        _fail(f"Triton not importable: {stderr[:200]}")
        print("    Triton is needed for kernel benchmarking.")
        print("    Install with: pip install triton")
        return False

    # Check numpy (needed by timing.py)
    rc, stdout, stderr = executor.exec(
        f'{python} -c "import numpy; print(numpy.__version__)"',
        timeout=30,
    )
    if rc == 0:
        _pass(f"NumPy available: v{stdout.strip()}")
    else:
        _fail(f"NumPy not importable: {stderr[:200]}")
        print("    NumPy is needed by timing.py.")
        print("    Install with: pip install numpy")
        return False

    return True


# ── Stage 4: Kernel benchmarking on pod ──────────────────────────────────

# Minimal problem.py and kernel.py for testing
_TEST_PROBLEM = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0

def get_inputs():
    return [torch.randn(1024, 1024)]

def get_init_inputs():
    return []
'''

_TEST_KERNEL = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _double_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)

def kernel_function(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _double_kernel[grid](x, out, n, BLOCK_SIZE=1024)
    return out
'''

_TEST_CODE = '''
import torch
import sys
sys.path.insert(0, ".")
from kernel import kernel_function

def test():
    x = torch.randn(1024, 1024, device="cuda")
    out = kernel_function(x)
    expected = x * 2.0
    assert torch.allclose(out, expected, atol=1e-5), "Correctness check failed!"
    print("PASS")

if __name__ == "__main__":
    test()
'''


def stage_4_benchmark(executor):
    _header(4, "Kernel benchmark on pod")

    remote_dir = ""
    try:
        remote_dir = executor.mkdtemp()

        with tempfile.TemporaryDirectory() as local_tmp:
            local = Path(local_tmp)

            # Write test files
            (local / "problem.py").write_text(_TEST_PROBLEM)
            (local / "kernel.py").write_text(_TEST_KERNEL)
            (local / "test_kernel.py").write_text(_TEST_CODE)

            # Write timing.py and kernel_subprocess.py from the local package
            (local / "timing.py").write_text(_kubectl._get_timing_py_source())
            (local / "kernel_subprocess.py").write_text(
                _kubectl._get_kernel_subprocess_source()
            )

            # Copy all to pod
            for fname in [
                "problem.py",
                "kernel.py",
                "test_kernel.py",
                "timing.py",
                "kernel_subprocess.py",
            ]:
                executor.copy_to(local / fname, f"{remote_dir}/{fname}")

            _pass("Files copied to pod")

            python = executor._config.python_path

            # 4a. Run correctness test
            rc, stdout, stderr = executor.exec(
                f"cd {remote_dir} && {python} test_kernel.py",
                timeout=120,
            )
            if rc == 0 and "PASS" in stdout:
                _pass("Kernel correctness test passed on pod")
            else:
                _fail(f"Kernel test failed: rc={rc}\n  stdout={stdout[:300]}\n  stderr={stderr[:300]}")
                return False

            # 4b. Run kernel_subprocess.py benchmark
            rc, stdout, stderr = executor.exec(
                f"cd {remote_dir} && {python} kernel_subprocess.py "
                f"--kernel kernel.py --problem problem.py "
                f"--json results.json --quiet",
                timeout=300,
            )
            if rc != 0:
                _fail(f"kernel_subprocess.py failed: rc={rc}\n  stderr={stderr[:500]}")
                return False

            # Copy results back
            result_local = local / "results.json"
            executor.copy_from(f"{remote_dir}/results.json", result_local)
            results = json.loads(result_local.read_text())

            time_ms = results.get("kernels", {}).get("kernel", {}).get("time_ms")
            if time_ms and time_ms != float("inf"):
                _pass(f"Kernel benchmark: {time_ms:.4f} ms")
            else:
                _fail(f"Unexpected benchmark results: {results}")
                return False

    except Exception as e:
        _fail(f"Benchmark stage failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if remote_dir:
            executor.rm(remote_dir)

    return True


# ── Stage 5: Full platform components ────────────────────────────────────

def stage_5_platform_components(config):
    _header(5, "Platform components (KubectlVerifier + KubectlBenchmarker)")

    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("test_kubectl")

    KubectlVerifier = _kubectl.KubectlVerifier
    KubectlBenchmarker = _kubectl.KubectlBenchmarker
    KubectlAcceleratorSpecsProvider = _kubectl.KubectlAcceleratorSpecsProvider

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write problem file
        problem_file = tmpdir / "problem.py"
        problem_file.write_text(_TEST_PROBLEM)

        # 5a. KubectlVerifier
        try:
            verifier = KubectlVerifier(
                log_dir=tmpdir, logger=logger, kubectl_config=config
            )
            result = verifier.verify(
                kernel_code=_TEST_KERNEL,
                problem_file=problem_file,
                test_code=_TEST_CODE,
            )
            if result:
                _pass("KubectlVerifier.verify() returned True")
            else:
                _fail("KubectlVerifier.verify() returned False")
                return False
        except Exception as e:
            _fail(f"KubectlVerifier failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 5b. KubectlBenchmarker
        try:
            import multiprocessing as mp

            benchmarker = KubectlBenchmarker(
                log_dir=tmpdir,
                logger=logger,
                benchmark_lock=mp.Lock(),
                kubectl_config=config,
            )

            # Benchmark kernel
            kernel_time = benchmarker.benchmark_kernel(_TEST_KERNEL, problem_file)
            if kernel_time != float("inf"):
                _pass(f"KubectlBenchmarker.benchmark_kernel(): {kernel_time:.4f} ms")
            else:
                _fail("KubectlBenchmarker.benchmark_kernel() returned inf")
                return False

            # Benchmark PyTorch reference
            ref_time = benchmarker.benchmark_reference(problem_file)
            if ref_time != float("inf"):
                _pass(f"KubectlBenchmarker.benchmark_reference(): {ref_time:.4f} ms")
            else:
                _fail("KubectlBenchmarker.benchmark_reference() returned inf")
                return False

            # Benchmark torch.compile
            compile_time = benchmarker.benchmark_reference_compiled(problem_file)
            if compile_time != float("inf"):
                _pass(f"KubectlBenchmarker.benchmark_reference_compiled(): {compile_time:.4f} ms")
            else:
                _skip("torch.compile benchmark returned inf (may not be supported on pod)")

        except Exception as e:
            _fail(f"KubectlBenchmarker failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # 5c. AcceleratorSpecsProvider
    try:
        specs_provider = KubectlAcceleratorSpecsProvider(
            kubectl_config=config, logger=logger
        )
        specs = specs_provider.get_specs()
        gpu_name = specs.get("name", "unknown")
        _pass(f"KubectlAcceleratorSpecsProvider: {gpu_name}")
    except Exception as e:
        _skip(f"AcceleratorSpecsProvider failed (non-fatal): {e}")

    return True


# ── Stage 6: Registry integration ────────────────────────────────────────

def stage_6_registry():
    _header(6, "Registry integration")

    try:
        # This needs the full package chain — skip on Python < 3.10
        _import_file(
            "triton_kernel_agent.platform.noop",
            _PROJECT_ROOT / "triton_kernel_agent" / "platform" / "noop.py",
        )
        _import_file(
            "triton_kernel_agent.platform.nvidia",
            _PROJECT_ROOT / "triton_kernel_agent" / "platform" / "nvidia.py",
        )
        registry_mod = _import_file(
            "triton_kernel_agent.platform.registry",
            _PROJECT_ROOT / "triton_kernel_agent" / "platform" / "registry.py",
        )
        registry = registry_mod.registry
    except Exception as e:
        _skip(f"Registry import failed (may need full deps): {e}")
        return True  # non-blocking

    try:
        components = registry.list_implementations("verifier")
        if "kubectl" in components:
            _pass(f"'kubectl' registered for verifier (all: {components})")
        else:
            _fail(f"'kubectl' not registered. Available: {components}")
            return False

        for key in [
            "verifier",
            "benchmarker",
            "worker_runner",
            "specs_provider",
            "profiler",
            "roofline_analyzer",
            "bottleneck_analyzer",
            "rag_prescriber",
        ]:
            if registry.has(key, "kubectl"):
                _pass(f"  {key}: kubectl registered")
            else:
                _fail(f"  {key}: kubectl NOT registered")
                return False

    except Exception as e:
        _fail(f"Registry check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


# ── Main ─────────────────────────────────────────────────────────────────

def _print_summary(results):
    names = {
        1: "Connectivity",
        2: "File transfer",
        3: "GPU execution",
        4: "Kernel benchmark",
        5: "Platform components",
        6: "Registry integration",
    }
    print(f"\n{Colors.BOLD}Summary{Colors.RESET}")
    print("-" * 40)
    for stage in sorted(results):
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if results[stage] else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  Stage {stage} ({names.get(stage, '?')}): {status}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    color = Colors.GREEN if passed == total else Colors.RED
    print(f"\n  {color}{passed}/{total} stages passed{Colors.RESET}")


def main():
    print(f"{Colors.BOLD}kubectl Platform Integration Test{Colors.RESET}")
    print("=" * 60)

    # Check env vars
    pod = os.environ.get("KUBECTL_POD_NAME", "")
    ns = os.environ.get("KUBECTL_NAMESPACE", "default")
    selector = os.environ.get("KUBECTL_LABEL_SELECTOR", "")

    if not pod and not selector:
        print(f"\n{Colors.RED}ERROR: Set KUBECTL_POD_NAME or KUBECTL_LABEL_SELECTOR{Colors.RESET}")
        print("\nExample:")
        print("  export KUBECTL_POD_NAME=my-gpu-pod")
        print("  export KUBECTL_NAMESPACE=my-namespace")
        print("  # If non-default context:")
        print("  # kubectl config use-context my-context")
        sys.exit(1)

    print(f"  Pod: {pod or '(auto-discover)'}")
    print(f"  Namespace: {ns}")
    if selector:
        print(f"  Label selector: {selector}")

    config = KubectlConfig.from_env()
    executor = KubectlExecutor(config)

    results = {}

    # Stage 6 first — no pod needed
    results[6] = stage_6_registry()

    # Stages that need the pod
    results[1] = stage_1_connectivity(executor)
    if not results[1]:
        print(f"\n{Colors.RED}Stopping: cannot reach pod.{Colors.RESET}")
        _print_summary(results)
        sys.exit(1)

    results[2] = stage_2_file_transfer(executor)
    if not results[2]:
        print(f"\n{Colors.RED}Stopping: file transfer broken.{Colors.RESET}")
        _print_summary(results)
        sys.exit(1)

    results[3] = stage_3_gpu(executor)
    if not results[3]:
        print(f"\n{Colors.RED}Stopping: GPU/libraries not available on pod.{Colors.RESET}")
        print("\nTo install missing packages on the pod:")
        print(f"  kubectl -n {ns} exec {executor.get_pod_name()} -- pip install triton numpy")
        _print_summary(results)
        sys.exit(1)

    results[4] = stage_4_benchmark(executor)
    results[5] = stage_5_platform_components(config)

    _print_summary(results)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
