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

"""Kubectl (remote GPU pod) implementations of platform interfaces.

This module enables running the optimization pipeline on a local machine
without a GPU by offloading GPU operations (kernel verification,
benchmarking, NCU profiling) to a Kubernetes pod via ``kubectl exec``
and ``kubectl cp``.

LLM calls and CSV parsing remain local (where internet access and
pandas are available); only GPU work is sent to the pod.

Usage::

    export KUBECTL_POD_NAME=gpu-pod-abc123
    export KUBECTL_NAMESPACE=default

    manager = OptimizationManager(
        platform="kubectl",
        kubectl_config=KubectlConfig.from_env(),
        ...
    )
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from triton_kernel_agent.platform.interfaces import (
    AcceleratorSpecsProvider,
    BottleneckAnalyzerBase,
    KernelBenchmarker,
    KernelProfilerBase,
    KernelVerifier,
    RooflineAnalyzerBase,
    WorkerRunner,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KubectlConfig:
    """Configuration for kubectl-based remote GPU execution.

    Populated from environment variables via :meth:`from_env`.
    """

    pod_name: str = ""
    namespace: str = "default"
    label_selector: str = ""
    container: str = ""
    python_path: str = "python3"
    ncu_bin_path: str = "ncu"

    @classmethod
    def from_env(cls) -> "KubectlConfig":
        """Create a config from environment variables.

        Environment variables:
            KUBECTL_POD_NAME        — explicit pod name (or empty for auto-discover)
            KUBECTL_NAMESPACE       — k8s namespace (default: ``"default"``)
            KUBECTL_LABEL_SELECTOR  — label selector for auto-discovery
            KUBECTL_CONTAINER       — container name in pod (optional)
            KUBECTL_PYTHON_PATH     — python binary on pod (default: ``"python3"``)
            KUBECTL_NCU_BIN_PATH    — NCU binary on pod (default: ``"ncu"``)
        """
        return cls(
            pod_name=os.environ.get("KUBECTL_POD_NAME", ""),
            namespace=os.environ.get("KUBECTL_NAMESPACE", "default"),
            label_selector=os.environ.get("KUBECTL_LABEL_SELECTOR", ""),
            container=os.environ.get("KUBECTL_CONTAINER", ""),
            python_path=os.environ.get("KUBECTL_PYTHON_PATH", "python3"),
            ncu_bin_path=os.environ.get("KUBECTL_NCU_BIN_PATH", "ncu"),
        )


# ---------------------------------------------------------------------------
# Low-level executor
# ---------------------------------------------------------------------------


class KubectlExecutor:
    """Wraps kubectl subprocesses for exec / cp operations."""

    def __init__(
        self,
        config: KubectlConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._cached_pod: str | None = None

    # -- Pod discovery -----------------------------------------------------

    def get_pod_name(self) -> str:
        """Return the target pod name, auto-discovering if needed."""
        if self._cached_pod:
            return self._cached_pod

        if self._config.pod_name:
            self._cached_pod = self._config.pod_name
            return self._cached_pod

        if not self._config.label_selector:
            raise ValueError(
                "Either KUBECTL_POD_NAME or KUBECTL_LABEL_SELECTOR must be set"
            )

        cmd = [
            "kubectl",
            "get",
            "pods",
            "-n",
            self._config.namespace,
            "-l",
            self._config.label_selector,
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                f"Failed to discover pod with selector "
                f"'{self._config.label_selector}': {result.stderr}"
            )

        self._cached_pod = result.stdout.strip()
        self._logger.info(f"Auto-discovered pod: {self._cached_pod}")
        return self._cached_pod

    # -- Core operations ---------------------------------------------------

    def _base_cmd(self) -> list[str]:
        """Base kubectl command with namespace."""
        return ["kubectl", "-n", self._config.namespace]

    def _container_args(self) -> list[str]:
        """Container flags if configured."""
        if self._config.container:
            return ["-c", self._config.container]
        return []

    def exec(
        self,
        cmd: str | list[str],
        timeout: int = 300,
    ) -> tuple[int, str, str]:
        """Run a command on the pod via ``kubectl exec``.

        Args:
            cmd: Command string or list.
            timeout: Timeout in seconds.

        Returns:
            ``(returncode, stdout, stderr)``
        """
        pod = self.get_pod_name()
        if isinstance(cmd, str):
            exec_cmd = (
                self._base_cmd()
                + ["exec", pod]
                + self._container_args()
                + ["--", "sh", "-c", cmd]
            )
        else:
            exec_cmd = (
                self._base_cmd()
                + ["exec", pod]
                + self._container_args()
                + ["--"]
                + cmd
            )

        self._logger.debug(f"kubectl exec: {' '.join(exec_cmd)}")
        result = subprocess.run(
            exec_cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr

    def copy_to(self, local_path: str | Path, remote_path: str) -> None:
        """Copy a local file/dir to the pod."""
        pod = self.get_pod_name()
        dest = f"{pod}:{remote_path}"
        cmd = self._base_cmd() + ["cp", str(local_path), dest]
        if self._config.container:
            cmd += ["-c", self._config.container]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"kubectl cp to pod failed: {result.stderr}")

    def copy_from(self, remote_path: str, local_path: str | Path) -> None:
        """Copy a file from the pod to local."""
        pod = self.get_pod_name()
        src = f"{pod}:{remote_path}"
        cmd = self._base_cmd() + ["cp", src, str(local_path)]
        if self._config.container:
            cmd += ["-c", self._config.container]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"kubectl cp from pod failed: {result.stderr}")

    def mkdtemp(self) -> str:
        """Create a temporary directory on the pod."""
        rc, stdout, stderr = self.exec("mktemp -d", timeout=30)
        if rc != 0:
            raise RuntimeError(f"mkdtemp on pod failed: {stderr}")
        return stdout.strip()

    def rm(self, remote_path: str) -> None:
        """Remove a path on the pod (best-effort)."""
        try:
            self.exec(f"rm -rf {remote_path}", timeout=30)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helper: read local script for transfer to pod
# ---------------------------------------------------------------------------


def _get_timing_py_source() -> str:
    """Read timing.py source from the local package."""
    timing_path = (
        Path(__file__).parent.parent
        / "opt_worker_component"
        / "benchmarking"
        / "timing.py"
    )
    return timing_path.read_text(encoding="utf-8")


def _get_kernel_subprocess_source() -> str:
    """Read kernel_subprocess.py and patch the import.

    The original file does ``from timing import ...`` (no package prefix),
    which works when both files are in the same directory on the pod.
    The file already uses this import style, so no patching needed.
    """
    script_path = (
        Path(__file__).parent.parent
        / "opt_worker_component"
        / "benchmarking"
        / "kernel_subprocess.py"
    )
    return script_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Embedded PyTorch benchmark script (self-contained, runs on pod)
# ---------------------------------------------------------------------------

_PYTORCH_BENCH_SCRIPT = '''
"""Self-contained PyTorch benchmark script for kubectl pods.

Usage:
    python pytorch_bench.py <problem.py> <result.json> [compile]

Only uses torch — no KernelAgent imports needed.
"""
import importlib.util
import json
import sys
import hashlib
from pathlib import Path

import torch

def import_module(path, name=None):
    path = Path(path)
    if name is None:
        name = f"mod_{hashlib.md5(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def main():
    problem_path = sys.argv[1]
    result_path = sys.argv[2]
    use_compile = len(sys.argv) > 3 and sys.argv[3] == "compile"

    mod = import_module(problem_path, "problem")
    Model = mod.Model
    get_inputs = mod.get_inputs
    get_init_inputs = getattr(mod, "get_init_inputs", None)

    init_inputs = get_init_inputs() if get_init_inputs else []
    if not isinstance(init_inputs, (tuple, list)):
        init_inputs = [init_inputs]

    model = Model(*init_inputs) if init_inputs else Model()
    model = model.cuda()

    has_params = any(p.numel() > 0 for p in model.parameters())
    is_loss = isinstance(model, torch.nn.modules.loss._Loss)
    dtype = torch.bfloat16

    if has_params or not is_loss:
        if has_params:
            model = model.to(dtype)
        inputs = get_inputs()
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        inputs = [
            inp.cuda().to(dtype) if isinstance(inp, torch.Tensor) and inp.is_floating_point()
            else inp.cuda() if isinstance(inp, torch.Tensor) else inp
            for inp in inputs
        ]
    else:
        inputs = get_inputs()
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        processed = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                if i == 0 and inp.is_floating_point():
                    processed.append(inp.cuda().to(torch.float32))
                else:
                    processed.append(inp.cuda())
            else:
                processed.append(inp)
        inputs = processed

    if use_compile:
        model = torch.compile(model)
        for _ in range(3):
            model(*inputs)
        torch.cuda.synchronize()

    # Warmup
    for _ in range(25):
        model(*inputs)
    torch.cuda.synchronize()

    # Timing with CUDA events
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(*inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    import numpy as np
    mean_ms = float(np.mean(times))

    result = {"time_ms": mean_ms}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"time_ms: {mean_ms:.4f}")

if __name__ == "__main__":
    main()
'''


# =========================================================================
# Manager-level components
# =========================================================================


class KubectlVerifier(KernelVerifier):
    """Verifies kernel correctness by running tests on the GPU pod."""

    def __init__(
        self,
        log_dir: Path | None = None,
        logger: logging.Logger | None = None,
        kubectl_config: KubectlConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir else Path(".")
        self.logger = logger or logging.getLogger(__name__)
        self._executor = KubectlExecutor(
            kubectl_config or KubectlConfig(), self.logger
        )

    def verify(
        self,
        kernel_code: str,
        problem_file: Path,
        test_code: str,
    ) -> bool:
        remote_dir = ""
        try:
            remote_dir = self._executor.mkdtemp()

            # Write files locally then copy
            with tempfile.TemporaryDirectory() as local_tmp:
                local_tmp_path = Path(local_tmp)

                (local_tmp_path / "kernel.py").write_text(
                    kernel_code, encoding="utf-8"
                )
                (local_tmp_path / "test_kernel.py").write_text(
                    test_code, encoding="utf-8"
                )

                self._executor.copy_to(
                    local_tmp_path / "kernel.py", f"{remote_dir}/kernel.py"
                )
                self._executor.copy_to(
                    problem_file, f"{remote_dir}/problem.py"
                )
                self._executor.copy_to(
                    local_tmp_path / "test_kernel.py",
                    f"{remote_dir}/test_kernel.py",
                )

            rc, stdout, stderr = self._executor.exec(
                f"cd {remote_dir} && {self._executor._config.python_path} test_kernel.py",
                timeout=120,
            )

            if rc == 0:
                self.logger.info("Initial kernel passed correctness verification (kubectl)")
            else:
                # Filter kubectl "Defaulted container" warning from real errors
                real_stderr = "\n".join(
                    line for line in stderr.splitlines()
                    if "Defaulted container" not in line
                ).strip()
                self.logger.error(
                    f"Initial kernel failed verification (kubectl):\n"
                    f"  stdout: {stdout[:500]}\n"
                    f"  stderr: {real_stderr[:1000]}"
                )

            return rc == 0

        except Exception as e:
            self.logger.error(f"Kubectl verification error: {e}")
            return False
        finally:
            if remote_dir:
                self._executor.rm(remote_dir)


class KubectlBenchmarker(KernelBenchmarker):
    """Benchmarks kernels and baselines on a remote GPU pod via kubectl."""

    def __init__(
        self,
        log_dir: Path | None = None,
        logger: logging.Logger | None = None,
        benchmark_lock: Any = None,
        kubectl_config: KubectlConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir else Path(".")
        self.logger = logger or logging.getLogger(__name__)
        self.benchmark_lock = benchmark_lock
        self._executor = KubectlExecutor(
            kubectl_config or KubectlConfig(), self.logger
        )

    def benchmark_kernel(
        self,
        kernel_code: str,
        problem_file: Path,
    ) -> float:
        remote_dir = ""
        try:
            if self.benchmark_lock:
                self.benchmark_lock.acquire()

            remote_dir = self._executor.mkdtemp()

            with tempfile.TemporaryDirectory() as local_tmp:
                local_tmp_path = Path(local_tmp)

                (local_tmp_path / "kernel.py").write_text(
                    kernel_code, encoding="utf-8"
                )
                (local_tmp_path / "timing.py").write_text(
                    _get_timing_py_source(), encoding="utf-8"
                )
                (local_tmp_path / "kernel_subprocess.py").write_text(
                    _get_kernel_subprocess_source(), encoding="utf-8"
                )

                self._executor.copy_to(
                    local_tmp_path / "kernel.py", f"{remote_dir}/kernel.py"
                )
                self._executor.copy_to(
                    problem_file, f"{remote_dir}/problem.py"
                )
                self._executor.copy_to(
                    local_tmp_path / "timing.py", f"{remote_dir}/timing.py"
                )
                self._executor.copy_to(
                    local_tmp_path / "kernel_subprocess.py",
                    f"{remote_dir}/kernel_subprocess.py",
                )

                python = self._executor._config.python_path
                rc, stdout, stderr = self._executor.exec(
                    f"cd {remote_dir} && {python} kernel_subprocess.py "
                    f"--kernel kernel.py --problem problem.py "
                    f"--json results.json --quiet",
                    timeout=300,
                )

                if rc != 0:
                    self.logger.error(
                        f"Kernel benchmark failed on pod: {stderr[:500]}"
                    )
                    return float("inf")

                # Copy result back
                local_result = local_tmp_path / "results.json"
                self._executor.copy_from(
                    f"{remote_dir}/results.json", local_result
                )
                results = json.loads(local_result.read_text(encoding="utf-8"))

            kernel_time = (
                results.get("kernels", {}).get("kernel", {}).get("time_ms", float("inf"))
            )
            if kernel_time != float("inf"):
                self.logger.info(f"Initial kernel time (kubectl): {kernel_time:.4f}ms")
            return kernel_time

        except Exception as e:
            self.logger.error(f"Kubectl kernel benchmark error: {e}")
            return float("inf")
        finally:
            if self.benchmark_lock:
                try:
                    self.benchmark_lock.release()
                except Exception:
                    pass
            if remote_dir:
                self._executor.rm(remote_dir)

    def _benchmark_pytorch(
        self, problem_file: Path, compile: bool = False
    ) -> float:
        """Common implementation for reference benchmarks."""
        remote_dir = ""
        try:
            if self.benchmark_lock:
                self.benchmark_lock.acquire()

            remote_dir = self._executor.mkdtemp()

            with tempfile.TemporaryDirectory() as local_tmp:
                local_tmp_path = Path(local_tmp)

                (local_tmp_path / "pytorch_bench.py").write_text(
                    _PYTORCH_BENCH_SCRIPT, encoding="utf-8"
                )

                self._executor.copy_to(
                    problem_file, f"{remote_dir}/problem.py"
                )
                self._executor.copy_to(
                    local_tmp_path / "pytorch_bench.py",
                    f"{remote_dir}/pytorch_bench.py",
                )

                python = self._executor._config.python_path
                compile_arg = " compile" if compile else ""
                rc, stdout, stderr = self._executor.exec(
                    f"cd {remote_dir} && {python} pytorch_bench.py "
                    f"problem.py results.json{compile_arg}",
                    timeout=300,
                )

                if rc != 0:
                    label = "compile " if compile else ""
                    self.logger.error(
                        f"PyTorch {label}benchmark failed on pod: {stderr[:500]}"
                    )
                    return float("inf")

                local_result = local_tmp_path / "results.json"
                self._executor.copy_from(
                    f"{remote_dir}/results.json", local_result
                )
                results = json.loads(local_result.read_text(encoding="utf-8"))

            time_ms = results.get("time_ms", float("inf"))
            label = "compile " if compile else ""
            if time_ms != float("inf"):
                self.logger.info(
                    f"PyTorch {label}baseline (kubectl): {time_ms:.4f}ms"
                )
            return time_ms

        except Exception as e:
            label = "compile " if compile else ""
            self.logger.error(f"Kubectl PyTorch {label}benchmark error: {e}")
            return float("inf")
        finally:
            if self.benchmark_lock:
                try:
                    self.benchmark_lock.release()
                except Exception:
                    pass
            if remote_dir:
                self._executor.rm(remote_dir)

    def benchmark_reference(self, problem_file: Path) -> float:
        return self._benchmark_pytorch(problem_file, compile=False)

    def benchmark_reference_compiled(self, problem_file: Path) -> float:
        return self._benchmark_pytorch(problem_file, compile=True)


class KubectlWorkerRunner(WorkerRunner):
    """Spawns OptimizationWorker processes locally (workers need LLM access).

    Identical in structure to ``NvidiaWorkerRunner`` — the kubectl config
    flows through ``worker_kwargs`` to each ``OptimizationWorker``, which
    then creates ``KubectlBenchmark`` / ``KubectlVerificationWorker``
    internally.
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        logger: logging.Logger | None = None,
        benchmark_lock: Any = None,
        profiling_semaphore: Any = None,
        openai_model: str = "claude-opus-4.5",
        high_reasoning_effort: bool = True,
        bottleneck_override: str | None = None,
        worker_kwargs: dict[str, Any] | None = None,
        kubectl_config: KubectlConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir else Path(".")
        self.logger = logger or logging.getLogger(__name__)
        self.benchmark_lock = benchmark_lock
        self.profiling_semaphore = profiling_semaphore
        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self.bottleneck_override = bottleneck_override
        self.worker_kwargs = worker_kwargs or {}
        self.kubectl_config = kubectl_config

    def run_workers(
        self,
        candidates: list[dict[str, Any]],
        round_num: int,
        problem_file: Path,
        test_code: str,
        pytorch_baseline: float,
        shared_history: list[dict],
        shared_reflexions: list[dict],
    ) -> list[dict[str, Any]]:
        result_queue = mp.Queue()
        workers = []

        for i, candidate in enumerate(candidates):
            workdir = self.log_dir / "workers" / f"w{i}" / f"r{round_num}"
            workdir.mkdir(parents=True, exist_ok=True)

            args = (
                i,
                candidate["parent"].kernel_code,
                candidate["parent"].metrics.time_ms,
                candidate["parent"].program_id,
                problem_file,
                test_code,
                workdir,
                workdir / "logs",
                result_queue,
                self.benchmark_lock,
                self.profiling_semaphore,
                pytorch_baseline,
                candidate["bottleneck_id"],
                self.openai_model,
                self.high_reasoning_effort,
                self.bottleneck_override,
                self.worker_kwargs,
                shared_history,
                shared_reflexions,
                self.kubectl_config,
            )

            p = mp.Process(target=_kubectl_worker_process, args=args)
            p.start()
            workers.append(p)

        # Wait for completion
        worker_timeout = 1800
        deadline = time.time() + worker_timeout
        for w in workers:
            remaining = max(0, deadline - time.time())
            w.join(timeout=remaining)
            if w.is_alive():
                self.logger.warning(f"Worker {w.pid} timed out, terminating")
                w.terminate()
                w.join(timeout=5)
                if w.is_alive():
                    self.logger.warning(f"Worker {w.pid} still alive, killing")
                    w.kill()
                    w.join(timeout=2)
            w.close()

        results: list[dict[str, Any]] = []
        while not result_queue.empty():
            try:
                results.append(result_queue.get_nowait())
            except Exception:
                break

        result_queue.close()
        result_queue.join_thread()

        successful = sum(1 for r in results if r.get("success"))
        self.logger.info(
            f"Round {round_num}: {successful}/{len(candidates)} workers succeeded "
            f"({len(results)} results received)"
        )

        return results


# ---------------------------------------------------------------------------
# Module-level worker process target (must be picklable)
# ---------------------------------------------------------------------------


def _kubectl_worker_process(
    worker_id: int,
    kernel_code: str,
    known_time: float,
    parent_id: str,
    problem_file: Path,
    test_code: str,
    workdir: Path,
    log_dir: Path,
    result_queue: mp.Queue,
    benchmark_lock: Any,
    profiling_semaphore: Any,
    pytorch_baseline: float,
    bottleneck_id: int,
    openai_model: str,
    high_reasoning_effort: bool,
    bottleneck_override: str | None,
    worker_kwargs: dict,
    prior_history: list[dict],
    prior_reflexions: list[dict],
    kubectl_config: KubectlConfig | None,
) -> None:
    """Worker process for kubectl platform.

    Workers run locally (for LLM access); GPU operations are sent to the
    pod via kubectl inside the worker's benchmarker and verifier.
    """
    import sys

    kernel_agent_path = Path(__file__).parent.parent.parent
    if str(kernel_agent_path) not in sys.path:
        sys.path.insert(0, str(kernel_agent_path))

    try:
        from triton_kernel_agent.opt_worker import OptimizationWorker

        workdir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(problem_file, workdir / "problem.py")

        # Remove kubectl_config from worker_kwargs if present to avoid
        # duplicate keyword argument (we pass it explicitly below).
        worker_kwargs.pop("kubectl_config", None)

        worker = OptimizationWorker(
            worker_id=worker_id,
            workdir=workdir,
            log_dir=log_dir,
            openai_model=openai_model,
            high_reasoning_effort=high_reasoning_effort,
            bottleneck_id=bottleneck_id,
            benchmark_lock=benchmark_lock,
            profiling_semaphore=profiling_semaphore,
            pytorch_baseline_time=pytorch_baseline,
            bottleneck_override=bottleneck_override,
            prior_history=prior_history,
            prior_reflexions=prior_reflexions,
            kubectl_config=kubectl_config,
            **worker_kwargs,
        )

        success, best_kernel, metrics = worker.optimize_kernel(
            kernel_code=kernel_code,
            problem_file=problem_file,
            test_code=test_code,
            known_kernel_time=known_time,
            max_opt_rounds=1,
        )

        attempt_data = metrics.get("last_attempt")
        reflexion_data = metrics.get("last_reflexion")

        result_queue.put(
            {
                "success": success,
                "worker_id": worker_id,
                "kernel_code": best_kernel,
                "time_ms": metrics.get("best_time_ms", float("inf")),
                "parent_id": parent_id,
                "attempt": attempt_data,
                "reflexion": reflexion_data,
            }
        )

    except Exception as e:
        result_queue.put(
            {
                "success": False,
                "worker_id": worker_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )


# =========================================================================
# Worker-level components
# =========================================================================


class KubectlKernelProfiler(KernelProfilerBase):
    """Profiles kernels on a remote GPU pod by running NCU via kubectl exec.

    Steps:
    1. Generate an NCU wrapper script with pod-relative paths (all files in
       the same remote temp directory).
    2. Copy kernel, problem, and wrapper files to the pod.
    3. Run NCU via ``kubectl exec``.
    4. Copy the CSV results back to the local artifacts directory.
    5. Parse metrics locally using ``load_ncu_metrics`` / ``metrics_to_prompt``
       (pure pandas — no GPU needed).
    """

    # Default timeouts
    NCU_TIMEOUT_SECONDS = 360
    SEMAPHORE_TIMEOUT_SECONDS = 900

    def __init__(
        self,
        kubectl_config: KubectlConfig | None = None,
        logger: logging.Logger | None = None,
        artifacts_dir: Path | None = None,
        logs_dir: Path | None = None,
        profiling_semaphore: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._config = kubectl_config or KubectlConfig()
        self._logger = logger or logging.getLogger(__name__)
        self._executor = KubectlExecutor(self._config, self._logger)
        self.artifacts_dir = artifacts_dir or Path(".")
        self.logs_dir = logs_dir or Path(".")
        self.profiling_semaphore = profiling_semaphore

        # Lazy-load the Jinja2 template from NCUWrapperFactory
        from triton_kernel_agent.opt_worker_component.profiling.ncu_wrapper_factory import (
            NCUWrapperFactory,
        )

        self._wrapper_factory = NCUWrapperFactory(self._logger)

    def _generate_pod_wrapper(self, output_dir: Path) -> Path:
        """Generate an NCU wrapper script with pod-relative paths.

        All files (kernel.py, problem.py, ncu_wrapper.py) will be in the
        same directory on the pod, so we use ``'.'`` as the path prefix.
        """
        wrapper_file = output_dir / "ncu_wrapper.py"
        wrapper_content = self._wrapper_factory.template.render(
            kernel_file_parent="'.'",
            problem_file_parent="'.'",
            kernel_module="kernel",
            problem_module="problem",
            dtype_inference=True,
            model_extraction=True,
        )
        wrapper_file.write_text(wrapper_content)
        self._logger.debug(f"Generated pod NCU wrapper: {wrapper_file}")
        return wrapper_file

    def _build_ncu_command(self, remote_dir: str) -> str:
        """Build the NCU command string to execute on the pod."""
        from kernel_perf_agent.kernel_opt.profiler.ncu_profiler import METRICS

        ncu_bin = self._config.ncu_bin_path
        python = self._config.python_path
        csv_path = f"{remote_dir}/ncu_output.csv"
        wrapper_path = f"{remote_dir}/ncu_wrapper.py"

        return (
            f"{ncu_bin} --csv --page=raw --kernel-name-base=demangled "
            f"--target-processes=all --replay-mode=kernel "
            f"--profile-from-start=on "
            f"--log-file={csv_path} "
            f"--metrics={METRICS} "
            f"--launch-skip=3 --launch-count=20 "
            f"{python} {wrapper_path}"
        )

    def _wait_with_backoff(self, attempt: int) -> None:
        """Wait with exponential backoff before retrying."""
        wait_time = 2**attempt
        self._logger.warning(f"Retrying in {wait_time}s...")
        time.sleep(wait_time)

    def profile_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int = 2,
    ) -> Any | None:
        """Profile kernel with NCU on the remote GPU pod.

        Acquires the profiling semaphore (if set) since NCU requires
        exclusive GPU access.

        Returns:
            ``ProfilerResults`` or ``None`` on failure.
        """
        # Acquire profiling semaphore (NCU needs exclusive GPU access)
        semaphore_acquired = False
        if self.profiling_semaphore is not None:
            self._logger.info(
                f"[Round {round_num}] Waiting for profiling semaphore..."
            )
            semaphore_acquired = self.profiling_semaphore.acquire(
                timeout=self.SEMAPHORE_TIMEOUT_SECONDS
            )
            if not semaphore_acquired:
                self._logger.warning(
                    f"[Round {round_num}] Semaphore timeout after "
                    f"{self.SEMAPHORE_TIMEOUT_SECONDS}s, skipping profiling"
                )
                return None
            self._logger.info(
                f"[Round {round_num}] Acquired profiling semaphore"
            )

        try:
            return self._profile_kernel_impl(
                kernel_file, problem_file, round_num, max_retries
            )
        finally:
            if semaphore_acquired:
                self.profiling_semaphore.release()
                self._logger.debug(
                    f"[Round {round_num}] Released profiling semaphore"
                )

    def _profile_kernel_impl(
        self,
        kernel_file: Path,
        problem_file: Path,
        round_num: int,
        max_retries: int,
    ) -> Any | None:
        """Internal profiling implementation (called with semaphore held)."""
        from datetime import datetime

        from kernel_perf_agent.kernel_opt.profiler.ncu_profiler import (
            load_ncu_metrics,
            metrics_to_prompt,
        )
        from triton_kernel_agent.opt_worker_component.profiling.kernel_profiler import (
            ProfilerMetadata,
            ProfilerResults,
        )

        for attempt in range(1, max_retries + 1):
            remote_dir = ""
            try:
                self._logger.info(
                    f"[Round {round_num}] NCU profiling attempt "
                    f"{attempt}/{max_retries} (kubectl)..."
                )

                # 1. Create remote temp dir
                remote_dir = self._executor.mkdtemp()

                # 2. Generate wrapper and copy files to pod
                with tempfile.TemporaryDirectory() as local_tmp:
                    local_tmp_path = Path(local_tmp)
                    wrapper_file = self._generate_pod_wrapper(local_tmp_path)

                    self._executor.copy_to(
                        kernel_file, f"{remote_dir}/kernel.py"
                    )
                    self._executor.copy_to(
                        problem_file, f"{remote_dir}/problem.py"
                    )
                    self._executor.copy_to(
                        wrapper_file, f"{remote_dir}/ncu_wrapper.py"
                    )

                # 3. Run NCU on the pod
                ncu_cmd = self._build_ncu_command(remote_dir)
                self._logger.info(
                    f"[Round {round_num}] Running NCU on pod: "
                    f"{ncu_cmd[:120]}..."
                )
                rc, stdout, stderr = self._executor.exec(
                    ncu_cmd, timeout=self.NCU_TIMEOUT_SECONDS
                )

                if rc != 0:
                    # Filter kubectl warnings from real errors
                    real_stderr = "\n".join(
                        line
                        for line in stderr.splitlines()
                        if "Defaulted container" not in line
                    ).strip()
                    raise RuntimeError(
                        f"NCU failed (rc={rc}): {real_stderr[:500]}"
                    )

                # 4. Copy CSV back to local artifacts
                csv_filename = f"ncu_round_{round_num}.csv"
                local_csv = self.artifacts_dir / csv_filename
                self._executor.copy_from(
                    f"{remote_dir}/ncu_output.csv", local_csv
                )

                if not local_csv.exists() or local_csv.stat().st_size < 100:
                    raise RuntimeError(
                        f"NCU CSV missing or too small: {local_csv}"
                    )

                # 5. Parse metrics locally (pure pandas, no GPU)
                metrics_df = load_ncu_metrics(local_csv, select="last")
                metrics = json.loads(metrics_to_prompt(metrics_df))

                # 6. Build and return ProfilerResults
                results = ProfilerResults(
                    metrics_df=metrics_df,
                    metrics=metrics,
                    metadata=ProfilerMetadata(
                        kernel_file=str(kernel_file),
                        problem_file=str(problem_file),
                        round_num=round_num,
                        timestamp=datetime.utcnow().isoformat() + "Z",
                        ncu_version=None,
                    ),
                )

                # Save metrics JSON
                metrics_file = (
                    self.logs_dir
                    / f"round{round_num:03d}_ncu_metrics.json"
                )
                with open(metrics_file, "w") as f:
                    f.write(results.to_json())

                self._logger.info(
                    f"NCU profiling completed for round {round_num} (kubectl)"
                )
                return results

            except subprocess.TimeoutExpired:
                if attempt >= max_retries:
                    self._logger.error(
                        f"NCU profiling timed out after "
                        f"{self.NCU_TIMEOUT_SECONDS}s "
                        f"(final attempt {attempt}/{max_retries}, kubectl)"
                    )
                    return None
                self._logger.debug(
                    f"NCU timed out (attempt {attempt}/{max_retries})"
                )
                self._wait_with_backoff(attempt)

            except json.JSONDecodeError as e:
                if attempt >= max_retries:
                    self._logger.error(
                        f"Failed to parse NCU metrics "
                        f"(final attempt, kubectl): {e}"
                    )
                    return None
                self._logger.debug(
                    f"NCU metrics parse error "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
                self._wait_with_backoff(attempt)

            except Exception as e:
                if attempt >= max_retries:
                    self._logger.error(
                        f"NCU profiling error "
                        f"(final attempt, kubectl): {e}",
                        exc_info=True,
                    )
                    return None
                self._logger.debug(
                    f"NCU error (attempt {attempt}/{max_retries}): {e}"
                )
                self._wait_with_backoff(attempt)

            finally:
                if remote_dir:
                    self._executor.rm(remote_dir)

        self._logger.error(
            f"NCU profiling failed after {max_retries} attempts "
            f"for round {round_num} (kubectl)"
        )
        return None


class KubectlRooflineAnalyzer(RooflineAnalyzerBase):
    """Roofline analyzer for kubectl platform.

    Delegates to the real ``RooflineAnalyzer`` which is pure computation
    (no GPU needed) — it just needs NCU metric values as input.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        roofline_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import (
            RooflineAnalyzer,
        )

        self._delegate = RooflineAnalyzer(
            config=roofline_config, logger=logger
        )

    def analyze(self, ncu_metrics: dict[str, Any]) -> Any:
        return self._delegate.analyze(ncu_metrics)

    def should_stop(self, result: Any = None) -> tuple[bool, str]:
        return self._delegate.should_stop(result)

    def reset_history(self) -> None:
        self._delegate.reset_history()


class KubectlBottleneckAnalyzer(BottleneckAnalyzerBase):
    """LLM-based bottleneck analyzer that works with or without NCU metrics.

    Delegates to the real ``BottleneckAnalyzer``. When GPU specs are
    provided (from ``KubectlAcceleratorSpecsProvider``), the LLM gets
    real hardware info for better analysis. When NCU profiling is enabled,
    real NCU metrics and roofline analysis are available.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_dir: Path | None = None,
        openai_model: str = "claude-opus-4.5",
        kubectl_config: KubectlConfig | None = None,
        gpu_specs: dict[str, Any] | None = None,
        roofline_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._log_dir = Path(log_dir) if log_dir else None
        self._openai_model = openai_model
        self._gpu_specs = gpu_specs
        self._delegate: Any | None = None
        self.roofline = KubectlRooflineAnalyzer(
            logger=self._logger, roofline_config=roofline_config
        )

    def _get_delegate(self) -> Any:
        if self._delegate is None:
            from triton_kernel_agent.opt_worker_component.prescribing.bottleneck_analyzer import (
                BottleneckAnalyzer,
            )
            from utils.providers import get_model_provider

            provider = get_model_provider(self._openai_model)
            gpu_specs = self._gpu_specs or {
                "name": "remote (kubectl)",
                "architecture": "unknown",
                "peak_fp32_tflops": 0.0,
                "peak_memory_bw_gbps": 0.0,
                "sm_count": 0,
            }

            self._delegate = BottleneckAnalyzer(
                provider=provider,
                model=self._openai_model,
                gpu_specs=gpu_specs,
                logs_dir=self._log_dir,
                logger=self._logger,
            )
        return self._delegate

    def analyze(
        self,
        kernel_code: str,
        ncu_metrics: dict[str, Any],
        round_num: int = 0,
        roofline_result: Any | None = None,
    ) -> list[Any]:
        return self._get_delegate().analyze(
            kernel_code, ncu_metrics, round_num, roofline_result
        )


class KubectlAcceleratorSpecsProvider(AcceleratorSpecsProvider):
    """Queries the GPU pod for device name, then looks up specs."""

    def __init__(
        self,
        kubectl_config: KubectlConfig | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> None:
        self._config = kubectl_config or KubectlConfig()
        self._logger = logger or logging.getLogger(__name__)
        self._executor = KubectlExecutor(self._config, self._logger)

    def get_specs(self, device_name: str | None = None) -> dict[str, Any]:
        if device_name is None:
            try:
                python = self._config.python_path
                rc, stdout, stderr = self._executor.exec(
                    f'{python} -c "import torch; print(torch.cuda.get_device_name(0))"',
                    timeout=60,
                )
                if rc == 0 and stdout.strip():
                    device_name = stdout.strip()
                    self._logger.info(
                        f"Detected GPU on pod: {device_name}"
                    )
            except Exception as e:
                self._logger.warning(f"Failed to detect GPU on pod: {e}")

        if device_name:
            try:
                from kernel_perf_agent.kernel_opt.diagnose_prompt.gpu_specs import (
                    get_gpu_specs,
                )

                specs = get_gpu_specs(device_name)
                if specs is not None:
                    return specs
                self._logger.warning(
                    f"GPU '{device_name}' not in specs database, using stub"
                )
            except Exception as e:
                self._logger.warning(
                    f"GPU specs lookup failed for '{device_name}': {e}"
                )

        return {
            "name": device_name or "remote (kubectl)",
            "architecture": "unknown",
            "peak_fp32_tflops": 0.0,
            "peak_memory_bw_gbps": 0.0,
            "sm_count": 0,
        }


# =========================================================================
# Worker-level GPU execution subclasses
# (Not in registry — created directly by OptimizationWorker)
# =========================================================================


class KubectlBenchmark:
    """Drop-in replacement for ``Benchmark`` that runs on a GPU pod.

    Only overrides ``benchmark_kernel``; PyTorch benchmarks are handled
    at the manager level by ``KubectlBenchmarker``.
    """

    def __init__(
        self,
        kubectl_config: KubectlConfig,
        benchmark_lock: Any,
        logger: logging.Logger,
        worker_id: int = 0,
        artifacts_dir: Path | None = None,
        warmup: int = 25,
        repeat: int = 100,
        **kwargs: Any,
    ) -> None:
        self.logger = logger
        self.warmup = warmup
        self.repeat = repeat
        self._executor = KubectlExecutor(kubectl_config, logger)
        self.artifacts_dir = artifacts_dir or Path(".")

        from triton_kernel_agent.opt_worker_component.benchmarking.benchmark import (
            BenchmarkLockManager,
        )

        self.lock_manager = BenchmarkLockManager(benchmark_lock, worker_id, logger)

    def benchmark_kernel(
        self,
        kernel_file: Path,
        problem_file: Path,
        baseline_file: Path | None = None,
    ) -> dict[str, Any]:
        """Benchmark a kernel on the GPU pod."""
        remote_dir = ""
        try:
            with self.lock_manager:
                remote_dir = self._executor.mkdtemp()

                with tempfile.TemporaryDirectory() as local_tmp:
                    local_tmp_path = Path(local_tmp)

                    # Copy timing.py and kernel_subprocess.py
                    (local_tmp_path / "timing.py").write_text(
                        _get_timing_py_source(), encoding="utf-8"
                    )
                    (local_tmp_path / "kernel_subprocess.py").write_text(
                        _get_kernel_subprocess_source(), encoding="utf-8"
                    )

                    self._executor.copy_to(
                        kernel_file, f"{remote_dir}/kernel.py"
                    )
                    self._executor.copy_to(
                        problem_file, f"{remote_dir}/problem.py"
                    )
                    self._executor.copy_to(
                        local_tmp_path / "timing.py", f"{remote_dir}/timing.py"
                    )
                    self._executor.copy_to(
                        local_tmp_path / "kernel_subprocess.py",
                        f"{remote_dir}/kernel_subprocess.py",
                    )

                    python = self._executor._config.python_path
                    rc, stdout, stderr = self._executor.exec(
                        f"cd {remote_dir} && {python} kernel_subprocess.py "
                        f"--kernel kernel.py --problem problem.py "
                        f"--warmup {self.warmup} --repeat {self.repeat} "
                        f"--json results.json --quiet",
                        timeout=300,
                    )

                    if rc != 0:
                        self.logger.error(
                            f"Kernel benchmark failed on pod: {stderr[:500]}"
                        )
                        return {"time_ms": float("inf"), "speedup": 0.0}

                    local_result = local_tmp_path / "results.json"
                    self._executor.copy_from(
                        f"{remote_dir}/results.json", local_result
                    )
                    results = json.loads(
                        local_result.read_text(encoding="utf-8")
                    )

                # We always copy the file as kernel.py to the pod,
                # so the results are keyed under "kernel" regardless
                # of the original local filename.
                kernel_results = results.get("kernels", {}).get("kernel", {})

                return {
                    "time_ms": kernel_results.get("time_ms", float("inf")),
                    "speedup": kernel_results.get("speedup", 1.0),
                }

        except Exception as e:
            self.logger.error(f"Kubectl kernel benchmark failed: {e}")
            return {"time_ms": float("inf"), "speedup": 0.0}
        finally:
            if remote_dir:
                self._executor.rm(remote_dir)

    # Stubs matching the Benchmark interface used by the orchestrator
    def benchmark_pytorch(self, problem_file: Path, **kwargs: Any) -> dict[str, Any]:
        """Not used at worker level — manager handles PyTorch benchmarks."""
        return {"time_ms": float("inf")}

    def benchmark_pytorch_compile(
        self, problem_file: Path, **kwargs: Any
    ) -> dict[str, Any]:
        """Not used at worker level — manager handles compile benchmarks."""
        return {"time_ms": float("inf")}


class KubectlVerificationWorker:
    """Drop-in replacement for ``VerificationWorker`` that runs tests on a GPU pod.

    Only overrides ``_run_test()``; LLM-based refinement still runs locally.
    Inherits from the real ``VerificationWorker`` to keep refinement logic.
    """

    def __init__(
        self,
        kubectl_config: KubectlConfig,
        worker_id: int = 0,
        workdir: Path | None = None,
        log_dir: Path | None = None,
        openai_model: str = "claude-opus-4.5",
        high_reasoning_effort: bool = True,
        target_platform: str = "cuda",
        test_timeout_s: int = 60,
        **kwargs: Any,
    ) -> None:
        # Import here to avoid circular imports
        from triton_kernel_agent.worker import VerificationWorker

        self._delegate = VerificationWorker(
            worker_id=worker_id,
            workdir=workdir or Path("."),
            log_dir=log_dir or Path("."),
            openai_model=openai_model,
            high_reasoning_effort=high_reasoning_effort,
            target_platform=target_platform,
            test_timeout_s=test_timeout_s,
            **kwargs,
        )
        self._kubectl_config = kubectl_config
        self._executor = KubectlExecutor(
            kubectl_config,
            self._delegate.logger,
        )
        self._test_timeout_s = test_timeout_s

        # Monkey-patch _run_test on the delegate so that all existing
        # methods (verify_with_refinement, etc.) use kubectl exec
        self._delegate._run_test = self._run_test

    def _run_test(self) -> tuple[bool, str, str]:
        """Run the test script on the GPU pod instead of locally."""
        remote_dir = ""
        try:
            remote_dir = self._executor.mkdtemp()

            # Copy kernel, test, and problem files
            workdir = self._delegate.workdir
            kernel_file = workdir / "kernel.py"
            test_file = self._delegate.test_file
            problem_file = workdir / "problem.py"

            if kernel_file.exists():
                self._executor.copy_to(kernel_file, f"{remote_dir}/kernel.py")
            if test_file.exists():
                self._executor.copy_to(
                    test_file, f"{remote_dir}/{test_file.name}"
                )
            if problem_file.exists():
                self._executor.copy_to(
                    problem_file, f"{remote_dir}/problem.py"
                )

            python = self._executor._config.python_path
            rc, stdout, stderr = self._executor.exec(
                f"cd {remote_dir} && {python} {test_file.name}",
                timeout=self._test_timeout_s,
            )

            if rc == 0:
                self._delegate.logger.info("Test passed (kubectl)")
            else:
                self._delegate.logger.error(
                    f"Test failed (kubectl). rc={rc}, stderr: {stderr[:2000]}"
                )

            return rc == 0, stdout, stderr

        except subprocess.TimeoutExpired:
            self._delegate.logger.error("Test timed out (kubectl)")
            return (
                False,
                "",
                f"Test execution timed out after {self._test_timeout_s}s",
            )
        except Exception as e:
            self._delegate.logger.error(f"Test execution error (kubectl): {e}")
            return False, "", str(e)
        finally:
            if remote_dir:
                self._executor.rm(remote_dir)

    # Proxy all other attribute access to the delegate
    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)
