# kernel-optimizer

Claude Code plugin for Triton kernel optimization using AI-powered subagents and remote GPU pods via kubectl.

## Overview

This plugin packages Meta's KernelAgent optimization pipeline as a Claude Code plugin. Instead of running a Python process that makes external LLM API calls, Claude Code itself orchestrates the optimization loop:

- **Claude Code subagents** handle all LLM reasoning (bottleneck analysis, kernel generation, refinement, reflexion)
- **kubectl** handles all GPU operations (verification, benchmarking, NCU profiling) on a remote Kubernetes pod
- **Beam search strategy** maintains a beam of top-performing kernels, generating N×M optimization candidates per round

## Prerequisites

- Access to a Kubernetes pod with an NVIDIA GPU (H100 recommended)
- `kubectl` configured and authenticated
- Python environment with the KernelAgent dependencies installed

## Environment Setup

```bash
export KUBECTL_POD_NAME=<your-gpu-pod>
export KUBECTL_NAMESPACE=<namespace>
# Optional:
export KUBECTL_CONTAINER=<container-name>
```

## Usage

```
/kernel-optimizer:optimize <path-to-kernel-directory>
```

The kernel directory must contain:
- `problem.py` — defines the computation (Model class, get_inputs, get_init_inputs)
- `input_kernel.py` (or `input.py`) — the initial Triton kernel to optimize

## How It Works

### Optimization Loop (per round)

1. **Profile** top kernels with NCU on the GPU pod
2. **Analyze** bottlenecks via parallel subagents (bottleneck-analyzer)
3. **Generate** optimized kernels via parallel subagents (kernel-optimizer)
4. **Verify** correctness on GPU pod, with auto-refinement (kernel-refiner) on failure
5. **Benchmark** verified kernels on GPU pod
6. **Reflect** on results via parallel subagents (reflexion)
7. **Update beam** — keep top-N kernels, repeat

### Architecture

```
Claude Code (orchestrator)
├── bottleneck-analyzer subagent  ── reads NCU metrics → writes bottleneck JSON
├── kernel-optimizer subagent     ── reads bottleneck + history → writes optimized .py
├── kernel-refiner subagent       ── reads error output → writes fixed .py
├── reflexion subagent            ── reads attempt results → writes lessons JSON
└── gpu_ops.py (Bash)             ── kubectl exec/cp → GPU pod
```

## Plugin Structure

```
kernel-optimizer/
├── .claude-plugin/plugin.json    # Plugin manifest
├── commands/optimize.md          # Entry point: /kernel-optimizer:optimize
├── skills/kernel-optimization/
│   └── SKILL.md                  # Main orchestration (7 phases)
├── agents/
│   ├── bottleneck-analyzer.md    # NCU metrics analysis
│   ├── kernel-optimizer.md       # Triton kernel generation
│   ├── kernel-refiner.md         # Fix broken kernels
│   └── reflexion.md              # Optimization attempt reflection
├── scripts/
│   └── gpu_ops.py                # kubectl GPU operations CLI
└── templates/
    └── default_test.py           # Default correctness test harness
```

## Code Reuse

The plugin reuses existing KernelAgent code with zero modifications:

| Component | Reuse |
|---|---|
| `platform/kubectl.py` (KubectlExecutor, Verifier, Benchmarker, Profiler) | Imported by gpu_ops.py |
| `ncu_profiler.py` (load_ncu_metrics, metrics_to_prompt) | Imported by gpu_ops.py |
| `ncu_roofline.py` (RooflineAnalyzer) | Imported by gpu_ops.py |
| `gpu_specs.py` / `gpu_specs_database.py` | Imported by gpu_ops.py |
| Jinja2 templates (kernel_optimization.j2, etc.) | Replaced by agent definitions |
| OptimizationManager / OptimizationOrchestrator | Replaced by SKILL.md |
