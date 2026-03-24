---
description: Orchestrates Triton kernel optimization using Claude Code subagents and kubectl GPU pods
inputs:
  - name: kernel_dir
    description: Path to directory containing problem.py and input_kernel.py (or input.py)
    required: true
---

# Kernel Optimization Skill

This skill orchestrates the full KernelAgent optimization pipeline using Claude Code
subagents for LLM reasoning and kubectl for GPU operations. It implements a beam search
strategy where each round profiles the best kernels, generates optimized variants via
parallel subagents, verifies and benchmarks them on a GPU pod, and reflects on results.

## Configuration Defaults

- `max_rounds`: 10
- `num_top_kernels`: 2 (beam width — top N kernels to profile each round)
- `num_bottlenecks`: 2 (per kernel — generates N × M workers per round)
- `strategy`: beam_search
- `max_refinement_attempts`: 3

## Environment Requirements

Set these before running:
```
KUBECTL_POD_NAME=<pod-name>        # or KUBECTL_LABEL_SELECTOR
KUBECTL_NAMESPACE=<namespace>       # default: "default"
KUBECTL_CONTAINER=<container>       # optional
```

## Plugin Paths

- `GPU_OPS`: `${CLAUDE_PLUGIN_ROOT}/scripts/gpu_ops.py`
- `TEST_TEMPLATE`: `${CLAUDE_PLUGIN_ROOT}/templates/default_test.py`
- Agents: `${CLAUDE_PLUGIN_ROOT}/agents/`

---

# Phase 1: Setup and Validation

1. **Validate kernel directory**:
   - Read `kernel_dir` from the initial request arguments
   - Verify `problem.py` exists in the directory
   - Find the kernel file: try `input_kernel.py` first, then `input.py`
   - If a custom test file exists (`test.py` or `test_kernel.py`), use it; otherwise use the default test template

2. **Create state directory**:
   ```
   state_dir = <kernel_dir>/optimization_state/
   ```
   Create subdirectories: `kernels/`, and per-round directories will be created as needed.

3. **Copy initial kernel**:
   - Copy the input kernel to `state_dir/kernels/initial.py`
   - Read and store the kernel code for later use

4. **Detect GPU on pod**:
   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_ops.py detect-gpu
   ```
   Save the JSON output to `state_dir/gpu_specs.json`.

5. **Read problem description**: Read `problem.py` and extract the docstring or class description for use in optimization prompts.

---

# Phase 2: Baselines

1. **Verify initial kernel passes correctness tests**:
   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_ops.py verify \
       --kernel-file state_dir/kernels/initial.py \
       --problem-file <kernel_dir>/problem.py \
       --test-file <test_file>
   ```
   If this fails, STOP and report the error. The initial kernel must pass tests.

2. **Benchmark the initial kernel**:
   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_ops.py benchmark \
       --kernel-file state_dir/kernels/initial.py \
       --problem-file <kernel_dir>/problem.py
   ```
   Record `initial_kernel_ms`.

3. **Benchmark PyTorch eager baseline**:
   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_ops.py benchmark-reference \
       --problem-file <kernel_dir>/problem.py
   ```
   Record `pytorch_eager_ms`.

4. **Benchmark PyTorch compile baseline**:
   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_ops.py benchmark-reference \
       --problem-file <kernel_dir>/problem.py --compile
   ```
   Record `pytorch_compile_ms`.

5. **Save baselines**:
   Write `state_dir/baselines.json`:
   ```json
   {
       "initial_kernel_ms": <value>,
       "pytorch_eager_ms": <value>,
       "pytorch_compile_ms": <value>
   }
   ```

---

# Phase 3: Initialize State

Write these state files:

1. **`state_dir/config.json`**:
   ```json
   {
       "max_rounds": 10,
       "num_top_kernels": 2,
       "num_bottlenecks": 2,
       "strategy": "beam_search",
       "max_refinement_attempts": 3
   }
   ```

2. **`state_dir/database.json`** — the beam of top kernels:
   ```json
   [
       {
           "id": "initial",
           "kernel_path": "kernels/initial.py",
           "time_ms": <initial_kernel_ms>,
           "generation": 0,
           "parent_id": null
       }
   ]
   ```

3. **`state_dir/round_num.txt`**: Write `0`

4. **`state_dir/history.json`**: Write `[]`

5. **`state_dir/reflexions.json`**: Write `[]`

---

# Phase 4: Optimization Round

Read current state:
- `round_num` from `round_num.txt` → increment to get current round
- `database.json` → top kernels (sorted by time_ms ascending)
- `config.json` → num_top_kernels, num_bottlenecks
- `history.json` → recent optimization attempts (last 10)
- `reflexions.json` → recent reflexions (last 10)
- `gpu_specs.json` → GPU specifications
- `baselines.json` → baseline times

Create round directory: `state_dir/round_<N>/`

## Step 4a: Select Candidates

Using beam search: for each of the top-N kernels × M bottleneck IDs (1..M), create a candidate list:
```
candidates = []
for i, kernel in enumerate(top_kernels[:num_top_kernels]):
    for bottleneck_id in range(num_bottlenecks):
        candidates.append({
            "kernel_id": kernel["id"],
            "kernel_path": kernel["kernel_path"],
            "bottleneck_id": bottleneck_id,
            "worker_id": len(candidates)
        })
```

## Step 4b: Profile Parent Kernels (SERIAL — GPU)

For each unique parent kernel in the candidate list, profile it with NCU:

```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_ops.py profile \
    --kernel-file state_dir/<kernel_path> \
    --problem-file <kernel_dir>/problem.py \
    --round <N> \
    --artifacts-dir state_dir/round_<N>/
```

Then run roofline analysis on the metrics:
```bash
python ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_ops.py roofline \
    --ncu-metrics-file state_dir/round_<N>/round<NNN>_ncu_metrics.json
```

## Step 4c: Analyze Bottlenecks (PARALLEL — Subagents)

For each unique parent kernel, launch a **bottleneck-analyzer** subagent via the Agent tool.

Each subagent receives:
- NCU metrics file path: `state_dir/round_<N>/round<NNN>_ncu_metrics.json`
- Roofline analysis file path: `state_dir/round_<N>/round<NNN>_roofline.json`
- Kernel source file path: `state_dir/<kernel_path>`
- GPU specs file path: `state_dir/gpu_specs.json`
- Output path: `state_dir/round_<N>/bottleneck_<kernel_id>.json`
- `num_bottlenecks`: from config

**Launch all bottleneck analyzers in a single message for parallelism.**

Subagent prompt template:
```
You are a bottleneck-analyzer subagent. Follow the instructions in
${CLAUDE_PLUGIN_ROOT}/agents/bottleneck-analyzer.md.

Analyze the NCU profiling metrics and identify performance bottlenecks.

Inputs:
- NCU metrics: <path>
- Roofline analysis: <path>
- Kernel source: <path>
- GPU specs: <path>
- num_bottlenecks: <N>

Write your analysis as a JSON array to: <output_path>
```

## Step 4d: Generate Optimized Kernels (PARALLEL — Subagents)

For each candidate (kernel × bottleneck_id), launch a **kernel-optimizer** subagent.

Each subagent receives:
- Problem file: `<kernel_dir>/problem.py`
- Current kernel: `state_dir/<kernel_path>`
- Bottleneck analysis: `state_dir/round_<N>/bottleneck_<kernel_id>.json`
- GPU specs: `state_dir/gpu_specs.json`
- Roofline analysis: `state_dir/round_<N>/round<NNN>_roofline.json`
- History: `state_dir/history.json`
- Reflexions: `state_dir/reflexions.json`
- `bottleneck_id`: which bottleneck from the array to target
- `pytorch_baseline_ms`: from baselines
- `current_best_ms`: best time from database
- Output path: `state_dir/kernels/r<N>_w<worker_id>.py`

**Launch ALL kernel-optimizer subagents in a single message for maximum parallelism.**

Subagent prompt template:
```
You are a kernel-optimizer subagent. Follow the instructions in
${CLAUDE_PLUGIN_ROOT}/agents/kernel-optimizer.md.

Generate an optimized Triton kernel targeting bottleneck #<bottleneck_id>.

Inputs:
- Problem: <path>
- Current kernel: <path>
- Bottleneck analysis: <path> (target bottleneck index: <bottleneck_id>)
- GPU specs: <path>
- Roofline: <path>
- History: <path>
- Reflexions: <path>
- PyTorch baseline: <value> ms
- Current best: <value> ms

Write the complete optimized kernel to: <output_path>
```

## Step 4e: Verify and Benchmark (SERIAL — GPU)

For each subagent output kernel, sequentially:

1. **Verify correctness**:
   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_ops.py verify \
       --kernel-file state_dir/kernels/r<N>_w<W>.py \
       --problem-file <kernel_dir>/problem.py \
       --test-file <test_file>
   ```

2. **If verification fails**, launch a **kernel-refiner** subagent (up to 3 attempts):
   - Read the error output from the verify command
   - Launch subagent with the failed kernel, test code, problem, and error
   - Re-verify the fixed kernel
   - If all 3 attempts fail, mark this worker as failed and continue

   Subagent prompt template:
   ```
   You are a kernel-refiner subagent. Follow the instructions in
   ${CLAUDE_PLUGIN_ROOT}/agents/kernel-refiner.md.

   Fix this kernel that failed verification.

   Inputs:
   - Failed kernel: <path>
   - Test code: <path>
   - Problem: <path>
   - Error stdout: <stdout>
   - Error stderr: <stderr>
   - Previous refinement attempts: <count>

   Write the fixed kernel to: <output_path>
   ```

3. **Benchmark the verified kernel**:
   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_ops.py benchmark \
       --kernel-file state_dir/kernels/r<N>_w<W>.py \
       --problem-file <kernel_dir>/problem.py
   ```

4. **Profile for SOL metrics** (for reflexion):
   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_ops.py profile \
       --kernel-file state_dir/kernels/r<N>_w<W>.py \
       --problem-file <kernel_dir>/problem.py \
       --round <N> \
       --artifacts-dir state_dir/round_<N>/
   ```
   ```bash
   python ${CLAUDE_PLUGIN_ROOT}/scripts/gpu_ops.py roofline \
       --ncu-metrics-file state_dir/round_<N>/round<NNN>_ncu_metrics.json
   ```

5. **Record worker result** in `state_dir/round_<N>/worker_<W>_result.json`:
   ```json
   {
       "worker_id": <W>,
       "kernel_id": "r<N>_w<W>",
       "kernel_path": "kernels/r<N>_w<W>.py",
       "parent_id": "<parent_kernel_id>",
       "passed_verification": true,
       "time_ms": <benchmark_time>,
       "bottleneck_category": "<from bottleneck analysis>",
       "root_cause": "<from bottleneck analysis>",
       "recommended_fix": "<from bottleneck analysis>",
       "compute_sol_pct": <from roofline>,
       "memory_sol_pct": <from roofline>,
       "improvement_pct": <calculated>,
       "is_improvement": <boolean>
   }
   ```

## Step 4f: Reflexion (PARALLEL — Subagents)

For each completed worker, launch a **reflexion** subagent.

Each subagent receives:
- Attempt result: `state_dir/round_<N>/worker_<W>_result.json`
- Output path: `state_dir/round_<N>/worker_<W>_reflexion.json`

**Launch all reflexion subagents in a single message for parallelism.**

Subagent prompt template:
```
You are a reflexion subagent. Follow the instructions in
${CLAUDE_PLUGIN_ROOT}/agents/reflexion.md.

Analyze this optimization attempt and extract lessons learned.

Input: <result_path>
Write reflexion JSON to: <output_path>
```

---

# Phase 5: Update State

1. **Update database.json** — beam search update:
   - Combine existing database entries with new successful workers
   - Sort by `time_ms` ascending
   - Keep top `num_top_kernels` entries
   - This is the updated beam for the next round

2. **Update history.json**:
   - Append all worker results from this round
   - Keep only the last 10 entries

3. **Update reflexions.json**:
   - Append all reflexion results from this round
   - Keep only the last 10 entries

4. **Update round_num.txt**: Write the current round number

5. **Log round summary**:
   - Report: round number, number of workers, best time this round, overall best time
   - Report improvement over initial kernel and PyTorch baselines

---

# Phase 6: Loop Decision

Check termination conditions:

1. **Max rounds reached**: If `round_num >= max_rounds`, go to Phase 7
2. **Convergence**: If the best kernel has not improved for 3 consecutive rounds, go to Phase 7
3. **Target achieved**: If best kernel is >2x faster than PyTorch eager, go to Phase 7

If none of these conditions are met, **go back to Phase 4** for the next round.

**Important**: When looping, re-read all state files at the start of Phase 4 to pick up the latest beam, history, and reflexions.

---

# Phase 7: Results

1. **Read final state**: database.json, baselines.json, history.json

2. **Present results summary**:
   ```
   ═══════════════════════════════════════════
   OPTIMIZATION COMPLETE
   ═══════════════════════════════════════════
   Best kernel time:     X.XXXX ms
   Initial kernel time:  X.XXXX ms
   PyTorch eager time:   X.XXXX ms
   PyTorch compile time: X.XXXX ms

   Speedup over initial:  X.Xx
   Speedup over PyTorch:  X.Xx

   Total rounds:          N
   Total workers run:     N

   Top kernels:
     1. X.XXXX ms (generation N, from round M)
     2. X.XXXX ms (generation N, from round M)
   ═══════════════════════════════════════════
   ```

3. **Save the best kernel**:
   - Copy the best kernel from state to `<kernel_dir>/optimized_kernel.py`
   - Report the file path to the user

4. **Clean up**: The state directory is preserved for inspection but the optimization is complete.

---

# Error Handling

- **GPU pod unreachable**: If any gpu_ops.py command fails with a connection error, report the error and suggest checking `KUBECTL_POD_NAME` / `KUBECTL_NAMESPACE` environment variables.
- **NCU profiling fails**: Skip profiling for that kernel and use a fallback "underutilized" bottleneck classification. Continue with optimization.
- **All workers fail verification**: Log a warning and continue to the next round. The beam retains the previous best kernels.
- **Subagent produces invalid output**: If a kernel-optimizer subagent writes a malformed file, treat it as a failed worker and continue.

---

# State File Reference

```
state_dir/
  config.json              # Optimization parameters
  round_num.txt            # Current round (integer)
  database.json            # Top-N kernels beam [{id, kernel_path, time_ms, generation, parent_id}]
  history.json             # Last 10 optimization attempts
  reflexions.json          # Last 10 reflexions
  gpu_specs.json           # GPU hardware specifications
  baselines.json           # {initial_kernel_ms, pytorch_eager_ms, pytorch_compile_ms}
  kernels/
    initial.py             # Original kernel
    r1_w0.py               # Round 1, worker 0
    r1_w1.py               # Round 1, worker 1
    ...
  round_1/
    round001_ncu_metrics.json
    round001_roofline.json
    bottleneck_initial.json
    worker_0_result.json
    worker_0_reflexion.json
    worker_1_result.json
    worker_1_reflexion.json
  round_2/
    ...
```
