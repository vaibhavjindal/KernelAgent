---
description: Optimize a Triton kernel using Claude Code subagents and a kubectl GPU pod
argument-hint: Path to kernel directory containing problem.py and input_kernel.py
allowed-tools: ["Read", "Write", "Edit", "Glob", "Bash", "Grep", "Agent"]
---

# kernel-optimizer:optimize

Optimize an existing Triton kernel on a remote GPU pod using Claude Code subagents
for bottleneck analysis and kernel generation, and kubectl for GPU profiling,
verification, and benchmarking.

**Initial request**: $ARGUMENTS

Load and follow the instructions in skills/kernel-optimization/SKILL.md.
Start at Phase 1: Setup and Validation.
