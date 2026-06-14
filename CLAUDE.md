# Agent Instructions

This repository is the Agents prompt and workflow release artifact for MLSys 2026 FlashInfer NVIDIA track kernel submissions.

## Repository Rules

- Use English for all repository-facing files, comments, documentation, prompts, and commit messages.
- During prompt-driven workflow reproduction, MUST NOT clone, inspect, copy from, or otherwise use the released submissions repository to obtain implementation answers.
- Do not copy final submission code into a fresh agent workspace when reproducing the search workflow.
- Use `~/data/flashinfer-trace` as the default dataset location. Respect `FIB_DATASET_PATH` when set.
- Generated outputs belong in `runs/`, `outputs/`, or `profile/`; these paths are ignored by git.
- Use Python 3.12 or 3.13 for the uv environment.

## Expected Agent Workflow

For a new kernel optimization task:

1. Set up the environment from `README.md`.
2. Use KernelWiki for Blackwell, CUDA, CuTe DSL, Triton, and prior kernel implementation research.
3. Use ncu-report-skill for Nsight Compute profiling and bottleneck analysis.
4. Record every performance-related commit in `benchmark.csv`.
5. Keep NCU profiling records for each major optimization direction.