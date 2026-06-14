# Kernel Design Agents Workspace
## Contents

| Path | Purpose |
|---|---|
| `verify.py` | Minimal example that evaluates one packed FlashInfer `solution.json` with `flashinfer-bench`. |
| `docs/reproduction.md` | Environment, dataset, and benchmark reproduction notes. |

## Fresh Workflow Setup

Install the benchmark environment, download the FlashInfer contest workloads, and prepare the agent workflow dependencies:

```bash
git clone https://github.com/flashinfer-ai/flashinfer-bench.git /tmp/flashinfer-bench-main
uv sync --python 3.12

# uv.lock pins the contest-tested stack:
# flashinfer-python==0.6.8.post1, torch==2.12.0+cu132, triton==3.7.0.
# Use Python 3.12 or 3.13; Python 3.14 is not supported by all CUDA wheels.

# Required by some baselines and generated solutions that use DeepGEMM/CUTLASS/CuTe headers.
# DeepGEMM needs its CUTLASS/fmt submodules during installation.
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM
uv pip install -e /tmp/DeepGEMM --no-build-isolation
```

Confirm that the workload dataset is visible:

```bash
uv run python -c "from flashinfer_bench import TraceSet; from pathlib import Path; ts = TraceSet.from_path(Path.home() / 'dataset' / 'flashinfer-trace'); print(sorted(ts.definitions)); print(sum(len(v) for v in ts.workloads.values()), 'workloads')"
```