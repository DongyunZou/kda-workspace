#!/usr/bin/env python3
"""Minimal FlashInfer solution verification example.
It loads one packed FlashInfer `solution.json`, selects workloads from a
FlashInfer trace dataset, and evaluates the solution with the official
`flashinfer_bench` entrypoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_DATASET = Path.home() / "dataset" / "flashinfer-trace"


def dataset_path(explicit: str | None) -> Path:
    raw = explicit or os.environ.get("FIB_DATASET_PATH")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_DATASET


def import_flashinfer_bench():
    try:
        from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "flashinfer_bench is not installed. Run `uv sync`, or install the official "
            "FlashInfer benchmark package following docs/reproduction.md."
        ) from exc
    return Benchmark, BenchmarkConfig, Solution, TraceSet


def benchmark_config(solution_definition: str, args: argparse.Namespace):
    _, BenchmarkConfig, _, _ = import_flashinfer_bench()
    kwargs = dict(
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
        use_isolated_runner=args.isolated_runner,
        profile_baseline=False,
    )
    if solution_definition.startswith("moe_fp8"):
        kwargs.update(atol=1.0, rtol=0.3, required_matched_ratio=0.9)
    return BenchmarkConfig(**kwargs)


def select_workloads(workloads: list, args: argparse.Namespace) -> list:
    if args.workload_uuid:
        wanted = set(args.workload_uuid)
        workloads = [w for w in workloads if w.workload.uuid in wanted]
    if args.limit is not None:
        workloads = workloads[: max(0, args.limit)]
    return workloads


def is_flashinfer_baseline_solution(solution) -> bool:
    fields = [
        solution.name,
        solution.author,
        *solution.spec.dependencies,
    ]
    return "flashinfer" in " ".join(fields).lower()


def select_flashinfer_baseline(trace_set, definition_name: str, baseline_name: str | None):
    solutions = list(trace_set.solutions.get(definition_name, []))
    if baseline_name:
        matches = [solution for solution in solutions if solution.name == baseline_name]
        if not matches:
            raise ValueError(
                f"Baseline solution {baseline_name!r} not found for {definition_name!r}"
            )
        return matches[0]

    candidates = [solution for solution in solutions if is_flashinfer_baseline_solution(solution)]
    if not candidates:
        raise ValueError(f"No FlashInfer baseline solution found for {definition_name!r}")
    if len(candidates) > 1:
        names = ", ".join(solution.name for solution in candidates)
        raise ValueError(
            "Multiple FlashInfer baseline solutions found for "
            f"{definition_name!r}: {names}. Pass --baseline-solution to choose one."
        )
    return candidates[0]


def evaluation_status(trace) -> str:
    if trace is None or trace.evaluation is None:
        return "NO_EVAL"
    status = trace.evaluation.status
    return getattr(status, "value", str(status))


def passed_latency_ms(trace) -> float | None:
    if trace is None or trace.evaluation is None:
        return None
    if str(evaluation_status(trace)).lower() != "passed":
        return None
    perf = getattr(trace.evaluation, "performance", None)
    if perf is None or perf.latency_ms is None or perf.latency_ms <= 0:
        return None
    return float(perf.latency_ms)


def run(args: argparse.Namespace) -> bool:
    Benchmark, _, Solution, TraceSet = import_flashinfer_bench()

    solution_path = Path(args.solution).expanduser().resolve()
    solution = Solution.model_validate_json(solution_path.read_text())

    dataset = dataset_path(args.dataset)
    if not dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset}\n"
            "Run ./scripts/download_data.sh or set FIB_DATASET_PATH."
        )
    trace_set = TraceSet.from_path(str(dataset))

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition {solution.definition!r} not found in {dataset}")

    definition = trace_set.definitions[solution.definition]
    workloads = select_workloads(list(trace_set.workloads.get(solution.definition, [])), args)
    if not workloads:
        raise ValueError(f"No workloads selected for {solution.definition}")

    baseline_solution = select_flashinfer_baseline(
        trace_set, solution.definition, args.baseline_solution
    )
    if solution.name == baseline_solution.name and solution.hash() != baseline_solution.hash():
        raise ValueError(
            f"Solution name {solution.name!r} collides with the FlashInfer baseline. "
            "Rename the candidate solution before verification."
        )
    solutions = [solution]
    if solution.name != baseline_solution.name:
        solutions.append(baseline_solution)

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: solutions},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    result_trace_set = Benchmark(
        bench_trace_set,
        benchmark_config(solution.definition, args),
    ).run_all(dump_traces=False)

    traces = list(result_trace_set.traces.get(definition.name, []))
    traces_by_solution_workload = {
        (trace.solution, trace.workload.uuid): trace
        for trace in traces
        if trace.solution is not None
    }

    passed = 0
    baseline_passed = 0
    latencies = []
    baseline_latencies = []
    speedups = []
    for workload_trace in workloads:
        uuid = workload_trace.workload.uuid
        solution_trace = traces_by_solution_workload.get((solution.name, uuid))
        baseline_trace = traces_by_solution_workload.get((baseline_solution.name, uuid))

        solution_status = evaluation_status(solution_trace)
        baseline_status = evaluation_status(baseline_trace)
        solution_latency = passed_latency_ms(solution_trace)
        baseline_latency = passed_latency_ms(baseline_trace)

        if solution_latency is not None:
            passed += 1
            latencies.append(solution_latency)
        if baseline_latency is not None:
            baseline_passed += 1
            baseline_latencies.append(baseline_latency)

        parts = [f"solution={solution_status}", f"baseline={baseline_status}"]
        if solution_latency is not None and baseline_latency is not None:
            speedup = baseline_latency / solution_latency
            speedups.append(speedup)
            parts.append(f"speedup_vs_flashinfer={speedup:.4f}x")
        print(f"{uuid}: " + ", ".join(parts))

    print(f"\nsolution:   {solution.name}")
    print(f"definition: {solution.definition}")
    print(f"baseline:   {baseline_solution.name}")
    print(f"passed:     {passed}/{len(workloads)}")
    print(f"baseline passed: {baseline_passed}/{len(workloads)}")
    if latencies:
        print(f"latency:    {sum(latencies) / len(latencies):.6f} ms mean")
    if baseline_latencies:
        print(f"flashinfer: {sum(baseline_latencies) / len(baseline_latencies):.6f} ms mean")
    if speedups:
        print(
            f"speedup:    {sum(speedups) / len(speedups):.4f}x mean "
            f"vs flashinfer baseline ({len(speedups)}/{len(workloads)} workloads)"
        )
    else:
        print("speedup:    n/a (no workloads where both solution and baseline passed)")
    return passed == len(workloads) and baseline_passed == len(workloads)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", required=True, help="Path to a packed FlashInfer solution.json.")
    parser.add_argument("--dataset", default=None, help="Path to flashinfer-trace dataset.")
    parser.add_argument("--baseline-solution", default=None, help="FlashInfer baseline solution name.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected workloads.")
    parser.add_argument("--workload-uuid", action="append", default=None, help="Run one specific workload UUID.")
    parser.add_argument("--fast", action="store_true", help="Use a short benchmark configuration.")
    parser.add_argument("--isolated-runner", action="store_true", help="Use flashinfer-bench isolated runner.")
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fast:
        args.warmup_runs = 1 if args.warmup_runs is None else args.warmup_runs
        args.iterations = 5 if args.iterations is None else args.iterations
        args.num_trials = 1 if args.num_trials is None else args.num_trials
        args.limit = 2 if args.limit is None and not args.workload_uuid else args.limit
    else:
        args.warmup_runs = 3 if args.warmup_runs is None else args.warmup_runs
        args.iterations = 50 if args.iterations is None else args.iterations
        args.num_trials = 3 if args.num_trials is None else args.num_trials

    return 0 if run(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
