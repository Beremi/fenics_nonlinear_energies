#!/usr/bin/env python3
"""Run bounded, factorized route-cost calibration kernels under MPI.

These synthetic kernels separate factors that the finite-element degree sweep
necessarily couples.  They are calibration evidence only: they do not replace
the production fixed-state or full-solve comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time

from mpi4py import MPI
import numpy as np

from src.core.benchmark.run_record import atomic_write_json


def _git_metadata() -> dict[str, object]:
    repo = Path(__file__).resolve().parents[2]

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--short"))}


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else 1024 * value


def _sha(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _design() -> list[dict[str, int | str]]:
    baseline = {
        "element_dofs": 30,
        "quadrature_points": 11,
        "constitutive_dimension": 6,
        "color_count": 64,
        "nonzeros_per_row": 48,
        "message_bytes": 65536,
        "imbalance_ratio": 1,
    }
    cases: list[dict[str, int | str]] = [{"case_id": "baseline", **baseline}]
    factors = {
        "element_dofs": (12, 105),
        "quadrature_points": (1, 24, 125),
        "constitutive_dimension": (3, 9),
        "color_count": (32, 128),
        "nonzeros_per_row": (20, 100),
        "message_bytes": (8192, 1048576),
        "imbalance_ratio": (2, 4),
    }
    for factor, levels in factors.items():
        for level in levels:
            row = dict(baseline)
            row[factor] = int(level)
            cases.append({"case_id": f"{factor}_{level}", **row})
    return cases


def _time_max(comm: MPI.Comm, operation) -> tuple[float, list[float], object]:
    comm.Barrier()
    started = time.perf_counter()
    value = operation()
    elapsed = float(time.perf_counter() - started)
    by_rank = [float(item) for item in comm.allgather(elapsed)]
    collective_max = float(comm.allreduce(elapsed, op=MPI.MAX))
    if not np.isclose(collective_max, max(by_rank), rtol=1.0e-13, atol=1.0e-15):
        raise RuntimeError("factor MPI_MAX timing disagrees with gathered rank timings")
    return collective_max, by_rank, value


def _balanced_rank_work(
    *, rank: int, size: int, baseline: int, target_max_over_mean: int
) -> tuple[int, bool]:
    """Assign integer work with fixed global total and a declared max/mean.

    The last rank is the heavy rank.  For a single rank, nonunit imbalance is
    impossible and the factor is explicitly marked inapplicable rather than
    being mislabeled as imbalance.
    """
    if size < 1 or not 0 <= rank < size or baseline < 1:
        raise ValueError("invalid rank-work design")
    target = int(target_max_over_mean)
    if target < 1:
        raise ValueError("imbalance target must be positive")
    if size == 1:
        return baseline, target == 1
    if target > size:
        raise ValueError("imbalance target cannot exceed the MPI size")
    heavy = baseline * target
    if rank == size - 1:
        return heavy, True
    remaining = baseline * size - heavy
    low, extra = divmod(remaining, size - 1)
    return low + int(rank < extra), True


def _run_case(
    row: dict[str, int | str], *, repetitions: int, comm: MPI.Comm
) -> dict[str, object]:
    me = int(row["element_dofs"])
    nq = int(row["quadrature_points"])
    s = int(row["constitutive_dimension"])
    colors = int(row["color_count"])
    nnz_per_row = int(row["nonzeros_per_row"])
    message_bytes = int(row["message_bytes"])
    imbalance = int(row["imbalance_ratio"])
    local_batch, imbalance_applicable = _balanced_rank_work(
        rank=comm.rank,
        size=comm.size,
        baseline=8,
        target_max_over_mean=imbalance,
    )
    seed = int(hashlib.sha256(str(row["case_id"]).encode()).hexdigest()[:8], 16) + comm.rank
    rng = np.random.default_rng(seed)

    allocation_started = time.perf_counter()
    bmat = rng.standard_normal((local_batch, nq, s, me), dtype=np.float64)
    raw = rng.standard_normal((local_batch, nq, s, s), dtype=np.float64)
    constitutive = np.einsum("...ik,...jk->...ij", raw, raw, optimize=True)
    hessian = rng.standard_normal((me, me), dtype=np.float64)
    hessian = 0.5 * (hessian + hessian.T)
    probes = rng.standard_normal((colors, me), dtype=np.float64)
    n_rows, insertion_imbalance_applicable = _balanced_rank_work(
        rank=comm.rank,
        size=comm.size,
        baseline=512,
        target_max_over_mean=imbalance,
    )
    if insertion_imbalance_applicable != imbalance_applicable:
        raise RuntimeError("factor stages disagree on imbalance applicability")
    coo_rows = np.repeat(np.arange(n_rows, dtype=np.int64), nnz_per_row)
    coo_values = rng.standard_normal(coo_rows.size, dtype=np.float64)
    send = rng.standard_normal(max(1, message_bytes // 8), dtype=np.float64)
    recv = np.empty_like(send)
    allocation_local = float(time.perf_counter() - allocation_started)
    allocation_times_by_rank = [float(value) for value in comm.allgather(allocation_local)]
    allocation_time = float(comm.allreduce(allocation_local, op=MPI.MAX))
    if not np.isclose(
        allocation_time,
        max(allocation_times_by_rank),
        rtol=1.0e-13,
        atol=1.0e-15,
    ):
        raise RuntimeError("factor allocation MPI_MAX disagrees with rank timings")
    tracked_bytes = sum(
        array.nbytes
        for array in (bmat, constitutive, hessian, probes, coo_rows, coo_values, send, recv)
    )

    stage_times = {name: [] for name in ("contraction", "color_hvp", "insertion", "communication")}
    stage_times_by_rank = {
        name: [] for name in ("contraction", "color_hvp", "insertion", "communication")
    }
    checksum = 0.0
    for _ in range(repetitions):
        elapsed, by_rank, contracted = _time_max(
            comm,
            lambda: np.einsum(
                "bqsm,bqst,bqtn->bmn", bmat, constitutive, bmat, optimize=True
            ),
        )
        stage_times["contraction"].append(elapsed)
        stage_times_by_rank["contraction"].append(by_rank)
        checksum += float(np.asarray(contracted).ravel()[0])

        elapsed, by_rank, colored = _time_max(comm, lambda: probes @ hessian.T)
        stage_times["color_hvp"].append(elapsed)
        stage_times_by_rank["color_hvp"].append(by_rank)
        checksum += float(np.asarray(colored).ravel()[0])

        elapsed, by_rank, inserted = _time_max(
            comm, lambda: np.bincount(coo_rows, weights=coo_values, minlength=n_rows)
        )
        stage_times["insertion"].append(elapsed)
        stage_times_by_rank["insertion"].append(by_rank)
        checksum += float(np.asarray(inserted).ravel()[0])

        def communicate() -> np.ndarray:
            comm.Allreduce(send, recv, op=MPI.SUM)
            return recv

        elapsed, by_rank, communicated = _time_max(comm, communicate)
        stage_times["communication"].append(elapsed)
        stage_times_by_rank["communication"].append(by_rank)
        checksum += float(np.asarray(communicated).ravel()[0])

    local_rss = _peak_rss_bytes()
    local_hashes = {
        "rank": int(comm.rank),
        "bmat": _sha(bmat),
        "constitutive": _sha(constitutive),
        "hessian": _sha(hessian),
        "probes": _sha(probes),
        "coo_rows": _sha(coo_rows),
    }
    hashes_by_rank = comm.gather(local_hashes, root=0)
    local_batches = [int(value) for value in comm.allgather(local_batch)]
    insertion_rows = [int(value) for value in comm.allgather(n_rows)]
    realized_batch_imbalance = float(max(local_batches) / np.mean(local_batches))
    realized_insertion_imbalance = float(max(insertion_rows) / np.mean(insertion_rows))
    return {
        **row,
        "mpi_ranks": int(comm.size),
        "local_batch_min": min(local_batches),
        "local_batch_max": max(local_batches),
        "local_batch_by_rank": local_batches,
        "insertion_rows_by_rank": insertion_rows,
        "imbalance_factor_applicable": bool(imbalance_applicable),
        "realized_batch_max_over_mean": realized_batch_imbalance,
        "realized_insertion_max_over_mean": realized_insertion_imbalance,
        "tracked_allocation_bytes_max": int(comm.allreduce(tracked_bytes, op=MPI.MAX)),
        "peak_rss_bytes_max": int(comm.allreduce(local_rss, op=MPI.MAX)),
        "allocation_collective_max_s": allocation_time,
        "allocation_times_by_rank_s": allocation_times_by_rank,
        "stage_collective_max_times_s": stage_times,
        "stage_times_by_rank_s": stage_times_by_rank,
        "stage_collective_max_medians_s": {
            name: float(np.median(np.asarray(values, dtype=np.float64)))
            for name, values in stage_times.items()
        },
        "cold_first_over_warm_median": {
            name: float(values[0] / max(np.median(values[1:]), np.finfo(float).tiny))
            for name, values in stage_times.items()
        },
        "timing_reduction": "mpi_collective_max",
        "checksum": checksum,
        "input_sha256_by_rank": hashes_by_rank or [],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    comm = MPI.COMM_WORLD
    cases = _design()
    # Deterministic rotation prevents a factor from always occupying the same
    # thermal/cache position across rank-point allocations.
    offset = int(args.block_repetition) % len(cases)
    ordered = cases[offset:] + cases[:offset]
    results = [
        _run_case(row, repetitions=int(args.repetitions), comm=comm) for row in ordered
    ]
    return {
        "schema_version": 1,
        "experiment_id": "EXP-ROUTE-001",
        "tier": "factorized_microbenchmark",
        "status": "completed",
        "scope": "synthetic_calibration_not_production_route_timing",
        "block_repetition": int(args.block_repetition),
        "repetitions": int(args.repetitions),
        "timing_reduction": "mpi_collective_max",
        "one_factor_at_a_time": True,
        "factor_order_policy": "deterministic_rotated_v1",
        "factors": [
            "element_dofs",
            "quadrature_points",
            "constitutive_dimension",
            "color_count",
            "nonzeros_per_row",
            "message_bytes",
            "imbalance_ratio",
        ],
        "results": results,
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "numerical_runtime": {
            "numpy": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
            "cpu_affinity": (
                sorted(int(value) for value in os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else []
            ),
        },
        "command": shlex.join([sys.executable, *sys.argv]),
        "git": _git_metadata(),
        "job_metadata": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME", ""),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--block-repetition", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if int(args.repetitions) < 5:
        raise SystemExit("factorized publication calibration requires >=5 repetitions")
    payload = run(args)
    if MPI.COMM_WORLD.rank == 0:
        atomic_write_json(Path(args.output).resolve(), payload)
        print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
