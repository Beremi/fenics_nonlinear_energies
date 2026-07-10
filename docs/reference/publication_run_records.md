# Publication Run Records

Publication-revision experiments use the versioned contract in
`src/core/benchmark/run_record.py`. The contract separates terminal scientific
outcomes from launcher behavior and prevents a dirty pilot from being ingested
as publication evidence.

## Required Workflow

1. Call `check_experiment_preflight(...)` immediately before starting a case.
2. For a publication run, use `run_kind="publication"`. The check fails unless
   `git status --porcelain` is empty.
3. For a diagnostic on a dirty tree, use `run_kind="pilot"`, set
   `pilot_dirty_override=True`, and provide a non-empty
   `pilot_override_reason`. This record remains a pilot and will be rejected by
   a publication ingestion boundary.
4. During a long solve, call `atomic_write_checkpoint(...)` periodically. Use a
   monotonically increasing sequence and include the latest counters, state
   path, residuals, and resource observations in `progress`.
5. On every terminal path—including exceptions, caps, and timeouts—construct a
   complete run record and persist it with `atomic_write_run_record(...)`.

The atomic writer serializes before touching the destination, fsyncs a
same-directory temporary file, replaces the destination atomically, and fsyncs
the directory. Invalid JSON values such as NaN do not overwrite the previous
checkpoint.

Raw solver histories may use non-finite in-memory sentinels for optional
quantities that are unavailable at a particular iteration. Persist those raw
payloads with `atomic_write_json(..., nonfinite_as_null=True)`. The recursive
sanitizer preserves finite numbers and maps only NaN and positive/negative
infinity to JSON `null`; serialization still runs with `allow_nan=False`.
Validated publication run records retain the strict default and reject
non-finite evidence. Consequently, consumers of newly written raw Plasticity3D
outputs must treat `null`, rather than the nonstandard bare tokens `NaN` or
`Infinity`, as an unavailable optional history value. Historical artifacts are
not rewritten.

## Terminal Statuses

| Status | Meaning | Additional rule |
| --- | --- | --- |
| `success` | The preregistered accuracy gate passed | `gate_passed=true`, `censored=false` |
| `failure` | The method or process failed without a declared cap | The reason and all available last-state evidence remain in the record |
| `capped` | A declared iteration, memory, campaign, or other ceiling stopped the case | Record `limit_kind`, `limit_value`, and `censored=true` |
| `timeout` | A wall-time ceiling stopped the case | Record the wall-time limit and `censored=true` |

Do not convert `capped` or `timeout` to `failure`, and do not omit them from a
campaign summary. A solver process that exits normally without passing the
accuracy contract is not `success`.

## Record Sections

Schema version 1 requires these sections for every terminal status:

- `identifiers`: campaign, experiment, case, method, route, and one-based
  repetition identifiers;
- `problem`: mesh, degree, quadrature, total/free degrees of freedom, and an
  applicability note;
- `solver`: exact algorithm, implementation, parameter map, preconditioner map,
  and stopping-contract identifier;
- `termination`: status, reason, timestamps, exit code, cap information, and
  censoring flag;
- `accuracy`: the contract and gate result; absolute, relative, and scaled
  residuals; relative correction; energy change; and custom metrics;
- `counts`: nonlinear, Krylov, function, gradient, Hessian, HVP, and
  preconditioner counts;
- `timing`: rank aggregation, cold/warm classification, barrier and JAX
  synchronization policy, phase-overlap declaration, phase decomposition, and
  total time;
- `resources`: nodes, ranks, threads, peak per-rank and per-node memory,
  tracked allocations, and measurement method;
- `diagnostics`: state, branch, feasibility, and KKT maps, empty only when not
  applicable;
- `environment`: Python/packages, JAX/XLA, FP64 state, PETSc, MPI, compiler,
  BLAS, hardware, scheduler, and affinity;
- `provenance`: clean commit evidence, exact argv and working directory,
  SHA-256 code/configuration/input hashes, seed policy, and record timestamp;
- `artifacts`: raw outputs, states, logs, tables, figures, and reports.

Fields remain present when a value is unavailable. Use JSON `null` only where
the schema permits it and explain the reason in that section's `notes`. Use the
literal string `not-applicable` for required environment or method labels that
do not apply. Never write NaN or infinity.

The exported `RUN_RECORD_JSON_SCHEMA` describes the stable top-level envelope.
`validate_run_record` is authoritative for conditional rules. Pass
`require_publication_ready=True` when ingesting records for paper tables or
figures; this rejects pilot records even if their numerical gate passed.

## Minimal Runner Integration

```python
from pathlib import Path

from src.core.benchmark.run_record import (
    atomic_write_checkpoint,
    atomic_write_run_record,
    check_experiment_preflight,
)

repo = Path(__file__).resolve().parents[2]
preflight = check_experiment_preflight(repo, run_kind="publication")

# Periodically, from the solver process:
atomic_write_checkpoint(
    output_dir / "progress_latest.json",
    record_id=record_id,
    sequence=nonlinear_iteration,
    progress={
        "nonlinear_iteration": nonlinear_iteration,
        "scaled_residual": scaled_residual,
        "state_path": str(last_state_path),
    },
)

# On every terminal path, merge preflight.provenance_fields() into the complete
# provenance section, then validate and replace the terminal record atomically.
atomic_write_run_record(output_dir / "run_record.json", record)
```

`sha256_file(...)` is provided for streaming hashes. Paths in the record should
be relative to the campaign root when possible. The campaign's experiment card
must still define tolerances, measurement applicability, repetitions, route
ordering, resource ceilings, and statistical contrasts; the schema does not
choose those scientific decisions.
