#!/usr/bin/env bash
# One row from the publication-revision Karolina campaign matrix.
# Resource options are supplied by prepare_campaign.py.

#SBATCH --account=fta-26-40
#SBATCH --qos=3571_6328
#SBATCH --partition=qcpu_exp
#SBATCH --time=00:15:00

set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 MATRIX CASE_ID OUT_ROOT EXPECTED_SOURCE_COMMIT EXPECTED_MATRIX_SHA256 SOURCE_FREEZE SOURCE_FREEZE_SHA256" >&2
  exit 2
fi
MATRIX="$1"
CASE_ID="$2"
OUT_ROOT="$3"
EXPECTED_SOURCE_COMMIT="$4"
EXPECTED_MATRIX_SHA256="$5"
SOURCE_FREEZE="$6"
SOURCE_FREEZE_SHA256="$7"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${ALLOCATION_REVALIDATED:-}" != "YES" || "${ACCOUNT_QOS_REVALIDATED:-}" != "YES" ]]; then
  echo "Refusing compute execution: allocation and account/QoS were not explicitly revalidated." >&2
  exit 2
fi
if [[ -z "${ALLOCATION_VALID_UNTIL:-}" ]] || [[ "${ALLOCATION_VALID_UNTIL}" < "$(date +%F)" ]] || [[ "${ALLOCATION_VALID_UNTIL}" == "$(date +%F)" ]]; then
  echo "Refusing compute execution: ALLOCATION_VALID_UNTIL must be after today." >&2
  exit 2
fi
if [[ "${SLURM_JOB_ACCOUNT:-}" != "fta-26-40" ]]; then
  echo "Expected Slurm account fta-26-40, got ${SLURM_JOB_ACCOUNT:-unset}." >&2
  exit 2
fi
if [[ "${SLURM_JOB_QOS:-}" != "3571_6328" ]]; then
  echo "Expected Slurm QoS 3571_6328, got ${SLURM_JOB_QOS:-unset}." >&2
  exit 2
fi

if [[ ! "$EXPECTED_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Refusing compute execution: expected source commit is malformed." >&2
  exit 2
fi
if [[ ! "$EXPECTED_MATRIX_SHA256" =~ ^[0-9a-f]{64}$ || ! "$SOURCE_FREEZE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Refusing compute execution: expected matrix/source-freeze SHA-256 is malformed." >&2
  exit 2
fi
if [[ ! -f "$MATRIX" || ! -f "$SOURCE_FREEZE" ]]; then
  echo "Refusing compute execution: matrix or reviewed source freeze is missing." >&2
  exit 2
fi
RUNTIME_SOURCE_COMMIT="$(git rev-parse HEAD)"
if [[ "$RUNTIME_SOURCE_COMMIT" != "$EXPECTED_SOURCE_COMMIT" ]]; then
  echo "Refusing compute execution: queued source commit differs from the submission-time commit." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Publication campaign requires a clean committed worktree." >&2
  exit 2
fi
RUNTIME_MATRIX_SHA256="$(sha256sum "$MATRIX" | awk '{print $1}')"
if [[ "$RUNTIME_MATRIX_SHA256" != "$EXPECTED_MATRIX_SHA256" ]]; then
  echo "Refusing compute execution: campaign matrix changed after command preparation." >&2
  exit 2
fi
RUNTIME_SOURCE_FREEZE_SHA256="$(sha256sum "$SOURCE_FREEZE" | awk '{print $1}')"
if [[ "$RUNTIME_SOURCE_FREEZE_SHA256" != "$SOURCE_FREEZE_SHA256" ]]; then
  echo "Refusing compute execution: reviewed source freeze changed after command preparation." >&2
  exit 2
fi

if [[ -n "${ENV_SETUP:-}" ]]; then
  if [[ ! -f "$ENV_SETUP" ]]; then
    echo "ENV_SETUP='$ENV_SETUP' does not exist." >&2
    exit 2
  fi
  # shellcheck source=/dev/null
  source "$ENV_SETUP"
fi
PYTHON="${PYTHON:-./.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable '$PYTHON' is unavailable; set ENV_SETUP/PYTHON." >&2
  exit 2
fi

# SOURCE_FREEZE_SHA256 is embedded in the submitted command, while this Python
# block is part of the batch script copied by Slurm at submission.  A queued job
# therefore cannot silently move to another clean commit or another reviewed
# runner/analyzer set before it starts.
"$PYTHON" - "$PWD" "$MATRIX" "$SOURCE_FREEZE" "$EXPECTED_SOURCE_COMMIT" "$EXPECTED_MATRIX_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


repo_root = Path(sys.argv[1]).resolve()
matrix = Path(sys.argv[2]).resolve()
freeze_path = Path(sys.argv[3]).resolve()
expected_commit = sys.argv[4]
expected_matrix_sha256 = sys.argv[5]
with freeze_path.open(encoding="utf-8") as handle:
    freeze = json.load(handle)
if not isinstance(freeze, dict) or set(freeze) != {
    "schema_id",
    "schema_version",
    "source_commit",
    "matrix",
    "reviewed_sources",
}:
    raise SystemExit("reviewed source freeze has an unexpected top-level shape")
if (
    freeze.get("schema_id") != "fenics-nonlinear-energies.queued-source-freeze"
    or freeze.get("schema_version") != 1
    or freeze.get("source_commit") != expected_commit
):
    raise SystemExit("reviewed source freeze identity or commit changed")
matrix_record = freeze.get("matrix")
if not isinstance(matrix_record, dict) or set(matrix_record) != {"path", "sha256"}:
    raise SystemExit("reviewed source freeze matrix record is malformed")
try:
    matrix_relative = str(matrix.relative_to(repo_root))
except ValueError as exc:
    raise SystemExit("campaign matrix is outside the submitted repository") from exc
if (
    matrix_record.get("path") != matrix_relative
    or matrix_record.get("sha256") != expected_matrix_sha256
    or sha256(matrix) != expected_matrix_sha256
):
    raise SystemExit("reviewed source freeze does not bind the executed matrix")
reviewed = freeze.get("reviewed_sources")
if not isinstance(reviewed, dict) or not reviewed:
    raise SystemExit("reviewed source freeze contains no reviewed sources")
resolved_paths = set()
for key, record in sorted(reviewed.items()):
    if not isinstance(key, str) or not key:
        raise SystemExit("reviewed source freeze contains an invalid source key")
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise SystemExit(f"reviewed source record {key} is malformed")
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute():
        raise SystemExit(f"reviewed source {key} uses an absolute path")
    source = (repo_root / relative).resolve()
    try:
        source.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit(f"reviewed source {key} escapes the repository") from exc
    if source in resolved_paths:
        raise SystemExit(f"reviewed source {key} duplicates another source path")
    resolved_paths.add(source)
    expected = record.get("sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or not source.is_file()
        or sha256(source) != expected
    ):
        raise SystemExit(f"reviewed source {key} is missing or changed")
print(f"verified queued source freeze for {len(reviewed)} reviewed sources")
PY

# The environment setup may select compilers, MPI, PETSc, and the Python
# executable. Publication-critical numerical settings are forced only after it
# has been sourced, so neither the submit environment nor ENV_SETUP can change
# the reviewed policy.
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
export PYTHONPATH="$PWD"
export MIX_LOCAL_ASSEMBLY_BACKEND=coo_local
export MIX_LOCAL_REUSE_HESSIAN_VALUE_BUFFERS=1
export MIX_LOCAL_P4_HESSIAN_CHUNK_SIZE=32
export MIX_LOCAL_P4_CHUNK_SCATTER_CACHE=auto
export MIX_LOCAL_P4_CHUNK_SCATTER_CACHE_MAX_GIB=2.0
unset PETSC_OPTIONS PETSC_OPTIONS_FILE PETSC_OPTIONS_YAML

# Validate every scheduler-supplied resource field against the reviewed row,
# including the compact SLURM_TASKS_PER_NODE form (for example, 128(x2)).
VALIDATED_RANKS_PER_NODE="$(
  "$PYTHON" - "$MATRIX" "$CASE_ID" \
    "${SLURM_JOB_ACCOUNT:?}" "${SLURM_JOB_QOS:?}" "${SLURM_JOB_PARTITION:?}" \
    "${SLURM_JOB_NUM_NODES:?}" "${SLURM_NTASKS:?}" \
    "${SLURM_CPUS_PER_TASK:-1}" "${SLURM_TASKS_PER_NODE:?}" <<'PY'
import csv
import re
import sys


def expand_tasks_per_node(value: str) -> list[int]:
    expanded: list[int] = []
    for token in value.split(","):
        match = re.fullmatch(r"([1-9][0-9]*)(?:\(x([1-9][0-9]*)\))?", token.strip())
        if match is None:
            raise SystemExit(f"unsupported SLURM_TASKS_PER_NODE value {value!r}")
        expanded.extend([int(match.group(1))] * int(match.group(2) or 1))
    return expanded


matrix, case_id, account, qos, partition, nodes, tasks, cpus, tasks_per_node = sys.argv[1:]
with open(matrix, newline="", encoding="utf-8") as handle:
    rows = [row for row in csv.DictReader(handle) if row["case_id"] == case_id]
if len(rows) != 1:
    raise SystemExit(f"expected one matrix row for {case_id}, got {len(rows)}")
row = rows[0]
expected_shape = (int(row["nodes"]), int(row["total_ranks"]), 1)
actual_shape = (int(nodes), int(tasks), int(cpus))
if actual_shape != expected_shape:
    raise SystemExit(f"Slurm shape {actual_shape} does not match matrix {expected_shape}")
if account != "fta-26-40" or qos != "3571_6328" or partition != row["partition"]:
    raise SystemExit("Slurm account/QoS/partition differs from the reviewed matrix")
expected_per_node = [int(row["ranks_per_node"])] * int(row["nodes"])
actual_per_node = expand_tasks_per_node(tasks_per_node)
if actual_per_node != expected_per_node:
    raise SystemExit(
        f"Slurm tasks-per-node {actual_per_node} does not match matrix {expected_per_node}"
    )
print(row["ranks_per_node"])
PY
)"

JOB_ROOT="$OUT_ROOT/jobs/${CASE_ID}/job_${SLURM_JOB_ID}"
mkdir -p "$JOB_ROOT"

cleanup() {
  set +e
  {
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "accounting_status=pending_post_job_collection"
  } >> "$JOB_ROOT/job_metadata.env"
}
trap cleanup EXIT

{
  echo "case_id=$CASE_ID"
  echo "job_id=$SLURM_JOB_ID"
  echo "job_name=${SLURM_JOB_NAME:-}"
  echo "account=${SLURM_JOB_ACCOUNT:-}"
  echo "qos=${SLURM_JOB_QOS:-}"
  echo "partition=${SLURM_JOB_PARTITION:-}"
  echo "cluster=${SLURM_CLUSTER_NAME:-}"
  echo "nodes=${SLURM_JOB_NUM_NODES:-}"
  echo "ntasks=${SLURM_NTASKS:-}"
  echo "ntasks_per_node=$VALIDATED_RANKS_PER_NODE"
  echo "tasks_per_node=${SLURM_TASKS_PER_NODE:-}"
  echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
  echo "nodelist=${SLURM_JOB_NODELIST:-}"
  echo "submit_dir=${SLURM_SUBMIT_DIR:-}"
  echo "matrix=$MATRIX"
  echo "matrix_sha256=$RUNTIME_MATRIX_SHA256"
  echo "expected_matrix_sha256=$EXPECTED_MATRIX_SHA256"
  echo "git_commit=$RUNTIME_SOURCE_COMMIT"
  echo "expected_source_commit=$EXPECTED_SOURCE_COMMIT"
  echo "git_dirty=false"
  echo "source_freeze=$SOURCE_FREEZE"
  echo "source_freeze_sha256=$RUNTIME_SOURCE_FREEZE_SHA256"
  echo "expected_source_freeze_sha256=$SOURCE_FREEZE_SHA256"
  echo "env_setup=${ENV_SETUP:-}"
  if [[ -n "${ENV_SETUP:-}" ]]; then
    echo "env_setup_sha256=$(sha256sum "$ENV_SETUP" | awk '{print $1}')"
  else
    echo "env_setup_sha256=not_applicable"
  fi
  echo "allocation_revalidated=${ALLOCATION_REVALIDATED}"
  echo "account_qos_revalidated=${ACCOUNT_QOS_REVALIDATED}"
  echo "allocation_valid_until=${ALLOCATION_VALID_UNTIL}"
  echo "started_at=$(date --iso-8601=seconds)"
} | tee "$JOB_ROOT/job_metadata.env"

{
  echo "=== git ==="
  git show -s --format=fuller HEAD
  echo "=== modules ==="
  if type module >/dev/null 2>&1; then module list 2>&1; else echo "module unavailable"; fi
  echo "=== python ==="
  "$PYTHON" --version
  echo "=== PETSc contract ==="
  "$PYTHON" - <<'PY'
from petsc4py import PETSc

version = PETSc.Sys.getVersion()
if tuple(version[:2]) != (3, 24):
    raise SystemExit(f"publication environment requires PETSc 3.24.x, got {version}")
if not hasattr(PETSc.Mat, "setPreallocationCOO"):
    raise SystemExit("PETSc Mat.setPreallocationCOO is unavailable")
for package in ("hypre", "mumps"):
    if not PETSc.Sys.hasExternalPackage(package):
        raise SystemExit(
            f"publication environment requires PETSc external package {package}"
        )
print(version)
PY
  echo "=== allocation ==="
  scontrol show job "$SLURM_JOB_ID"
  echo "=== nodes ==="
  scontrol show hostnames "${SLURM_JOB_NODELIST:-}"
  echo "=== reviewed environment whitelist ==="
  for name in PATH LD_LIBRARY_PATH PYTHONPATH PYTHON JAX_PLATFORMS XLA_FLAGS \
    OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS \
    BLIS_NUM_THREADS VECLIB_MAXIMUM_THREADS MIX_LOCAL_ASSEMBLY_BACKEND \
    MIX_LOCAL_REUSE_HESSIAN_VALUE_BUFFERS MIX_LOCAL_P4_HESSIAN_CHUNK_SIZE \
    MIX_LOCAL_P4_CHUNK_SCATTER_CACHE MIX_LOCAL_P4_CHUNK_SCATTER_CACHE_MAX_GIB \
    SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_QOS SLURM_JOB_PARTITION \
    SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_TASKS_PER_NODE \
    SLURM_CPUS_PER_TASK SLURM_JOB_NODELIST; do
    printf '%s=%q\n' "$name" "${!name:-}"
  done
} > "$JOB_ROOT/environment.txt" 2>&1

"$PYTHON" -u "$SCRIPT_DIR/execute_case.py" \
  --matrix "$MATRIX" \
  --case-id "$CASE_ID" \
  --out-root "$OUT_ROOT" \
  --python "$PYTHON" \
  2>&1 | tee "$JOB_ROOT/execute.log"
