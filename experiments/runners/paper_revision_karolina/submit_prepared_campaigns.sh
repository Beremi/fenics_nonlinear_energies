#!/usr/bin/env bash
# Prepare (default), admission-test, or explicitly submit the revision matrix.
#
# Safety defaults are intentionally non-submitting.  The allocation record in
# the local HPC profile ends on 2026-07-11, so both admission tests and real
# submission require current allocation/account/QoS revalidation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

DRY_RUN="${DRY_RUN:-1}"
SBATCH_TEST_ONLY="${SBATCH_TEST_ONLY:-0}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-0}"
ONLY_OPTIONAL="${ONLY_OPTIONAL:-0}"
EXPERIMENTS="${EXPERIMENTS:-}"
TIERS="${TIERS:-}"
ADMISSION_GATE="${ADMISSION_GATE:-}"
STOPPING_ADJUDICATION="${STOPPING_ADJUDICATION:-}"
ROUTE_PHASE="${ROUTE_PHASE:-}"
MODEL_FREEZE_RECEIPT="${MODEL_FREEZE_RECEIPT:-}"
ENV_SETUP="${ENV_SETUP:-}"
ENV_LOCK="${ENV_LOCK:-}"
MAX_NODE_HOURS="${MAX_NODE_HOURS:-100}"
CAMPAIGN_ID="${CAMPAIGN_ID:-paper_revision_karolina_prepared_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
OUT_ROOT="${OUT_ROOT:-artifacts/reproduction/paper_revision_karolina/${CAMPAIGN_ID}}"
PYTHON="${PYTHON:-./.venv/bin/python}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
case "$SBATCH_TEST_ONLY" in 0|1) ;; *) echo "SBATCH_TEST_ONLY must be 0 or 1" >&2; exit 2 ;; esac
case "$INCLUDE_OPTIONAL" in 0|1) ;; *) echo "INCLUDE_OPTIONAL must be 0 or 1" >&2; exit 2 ;; esac
case "$ONLY_OPTIONAL" in 0|1) ;; *) echo "ONLY_OPTIONAL must be 0 or 1" >&2; exit 2 ;; esac
if [[ ! "$CAMPAIGN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "CAMPAIGN_ID must contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable '$PYTHON' is unavailable" >&2
  exit 2
fi

args=(
  "$PYTHON" "$SCRIPT_DIR/prepare_campaign.py"
  --matrix "$SCRIPT_DIR/campaign_matrix.csv"
  --out-root "$OUT_ROOT"
  --max-node-hours "$MAX_NODE_HOURS"
)
if [[ "$INCLUDE_OPTIONAL" == "1" ]]; then
  args+=(--include-optional)
fi
if [[ "$ONLY_OPTIONAL" == "1" ]]; then
  args+=(--only-optional)
fi
if [[ -n "$EXPERIMENTS" ]]; then
  IFS=',' read -r -a experiment_values <<< "$EXPERIMENTS"
  for experiment in "${experiment_values[@]}"; do
    args+=(--experiment "$experiment")
  done
fi
if [[ -n "$TIERS" ]]; then
  IFS=',' read -r -a tier_values <<< "$TIERS"
  for tier in "${tier_values[@]}"; do
    args+=(--tier "$tier")
  done
fi
if [[ -n "$ADMISSION_GATE" ]]; then
  args+=(--admission-gate "$ADMISSION_GATE")
fi
if [[ -n "$STOPPING_ADJUDICATION" ]]; then
  args+=(--stopping-adjudication "$STOPPING_ADJUDICATION")
fi
if [[ -n "$ROUTE_PHASE" ]]; then
  args+=(--route-phase "$ROUTE_PHASE")
fi
if [[ -n "$MODEL_FREEZE_RECEIPT" ]]; then
  args+=(--model-freeze-receipt "$MODEL_FREEZE_RECEIPT")
fi
if [[ -n "$ENV_SETUP" || -n "$ENV_LOCK" ]]; then
  if [[ -z "$ENV_SETUP" || -z "$ENV_LOCK" ]]; then
    echo "ENV_SETUP and ENV_LOCK must be supplied together" >&2
    exit 2
  fi
  args+=(--env-setup "$ENV_SETUP" --env-lock "$ENV_LOCK")
fi
if [[ "$SBATCH_TEST_ONLY" == "1" ]]; then
  args+=(--test-only)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  "${args[@]}"
  echo "DRY_RUN=1: prepared commands only; no sbatch process was invoked."
  exit 0
fi

# prepare_campaign.py independently enforces all revalidation variables before
# it can execute sbatch, so invoking it directly cannot bypass these gates.
args+=(--execute)
"${args[@]}"
