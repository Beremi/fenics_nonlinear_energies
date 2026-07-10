#!/usr/bin/env bash
# Execute one hash-bound row prepared by karolina_reviewed_campaign.py.
# All scheduler resource options are present in the prepared command inventory.

#SBATCH --account=fta-26-40
#SBATCH --qos=3571_6328
#SBATCH --partition=qcpu_exp
#SBATCH --time=00:15:00

set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 CAMPAIGN_ROOT CASE_ID PLAN_SHA256 SOURCE_FREEZE_SHA256 ENV_SETUP ENV_LOCK" >&2
  exit 2
fi

CAMPAIGN_ROOT="$1"
CASE_ID="$2"
EXPECTED_PLAN_SHA256="$3"
EXPECTED_FREEZE_SHA256="$4"
ENV_SETUP="$5"
ENV_LOCK="$6"

if [[ "${ALLOCATION_REVALIDATED:-}" != "YES" || "${ACCOUNT_QOS_REVALIDATED:-}" != "YES" ]]; then
  echo "Refusing execution: allocation and account/QoS were not revalidated." >&2
  exit 2
fi
if [[ -z "${ALLOCATION_VALID_UNTIL:-}" || "${ALLOCATION_VALID_UNTIL}" < "$(date +%F)" || "${ALLOCATION_VALID_UNTIL}" == "$(date +%F)" ]]; then
  echo "Refusing execution: ALLOCATION_VALID_UNTIL must be after today." >&2
  exit 2
fi
if [[ "${SLURM_JOB_ACCOUNT:-}" != "fta-26-40" || "${SLURM_JOB_QOS:-}" != "3571_6328" ]]; then
  echo "Refusing execution: Slurm account/QoS differ from the reviewed contract." >&2
  exit 2
fi
if [[ ! -f "$ENV_SETUP" || ! -f "$ENV_LOCK" ]]; then
  echo "Refusing execution: reviewed environment setup/lock are missing." >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$ENV_SETUP"
PYTHON="${PYTHON:-./.venv/bin/python}"
exec "$PYTHON" -u experiments/runners/execute_reviewed_karolina_case.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --case-id "$CASE_ID" \
  --expected-plan-sha256 "$EXPECTED_PLAN_SHA256" \
  --expected-source-freeze-sha256 "$EXPECTED_FREEZE_SHA256" \
  --environment-lock "$ENV_LOCK"

