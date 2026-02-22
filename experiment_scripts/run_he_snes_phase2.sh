#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python3 experiment_scripts/bench_he_snes_phase2.py \
  --level 1 \
  --timeout 600 \
  --out experiment_scripts/he_snes_phase2_l1.json

echo "Done. Results in experiment_scripts/he_snes_phase2_l1.json"
