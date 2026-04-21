#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-artifacts/raw_results/scaling_probe/p4_l1_2_uniform_tail_maxit20_constitutivead_localpmg_mix_tol1e1_threads1}"
REPORT_SCRIPT="${REPORT_SCRIPT:-experiments/analysis/generate_p4_l1_2_uniform_tail_scaling_assets.py}"
REPORT_OUTDIR="${REPORT_OUTDIR:-$ROOT/assets}"
REPORT_PATH="${REPORT_PATH:-$ROOT/REPORT.md}"
RANKS_CSV="${RANKS_CSV:-1,2,4,8,16,32}"
NEWTON_MAXIT="${NEWTON_MAXIT:-20}"
IFS=',' read -r -a RANKS <<< "$RANKS_CSV"

mkdir -p "$ROOT"
SUMMARY="$ROOT/run_summary.tsv"
printf 'np\tstatus\ttotal_time\tsolve_time\tlinear1_assemble\tlinear1_setup\tlinear1_solve\n' > "$SUMMARY"

for NP in "${RANKS[@]}"; do
  OUTDIR="$ROOT/np${NP}"
  mkdir -p "$OUTDIR"
  rm -f "$OUTDIR/output.json" "$OUTDIR/run.log" "$OUTDIR/time.txt"
  rm -rf "$OUTDIR/data"
  echo "[$(date --iso-8601=seconds)] starting np=${NP}" | tee -a "$ROOT/serial_master.log"
  TIMEFORMAT=$'real %3R\nuser %3U\nsys %3S'
  {
    time env \
      OMP_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 \
      XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' \
      mpiexec -n "$NP" ./.venv/bin/python \
        experiments/runners/run_plasticity3d_backend_mix_case.py \
        --assembly-backend local_constitutiveAD \
        --solver-backend local_pmg \
        --mesh-name hetero_ssr_L1_2 \
        --lambda-target 1.5 \
        --pmg-strategy uniform_refined_p4_p2_p1_p1 \
        --ksp-rtol 1e-1 \
        --ksp-max-it 100 \
        --stop-tol 0.0 \
        --maxit "$NEWTON_MAXIT" \
        --out-dir "$OUTDIR" \
        --output-json "$OUTDIR/output.json" \
        > "$OUTDIR/run.log" 2>&1
  } 2> "$OUTDIR/time.txt"

  ./.venv/bin/python - <<'PY' "$OUTDIR/output.json" "$SUMMARY" "$NP"
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
summary = Path(sys.argv[2])
np_ranks = int(sys.argv[3])
obj = json.loads(out_path.read_text())
first_linear = (obj.get("linear_history") or [{}])[0]
with summary.open("a", encoding="utf-8") as f:
    f.write(
        f"{np_ranks}\t"
        f"{obj['status']}\t"
        f"{obj['total_time']}\t"
        f"{obj['solve_time']}\t"
        f"{first_linear.get('t_assemble', 0.0)}\t"
        f"{first_linear.get('t_setup', 0.0)}\t"
        f"{first_linear.get('t_solve', 0.0)}\n"
    )
PY
  echo "[$(date --iso-8601=seconds)] finished np=${NP}" | tee -a "$ROOT/serial_master.log"
done

P4_L1_2_SCALING_ROOT="$ROOT" \
P4_L1_2_SCALING_OUTDIR="$REPORT_OUTDIR" \
P4_L1_2_SCALING_REPORT="$REPORT_PATH" \
P4_L1_2_SCALING_RANKS="$RANKS_CSV" \
P4_L1_2_SCALING_TITLE='`P4(L1_2), lambda = 1.5` Parallel Scaling Report' \
P4_L1_2_SCALING_PROBLEM='`hetero_ssr_L1_2`' \
P4_L1_2_SCALING_HIERARCHY='`P4(L1_2) -> P2(L1_2) -> P1(L1_2) -> P1(L1)`' \
P4_L1_2_SCALING_LINEAR_STACK='`fgmres + PMG` local_pmg solver, uniform-refined hierarchy, `ksp_rtol = 1e-1`' \
P4_L1_2_SCALING_NONLINEAR_STACK='elastic initial guess, constitutive autodiff assembly, local residual-bisection Newton' \
P4_L1_2_SCALING_DISTRIBUTION='`overlap_p2p`, `rank_local`, `owned_rows`, `block_xyz`' \
P4_L1_2_SCALING_FIXED_WORK="fixed \`${NEWTON_MAXIT}\` Newton iterations (\`stop_tol = 0\`, \`maxit = ${NEWTON_MAXIT}\`)" \
P4_L1_2_SCALING_THREAD_CAPS='all BLAS/OpenMP/JAX thread counts forced to `1` per MPI rank' \
./.venv/bin/python "$REPORT_SCRIPT" >> "$ROOT/serial_master.log" 2>&1
echo "[$(date --iso-8601=seconds)] report generated" | tee -a "$ROOT/serial_master.log"
