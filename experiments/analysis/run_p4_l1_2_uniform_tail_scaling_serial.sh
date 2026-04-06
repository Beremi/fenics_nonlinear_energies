#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-artifacts/raw_results/scaling_probe/p4_l1_2_uniform_tail_maxit1_threads1}"
CHUNK_SIZE="${P4_HESSIAN_CHUNK_SIZE:-8}"
RANKS=(1 2 4 8 16 32)

mkdir -p "$ROOT"
SUMMARY="$ROOT/run_summary.tsv"
printf 'np\tstatus\ttotal_time\tsolve_time\tlinear1_assemble\tlinear1_setup\tlinear1_solve\n' > "$SUMMARY"

COMMON_ARGS=(
  -u -m src.problems.slope_stability_3d.jax_petsc.solve_slope_stability_3d_dof
  --mesh_name hetero_ssr_L1_2
  --elem_degree 4
  --lambda-target 1.5
  --profile performance
  --ksp_type fgmres
  --pc_type mg
  --ksp_rtol 1e-2
  --ksp_max_it 100
  --accept_ksp_maxit_direction
  --ksp_maxit_direction_true_rel_cap 0.06
  --distribution_strategy overlap_p2p
  --problem_build_mode rank_local
  --mg_level_build_mode rank_local
  --mg_transfer_build_mode owned_rows
  --element_reorder_mode block_xyz
  --mg_strategy uniform_refined_p4_p2_p1_p1
  --use_near_nullspace
  --mg_coarse_backend hypre
  --mg_coarse_ksp_type cg
  --mg_coarse_pc_type hypre
  --mg_coarse_hypre_nodal_coarsen 6
  --mg_coarse_hypre_vec_interp_variant 3
  --mg_coarse_hypre_strong_threshold 0.5
  --mg_coarse_hypre_coarsen_type HMIS
  --mg_coarse_hypre_max_iter 2
  --mg_coarse_hypre_tol 0.0
  --mg_coarse_hypre_relax_type_all symmetric-SOR/Jacobi
  --mg_p1_smoother_ksp_type chebyshev
  --mg_p1_smoother_pc_type jacobi
  --mg_p1_smoother_steps 5
  --mg_p2_smoother_ksp_type chebyshev
  --mg_p2_smoother_pc_type jacobi
  --mg_p2_smoother_steps 5
  --mg_p4_smoother_ksp_type chebyshev
  --mg_p4_smoother_pc_type jacobi
  --mg_p4_smoother_steps 5
  --p4_hessian_chunk_size "$CHUNK_SIZE"
  --line_search armijo
  --armijo_alpha0 1.0
  --armijo_c1 1e-4
  --armijo_shrink 0.5
  --armijo_max_ls 40
  --elastic_initial_guess
  --no-regularized_newton_tangent
  --tolg 1e-2
  --tolg_rel 0.0
  --maxit 1
  --save_history
  --debug_setup
  --quiet
)

for NP in "${RANKS[@]}"; do
  OUTDIR="$ROOT/np${NP}"
  mkdir -p "$OUTDIR"
  rm -f "$OUTDIR/output.json" "$OUTDIR/progress.json" "$OUTDIR/run.log" "$OUTDIR/time.txt"
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
      mpiexec -n "$NP" ./.venv/bin/python "${COMMON_ARGS[@]}" \
        --nproc "$NP" \
        --out "$OUTDIR/output.json" \
        --progress-out "$OUTDIR/progress.json" \
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
diag = obj["parallel_diagnostics"][0]
lin = diag["linear_history"][0]
with summary.open("a", encoding="utf-8") as f:
    f.write(
        f"{np_ranks}\t"
        f"{obj['status']}\t"
        f"{obj['total_time']}\t"
        f"{obj['solve_time']}\t"
        f"{lin['t_assemble']}\t"
        f"{lin['t_setup']}\t"
        f"{lin['t_solve']}\n"
    )
PY
  echo "[$(date --iso-8601=seconds)] finished np=${NP}" | tee -a "$ROOT/serial_master.log"
done

./.venv/bin/python experiments/analysis/generate_p4_l1_2_uniform_tail_scaling_assets.py >> "$ROOT/serial_master.log" 2>&1
echo "[$(date --iso-8601=seconds)] report generated" | tee -a "$ROOT/serial_master.log"
