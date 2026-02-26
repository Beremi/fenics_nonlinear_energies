#!/bin/bash
# Detailed breakdown benchmarks: verbose output for key configurations
set -e
cd /workdir

OUT=/workdir/experiment_scripts/bench_local_coloring_detail.txt
> "$OUT"

for PC in gamg hypre; do
  for NP in 1 4 16; do
    for MODE in "" "--local-coloring"; do
      TAG="ORIGINAL"
      if [ -n "$MODE" ]; then TAG="LOCAL-COLORING"; fi
      echo "=== $TAG pc=$PC np=$NP ===" | tee -a "$OUT"
      mpirun -n $NP python3 pLaplace2D_jax_petsc/solve_pLaplace_dof.py \
        --level 9 --pc-type $PC $MODE 2>&1 | tee -a "$OUT"
      echo "" | tee -a "$OUT"
    done
  done
done

echo "DONE" | tee -a "$OUT"
