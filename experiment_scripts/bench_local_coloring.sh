#!/bin/bash
# Benchmark: original vs local coloring, hypre vs gamg, np=1,2,4,8,16
set -e
cd /workdir

OUT=/workdir/experiment_scripts/bench_local_coloring_results.txt
> "$OUT"

for PC in hypre gamg; do
  for NP in 1 2 4 8 16; do
    echo "=== ORIGINAL pc=$PC np=$NP ===" | tee -a "$OUT"
    mpirun -n $NP python3 pLaplace2D_jax_petsc/solve_pLaplace_dof.py \
      --level 9 --pc-type $PC --quiet 2>&1 | tee -a "$OUT"
    echo "" | tee -a "$OUT"

    echo "=== LOCAL-COLORING pc=$PC np=$NP ===" | tee -a "$OUT"
    mpirun -n $NP python3 pLaplace2D_jax_petsc/solve_pLaplace_dof.py \
      --level 9 --pc-type $PC --quiet --local-coloring 2>&1 | tee -a "$OUT"
    echo "" | tee -a "$OUT"
  done
done

echo "DONE" | tee -a "$OUT"
