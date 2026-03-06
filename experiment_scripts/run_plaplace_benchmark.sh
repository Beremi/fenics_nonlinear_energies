#!/bin/bash
# Benchmark: pLaplace level 9, 32 MPI, GAMG — SFD vs Element vs FEniCS
# All three solvers use identical Newton/linear settings:
#   ksp=cg, pc=gamg, ksp_rtol=1e-3
#   tolf=1e-5, tolg=1e-3, linesearch=(-0.5, 2.0), maxit=100
#
# Usage (from repo root):
#   bash experiment_scripts/run_plaplace_benchmark.sh
#
# Outputs:
#   experiment_results_cache/bench_plaplace_sfd_32p.json
#   experiment_results_cache/bench_plaplace_element_32p.json
#   experiment_results_cache/bench_plaplace_fenics_32p.json

set -e
cd "$(dirname "$0")/.."
source local_env/activate.sh

NP=32
LEVEL=9
OUT_DIR=experiment_results_cache

mkdir -p "$OUT_DIR"

echo "========================================================"
echo "  pLaplace benchmark: level=$LEVEL np=$NP pc=gamg"
echo "========================================================"

echo ""
echo "--- [1/3] JAX-PETSc SFD ---"
mpirun -n $NP python3 pLaplace2D_jax_petsc/solve_pLaplace_dof.py \
    --level $LEVEL \
    --pc-type gamg \
    --ksp-rtol 1e-3 \
    --tolf 1e-5 \
    --tolg 1e-3 \
    --local-coloring \
    --assembly-mode sfd \
    --json "$OUT_DIR/bench_plaplace_sfd_32p.json"

echo ""
echo "--- [2/3] JAX-PETSc Element ---"
mpirun -n $NP python3 pLaplace2D_jax_petsc/solve_pLaplace_dof.py \
    --level $LEVEL \
    --pc-type gamg \
    --ksp-rtol 1e-3 \
    --tolf 1e-5 \
    --tolg 1e-3 \
    --local-coloring \
    --assembly-mode element \
    --json "$OUT_DIR/bench_plaplace_element_32p.json"

echo ""
echo "--- [3/3] FEniCS custom ---"
mpirun -n $NP python3 pLaplace2D_fenics/solve_pLaplace_custom_jaxversion.py \
    --levels $LEVEL \
    --pc-type gamg \
    --ksp-rtol 1e-3 \
    --json "$OUT_DIR/bench_plaplace_fenics_32p.json"

echo ""
echo "========================================================"
echo "  All runs complete. Analysing results..."
echo "========================================================"

python3 experiment_scripts/analyze_plaplace_benchmark.py \
    "$OUT_DIR/bench_plaplace_sfd_32p.json" \
    "$OUT_DIR/bench_plaplace_element_32p.json" \
    "$OUT_DIR/bench_plaplace_fenics_32p.json"
