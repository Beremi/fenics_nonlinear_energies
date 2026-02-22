#!/bin/bash
# Run all GL custom Newton benchmarks
set -e
cd /work

SOLVER="GinzburgLandau2D_fenics/solve_GL_custom_jaxversion.py"
OUTDIR="results_GL/experiment_001"
LEVELS="5 6 7 8 9"

echo "=== Custom Newton benchmarks ==="

for NP in 1 4 8 16; do
    for RUN in 1 2 3; do
        OUTFILE="${OUTDIR}/custom_jaxversion_np${NP}_run${RUN}.json"
        echo "--- np=${NP}, run=${RUN} ---"
        if [ "$NP" -eq 1 ]; then
            python3 ${SOLVER} --levels ${LEVELS} --quiet --json ${OUTFILE}
        else
            mpirun -n ${NP} python3 ${SOLVER} --levels ${LEVELS} --quiet --json ${OUTFILE}
        fi
    done
done

echo ""
echo "=== SNES Newton benchmarks ==="

SOLVER="GinzburgLandau2D_fenics/solve_GL_snes_newton.py"

for NP in 1 4 8 16; do
    for RUN in 1 2 3; do
        OUTFILE="${OUTDIR}/snes_newton_np${NP}_run${RUN}.json"
        echo "--- np=${NP}, run=${RUN} ---"
        if [ "$NP" -eq 1 ]; then
            python3 ${SOLVER} --levels ${LEVELS} --json ${OUTFILE}
        else
            mpirun -n ${NP} python3 ${SOLVER} --levels ${LEVELS} --json ${OUTFILE}
        fi
    done
done

echo ""
echo "=== ALL DONE ==="
ls -la ${OUTDIR}/
