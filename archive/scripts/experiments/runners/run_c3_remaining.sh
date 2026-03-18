#!/bin/bash
set -e
cd /work

echo "=== C3: basic + EW + lag_pc=2 + fgmres + HYPRE (remaining) ==="
echo "Date: $(date)"

# ksp=1e-1 np=1 — partial (4 out of 5 levels done already, rerun all to be safe)
rm -f experiment_scripts/c3_ksp01_np1.txt
for np in 1 2 4 8 16; do
    echo "--- ksp_rtol=1e-1 np=$np ---"
    if [ "$np" -eq 1 ]; then
        KSP_RTOL=1e-1 python3 experiment_scripts/bench_c3.py
    else
        KSP_RTOL=1e-1 mpirun -n $np python3 experiment_scripts/bench_c3.py
    fi
    echo ""
done

echo "=== ALL DONE ==="
