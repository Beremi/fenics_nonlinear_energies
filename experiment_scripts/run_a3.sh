#!/bin/bash
set -e
cd /work
rm -f tmp_work/a3_ksp*.txt

echo "=== A3: newtontr + ASM+ILU Benchmark ==="
echo "Date: $(date)"

for ksp in 1e-3 1e-2 1e-1; do
    for np in 1 2 4 8 16; do
        echo "--- ksp_rtol=$ksp np=$np ---"
        if [ "$np" -eq 1 ]; then
            KSP_RTOL=$ksp python3 tmp_work/bench_a3.py
        else
            KSP_RTOL=$ksp mpirun -n $np python3 tmp_work/bench_a3.py
        fi
        echo ""
    done
done

echo "=== ALL DONE ==="
