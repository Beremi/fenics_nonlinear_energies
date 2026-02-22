#!/bin/bash
# Run all TR + HYPRE benchmarks with optimized timing.
# Output: tmp_work/tr2_ksp*_np*.txt files + summary to stdout

set -e
cd /work

# Clean old results
rm -f tmp_work/tr2_ksp*.txt

echo "=== Trust Region + HYPRE Benchmark ==="
echo "Date: $(date)"

for ksp in 1e-3 1e-2 1e-1; do
    for np in 1 2 4 8 16; do
        echo "--- ksp_rtol=$ksp np=$np ---"
        if [ "$np" -eq 1 ]; then
            KSP_RTOL=$ksp python3 tmp_work/bench_tr2.py
        else
            KSP_RTOL=$ksp mpirun -n $np python3 tmp_work/bench_tr2.py
        fi
        echo ""
    done
done

echo "=== ALL DONE ==="
