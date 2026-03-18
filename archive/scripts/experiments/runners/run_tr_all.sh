#!/bin/bash
# Run all trust-region benchmarks: ksp_rtol × nprocs
# Results go to tmp_work/tr_all_results.txt
cd /work
OUTFILE=/work/tmp_work/tr_all_results.txt
echo "=== Trust Region Benchmark ===" > $OUTFILE
echo "Date: $(date)" >> $OUTFILE

for KRTOL in 1e-3 1e-2 1e-1; do
    for NP in 1 2 4 8 16; do
        echo "--- ksp_rtol=$KRTOL np=$NP ---" >> $OUTFILE
        echo "Running ksp_rtol=$KRTOL np=$NP ..."
        if [ "$NP" = "1" ]; then
            KSP_RTOL=$KRTOL python3 /work/tmp_work/bench_tr.py >> $OUTFILE 2>/dev/null
        else
            mpirun -n $NP env KSP_RTOL=$KRTOL python3 /work/tmp_work/bench_tr.py >> $OUTFILE 2>/dev/null
        fi
        echo "" >> $OUTFILE
    done
done
echo "=== ALL DONE ===" >> $OUTFILE
echo "ALL DONE"
