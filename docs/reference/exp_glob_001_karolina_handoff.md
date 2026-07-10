# EXP-GLOB-001 Karolina handoff

This handoff covers the prepared, not submitted, full-rank controlled
globalization campaign. It is separate from the route/discretization/scaling
matrix and does not authorize scheduler contact.

## Frozen scope

| Problem | Ranks | Deterministic starts | Methods | Repetitions | Jobs | Per-job ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ginzburg--Landau level 10 | 16 | 3 | 2 | 5 | 30 | 10 min |
| HyperElasticity level 4, first load | 32 | 3 | 2 | 5 | 30 | 15 min |

The complete ceiling is 12.5 Karolina CPU node-hours. All jobs use one
`qcpu_exp` node, one thread per MPI rank, block placement, local NUMA binding,
account `fta-26-40`, and QoS `3571_6328`.

## Scheduler-free preparation

```bash
./.venv/bin/python experiments/runners/prepare_exp_glob_001_karolina.py \
  prepare --output-root <fresh-campaign-root>

./.venv/bin/python experiments/runners/prepare_exp_glob_001_karolina.py \
  preflight --campaign-root <fresh-campaign-root>
```

Preparation requires one clean 40-character commit, generates and hash-binds
the six canonical starts, freezes all 60 payloads, and writes
`sbatch_commands.txt`. It never executes that file. A preparation without both
`--env-setup` and `--env-lock` is deliberately marked
`submission_admissible: false`.

## Future authorized execution and archive order

1. Revalidate the allocation, account, QoS, partitions, source commit, and
   reviewed environment outside this workflow.
2. Use the guarded reviewed-campaign submission utility only under separate
   explicit authorization. It journals an intent before each scheduler call
   and its result afterward.
3. After all jobs copy back, run `prepare_exp_glob_001_karolina.py analyze`.
   This reconstructs 60 strict run records and the common-start/endpoint audit,
   but exposes no claim before accounting settlement.
4. Capture raw `sacct --parsable2` text outside this offline workflow. Build
   and verify the deterministic index with
   `experiments/analysis/generate_offline_accounting_index.py`, then pass it to
   `finalize_reviewed_karolina_archive.py`; the finalizer reparses all 60
   records and writes the archive checksum manifest.
5. Preserve the checksum-manifest digest before copy-back. Run
   `prepare_exp_glob_001_karolina.py adjudicate` with that digest and a detached
   output path. Any missing, additional, changed, or symlinked archive member
   fails closed.

The final receipt can admit comparisons only on the six prescribed starts.
It always keeps `robustness_generalization_claim_admissible: false`. No job has
been submitted or run while preparing this handoff.
