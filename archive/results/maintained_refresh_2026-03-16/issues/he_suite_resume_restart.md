# HE Suite Resume Restart

- status: `resolved`
- family: `hyperelasticity`
- affected area: `experiments/runners/run_replications.py`

## Symptom

When the maintained HE suite was interrupted and restarted through the replication launcher, the suite task restarted from the beginning instead of resuming from the already written campaign rows.

## Smallest reproducer

1. Start the maintained suite campaign.
2. Interrupt during `he_final_suite_best`.
3. Restart with:

```bash
./.venv/bin/python experiments/runners/run_replications.py \
  --out-dir replications/2026-03-16_maintained_refresh \
  --only suites \
  --resume
```

## Cause

The replication launcher correctly used campaign-level `--resume`, but it still passed `--no-resume` through to the maintained p-Laplace, GL, and HE suite runners themselves. That behavior is fine for a brand-new run and wrong for an interrupted run.

## Repair

- removed `--no-resume` from the maintained suite task commands in the replication launcher
- added a regression test that suite tasks allow resume after interruption
- rebuilt the HE suite summary from the already written per-case JSON files before restarting the detached continuation

## Validation

- `tests/test_replication_runner.py`
- `replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_suite_best/summary.json`
- `replications/2026-03-16_maintained_refresh/continue_campaign.sh`
