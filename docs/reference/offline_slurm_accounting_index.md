# Offline Slurm accounting index

`generate_offline_accounting_index.py` turns already captured raw
`sacct --parsable2` text into the only index shape accepted by the Karolina
archive finalizers. It never invokes or queries Slurm.

## Capture contract

Capture accounting only after every submitted job has settled, outside this
offline workflow. Put the raw UTF-8 outputs in a fresh directory detached from
the campaign archive. The directory must contain exactly one regular file per
accepted job, named `<job-id>.sacct`; it may contain no subdirectory, symlink,
missing job, or additional file.

Generate and then independently verify the index:

```bash
./.venv/bin/python experiments/analysis/generate_offline_accounting_index.py \
  generate --campaign-root <submitted-campaign-root> \
  --snapshot-root <detached-snapshot-root> \
  --output <detached-snapshot-root>/accounting-index.json

./.venv/bin/python experiments/analysis/generate_offline_accounting_index.py \
  verify --campaign-root <submitted-campaign-root> \
  --snapshot-root <detached-snapshot-root> \
  --output <detached-snapshot-root>/accounting-index.json
```

Generation refuses to overwrite. Verification reconstructs the complete index
and requires byte-canonical JSON. The index intentionally contains no creation
time, host, or directory-dependent absolute path, so the same campaign
manifest and raw bytes produce the same bytes.

For each allocation row the generator checks the submitted case/job identity,
Karolina cluster, account `fta-26-40`, QoS `3571_6328`, partition, node count,
CPU/rank count, terminal `COMPLETED` state, and `0:0` exit code. It hashes the
raw file and binds the index to the current submitted-manifest hash. Changed
raw bytes, a changed manifest, stale job identity, failed allocation, resource
mismatch, missing coverage, extra files, path escape, or symlink fails closed.

Pass the verified index to `finalize_karolina_campaign_archive.py` for the
route/discretization/scaling matrix or to
`finalize_reviewed_karolina_archive.py` for the reviewed STOP/GLOB campaigns.
Those finalizers copy the raw text into each job directory, reparse it again,
and checksum-seal the complete campaign. An index is provenance, not evidence
that a job was run in the current session, and it supports no scientific claim
on its own.
