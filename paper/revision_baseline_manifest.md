# Revision Baseline Manifest

Captured: 2026-07-10.

## Source State

- Branch: `paper-publish-readiness`.
- HEAD: `4d2bea0abbb01e4834e6f0b72312280a4a35d7de`.
- Worktree at capture: dirty, containing pre-existing paper-readiness work and
  the new publication-plan work.
- Dirty-entry count at capture: 82 (`69` modified, `5` deleted, and `8`
  untracked entries).

The dirty files are user-owned working state and must not be reset or silently
discarded. This manifest records the baseline for development and local smoke
tests only. Publication-grade timing campaigns require a later clean,
identified experiment commit.

## Historical Paper Artifact

- `paper/build/main.pdf` SHA-256:
  `01698776cd6236345a8f537d288ddfa8cf64ba15dd3e2bbacaf72252ca1b2e43`.
- Historical submission manifest:
  `artifacts/reproduction/paper_submission_2026_07_08/manifest.json`.
- Historical manifest SHA-256:
  `5c731ed0b592d29ae1fa3f2912f23256de5c5b38a47cd553cb9c33fbc9abce1e`.

The July 8 bundle is historical evidence. The revision pipeline must not delete
or overwrite it. New experiment outputs will use a distinct revision campaign
root and explicit campaign/run identifiers.

## Scientific Baseline

- The current PDF predates the direct derivative-verification, common stopping,
  replicated timing, crossover, and KKT campaigns specified in
  [`publication_action_plan.md`](publication_action_plan.md).
- Historical exact timings and topology endpoints remain pilot/historical
  evidence until rerun under the new contracts.
- The selected revision route and current closest-work audit are recorded in
  [`venue_and_contribution_decision.md`](venue_and_contribution_decision.md).

## Gate

This baseline is frozen for comparison. Before a final experiment is run, add a
clean experiment-commit record containing an empty `git status --porcelain`,
the exact commit, environment, inputs, commands, and campaign manifest.
