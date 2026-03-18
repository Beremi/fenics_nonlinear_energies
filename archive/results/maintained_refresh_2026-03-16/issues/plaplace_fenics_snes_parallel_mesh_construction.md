# p-Laplace FEniCS SNES Parallel Mesh Construction

- status: `resolved`
- family: `plaplace`
- affected area: `src/problems/plaplace/fenics/solver_snes.py`

## Symptom

The maintained p-Laplace SNES solver crashed in MPI on the level-5 comparison case with allocator corruption.

## Smallest reproducer

```bash
mpiexec -n 2 ./.venv/bin/python src/problems/plaplace/fenics/solve_pLaplace_snes_newton.py \
  --levels 5 \
  --json /tmp/plaplace_snes_np2.json
```

## Cause

The SNES mesh loader used a root-load plus `bcast` pattern for the mesh arrays. In this path the DOLFINx mesh
construction was less stable than the maintained custom solver pattern that lets rank 0 load the mesh and gives
other ranks empty arrays before `mesh.create_mesh(...)`.

## Repair

- switched the SNES mesh construction to the same safer pattern already used by the maintained custom solver
- added a regression test for the `np=2` level-5 MPI solve
- reran the direct comparison matrix for p-Laplace

## Validation

- `tests/test_plaplace_snes_mpi.py`
- `replications/2026-03-16_maintained_refresh/comparisons/plaplace/`
