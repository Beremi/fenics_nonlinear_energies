# Documentation

`docs/` is the canonical documentation surface for the current repository state.
If you are new to the codebase, start here rather than in `archive/`.

## Start Here

- Setup and environment:
  - [Quickstart](setup/quickstart.md)
  - [Local build](setup/local_build.md)
- Problem overviews:
  - [pLaplace](problems/pLaplace.md)
  - [pLaplace_u3 thesis replications](problems/pLaplace_u3_thesis_replications.md)
  - [pLaplace_up_arctan](problems/pLaplace_up_arctan.md)
  - [GinzburgLandau](problems/GinzburgLandau.md)
  - [HyperElasticity](problems/HyperElasticity.md)
  - [Plasticity](problems/Plasticity.md)
  - [Plasticity3D](problems/Plasticity3D.md)
  - [Topology](problems/Topology.md)
- Current maintained results:
  - [pLaplace results](results/pLaplace.md)
  - [GinzburgLandau results](results/GinzburgLandau.md)
  - [HyperElasticity results](results/HyperElasticity.md)
  - [Plasticity results](results/Plasticity.md)
  - [Plasticity3D results](results/Plasticity3D.md)
  - [Topology results](results/Topology.md)

## Structure

- `setup/`: how to build and run the maintained solvers
- `problems/`: mathematical setup, geometry, maintained implementations, and one curated sample result per family
- `results/`: current maintained comparison tables, scaling figures, and reproduction commands
- `implementation/`: current implementation notes that still describe the active code
- `reference/`: solver-fit and formulation notes that are useful across problem families
- `assets/`: curated figures used by the current docs

## Deeper Notes

Current implementation and reference material:

- [Trust-region + line-search algorithm](implementation/trust_region_linesearch_algorithm.md)
- [Plasticity3D autodiff modes](implementation/plasticity3d_autodiff_modes.md)
- [HyperElasticity JAX+PETSc implementation](implementation/hyperelasticity_jax_petsc.md)
- [Topology JAX+PETSc implementation](implementation/topology_jax_petsc.md)
- [Problem formulation brief](reference/problem_formulation_brief.md)
- [GAMG setup for elastic-like systems](reference/he_gamg_elasticity_setup.md)

## Historical Material

Historical benchmark reports, refactor logs, tuning notes, and superseded
overview pages are preserved under `archive/docs/`. They remain useful for
provenance, but they are no longer the primary place to find the current run
instructions or the current maintained results.
