# Maintained Replications

- campaign root: `replications/2026-03-16_maintained_refresh`
- manifest: `replications/2026-03-16_maintained_refresh/manifest.json`
- commands: `replications/2026-03-16_maintained_refresh/commands.md`

## Benchmark Suites

| id | family | status | path | duration [s] |
| --- | --- | --- | --- | --- |
| gl_final_figures | ginzburg_landau | completed | replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_report_figures | 1.6620437790406868 |
| gl_final_suite | ginzburg_landau | completed | replications/2026-03-16_maintained_refresh/runs/ginzburg_landau/final_suite | 647.7487799039809 |
| he_final_figures | hyperelasticity | completed | replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_report_figures | 2.202758418978192 |
| he_final_suite_best | hyperelasticity | completed | replications/2026-03-16_maintained_refresh/runs/hyperelasticity/final_suite_best | 22516.83976472309 |
| he_pure_jax_suite_best | hyperelasticity | completed | replications/2026-03-16_maintained_refresh/runs/hyperelasticity/pure_jax_suite_best | 5520.834975063102 |
| plaplace_final_figures | plaplace | completed | replications/2026-03-16_maintained_refresh/runs/plaplace/final_report_figures | 1.7526085280114785 |
| plaplace_final_suite | plaplace | completed | replications/2026-03-16_maintained_refresh/runs/plaplace/final_suite | 482.9918622710975 |
| topology_parallel_final_report | topology | completed | replications/2026-03-16_maintained_refresh/_tasks/topology_parallel_final_report | 2.6775666599860415 |
| topology_parallel_final_solver | topology | completed | replications/2026-03-16_maintained_refresh/runs/topology/parallel_final | 24.685620011994615 |
| topology_parallel_scaling | topology | completed | replications/2026-03-16_maintained_refresh/runs/topology/parallel_scaling | 598.7573569130618 |
| topology_serial_report | topology | completed | replications/2026-03-16_maintained_refresh/runs/topology/serial_reference | 37.101236413931474 |

## Example Runs

| id | family | status | path |
| --- | --- | --- | --- |
| gl_fenics_custom | ginzburg_landau | completed | replications/2026-03-16_maintained_refresh/runs/examples/gl_fenics_custom |
| gl_fenics_snes | ginzburg_landau | completed | replications/2026-03-16_maintained_refresh/runs/examples/gl_fenics_snes |
| gl_jax_petsc_element | ginzburg_landau | completed | replications/2026-03-16_maintained_refresh/runs/examples/gl_jax_petsc_element |
| he_fenics_custom | hyperelasticity | completed | replications/2026-03-16_maintained_refresh/runs/examples/he_fenics_custom |
| he_fenics_snes | hyperelasticity | completed | replications/2026-03-16_maintained_refresh/runs/examples/he_fenics_snes |
| he_jax_petsc_element | hyperelasticity | completed | replications/2026-03-16_maintained_refresh/runs/examples/he_jax_petsc_element |
| he_jax_serial | hyperelasticity | completed | replications/2026-03-16_maintained_refresh/runs/examples/he_jax_serial |
| plaplace_fenics_custom | plaplace | completed | replications/2026-03-16_maintained_refresh/runs/examples/plaplace_fenics_custom |
| plaplace_fenics_snes | plaplace | completed | replications/2026-03-16_maintained_refresh/runs/examples/plaplace_fenics_snes |
| plaplace_jax_petsc_element | plaplace | completed | replications/2026-03-16_maintained_refresh/runs/examples/plaplace_jax_petsc_element |
| plaplace_jax_serial | plaplace | completed | replications/2026-03-16_maintained_refresh/runs/examples/plaplace_jax_serial |
| topology_parallel_reference | topology | completed | replications/2026-03-16_maintained_refresh/runs/examples/topology_parallel_reference |
| topology_serial_reference | topology | completed | replications/2026-03-16_maintained_refresh/runs/examples/topology_serial_reference |

## Model Cards

- `model_cards/plaplace.md`
- `model_cards/ginzburg_landau.md`
- `model_cards/hyperelasticity.md`
- `model_cards/topology.md`
- `final_summary.md`

## Comparison Reports

- `comparisons/plaplace/direct_speed.md`
- `comparisons/ginzburg_landau/direct_speed.md`
- `comparisons/hyperelasticity/direct_speed.md`
- `comparisons/topology/direct_speed.md`

## Issues

- `issues/gl_jax_petsc_direct_output_schema.md`
- `issues/he_jax_petsc_trust_region_cli_flags.md`
- `issues/he_suite_resume_restart.md`
- `issues/plaplace_fenics_snes_parallel_mesh_construction.md`
