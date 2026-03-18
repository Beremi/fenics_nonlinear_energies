# JAX Topology Optimisation Benchmark

Date: 2026-03-15

This report fixes the JAX topology benchmark to a single clean reference
configuration: a fine `192 x 96` cantilever mesh, a staggered
displacement/design solve, and a fixed staircase SIMP continuation.
The intent is no longer to compare continuation heuristics; it is to
document one compact working implementation that demonstrates how the
repository can define energies in JAX, autodifferentiate them, assemble
sparse Hessians on fixed graphs, and solve a nontrivial benchmark.

## Benchmark Definition

The domain is the cantilever rectangle

$$
\Omega = [0, L] \times [0, H],
$$

with the left edge clamped and a downward traction patch on the right
edge. The design variable is a latent nodal field $z$, mapped to a
physical density by

$$
\theta(z) = \theta_{\min} + (1 - \theta_{\min})\,\sigma(z),
\qquad
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

The mechanics energy for fixed design is

$$
\Pi_h(u; z_k)
=
\sum_e A_e\,\frac{1}{2}\,\theta_e(z_k)^p\,\varepsilon_e(u)^T C\,\varepsilon_e(u)
- f^T u,
$$

and the frozen design energy is

$$
G_h(z; u_{k+1})
=
\sum_e A_e
\left[
e_{k,e}\,\theta_e(z)^{-p}
+ \lambda_k\,\theta_e(z)
+ \alpha\left(\frac{\ell}{2}|\nabla \theta_e(z)|^2 + \frac{W(\theta_e(z))}{\ell}\right)
+ \frac{\mu}{2}(\bar z_e - \bar z^{old}_e)^2
\right],
$$

with

$$
W(\theta) = \theta^2(1-\theta)^2.
$$

The SIMP exponent follows the fixed staircase schedule

$$
p_{k+1} =
\min\bigl(p_{\max},\, p_k + \Delta p\bigr)
\quad \text{every } m \text{ outer iterations},
$$

using `\Delta p = 0.5` and `m = 20`.

## Reference Configuration

| Knob | Value |
| --- | --- |
| Mesh | 192 x 96 |
| Elements | 36864 |
| Free displacement DOFs | 37248 |
| Free design DOFs | 16205 |
| Target volume fraction | 0.4000 |
| Staircase schedule | p = p + 0.5 every 20 outer iterations |
| Final p target | 4.00 |
| Volume control | beta_lambda = 12.0, volume_penalty = 10.0 |
| Regularisation | alpha = 0.005, ell = 0.08, mu_move = 0.01 |

## Minimal JAX Problem Definition

The problem-specific input to JAX is just the energy definition. The
following two functions are the parts that are actually autodifferentiated:

```python
def mechanics_energy(
    u_free, u_0, freedofs, elems, elem_B, elem_area, material_scale, constitutive, force
):
    u_full = expand_free_dofs(u_free, u_0, freedofs)
    u_elem = u_full[elems]
    strain = jnp.einsum("eij,ej->ei", elem_B, u_elem)
    elastic_density = 0.5 * jnp.einsum("ei,ij,ej->e", strain, constitutive, strain)
    return jnp.sum(elem_area * material_scale * elastic_density) - jnp.dot(force, u_full)


def design_energy(
    z_free, z_0, freedofs, elems, elem_grad_phi, elem_area, e_frozen,
    z_old_full, lambda_volume, alpha_reg, ell_pf, mu_move, theta_min, p_penal
):
    z_full = expand_free_dofs(z_free, z_0, freedofs)
    theta_full = theta_from_latent(z_full, theta_min)
    theta_elem = theta_full[elems]
    theta_centroid = jnp.mean(theta_elem, axis=1)
    grad_theta = jnp.einsum("eia,ei->ea", elem_grad_phi, theta_elem)
    z_delta_centroid = jnp.mean(z_full[elems] - z_old_full[elems], axis=1)

    double_well = theta_centroid**2 * (1.0 - theta_centroid) ** 2
    reg_density = 0.5 * ell_pf * jnp.sum(grad_theta * grad_theta, axis=1) + double_well / ell_pf
    proximal_density = 0.5 * mu_move * z_delta_centroid**2
    design_density = e_frozen * theta_centroid ** (-p_penal) + lambda_volume * theta_centroid
    return jnp.sum(elem_area * (design_density + alpha_reg * reg_density + proximal_density))
```

## Where JAX Is Used

The solver asks JAX for derivatives with respect to the free unknowns of
each subproblem only:

| Energy | Differentiated with respect to | What is generated |
| --- | --- | --- |
| $\Pi_h(u; z_k)$ | `u_free` | gradient and sparse Hessian of the mechanics subproblem |
| $G_h(z; u_{k+1})$ | `z_free` | gradient and sparse Hessian of the design subproblem |

The corresponding calls are:

```python
mechanics_drv = EnergyDerivator(
    mechanics_energy,
    mechanics_params,
    mesh.adjacency_u,
    jnp.asarray(u_free, dtype=jnp.float64),
)
mechanics_F, mechanics_dF, mechanics_ddF = mechanics_drv.get_derivatives()
mechanics_hess_solver = HessSolverGenerator(
    ddf=mechanics_ddF,
    solver_type="amg",
    elastic_kernel=mesh.elastic_kernel,
    tol=ksp_rtol,
    maxiter=ksp_max_it,
)

design_drv = EnergyDerivator(
    design_energy,
    design_params,
    mesh.adjacency_z,
    jnp.asarray(z_free, dtype=jnp.float64),
)
design_F, design_dF, design_ddF = design_drv.get_derivatives()
design_hess_solver = HessSolverGenerator(
    ddf=design_ddF,
    solver_type="direct",
    tol=ksp_rtol,
    maxiter=ksp_max_it,
)
```

`EnergyDerivator` provides the energy, gradient, and sparse Hessian-value
callbacks. `HessSolverGenerator` then builds the linear solve stage used
inside Newton: AMG for mechanics and a direct sparse solve for the design
subproblem.

The continuation itself is intentionally fixed and minimal:

```python
def staircase_p_step(p_penal, *, p_max, p_increment, continuation_interval, outer_it):
    if p_penal >= p_max or outer_it % continuation_interval != 0:
        return 0.0
    return min(p_increment, p_max - p_penal)
```

## Solver Structure

### Outer staggered loop

```text
build mesh, free-DOF masks, element operators
initialize z_0 from the target volume fraction and set u_0 = 0
define Π_h(u_free; z) and G_h(z_free; u, z_old, lambda, p)
use JAX to generate:
    grad_u Π_h, Hess_u Π_h on the displacement graph
    grad_z G_h, Hess_z G_h on the design graph
build sparse linear solvers for both Hessians

for outer iteration k = 1, 2, ...:
    theta_k <- theta_from_latent(z_k)
    material_scale <- theta(theta_k)^p_k

    solve mechanics subproblem in u_free:
        Newton steps use grad_u Π_h and Hess_u Π_h
        each Newton step solves a sparse linear system in the displacement unknowns

    freeze element strain-energy density e_k from u_{k+1}
    build lambda_effective from the sensitivity quantile, lambda_k, and volume penalty

    solve design subproblem in z_free:
        Newton steps use grad_z G_h and Hess_z G_h
        each Newton step solves a sparse linear system in the design unknowns

    update lambda_k from the achieved volume error
    record compliance, volume, design change, and Newton counts
    if p_k is already at p_max and all outer tolerances are satisfied:
        stop
    otherwise update p_k with the fixed staircase rule
```

### Inner Newton solve

```text
given x_n:
    evaluate F(x_n)
    evaluate g_n = grad F(x_n)        <- JAX autodiff
    evaluate H_n = Hess F(x_n)        <- JAX autodiff on fixed sparse graph
    solve H_n * delta = -g_n          <- sparse linear solver
    line-search / trust-region accept or reject the step
    x_{n+1} = x_n + alpha * delta
repeat until function and gradient tolerances are met
```

## Geometry And Final State

![Mesh preview](artifacts/figures/benchmark_reports/topology_serial_reference/mesh_preview.png)

![Final state](artifacts/figures/benchmark_reports/topology_serial_reference/final_state.png)

The density plot is elementwise constant: each triangle is coloured by
its average density $\theta_e$.

## Convergence History

![Convergence history](artifacts/figures/benchmark_reports/topology_serial_reference/convergence_history.png)

## Density Evolution

![Density evolution](artifacts/figures/benchmark_reports/topology_serial_reference/density_evolution.gif)

## Run Summary

| Metric | Value |
| --- | --- |
| Result | completed |
| Outer iterations | 121 |
| Final p | 4.0000 |
| Wall time [s] | 14.664 |
| JAX setup [s] | 1.624 |
| Solve time [s] | 13.039 |
| Final compliance | 4.155670 |
| Final volume fraction | 0.400000 |
| Final volume error | 0.000000 |
| Final state change | 0.000000 |
| Final design change | 0.000000 |
| Final compliance change | 0.000000 |
| Total mechanics Newton iterations | 17 |
| Total design Newton iterations | 0 |

## Density Quality Indicators

| Indicator | Value |
| --- | --- |
| Gray fraction on 0.1 < theta < 0.9 | 0.8656 |
| Gray fraction on 0.05 < theta < 0.95 | 0.8656 |
| theta_min | 0.309095 |
| theta_max | 0.999955 |

## JAX Setup Timings

| Stage | Time [s] |
| --- | --- |
| mechanics: coloring | 0.234420 |
| mechanics: compilation | 0.227010 |
| design: coloring | 0.044850 |
| design: compilation | 0.272606 |

## Outer Iteration Table

| k | p | dp | lambda_eff | mech iters | design iters | compliance | volume | vol error | state change | comp change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.000 | 0.000 | 4.07952 | 17 | 0 | 4.155670 | 0.400000 | 0.000000 | inf | inf |
| 2 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 6 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 10 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 12 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 13 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 14 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 15 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 16 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 17 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 18 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 19 | 1.000 | 0.000 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 20 | 1.000 | 0.500 | 4.07952 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 21 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 22 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 23 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 24 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 25 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 26 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 27 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 28 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 29 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 30 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 31 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 32 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 33 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 34 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 35 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 36 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 37 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 38 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 39 | 1.500 | 0.000 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 40 | 1.500 | 0.500 | 3.56474 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 41 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 42 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 43 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 44 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 45 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 46 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 47 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 48 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 49 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 50 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 51 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 52 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 53 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 54 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 55 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 56 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 57 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 58 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 59 | 2.000 | 0.000 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 60 | 2.000 | 0.500 | 2.77520 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 61 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 62 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 63 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 64 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 65 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 66 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 67 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 68 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 69 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 70 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 71 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 72 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 73 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 74 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 75 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 76 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 77 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 78 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 79 | 2.500 | 0.000 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 80 | 2.500 | 0.500 | 2.03846 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 81 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 82 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 83 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 84 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 85 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 86 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 87 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 88 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 89 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 90 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 91 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 92 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 93 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 94 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 95 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 96 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 97 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 98 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 99 | 3.000 | 0.000 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 100 | 3.000 | 0.500 | 1.44593 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 101 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 102 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 103 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 104 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 105 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 106 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 107 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 108 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 109 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 110 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 111 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 112 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 113 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 114 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 115 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 116 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 117 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 118 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 119 | 3.500 | 0.000 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 120 | 3.500 | 0.500 | 1.02966 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |
| 121 | 4.000 | 0.000 | 0.70860 | 0 | 0 | 4.155670 | 0.400000 | 0.000000 | 0.000000 | 0.000000 |

## Artifacts

- JSON result: `artifacts/figures/benchmark_reports/topology_serial_reference/report_run.json`
- Final state: `artifacts/figures/benchmark_reports/topology_serial_reference/report_state.npz`
- Outer-history CSV: `artifacts/figures/benchmark_reports/topology_serial_reference/report_outer_history.csv`
- Mesh figure: `artifacts/figures/benchmark_reports/topology_serial_reference/mesh_preview.png`
- Final-state figure: `artifacts/figures/benchmark_reports/topology_serial_reference/final_state.png`
- Convergence figure: `artifacts/figures/benchmark_reports/topology_serial_reference/convergence_history.png`
- Density-evolution GIF: `artifacts/figures/benchmark_reports/topology_serial_reference/density_evolution.gif`

## Reproduction

Regenerate the benchmark report and assets with:

```bash
        ./.venv/bin/python experiments/analysis/generate_report_assets.py
        ```

        Run the solver directly with the same fine-grid staircase setup:

        ```bash
        ./.venv/bin/python src/problems/topology/jax/solve_topopt_jax.py \
    --nx 192 --ny 96 --length 2.0 --height 1.0 \
    --traction 1.0 --load_fraction 0.2 \
    --fixed_pad_cells 16 --load_pad_cells 16 \
    --volume_fraction_target 0.4 --theta_min 0.001 \
    --solid_latent 10.0 --young 1.0 --poisson 0.3 \
    --alpha_reg 0.005 --ell_pf 0.08 --mu_move 0.01 \
    --beta_lambda 12.0 --volume_penalty 10.0 \
    --p_start 1.0 --p_max 4.0 --p_increment 0.5 \
    --continuation_interval 20 --outer_maxit 180 \
    --outer_tol 0.02 --volume_tol 0.001 \
    --mechanics_maxit 200 --design_maxit 400 \
    --tolf 1e-06 --tolg 0.001 \
    --ksp_rtol 0.01 --ksp_max_it 80 --save_outer_state_history --quiet \
    --json_out artifacts/figures/benchmark_reports/topology_serial_reference/report_run.json --state_out artifacts/figures/benchmark_reports/topology_serial_reference/report_state.npz
        ```
