# Porting the 3D heterogeneous Mohr–Coulomb SSR benchmark into `jax_petsc_nonlinear_energies`

This document is a full implementation guide for moving the **3D heterogeneous slope-stability benchmark** from

- `Beremi/slope_stability_petsc4py`

into

- `Beremi/jax_petsc_nonlinear_energies`

with the following goals:

1. keep the **3D heterogeneous benchmark definition** and the source repo’s **Mohr–Coulomb plasticity logic**,
2. implement the local constitutive model as a **scalar JAX element energy** so the residual and tangent come from `jax.grad` and `jax.hessian`,
3. keep a runtime **`lambda` / strength-reduction parameter** so the plastic difficulty can be tuned without changing the mesh or the base material tables,
4. support **same-mesh `P1` / `P2` / `P4` tetrahedral spaces**,
5. integrate with the target repo’s **PETSc reordered overlap-domain assembly** and its **PMG hierarchy machinery**,
6. avoid common 3D pitfalls: huge dense element tensors, wrong boundary lifting, wrong gravity axis, wrong shear convention, and trying to force a non-nested h-hierarchy from the source adaptive meshes.

The recommendations below are deliberately biased toward a **correct first port** that can later be optimized, rather than a “perfect on day one” redesign.

---

## 1. The porting strategy in one page

### Recommended end-state

Implement a new problem family, for example:

```text
src/problems/slope_stability_3d/
  support/
    import_source_mesh.py
    mesh.py
    materials.py
    reduction.py
    simplex_lagrange.py
  jax/
    jax_energy_3d.py
  jax_petsc/
    reordered_element_assembler.py
    solver.py
    multigrid.py
    solve_slope_stability_3d_dof.py
data/meshes/SlopeStability3D/
  hetero_ssr/
    SSR_hetero_ada_L1.msh
    SSR_hetero_ada_L2.msh
    SSR_hetero_ada_L3.msh
    SSR_hetero_ada_L4.msh
    SSR_hetero_ada_L5.msh
    definition.py
    hetero_ssr_L1_p1_same_mesh.h5
    hetero_ssr_L1_p2_same_mesh.h5
    hetero_ssr_L1_p4_same_mesh.h5
    ...
```

### Core modeling choice

**Do not port the hand-coded 3D consistent tangent first.**

The target repo is already organized around **scalar element energies** and JAX autodiff. The cleanest 3D transfer is:

1. port the source repo’s **3D Mohr–Coulomb scalar potential**,
2. keep the source repo’s **Davis reduction**,
3. compute
   - local residual with `jax.grad(element_energy_3d)`,
   - local tangent with `jax.hessian(element_energy_3d)`,
4. assemble globally with the target repo’s reordered PETSc assembler scaffold.

That gets you a 3D plasticity path that is faithful to the benchmark and natural inside the target architecture.

### Most important implementation decision

For 3D, especially `P4`, **do not build the hot path around a dense stored `elem_B` tensor unless you need it for debugging**.

Use the hyperelasticity pattern instead:

- store `dphix`, `dphiy`, `dphiz`, `quad_weight`,
- form the 6 strain components inside the JAX kernel,
- differentiate the scalar element energy.

This is much more memory-friendly for tetrahedral `P4`.

### Recommended delivery order

1. import one source mesh (`SSR_hetero_ada_L1.msh`),
2. generate same-mesh `P1` and `P2` assets,
3. port the 3D scalar potential and Davis reduction,
4. build a single-rank `P2` solve with AMG/HYPRE first,
5. add the reordered PETSc assembler,
6. add same-mesh `P2 -> P1` PMG,
7. only then add `P4`,
8. only later consider h-tails across distinct meshes.

---

## 2. What to read in the two repositories

### Source repo: what matters

Read these first:

```text
benchmarks/run_3D_hetero_SSR_capture/case.toml
meshes/3d_hetero_ssr/definition.py
meshes/3d_hetero_ssr/*.msh
src/slope_stability/constitutive/problem.py
src/slope_stability/constitutive/reduction.py
src/slope_stability/mesh/materials.py
src/slope_stability/fem/assembly.py
src/slope_stability/fem/basis.py
src/slope_stability/fem/quadrature.py
src/slope_stability/core/simplex_lagrange.py
src/slope_stability/core/elements.py
```

What those files tell you:

- the heterogeneous 3D benchmark uses **material regions** and **physical-group boundary labels**,
- the benchmark case is built around **SSR / strength reduction**,
- the 3D constitutive model already has
  - a **scalar potential**,
  - a **stress update**,
  - a **hand-coded tangent**,
- same-mesh `P1`, `P2`, and `P4` tetrahedral logic already exists in the source repo,
- the source benchmark is the right reference for
  - material tables,
  - boundary labeling,
  - gravity direction,
  - quadrature order,
  - Voigt ordering,
  - and the Davis reduction formulas.

### Target repo: what matters

Read these first:

```text
src/problems/slope_stability/jax/jax_energy.py
src/problems/slope_stability/support/mesh.py
src/problems/slope_stability/support/reduction.py
src/problems/slope_stability/jax_petsc/reordered_element_assembler.py
src/problems/slope_stability/jax_petsc/solver.py
src/problems/slope_stability/jax_petsc/multigrid.py
src/problems/hyperelasticity/jax_petsc/reordered_element_assembler.py
src/problems/hyperelasticity/jax_petsc/parallel_hessian_dof.py
src/core/petsc/reordered_element_base.py
src/core/petsc/jax_tools/parallel_assembler.py
docs/problems/Plasticity.md
docs/results/Plasticity.md
```

What those files tell you:

- current maintained plasticity is **2D plane strain**,
- the repo already has the right **JAX + PETSc assembly architecture**,
- the 3D hyperelasticity path is the best template for a 3D vector-valued finite-element kernel,
- the current PMG implementation already knows how to do:
  - same-mesh `P2/P1`,
  - same-mesh `P4/P2/P1`,
  - rank-local loading,
  - owned-row transfer builds,
  - overlap-domain assembly,
- the maintained 2D PMG path strongly suggests:
  - keep **reordered ownership**,
  - use **rank-local heavy tensors**,
  - prefer **owned-row transfer operators**,
  - start with **same-mesh hierarchies**,
  - and only then add deeper h-tails.

---

## 3. Benchmark facts you should preserve

### 3.1 Mesh family

The source heterogeneous 3D mesh folder contains:

```text
SSR_hetero_ada_L1.msh
SSR_hetero_ada_L2.msh
SSR_hetero_ada_L3.msh
SSR_hetero_ada_L4.msh
SSR_hetero_ada_L5.msh
SSR_hetero_uni.msh
definition.py
```

Treat these as **distinct benchmark meshes**, not automatically as a nested multigrid hierarchy.

### 3.2 Material regions

The source `definition.py` exposes four materials:

| material | `c0` | `phi` | `psi` | `E` | `nu` | `gamma` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cover layer | 15 | 30 | 0 | 10000 | 0.33 | 19 |
| general foundation | 15 | 38 | 0 | 50000 | 0.30 | 22 |
| weak foundation | 10 | 35 | 0 | 50000 | 0.30 | 21 |
| slope mass | 18 | 32 | 0 | 20000 | 0.33 | 20 |

For the JAX port, keep the **base material arrays** at quadrature points:

- `c0_q`
- `phi_q` (radians)
- `psi_q` (radians)
- `shear_q`
- `bulk_q`
- `lame_q`
- `gamma_q`

Then derive reduced arrays from `lambda`.

### 3.3 Boundary labeling

The source benchmark’s boundary labeling is component-wise:

- x-Dirichlet labels: `[1, 2]`
- y-Dirichlet labels: `[5]`
- z-Dirichlet labels: `[3, 4]`

That means the 3D case is **component-constrained**, not “all components fixed on one surface”.

### 3.4 Gravity axis

A subtle but important source-benchmark detail:

- the source 3D body force is assembled as **`[0, -gamma, 0]`**,

so the “vertical” direction in the benchmark is the **second coordinate**, not the third.

**Do not silently change this in the port.**

If you want to rename axes internally, do it once and document it. Do not half-port the convention.

### 3.5 The `lambda` parameter

You said you want to keep `lambda` so the plasticity difficulty can be tuned. The right design is:

- store **unreduced** base materials in the mesh/problem asset,
- at solve setup, compute reduced
  - `c_bar_q`
  - `sin_phi_q`
- from `lambda_target`.

That keeps the benchmark material definition intact and lets you sweep `lambda` without writing new mesh snapshots.

---

## 4. Recommended architecture for the port

## 4.1 New problem package

Do **not** force the 3D heterogeneous benchmark into the existing 2D `src/problems/slope_stability/` subtree.

Keep the 2D maintained path stable and create a new problem family:

```text
src/problems/slope_stability_3d/
```

That avoids mixing:

- 2D plane strain,
- 3D tetrahedral vector FE,
- homogeneous and heterogeneous material logic,
- and distinct multigrid assumptions.

## 4.2 Internal data contract

Create a 3D problem-data object that mirrors the target repo’s current style but is explicit about 3D tensors and heterogeneity.

### Code snapshot: 3D case data

```python
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class SlopeStability3DCaseData:
    case_name: str
    mesh_name: str
    degree: int

    nodes: np.ndarray          # (n_nodes, 3)
    elems_scalar: np.ndarray   # (n_elem, n_p)
    elems: np.ndarray          # (n_elem, 3 * n_p) vector-expanded dof connectivity
    surf: np.ndarray           # degree-aware boundary-face connectivity
    q_mask: np.ndarray         # (3 * n_nodes,) bool; True means fixed DOF
    freedofs: np.ndarray       # free DOF ids in the full vector layout

    dphix: np.ndarray          # (n_elem, n_q, n_p)
    dphiy: np.ndarray          # (n_elem, n_q, n_p)
    dphiz: np.ndarray          # (n_elem, n_q, n_p)
    quad_weight: np.ndarray    # (n_elem, n_q)

    force: np.ndarray          # (3 * n_nodes,)
    u_0: np.ndarray            # (3 * n_nodes,)

    material_id: np.ndarray    # (n_elem,)
    c0_q: np.ndarray           # (n_elem, n_q)
    phi_q: np.ndarray          # (n_elem, n_q), radians
    psi_q: np.ndarray          # (n_elem, n_q), radians
    shear_q: np.ndarray        # (n_elem, n_q)
    bulk_q: np.ndarray         # (n_elem, n_q)
    lame_q: np.ndarray         # (n_elem, n_q)
    gamma_q: np.ndarray        # (n_elem, n_q)

    # Keep a placeholder history field only if you want interface symmetry
    # with the current 2D path or future hardening extensions.
    eps_p_old: np.ndarray      # (n_elem, n_q, 6)

    adjacency: sp.coo_matrix

    davis_type: str = "B"
    lambda_target_default: float = 1.0
    gravity_axis: int = 1
```

### Why this contract works

- it is natural for the reordered PETSc assembler,
- it is natural for JAX `vmap`,
- it keeps heterogeneity at **quadrature-point resolution**,
- it does not force the solver to know about Gmsh physical groups,
- it allows a one-time `.msh -> .h5` import path,
- and it stays close to how the target repo already handles heavy FE data.

---

## 5. How to bring the source meshes into the target repo

There are two sensible ways to do this.

## 5.1 Recommended workflow: copy the source meshes, then import once into HDF5

The best workflow for performance is:

1. download `.msh` files from the source repo,
2. import them once into target-repo HDF5 snapshots,
3. run the solver from those HDF5 snapshots,
4. use rank-local loading for large runs.

### Shell commands: download the meshes

#### Option A: clone the source repo and copy the folder

```bash
git clone https://github.com/Beremi/slope_stability_petsc4py.git /tmp/slope_stability_petsc4py

mkdir -p data/meshes/SlopeStability3D/hetero_ssr
cp /tmp/slope_stability_petsc4py/meshes/3d_hetero_ssr/*.msh data/meshes/SlopeStability3D/hetero_ssr/
cp /tmp/slope_stability_petsc4py/meshes/3d_hetero_ssr/definition.py data/meshes/SlopeStability3D/hetero_ssr/
```

#### Option B: download directly from GitHub raw

```bash
mkdir -p data/meshes/SlopeStability3D/hetero_ssr

for f in \
  SSR_hetero_ada_L1.msh \
  SSR_hetero_ada_L2.msh \
  SSR_hetero_ada_L3.msh \
  SSR_hetero_ada_L4.msh \
  SSR_hetero_ada_L5.msh \
  SSR_hetero_uni.msh \
  definition.py
do
  curl -L \
    -o "data/meshes/SlopeStability3D/hetero_ssr/${f}" \
    "https://raw.githubusercontent.com/Beremi/slope_stability_petsc4py/main/meshes/3d_hetero_ssr/${f}"
done
```

## 5.2 Why not read `.msh` on every solve?

Because the target repo’s maintained large-scale path already favors:

- HDF5 snapshots,
- rank-local loading,
- owned-row transfer construction,
- and avoiding repeated heavy preprocessing.

For the 3D port, do the same:

- `.msh` is the **source asset**,
- `.h5` is the **runtime asset**.

---

## 6. Same-mesh `P1` / `P2` / `P4`: how to build them correctly

This is where most ports go wrong.

### Key recommendation

Treat the source `.msh` files as **macro tetrahedral meshes** and generate the FE spaces yourself, just like the target repo already does for same-mesh 2D.

That gives you:

- one macro mesh,
- degree `1`, `2`, or `4`,
- consistent node ordering,
- easy p-transfer operators,
- and no need to maintain separate quadratic or quartic `.msh` files.

### Why this is the right approach here

The source repo already separates:

- mesh definition / material regions / boundary groups,
- and FE degree (`P1`, `P2`, `P4`).

So the clean transfer is:

1. import the macro tetra mesh and its physical groups,
2. generate same-mesh `P1`, `P2`, or `P4` nodal spaces,
3. assemble geometry-dependent FE operators for that degree.

## 6.1 Port the source simplex helper

Port this file essentially as-is into the new 3D support layer:

```text
src/slope_stability/core/simplex_lagrange.py
```

That helper gives you:

- tetrahedral node tuples for arbitrary order,
- source-consistent tetra reference nodes,
- source-consistent P4 basis evaluation.

That is exactly what you want for:

- same-mesh node generation,
- face lifting,
- prolongation/restriction assembly,
- and source-parity basis ordering.

## 6.2 Generate same-mesh tetra connectivity

### Code snapshot: generate same-mesh tetra nodes/connectivity

```python
import numpy as np
from .simplex_lagrange import tetra_reference_nodes


COORD_DECIMALS = 12


def build_same_mesh_tetra_connectivity(
    macro_nodes: np.ndarray,      # (n_macro_nodes, 3)
    macro_tets: np.ndarray,       # (n_elem, 4)
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    degree = int(degree)
    if degree == 1:
        return np.asarray(macro_nodes, dtype=np.float64), np.asarray(macro_tets, dtype=np.int64)

    ref = tetra_reference_nodes(degree).T   # (n_p, 3), xi=(xi1,xi2,xi3)
    node_map: dict[tuple[float, float, float], int] = {}
    nodes_out: list[np.ndarray] = []
    elems_out = np.empty((macro_tets.shape[0], ref.shape[0]), dtype=np.int64)

    for e, tet in enumerate(np.asarray(macro_tets, dtype=np.int64)):
        X = np.asarray(macro_nodes[tet], dtype=np.float64)  # (4, 3)
        for a, xi in enumerate(ref):
            l0 = 1.0 - xi[0] - xi[1] - xi[2]
            x = l0 * X[0] + xi[0] * X[1] + xi[1] * X[2] + xi[2] * X[3]
            key = tuple(np.round(x, COORD_DECIMALS))
            idx = node_map.get(key)
            if idx is None:
                idx = len(nodes_out)
                node_map[key] = idx
                nodes_out.append(x)
            elems_out[e, a] = idx

    return np.asarray(nodes_out, dtype=np.float64), elems_out
```

### Notes

- This is the 3D analogue of the target repo’s same-mesh 2D generation.
- Coordinate rounding is fine here because same-mesh nodes are generated from the same macro tetra vertices.
- Use the **same ordering helper everywhere**:
  - case generation,
  - assembly,
  - p-transfer operators.

## 6.3 Expand scalar connectivity to vector DOF connectivity

### Code snapshot

```python
def expand_tetra_connectivity_to_dofs(elems_scalar: np.ndarray) -> np.ndarray:
    elems_scalar = np.asarray(elems_scalar, dtype=np.int64)
    n_elem, n_p = elems_scalar.shape
    elems = np.empty((n_elem, 3 * n_p), dtype=np.int64)
    elems[:, 0::3] = 3 * elems_scalar
    elems[:, 1::3] = 3 * elems_scalar + 1
    elems[:, 2::3] = 3 * elems_scalar + 2
    return elems
```

This is the layout expected by the 3D reordered PETSc assembler.

## 6.4 Boundary lifting for `P2` and `P4`

Do **not** constrain only the macro corner nodes.

You must lift boundary faces to the chosen polynomial degree.

### Algorithm

For each macro boundary triangle and each constrained component family:

1. generate the degree-aware face nodes,
2. map them to global FE node ids,
3. mark the corresponding vector components as fixed.

This is essential for `P2` and `P4`, because many boundary-constrained DOFs are **not corner vertices**.

### Practical recommendation

Use the same simplex-ordering machinery for boundary faces too, so the face-node ordering matches the volume node ordering. If you already have a degree-aware `surf` array, deriving the component-constrained full-DOF mask becomes straightforward.

---

## 7. Quadrature and geometric operators

For correctness, use the source repo’s 3D tetrahedral quadrature and basis definitions for the first port.

That means porting/adapting:

```text
src/slope_stability/fem/quadrature.py
src/slope_stability/fem/basis.py
src/slope_stability/fem/assembly.py
```

You do **not** need to keep the exact global sparse `B` matrix path in the target port, but you do want the same local FE geometry logic.

## 7.1 Recommended stored local data for the JAX kernel

For 3D plasticity, store:

- `dphix` : `(n_elem, n_q, n_p)`
- `dphiy` : `(n_elem, n_q, n_p)`
- `dphiz` : `(n_elem, n_q, n_p)`
- `quad_weight` : `(n_elem, n_q)`

and **not** a dense `elem_B` in the hot path.

### Why

For tetrahedral `P4`:

- `n_p = 35`,
- vector DOFs per element = `3 * 35 = 105`,
- a dense stored `B` at every quadrature point is much heavier than needed.

The hyperelastic 3D path in the target repo already demonstrates the better pattern:
store shape gradients, not the expanded operator.

## 7.2 Assemble `dphix`, `dphiy`, `dphiz`, `quad_weight`

### Code snapshot: output layout to target

```python
@dataclass(frozen=True)
class LocalTetOps:
    dphix: np.ndarray       # (n_elem, n_q, n_p)
    dphiy: np.ndarray       # (n_elem, n_q, n_p)
    dphiz: np.ndarray       # (n_elem, n_q, n_p)
    quad_weight: np.ndarray # (n_elem, n_q)
```

The implementation can be adapted from the source repo’s `_assemble_3d`, but reshape outputs into element-major tensors instead of flattening everything into one global integration-point axis.

### Implementation tip

Port the source formulas first, then refactor. The order I recommend is:

1. port source local basis + quadrature,
2. compute Jacobians, inverse Jacobians, gradients, and weights,
3. compare against the source outputs on a tiny mesh,
4. only then refactor storage layout.

---

## 8. Material handling for heterogeneous 3D

The current target 2D path is homogeneous and stores:

- one `E`,
- one `nu`,
- one `c0`,
- one `phi_deg`.

For the heterogeneous 3D port, you need per-quadrature material arrays.

## 8.1 Keep base arrays unreduced

Store the base arrays in the HDF5 case snapshot:

```python
c0_q, phi_q, psi_q, shear_q, bulk_q, lame_q, gamma_q
```

all shaped `(n_elem, n_q)`.

Then reduce **at solve setup** using the runtime `lambda_target`.

This avoids regenerating assets for every `lambda`.

## 8.2 Recommended support function

### Code snapshot: array-valued Davis-B reduction

```python
import jax.numpy as jnp


def davis_b_reduction_qp(
    c0_q: jnp.ndarray,
    phi_q: jnp.ndarray,   # radians
    psi_q: jnp.ndarray,   # radians
    lam: float,
    tiny: float = 1.0e-15,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    lam = jnp.asarray(lam, dtype=jnp.float64)
    c01 = c0_q / lam
    phi1 = jnp.arctan(jnp.tan(phi_q) / lam)
    psi1 = jnp.arctan(jnp.tan(psi_q) / lam)

    denom = 1.0 - jnp.sin(phi1) * jnp.sin(psi1)
    beta = (jnp.cos(phi1) * jnp.cos(psi1)) / jnp.maximum(tiny, denom)

    c0_lambda = beta * c01
    phi_lambda = jnp.arctan(beta * jnp.tan(phi1))

    c_bar_q = 2.0 * c0_lambda * jnp.cos(phi_lambda)
    sin_phi_q = jnp.sin(phi_lambda)
    return c_bar_q, sin_phi_q
```

### Why return `c_bar_q` and `sin_phi_q`?

Because that is exactly what the source 3D constitutive and potential routines consume.

## 8.3 Where to apply the reduction

There are two reasonable choices.

### Simpler first version

At solve setup:

1. load base `c0_q`, `phi_q`, `psi_q`,
2. compute reduced arrays,
3. store reduced arrays in `params`,
4. JIT kernels consume reduced arrays only.

This is the simplest if `lambda_target` is fixed for one solve.

### Continuation-ready version

Keep base arrays in `params`, and regenerate reduced arrays when `lambda_target` changes.

Do this only if you are implementing a within-process continuation loop.

For the first benchmark port, the **simpler first version is enough**.

---

## 9. The 3D JAX constitutive kernel: what to port and what not to port

## 9.1 Port the scalar potential, not the hand tangent

The source repo exposes both:

- `constitutive_problem_3D(...)`
- `potential_3D(...)`

The target repo is energy-based. So the direct match is:

- port `potential_3D(...)`,
- differentiate it with JAX.

That is the central transfer idea.

## 9.2 Important warning on Voigt/shear convention

Do **not** “clean up” the source convention during the first port.

Mirror the source ordering and local FE formulas exactly.

In particular:

- keep the 6-strain ordering consistent everywhere,
- keep the same local row ordering in the FE operator,
- keep the same branch formulas in the potential,
- keep the same gravity axis,
- keep the same Davis reduction interpretation.

A 3D plasticity port can look “mathematically nicer” and still fail to match the source benchmark.

## 9.3 Recommended 3D local strain helper

Use the hyperelasticity-style gradient contraction instead of a dense `B`.

### Code snapshot: 6-strain construction from local gradients

```python
import jax.numpy as jnp


def strain6_from_local_gradients(
    u_elem: jnp.ndarray,    # (3 * n_p,)
    dphix_q: jnp.ndarray,   # (n_p,)
    dphiy_q: jnp.ndarray,   # (n_p,)
    dphiz_q: jnp.ndarray,   # (n_p,)
) -> jnp.ndarray:
    ux = u_elem[0::3]
    uy = u_elem[1::3]
    uz = u_elem[2::3]

    # Keep the source ordering convention.
    e_xx = jnp.dot(ux, dphix_q)
    e_yy = jnp.dot(uy, dphiy_q)
    e_zz = jnp.dot(uz, dphiz_q)

    g_xy = jnp.dot(ux, dphiy_q) + jnp.dot(uy, dphix_q)
    g_yz = jnp.dot(uy, dphiz_q) + jnp.dot(uz, dphiy_q)
    g_xz = jnp.dot(ux, dphiz_q) + jnp.dot(uz, dphix_q)

    return jnp.array([e_xx, e_yy, e_zz, g_xy, g_yz, g_xz], dtype=jnp.float64)
```

## 9.4 Recommended principal-value helper

The source potential computes the principal values from invariants. That is a good starting point because it avoids eigenvectors and stays close to the benchmark.

### Code snapshot: source-style principal-value helper

```python
import jax.numpy as jnp


def _safe_signed_denom(x: jnp.ndarray, tiny: float = 1.0e-15) -> jnp.ndarray:
    sign = jnp.where(x >= 0.0, 1.0, -1.0)
    return jnp.where(jnp.abs(x) < tiny, sign * tiny, x)


def principal_values_from_sym6(eps6: jnp.ndarray, tiny: float = 1.0e-15):
    e11, e22, e33, e12, e23, e13 = eps6

    I1 = e11 + e22 + e33
    I2 = e11 * e22 + e11 * e33 + e22 * e33 - e12 * e12 - e13 * e13 - e23 * e23
    I3 = (
        e11 * e22 * e33
        - e33 * e12 * e12
        - e22 * e13 * e13
        - e11 * e23 * e23
        + 2.0 * e12 * e13 * e23
    )

    Q = jnp.maximum(0.0, (I1 * I1 - 3.0 * I2) / 9.0)
    R = (-2.0 * I1**3 + 9.0 * I1 * I2 - 27.0 * I3) / 54.0

    theta0 = jnp.where(
        Q > 0.0,
        R / jnp.maximum(tiny, jnp.sqrt(Q**3)),
        1.0,
    )
    theta = jnp.where(
        Q > 0.0,
        jnp.arccos(jnp.clip(theta0, -1.0, 1.0)) / 3.0,
        0.0,
    )

    sqrtQ = jnp.sqrt(Q)
    e1 = -2.0 * sqrtQ * jnp.cos(theta + 2.0 * jnp.pi / 3.0) + I1 / 3.0
    e2 = -2.0 * sqrtQ * jnp.cos(theta - 2.0 * jnp.pi / 3.0) + I1 / 3.0
    e3 = -2.0 * sqrtQ * jnp.cos(theta) + I1 / 3.0
    return e1, e2, e3, I1
```

### Note

If you later want to compare this with `jnp.linalg.eigvalsh` on CPU, do it as a profiling experiment, not in the first parity port.

## 9.5 Port the source `potential_3D`

### Code snapshot: 3D Mohr–Coulomb potential density in JAX

```python
import jax.numpy as jnp


def mc_potential_density_3d(
    eps6: jnp.ndarray,
    c_bar: jnp.ndarray,
    sin_phi: jnp.ndarray,
    shear: jnp.ndarray,
    bulk: jnp.ndarray,
    lame: jnp.ndarray,
    tiny: float = 1.0e-15,
) -> jnp.ndarray:
    e1, e2, e3, I1 = principal_values_from_sym6(eps6, tiny=tiny)

    f_tr = (
        2.0 * shear * ((1.0 + sin_phi) * e1 - (1.0 - sin_phi) * e3)
        + 2.0 * lame * sin_phi * I1
        - c_bar
    )

    gamma_sl = (e1 - e2) / jnp.maximum(tiny, 1.0 + sin_phi)
    gamma_sr = (e2 - e3) / jnp.maximum(tiny, 1.0 - sin_phi)
    gamma_la = (e1 + e2 - 2.0 * e3) / jnp.maximum(tiny, 3.0 - sin_phi)
    gamma_ra = (2.0 * e1 - e2 - e3) / jnp.maximum(tiny, 3.0 + sin_phi)

    denom_s = 4.0 * lame * sin_phi**2 + 4.0 * shear * (1.0 + sin_phi**2)
    denom_l = (
        4.0 * lame * sin_phi**2
        + shear * (1.0 + sin_phi) ** 2
        + 2.0 * shear * (1.0 - sin_phi) ** 2
    )
    denom_r = (
        4.0 * lame * sin_phi**2
        + 2.0 * shear * (1.0 + sin_phi) ** 2
        + shear * (1.0 - sin_phi) ** 2
    )
    denom_a = 4.0 * bulk * sin_phi**2

    lambda_s = f_tr / _safe_signed_denom(denom_s, tiny)
    lambda_l = (
        shear * ((1.0 + sin_phi) * (e1 + e2) - 2.0 * (1.0 - sin_phi) * e3)
        + 2.0 * lame * sin_phi * I1
        - c_bar
    ) / _safe_signed_denom(denom_l, tiny)
    lambda_r = (
        shear * (2.0 * (1.0 + sin_phi) * e1 - (1.0 - sin_phi) * (e2 + e3))
        + 2.0 * lame * sin_phi * I1
        - c_bar
    ) / _safe_signed_denom(denom_r, tiny)
    lambda_a = (2.0 * bulk * sin_phi * I1 - c_bar) / _safe_signed_denom(denom_a, tiny)

    psi_el = 0.5 * lame * I1**2 + shear * (e1**2 + e2**2 + e3**2)
    psi_s = psi_el - 0.5 * denom_s * lambda_s**2
    psi_l = (
        0.5 * lame * I1**2
        + shear * (e3**2 + 0.5 * (e1 + e2) ** 2)
        - 0.5 * denom_l * lambda_l**2
    )
    psi_r = (
        0.5 * lame * I1**2
        + shear * (e1**2 + 0.5 * (e2 + e3) ** 2)
        - 0.5 * denom_r * lambda_r**2
    )
    psi_a = 0.5 * bulk * I1**2 - 0.5 * denom_a * lambda_a**2

    test_el = f_tr <= 0.0
    test_s = (~test_el) & (lambda_s <= jnp.minimum(gamma_sl, gamma_sr))
    test_l = (~(test_el | test_s)) & (gamma_sl < gamma_sr) & (lambda_l >= gamma_sl) & (lambda_l <= gamma_la)
    test_r = (~(test_el | test_s | test_l)) & (gamma_sl > gamma_sr) & (lambda_r >= gamma_sr) & (lambda_r <= gamma_ra)

    return jnp.where(
        test_el,
        psi_el,
        jnp.where(
            test_s,
            psi_s,
            jnp.where(
                test_l,
                psi_l,
                jnp.where(test_r, psi_r, psi_a),
            ),
        ),
    )
```

### Why this is the right first kernel

- it mirrors the source benchmark,
- it stays scalar and autodiff-friendly,
- it fits the target repo’s design,
- and it lets you get the tangent for free with `jax.hessian`.

## 9.6 Element energy

### Code snapshot: 3D element energy from the scalar potential

```python
import jax
import jax.numpy as jnp


def element_energy_3d(
    u_elem: jnp.ndarray,        # (3 * n_p,)
    dphix_e: jnp.ndarray,       # (n_q, n_p)
    dphiy_e: jnp.ndarray,       # (n_q, n_p)
    dphiz_e: jnp.ndarray,       # (n_q, n_p)
    quad_weight_e: jnp.ndarray, # (n_q,)
    c_bar_e: jnp.ndarray,       # (n_q,)
    sin_phi_e: jnp.ndarray,     # (n_q,)
    shear_e: jnp.ndarray,       # (n_q,)
    bulk_e: jnp.ndarray,        # (n_q,)
    lame_e: jnp.ndarray,        # (n_q,)
) -> jnp.ndarray:
    eps_q = jax.vmap(
        strain6_from_local_gradients,
        in_axes=(None, 0, 0, 0),
    )(u_elem, dphix_e, dphiy_e, dphiz_e)

    psi_q = jax.vmap(
        mc_potential_density_3d,
        in_axes=(0, 0, 0, 0, 0, 0),
    )(eps_q, c_bar_e, sin_phi_e, shear_e, bulk_e, lame_e)

    return jnp.sum(quad_weight_e * psi_q)
```

Then:

```python
element_residual_3d = jax.grad(element_energy_3d, argnums=0)
element_hessian_3d = jax.hessian(element_energy_3d, argnums=0)
```

---

## 10. Efficiency guidance for JAX in 3D

This matters a lot, especially once `P4` is enabled.

## 10.1 Use `float64`

The target repo already does this. Keep it.

```python
from jax import config
config.update("jax_enable_x64", True)
```

## 10.2 Keep shapes static

Compile one kernel per `(degree, n_q, n_p)` family.

That means:

- `P1` compiles once,
- `P2` compiles once,
- `P4` compiles once.

Do not feed ragged element layouts.

## 10.3 Store gradients, not `B`

For `P4`, this is the difference between a manageable and a painful memory footprint.

## 10.4 Batch over elements, but chunk for `P4`

### Recommended policy

- `P1`: full local batch is fine,
- `P2`: often full local batch is still fine,
- `P4`: use chunked local element batches.

A safe first implementation is:

```python
for start in range(0, n_local_elem, chunk_size):
    stop = min(start + chunk_size, n_local_elem)
    H_chunk = elem_hess_batch(v_local, start, stop)
    scatter_chunk(H_chunk)
```

### Why chunk

A tetrahedral `P4` element has

- `35` scalar shape functions,
- `105` vector DOFs,
- one dense local Hessian of size `105 x 105`.

Even before global assembly, that is a significant local tensor.

## 10.5 Warm up once per degree

Do one JIT warmup per active degree and local data shape during setup, before timing the real solve.

## 10.6 Avoid re-JIT on `lambda` unless you really need in-process continuation

The simplest high-performance design is:

1. load base materials,
2. compute reduced arrays once for the chosen `lambda`,
3. close over those reduced arrays in the JAX kernels.

If you later implement within-process continuation on `lambda`, you can switch to a runtime scalar argument or rebuild reduced arrays between solves.

## 10.7 Keep the top-level element kernel branch-free at the Python level

Use `jnp.where` or `jax.lax.cond`, not Python `if`, for constitutive branching.

---

## 11. PETSc assembler integration: use the reordered overlap-domain scaffold

The clean 3D template is the target repo’s hyperelasticity reordered assembler:

```text
src/problems/hyperelasticity/jax_petsc/reordered_element_assembler.py
```

Port the 3D plasticity path the same way.

## 11.1 New assembler class

### Code snapshot: 3D plasticity reordered assembler skeleton

```python
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.core.petsc.reordered_element_base import ReorderedElementAssemblerBase
from src.problems.slope_stability_3d.jax.jax_energy_3d import element_energy_3d


class MC3DReorderedElementAssembler(ReorderedElementAssemblerBase):
    block_size = 3
    coordinate_key = "nodes"
    dirichlet_key = "u_0"
    local_elem_data_keys = (
        "dphix",
        "dphiy",
        "dphiz",
        "quad_weight",
        "c_bar_q",
        "sin_phi_q",
        "shear_q",
        "bulk_q",
        "lame_q",
    )
    near_nullspace_key = "elastic_kernel"

    def _make_local_element_kernels(self):
        elems = jnp.asarray(self.local_data.elems_local_np, dtype=jnp.int32)

        dphix = jnp.asarray(self.local_data.local_elem_data["dphix"], dtype=jnp.float64)
        dphiy = jnp.asarray(self.local_data.local_elem_data["dphiy"], dtype=jnp.float64)
        dphiz = jnp.asarray(self.local_data.local_elem_data["dphiz"], dtype=jnp.float64)
        quad_weight = jnp.asarray(self.local_data.local_elem_data["quad_weight"], dtype=jnp.float64)

        c_bar_q = jnp.asarray(self.local_data.local_elem_data["c_bar_q"], dtype=jnp.float64)
        sin_phi_q = jnp.asarray(self.local_data.local_elem_data["sin_phi_q"], dtype=jnp.float64)
        shear_q = jnp.asarray(self.local_data.local_elem_data["shear_q"], dtype=jnp.float64)
        bulk_q = jnp.asarray(self.local_data.local_elem_data["bulk_q"], dtype=jnp.float64)
        lame_q = jnp.asarray(self.local_data.local_elem_data["lame_q"], dtype=jnp.float64)

        energy_weights = jnp.asarray(self.local_data.energy_weights, dtype=jnp.float64)

        def elem_energy_local(v_e, dphix_e, dphiy_e, dphiz_e, w_e, cbar_e, sphi_e, mu_e, K_e, lam_e):
            return element_energy_3d(
                v_e,
                dphix_e,
                dphiy_e,
                dphiz_e,
                w_e,
                cbar_e,
                sphi_e,
                mu_e,
                K_e,
                lam_e,
            )

        vmapped_hess = jax.vmap(
            jax.hessian(elem_energy_local),
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )

        def local_full_energy(v_local):
            v_e = v_local[elems]
            e = jax.vmap(elem_energy_local, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
                v_e, dphix, dphiy, dphiz, quad_weight,
                c_bar_q, sin_phi_q, shear_q, bulk_q, lame_q
            )
            return jnp.sum(e)

        grad_local = jax.grad(local_full_energy)

        @jax.jit
        def energy_fn(v_local):
            v_e = v_local[elems]
            e = jax.vmap(elem_energy_local, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
                v_e, dphix, dphiy, dphiz, quad_weight,
                c_bar_q, sin_phi_q, shear_q, bulk_q, lame_q
            )
            return jnp.sum(e * energy_weights)

        @jax.jit
        def local_grad_fn(v_local):
            return grad_local(v_local)

        @jax.jit
        def elem_hess_fn(v_local):
            v_e = v_local[elems]
            return vmapped_hess(
                v_e, dphix, dphiy, dphiz, quad_weight,
                c_bar_q, sin_phi_q, shear_q, bulk_q, lame_q
            )

        return energy_fn, local_grad_fn, elem_hess_fn, grad_local

    def _build_rhs_owned(self) -> np.ndarray:
        rhs = np.asarray(self.params["force"], dtype=np.float64)
        freedofs = np.asarray(self.params["freedofs"], dtype=np.int64)
        rhs_free = rhs[freedofs]
        rhs_reordered = rhs_free[self.layout.perm]
        return np.asarray(rhs_reordered[self.layout.lo : self.layout.hi], dtype=np.float64)
```

## 11.2 Why `block_size = 3` matters

It makes PETSc ownership and reorderings respect xyz blocks. That is important for:

- block-aware permutations,
- smoother behavior,
- near-nullspace handling.

## 11.3 Use the target repo’s overlap-domain machinery directly

Do not re-invent:

- local overlap extraction,
- owned-row COO insertion,
- overlap P2P exchange,
- reordered ownership ranges.

That infrastructure already exists in `src/core/petsc/reordered_element_base.py`.

---

## 12. Near-nullspace for 3D vector elasticity/plasticity

The 3D path should provide a 6-vector rigid-body near-nullspace:

- 3 translations,
- 3 rotations.

Even with Dirichlet boundaries, this is valuable for AMG-like coarse spaces and for consistency with the existing vector-valued PETSc setup.

### Code snapshot: 3D rigid-body modes

```python
import numpy as np


def build_near_nullspace_modes_3d(nodes: np.ndarray, freedofs: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    freedofs = np.asarray(freedofs, dtype=np.int64)

    n_nodes = nodes.shape[0]
    full = np.zeros((3 * n_nodes, 6), dtype=np.float64)

    center = np.mean(nodes, axis=0)
    x = nodes[:, 0] - center[0]
    y = nodes[:, 1] - center[1]
    z = nodes[:, 2] - center[2]

    # translations
    full[0::3, 0] = 1.0
    full[1::3, 1] = 1.0
    full[2::3, 2] = 1.0

    # rotations
    # about x: (0, -z, y)
    full[1::3, 3] = -z
    full[2::3, 3] =  y

    # about y: (z, 0, -x)
    full[0::3, 4] =  z
    full[2::3, 4] = -x

    # about z: (-y, x, 0)
    full[0::3, 5] = -y
    full[1::3, 5] =  x

    return full[freedofs, :]
```

Store this in `params["elastic_kernel"]`.

---

## 13. Solver integration

The 3D solver should look much closer to the current 2D PETSc plasticity solver than to a standalone script.

## 13.1 New CLI entry point

Create something like:

```text
src/problems/slope_stability_3d/jax_petsc/solve_slope_stability_3d_dof.py
```

Reuse the style of the current 2D solver, but add 3D-specific case loading and simpler first-stage MG expectations.

### Recommended CLI arguments

At minimum:

- `--mesh_name`
- `--elem_degree {1,2,4}`
- `--lambda-target`
- `--pc_type`
- `--ksp_type`
- `--ksp_rtol`
- `--ksp_max_it`
- `--problem_build_mode`
- `--mg_level_build_mode`
- `--mg_transfer_build_mode`
- `--distribution_strategy`
- `--element_reorder_mode`
- `--mg_strategy`
- `--mg_variant`

## 13.2 Build reduced materials in solver setup

### Code snapshot

```python
def apply_strength_reduction(params: dict[str, object], lambda_target: float) -> None:
    c0_q = np.asarray(params["c0_q"], dtype=np.float64)
    phi_q = np.asarray(params["phi_q"], dtype=np.float64)
    psi_q = np.asarray(params["psi_q"], dtype=np.float64)

    c_bar_q, sin_phi_q = reduction(
        c0_q,
        phi_q,
        psi_q,
        float(lambda_target),
        Davis_type=str(params.get("davis_type", "B")),
    )

    params["c_bar_q"] = np.asarray(c_bar_q, dtype=np.float64)
    params["sin_phi_q"] = np.asarray(sin_phi_q, dtype=np.float64)
```

The reduced arrays should be the tensors the assembler consumes.

## 13.3 Two-phase bring-up

### Phase A: correctness first

Use:

- `pc_type = hypre` or `gamg`,
- single rank or a few ranks,
- `P2`,
- `lambda = 1.0`,
- no PMG yet.

### Phase B: PMG integration

Once the local kernel, residual, tangent, load vector, and boundary conditions are correct, add the 3D PMG hierarchy.

---

## 14. PMG: how to adapt and how to use it

The target repo’s PMG machinery is currently problem-specific and 2D-specific, but the design is exactly the pattern you want.

## 14.1 What to reuse conceptually

Reuse these ideas from the existing PMG path:

- `MGLevelSpace`-style lightweight level metadata,
- same-mesh p-transfer operators,
- rank-local heavy-data loading,
- owned-row transfer assembly,
- reordered ownership-aware prolongation/restriction,
- overlap P2P for hot-path solves.

## 14.2 What not to assume

Do **not** assume that the source mesh files

```text
SSR_hetero_ada_L1.msh ... SSR_hetero_ada_L5.msh
```

form a nested geometric hierarchy.

They might be refinement levels, or adapted meshes, or just benchmark variants. Unless you explicitly verify node nesting and construct valid transfer operators, they should be treated as **separate benchmark meshes**, not a ready-made h-multigrid ladder.

## 14.3 First PMG target: same-mesh p-hierarchy on one macro mesh

That means:

- on `SSR_hetero_ada_L1.msh`, generate `P1`, `P2`, optionally `P4`,
- build PMG on the same macro mesh.

### Recommended first 3D PMG strategies

For a `P2` fine space:

```text
same_mesh_p2_p1
```

For a `P4` fine space:

```text
same_mesh_p4_p2_p1
```

### What to postpone

Postpone these until you have a verified nested h-hierarchy:

- `same_mesh_p2_p1_lminus1_p1`
- `same_mesh_p4_p1_lminus1_p1`
- `same_mesh_p4_p2_p1_lminus1_p1`
- deep custom `P1` tails across multiple mesh levels

Those are powerful in the current 2D maintained path, but they rely on hierarchy structure you do not automatically have in the imported 3D source meshes.

## 14.4 Transfer operators for same-mesh p-multigrid

Because all spaces are nodal Lagrange spaces on the same macro mesh, the prolongation from degree `p_c` to `p_f` is just **nodal interpolation**.

### Recommended construction

Use the same tetrahedral basis ordering helper that you used for connectivity generation.

At the element level:

1. get fine reference nodes,
2. evaluate coarse basis at those fine nodes,
3. build the scalar prolongation block,
4. Kronecker it with `I_3` for vector DOFs,
5. assemble globally in reordered free-DOF space.

### Code snapshot: scalar local p-prolongation idea

```python
def build_local_p_prolongation_scalar(
    p_coarse: int,
    p_fine: int,
) -> np.ndarray:
    x_fine = tetra_reference_nodes(p_fine)               # (3, n_pf)
    phi_c, _, _, _ = evaluate_tetra_lagrange_basis(p_coarse, x_fine)
    # phi_c shape: (n_pc, n_pf)
    return phi_c.T   # (n_pf, n_pc)
```

Then:

```python
P_local_vec = np.kron(P_local_scalar, np.eye(3))
```

### Important consistency rule

Use the **same node ordering helper** for:

- `P1/P2/P4` connectivity,
- local basis evaluation,
- local p-transfer blocks.

If you mix ordering conventions, PMG will be broken even if every individual piece “looks right”.

## 14.5 New 3D PMG module

Create:

```text
src/problems/slope_stability_3d/jax_petsc/multigrid.py
```

and adapt the existing 2D `multigrid.py`.

### Required changes

- vector block size `2 -> 3`,
- 2D triangle assumptions -> 3D tetra assumptions,
- 2D case loaders -> 3D case loaders,
- 2D near-nullspace -> 3D six-mode rigid-body nullspace,
- triangle-based transfer/basis logic -> tetra-based logic.

## 14.6 Recommended first PMG runtime settings

For the first 3D `P2` benchmark path:

```bash
mpirun -n 8 python -m src.problems.slope_stability_3d.jax_petsc.solve_slope_stability_3d_dof \
  --mesh_name hetero_ssr_L1 \
  --elem_degree 2 \
  --lambda-target 1.0 \
  --ksp_type fgmres \
  --pc_type mg \
  --ksp_rtol 1e-2 \
  --ksp_max_it 15 \
  --problem_build_mode rank_local \
  --mg_level_build_mode rank_local \
  --mg_transfer_build_mode owned_rows \
  --distribution_strategy overlap_p2p \
  --element_reorder_mode block_xyz \
  --mg_strategy same_mesh_p2_p1 \
  --mg_variant explicit_pmg
```

### Why `block_xyz` first?

Because it works cleanly with rank-local builds without forcing a global adjacency in the first version.

You can add `block_metis` later if you either:

- keep/load a global adjacency,
- or precompute/store a permutation.

## 14.7 PMG development order

1. `P2 -> P1` same-mesh hierarchy,
2. `P4 -> P2 -> P1` same-mesh hierarchy,
3. optional custom mixed hierarchy,
4. optional h-tail only after a true nested 3D mesh ladder exists.

---

## 15. `P1`, `P2`, `P4`: what changes and what to expect

## 15.1 `P1`

Use `P1` for:

- debugging the import pipeline,
- boundary lifting tests,
- local constitutive parity tests,
- and the coarsest same-mesh PMG level.

### Characteristics

- nodes per tetra: `4`
- vector DOFs per element: `12`
- volume quadrature points: `1`

This is the cheapest degree to validate end-to-end.

## 15.2 `P2`

This should be the first real benchmark target.

### Why

- the source 3D heterogeneous benchmark case is already set up around `P2`,
- `P2` is much more practical than `P4` for a first 3D plasticity port,
- and `P2 -> P1` PMG is straightforward.

### Characteristics

- nodes per tetra: `10`
- vector DOFs per element: `30`
- volume quadrature points: `11`

That makes local Hessians completely manageable.

## 15.3 `P4`

Add `P4` only after `P2` is correct and PMG works.

### Characteristics

- nodes per tetra: `35`
- vector DOFs per element: `105`
- volume quadrature points: `24`

### Consequences

- local Hessian size is `105 x 105`,
- element-batch memory becomes a serious issue,
- full dense `elem_B` storage is unattractive,
- chunked Hessian assembly is strongly recommended,
- and top-level smoothing cost will be the first scalability bottleneck.

### Practical recommendation

For the first `P4` version:

- use same-mesh `P4 -> P2 -> P1`,
- use chunked local Hessians,
- and do not attempt a deep h-tail until the `P4` same-mesh path is stable.

---

## 16. Gravity/load assembly and heterogeneous body force

Because the source materials have different `gamma`, the body force is also heterogeneous.

That means the nodal force vector must be assembled from `gamma_q`, not one global scalar.

### Code snapshot: heterogeneous body force per quadrature point

```python
def body_force_q(gamma_q: np.ndarray) -> np.ndarray:
    # Preserve source axis convention: vertical is y.
    out = np.zeros((gamma_q.shape[0], 3), dtype=np.float64)
    out[:, 1] = -gamma_q
    return out
```

At the element level, use the same interpolation operator / quadrature data you already built for the FE space.

### Strong recommendation

Write one standalone regression test that compares:

- your new 3D assembled load vector,
- against a tiny source-reproduced case.

This catches the most common silent sign/axis mistakes.

---

## 17. HDF5 import path and rank-local loading

The target repo’s maintained large-scale path already shows the right pattern:

- one-time HDF5 asset generation,
- optional rank-local loading of the heavy element tensors,
- and reuse of those tensors across solves.

Do the same for 3D.

## 17.1 Import script

Create a script or module like:

```text
src/problems/slope_stability_3d/support/import_source_mesh.py
```

that:

1. reads the source `.msh`,
2. reads the source `definition.py`-equivalent metadata,
3. builds degree-aware same-mesh FE spaces,
4. assembles local FE operators,
5. expands materials to quadrature-point arrays,
6. assembles the body force,
7. builds `freedofs` and adjacency,
8. writes an HDF5 snapshot.

### Recommended CLI

```bash
python -m src.problems.slope_stability_3d.support.import_source_mesh \
  --mesh data/meshes/SlopeStability3D/hetero_ssr/SSR_hetero_ada_L1.msh \
  --case hetero_ssr_L1 \
  --degree 2 \
  --out data/meshes/SlopeStability3D/hetero_ssr/hetero_ssr_L1_p2_same_mesh.h5
```

## 17.2 Rank-local loading

For large 3D runs, add a 3D analogue of the current target function:

```text
load_same_mesh_case_hdf5_rank_local(...)
```

The idea is the same:

- keep light metadata replicated,
- load only local element tensors on each rank.

### Recommended distributed fields

For 3D plasticity, the heavy fields are:

- `dphix`
- `dphiy`
- `dphiz`
- `quad_weight`
- `c0_q`
- `phi_q`
- `psi_q`
- `shear_q`
- `bulk_q`
- `lame_q`
- `gamma_q`
- optional `eps_p_old`

---

## 18. Validation plan

Do not wait until the full MPI benchmark to test the port.

## 18.1 Unit tests

### A. Davis reduction parity

Given random arrays `(c0, phi, psi, lambda)`:

- compare the new 3D reduction helper against the source `reduction(...)`.

### B. Potential parity

Given random strains and materials:

- compare the new JAX `mc_potential_density_3d`
- against the source `potential_3D(...)`.

This is the single most important constitutive test.

### C. Local gradient/Hessian finite-difference sanity

For one element:

- compare `jax.grad` against centered finite differences,
- compare `jax.hessian` action against directional differences of the gradient.

### D. Elastic limit sanity

Set `c0` very large so the point stays elastic, and compare against linear elasticity.

### E. Boundary lifting sanity

Check that degree-aware constrained nodes on boundary faces are actually fixed for `P2` and `P4`.

### F. Body-force sign/axis sanity

Build a tiny mesh and verify that gravity acts in the source’s vertical direction (`y`).

### G. Same-mesh transfer sanity

For a smooth nodal field:

- prolong `P1 -> P2`,
- restrict back,
- verify exactness where expected.

## 18.2 Integration tests

### Test 1: single-rank `P1`

- import one mesh,
- solve one easy case,
- verify assembly and signs.

### Test 2: single-rank `P2`

- run `lambda = 1.0`,
- compare energy and displacement magnitude against a reference run.

### Test 3: multi-rank `P2` with AMG

- verify distributed assembly consistency before enabling PMG.

### Test 4: multi-rank `P2` with PMG

- `same_mesh_p2_p1`,
- overlap P2P,
- owned-row transfers.

### Test 5: `P4`

Only after the previous tests pass.

---

## 19. What to postpone on purpose

These are good ideas, but they should not block the first working port.

### 19.1 Full path-dependent plastic history evolution

The target 2D docs still emphasize endpoint states with `eps_p_old = 0` rather than a full history loop.

For the first 3D transfer, follow the source benchmark’s scalar-potential formulation and keep history variables as placeholders only.

### 19.2 Hand-coded 3D consistent tangent

You already get a tangent from JAX.

Port the analytic tangent only if later profiling proves it is worth the added complexity.

### 19.3 Deep h-tail PMG across the source `L1..L5` meshes

Do this only after you prove those meshes form a valid nested hierarchy and build correct transfer operators.

### 19.4 Fancy solver policies on day one

The target repo’s 2D maintained PMG path contains a lot of tuned policy logic. Reuse what is easy, but do not let every nonlinear/linear tuning knob block the 3D constitutive and mesh port.

---

## 20. A concrete implementation order that is likely to work

## Step 1 — vendor the source 3D helper logic

Port/adapt:

- `simplex_lagrange.py`
- tetra basis/quadrature
- Davis reduction
- `potential_3D`

into the new 3D support/JAX layer.

## Step 2 — build the mesh importer

Implement:

- macro mesh read,
- same-mesh `P1/P2/P4` connectivity generation,
- boundary lifting,
- material expansion,
- force assembly,
- adjacency build,
- HDF5 write.

## Step 3 — build the `P2` case path

Implement:

- `ensure_case_hdf5_3d(...)`
- `load_case_hdf5_3d(...)`
- `load_case_hdf5_rank_local_3d(...)`

## Step 4 — build the JAX local kernel

Implement:

- strain helper,
- principal-value helper,
- 3D potential,
- element energy,
- local grad/hessian.

## Step 5 — build the reordered PETSc assembler

Copy the hyperelasticity 3D pattern and swap in the plasticity kernel.

## Step 6 — solve with AMG/HYPRE first

Get:

- residual,
- tangent,
- gravity,
- constraints,
- and heterogeneity

correct before PMG.

## Step 7 — add same-mesh `P2 -> P1` PMG

Clone/adapt the PMG module.

## Step 8 — add `P4`

Only after `P2` is stable.

## Step 9 — optional continuation wrapper

If you want more source-like SSR behavior, wrap repeated solves over a `lambda` schedule.

---

## 21. Example command set once the port exists

## 21.1 Import one mesh

```bash
python -m src.problems.slope_stability_3d.support.import_source_mesh \
  --mesh data/meshes/SlopeStability3D/hetero_ssr/SSR_hetero_ada_L1.msh \
  --case hetero_ssr_L1 \
  --degree 1

python -m src.problems.slope_stability_3d.support.import_source_mesh \
  --mesh data/meshes/SlopeStability3D/hetero_ssr/SSR_hetero_ada_L1.msh \
  --case hetero_ssr_L1 \
  --degree 2
```

## 21.2 Correctness run with AMG

```bash
mpirun -n 4 python -m src.problems.slope_stability_3d.jax_petsc.solve_slope_stability_3d_dof \
  --mesh_name hetero_ssr_L1 \
  --elem_degree 2 \
  --lambda-target 1.0 \
  --ksp_type fgmres \
  --pc_type hypre \
  --ksp_rtol 1e-2 \
  --ksp_max_it 30 \
  --problem_build_mode rank_local \
  --distribution_strategy overlap_p2p \
  --element_reorder_mode block_xyz
```

## 21.3 First PMG run

```bash
mpirun -n 8 python -m src.problems.slope_stability_3d.jax_petsc.solve_slope_stability_3d_dof \
  --mesh_name hetero_ssr_L1 \
  --elem_degree 2 \
  --lambda-target 1.0 \
  --ksp_type fgmres \
  --pc_type mg \
  --ksp_rtol 1e-2 \
  --ksp_max_it 15 \
  --problem_build_mode rank_local \
  --mg_level_build_mode rank_local \
  --mg_transfer_build_mode owned_rows \
  --distribution_strategy overlap_p2p \
  --element_reorder_mode block_xyz \
  --mg_strategy same_mesh_p2_p1 \
  --mg_variant explicit_pmg
```

## 21.4 Difficulty sweep via `lambda`

```bash
for lam in 1.30 1.20 1.10 1.00 0.95; do
  mpirun -n 8 python -m src.problems.slope_stability_3d.jax_petsc.solve_slope_stability_3d_dof \
    --mesh_name hetero_ssr_L1 \
    --elem_degree 2 \
    --lambda-target "${lam}" \
    --ksp_type fgmres \
    --pc_type mg \
    --ksp_rtol 1e-2 \
    --ksp_max_it 15 \
    --problem_build_mode rank_local \
    --mg_level_build_mode rank_local \
    --mg_transfer_build_mode owned_rows \
    --distribution_strategy overlap_p2p \
    --element_reorder_mode block_xyz \
    --mg_strategy same_mesh_p2_p1 \
    --mg_variant explicit_pmg
done
```

---

## 22. Common failure modes and how to avoid them

## 22.1 Wrong gravity direction

Symptom:

- deformation looks rotated or physically nonsensical.

Cause:

- assuming gravity is along `z` instead of the source benchmark’s `y`.

Fix:

- preserve the source axis convention until everything matches.

## 22.2 Incomplete boundary lifting on `P2/P4`

Symptom:

- topological leaks at constrained boundaries,
- strange extra motion on higher-order runs.

Cause:

- only corner nodes were constrained.

Fix:

- lift face constraints to all degree-aware boundary nodes.

## 22.3 Inconsistent node ordering between connectivity, basis, and transfers

Symptom:

- PMG diverges immediately,
- same-mesh prolongation looks dense or wrong.

Cause:

- mixed ordering conventions.

Fix:

- use one tetrahedral ordering helper everywhere.

## 22.4 Huge memory blow-up for `P4`

Symptom:

- setup or Hessian build runs out of memory.

Cause:

- storing dense `elem_B`,
- or vmapping too many `P4` element Hessians at once.

Fix:

- store gradient tensors only,
- chunk the local Hessian work.

## 22.5 Trying to force a deep h-tail from the source adapted meshes

Symptom:

- invalid transfer operators,
- nonsense coarse corrections.

Cause:

- assuming `L1..L5` are nested.

Fix:

- use same-mesh p-hierarchies first.

## 22.6 “Correct” Mohr–Coulomb formulas that do not match the benchmark

Symptom:

- the solver works, but results do not match the source benchmark.

Cause:

- silently changing conventions, especially shear or branch formulas.

Fix:

- mirror the source benchmark exactly in v1.

---

## 23. Final recommendation

If you want this port to succeed with the least wasted effort, do it this way:

1. **import the source heterogeneous 3D meshes as macro tetra meshes**,
2. **generate same-mesh `P1/P2/P4` spaces yourself**,
3. **port the source `potential_3D` and Davis reduction into JAX**,
4. **differentiate the scalar element energy with JAX**,
5. **reuse the target repo’s reordered PETSc overlap assembler architecture**,
6. **start with `P2` + AMG/HYPRE**, then
7. **add same-mesh `P2 -> P1` PMG**, then
8. **add `P4`**, and only then
9. **consider deeper multilevel hierarchies**.

That path is the shortest route to a correct, maintainable 3D heterogeneous Mohr–Coulomb benchmark inside the JAX + PETSc repo.

---

## 24. Implementation checklist

Use this as the literal task list.

### Source parity
- [ ] port Davis reduction for arrays
- [ ] port `potential_3D`
- [ ] preserve source gravity axis
- [ ] preserve source material tables
- [ ] preserve source boundary component labels
- [ ] preserve source tetra node ordering

### Mesh/import
- [ ] import `.msh`
- [ ] create same-mesh `P1/P2/P4`
- [ ] lift boundary faces to high-order nodes
- [ ] build `freedofs`
- [ ] build adjacency
- [ ] assemble body force
- [ ] write HDF5 snapshots

### JAX kernel
- [ ] strain helper from `dphix/dphiy/dphiz`
- [ ] principal-value helper
- [ ] scalar 3D plastic potential
- [ ] element grad
- [ ] element Hessian
- [ ] chunked `P4` path

### PETSc integration
- [ ] 3D reordered assembler
- [ ] 3D near-nullspace
- [ ] 3D solver CLI
- [ ] rank-local heavy-data loading
- [ ] overlap P2P path

### PMG
- [ ] same-mesh `P2 -> P1`
- [ ] same-mesh `P4 -> P2 -> P1`
- [ ] owned-row transfer operators
- [ ] 3D `MGLevelSpace`
- [ ] 3D prolongation/restriction assembly

### Validation
- [ ] reduction parity test
- [ ] potential parity test
- [ ] one-element FD gradient/Hessian test
- [ ] gravity sign/axis test
- [ ] `P2` correctness run
- [ ] `P2` PMG run
- [ ] `P4` smoke test