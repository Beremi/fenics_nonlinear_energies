# HyperElasticity 3D — JAX + PETSc Implementation Prompt

## Current Repo Note

This prompt describes the original implementation plan. The codebase has since been refactored.

Current locations:

- CLI wrapper: `HyperElasticity3D_jax_petsc/solve_HE_dof.py`
- solver logic: `HyperElasticity3D_jax_petsc/solver.py`
- problem-specific assembler glue: `HyperElasticity3D_jax_petsc/parallel_hessian_dof.py`
- shared mesh / BC helpers: `HyperElasticity3D_petsc_support/`
- shared JAX assembler infrastructure: `tools_petsc4py/jax_tools/parallel_assembler.py`

For current solver-parameter behavior, especially the optional trust-region path, see
`TRUST_REGION_LINESEARCH_TUNING.md`.

## Goal

Create `HyperElasticity3D_jax_petsc/` — a MPI-parallel solver for 3D
hyperelasticity using the **generic reusable infrastructure** in
`tools_petsc4py/`.  Follow the same pattern as `pLaplace2D_jax_petsc/`:
thin problem-specific subclasses + a CLI solver script.

---

## Architecture (already implemented — just reuse)

The generic infrastructure now lives across `tools_petsc4py/` and
`tools_petsc4py/jax_tools/`:

| File                                   | Purpose                                                                                                                                                                    |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tools_petsc4py/dof_partition.py`      | `DOFPartition` — RCM reordering, PETSc block ownership, P2P ghost exchange, element weight assignment, `update_dirichlet()`                                                |
| `tools_petsc4py/jax_tools/parallel_assembler.py` | `DOFHessianAssemblerBase` (global coloring) and `LocalColoringAssemblerBase` (local coloring + vmap) plus generic problem-spec wrappers |
| `tools_petsc4py/minimizers.py`         | `newton()` — golden-section line search, compatible callbacks                                                                                                              |

The base assembler constructor already supports:
- `near_nullspace_vecs` — list of `(n_free,)` arrays in **original** free-DOF ordering; automatically reordered to PETSc space and set as `Mat.setNearNullSpace`
- `pc_options` — dict passed as `PETSc.Options` (e.g. Hypre BoomerAMG nodal coarsening)
- `ksp_max_it` — optional max KSP iterations
- `update_dirichlet(u_0_new)` — updates Dirichlet values in `v_template` for load stepping
- `f=None` — load vector defaults to zeros (HE has no body forces)

### Pattern to follow

Look at `pLaplace2D_jax_petsc/` for the exact pattern:

```
pLaplace2D_jax_petsc/
  dof_partition.py      # Thin backward-compat wrapper (~50 lines)
  parallel_hessian_dof.py  # Thin subclasses with _make_local_energy_fns() (~120 lines)
  mesh.py               # HDF5 mesh loader
  solve_pLaplace_dof.py # CLI solver script
```

The original plan was to create:

```
HyperElasticity3D_jax_petsc/
  __init__.py                # docstring
  parallel_hessian_dof.py    # HE-specific assembler subclasses
  mesh.py                    # HDF5 mesh loader (copy from HyperElasticity3D_jax/mesh.py)
  rotate_boundary.py         # Boundary rotation (copy from HyperElasticity3D_jax/)
  solve_HE_dof.py            # CLI solver with load stepping
```

Current refactored layout is:

```
HyperElasticity3D_jax_petsc/
  __init__.py
  parallel_hessian_dof.py
  solver.py
  solve_HE_dof.py

HyperElasticity3D_petsc_support/
  __init__.py
  mesh.py
  rotate_boundary.py
```

---

## Historical Implementation Sketch

The sections below describe the original implementation plan. Read the file
paths through the current layout above: the mesh and boundary helpers now live
in `HyperElasticity3D_petsc_support/`, and the shared assembler lives in
`tools_petsc4py/jax_tools/parallel_assembler.py`.

### 1. `HyperElasticity3D_jax_petsc/parallel_hessian_dof.py`

Thin subclasses providing the Neo-Hookean energy function.

**Key differences from pLaplace:**
- 3 DOFs per scalar node: `v[0::3]`, `v[1::3]`, `v[2::3]`
- `elem_data` has 4 keys: `"dvx"`, `"dvy"`, `"dvz"`, `"vol"`
- Material params `C1`, `D1` stored on `self` (like pLaplace stores `self._p`)
- No load vector (`f=None`)

```python
"""HyperElasticity 3D assemblers — thin subclasses of generic base."""

import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from tools_petsc4py.jax_tools.parallel_assembler import (
    DOFHessianAssemblerBase,
    LocalColoringAssemblerBase,
)


class _HyperElasticityMixin:
    """Mixin providing Neo-Hookean energy via _make_local_energy_fns()."""

    def _make_local_energy_fns(self):
        C1 = self._C1
        D1 = self._D1
        elems = jnp.array(self.part.elems_local_np, dtype=jnp.int32)
        dvx = jnp.array(self.part.local_elem_data["dvx"], dtype=jnp.float64)
        dvy = jnp.array(self.part.local_elem_data["dvy"], dtype=jnp.float64)
        dvz = jnp.array(self.part.local_elem_data["dvz"], dtype=jnp.float64)
        vol = jnp.array(self.part.local_elem_data["vol"], dtype=jnp.float64)
        vol_w = jnp.array(
            self.part.local_elem_data["vol"] * self.part.elem_weights,
            dtype=jnp.float64,
        )

        def _neo_hookean(v_local, volumes):
            # v_local is the FULL local node vector (3*N_scalar_nodes)
            # elems indexes into SCALAR node space
            vx_e = v_local[0::3][elems]
            vy_e = v_local[1::3][elems]
            vz_e = v_local[2::3][elems]

            F11 = jnp.sum(vx_e * dvx, axis=1)
            F12 = jnp.sum(vx_e * dvy, axis=1)
            F13 = jnp.sum(vx_e * dvz, axis=1)
            F21 = jnp.sum(vy_e * dvx, axis=1)
            F22 = jnp.sum(vy_e * dvy, axis=1)
            F23 = jnp.sum(vy_e * dvz, axis=1)
            F31 = jnp.sum(vz_e * dvx, axis=1)
            F32 = jnp.sum(vz_e * dvy, axis=1)
            F33 = jnp.sum(vz_e * dvz, axis=1)

            I1 = (F11**2 + F12**2 + F13**2 +
                  F21**2 + F22**2 + F23**2 +
                  F31**2 + F32**2 + F33**2)
            det = jnp.abs(F11*F22*F33 - F11*F23*F32
                        - F12*F21*F33 + F12*F23*F31
                        + F13*F21*F32 - F13*F22*F31)
            W = C1 * (I1 - 3 - 2*jnp.log(det)) + D1 * (det - 1)**2
            return jnp.sum(W * volumes)

        def energy_weighted(v_local):
            return _neo_hookean(v_local, vol_w)

        def energy_full(v_local):
            return _neo_hookean(v_local, vol)

        return energy_weighted, energy_full


class ParallelDOFHessianAssembler(_HyperElasticityMixin, DOFHessianAssemblerBase):
    """HE assembler with global coloring."""

    def __init__(self, params, comm, adjacency=None,
                 coloring_trials_per_rank=10, ksp_rtol=1e-3,
                 ksp_type="cg", pc_type="hypre",
                 ksp_max_it=None, near_nullspace_vecs=None,
                 pc_options=None):
        self._C1 = float(params["C1"])
        self._D1 = float(params["D1"])
        super().__init__(
            freedofs=np.asarray(params["freedofs"]),
            elems=np.asarray(params["elems"]),
            u_0=np.asarray(params["u_0"]),
            comm=comm, adjacency=adjacency,
            f=None,  # no body forces
            elem_data={
                "dvx": np.asarray(params["dvx"]),
                "dvy": np.asarray(params["dvy"]),
                "dvz": np.asarray(params["dvz"]),
                "vol": np.asarray(params["vol"]),
            },
            coloring_trials_per_rank=coloring_trials_per_rank,
            ksp_rtol=ksp_rtol, ksp_type=ksp_type, pc_type=pc_type,
            ksp_max_it=ksp_max_it,
            near_nullspace_vecs=near_nullspace_vecs,
            pc_options=pc_options,
        )


class LocalColoringAssembler(_HyperElasticityMixin, LocalColoringAssemblerBase):
    """HE assembler with per-rank local coloring + vmap."""

    def __init__(self, params, comm, adjacency=None,
                 coloring_trials_per_rank=10, ksp_rtol=1e-3,
                 ksp_type="cg", pc_type="hypre",
                 ksp_max_it=None, near_nullspace_vecs=None,
                 pc_options=None):
        self._C1 = float(params["C1"])
        self._D1 = float(params["D1"])
        super().__init__(
            freedofs=np.asarray(params["freedofs"]),
            elems=np.asarray(params["elems"]),
            u_0=np.asarray(params["u_0"]),
            comm=comm, adjacency=adjacency,
            f=None,
            elem_data={
                "dvx": np.asarray(params["dvx"]),
                "dvy": np.asarray(params["dvy"]),
                "dvz": np.asarray(params["dvz"]),
                "vol": np.asarray(params["vol"]),
            },
            coloring_trials_per_rank=coloring_trials_per_rank,
            ksp_rtol=ksp_rtol, ksp_type=ksp_type, pc_type=pc_type,
            ksp_max_it=ksp_max_it,
            near_nullspace_vecs=near_nullspace_vecs,
            pc_options=pc_options,
        )
```

### 2. `HyperElasticity3D_petsc_support/mesh.py`

Copy from `HyperElasticity3D_jax/mesh.py` with minimal changes.
The `get_data_jax()` method returns `(params, adjacency, u_init)` — same
interface as pLaplace.

**Important:** The mesh also provides:
- `elastic_kernel` — `(n_free, 6)` array of rigid body modes restricted to free DOFs
- These become `near_nullspace_vecs` for the assembler (one column per mode)

### 3. `HyperElasticity3D_petsc_support/rotate_boundary.py`

Copy from `HyperElasticity3D_jax/rotate_boundary.py`. Use numpy instead of jax
since this runs outside JAX:

```python
import numpy as np

def rotate_boundary(u_0, angle):
    """Rotate right-face Dirichlet BCs by angle (radians).
    
    Parameters
    ----------
    u_0 : ndarray, shape (3*N_nodes,)
        Full DOF vector with current Dirichlet values.
    angle : float
        Rotation angle around X-axis.
    
    Returns
    -------
    u_0_new : ndarray
        Updated full DOF vector.
    """
    coords = u_0.reshape(-1, 3)
    u_0_new = u_0.copy()
    lx = np.max(coords[:, 0])
    nodes = np.where(coords[:, 0] == lx)[0]
    u_0_new[nodes * 3 + 1] = (np.cos(angle) * coords[nodes, 1]
                              + np.sin(angle) * coords[nodes, 2])
    u_0_new[nodes * 3 + 2] = (-np.sin(angle) * coords[nodes, 1]
                              + np.cos(angle) * coords[nodes, 2])
    return u_0_new
```

**Note:** `u_0` here is the full `(3*N_nodes,)` vector where entries
contain nodal coordinates (identity deformation). `rotate_boundary` updates
only the right-face Dirichlet DOFs.

### 4. `HyperElasticity3D_jax_petsc/solve_HE_dof.py`

CLI solver script. Follow `pLaplace2D_jax_petsc/solve_pLaplace_dof.py` pattern
but add **load stepping**.

**Key details:**
- **24 load steps**, rotation per step = `4 * 2π / 24`
- At each step `s`:
  1. `angle = s * rotation_per_step`
  2. `u_0_new = rotate_boundary(u_0_original, angle)` — rotate the ORIGINAL coordinates
  3. `assembler.update_dirichlet(u_0_new)` — update Dirichlet values in partition
  4. Run Newton with previous solution as starting guess
  5. Store solution for next step
- **Initial guess for step 0**: coordinates at free DOFs (from `mesh.get_data_jax()`)
- **Near-nullspace**: pass `elastic_kernel` columns as `near_nullspace_vecs`
- **PC options**: `{"pc_hypre_boomeramg_nodal_coarsen": 6, "pc_hypre_boomeramg_vec_interp_variant": 3}`
- **KSP type**: `"cg"` (Hessian is SPD for well-posed elasticity)
- **PC type**: `"hypre"` (default)
- **Newton tolerances**: `tolf=1e-4`, `tolg=1e-3` (from FEniCS reference)
- **Thread control**: same as pLaplace (XLA single-thread, OMP_NUM_THREADS=1)

**Solver loop structure:**

```python
# u_0_original = params["u_0"]  (original nodal coordinates, never modified)
# x = assembler.create_vec(u_init_reordered)  (PETSc Vec)

total_steps = 24
rotation_per_step = 4 * 2 * np.pi / total_steps

for step in range(1, total_steps + 1):
    angle = step * rotation_per_step
    u_0_new = rotate_boundary(u_0_original, angle)
    assembler.update_dirichlet(u_0_new)
    
    result = newton(
        energy_fn=assembler.energy_fn,
        gradient_fn=assembler.gradient_fn,
        hessian_solve_fn=assembler.hessian_solve_fn,
        x=x,  # previous solution as starting guess
        tolf=1e-4, tolg=1e-3,
        linesearch_tol=1e-3,
        linesearch_interval=(-0.5, 2.0),
        maxit=100,
        verbose=verbose,
        comm=comm,
    )
    # x now contains the solution for this step
    if rank == 0:
        print(f"Step {step}/{total_steps}: {result['nit']} iters, J={result['fun']:.6f}")
```

---

## Critical Implementation Details

### DOF Indexing (v_local space)

The DOFPartition works with **total node indices** — the `elems` array
contains indices into a flat node vector. For 3D elasticity with 3 DOFs per
node, the full DOF vector has length `3 * N_scalar_nodes`, and `freedofs`
are indices into this flat vector.

In the `v_local` (local node-value vector), entries are also flat DOF indices.
The energy function accesses scalar-node-level data via `v_local[0::3]`,
`v_local[1::3]`, `v_local[2::3]` — slicing every 3rd entry to get x, y, z
displacement components.

**IMPORTANT**: The `elems` array indexes into **scalar node space** (not
DOF space). So `v_local[0::3][elems]` extracts x-components at element
nodes. This matches the original `HyperElasticity3D_jax/jax_energy.py`.

**BUT WAIT**: The generic DOFPartition works with a flat DOF vector where
`freedofs` are indices into that flat vector. For elasticity, `freedofs`
contains indices like `[0, 1, 2, 9, 10, 11, ...]` (groups of 3 per free
node). The `elems` array contains scalar node indices (not DOF indices).
The DOFPartition's `local_to_total` maps local node indices to total node
indices — these are **DOF indices** (flat), not scalar node indices.

**Check**: Look at how `HyperElasticity3D_jax/jax_energy.py` works:
- `v = u_0.at[freedofs].set(u)` — flat DOF vector (length 3*N_nodes)
- `v[0::3][elems]` — slice every 3rd entry to get x-components, then index by scalar-node-level elems

The DOFPartition's `n_total = len(u_0) = 3*N_nodes`. Its `elems` are in
scalar-node-index space. The `v_template` and `v_local` are in DOF-index
space (length `n_local` which maps to flat DOF indices).

**This means**: `v_local[0::3][elems_local]` works correctly IF the local
node mapping preserves the DOF-to-scalar-node structure. Let's verify:
- `local_to_total` contains the total DOF indices of local nodes
- `np.unique(local_elems_total.ravel())` — local_elems_total contains scalar
  node indices, so `local_to_total` will contain scalar node indices
- Then `v_local` is indexed by these local indices, and it has DOF entries

**ACTUALLY**: The critical question is whether `elems` contains scalar node
indices or DOF indices. Looking at the mesh file:
- `elems2nodes` — scalar node indices per element
- `u_0` is length `3*N_nodes` (DOF vector)
- `freedofs` = `dofsMinim` — indices into the DOF vector

The DOFPartition takes `elems` (scalar node indices) and `u_0` (DOF vector).
Inside DOFPartition, `total_to_free_reord` maps total (DOF) indices to
reordered free-DOF indices. It indexes `total_to_free_reord[elems]` — but
`elems` contains scalar node indices, not DOF indices!

**THIS IS A PROBLEM**: The generic DOFPartition assumes `elems` indexes into
the same space as `u_0` and `freedofs`. For pLaplace (1 DOF/node), scalar
node index = DOF index. For elasticity (3 DOFs/node), they differ.

### Resolution Options

**Option A — Pre-expand elems to DOF connectivity:**
Convert scalar-node `elems` to DOF-level connectivity before passing to
DOFPartition. Each scalar element `[n0, n1, n2, n3]` becomes
`[3*n0, 3*n0+1, 3*n0+2, 3*n1, 3*n1+1, ...]`. Then all indexing is
consistent in flat DOF space. The energy function would use this expanded
connectivity directly: `v_e = v_local[dof_elems]` then reshape.

**Option B — Pass scalar-node-level u_0 and freedofs:**
Reshape u_0 and freedofs to scalar-node level. But this doesn't work
because individual DOF components can be separately free/fixed.

**Option A is strongly recommended.** The DOFPartition and assembler
infrastructure work correctly in flat DOF space. The expansion is done once
at mesh loading time.

**Implementation for Option A:**

In `mesh.py`, after loading:
```python
# Original scalar-node connectivity
elems_scalar = params["elems2nodes"]  # (n_elems, 4) for tets

# Expand to DOF connectivity: each scalar node → 3 DOFs
# elems_dof[e, :] = [3*n0, 3*n0+1, 3*n0+2, 3*n1, 3*n1+1, 3*n1+2, ...]
n_elems, npe = elems_scalar.shape
elems_dof = np.empty((n_elems, npe * 3), dtype=np.int64)
for k in range(3):
    elems_dof[:, k::3] = 3 * elems_scalar + k
```

Then the energy function in `_make_local_energy_fns()` uses:
```python
elems_dof = jnp.array(self.part.elems_local_np)  # (n_elems, 12) for tets
# Extract per-component:
vx_e = v_local[elems_dof[:, 0::3]]  # x-component at element nodes
vy_e = v_local[elems_dof[:, 1::3]]  # y-component
vz_e = v_local[elems_dof[:, 2::3]]  # z-component
```

And `dvx, dvy, dvz` remain `(n_elems, npe)` — they are multiplied with
the `(n_elems, npe)` node-value slices.

**Also**: `elem_data` arrays (`dvx`, `dvy`, `dvz`, `vol`) remain at
scalar-element level — they have shape `(n_elems, npe)` where `npe` = 4
for tets. The DOFPartition slices them by `local_elem_idx` which is correct.

---

## Validation

### Reference values from FEniCS solver

Run `HyperElasticity3D_fenics/solve_HE_custom_jaxversion.py` for reference:
- Check final energy after all 24 load steps
- Check per-step Newton iteration counts
- Check per-step energy values

### Validation strategy

1. **Single rank, level 1**: Run all 24 steps, compare per-step energy with
   FEniCS reference (should match to ~6 digits)
2. **4 ranks, level 1**: Same comparison — verify MPI doesn't change results
3. **Single rank, level 2+**: Verify convergence, compare with FEniCS

### Expected physical behavior

- Step 1: small rotation, 2-4 Newton iters
- Steps 2-24: continuation from previous, typically 2-5 iters each
- Total rotation = `4 * 2π ≈ 25.13 rad` (four full turns of right face)
- Energy should increase with rotation (elastic strain energy)

---

## Mesh Data

Pre-generated HDF5 files in `mesh_data/HyperElasticity/`:
- `HyperElasticity_level1.h5` through `level4.h5`

H5 keys: `u0`, `dofsMinim`, `elems2nodes`, `dphix`, `dphiy`, `dphiz`,
`vol`, `C1`, `D1`, `nodes2coord`, `adjacency/` (COO group with `data`,
`row`, `col`, `shape`).

---

## Summary of what's already implemented vs what to create

### Already done (just import and use):
- `tools_petsc4py.dof_partition.DOFPartition` — generic partitioning
- `tools_petsc4py.jax_tools.parallel_assembler.LocalColoringAssemblerBase` — abstract base
- `tools_petsc4py.jax_tools.parallel_assembler.DOFHessianAssemblerBase` — abstract base
- `tools_petsc4py.minimizers.newton` — Newton solver with line search
- Near-nullspace support in `_setup_petsc`
- `update_dirichlet()` in base assembler
- `pc_options` dict support

### Current implementation targets:
1. `HyperElasticity3D_jax_petsc/__init__.py` — docstring
2. `HyperElasticity3D_petsc_support/mesh.py` — HDF5 mesh loader (adapt from `HyperElasticity3D_jax/mesh.py`)
3. `HyperElasticity3D_petsc_support/rotate_boundary.py` — numpy boundary rotation
4. `HyperElasticity3D_jax_petsc/parallel_hessian_dof.py` — thin subclasses (~110 lines)
5. `HyperElasticity3D_jax_petsc/solver.py` / `solve_HE_dof.py` — solver logic + CLI wrapper

The problem-specific code is ~300 lines total. All the heavy infrastructure
(~1300 lines) is reused from `tools_petsc4py/`.
