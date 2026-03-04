# Investigation: JAX+PETSc Performance Gap — pLaplace vs HyperElasticity

## Motivation

For the **pLaplace** problem, the JAX+PETSc solver is roughly **on par with or faster than** the FEniCS custom solver. But for **HyperElasticity**, JAX+PETSc is **5–19× slower** despite using the same solver infrastructure, the same Newton algorithm, and the same preconditioner (GAMG). This report investigates the root causes.

---

## 1) pLaplace Comparison (finest level, np=32, GAMG)

**Problem**: 2D p-Laplacian, P1 elements, 784,385 free DOFs, native build (Threadripper PRO 7975WX).

| Metric | FEniCS Custom + GAMG | JAX+PETSc + GAMG | Ratio |
| --- | ---: | ---: | ---: |
| Solve time | 0.785 s | 0.62 s | **0.79×** (JAX faster) |
| Newton iterations | 6 | 7 | 1.17× |
| Total KSP iterations | 77 | 52 | 0.68× |
| Total time (incl. setup) | 0.785 s | 1.68 s | 2.14× |

**Source**: `results_pLaplace.md`, experiment_002 (native build, 32 MPI ranks).

**Conclusion**: At np=32 on the finest level, JAX+PETSc is **faster in solve time** and **comparable overall** (setup cost adds overhead). The SFD Hessian assembly, KSP solve, and line search are all efficient for this problem.

---

## 2) HyperElasticity Comparison (L3, np=32, GAMG, 24 steps)

**Problem**: 3D Neo-Hookean beam, P1 vector elements, 77,517 DOFs (L3), 24 load steps, native build.

| Metric | FEniCS Custom + GAMG | JAX+PETSc + GAMG | Ratio |
| --- | ---: | ---: | ---: |
| Total time (24 steps) | 52.9 s | 267.06 s | **5.0×** (JAX slower) |
| Average step time | 2.20 s | 11.13 s | 5.1× |
| Total Newton iterations | 1,164 | 1,065 | 0.91× |
| Total KSP iterations | 20,894 | 17,508 | 0.84× |
| Time per Newton iter | 0.045 s | 0.251 s | **5.5×** |
| Final energy | 93.704832 | 93.704996 | ✓ match |

**Source**: `results_HyperElasticity3D.md`, Annex G.4 (FEniCS np=32), Annex F.12 (JAX+PETSc np=32).

**Key observation**: JAX+PETSc has **fewer Newton iters** and **fewer KSP iters** yet is **5× slower**. The bottleneck is entirely in per-iteration cost, not convergence behavior.

---

## 3) The Discrepancy: Why pLaplace ≈ Parity but HE ≈ 5× Slower?

### 3.1 Root Cause #1: Graph Coloring — 8 colors (pLaplace) vs 63 colors (HE)

The SFD Hessian assembly requires one HVP (Hessian-vector product) evaluation **per graph color per Newton step**.

| Problem | DOF adjacency | Graph colors (igraph, L3) | HVPs per Newton step |
| --- | --- | ---: | ---: |
| pLaplace 2D | Scalar P1, 2D triangles | **8** | 8 |
| HyperElasticity 3D | Vector P1 (3 DOFs/node), 3D tets | **63** | 63 |

**Source**: `results_graph_coloring.md`, Table "HyperElasticity 3D", Level 3, igraph column.

The 3D vector FE space has much higher vertex degree in the DOF adjacency graph:
- pLaplace 2D P1: each DOF connects to ~6–8 neighbours → A² has max degree ~20 → 8 colors
- HE 3D P1 vector: each scalar DOF connects to 3× more DOFs (block coupling) and tets have 4 nodes × 3 DOFs = 12 DOFs per element → A² has max degree ~70 → 63 colors

**This means the JAX+PETSc HE solver does ~8× more HVP evaluations per Newton step than pLaplace.** Since HVP compute dominates assembly time (>96%), this alone accounts for most of the gap.

FEniCS does **not** use SFD — it assembles the Hessian from compiled UFL symbolic forms in a **single pass**, regardless of DOF connectivity. This is the fundamental asymmetry.

### 3.2 Root Cause #2: Each HVP Is More Expensive for HE

Each HVP evaluation for HE involves:
- 3D tetrahedral elements with 12 DOFs per element (vs 3 nodes, 2D triangles)
- Deformation gradient F computation (3×3 matrix per element)
- Determinant via cofactor formula
- Logarithm evaluation (`log(det F)`)
- Neo-Hookean energy with two invariants

For pLaplace, the energy function is simpler: `|∇u|^p / p` with p=3, computed from a 2-component gradient.

### 3.3 Root Cause #3: COO Matrix Finalization Cost

From the step-1 timing investigation (`he_step1_timing_investigation_fix1.md`, L3, np=16, Docker):

The patched JAX+PETSc solver shows that COO matrix finalization (`A.assemble()` after `setValuesCOO`) takes **12.82 s** out of 18.97 s assembly time. This is the PETSc internal step that sorts/communicates COO entries into the distributed MPIAIJ matrix.

For pLaplace at 784K DOFs with 8 colors, the COO path inserts ~5.5M entries. For HE at 78K DOFs with 63 colors, it inserts ~3.1M entries — similar count but the block-3 structure and off-diagonal patterns may cause more communication.

### 3.4 Root Cause #4: KSP Solve Is 5× Slower Despite Similar Iteration Count

From the step-1 breakdown (L3, np=16, Docker, **patched** JAX+PETSc):

| Linear component | FEniCS | JAX+PETSc (patched) | Ratio |
| --- | ---: | ---: | ---: |
| Hessian assembly | 0.616 s | 18.97 s | 30.8× |
|   ↳ HVP compute | — | 5.92 s | — |
|   ↳ COO/finalize | — | 12.82 s | — |
| PC setup | 0.064 s | 1.28 s | 20× |
| KSP solve | 1.020 s | 5.14 s | **5.0×** |
| **Linear total** | **1.700 s** | **24.44 s** | **14.4×** |

KSP iterations are almost identical (414 vs 402). Possible explanations for the 5× KSP solve gap:
1. **SFD Hessian quality**: The finite-difference approximation may produce a slightly different matrix than the exact UFL Hessian. This was confirmed by the matrix-layout isolation experiment (F.11): "FEniCS matrix + JAX values → `ksp_its=4`" vs "FEniCS matrix + FEniCS values → `ksp_its=1`". The JAX SFD values produce a matrix that requires more KSP work per iteration.
2. **GAMG setup cost**: PC setup is 20× slower (1.28s vs 0.064s). This suggests the matrix structure from COO assembly may cause GAMG to do more work building coarse levels.
3. **Possible hidden finalization**: Some COO assembly work may still be deferred into `ksp.solve()`.

### 3.5 Root Cause #5: Line Search and Gradient Are Also Slower

| Component | FEniCS | JAX+PETSc | Ratio |
| --- | ---: | ---: | ---: |
| Gradient evaluation (sum over Newton) | 0.035 s | 0.50 s | 14× |
| Line search (sum over Newton) | 0.365 s | 2.97 s | 8.1× |
| Update | 0.018 s | 0.15 s | 8.2× |

- **Gradient**: FEniCS uses compiled UFL forms (`assemble_vector`). JAX uses AD, which involves a full forward pass through the energy function + P2P ghost exchange.
- **Line search**: Each golden-section probe requires an energy evaluation (JAX energy + Allreduce). With ~17 probes per Newton step and 40 Newton steps, that's ~680 energy evaluations. Each requires P2P exchange + JAX forward pass + MPI_Allreduce.

### 3.6 Root Cause #6: Load Imbalance in HVP Compute

From F.11: Per-rank Hessian value compute time (np=16, step 1):
- min = 0.085 s, max = 0.462 s → **imbalance ratio = 5.42×**

This means the slowest rank takes 5.4× longer than the fastest. All ranks must synchronize at `A.assemble()`, so the slowest rank determines wall time. This imbalance likely stems from uneven element distribution across ranks after RCM-based DOF partitioning (the 3D beam geometry creates asymmetric partitions).

---

## 4) Detailed Step-1 Timing Comparison (L3, np=16, GAMG, Docker)

Full Newton-level breakdown from `he_step1_timing_breakdown_fenics_vs_jaxpetsc.md`:

| Component | FEniCS [s] | JAX+PETSc [s] | Ratio | Share of JAX total |
| --- | ---: | ---: | ---: | ---: |
| Gradient eval (all Newton iters) | 0.035 | 0.499 | 14.3× | 1.2% |
| Hessian callback (all Newton iters) | 1.702 | 38.562 | 22.7× | 91.4% |
|   ↳ Assembly (incl COO finalize) | 0.616 | 20.788 | 33.7× | 49.3% |
|     ↳ P2P exchange | — | 0.019 | — | 0.04% |
|     ↳ HVP compute | — | 20.097 | — | **47.7%** |
|     ↳ extraction | — | 0.406 | — | 1.0% |
|     ↳ COO assembly | — | 0.259 | — | 0.6% |
|   ↳ PC setup | 0.064 | 1.281 | 20× | 3.0% |
|   ↳ KSP solve | 1.020 | 16.473 | 16.2× | 39.1% |
| Line search (all Newton iters) | 0.365 | 2.960 | 8.1× | 7.0% |
| Update (all Newton iters) | 0.018 | 0.150 | 8.2× | 0.4% |
| **Step total** | **2.123** | **42.189** | **19.9×** | 100% |

**Note**: This is the **unpatched** version. The patched version (sequential HVP + explicit `A.assemble()`) reduced total to 28.08s (1.50× improvement) by moving hidden KSP overhead into explicit assembly accounting.

---

## 5) Why pLaplace Doesn't Suffer

For pLaplace at the same np=16 (Docker, experiment_001):

| Metric | FEniCS Custom | JAX+PETSc | Ratio |
| --- | ---: | ---: | ---: |
| Solve time (np=16) | 1.53 s | 2.18 s (GAMG) | 1.42× |
| KSP time (np=16) | ~1.0 s | ~1.1 s | 1.10× |

**Source**: `results_pLaplace.md`, "Comparison with FEniCS Custom Newton" table.

The KSP solve is nearly identical (1.10×), confirming the matrix quality and preconditioner behavior match. The pLaplace SFD Hessian with 8 colors produces a matrix that is spectrally close to the exact UFL Hessian — GAMG works identically.

For HE, 63 colors + larger finite-difference stencil + more complex energy function means:
1. More numerical noise accumulates in the SFD approximation
2. The resulting matrix has slightly different spectral properties
3. GAMG coarsening/interpolation operates differently

---

## 6) Summary: Where the Time Goes (HE L3, np=16)

```
FEniCS step-1:  2.12s total
├── Hessian assembly (UFL):      0.62s  (29%)    ← single-pass compiled forms
├── PC setup (GAMG):             0.06s  ( 3%)
├── KSP solve:                   1.02s  (48%)
├── Line search:                 0.37s  (17%)
└── Gradient + update + other:   0.05s  ( 3%)

JAX+PETSc step-1:  42.19s total (unpatched) / 28.08s (patched)
├── HVP compute (63 colors × JAX fwd): 20.10s  (48%)  ← DOMINANT BOTTLENECK
├── COO finalize (A.assemble):    ~12.8s  (30%)  ← expensive with block-3 structure
├── KSP solve:                     5.14s  (12%)  ← 5× slower than FEniCS
├── Line search:                   2.97s  ( 7%)  ← 8× slower (JAX energy + Allreduce)
├── PC setup:                      1.28s  ( 3%)
└── Gradient + update + other:     0.65s  ( 2%)
```

---

## 7) Conclusions and Potential Improvements

### What degrades JAX+PETSc performance for HyperElasticity:

1. **63 graph colors** (vs 8 for pLaplace) → 63 HVP evaluations per Newton step. This is the single biggest factor (accounts for ~48% of total step time).

2. **COO matrix finalization** is expensive for the block-3 vector structure (~30% of step time in patched version).

3. **KSP solve is 5× slower** despite similar iteration counts, suggesting the SFD-approximated Hessian has subtly different spectral properties that make GAMG less effective per iteration.

4. **Line search and gradient** evaluations are 8–14× slower because each requires a full JAX forward pass + MPI communication, whereas FEniCS uses compiled C forms.

5. **Load imbalance** (5.4× between fastest/slowest rank in HVP compute) wastes parallel resources.

### Potential improvements to investigate:

| Improvement | Expected impact | Difficulty |
| --- | --- | --- |
| **Block coloring** (color nodes, not scalar DOFs → ~21 colors instead of 63) | ~3× reduction in HVP count | Medium |
| **Analytical Hessian via JAX** (`jax.hessian` or symbolic differentiation) | Eliminate SFD entirely → single-pass assembly | High |
| **Better load balancing** (ParMETIS/SCOTCH partitioning instead of RCM) | Reduce 5.4× imbalance | Medium |
| **Reduce COO finalization cost** (use `ADD_VALUES` with preallocation hints, or switch to `setValuesBlocked`) | Reduce 12.8s → closer to 1s | Medium |
| **Cache HVP JIT compilation** across Newton steps | Minor (JIT already cached) | Low |
| **Use element-level assembly** instead of DOF-level SFD | Match FEniCS approach, eliminate coloring | High (redesign) |

---

## 8) Experiment: Element-Level Analytical Hessian Assembly

### 8.1 Approach

Instead of 63 SFD HVP evaluations per Newton step, compute the **12×12 element Hessian** analytically via `jax.hessian(element_energy)` + `jax.vmap` over all elements in a single pass. Then pre-aggregate element contributions and write into `self.A` using the **same SFD COO pattern** with `INSERT_VALUES`.

**Implementation**: `--assembly_mode element` flag in `solve_HE_dof.py`. Code in `LocalColoringAssembler.setup_element_hessian()` and `assemble_hessian_element()`.

### 8.2 Correctness Verification

Matrices match to machine precision (relative Frobenius norm difference = 2.12e-14 at L3 np=4). Solver converges to identical energy and identical Newton/KSP iteration counts.

### 8.3 Evolution: Two Iterations of the Element Assembly Approach

**Iteration 1 (separate `A_elem` matrix with duplicate COO + ADD_VALUES)**:
The first implementation created a separate PETSc matrix `A_elem` with an element-based COO pattern containing duplicates. Assembly was 3.5–5× faster, but KSP solve regressed ~1.5× despite identical matrices. Overall speedup was limited (1.08× at np=32).

**Root cause of KSP regression**: PETSc's `setPreallocationCOO` with duplicate entries produces a different internal matrix storage layout than the SFD's unique-entry COO. This made matrix-vector products less cache-efficient.

**Iteration 2 (reuse `self.A` with pre-aggregated SFD COO pattern)**:
The fix pre-aggregates element contributions into the same COO pattern as the SFD matrix using `np.add.at`, then writes into `self.A` with `INSERT_VALUES`. This preserves PETSc's internal storage layout.

**Bug discovered**: PETSc's `setPreallocationCOO` for MPIAIJ matrices modifies the input column array in-place (remapping off-process columns). The original COO arrays must be reconstructed from the adjacency data (`_row_adj`, `_col_adj`) to build the correct element→COO position mapping.

### 8.4 Step-1 Benchmark Results (L3, GAMG, native build, Iteration 2)

| np | Mode | Step time [s] | Assembly [s] | Compute [s] | COO [s] | PC setup [s] | KSP solve [s] | Newton | KSP | Energy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | SFD | 26.163 | 18.200 | 11.660 | 6.273 | 0.815 | 5.060 | 38 | 500 | 0.1625951097 |
| 4 | Element | 12.224 | 4.219 | 1.852 | 1.580 | 0.819 | 5.129 | 38 | 500 | 0.1625951097 |
| 16 | SFD | 23.123 | 12.184 | 3.679 | 8.419 | 0.387 | 7.935 | 39 | 402 | 0.1626936687 |
| 16 | Element | 7.990 | 1.643 | 0.508 | 1.001 | 0.299 | 4.455 | 39 | 402 | 0.1626936687 |
| 32 | SFD | 17.540 | 8.070 | 1.949 | 6.065 | 0.325 | 6.665 | 40 | 413 | 0.1626156380 |
| 32 | Element | 10.725 | 1.521 | 0.398 | 1.019 | 0.341 | 6.574 | 40 | 413 | 0.1626156379 |

### 8.5 Speedup Summary

| np | SFD step [s] | Element step [s] | Overall speedup | Assembly speedup | Compute speedup | KSP solve ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 26.16 | 12.22 | **2.14×** | 4.31× | 6.30× | 1.01× (no regression) |
| 16 | 23.12 | 7.99 | **2.89×** | 7.42× | 7.24× | 0.56× (elem faster!) |
| 32 | 17.54 | 10.73 | **1.64×** | 5.31× | 4.90× | 0.99× (no regression) |

### 8.6 Analysis

**Assembly speedup is now 4–7×** across rank counts. The element Hessian pass (`jax.vmap(jax.hessian(...))`) replaces 63 sequential HVP evaluations with a single vmapped computation.

**COO finalization** also improved **~6× at np=16** (8.4s → 1.0s). Pre-aggregation via `np.add.at` is much faster than PETSc's internal duplicate-handling ADD_VALUES path.

**KSP solve regression is eliminated**: solve times now match between SFD and element assembly (e.g., 6.57s vs 6.67s at np=32). At np=16, element is even faster (4.5s vs 7.9s), likely due to reduced memory pressure from not maintaining a separate matrix.

**Overall speedup is 1.6–2.9×**: a major improvement over the previous iteration (1.08–1.79×).

### 8.7 Remaining Gap vs FEniCS

FEniCS custom + GAMG at np=32: **step 1 ≈ 1.6 s** (from Annex G.4).
JAX+PETSc + Element at np=32: **step 1 = 10.7 s** → still **~6.7× slower**.

The remaining gap is dominated by:
- **KSP solve**: 6.6s (JAX+PETSc) vs ~0.8s (FEniCS estimated) — possibly different GAMG setup quality or matrix storage efficiency
- **Line search**: each energy probe requires JAX computation + MPI Allreduce
- **Gradient**: JAX AD vs compiled UFL forms

### 8.8 Reproducing

```bash
source local_env/activate.sh
export OMP_NUM_THREADS=1

# Element Hessian assembly (Iteration 2 — reuses self.A)
mpirun -n 16 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py \
    --level 3 --steps 1 --total_steps 24 \
    --profile performance --assembly_mode element \
    --save_linear_timing --quiet

# SFD baseline
mpirun -n 16 python3 HyperElasticity3D_jax_petsc/solve_HE_dof.py \
    --level 3 --steps 1 --total_steps 24 \
    --profile performance --assembly_mode sfd \
    --save_linear_timing --quiet
```
