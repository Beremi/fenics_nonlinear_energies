# Investigation of JAX+PETSc vs FEniCS Performance Gap

*Reference: `jax_petsc_fenics_performance_investigation_plan.md`*

## Experiment: Isolating Local Compute vs. MPI Assembly (The "Hidden Problem")
Following the investigation plan, we aimed to strictly separate the **local Hessian computation** from the **PETSc matrix assembly** (`setValuesCOO`) pipeline to identify the root cause of JAX+PETSc appearing drastically slower than FEniCS Custom.

### Hypothesis
Initially, profiling suggested that JAX completed its HVP computation rapidly, but then spent an excessive amount of time inside `self.A.setValuesCOO(...)`. The working theory was that PETSc's Python wrapping (`petsc4py`), internal sorting, or coordinate caching was introducing huge overheads compared to FEniCS's native C++ assembly (`MatSetValuesLocal`). 

### Methodology
To test this, we introduced a strict `self.comm.Barrier()` into `tools_petsc4py/parallel_assembler.py` and `HyperElasticity3D_jax_petsc/parallel_hessian_dof.py` right after the JAX compute block and just before `setValuesCOO`. We captured the time spent waiting at this barrier as a new metric: `wait_imbalance`.

### Results (Mesh Level 3, np=16, 39 Newton Iterations)

**Before the barrier (Original Timing Profile):**
* **Total Assembly Time:** 1.544 s
* **HVP Compute:** 0.529 s
* **PETSc COO Assembly:** 0.883 s  *(Looked like the bottleneck)*

**After the barrier (Isolated Timing Profile):**
* **Total Assembly Time:** 1.541 s
* **HVP Compute:** 0.522 s
* **Wait (Load Imbalance):** **0.832 s**
* **PETSc COO Assembly:** **0.055 s** *(Actual PETSc assembly speed)*

Looking at a per-iteration breakdown for Rank 0:
* Iter 1: Compute = `0.0131 s`, Wait = `0.0220 s`, COO = `0.0014 s`
* Iter 2: Compute = `0.0133 s`, Wait = `0.0212 s`, COO = `0.0014 s`
* *(Consistent across all iterations)*

### Conclusion: The "Hidden Problem" is Identified
The bottleneck is **NOT** PETSc's `A.setValuesCOO` implementation. The actual insertion of elements into the PETSc sparse matrix takes an incredibly fast `~1.4 ms` per iteration (`0.055 s` total).

The massive time sink (`~883 ms`) previously attributed to PETSc is actually **MPI Load Imbalance**. 
When ranks call `setValuesCOO(...)` or `A.assemble()`, they enter a collective MPI state. Because Rank 0 finishes its JAX HVP evaluate in ~13ms, it arrives at `setValuesCOO` early and sits completely idle for ~21ms waiting for the slowest MPI rank to finish its JAX compilation/evaluation.

### Next Steps for Investigation
Now that we have definitively isolated the gap to JAX compute load-imbalance, the next phase from our plan is to answer: **Why does JAX evaluation timescale vary so aggressively across MPI ranks while FEniCS Custom executes uniformly?**

1. **JAX Padding / Ghosting overhead:** Are some MPI ranks padding their JAX arrays to a larger static shape (for XLA compilation) causing them to do wildly more FLOPs than their actively owned elements require?
2. **CPU Thread Contention:** JAX might be over-provisioning CPU threads (`OMP_NUM_THREADS`, `XLA_FLAGS`) causing core thrashing on ranks handling boundary conditions, while FEniCS gracefully handles threaded assembly.
3. **Partition Size:** Double-check if the actual integer cell ownership is significantly imbalanced by `dolfinx.mesh.create_box` across 16 ranks.

*Logs and parsing scripts strictly stored in `tmp_work/` per environment setup restrictions.*

## Root Cause Discovered: Naive 1D Algebraic Chunking vs Graph Partitioning
To verify why JAX execution times varied so vastly between MPI ranks, we instrumented the code to print exactly how many local elements each rank processed during `assemble_hessian_element()`.

### The Element Count Distribution (`mpirun -n 16`)
We found a staggering disparity in the number of elements processed per rank:
* **Rank 15:** 8,742 elements
* **Rank 0:** 8,748 elements
* **Rank 1, 14:** ~10,900 elements
* **Rank 3, 5, 10, 12:** ~17,492 elements
* **Rank 2, 4, 11, 13:** **21,326 elements**

**The slowest ranks process ~2.4x as many elements as the fastest ranks.** 
Since JAX is running a synchronous mathematical `vmap` over all locally designated elements, a rank with 2.4x the elements takes approximately 2.4x the compute time. 
From our earlier timing: `0.013s` (Rank 0 compute) x `2.4` = `~0.031s`, perfectly explaining the `0.022s` wait imbalance!

### Why does this happen?
Within FEniCS, mesh distribution is typically handled by advanced topological graph partitioners (like ParMETIS or SCOTCH). These partitioners distribute an *equal number of cells* to each MPI process while mathematically minimizing the surface area of the partition boundaries (the "ghost cells").

However, the JAX+PETSc codebase does **not** rely on FEniCS topology distribution. Instead:
1. It loads the exact same monolithic HDF5 mesh onto all MPI ranks.
2. Rank 0 calculates a **Reverse Cuthill-McKee (RCM)** ordering of the DOF adjacency matrix to reduce bandwidth.
3. The `tools_petsc4py/dof_partition.py` script naively chunks the 1D reordered DOF list into perfectly equal chunks using `petsc_ownership_range`.
4. Finally, any element that touches an owned DOF is dragged into that rank's local element list.

**The Flaw in Algebraic Decomposition:** While slicing an RCM-sorted 1D array perfectly balances the number of DOFs each process *"owns"*, it creates chaotic and wildly varying topological surface areas. Processes assigned DOFs in the interior clusters might have to process very few shared boundary elements, while processes at certain RCM structural boundaries get "spaghetti" partitions with massive ghost-layers. 

This causes massive redundant element evaluations: Rank 13 ends up evaluating almost a third of the mesh just to assemble its contiguous chunk of the matrix, resulting in crippling load imbalance (`MPI.Barrier()` wait time).

## Conclusion & Proposed Fixes
The primary performance gap slowing down JAX+PETSc vs FEniCS Custom in multi-core deployments is **not PETSc assembly bindings, nor JAX compilation limits, but fundamentally unequal element distributions causing MPI sync imbalance.**

To close this gap and match FEniCS scaling efficiency:
* **Adopt FEniCS Graph Partitioning:** Instead of using basic algebraic 1D DOF splitting (`dof_partition.py`), the codebase should natively pull the mesh partition map from DolfinX's ParMETIS/SCOTCH distribution.
* **Element-Driven Distribution:** Assign MPI ownership by *Elements* rather than *DOFs*, assigning the computation of an element strictly to one rank, and handling the accumulation of shared interface DOFs naturally using `petsc4py`'s `ADD_VALUES` natively over the MPI interconnect.
