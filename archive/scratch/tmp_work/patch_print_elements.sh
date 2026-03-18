sed -i '/def assemble_hessian_element(self, u_owned):/a \
        num_elems = len(self._elem_weights)\n\
        if not hasattr(self, "_printed_elems"):\n\
            print(f"Rank {self.comm.rank}: Evaluating {num_elems} elements")\n\
            self._printed_elems = True\n' HyperElasticity3D_jax_petsc/parallel_hessian_dof.py
