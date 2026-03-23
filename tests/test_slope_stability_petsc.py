from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from src.problems.slope_stability.jax_petsc import multigrid
from src.problems.slope_stability.jax_petsc import solve_slope_stability_dof
from src.problems.slope_stability.jax_petsc.reordered_element_assembler import (
    SlopeStabilityReorderedElementAssembler,
)
from src.problems.slope_stability.support import (
    build_near_nullspace_modes,
    build_same_mesh_lagrange_case_data,
    case_name_for_level,
    davis_b_reduction,
    ensure_same_mesh_case_hdf5,
)
from src.problems.slope_stability.support.mesh import _build_same_mesh_lagrange_case_data_impl


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PETSC_SOLVER = "src/problems/slope_stability/jax_petsc/solve_slope_stability_dof.py"
PETSC_REFINED_P1_SOLVER = "src/problems/slope_stability/jax_petsc/solve_slope_stability_refined_p1_dof.py"
SERIAL_SOLVER = "src/problems/slope_stability/jax/solve_slope_stability_jax.py"


def _run_json(command: list[str], output_path: Path) -> dict[str, object]:
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _petsc_command(
    output_path: Path,
    *,
    nprocs: int = 1,
    local_hessian_mode: str = "element",
) -> list[str]:
    return [
        "mpiexec",
        "-n",
        str(nprocs),
        str(PYTHON),
        "-u",
        PETSC_SOLVER,
        "--level",
        "1",
        "--lambda-target",
        "1.2",
        "--profile",
        "performance",
        "--ksp_type",
        "stcg",
        "--pc_type",
        "gamg",
        "--ksp_rtol",
        "1e-1",
        "--ksp_max_it",
        "30",
        "--assembly_mode",
        "element",
        "--element_reorder_mode",
        "block_xyz",
        "--local_hessian_mode",
        local_hessian_mode,
        "--local_coloring",
        "--use_trust_region",
        "--trust_subproblem_line_search",
        "--linesearch_tol",
        "1e-1",
        "--trust_radius_init",
        "0.5",
        "--trust_radius_min",
        "1e-8",
        "--trust_radius_max",
        "1e6",
        "--trust_shrink",
        "0.5",
        "--trust_expand",
        "1.5",
        "--trust_eta_shrink",
        "0.05",
        "--trust_eta_expand",
        "0.75",
        "--trust_max_reject",
        "6",
        "--tolf",
        "1e-4",
        "--tolg",
        "1e-3",
        "--tolg_rel",
        "1e-3",
        "--tolx_rel",
        "1e-3",
        "--tolx_abs",
        "1e-10",
        "--maxit",
        "100",
        "--quiet",
        "--out",
        str(output_path),
    ]


def test_slope_stability_petsc_cli_accepts_trust_and_state_flags():
    parser = solve_slope_stability_dof._build_parser({"reference": {}, "performance": {}})
    args = parser.parse_args(
        [
            "--assembly_mode",
            "element",
            "--operator_mode",
            "matfree_overlap",
            "--local_hessian_mode",
            "sfd_local",
            "--use_trust_region",
            "--trust_radius_init",
            "0.5",
            "--trust_shrink",
            "0.5",
            "--trust_expand",
            "1.5",
            "--trust_eta_shrink",
            "0.05",
            "--trust_eta_expand",
            "0.75",
            "--trust_max_reject",
            "6",
            "--trust_subproblem_line_search",
            "--state-out",
            "state.npz",
            "--preconditioner_operator",
            "refined_p1_same_nodes",
            "--mg_coarsest_level",
            "2",
            "--elem_degree",
            "4",
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_variant",
            "outer_pcksp",
            "--fine_pmat_policy",
            "elastic_frozen",
            "--fine_pmat_stagger_period",
            "2",
            "--mg_lower_operator_policy",
            "fixed_setup",
            "--mg_p4_smoother_ksp_type",
            "chebyshev",
            "--mg_p4_smoother_pc_type",
            "jacobi",
            "--mg_p4_smoother_steps",
            "4",
            "--mg_p2_smoother_ksp_type",
            "richardson",
            "--mg_p2_smoother_pc_type",
            "sor",
            "--mg_p2_smoother_steps",
            "3",
            "--mg_p1_smoother_ksp_type",
            "richardson",
            "--mg_p1_smoother_pc_type",
            "jacobi",
            "--mg_p1_smoother_steps",
            "2",
            "--mg_fine_ksp_type",
            "gmres",
            "--mg_degree2_pc_type",
            "sor",
            "--pc_reuse_preconditioner",
            "--mg_coarse_backend",
            "hypre",
            "--mg_coarse_hypre_nodal_coarsen",
            "6",
            "--mg_coarse_hypre_vec_interp_variant",
            "3",
            "--mg_coarse_hypre_strong_threshold",
            "0.45",
            "--mg_coarse_hypre_coarsen_type",
            "HMIS",
            "--hypre_nodal_coarsen",
            "6",
            "--hypre_vec_interp_variant",
            "3",
            "--hypre_strong_threshold",
            "0.5",
            "--hypre_coarsen_type",
            "HMIS",
            "--hypre_max_iter",
            "3",
            "--hypre_tol",
            "0.0",
            "--hypre_relax_type_all",
            "symmetric-SOR/Jacobi",
            "--line_search",
            "armijo",
            "--armijo_alpha0",
            "0.75",
            "--armijo_c1",
            "1e-5",
            "--armijo_shrink",
            "0.6",
            "--armijo_max_ls",
            "18",
            "--benchmark_mode",
            "warmup_once_then_solve",
            "--mg_coarse_hypre_max_iter",
            "4",
            "--mg_coarse_hypre_tol",
            "0.0",
            "--mg_coarse_hypre_relax_type_all",
            "symmetric-SOR/Jacobi",
            "--accept_ksp_maxit_direction",
            "--guard_ksp_maxit_direction",
            "--ksp_maxit_direction_true_rel_cap",
            "5e-2",
            "--no-reuse_hessian_value_buffers",
        ]
    )

    assert args.assembly_mode == "element"
    assert args.operator_mode == "matfree_overlap"
    assert args.local_hessian_mode == "sfd_local"
    assert args.use_trust_region is True
    assert args.trust_radius_init == 0.5
    assert args.trust_shrink == 0.5
    assert args.trust_expand == 1.5
    assert args.trust_eta_shrink == 0.05
    assert args.trust_eta_expand == 0.75
    assert args.trust_max_reject == 6
    assert args.trust_subproblem_line_search is True
    assert args.state_out == "state.npz"
    assert args.preconditioner_operator == "refined_p1_same_nodes"
    assert args.mg_coarsest_level == 2
    assert args.elem_degree == 4
    assert args.mg_strategy == "same_mesh_p4_p2_p1_lminus1_p1"
    assert args.mg_variant == "outer_pcksp"
    assert args.fine_pmat_policy == "elastic_frozen"
    assert args.fine_pmat_stagger_period == 2
    assert args.mg_lower_operator_policy == "fixed_setup"
    assert args.mg_p4_smoother_ksp_type == "chebyshev"
    assert args.mg_p4_smoother_pc_type == "jacobi"
    assert args.mg_p4_smoother_steps == 4
    assert args.mg_p2_smoother_ksp_type == "richardson"
    assert args.mg_p2_smoother_pc_type == "sor"
    assert args.mg_p2_smoother_steps == 3
    assert args.mg_p1_smoother_ksp_type == "richardson"
    assert args.mg_p1_smoother_pc_type == "jacobi"
    assert args.mg_p1_smoother_steps == 2
    assert args.mg_fine_ksp_type == "gmres"
    assert args.mg_degree2_pc_type == "sor"
    assert args.pc_reuse_preconditioner is True
    assert args.mg_coarse_backend == "hypre"
    assert args.mg_coarse_hypre_nodal_coarsen == 6
    assert args.mg_coarse_hypre_vec_interp_variant == 3
    assert args.mg_coarse_hypre_strong_threshold == 0.45
    assert args.mg_coarse_hypre_coarsen_type == "HMIS"
    assert args.hypre_nodal_coarsen == 6
    assert args.hypre_vec_interp_variant == 3
    assert args.hypre_strong_threshold == 0.5
    assert args.hypre_coarsen_type == "HMIS"
    assert args.hypre_max_iter == 3
    assert args.hypre_tol == 0.0
    assert args.hypre_relax_type_all == "symmetric-SOR/Jacobi"
    assert args.line_search == "armijo"
    assert args.armijo_alpha0 == 0.75
    assert args.armijo_c1 == 1.0e-5
    assert args.armijo_shrink == 0.6
    assert args.armijo_max_ls == 18
    assert args.benchmark_mode == "warmup_once_then_solve"
    assert args.mg_coarse_hypre_max_iter == 4
    assert args.mg_coarse_hypre_tol == 0.0
    assert args.mg_coarse_hypre_relax_type_all == "symmetric-SOR/Jacobi"
    assert args.accept_ksp_maxit_direction is True
    assert args.guard_ksp_maxit_direction is True
    assert args.ksp_maxit_direction_true_rel_cap == 5.0e-2
    assert args.reuse_hessian_value_buffers is False


def test_slope_stability_petsc_cli_accepts_custom_mixed_hierarchy():
    parser = solve_slope_stability_dof._build_parser({"reference": {}, "performance": {}})
    args = parser.parse_args(
        [
            "--elem_degree",
            "4",
            "--pc_type",
            "mg",
            "--mg_strategy",
            "custom_mixed",
            "--mg_custom_hierarchy",
            "1:1,2:1,6:2,6:4",
        ]
    )

    assert args.mg_strategy == "custom_mixed"
    assert args.mg_custom_hierarchy == "1:1,2:1,6:2,6:4"


def test_parse_custom_mg_hierarchy_specs():
    specs = multigrid.parse_custom_mg_hierarchy_specs("L1P1, 2:1, L6P2, 6:4")
    assert [(spec.level, spec.degree) for spec in specs] == [(1, 1), (2, 1), (6, 2), (6, 4)]


def test_same_mesh_hdf5_asset_matches_level2_p4_builder():
    path = ensure_same_mesh_case_hdf5(2, 4)
    assert path.exists()

    asset_case = build_same_mesh_lagrange_case_data(case_name_for_level(2), degree=4)
    procedural_case = _build_same_mesh_lagrange_case_data_impl(
        case_name_for_level(2),
        degree=4,
    )

    assert asset_case.case_name == procedural_case.case_name
    assert asset_case.level == procedural_case.level
    np.testing.assert_allclose(asset_case.nodes, procedural_case.nodes)
    np.testing.assert_array_equal(asset_case.elems_scalar, procedural_case.elems_scalar)
    np.testing.assert_array_equal(asset_case.freedofs, procedural_case.freedofs)
    np.testing.assert_allclose(asset_case.elem_B, procedural_case.elem_B)
    np.testing.assert_allclose(asset_case.quad_weight, procedural_case.quad_weight)
    np.testing.assert_allclose(asset_case.force, procedural_case.force)


def _same_mesh_p4_level2_params() -> tuple[dict[str, object], object]:
    case_data = build_same_mesh_lagrange_case_data(case_name_for_level(2), degree=4)
    params = dict(case_data.__dict__)
    params["elastic_kernel"] = build_near_nullspace_modes(
        np.asarray(case_data.nodes, dtype=np.float64),
        np.asarray(case_data.freedofs, dtype=np.int64),
    )
    cohesion, phi_deg = davis_b_reduction(
        float(params["c0"]),
        float(params["phi_deg"]),
        float(params["psi_deg"]),
        1.0,
    )
    params["cohesion"] = float(cohesion)
    params["phi_deg"] = float(phi_deg)
    params["reg"] = 1.0e-12
    params["elem_type"] = "P4"
    params["element_degree"] = 4
    return params, case_data.adjacency


def _p4_level2_frozen_pmat_command(output_path: Path, policy: str) -> list[str]:
    return [
        "mpiexec",
        "-n",
        "1",
        str(PYTHON),
        "-u",
        PETSC_SOLVER,
        "--level",
        "2",
        "--elem_degree",
        "4",
        "--lambda-target",
        "1.0",
        "--profile",
        "performance",
        "--pc_type",
        "mg",
        "--ksp_type",
        "fgmres",
        "--ksp_rtol",
        "1e-2",
        "--ksp_max_it",
        "100",
        "--operator_mode",
        "matfree_overlap",
        "--mg_strategy",
        "same_mesh_p4_p2_p1",
        "--mg_variant",
        "legacy_pmg",
        "--fine_pmat_policy",
        str(policy),
        "--save-linear-timing",
        "--quiet",
        "--no-use_trust_region",
        "--out",
        str(output_path),
    ]


def _p4_level2_staggered_pmat_command(
    output_path: Path,
    *,
    policy: str,
    mg_variant: str,
    mg_lower_operator_policy: str | None = None,
    mg_fine_ksp_type: str | None = None,
    mg_fine_pc_type: str | None = None,
    mg_fine_steps: int | None = None,
    mg_degree2_pc_type: str | None = None,
    mg_degree1_pc_type: str | None = None,
) -> list[str]:
    command = [
        "mpiexec",
        "-n",
        "1",
        str(PYTHON),
        "-u",
        PETSC_SOLVER,
        "--level",
        "2",
        "--elem_degree",
        "4",
        "--lambda-target",
        "1.0",
        "--profile",
        "performance",
        "--pc_type",
        "mg",
        "--ksp_type",
        "fgmres",
        "--ksp_rtol",
        "1e-2",
        "--ksp_max_it",
        "100",
        "--operator_mode",
        "matfree_overlap",
        "--mg_strategy",
        "same_mesh_p4_p2_p1",
        "--mg_variant",
        str(mg_variant),
        "--fine_pmat_policy",
        str(policy),
        "--fine_pmat_stagger_period",
        "2",
        "--save-linear-timing",
        "--quiet",
        "--no-use_trust_region",
        "--out",
        str(output_path),
    ]
    if str(policy) == "staggered_smoother_only":
        command.extend(
            [
                "--mg_lower_operator_policy",
                str(mg_lower_operator_policy or "fixed_setup"),
                "--mg_fine_ksp_type",
                str(mg_fine_ksp_type or "chebyshev"),
                "--mg_fine_pc_type",
                str(mg_fine_pc_type or "jacobi"),
                "--mg_fine_steps",
                str(mg_fine_steps or 4),
                "--mg_degree2_pc_type",
                str(mg_degree2_pc_type or "sor"),
                "--mg_degree1_pc_type",
                str(mg_degree1_pc_type or "jacobi"),
            ]
        )
    return command


def _csr_snapshot(mat: PETSc.Mat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indptr, indices, data = mat.getValuesCSR()
    return (
        np.asarray(indptr, dtype=np.int64).copy(),
        np.asarray(indices, dtype=np.int64).copy(),
        np.asarray(data, dtype=np.float64).copy(),
    )


def test_slope_stability_matrixfree_matvec_matches_assembled_p4_level2():
    params, adjacency = _same_mesh_p4_level2_params()
    assembler = SlopeStabilityReorderedElementAssembler(
        params=params,
        comm=MPI.COMM_SELF,
        adjacency=adjacency,
        ksp_rtol=1.0e-2,
        ksp_type="cg",
        pc_type="jacobi",
        ksp_max_it=20,
        use_near_nullspace=False,
        reorder_mode="block_xyz",
        local_hessian_mode="element",
    )
    try:
        rng = np.random.default_rng(12)
        u_full = 1.0e-3 * rng.standard_normal(assembler.layout.n_free)
        u_owned = np.asarray(u_full[assembler.layout.lo : assembler.layout.hi], dtype=np.float64)
        trial = assembler.create_vec(u_full)
        assembled_out = assembler.create_vec()
        elem_out = assembler.create_vec()
        overlap_out = assembler.create_vec()

        assembler.assemble_hessian(u_owned)
        assembler.A.mult(trial, assembled_out)

        elem_op = assembler.prepare_matrix_free_operator(u_owned, mode="matfree_element")
        elem_op.mult(trial, elem_out)

        overlap_op = assembler.prepare_matrix_free_operator(u_owned, mode="matfree_overlap")
        overlap_op.mult(trial, overlap_out)

        assert np.allclose(elem_out.array, assembled_out.array, atol=1.0e-8, rtol=1.0e-8)
        assert np.allclose(overlap_out.array, assembled_out.array, atol=1.0e-8, rtol=1.0e-8)
    finally:
        trial.destroy()
        assembled_out.destroy()
        elem_out.destroy()
        overlap_out.destroy()
        assembler.cleanup()


def test_slope_stability_matrixfree_diagonal_matches_assembled_p4_level2():
    params, adjacency = _same_mesh_p4_level2_params()
    assembler = SlopeStabilityReorderedElementAssembler(
        params=params,
        comm=MPI.COMM_SELF,
        adjacency=adjacency,
        ksp_rtol=1.0e-2,
        ksp_type="cg",
        pc_type="jacobi",
        ksp_max_it=20,
        use_near_nullspace=False,
        reorder_mode="block_xyz",
        local_hessian_mode="element",
    )
    trial = None
    assembled_diag = None
    shell_diag = None
    try:
        rng = np.random.default_rng(21)
        u_full = 1.0e-3 * rng.standard_normal(assembler.layout.n_free)
        u_owned = np.asarray(u_full[assembler.layout.lo : assembler.layout.hi], dtype=np.float64)
        trial = assembler.create_vec(u_full)
        assembled_diag = assembler.create_vec()
        shell_diag = assembler.create_vec()

        assembler.assemble_hessian(u_owned)
        assembler.A.getDiagonal(assembled_diag)

        overlap_op = assembler.prepare_matrix_free_operator(u_owned, mode="matfree_overlap")
        overlap_op.getDiagonal(shell_diag)

        assert np.allclose(shell_diag.array, assembled_diag.array, atol=1.0e-8, rtol=1.0e-8)
    finally:
        if trial is not None:
            trial.destroy()
        if assembled_diag is not None:
            assembled_diag.destroy()
        if shell_diag is not None:
            shell_diag.destroy()
        assembler.cleanup()


def test_slope_stability_elastic_and_initial_tangent_p4_matrices_match_at_zero_level2():
    params, adjacency = _same_mesh_p4_level2_params()
    assembler = SlopeStabilityReorderedElementAssembler(
        params=params,
        comm=MPI.COMM_SELF,
        adjacency=adjacency,
        ksp_rtol=1.0e-2,
        ksp_type="cg",
        pc_type="none",
        ksp_max_it=20,
        use_near_nullspace=False,
        reorder_mode="block_xyz",
        local_hessian_mode="element",
    )
    try:
        u_owned = np.zeros(assembler.part.n_owned, dtype=np.float64)
        assembler.assemble_hessian_with_mode(u_owned, constitutive_mode="elastic")
        elastic_csr = _csr_snapshot(assembler.A)

        assembler.assemble_hessian(u_owned)
        initial_csr = _csr_snapshot(assembler.A)

        assert np.array_equal(elastic_csr[0], initial_csr[0])
        assert np.array_equal(elastic_csr[1], initial_csr[1])
        assert np.allclose(elastic_csr[2], initial_csr[2], atol=1.0e-10, rtol=1.0e-10)
    finally:
        assembler.cleanup()


def test_slope_stability_petsc_element_and_local_sfd_agree_on_level1(tmp_path: Path):
    element_out = tmp_path / "element.json"
    sfd_out = tmp_path / "sfd.json"
    element = _run_json(_petsc_command(element_out, local_hessian_mode="element"), element_out)
    sfd = _run_json(_petsc_command(sfd_out, local_hessian_mode="sfd_local"), sfd_out)

    e_step = element["result"]["steps"][0]
    s_step = sfd["result"]["steps"][0]
    assert element["result"]["solver_success"] is True
    assert sfd["result"]["solver_success"] is True
    assert np.isclose(float(e_step["energy"]), float(s_step["energy"]), atol=1.0e-10)
    assert np.isclose(float(e_step["omega"]), float(s_step["omega"]), atol=1.0e-10)
    assert np.isclose(float(e_step["u_max"]), float(s_step["u_max"]), atol=1.0e-12)


def test_slope_stability_serial_jax_and_petsc_agree_on_level1(tmp_path: Path):
    petsc_out = tmp_path / "petsc.json"
    serial_out = tmp_path / "serial.json"
    petsc = _run_json(_petsc_command(petsc_out, local_hessian_mode="element"), petsc_out)
    serial = _run_json(
        [
            str(PYTHON),
            "-u",
            SERIAL_SOLVER,
            "--level",
            "1",
            "--lambda-target",
            "1.2",
            "--quiet",
            "--json",
            str(serial_out),
        ],
        serial_out,
    )

    petsc_step = petsc["result"]["steps"][0]
    serial_result = serial["result"]
    assert petsc["result"]["solver_success"] is True
    assert serial["result"]["solver_success"] is True
    assert np.isclose(
        float(petsc_step["energy"]),
        float(serial_result["final_energy"]),
        atol=5.0e-4,
    )
    assert np.isclose(float(petsc_step["omega"]), float(serial_result["omega"]), atol=1.0e-2)
    assert np.isclose(float(petsc_step["u_max"]), float(serial_result["u_max"]), atol=5.0e-4)


def test_slope_stability_petsc_mpi_smoke_level1_np2(tmp_path: Path):
    output = tmp_path / "mpi2.json"
    payload = _run_json(_petsc_command(output, nprocs=2), output)
    step = payload["result"]["steps"][0]
    assert payload["metadata"]["nprocs"] == 2
    assert payload["result"]["solver_success"] is True
    assert payload["result"]["status"] == "completed"
    assert np.isfinite(float(step["energy"]))
    assert np.isfinite(float(step["omega"]))
    assert np.isfinite(float(step["u_max"]))


def test_slope_stability_refined_p1_petsc_smoke_level1(tmp_path: Path):
    output = tmp_path / "refined_p1.json"
    payload = _run_json(
        [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_REFINED_P1_SOLVER,
            "--level",
            "1",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "hypre",
            "--quiet",
            "--out",
            str(output),
        ],
        output,
    )
    step = payload["result"]["steps"][0]
    assert payload["case"]["elem_type"] == "P1_refined_same_nodes"
    assert payload["result"]["solver_success"] is True
    assert np.isfinite(float(step["energy"]))
    assert np.isfinite(float(step["omega"]))
    assert np.isfinite(float(step["u_max"]))


def test_slope_stability_p2_with_refined_p1_preconditioner_level1(tmp_path: Path):
    baseline_out = tmp_path / "baseline.json"
    mixed_out = tmp_path / "mixed_pc.json"
    baseline = _run_json(
        [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "1",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "hypre",
            "--quiet",
            "--out",
            str(baseline_out),
        ],
        baseline_out,
    )
    mixed = _run_json(
        [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "1",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "hypre",
            "--preconditioner_operator",
            "refined_p1_same_nodes",
            "--quiet",
            "--out",
            str(mixed_out),
        ],
        mixed_out,
    )
    base_step = baseline["result"]["steps"][0]
    mixed_step = mixed["result"]["steps"][0]
    assert mixed["result"]["solver_success"] is True
    assert mixed["case"]["elem_type"] == "P2"
    assert mixed["metadata"]["linear_solver"]["preconditioner_operator"] == "refined_p1_same_nodes"
    assert mixed["metadata"]["linear_solver"]["preconditioner_elem_type"] == "P1_refined_same_nodes"
    assert np.isclose(float(base_step["energy"]), float(mixed_step["energy"]), atol=1.0e-6)
    assert np.isclose(float(base_step["omega"]), float(mixed_step["omega"]), atol=1.0e-6)
    assert np.isclose(float(base_step["u_max"]), float(mixed_step["u_max"]), atol=1.0e-8)


def test_slope_stability_petsc_mg_smoke_level2(tmp_path: Path):
    output = tmp_path / "mg_level2.json"
    payload = _run_json(
        [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "2",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "mg",
            "--ksp_type",
            "fgmres",
            "--ksp_rtol",
            "1e-2",
            "--ksp_max_it",
            "50",
            "--quiet",
            "--out",
            str(output),
        ],
        output,
    )
    step = payload["result"]["steps"][0]
    assert payload["result"]["solver_success"] is True
    assert payload["metadata"]["linear_solver"]["pc_type"] == "mg"
    assert payload["metadata"]["linear_solver"]["mg_coarsest_level"] == 1
    assert payload["metadata"]["linear_solver"]["mg_coarse_backend"] == "hypre"
    assert payload["metadata"]["linear_solver"]["mg_coarse_hypre_max_iter"] >= 2
    assert payload["metadata"]["linear_solver"]["mg_coarse_hypre_relax_type_all"] == "symmetric-SOR/Jacobi"
    assert np.isfinite(float(step["energy"]))
    assert np.isfinite(float(step["omega"]))
    assert np.isfinite(float(step["u_max"]))


def test_slope_stability_petsc_mg_smoke_level3_coarsest2(tmp_path: Path):
    output = tmp_path / "mg_level3_c2.json"
    payload = _run_json(
        [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "3",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "mg",
            "--ksp_type",
            "fgmres",
            "--ksp_rtol",
            "1e-2",
            "--ksp_max_it",
            "50",
            "--mg_coarsest_level",
            "2",
            "--quiet",
            "--out",
            str(output),
        ],
        output,
    )
    assert payload["result"]["solver_success"] is True
    assert payload["metadata"]["linear_solver"]["mg_coarsest_level"] == 2


def test_slope_stability_petsc_p4_mixed_mg_smoke_level2(tmp_path: Path):
    output = tmp_path / "mg_level2_p4.json"
    payload = _run_json(
        [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "2",
            "--elem_degree",
            "4",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "mg",
            "--ksp_type",
            "fgmres",
            "--ksp_rtol",
            "1e-2",
            "--ksp_max_it",
            "100",
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--save-linear-timing",
            "--no-use_trust_region",
            "--quiet",
            "--out",
            str(output),
        ],
        output,
    )
    step = payload["result"]["steps"][0]
    assert payload["case"]["elem_type"] == "P4"
    assert payload["case"]["element_degree"] == 4
    assert payload["result"]["solver_success"] is True
    assert payload["metadata"]["linear_solver"]["mg_strategy"] == "same_mesh_p4_p2_p1_lminus1_p1"
    assert payload["metadata"]["linear_solver"]["mg_level_metadata"]
    assert payload["metadata"]["linear_solver"]["mg_hierarchy_levels"]
    assert payload["metadata"]["linear_solver"]["mg_hierarchy_transfers"]
    assert payload["metadata"]["linear_solver"]["mg_legacy_level_smoothers"]["fine"]["pc_type"] == "sor"
    assert step["linear_timing"][0]["mg_runtime_diagnostics"]
    assert np.isfinite(float(step["energy"]))
    assert np.isfinite(float(step["omega"]))
    assert np.isfinite(float(step["u_max"]))


def test_slope_stability_petsc_explicit_pmg_transfer_diagnostics_level2(tmp_path: Path):
    output = tmp_path / "mg_level2_p4_explicit_transfer_diag.json"
    payload = _run_json(
        [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "2",
            "--elem_degree",
            "4",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "mg",
            "--mg_variant",
            "explicit_pmg",
            "--mg_strategy",
            "same_mesh_p4_p2_p1_lminus1_p1",
            "--mg_lower_operator_policy",
            "galerkin_refresh",
            "--ksp_type",
            "fgmres",
            "--ksp_rtol",
            "1e-2",
            "--ksp_max_it",
            "100",
            "--mg_fine_ksp_type",
            "richardson",
            "--mg_fine_pc_type",
            "sor",
            "--mg_fine_steps",
            "2",
            "--mg_intermediate_steps",
            "2",
            "--mg_degree2_pc_type",
            "sor",
            "--mg_degree1_pc_type",
            "sor",
            "--save-linear-timing",
            "--no-use_trust_region",
            "--quiet",
            "--out",
            str(output),
        ],
        output,
    )
    step = payload["result"]["steps"][0]
    diagnostics = step["linear_timing"][0]["mg_runtime_diagnostics"]
    transfer_roles = {entry["sweep_role"] for entry in diagnostics if entry.get("kind") == "transfer"}
    assert payload["result"]["solver_success"] is True
    assert transfer_roles == {"prolongation", "restriction"}


def test_slope_stability_petsc_p4_matrixfree_modes_level2_hypre(tmp_path: Path):
    assembled_out = tmp_path / "hypre_level2_p4_assembled.json"
    elem_out = tmp_path / "hypre_level2_p4_matfree_element.json"
    overlap_out = tmp_path / "hypre_level2_p4_matfree_overlap.json"

    def _cmd(path: Path, mode: str) -> list[str]:
        command = [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "2",
            "--elem_degree",
            "4",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "hypre",
            "--ksp_type",
            "cg",
            "--ksp_rtol",
            "1e-2",
            "--ksp_max_it",
            "50",
            "--operator_mode",
            mode,
            "--quiet",
            "--out",
            str(path),
        ]
        return command

    assembled = _run_json(_cmd(assembled_out, "assembled"), assembled_out)
    elem = _run_json(_cmd(elem_out, "matfree_element"), elem_out)
    overlap = _run_json(_cmd(overlap_out, "matfree_overlap"), overlap_out)

    assembled_step = assembled["result"]["steps"][0]
    elem_step = elem["result"]["steps"][0]
    overlap_step = overlap["result"]["steps"][0]

    assert elem["result"]["solver_success"] is True
    assert overlap["result"]["solver_success"] is True
    assert assembled["result"]["solver_success"] is True
    assert elem["metadata"]["linear_solver"]["operator_mode"] == "matfree_element"
    assert overlap["metadata"]["linear_solver"]["operator_mode"] == "matfree_overlap"
    assert np.isclose(float(elem_step["energy"]), float(assembled_step["energy"]), atol=1.0e-6)
    assert np.isclose(float(overlap_step["energy"]), float(assembled_step["energy"]), atol=1.0e-6)
    assert np.isclose(float(elem_step["omega"]), float(assembled_step["omega"]), atol=1.0e-6)
    assert np.isclose(float(overlap_step["omega"]), float(assembled_step["omega"]), atol=1.0e-6)
    assert np.isclose(float(elem_step["u_max"]), float(assembled_step["u_max"]), atol=1.0e-8)
    assert np.isclose(float(overlap_step["u_max"]), float(assembled_step["u_max"]), atol=1.0e-8)


def test_slope_stability_petsc_p4_matrixfree_pmg_uses_fixed_lower_levels(tmp_path: Path):
    output = tmp_path / "mg_level2_p4_matfree.json"
    payload = _run_json(
        [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "2",
            "--elem_degree",
            "4",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "mg",
            "--operator_mode",
            "matfree_overlap",
            "--mg_strategy",
            "same_mesh_p4_p2_p1",
            "--mg_variant",
            "explicit_pmg",
            "--mg_lower_operator_policy",
            "fixed_setup",
            "--maxit",
            "1",
            "--save-linear-timing",
            "--quiet",
            "--out",
            str(output),
        ],
        output,
    )
    assert payload["metadata"]["linear_solver"]["pc_type"] == "mg"
    assert payload["metadata"]["linear_solver"]["operator_mode"] == "matfree_overlap"
    assert payload["metadata"]["linear_solver"]["ksp_type"] == "fgmres"
    assert (
        payload["metadata"]["linear_solver"]["mg_operator_policy"]
        == "explicit_pmg_matrixfree_fine_fixed_setup"
    )
    step = payload["result"]["steps"][0]
    record = step["linear_timing"][0]
    assert step["nit"] == 1
    assert record["assemble_total_time"] == 0.0
    assert record["pc_operator_assemble_total_time"] == 0.0
    assert record["operator_prepare_total_time"] > 0.0
    assert record["ksp_reason_name"] == "DIVERGED_MAX_IT"
    assert record["true_relative_residual"] > 0.0
    level_records = record["pc_operator_mg_level_records"]
    assert [entry["degree"] for entry in level_records] == [1, 2]
    assert all(entry["operator_source"] == "fixed_setup" for entry in level_records)


def test_slope_stability_petsc_explicit_pmg_control_records_diagnostics(tmp_path: Path):
    output = tmp_path / "mg_level2_p4_explicit_control.json"
    payload = _run_json(
        [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "2",
            "--elem_degree",
            "4",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "mg",
            "--operator_mode",
            "assembled",
            "--mg_strategy",
            "same_mesh_p4_p2_p1",
            "--mg_variant",
            "explicit_pmg",
            "--mg_lower_operator_policy",
            "refresh_each_newton",
            "--maxit",
            "1",
            "--save-linear-timing",
            "--quiet",
            "--out",
            str(output),
        ],
        output,
    )
    step = payload["result"]["steps"][0]
    record = step["linear_timing"][0]
    assert payload["metadata"]["linear_solver"]["mg_variant"] == "explicit_pmg"
    assert payload["metadata"]["linear_solver"]["mg_lower_operator_policy"] == "refresh_each_newton"
    assert step["linear_summary"]["n_solves"] == 1
    assert "ksp_reason_name" in record
    assert "true_relative_residual" in record
    assert record["pc_operator_assemble_total_time"] > 0.0
    level_records = record["pc_operator_mg_level_records"]
    assert [entry["degree"] for entry in level_records] == [1, 2]
    assert all(entry["operator_source"] == "refresh_each_newton" for entry in level_records)


def test_slope_stability_petsc_p4_matrixfree_explicit_pmg_fixed_setup_succeeds_level2(tmp_path: Path):
    output = tmp_path / "mg_level2_p4_matfree_fixed_success.json"
    payload = _run_json(
        [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "2",
            "--elem_degree",
            "4",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "mg",
            "--operator_mode",
            "matfree_overlap",
            "--mg_strategy",
            "same_mesh_p4_p2_p1",
            "--mg_variant",
            "explicit_pmg",
            "--mg_lower_operator_policy",
            "fixed_setup",
            "--mg_degree2_pc_type",
            "sor",
            "--mg_degree1_pc_type",
            "jacobi",
            "--mg_intermediate_steps",
            "4",
            "--mg_fine_steps",
            "4",
            "--mg_fine_ksp_type",
            "fgmres",
            "--ksp_type",
            "fgmres",
            "--ksp_rtol",
            "1e-2",
            "--ksp_max_it",
            "100",
            "--save-linear-timing",
            "--quiet",
            "--no-use_trust_region",
            "--out",
            str(output),
        ],
        output,
    )
    step = payload["result"]["steps"][0]
    summary = step["linear_summary"]
    assert payload["result"]["solver_success"] is True
    assert payload["result"]["status"] == "completed"
    assert payload["metadata"]["linear_solver"]["mg_operator_policy"] == "explicit_pmg_matrixfree_fine_fixed_setup"
    assert summary["all_converged"] is True
    assert summary["n_failed"] == 0
    assert summary["worst_true_relative_residual"] < 1.0e-2


def test_slope_stability_petsc_p4_matrixfree_explicit_pmg_chebyshev_jacobi_level2(tmp_path: Path):
    output = tmp_path / "mg_level2_p4_matfree_chebyshev.json"
    payload = _run_json(
        [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "2",
            "--elem_degree",
            "4",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "mg",
            "--operator_mode",
            "matfree_overlap",
            "--mg_strategy",
            "same_mesh_p4_p2_p1",
            "--mg_variant",
            "explicit_pmg",
            "--mg_lower_operator_policy",
            "fixed_setup",
            "--mg_fine_ksp_type",
            "chebyshev",
            "--mg_fine_pc_type",
            "jacobi",
            "--mg_fine_steps",
            "4",
            "--mg_degree2_pc_type",
            "jacobi",
            "--mg_degree1_pc_type",
            "jacobi",
            "--ksp_type",
            "fgmres",
            "--ksp_rtol",
            "1e-2",
            "--ksp_max_it",
            "100",
            "--save-linear-timing",
            "--quiet",
            "--no-use_trust_region",
            "--out",
            str(output),
        ],
        output,
    )
    record = payload["result"]["steps"][0]["linear_timing"][0]
    assert payload["metadata"]["linear_solver"]["mg_fine_down"]["ksp_type"] == "chebyshev"
    assert payload["metadata"]["linear_solver"]["mg_fine_down"]["pc_type"] == "jacobi"
    assert record["operator_diagonal_source"] == "element_hessian_diagonal"
    assert record["assemble_total_time"] == 0.0


def test_slope_stability_petsc_p4_matrixfree_python_pc_overlap_lu_level2(tmp_path: Path):
    output = tmp_path / "p4_level2_python_pc_overlap_lu.json"
    payload = _run_json(
        [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "2",
            "--elem_degree",
            "4",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "python",
            "--python_pc_variant",
            "overlap_lu",
            "--operator_mode",
            "matfree_overlap",
            "--ksp_type",
            "fgmres",
            "--ksp_rtol",
            "1e-2",
            "--ksp_max_it",
            "50",
            "--save-linear-timing",
            "--quiet",
            "--no-use_trust_region",
            "--out",
            str(output),
        ],
        output,
    )
    step = payload["result"]["steps"][0]
    record = step["linear_timing"][0]
    assert payload["result"]["solver_success"] is True
    assert payload["metadata"]["linear_solver"]["pc_type"] == "python"
    assert payload["metadata"]["linear_solver"]["python_pc_variant"] == "overlap_lu"
    assert record["python_pc_variant"] == "overlap_lu"
    assert record["assemble_total_time"] == 0.0
    assert record["python_pc_prepare_total_time"] > 0.0


def test_slope_stability_petsc_p4_matrixfree_legacy_pmg_elastic_frozen_level2(tmp_path: Path):
    output = tmp_path / "p4_level2_legacy_elastic_frozen.json"
    payload = _run_json(
        _p4_level2_frozen_pmat_command(output, "elastic_frozen"),
        output,
    )
    step = payload["result"]["steps"][0]
    assert payload["result"]["solver_success"] is True
    assert payload["metadata"]["linear_solver"]["mg_variant"] == "legacy_pmg"
    assert payload["metadata"]["linear_solver"]["fine_pmat_policy"] == "elastic_frozen"
    assert payload["metadata"]["linear_solver"]["fine_pmat_setup_assembly_time"] > 0.0
    assert step["linear_summary"]["all_converged"] is True
    assert all(
        float(record["assemble_total_time"]) == 0.0
        and float(record["fine_pmat_step_assembly_time"]) == 0.0
        for record in step["linear_timing"]
    )


def test_slope_stability_petsc_p4_matrixfree_legacy_pmg_initial_tangent_frozen_level2(tmp_path: Path):
    output = tmp_path / "p4_level2_legacy_initial_frozen.json"
    payload = _run_json(
        _p4_level2_frozen_pmat_command(output, "initial_tangent_frozen"),
        output,
    )
    step = payload["result"]["steps"][0]
    assert payload["result"]["solver_success"] is True
    assert payload["metadata"]["linear_solver"]["mg_variant"] == "legacy_pmg"
    assert payload["metadata"]["linear_solver"]["fine_pmat_policy"] == "initial_tangent_frozen"
    assert payload["metadata"]["linear_solver"]["fine_pmat_setup_assembly_time"] > 0.0
    assert step["linear_summary"]["all_converged"] is True
    assert all(
        float(record["assemble_total_time"]) == 0.0
        and float(record["fine_pmat_step_assembly_time"]) == 0.0
        for record in step["linear_timing"]
    )


def test_slope_stability_petsc_p4_matrixfree_legacy_pmg_staggered_whole_level2(tmp_path: Path):
    output = tmp_path / "p4_level2_legacy_staggered_whole.json"
    payload = _run_json(
        _p4_level2_staggered_pmat_command(
            output,
            policy="staggered_whole",
            mg_variant="legacy_pmg",
        ),
        output,
    )
    step = payload["result"]["steps"][0]
    records = step["linear_timing"]
    assert payload["result"]["solver_success"] is True
    assert payload["metadata"]["linear_solver"]["fine_pmat_policy"] == "staggered_whole"
    assert payload["metadata"]["linear_solver"]["fine_pmat_stagger_period"] == 2
    assert all(float(record["assemble_total_time"]) == 0.0 for record in records)
    assert any(float(record["fine_pmat_step_assembly_time"]) > 0.0 for record in records)
    for idx, record in enumerate(records, start=1):
        expect_update = (idx % 2) == 1
        assert bool(record["fine_pmat_updated_this_step"]) is expect_update
        if expect_update:
            assert float(record["fine_pmat_step_assembly_time"]) > 0.0
        else:
            assert float(record["fine_pmat_step_assembly_time"]) == 0.0


def test_slope_stability_petsc_p4_matrixfree_explicit_pmg_staggered_smoother_level2(tmp_path: Path):
    output = tmp_path / "p4_level2_explicit_staggered_smoother.json"
    payload = _run_json(
        _p4_level2_staggered_pmat_command(
            output,
            policy="staggered_smoother_only",
            mg_variant="explicit_pmg",
            mg_lower_operator_policy="fixed_setup",
            mg_fine_ksp_type="chebyshev",
            mg_fine_pc_type="jacobi",
            mg_fine_steps=4,
            mg_degree2_pc_type="sor",
            mg_degree1_pc_type="jacobi",
        ),
        output,
    )
    step = payload["result"]["steps"][0]
    records = step["linear_timing"]
    assert payload["result"]["solver_success"] is True
    assert payload["metadata"]["linear_solver"]["fine_pmat_policy"] == "staggered_smoother_only"
    assert payload["metadata"]["linear_solver"]["fine_pmat_stagger_period"] == 2
    assert payload["metadata"]["linear_solver"]["mg_lower_operator_policy"] == "fixed_setup"
    assert all(float(record["assemble_total_time"]) == 0.0 for record in records)
    assert all(float(record["pc_operator_assemble_total_time"]) == 0.0 for record in records)
    for idx, record in enumerate(records, start=1):
        expect_update = (idx % 2) == 1
        assert bool(record["fine_pmat_updated_this_step"]) is expect_update
        if expect_update:
            assert float(record["fine_pmat_step_assembly_time"]) > 0.0
        else:
            assert float(record["fine_pmat_step_assembly_time"]) == 0.0


def test_slope_stability_petsc_p4_legacy_pmg_reuse_preconditioner_level2(tmp_path: Path):
    current_output = tmp_path / "p4_level2_legacy_current.json"
    reused_output = tmp_path / "p4_level2_legacy_reused.json"

    def _cmd(out: Path, *, reuse: bool) -> list[str]:
        cmd = [
            "mpiexec",
            "-n",
            "1",
            str(PYTHON),
            "-u",
            PETSC_SOLVER,
            "--level",
            "2",
            "--elem_degree",
            "4",
            "--lambda-target",
            "1.0",
            "--profile",
            "performance",
            "--pc_type",
            "mg",
            "--ksp_type",
            "fgmres",
            "--ksp_rtol",
            "1e-2",
            "--ksp_max_it",
            "100",
            "--mg_strategy",
            "same_mesh_p4_p2_p1",
            "--mg_variant",
            "legacy_pmg",
            "--save-linear-timing",
            "--quiet",
            "--no-use_trust_region",
            "--out",
            str(out),
        ]
        if reuse:
            cmd.append("--pc_reuse_preconditioner")
        return cmd

    current = _run_json(_cmd(current_output, reuse=False), current_output)
    reused = _run_json(_cmd(reused_output, reuse=True), reused_output)

    current_step = current["result"]["steps"][0]
    reused_step = reused["result"]["steps"][0]

    assert current["result"]["solver_success"] is True
    assert reused["result"]["solver_success"] is True
    assert current["metadata"]["linear_solver"]["pc_reuse_preconditioner"] is False
    assert reused["metadata"]["linear_solver"]["pc_reuse_preconditioner"] is True
    assert abs(float(current_step["omega"]) - float(reused_step["omega"])) < 1.0e-3
    assert abs(float(current_step["u_max"]) - float(reused_step["u_max"])) < 1.0e-4
