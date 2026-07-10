from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np

from experiments.analysis.collect_slurm_accounting import SACCT_FIELDS, parse_sacct


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments/analysis/analyze_plasticity3d_route_endpoints.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("route_endpoint_analysis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = _load_module()


FIELDS = (
    "experiment_id",
    "tier",
    "case_id",
    "comparison_id",
    "block_repetition",
    "route_order",
    "route_order_policy",
    "timing_reduction",
    "probe_count",
    "optional",
    "nodes",
    "ranks_per_node",
    "total_ranks",
    "repetitions",
    "warmups",
    "partition",
    "time_limit",
    "estimated_node_hours",
    "runner",
    "mesh_name",
    "element_degree",
    "quadrature_rule",
    "route",
    "state_label",
    "state_amplitude",
    "assembly_backend",
    "solver_backend",
    "pmg_strategy",
    "maxit",
    "ksp_rtol",
    "ksp_max_it",
    "stop_tol",
    "grad_stop_tol",
    "convergence_metric",
    "notes",
)


def _matrix_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tier, degree, rule, ranks_values, strategy, tag in (
        (
            "full_solve_confirmation",
            4,
            "tetra_24point",
            (8, 32),
            "same_mesh_p4_p2_p1",
            "p4l1",
        ),
        (
            "low_order_confirmation",
            1,
            "tetra_1point",
            (8,),
            "uniform_refined_p1_chain",
            "p1l1",
        ),
    ):
        for ranks in ranks_values:
            comparison_id = f"route_{tag}_np{ranks}"
            for repetition in range(1, 11):
                first = "element_ad" if repetition % 2 else "constitutive_ad"
                second = "constitutive_ad" if first == "element_ad" else "element_ad"
                rows.append(
                    {
                        "experiment_id": "EXP-ROUTE-001",
                        "tier": tier,
                        "case_id": f"{comparison_id}_block_{repetition:02d}",
                        "comparison_id": comparison_id,
                        "block_repetition": str(repetition),
                        "route_order": f"{first}|{second}",
                        "route_order_policy": "seeded_balanced_alternating_v1",
                        "timing_reduction": "mpi_collective_max",
                        "probe_count": "0",
                        "optional": "1",
                        "nodes": "1",
                        "ranks_per_node": str(ranks),
                        "total_ranks": str(ranks),
                        "repetitions": "1",
                        "warmups": "0",
                        "partition": "qcpu_exp",
                        "time_limit": "02:00:00",
                        "estimated_node_hours": "2.0",
                        "runner": "p3d_solve_block",
                        "mesh_name": "hetero_ssr_L1",
                        "element_degree": str(degree),
                        "quadrature_rule": rule,
                        "route": "paired_element_constitutive",
                        "state_label": "solver_initial_state",
                        "state_amplitude": "0.0",
                        "assembly_backend": "paired_block",
                        "solver_backend": "local_pmg",
                        "pmg_strategy": strategy,
                        "maxit": "80",
                        "ksp_rtol": "1e-8",
                        "ksp_max_it": "500",
                        "stop_tol": "0.002",
                        "grad_stop_tol": "0.0001",
                        "convergence_metric": "reference_elastic_energy",
                        "notes": "paired randomized route block",
                    }
                )
    return rows


def _write_matrix(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (
            "nodes",
            "elements_scalar",
            "material_id",
            "free_dofs",
            "free_mask",
            "free_dof_permutation",
            "shear_q",
            "bulk_q",
            "lame_q",
            "quad_weight",
        )
    }


def _output(row: dict[str, str], route: str) -> dict[str, object]:
    backend = {
        "element_ad": "local",
        "constitutive_ad": "local_constitutiveAD",
    }[route]
    mode = "element" if route == "element_ad" else "constitutive"
    hashes = _hashes()
    requested = {
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "rtol": 1.0e-10,
        "atol": 1.0e-14,
        "max_it": 1000,
        "true_residual_rtol": 1.0e-8,
        "spd_factor_solver_type": "mumps",
        "symmetry_relative_tolerance": 1.0e-12,
    }
    metric = {
        "name": "plasticity3d_reference_elastic_energy",
        "riesz_operator": "petsc_matrix",
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "requested_rtol": 1.0e-10,
        "requested_atol": 1.0e-14,
        "requested_max_it": 1000,
        "effective_rtol": 1.0e-10,
        "effective_atol": 1.0e-14,
        "effective_max_it": 1000,
        "true_residual_rtol_gate": 1.0e-8,
        "set_from_petsc_options": False,
        "petsc_options_prefix": "",
        "provenance": {
            "problem": "Plasticity3D",
            "operator_source": "elastic_tangent_at_zero_displacement",
            "constitutive_mode": "elastic",
            "reference_operator_tangent_mode": "element_ad_for_all_routes",
            "constraint_variant": "glued_bottom",
            "free_space": "glued_free_dofs",
            "ordering": "backend_mix_reordered_constrained_free_dofs",
            "ownership": "petsc_distributed_rows",
            "mesh_name": row["mesh_name"],
            "element_degree": int(row["element_degree"]),
            "quadrature_rule_id": row["quadrature_rule"],
            "free_dofs": 6,
            "matrix_type": "mpiaij",
            "matrix_nonzeros": 24,
            "assembly_backend": "coo_local",
            "backend_mix_route": backend,
            "autodiff_tangent_mode": mode,
            "local_hessian_mode": "element",
            "material_parameter_ranges": {
                "shear": {"minimum": 1.0, "maximum": 2.0},
                "bulk": {"minimum": 3.0, "maximum": 4.0},
                "lame": {"minimum": 5.0, "maximum": 6.0},
            },
            "input_identity": {
                "array_sha256": {
                    key: hashes[key]
                    for key in (
                        "nodes",
                        "elements_scalar",
                        "material_id",
                        "free_dofs",
                        "free_mask",
                        "free_dof_permutation",
                    )
                },
                "hdf5": {
                    "path": "/synthetic/shared_input.h5",
                    "size_bytes": 100,
                    "dataset_sha256": {
                        key: hashes[key]
                        for key in ("shear_q", "bulk_q", "lame_q", "quad_weight")
                    },
                },
                "tangent_route": {
                    "constitutive_mode": "elastic",
                    "autodiff_tangent_mode": "element",
                    "reference_operator_forced_common": True,
                    "solve_route_autodiff_tangent_mode": mode,
                    "local_hessian_mode": "element",
                    "assembly_backend": "coo_local",
                },
            },
            "spd_certificate": {
                "certified_spd": True,
                "method": "symmetric_direct_factorization_inertia",
                "symmetry_checked": True,
                "factor_solver_type": "mumps",
                "symmetry_relative_tolerance": 1.0e-12,
                "matrix_infinity_norm": 10.0,
                "symmetry_absolute_tolerance": 1.0e-11,
                "matrix_rows": 6,
                "matrix_columns": 6,
                "inertia": {"negative": 0, "zero": 0, "positive": 6},
            },
        },
    }
    last_riesz = {
        "riesz_solve": "iterative",
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "iterations": 3,
        "reason": 2,
        "rhs_norm": 2.5,
        "relative_true_residual": 2.0e-10,
        "requested_rtol": 1.0e-10,
        "requested_atol": 1.0e-14,
        "requested_max_it": 1000,
        "effective_rtol": 1.0e-10,
        "effective_atol": 1.0e-14,
        "effective_max_it": 1000,
        "true_residual_rtol_gate": 1.0e-8,
    }
    return {
        "status": "completed",
        "solver_success": True,
        "message": "Converged: all required criteria met",
        "assembly_backend": backend,
        "solver_backend": "local_pmg",
        "mesh_name": row["mesh_name"],
        "elem_degree": int(row["element_degree"]),
        "quadrature_rule_id": row["quadrature_rule"],
        "quadrature_points": 24 if int(row["element_degree"]) == 4 else 1,
        "constraint_variant": "glued_bottom",
        "pmg_strategy": row["pmg_strategy"],
        "ranks": int(row["total_ranks"]),
        "maxit": 80,
        "ksp_max_it": 500,
        "ksp_rtol": 1.0e-8,
        "stop_tol": 2.0e-3,
        "grad_stop_tol": 1.0e-4,
        "line_search": "armijo",
        "linesearch_tol": 1.0e-3,
        "use_trust_region": True,
        "trust_subproblem_line_search": True,
        "lambda_target": 1.55,
        "energy": 3.0,
        "omega": 2.0,
        "u_max": float(np.sqrt(0.14)),
        "total_time": 1.0,
        "total_time_reduction": "mpi_collective_max",
        "total_time_by_rank_s": [
            1.0 - 0.01 * index for index in range(int(row["total_ranks"]))
        ],
        "state_out": "state.npz",
        "git": {
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "dirty": False,
        },
        "job_metadata": {"slurm_job_id": "123", "slurm_cluster_name": "karolina"},
        "branch_diagnostics": {
            "definition": "mohr_coulomb_owned_quadrature_branch_v2",
            "owned_quadrature_points": 4,
            "counts": {
                "elastic": 2,
                "shear": 1,
                "left_edge": 1,
                "right_edge": 0,
                "apex": 0,
            },
            "normalized_boundary_margin_min": 0.25,
            "near_boundary_threshold": 1.0e-8,
            "near_boundary_fraction": 0.0,
            "canonical_map_definition": "global_element_id_then_quadrature_index_int8_v1",
            "canonical_map_sha256": hashlib.sha256(b"branch-map").hexdigest(),
        },
        "nit": 1,
        "linear_iterations_total": 7,
        "linear_history": [
            {
                "ksp_its": 7,
                "ksp_reason_code": 2,
                "ksp_reason_name": "CONVERGED_RTOL",
                "effective_ksp": {
                    "ksp_type": "fgmres",
                    "pc_type": "mg",
                    "options_prefix": "mix_newton_",
                    "rtol": 1.0e-8,
                    "atol": 1.0e-50,
                    "dtol": 1.0e5,
                    "max_it": 500,
                    "captured_after_set_from_options": True,
                },
            }
        ],
        "initial_guess": {
            "success": True,
            "ksp_iterations": 5,
            "ksp_reason_code": 2,
            "effective_ksp": {
                "ksp_type": "fgmres",
                "pc_type": "mg",
                "options_prefix": "mix_init_",
                "rtol": 1.0e-8,
                "atol": 1.0e-50,
                "dtol": 1.0e5,
                "max_it": 500,
                "captured_after_set_from_options": True,
            },
        },
        "convergence_metric_requested": "reference_elastic_energy",
        "convergence_metric": "reference_elastic_energy",
        "final_grad_norm": 2.5,
        "riesz_solver_requested": requested,
        "parallel_setup": {"owned_free_dofs_sum": 6},
        "nonlinear_convergence": {
            "configuration": {
                "selection": "reference_elastic_energy",
                "correction_normalization": "metric_current_state",
                "state_scale_source": "initial_nonlinear_iterate_primal_norm",
                "state_scale": 1.5,
            },
            "metric": metric,
            "initial_absolute_dual_residual": {"value": 1.0},
            "absolute_dual_residual": {"value": 5.0e-5},
            "state_norm": {"value": 2.0},
            "relative_correction": {"value": 1.0e-3},
            "coefficient_gradient_l2": 2.5,
            "last_riesz_solve": last_riesz,
            "residual_gate": {
                "absolute_tolerance": 1.0e-4,
                "effective_absolute_target": 1.0e-4,
                "passed": True,
            },
        },
    }


def _write_state(path: Path, row: dict[str, str], route: str, *, delta: float = 0.0) -> None:
    coords = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    requested_displacement = np.asarray(
        [[0.0, 0.0, 0.0], [0.1 + delta, 0.2, 0.3]]
    )
    final = coords + requested_displacement
    displacement = final - coords
    free_displacement = displacement.reshape(-1)
    elastic_action = free_displacement.copy()
    np.savez(
        path,
        coords_ref=coords,
        coords_final=final,
        displacement=displacement,
        free_displacement_reordered=free_displacement,
        reference_elastic_action=elastic_action,
        reference_elastic_state_quadratic=float(
            np.dot(free_displacement, elastic_action)
        ),
        tetrahedra=np.asarray([[0, 1, 1, 0]], dtype=np.int32),
        surface_faces=np.asarray([[0, 1, 1]], dtype=np.int32),
        boundary_label=np.asarray([1], dtype=np.int32),
        mesh_name=row["mesh_name"],
        element_degree=int(row["element_degree"]),
        quadrature_rule_id=row["quadrature_rule"],
        constraint_variant="glued_bottom",
        mpi_ranks=int(row["total_ranks"]),
        assembly_backend={
            "element_ad": "local",
            "constitutive_ad": "local_constitutiveAD",
        }[route],
        lambda_target=1.55,
        energy=3.0,
    )


def _write_job(
    root: Path,
    row: dict[str, str],
    *,
    returncode: int = 0,
    state_delta: float = 0.0,
    mutate_output=None,
    route_times: dict[str, float] | None = None,
) -> None:
    job = root / "cases" / row["case_id"] / "job_123"
    measure = job / "measure_01"
    measure.mkdir(parents=True)
    batch = root / "jobs" / row["case_id"] / "job_123"
    batch.mkdir(parents=True)
    slurm = root / "slurm"
    slurm.mkdir(exist_ok=True)
    (batch / "job_metadata.env").write_text(
        "\n".join(
            (
                f"case_id={row['case_id']}",
                "job_id=123",
                "account=fta-26-40",
                "qos=3571_6328",
                f"partition={row['partition']}",
                "cluster=karolina",
                f"nodes={row['nodes']}",
                f"ntasks={row['total_ranks']}",
                f"ntasks_per_node={row['ranks_per_node']}",
                f"tasks_per_node={row['ranks_per_node']}",
                "cpus_per_task=1",
                f"matrix_sha256={analysis.REVIEWED_MATRIX_SHA256}",
                "git_commit=0123456789abcdef0123456789abcdef01234567",
                "git_dirty=false",
                "allocation_revalidated=YES",
                "account_qos_revalidated=YES",
                "allocation_valid_until=2026-12-31",
                "started_at=2026-07-10T10:00:00+00:00",
                "finished_at=2026-07-10T10:05:00+00:00",
                "accounting_status=pending_post_job_collection",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (batch / "environment.txt").write_text(
        "\n".join(
            (
                "=== git ===",
                "reviewed commit",
                "=== modules ===",
                "reviewed modules",
                "=== python ===",
                "Python 3",
                "=== PETSc contract ===",
                "(3, 24, 0)",
                "=== allocation ===",
                "reviewed allocation",
                "=== nodes ===",
                "node001",
                "=== reviewed environment whitelist ===",
                "SLURM_JOB_ACCOUNT=fta-26-40",
                "SLURM_JOB_QOS=3571_6328",
                f"SLURM_JOB_PARTITION={row['partition']}",
                f"SLURM_JOB_NUM_NODES={row['nodes']}",
                f"SLURM_NTASKS={row['total_ranks']}",
                "SLURM_CPUS_PER_TASK=1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (batch / "execute.log").write_text("completed\n", encoding="utf-8")
    (slurm / f"{row['case_id']}-123.out").write_text("stdout\n", encoding="utf-8")
    (slurm / f"{row['case_id']}-123.err").write_text("", encoding="utf-8")
    values = {
        "JobIDRaw": "123",
        "JobID": "123",
        "JobName": row["case_id"],
        "Cluster": "karolina",
        "Account": "fta-26-40",
        "Partition": row["partition"],
        "QOS": "3571_6328",
        "State": "COMPLETED",
        "ElapsedRaw": "300",
        "AllocNodes": row["nodes"],
        "AllocCPUS": row["total_ranks"],
        "TotalCPU": "00:01:00",
        "CPUTimeRAW": str(300 * int(row["total_ranks"])),
        "MaxRSS": "1K",
        "MaxVMSize": "2K",
        "ConsumedEnergyRaw": "0",
        "ExitCode": "0:0",
        "Start": "2026-07-10T10:00:00",
        "End": "2026-07-10T10:05:00",
        "NodeList": "node001",
    }
    raw = "|".join(SACCT_FIELDS) + "\n" + "|".join(values[name] for name in SACCT_FIELDS) + "\n"
    parsed = parse_sacct(raw, job_id="123")
    (batch / "sacct_final.json").write_text(
        json.dumps(
            {
                "schema_id": "fenics-nonlinear-energies.slurm-accounting-snapshot",
                "schema_version": 1,
                "collected_at_utc": "2026-07-10T10:10:00+00:00",
                "job_id": "123",
                "source": {
                    "mode": "offline_file",
                    "raw_parsable2": raw,
                    "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "byte_count": len(raw.encode()),
                },
                "allocation": parsed["allocation"],
                "rows": parsed["rows"],
                "derived": parsed["derived"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (job / "matrix_row.json").write_text(json.dumps(row), encoding="utf-8")
    order = row["route_order"].split("|")
    (job / "run_records.json").write_text(
        json.dumps(
            [
                {
                    "route": route,
                    "returncode": returncode if index == 0 else 0,
                    "timed_out": returncode == 124 and index == 0,
                    "scientific_validation": {"status": "passed"},
                }
                for index, route in enumerate(order)
            ]
        ),
        encoding="utf-8",
    )
    if returncode != 0:
        return
    route_results = {}
    route_times = dict(route_times or {})
    for route in analysis.ROUTES:
        route_dir = measure / route
        route_dir.mkdir()
        output = _output(row, route)
        route_time = float(route_times.get(route, 1.0))
        output["total_time"] = route_time
        output["total_time_by_rank_s"] = [
            route_time - 0.001 * index
            for index in range(int(row["total_ranks"]))
        ]
        if mutate_output is not None:
            mutate_output(row, route, output)
        (route_dir / "output.json").write_text(json.dumps(output), encoding="utf-8")
        _write_state(
            route_dir / "state.npz",
            row,
            route,
            delta=state_delta if route == "constitutive_ad" else 0.0,
        )
        route_results[route] = {
            "status": "completed",
            "solver_success": True,
            "collective_max_wall_time_s": route_time,
            "per_rank_wall_times_s": [
                float(route_times.get(route, 1.0)) - 0.001 * index
                for index in range(int(row["total_ranks"]))
            ],
            "timing_rank_count": int(row["total_ranks"]),
            "timing_provenance": "solver_allgather_then_MPI_MAX",
            "output_json": f"{route}/output.json",
            "state_npz": f"{route}/state.npz",
        }
    block_result = {
        "schema_version": 1,
        "experiment_id": "EXP-ROUTE-001",
        "tier": row["tier"],
        "comparison_id": row["comparison_id"],
        "block_repetition": int(row["block_repetition"]),
        "status": "routes_completed_pending_endpoint_analysis",
        "route_order": order,
        "route_order_policy": "seeded_balanced_alternating_v1",
        "timing_reduction": "mpi_collective_max",
        "routes": route_results,
        "job_metadata": {"slurm_job_id": "123", "slurm_cluster_name": "karolina"},
        "timing_claim_released": False,
    }
    (measure / "block_result.json").write_text(
        json.dumps(block_result), encoding="utf-8"
    )


def _campaign(
    tmp_path: Path,
    *,
    omit_case: str = "",
    censored_case: str = "",
    route_times: dict[str, float] | None = None,
    **job_kwargs,
):
    matrix = tmp_path / "matrix.csv"
    root = tmp_path / "campaign"
    rows = _matrix_rows()
    _write_matrix(matrix, rows)
    analysis.REVIEWED_MATRIX_SHA256 = hashlib.sha256(matrix.read_bytes()).hexdigest()
    for row in rows:
        if row["case_id"] == omit_case:
            continue
        kwargs = dict(job_kwargs) if row["case_id"] == job_kwargs.get("target_case") else {}
        kwargs.pop("target_case", None)
        _write_job(
            root,
            row,
            returncode=124 if row["case_id"] == censored_case else 0,
            route_times=route_times,
            **kwargs,
        )
    root.mkdir(exist_ok=True)
    reviewed_dir = root / "reviewed_artifacts"
    reviewed_dir.mkdir()
    reviewed = reviewed_dir / "tier_b_review.json"
    reviewed.write_text(json.dumps({"decision": "reviewed"}) + "\n", encoding="utf-8")
    reviewed_sha = hashlib.sha256(reviewed.read_bytes()).hexdigest()
    release_record = {
        "schema_id": "fenics-nonlinear-energies.human-release-authorization",
        "schema_version": 1,
        "status": "approved",
        "decision": "explicit_human_release_after_review",
        "matrix_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
        "source_commit": "0123456789abcdef0123456789abcdef01234567",
        "authorizes_experiment": "EXP-ROUTE-001",
        "authorizes_tiers": [
            "full_solve_confirmation",
            "low_order_confirmation",
        ],
        "reviewer": "synthetic-test-reviewer",
        "reviewed_artifacts": [
            {
                "path": "reviewed_artifacts/tier_b_review.json",
                "sha256": reviewed_sha,
            }
        ],
    }
    release_path = root / "release_authorization.json"
    release_path.write_text(json.dumps(release_record) + "\n", encoding="utf-8")
    manifest = {
        "status": "submitted",
        "matrix_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
        "selected_experiments": ["EXP-ROUTE-001"],
        "include_optional": True,
        "only_optional": True,
        "selected_tiers": [
            "full_solve_confirmation",
            "low_order_confirmation",
        ],
        "test_only_commands": False,
        "case_count": 30,
        "source_commit": "0123456789abcdef0123456789abcdef01234567",
        "source_dirty": False,
        "release_authorization": {
            "schema_id": "fenics-nonlinear-energies.human-release-authorization",
            "path": "release_authorization.json",
            "sha256": hashlib.sha256(release_path.read_bytes()).hexdigest(),
            "reviewer": "synthetic-test-reviewer",
        },
    }
    (root / "prepared_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return matrix, root, rows


def test_complete_balanced_campaign_admits_only_collective_max_timing(tmp_path: Path) -> None:
    matrix, root, _rows = _campaign(tmp_path)
    result = analysis.analyze(matrix, root, root / "prepared_manifest.json")
    assert result["schema"] == {"id": analysis.SCHEMA_ID, "version": 1}
    assert result["timing_admissible"] is True
    assert result["endpoint_correct_timing_admissible"] is True
    assert result["descriptive_timing_available"] is True
    assert result["comparative_ranking_admissible"] is False
    assert result["representative_low_order_confirmation_passed"] is True
    assert result["terminal_decision"] == "tier_b_descriptive_timing_only"
    assert result["status_counts"] == {"timing_admitted": 30}
    assert len(result["timing_summary"]) == 3
    assert all(
        row["uncertainty_method"]["resamples"] == 10000
        and len(
            row["paired_constitutive_over_element_ratio"][
                "bootstrap_median_confidence_interval"
            ]
        )
        == 2
        for row in result["timing_summary"]
    )
    assert len(result["structural_censors"]) == 2
    for block in result["blocks"]:
        assert block["status"] == "timing_admitted"
        assert block["routes"]["element_ad"][
            "admitted_collective_max_wall_time_s"
        ] == 1.0
        assert block["routes"]["constitutive_ad"][
            "admitted_collective_max_wall_time_s"
        ] == 1.0


def test_tier_b_rejects_incomplete_environment_and_accounting_shape(
    tmp_path: Path,
) -> None:
    matrix, root, rows = _campaign(tmp_path)
    target = rows[0]
    batch = root / "jobs" / target["case_id"] / "job_123"
    environment = batch / "environment.txt"
    original_environment = environment.read_text(encoding="utf-8")
    environment.write_text(
        original_environment.replace("SLURM_JOB_QOS=3571_6328\n", ""),
        encoding="utf-8",
    )
    result = analysis.analyze(matrix, root, root / "prepared_manifest.json")
    failed = next(
        block for block in result["blocks"] if block["case_id"] == target["case_id"]
    )
    assert failed["status"] == "invalid"
    assert "SLURM_JOB_QOS" in failed["reason"]

    environment.write_text(original_environment, encoding="utf-8")
    accounting_path = batch / "sacct_final.json"
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    accounting["allocation"]["alloc_cpus"] += 1
    accounting_path.write_text(json.dumps(accounting) + "\n", encoding="utf-8")
    result = analysis.analyze(matrix, root, root / "prepared_manifest.json")
    failed = next(
        block for block in result["blocks"] if block["case_id"] == target["case_id"]
    )
    assert failed["status"] == "invalid"
    assert "disagrees with raw evidence" in failed["reason"]


def test_complete_tier_b_record_tree_is_relocatable_and_rejects_escape(
    tmp_path: Path,
) -> None:
    matrix, root, rows = _campaign(tmp_path)
    before = analysis.analyze(matrix, root, root / "prepared_manifest.json")
    relocated = tmp_path / "relocated_campaign"
    root.rename(relocated)
    after = analysis.analyze(
        matrix, relocated, relocated / "prepared_manifest.json"
    )
    assert after["terminal_decision"] == before["terminal_decision"]
    assert after["status_counts"] == before["status_counts"]
    assert after["timing_summary"] == before["timing_summary"]

    target = rows[0]
    block_path = (
        relocated
        / "cases"
        / target["case_id"]
        / "job_123"
        / "measure_01"
        / "block_result.json"
    )
    block = json.loads(block_path.read_text(encoding="utf-8"))
    block["routes"]["element_ad"]["output_json"] = "../../matrix_row.json"
    block_path.write_text(json.dumps(block) + "\n", encoding="utf-8")
    rejected = analysis.analyze(
        matrix, relocated, relocated / "prepared_manifest.json"
    )
    failed = next(
        row for row in rejected["blocks"] if row["case_id"] == target["case_id"]
    )
    assert failed["status"] == "invalid"
    assert "escapes comparison block" in failed["reason"]


def test_comparative_ranking_requires_paired_and_order_stratified_intervals(
    tmp_path: Path,
) -> None:
    matrix, root, _rows = _campaign(
        tmp_path,
        route_times={"element_ad": 1.0, "constitutive_ad": 1.5},
    )
    result = analysis.analyze(matrix, root, root / "prepared_manifest.json")
    assert result["endpoint_correct_timing_admissible"] is True
    assert result["comparative_ranking_admissible"] is True
    assert result["terminal_decision"] == "tier_b_comparative_ranking_admissible"
    assert all(
        row["comparative_ranking"]["ranking_admissible"] is True
        and row["comparative_ranking"]["overall_winner_beyond_tie_band"]
        == "element_ad"
        and row["comparative_ranking"]["order_sensitivity_passed"] is True
        for row in result["timing_summary"]
    )


def test_tier_b_requires_hash_bound_human_release_authorization(
    tmp_path: Path,
) -> None:
    matrix, root, _rows = _campaign(tmp_path)
    manifest_path = root / "prepared_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("release_authorization")
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    result = analysis.analyze(matrix, root, manifest_path)
    assert result["endpoint_correct_timing_admissible"] is False
    assert "manifest_release_authorization_missing" in result[
        "coverage_and_campaign_failure_reasons"
    ]


def test_endpoint_state_failure_withholds_every_campaign_timing(tmp_path: Path) -> None:
    target = "route_p4l1_np8_block_01"
    matrix, root, _rows = _campaign(
        tmp_path,
        target_case=target,
        state_delta=1.0e-4,
    )
    result = analysis.analyze(matrix, root, root / "prepared_manifest.json")
    assert result["timing_admissible"] is False
    failed = next(block for block in result["blocks"] if block["case_id"] == target)
    assert failed["status"] == "invalid"
    assert "state" in failed["reason"]
    assert result["timing_summary"] == []
    assert all(
        route["timing_exposed"] is False
        for block in result["blocks"]
        for route in block["routes"].values()
    )


def test_pointwise_branch_map_mismatch_withholds_timing(tmp_path: Path) -> None:
    target = "route_p4l1_np8_block_01"

    def swap_map(_row, route, output):
        if route == "constitutive_ad":
            output["branch_diagnostics"]["canonical_map_sha256"] = hashlib.sha256(
                b"pointwise-swap-with-equal-counts"
            ).hexdigest()

    matrix, root, _rows = _campaign(
        tmp_path,
        target_case=target,
        mutate_output=swap_map,
    )
    result = analysis.analyze(matrix, root, root / "prepared_manifest.json")
    failed = next(block for block in result["blocks"] if block["case_id"] == target)
    assert failed["status"] == "invalid"
    assert "pointwise branch maps" in failed["reason"]
    assert result["timing_admissible"] is False
    assert all(
        route["admitted_collective_max_wall_time_s"] is None
        for block in result["blocks"]
        for route in block["routes"].values()
    )


def test_missing_low_order_and_censored_rows_remain_visible(tmp_path: Path) -> None:
    missing = "route_p1l1_np8_block_10"
    censored = "route_p4l1_np32_block_10"
    matrix, root, _rows = _campaign(
        tmp_path,
        omit_case=missing,
        censored_case=censored,
    )
    result = analysis.analyze(matrix, root, root / "prepared_manifest.json")
    by_case = {block["case_id"]: block for block in result["blocks"]}
    assert by_case[missing]["status"] == "missing"
    assert by_case[censored]["status"] == "censored"
    assert by_case[censored]["reason"] == "runner_timeout"
    assert result["representative_low_order_confirmation_passed"] is False
    assert result["timing_admissible"] is False


def test_stale_riesz_evidence_is_invalid_and_cli_returns_two_when_required(
    tmp_path: Path,
) -> None:
    target = "route_p4l1_np8_block_01"

    def stale(_row, route, output):
        if route == "constitutive_ad":
            output["nonlinear_convergence"]["last_riesz_solve"]["rhs_norm"] = 2.0

    matrix, root, _rows = _campaign(
        tmp_path,
        target_case=target,
        mutate_output=stale,
    )
    output_json = tmp_path / "analysis.json"
    output_csv = tmp_path / "analysis.csv"
    completed = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            str(SCRIPT),
            "--matrix",
            str(matrix),
            "--campaign-root",
            str(root),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--require-timing-admissible",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["timing_admissible"] is False
    failed = next(block for block in payload["blocks"] if block["case_id"] == target)
    assert "stale" in failed["reason"]
    assert output_csv.is_file()


def test_matrix_policy_mutation_is_reported_without_reading_timing(tmp_path: Path) -> None:
    matrix, root, rows = _campaign(tmp_path)
    rows[0]["ksp_rtol"] = "1e-6"
    _write_matrix(matrix, rows)
    # Keep the manifest digest current to isolate the matrix-policy failure.
    manifest = json.loads((root / "prepared_manifest.json").read_text(encoding="utf-8"))
    manifest["matrix_sha256"] = hashlib.sha256(matrix.read_bytes()).hexdigest()
    (root / "prepared_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = analysis.analyze(matrix, root, root / "prepared_manifest.json")
    assert result["timing_admissible"] is False
    assert any("ksp_rtol" in row["reason"] for row in result["matrix_policy_violations"])
    first = next(block for block in result["blocks"] if block["case_id"] == rows[0]["case_id"])
    assert first["status"] == "invalid"
    assert all(
        route["admitted_collective_max_wall_time_s"] is None
        for block in result["blocks"]
        for route in block["routes"].values()
    )


def test_rank_timing_vector_must_prove_the_declared_collective_max(tmp_path: Path) -> None:
    matrix, root, rows = _campaign(tmp_path)
    target = rows[0]
    block_path = (
        root
        / "cases"
        / target["case_id"]
        / "job_123"
        / "measure_01"
        / "block_result.json"
    )
    block = json.loads(block_path.read_text(encoding="utf-8"))
    block["routes"]["element_ad"]["per_rank_wall_times_s"][0] = 0.5
    block_path.write_text(json.dumps(block), encoding="utf-8")
    result = analysis.analyze(matrix, root, root / "prepared_manifest.json")
    failed = next(
        row for row in result["blocks"] if row["case_id"] == target["case_id"]
    )
    assert failed["status"] == "invalid"
    assert "rank-wise maximum" in failed["reason"]
    assert result["timing_admissible"] is False
