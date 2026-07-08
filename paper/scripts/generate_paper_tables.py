#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from common import REPO_ROOT, TABLES_ROOT, ensure_paper_dirs, read_csv_rows, read_json, write_json, write_text


LOCAL_P3D_SUMMARY = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_local_pmg_scaling/comparison_summary.json"
)
MIXED_P3D_SUMMARY = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_scaling/comparison_summary.json"
)
SOURCEFIXED_P3D_SUMMARY = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/plasticity3d_l1_2_lambda1_grad1e2_scaling_all_pmg/comparison_summary.json"
)
P3D_DEGREE_ENERGY_STUDY_SUMMARY = (
    REPO_ROOT
    / "artifacts/raw_results/plasticity3d_lambda1p55_degree_mesh_energy_study/comparison_summary.json"
)
P3D_VALIDATION_SUMMARY = REPO_ROOT / "artifacts/raw_results/plasticity3d_validation/comparison_summary.json"
P3D_DERIVATIVE_ABLATION_SUMMARY = (
    REPO_ROOT / "artifacts/raw_results/plasticity3d_derivative_ablation/comparison_summary.json"
)
JAX_FEM_BASELINE_SUMMARY = REPO_ROOT / "artifacts/raw_results/jax_fem_hyperelastic_baseline/comparison_summary.json"
GLOBALIZATION_METHOD_COMPARE = REPO_ROOT / "artifacts/reports/globalization_method_compare/full_summary.csv"
DERIVATIVE_ROUTE_COMPARE = REPO_ROOT / "artifacts/reports/derivative_route_compare/full_summary.csv"
SUPPLEMENTAL_REPORT_ROOT = REPO_ROOT / "artifacts/reports/paper_reviewer_gap_experiments"
SUPPLEMENTAL_HE_DISTRIBUTION = SUPPLEMENTAL_REPORT_ROOT / "full_he_distribution.csv"
SUPPLEMENTAL_HE_PMG = SUPPLEMENTAL_REPORT_ROOT / "full_he_pmg.csv"
SUPPLEMENTAL_TOPOLOGY_CONSISTENCY = SUPPLEMENTAL_REPORT_ROOT / "full_topology_consistency.csv"
SUPPLEMENTAL_GL_GLOBALIZATION = SUPPLEMENTAL_REPORT_ROOT / "full_gl_globalization.csv"
SUPPLEMENTAL_P3D_DERIVATIVE_DEGREE = SUPPLEMENTAL_REPORT_ROOT / "full_p3d_derivative_degree.csv"
P3D_LOCAL_LAMBDA155_SCALING = (
    REPO_ROOT
    / "artifacts/reports/plasticity3d_p4_l1_2_mumps_pmg_step_grad_local_karolina_scaling/local_solver_total_scaling.csv"
)
P3D_KAROLINA_LAMBDA155_SCALING = (
    REPO_ROOT
    / "artifacts/reports/plasticity3d_p4_l1_2_mumps_pmg_step_grad_local_karolina_scaling/karolina_rpn16_solver_total_scaling.csv"
)
P3D_LAMBDA155_STOP_SUMMARY = (
    REPO_ROOT
    / "artifacts/raw_results/example_runs/plasticity3d_p4_l1_2_lambda1p55_mumps_pmg_step_grad_convergence_20260507_190225/step_grad_convergence_summary.csv"
)

PLAPLACE_PARITY = REPO_ROOT / "experiments/analysis/docs_assets/data/plaplace/parity_showcase.csv"
GL_PARITY = REPO_ROOT / "experiments/analysis/docs_assets/data/ginzburg_landau/parity_showcase.csv"
HE_PARITY = REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/parity_showcase.csv"

PLAPLACE_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/plaplace/strong_scaling.csv"
GL_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/ginzburg_landau/strong_scaling.csv"
HE_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/strong_scaling.csv"
HE_KAROLINA_PMG_SCALING = (
    REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/karolina_l5_pmg_scaling.csv"
)
TOPO_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/topology/strong_scaling.csv"
TOPO_RESOLUTION = REPO_ROOT / "experiments/analysis/docs_assets/data/topology/resolution_objectives.csv"

P2D_SHOWCASE = REPO_ROOT / "artifacts/raw_results/docs_showcase/mc_plasticity_p4_l5/output.json"
P2D_L6_SUMMARY = REPO_ROOT / "artifacts/raw_results/slope_stability_l6_p4_deep_p1_tail_scaling_lambda1_maxit20/summary.json"
P2D_L7_SUMMARY = REPO_ROOT / "artifacts/raw_results/slope_stability_l7_p4_deep_p1_tail_scaling_lambda1_maxit20/summary.json"
SOURCE_CONT_NP8 = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/ssr_indirect_p4_l1_omega6p7e6_np8_shell_default_afterfix/data/run_info.json"
)
SOURCE_CONT_NP32 = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/ssr_indirect_p4_l1_omega6p7e6_np32_shell_default_afterfix/data/run_info.json"
)
SOURCE_CONT_NP8_PROGRESS = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/ssr_indirect_p4_l1_omega6p7e6_np8_shell_default_afterfix/data/progress_latest.json"
)
SOURCE_CONT_NP32_PROGRESS = (
    REPO_ROOT
    / "artifacts/raw_results/source_compare/ssr_indirect_p4_l1_omega6p7e6_np32_shell_default_afterfix/data/progress_latest.json"
)

LOCAL_IMPL = "local_constitutiveAD_local_pmg_armijo"
SOURCE_IMPL = "source_local_pmg_armijo"
LOCAL_SOURCEFIXED_IMPL = "local_constitutiveAD_local_pmg_sourcefixed_armijo"
SOURCE_SOURCEFIXED_IMPL = "source_local_pmg_sourcefixed_armijo"

IMPLEMENTATION_LABELS = {
    "fenics_custom": "FEniCS custom Newton",
    "jax_petsc_element": "JAX+PETSc element AD",
    "jax_petsc_local_sfd": "JAX+PETSc colored SFD",
    "jax_serial": "serial JAX",
    LOCAL_IMPL: "constitutive-AD PMG solver",
    SOURCE_IMPL: "reference-operator PMG variant",
    LOCAL_SOURCEFIXED_IMPL: "constitutive-AD PMG solver",
    SOURCE_SOURCEFIXED_IMPL: "reference-formula assembly PMG variant",
}

GLOBALIZATION_BENCHMARK_ORDER = {
    "plaplace_l10_np32": 0,
    "gl_l10_np16": 1,
    "he_l4_np32_steps8": 2,
    "plasticity3d_p2_l1_np32_lambda155": 3,
}

GLOBALIZATION_METHOD_ORDER = {
    "newton_linesearch": 0,
    "steihaug_trust": 1,
    "hybrid_trust_linesearch": 2,
}

GLOBALIZATION_METHOD_LABELS = {
    "newton_linesearch": "Newton + LS",
    "steihaug_trust": "Steihaug TR",
    "hybrid_trust_linesearch": "Hybrid TR+LS",
}

GLOBALIZATION_BENCHMARK_LABELS = {
    "plaplace_l10_np32": "$p$-Laplace $L_{10}$",
    "gl_l10_np16": "\\shortstack[l]{Ginzburg--Landau\\\\$L_{10}$}",
    "he_l4_np32_steps8": "Hyperelasticity $L_4$",
    "plasticity3d_p2_l1_np32_lambda155": "\\shortstack[l]{Plasticity3D\\\\$P_2(L_1)$}",
}

DERIVATIVE_BENCHMARK_ORDER = {
    "plaplace_l9_np32": 0,
    "he_l4_step1_np32": 1,
    "plasticity3d_p2_l1_lambda155_np32": 2,
}

DERIVATIVE_ROUTE_ORDER = {
    "element_ad": 0,
    "colored_sfd": 1,
    "constitutive_ad": 2,
}

DERIVATIVE_BENCHMARK_LABELS = {
    "plaplace_l9_np32": "$p$-Laplace $L_9$",
    "he_l4_step1_np32": "Hyperelasticity $L_4$ step 1",
    "plasticity3d_p2_l1_lambda155_np32": (
        "\\shortstack[l]{Plasticity3D $P_2(L_1)$\\\\$\\lambda_{\\mathrm{sr}}=\\num{1.55}$}"
    ),
}

DERIVATIVE_ROUTE_LABELS = {
    "element_ad": "Element AD",
    "colored_sfd": "Colored SFD",
    "constitutive_ad": "Constitutive AD",
}

REVIEWER_HE_BUILD_LABELS = {
    "replicated": "replicated",
    "rank_local": "rank-local",
}

HE_DISTRIBUTION_PURPOSE_LABELS = {
    "correctness": "agreement check",
    "memory": "memory comparison",
}

HE_DISTRIBUTION_OUTCOME_LABELS = {
    "completed": "completed",
    "fixed_work": "one linearization",
    "fixed_work_completed": "one linearization",
}

GLOBALIZATION_OUTCOME_LABELS = {
    "completed": "completed",
    "failed": "iteration cap",
    "timeout": "timeout",
}

HE_PMG_WORK_LABELS = {
    "fixed_work": "8 Newton linearizations",
    "fixed_work_completed": "8 Newton linearizations",
}

TOPOLOGY_SCHEDULE_LABELS = {
    "fixed_work": "fixed schedule",
    "fixed_work_completed": "fixed schedule",
}

P3D_DERIVATIVE_DEGREE_WORK_LABELS = {
    "fixed_work": "one Newton linearization",
    "fixed_work_completed": "one Newton linearization",
}

REVIEWER_HE_PMG_LABELS = {
    "gamg": "GAMG",
    "pmg_l2_hypre": "PMG $L_2$ + Hypre",
    "pmg_l2_redundant_mumps": "PMG $L_2$ + MUMPS",
    "pmg_l3_redundant_mumps": "PMG $L_3$ + MUMPS",
}

REVIEWER_P3D_ROUTE_LABELS = {
    "element_ad": "Element AD",
    "colored_sfd": "Colored SFD",
    "constitutive_ad": "Constitutive AD",
}

MESH_ALIAS_MATH = {
    "L1": "L_{1}",
    "L1_2": "L_{2}",
    "L1_2_3": "L_{3}",
    "L1_2_3_4": "L_{4}",
}


def _trim_decimal(text: str) -> str:
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")


def num(text: str) -> str:
    return rf"\num{{{text}}}"


def fmt_float(value: float, digits: int = 3) -> str:
    return num(_trim_decimal(f"{float(value):.{digits}f}"))


def fmt_int(value: object) -> str:
    return str(int(float(value)))


def fmt_count(value: object) -> str:
    return fmt_int(value)


def fmt_dofs(value: object) -> str:
    return num(str(int(float(value))))


def fmt_wall_time(value: float) -> str:
    value = float(value)
    if abs(value) >= 100:
        return fmt_float(value, 0)
    if abs(value) >= 10:
        return fmt_float(value, 1)
    if abs(value) >= 1:
        return fmt_float(value, 2)
    if abs(value) >= 0.1:
        return fmt_float(value, 3)
    return fmt_float(value, 4)


def fmt_energy(value: float, *, precision: int | None = None) -> str:
    value = float(value)
    if precision is not None:
        return fmt_float(value, precision)
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return fmt_float(value, 0)
    if magnitude >= 1_000:
        return fmt_float(value, 1)
    if magnitude >= 100:
        return fmt_float(value, 3)
    if magnitude >= 1:
        return fmt_float(value, 6)
    return fmt_float(value, 10)


def fmt_sig(value: float, sig: int = 3) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    digits = max(sig - 1 - int(math.floor(math.log10(abs(value)))), 0)
    return fmt_float(value, digits)


def fmt_sci(value: float, sig: int = 3) -> str:
    value = float(value)
    if value == 0.0:
        return num("0")
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    digits = max(sig - 1, 0)
    return rf"$\num{{{_trim_decimal(f'{mantissa:.{digits}f}')}}}\times 10^{{{exponent}}}$"


def implementation_label(name: object) -> str:
    key = str(name)
    if key in IMPLEMENTATION_LABELS:
        return IMPLEMENTATION_LABELS[key]
    if "local_constitutiveAD" in key and "local_pmg" in key:
        return "constitutive-AD PMG solver"
    if "sourcefixed" in key:
        return "reference-formula assembly PMG variant"
    if key.startswith("source") or "_source" in key:
        return "reference-operator PMG variant"
    return key.replace("_", r"\_")


def mesh_label(alias: object) -> str:
    key = str(alias)
    if key in MESH_ALIAS_MATH:
        return rf"${MESH_ALIAS_MATH[key]}$"
    if key.startswith("L") and key[1:].isdigit():
        return rf"$L_{{{key[1:]}}}$"
    return key.replace("_", r"\_")


def _math_mesh(alias: object) -> str:
    key = str(alias)
    if key in MESH_ALIAS_MATH:
        return MESH_ALIAS_MATH[key]
    if key.startswith("L") and key[1:].isdigit():
        return f"L_{{{key[1:]}}}"
    return key.replace("_", r"\_")


def degree_label(degree: object) -> str:
    key = str(degree)
    if key.startswith("P") and key[1:].isdigit():
        return rf"$P_{{{key[1:]}}}$"
    return key.replace("_", r"\_")


def element_label(degree: object, mesh_alias: object) -> str:
    key = str(degree)
    if key.startswith("P") and key[1:].isdigit():
        return rf"$P_{{{key[1:]}}}({_math_mesh(mesh_alias)})$"
    return f"{degree_label(degree)} {mesh_label(mesh_alias)}"


def find_csv_row(rows: list[dict[str, str]], solver: str, ranks: int) -> dict[str, str]:
    return next(row for row in rows if row.get("solver") == solver and int(row["nprocs"]) == ranks)


def xcol(weight: float, align: str = "RaggedRight") -> str:
    return rf">{{\hsize={float(weight):.3f}\hsize\linewidth=\hsize\{align}\arraybackslash}}X"


def xspec(*columns: tuple[float, str]) -> str:
    """Return normalized tabularx X columns.

    The hsize weights must sum to the number of X columns; otherwise tabularx
    can stretch the table poorly and emit alignment warnings.
    """
    total = sum(weight for weight, _align in columns)
    scale = len(columns) / total
    return "".join(xcol(weight * scale, align) for weight, align in columns)


def fill_spec(columns: str) -> str:
    """Return a tabular* spec with compact outer edges and stretched interiors."""
    parts = columns.split()
    if not parts:
        raise ValueError("tabular* spec needs at least one column")
    rest = " ".join(parts[1:])
    return "@{}" + parts[0] + r"@{\extracolsep{\fill}}" + (f" {rest}" if rest else "") + "@{}"


def pcol(width: str, align: str = "RaggedRight") -> str:
    return rf">{{\{align}\arraybackslash}}p{{{width}}}"


LatexRow = list[str] | str
LatexBlock = tuple[str, str, list[str], list[LatexRow]]


def _latex_lines(header: list[str], rows: list[LatexRow]) -> list[str]:
    lines = [r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        if isinstance(row, str):
            lines.append(row)
        else:
            lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    return lines


def latex_table(spec: str, header: list[str], rows: list[LatexRow]) -> str:
    lines = [rf"\begin{{tabular}}{{{spec}}}", *_latex_lines(header, rows), r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_tabularstar(spec: str, header: list[str], rows: list[LatexRow], *, width: str = r"\textwidth") -> str:
    lines = [
        rf"\begin{{tabular*}}{{{width}}}{{{spec}}}",
        *_latex_lines(header, rows),
        r"\end{tabular*}",
    ]
    return "\n".join(lines) + "\n"


def latex_tabularx(spec: str, header: list[str], rows: list[LatexRow], *, width: str = r"\textwidth") -> str:
    lines = [
        rf"\begin{{tabularx}}{{{width}}}{{{spec}}}",
        *_latex_lines(header, rows),
        r"\end{tabularx}",
    ]
    return "\n".join(lines) + "\n"


def latex_tabularx_blocks(blocks: list[LatexBlock], *, width: str = r"\textwidth") -> str:
    lines = [rf"\begin{{minipage}}{{{width}}}", r"\centering"]
    for index, (title, spec, header, rows) in enumerate(blocks):
        if title:
            lines.extend([rf"\textit{{{title}}}", r"\par\vspace{0.2em}"])
        lines.extend(
            [
                rf"\begin{{tabularx}}{{{width}}}{{{spec}}}",
                *_latex_lines(header, rows),
                r"\end{tabularx}",
            ]
        )
        if index != len(blocks) - 1:
            lines.extend(["", r"\vspace{0.45em}", ""])
    lines.append(r"\end{minipage}")
    return "\n".join(lines) + "\n"


def load_rows(path: Path) -> list[dict[str, object]]:
    data = read_json(path)
    rows = [dict(row) for row in data["rows"]]
    rows.sort(key=lambda row: (int(row.get("ranks", 10**6)), str(row.get("implementation", ""))))
    return rows


def find_rows(rows: list[dict[str, object]], impl: str) -> list[dict[str, object]]:
    selected = [row for row in rows if str(row.get("implementation", "")) == impl]
    selected.sort(key=lambda row: int(row["ranks"]))
    return selected


def write_table(name: str, spec: str, header: list[str], rows: list[LatexRow]) -> None:
    write_text(TABLES_ROOT / name, latex_table(spec, header, rows))


def write_table_star(
    name: str, spec: str, header: list[str], rows: list[LatexRow], *, width: str = r"\textwidth"
) -> None:
    write_text(TABLES_ROOT / name, latex_tabularstar(spec, header, rows, width=width))


def write_tablex(name: str, spec: str, header: list[str], rows: list[LatexRow], *, width: str = r"\textwidth") -> None:
    write_text(TABLES_ROOT / name, latex_tabularx(spec, header, rows, width=width))


def write_tablex_blocks(name: str, blocks: list[LatexBlock], *, width: str = r"\textwidth") -> None:
    write_text(TABLES_ROOT / name, latex_tabularx_blocks(blocks, width=width))


def select_csv_rows(path: Path, implementations: tuple[str, ...]) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    return [row for row in rows if row.get("implementation") in implementations]


def select_topology_rows(labels: tuple[str, ...]) -> list[dict[str, str]]:
    rows = read_csv_rows(TOPO_RESOLUTION)
    return [row for row in rows if row.get("label") in labels]


def plasticity3d_local_karolina_rows() -> list[dict[str, object]]:
    stop_rows = {int(row["ranks"]): row for row in read_csv_rows(P3D_LAMBDA155_STOP_SUMMARY)}
    grad_targets = [
        float(row["grad_target"])
        for row in stop_rows.values()
        if str(row.get("grad_target", "")).strip()
    ]
    grad_target = grad_targets[0] if grad_targets else math.nan
    rows: list[dict[str, object]] = []
    for row in read_csv_rows(P3D_LOCAL_LAMBDA155_SCALING):
        owned = float(row["owned_free_dofs_sum"])
        overlap = float(row["overlap_total_dofs_sum"])
        stop_row = stop_rows.get(int(row["ranks"]))
        grad = float(row["final_grad_norm"])
        rows.append(
            {
                "source": "single-node CPU",
                "ranks": int(row["ranks"]),
                "nodes": None,
                "solver_total_s": float(row["solver_total_s"]),
                "solve_time_s": float(row["nonlinear_solve_s"]),
                "newton_iterations": int(row["newton_iterations"]),
                "linear_iterations_total": int(row["linear_iterations_total"]),
                "energy": float(row["energy"]),
                "grad": grad,
                "grad_over_target": (
                    float(stop_row["final_grad_over_target"])
                    if stop_row is not None and str(stop_row.get("final_grad_over_target", "")).strip()
                    else grad / grad_target
                ),
                "max_local_dofs": int(row["max_local_dofs"]),
                "owned_free_dofs_sum": int(owned),
                "overlap_total_dofs_sum": int(overlap),
                "replication_ratio": float(row.get("replication_ratio") or overlap / owned),
            }
        )
    for row in read_csv_rows(P3D_KAROLINA_LAMBDA155_SCALING):
        if row.get("output") != "yes" or not row.get("solver_total"):
            continue
        owned = float(row["owned_free_dofs_sum"])
        overlap = float(row["overlap_total_dofs_sum"])
        stop_row = stop_rows.get(int(row["ranks"]))
        grad = float(row["grad"])
        rows.append(
            {
                "source": "multi-node CPU",
                "ranks": int(row["ranks"]),
                "nodes": int(row["nodes"]),
                "solver_total_s": float(row["solver_total"]),
                "solve_time_s": float(row["solve_time"]),
                "newton_iterations": int(row["nit"]),
                "linear_iterations_total": int(row["ksp_its"]),
                "energy": float(row["energy"]),
                "grad": grad,
                "grad_over_target": (
                    float(stop_row["final_grad_over_target"])
                    if stop_row is not None and str(stop_row.get("final_grad_over_target", "")).strip()
                    else grad / grad_target
                ),
                "max_local_dofs": int(row["max_local_dofs"]),
                "owned_free_dofs_sum": int(owned),
                "overlap_total_dofs_sum": int(overlap),
                "replication_ratio": float(row.get("replication_ratio") or overlap / owned),
            }
        )
    rows.sort(key=lambda row: (str(row["source"]) != "single-node CPU", int(row["ranks"])))
    return rows


def globalization_method_rows(path: Path = GLOBALIZATION_METHOD_COMPARE) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    rows.sort(
        key=lambda row: (
            GLOBALIZATION_BENCHMARK_ORDER.get(row.get("benchmark", ""), 10**6),
            GLOBALIZATION_METHOD_ORDER.get(row.get("method", ""), 10**6),
        )
    )
    return rows


def derivative_route_rows(path: Path = DERIVATIVE_ROUTE_COMPARE) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    rows.sort(
        key=lambda row: (
            DERIVATIVE_BENCHMARK_ORDER.get(row.get("benchmark", ""), 10**6),
            DERIVATIVE_ROUTE_ORDER.get(row.get("route", ""), 10**6),
        )
    )
    return rows


def supplemental_he_distribution_rows(path: Path = SUPPLEMENTAL_HE_DISTRIBUTION) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    rows.sort(
        key=lambda row: (
            0 if row.get("probe") == "correctness" else 1,
            int(float(row.get("level") or 0)),
            int(float(row.get("nprocs") or 0)),
            0 if row.get("build_mode") == "replicated" else 1,
        )
    )
    return rows


def supplemental_he_pmg_rows(path: Path = SUPPLEMENTAL_HE_PMG) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    order = {key: index for index, key in enumerate(REVIEWER_HE_PMG_LABELS)}
    rows.sort(key=lambda row: order.get(row.get("candidate", ""), 10**6))
    return rows


def supplemental_topology_consistency_rows(path: Path = SUPPLEMENTAL_TOPOLOGY_CONSISTENCY) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    rows.sort(key=lambda row: int(float(row.get("nprocs") or 0)))
    return rows


def supplemental_gl_globalization_rows(path: Path = SUPPLEMENTAL_GL_GLOBALIZATION) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    rows.sort(key=lambda row: GLOBALIZATION_METHOD_ORDER.get(row.get("method", ""), 10**6))
    return rows


def supplemental_p3d_derivative_degree_rows(path: Path = SUPPLEMENTAL_P3D_DERIVATIVE_DEGREE) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    route_order = {key: index for index, key in enumerate(REVIEWER_P3D_ROUTE_LABELS)}
    rows.sort(
        key=lambda row: (
            int(float(row.get("free_dofs") or 0)),
            str(row.get("mesh_case", "")),
            route_order.get(row.get("route", ""), 10**6),
        )
    )
    return rows


def _fmt_optional_wall(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "--"
    return fmt_wall_time(float(text))


def _fmt_optional_energy(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "--"
    return fmt_energy(float(text))


def _fmt_optional_float(value: object, digits: int = 3) -> str:
    text = str(value).strip()
    if not text:
        return "--"
    return fmt_float(float(text), digits)


def _fmt_optional_sci(value: object, sig: int = 3) -> str:
    text = str(value).strip()
    if not text:
        return "--"
    return fmt_sci(float(text), sig)


def _fmt_optional_count(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "--"
    return fmt_count(text)


def _fmt_optional_dofs(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "--"
    return fmt_dofs(text)


def _result_label(value: object) -> str:
    text = str(value).strip()
    labels = {
        "completed": "completed",
        "fixed_work": "fixed work",
        "fixed_work_completed": "fixed work",
        "failed_design": "design failed",
        "timeout": "timeout",
        "skipped_oom_guard": "OOM guarded",
        "missing_json": "missing JSON",
        "failed": "failed",
    }
    return labels.get(text, text.replace("_", r"\_"))


def _p3d_discretization_label(row: dict[str, str]) -> str:
    text = str(row.get("discretization", "")).strip()
    if text.startswith("P") and "(L" in text and text.endswith(")"):
        degree, mesh = text.split("(", 1)
        return element_label(degree, mesh[:-1])
    if text:
        return text.replace("_", r"\_")
    return degree_label(f"P{int(float(row.get('degree') or 0))}")


def _fmt_mib_pair(max_mib: object, total_mib: object) -> str:
    max_text = str(max_mib).strip()
    total_text = str(total_mib).strip()
    if not max_text and not total_text:
        return "--"
    max_gib = float(max_text or 0.0) / 1024.0
    total_gib = float(total_text or 0.0) / 1024.0
    return f"{fmt_float(max_gib, 1)}/{fmt_float(total_gib, 1)}"


def _fmt_optional_gib(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "--"
    return fmt_float(float(text), 2)


def _fmt_he_coarse(row: dict[str, str]) -> str:
    coarse_pc = str(row.get("coarse_pc", "")).strip()
    if not coarse_pc:
        return "--"
    if coarse_pc == "redundant":
        factor = str(row.get("coarse_factor_solver", "")).strip() or "LU"
        groups = _fmt_optional_count(row.get("coarse_redundant_number", ""))
        return f"{factor.replace('_', r'\_')}, {groups} grp."
    return coarse_pc.replace("_", r"\_")


def _derivative_row_time(row: dict[str, str]) -> str:
    for key in ("solve_time_s", "total_time_s", "wall_time_s"):
        text = str(row.get(key, "")).strip()
        if text:
            return fmt_wall_time(float(text))
    return "--"


def _derivative_hessian_time(row: dict[str, str]) -> str:
    text = str(row.get("hessian_time_s", "")).strip()
    return "--" if not text else fmt_wall_time(float(text))


def _derivative_sfd_colors(row: dict[str, str]) -> str:
    if str(row.get("route", "")) != "colored_sfd":
        return "--"
    lo = str(row.get("sfd_colors_min", "")).strip()
    hi = str(row.get("sfd_colors_max", "")).strip()
    if not lo or not hi or int(float(hi)) <= 0:
        return "--"
    lo_i = int(float(lo))
    hi_i = int(float(hi))
    return str(hi_i) if lo_i == hi_i else f"{lo_i}--{hi_i}"


def plasticity2d_resolution_rows() -> list[dict[str, object]]:
    showcase = read_json(P2D_SHOWCASE)
    l5_result = showcase["result"]["steps"][0]
    rows: list[dict[str, object]] = [
        {
            "label": element_label("P4", "L5"),
            "free_dofs": int(showcase["mesh"]["free_dofs"]),
            "energy": float(l5_result["energy"]),
            "total_time_s": float(showcase["timings"]["total_time"]),
            "status": "endpoint converged",
            "note": "completed endpoint",
        }
    ]
    for path, ranks, label in (
        (P2D_L6_SUMMARY, 8, element_label("P4", "L6")),
        (P2D_L7_SUMMARY, 16, element_label("P4", "L7")),
    ):
        summary_rows = read_json(path)
        selected = next(row for row in summary_rows if int(row["ranks"]) == ranks)
        rows.append(
            {
                "label": label,
                "free_dofs": int(selected["free_dofs"]),
                "energy": float(selected["energy"]),
                "total_time_s": float(selected["total_time_sec"]),
                "status": str(selected["status"]),
                "note": f"fixed-iteration diagnostic at {ranks} ranks",
            }
        )
    return rows


def pass_fail(value: object) -> str:
    return "pass" if bool(value) else "fail"


def criterion_status(value: object) -> str:
    return "satisfied" if bool(value) else "not satisfied"


def _layer2_criterion_status(layer2_metrics: dict[str, object], key: str) -> str:
    acceptance = dict(layer2_metrics.get("acceptance", {}))
    if key not in acceptance:
        return "--"
    return criterion_status(acceptance[key])


def final_metric_header(rows: list[dict[str, object]]) -> str:
    names = {str(row.get("final_metric_name", "")).strip() for row in rows if row.get("final_metric_name")}
    if names == {"grad_norm"}:
        return "Final gradient norm"
    if names == {"relative_correction"}:
        return "Final relative correction"
    if len(names) == 1:
        return "Final " + next(iter(names)).replace("_", " ")
    return "Final convergence metric"


def sourcefixed_long_rows(
    local_rows: list[dict[str, object]], source_rows: list[dict[str, object]]
) -> list[list[str]]:
    rows_by_key = {
        (int(row["ranks"]), str(row["implementation"])): row
        for row in [*local_rows, *source_rows]
    }
    table_rows: list[list[str]] = []
    for rank in (4, 8, 16, 32):
        for implementation in (LOCAL_SOURCEFIXED_IMPL, SOURCE_SOURCEFIXED_IMPL):
            row = rows_by_key.get((rank, implementation))
            if row is None:
                table_rows.append([fmt_count(rank), implementation_label(implementation), "--", "--", "--", "--", "not run"])
                continue
            table_rows.append(
                [
                    fmt_count(rank),
                    implementation_label(implementation),
                    fmt_wall_time(float(row["wall_time_s"])),
                    fmt_count(row["nit"]),
                    fmt_count(row["linear_iterations_total"]),
                    fmt_sci(float(row["final_metric"])),
                    str(row["status"]),
                ]
            )
    return table_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready LaTeX tables.")
    parser.add_argument("--out-dir", type=Path, default=TABLES_ROOT)
    args = parser.parse_args()
    ensure_paper_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    local_rows = load_rows(LOCAL_P3D_SUMMARY)
    mixed_rows = load_rows(MIXED_P3D_SUMMARY)
    sourcefixed_rows = load_rows(SOURCEFIXED_P3D_SUMMARY)
    degree_energy_rows = load_rows(P3D_DEGREE_ENERGY_STUDY_SUMMARY)

    pl_rows = read_csv_rows(PLAPLACE_SCALING)
    gl_rows = read_csv_rows(GL_SCALING)
    he_rows = read_csv_rows(HE_SCALING)
    he_karolina_rows = [
        row for row in read_csv_rows(HE_KAROLINA_PMG_SCALING) if row.get("result", "completed") == "completed"
    ]
    he_karolina_rows.sort(key=lambda row: int(row["ranks"]))
    topo_rows = read_csv_rows(TOPO_SCALING)

    source8 = read_json(SOURCE_CONT_NP8)
    source32 = read_json(SOURCE_CONT_NP32)
    source8_progress = read_json(SOURCE_CONT_NP8_PROGRESS)
    source32_progress = read_json(SOURCE_CONT_NP32_PROGRESS)
    p3d_validation = read_json(P3D_VALIDATION_SUMMARY)
    p3d_ablation = read_json(P3D_DERIVATIVE_ABLATION_SUMMARY)
    jax_fem_baseline = read_json(JAX_FEM_BASELINE_SUMMARY)
    globalization_rows = globalization_method_rows()
    derivative_rows = derivative_route_rows()
    supplemental_he_distribution = supplemental_he_distribution_rows()
    supplemental_he_pmg = supplemental_he_pmg_rows()
    supplemental_topology = supplemental_topology_consistency_rows()
    supplemental_gl = supplemental_gl_globalization_rows()
    supplemental_p3d_degree = supplemental_p3d_derivative_degree_rows()

    local_scaling_rows = find_rows(local_rows, LOCAL_IMPL)
    mixed_local_rows = find_rows(mixed_rows, LOCAL_IMPL)
    mixed_source_rows = find_rows(mixed_rows, SOURCE_IMPL)
    sourcefixed_local_rows = find_rows(sourcefixed_rows, LOCAL_SOURCEFIXED_IMPL)
    sourcefixed_source_rows = find_rows(sourcefixed_rows, SOURCE_SOURCEFIXED_IMPL)
    p3d_local_karolina_rows = plasticity3d_local_karolina_rows()

    pl_showcase = select_csv_rows(PLAPLACE_PARITY, ("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"))
    gl_showcase = select_csv_rows(GL_PARITY, ("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"))
    he_showcase = select_csv_rows(HE_PARITY, ("fenics_custom", "jax_petsc_element", "jax_serial"))
    p2d_rows = plasticity2d_resolution_rows()
    topo_benchmark_rows = select_topology_rows(("serial_reference", "parallel_final"))

    pl_highlight = find_csv_row(pl_rows, "jax_petsc_element", 32)
    gl_highlight = find_csv_row(gl_rows, "jax_petsc_element", 32)
    he_highlight = find_csv_row(he_rows, "jax_petsc_element", 32)
    p2d_highlight = p2d_rows[-1]
    p3d_highlight = next(row for row in local_scaling_rows if int(row["ranks"]) == 32)
    topo_highlight = next(row for row in topo_rows if row["solver"] == "jax_parallel" and int(row["ranks"]) == 32)

    write_tablex(
        "family_highlights.tex",
        "@{}"
        + xspec((0.68, "RaggedRight"), (1.55, "RaggedRight"), (2.10, "RaggedRight"))
        + "@{}",
        ["Family", "Representative result", "Highlight"],
        [
            [
                "$p$-Laplace",
                f"JAX+PETSc element AD, {mesh_label('L9')}, 32 ranks: {fmt_wall_time(float(pl_highlight['total_time_s']))} s",
                "Exact element Hessians are competitive with FEniCS and faster than colored SFD on the finest reported case.",
            ],
            [
                "Ginzburg--Landau",
                f"JAX+PETSc element AD, {mesh_label('L9')}, 32 ranks: {fmt_wall_time(float(gl_highlight['total_time_s']))} s",
                "Element AD remains effectively tied with FEniCS custom Newton on the fine-grid benchmark.",
            ],
            [
                "Hyperelasticity",
                f"JAX+PETSc element AD, {mesh_label('L4')}, 32 ranks: {fmt_wall_time(float(he_highlight['total_time_s']))} s",
                "Hybrid trust-region/line-search globalization sustains large-deformation solves in distributed mode.",
            ],
            [
                "Plasticity (2D)",
                f"JAX+PETSc deep-tail PMG, {p2d_highlight['label']}, 16 ranks: {fmt_wall_time(float(p2d_highlight['total_time_s']))} s",
                "The dominant bottlenecks shift from the coarse end to the top smoother and repeated Krylov work.",
            ],
            [
                "Plasticity3D",
                f"constitutive-AD PMG solver, {element_label('P4', 'L1_2')}, $\\lambda_{{\\mathrm{{sr}}}}=\\num{{1.0}}$, 32 ranks: {fmt_wall_time(float(p3d_highlight['wall_time_s']))} s",
                "Auxiliary timing context for this load factor; the main glued-bottom discretization study uses $\\lambda_{\\mathrm{sr}}=\\num{1.55}$.",
            ],
            [
                "Topology",
                f"parallel JAX+PETSc, $768\\times384$, 32 ranks: {fmt_wall_time(float(topo_highlight['wall_time_s']))} s",
                "Distributed design updates and PETSc mechanics deliver stable fine-grid end-to-end timing while pure JAX remains the serial formulation reference.",
            ],
        ],
    )

    write_table(
        "implementation_capability_matrix.tex",
        "@{}l c c c " + pcol(r"0.38\textwidth") + "@{}",
        ["Family", "FEniCS", "pure JAX", "JAX+PETSc", "Advanced solver / derivative features"],
        [
            ["$p$-Laplace", "yes", "yes", "yes", "element AD and colored-SFD recovery"],
            ["Ginzburg--Landau", "yes", "no", "yes", "element AD and colored-SFD recovery"],
            ["Hyperelasticity", "yes", "yes", "yes", "element AD, colored SFD comparison, and trust-region solves"],
            ["Plasticity2D", "no", "no", "yes", "scalarized endpoint potential and same-mesh PMG"],
            ["Plasticity3D", "no", "no", "yes", "constitutive AD, element AD, and same-mesh PMG"],
            ["Topology optimization", "no", "yes", "yes", "distributed design updates and PETSc mechanics"],
        ],
    )

    write_table(
        "benchmark_specification_matrix.tex",
        "@{}"
        + "l c "
        + pcol(r"0.14\textwidth")
        + " "
        + pcol(r"0.23\textwidth")
        + " "
        + pcol(r"0.26\textwidth")
        + "@{}",
        ["Family", "Grid / mesh", "Solve policy", "Compared paths", "Main difficulty"],
        [
            ["$p$-Laplace", mesh_label("L9"), "Newton + line search", "FEniCS, pure JAX, JAX+PETSc", "nonlinear elliptic solve with exact sparse Hessians"],
            ["Ginzburg--Landau", mesh_label("L9"), "Newton + line search", "FEniCS, JAX+PETSc", "indefinite local curvature from the double well"],
            ["Hyperelasticity", f"{mesh_label('L4')}, 24 steps", "trust-region path", "FEniCS, pure JAX, JAX+PETSc", "nonconvex large-deformation mechanics"],
            ["Plasticity2D", f"{element_label('P4', 'L5')}--{element_label('P4', 'L7')}", "endpoint solve or fixed nonlinear work", "JAX+PETSc only", "same-mesh PMG and nonlinear tail behavior"],
            ["Plasticity3D", f"{degree_label('P1')}/{degree_label('P2')}/{degree_label('P4')}", "configuration-specific", "constitutive and reference PMG variants", "heterogeneous 3D Mohr--Coulomb with constitutive AD"],
            ["Topology", "$768\\times384$", "stall-stop continuation", "pure JAX, JAX+PETSc", "distributed design-mechanics coupling"],
        ],
    )

    write_tablex(
        "reference_availability.tex",
        "@{}"
        + xcol(1.08)
        + "@{\\hspace{0.8em}}"
        + xcol(0.58, "Centering")
        + "@{\\hspace{0.7em}}"
        + xcol(0.62, "Centering")
        + "@{\\hspace{1.0em}}"
        + xcol(1.72)
        + "@{}",
        ["Family", "FEniCS", "pure JAX", "Notes"],
        [
            ["$p$-Laplace", "yes", "yes", "All three stacks exist on representative reported cases."],
            ["Ginzburg--Landau", "yes", "no", "FEniCS and JAX+PETSc form the reported comparison."],
            ["Hyperelasticity", "yes", "yes", "pure JAX is a serial formulation reference only."],
            ["Plasticity2D", "no", "no", "The reported solver path is JAX+PETSc only."],
            [
                "Plasticity3D",
                "no",
                "no",
                "Reference-formula assembly exists as a supporting comparison route.",
            ],
            ["Topology", "no", "yes", "Parallel fine-grid path is JAX+PETSc; pure JAX remains the serial design reference."],
        ],
    )

    write_table_star(
        "sota_framework_comparison.tex",
        fill_spec(
            " ".join(
                [
                    pcol(r"0.16\textwidth"),
                    pcol(r"0.13\textwidth"),
                    pcol(r"0.15\textwidth"),
                    pcol(r"0.15\textwidth"),
                    pcol(r"0.14\textwidth"),
                    pcol(r"0.18\textwidth"),
                ]
            )
        ),
        ["Family", "Modeling", "Differentiation", "Second-order route", "Parallel", "Closest overlap"],
        [
            [
                "\\shortstack[l]{FEniCS\\\\DOLFINx\\\\\\citep{logg2012fenicsbook,baratta2025dolfinx}}",
                "High-level variational forms",
                "Symbolic form derivatives and generated kernels",
                "Manual or application-specific",
                "Distributed FEM assembly and solve",
                "Elliptic and finite-strain mechanics",
            ],
            [
                "\\shortstack[l]{dolfin-adjoint\\\\pyadjoint\\\\cashocs\\\\\\citep{farrell2013dolfinadjoint,mitusch2019pyadjoint,blauth2023cashocsv2}}",
                "High-level PDE plus optimization loop",
                "Adjoint-based first-order sensitivities",
                "Reduced-gradient and adjoint optimization emphasis",
                "MPI via the host FEM stack",
                "PDE control, shape, and topology",
            ],
            [
                "\\shortstack[l]{JAX-FEM\\\\Xue 2026\\\\\\citep{xue2023jaxfem,xue2026implicit}}",
                "JAX-native nonlinear FEM",
                "Program-level forward and reverse AD",
                "Implicit Hessian-vector products in inverse-problem tests",
                "JAX / GPU-oriented execution",
                "Nonlinear mechanics and inverse design",
            ],
            [
                "\\shortstack[l]{AutoPDEx\\\\\\citep{bode2025autopdex}}",
                "JAX-native PDE discretizations",
                "JAX AD and implicit differentiation",
                "Nonlinear minimizers and implicit derivatives",
                "JAX execution with optional PETSc integration",
                "Differentiable PDE solvers",
            ],
            [
                "\\shortstack[l]{JetSCI\\\\\\citep{cattaneo2026jetsci}}",
                "JAX local discretizations plus PETSc sparse solves",
                "JAX-differentiated discretization kernels",
                "Differentiable simulation kernels with PETSc solves",
                "JAX/GPU within node; PETSc MPI across nodes",
                "Heterogeneous micromechanics",
            ],
            [
                "\\shortstack[l]{Firedrake--JAX\\\\FEniCSx ext. ops\\\\\\citep{yashchuk2023bringing,latyshev2025externaloperators}}",
                "Host FEM stack plus AD bridge",
                "Tangent, adjoint, or local constitutive AD",
                "Local external-operator derivatives",
                "Host framework parallel back end",
                "Parameterized PDEs and constitutive models",
            ],
            [
                "\\shortstack[l]{JAX-CPFEM\\\\\\citep{hu2025jaxcpfem}}",
                "JAX-native crystal-plasticity FEM",
                "Differentiable constitutive simulator",
                "AD constitutive derivatives",
                "GPU-oriented execution",
                "Crystal plasticity",
            ],
            [
                "\\shortstack[l]{FEniTop\\\\\\citep{jia2024fenitop}}",
                "FEniCSx topology code",
                "Sensitivity-based design updates",
                "Compliance and sensitivity optimization loop",
                "Parallel FEniCSx realization",
                "2D and 3D topology optimization",
            ],
            [
                "This work",
                "JAX local energies plus PETSc sparse solvers",
                "Element AD, constitutive AD, and colored sparse finite differences",
                "Local Hessians/tangents or sparse recovery",
                "PETSc MPI vectors, matrices, nonlinear solvers/globalization, and multigrid",
                "$p$-Laplace, Ginzburg--Landau, hyperelasticity, plasticity, topology",
            ],
        ],
    )

    write_table_star(
        "plaplace_benchmark_summary.tex",
        fill_spec("l c c c c"),
        ["Path", "Energy", "Newton iters", "Krylov iters", "Wall time [s]"],
        [
            [
                implementation_label(row["implementation"]),
                fmt_energy(float(row["final_energy"]), precision=9),
                fmt_count(row["newton_iters"]),
                fmt_count(row["linear_iters"]),
                fmt_wall_time(float(row["wall_time_s"])),
            ]
            for row in pl_showcase
        ],
    )

    write_table_star(
        "ginzburg_landau_benchmark_summary.tex",
        fill_spec("l c c c c"),
        ["Path", "Energy", "Newton iters", "Krylov iters", "Wall time [s]"],
        [
            [
                implementation_label(row["implementation"]),
                fmt_energy(float(row["final_energy"]), precision=10),
                fmt_count(row["newton_iters"]),
                fmt_count(row["linear_iters"]),
                fmt_wall_time(float(row["wall_time_s"])),
            ]
            for row in gl_showcase
        ],
    )

    write_table_star(
        "hyperelasticity_benchmark_summary.tex",
        fill_spec("l c c c c"),
        ["Path", "Energy", "Steps", "Krylov iters", "Wall time [s]"],
        [
            [
                implementation_label(row["implementation"]),
                fmt_energy(float(row["final_energy"]), precision=6),
                fmt_count(row["completed_steps"]),
                fmt_count(row["total_linear_iters"]),
                fmt_wall_time(float(row["wall_time_s"])),
            ]
            for row in he_showcase
        ],
    )

    write_table_star(
        "hyperelasticity_karolina_pmg_scaling.tex",
        fill_spec("c c c c c c c c"),
        [
            "Ranks",
            "Nodes",
            "Coarse groups",
            "Ranks/group",
            "Solver total [s]",
            "First step [s]",
            "Newton iters",
            "Krylov iters",
        ],
        [
            [
                fmt_count(row["ranks"]),
                fmt_count(row["nodes"]),
                fmt_count(row["coarse_groups"]),
                fmt_count(row["coarse_group_ranks"]),
                fmt_wall_time(float(row["solver_total_s"])),
                fmt_wall_time(float(row["first_step_s"])),
                fmt_count(row["newton_iters"]),
                fmt_count(row["linear_iters"]),
            ]
            for row in he_karolina_rows
        ],
    )

    write_tablex_blocks(
        "globalization_method_compare.tex",
        [
            (
                "Outcome and terminal value",
                "@{}"
                + xspec((1.20, "RaggedRight"), (0.90, "RaggedRight"))
                + r"@{\hspace{0.45em}}c c c c c@{}",
                ["Benchmark", "Method", "Ranks", "Outcome", "Steps", "Time [s]", "Energy"],
                [
                    [
                        GLOBALIZATION_BENCHMARK_LABELS.get(str(row["benchmark"]), str(row["benchmark_label"])),
                        GLOBALIZATION_METHOD_LABELS.get(str(row["method"]), str(row["method"]).replace("_", r"\_")),
                        fmt_count(row["nprocs"]),
                        GLOBALIZATION_OUTCOME_LABELS.get(str(row["result"]), _result_label(row["result"])),
                        f"{fmt_count(row['completed_steps'])}/{fmt_count(row['steps_requested'])}",
                        _fmt_optional_wall(row.get("solve_time_s") or row.get("wall_time_s")),
                        _fmt_optional_energy(row.get("final_energy")),
                    ]
                    for row in globalization_rows
                ],
            ),
            (
                "Nonlinear and Krylov work",
                "@{}"
                + xspec((1.20, "RaggedRight"), (0.90, "RaggedRight"))
                + r"@{\hspace{0.45em}}c c c c c@{}",
                ["Benchmark", "Method", "Ranks", "Newton", "Krylov", "LS evals", "TR rejects"],
                [
                    [
                        GLOBALIZATION_BENCHMARK_LABELS.get(str(row["benchmark"]), str(row["benchmark_label"])),
                        GLOBALIZATION_METHOD_LABELS.get(str(row["method"]), str(row["method"]).replace("_", r"\_")),
                        fmt_count(row["nprocs"]),
                        fmt_count(row["newton_iters"]),
                        fmt_count(row["krylov_iters"]),
                        fmt_count(row["line_search_evals"]),
                        fmt_count(row["trust_rejects"]),
                    ]
                    for row in globalization_rows
                ],
            ),
        ],
    )

    write_tablex_blocks(
        "derivative_route_compare.tex",
        [
            (
                "Outcome and timing",
                "@{}"
                + xspec((1.25, "RaggedRight"), (0.95, "RaggedRight"))
                + r"@{\hspace{0.45em}}c c c@{}",
                ["Benchmark", "Route", "Outcome", "Time [s]", "Energy"],
                [
                    [
                        DERIVATIVE_BENCHMARK_LABELS.get(str(row["benchmark"]), str(row["benchmark_label"])),
                        DERIVATIVE_ROUTE_LABELS.get(str(row["route"]), str(row["route_label"])),
                        _result_label(row["result"]),
                        _derivative_row_time(row),
                        _fmt_optional_energy(row.get("final_energy")),
                    ]
                    for row in derivative_rows
                ],
            ),
            (
                "Work and Hessian construction",
                "@{}"
                + xspec((1.25, "RaggedRight"), (0.95, "RaggedRight"))
                + r"@{\hspace{0.45em}}c c c c c@{}",
                ["Benchmark", "Route", "Ranks", "Newton", "Krylov", "Hessian [s]", "SFD colors"],
                [
                    [
                        DERIVATIVE_BENCHMARK_LABELS.get(str(row["benchmark"]), str(row["benchmark_label"])),
                        DERIVATIVE_ROUTE_LABELS.get(str(row["route"]), str(row["route_label"])),
                        fmt_count(row["nprocs"]),
                        fmt_count(row["newton_iters"]),
                        fmt_count(row["krylov_iters"]),
                        _derivative_hessian_time(row),
                        _derivative_sfd_colors(row),
                    ]
                    for row in derivative_rows
                ],
            ),
        ],
    )

    write_tablex_blocks(
        "hyperelasticity_distribution_memory.tex",
        [
            (
                "Solve work",
                "@{}"
                + xspec((0.90, "RaggedRight"), (0.95, "RaggedRight"), (1.00, "RaggedRight"))
                + r"@{\hspace{0.45em}}c c c c c@{}",
                [
                    "Purpose",
                    "Assembly layout",
                    "Outcome",
                    "Level",
                    "Ranks",
                    "Newton",
                    "Krylov",
                    "Solve [s]",
                ],
                [
                    [
                        HE_DISTRIBUTION_PURPOSE_LABELS.get(
                            str(row.get("probe", "")),
                            str(row.get("probe", "")).replace("_", r"\_"),
                        ),
                        REVIEWER_HE_BUILD_LABELS.get(str(row.get("build_mode", "")), str(row.get("build_mode", ""))),
                        HE_DISTRIBUTION_OUTCOME_LABELS.get(
                            str(row.get("result", "")),
                            _result_label(row.get("result", "")),
                        ),
                        mesh_label(f"L{int(float(row.get('level') or 0))}"),
                        fmt_count(row["nprocs"]),
                        _fmt_optional_count(row.get("newton_iters", "")),
                        _fmt_optional_count(row.get("krylov_iters", "")),
                        _fmt_optional_wall(row.get("solve_time_s", "")),
                    ]
                    for row in supplemental_he_distribution
                ],
            ),
            (
                "Memory and overlap",
                "@{}"
                + xspec((0.95, "RaggedRight"), (1.05, "RaggedRight"))
                + r"@{\hspace{0.45em}}c c c c c@{}",
                [
                    "Purpose",
                    "Assembly layout",
                    "Level",
                    "Ranks",
                    "RSS max/sum [GiB]",
                    "Tracked sum [GiB]",
                    "Overlap/owned",
                ],
                [
                    [
                        HE_DISTRIBUTION_PURPOSE_LABELS.get(
                            str(row.get("probe", "")),
                            str(row.get("probe", "")).replace("_", r"\_"),
                        ),
                        REVIEWER_HE_BUILD_LABELS.get(str(row.get("build_mode", "")), str(row.get("build_mode", ""))),
                        mesh_label(f"L{int(float(row.get('level') or 0))}"),
                        fmt_count(row["nprocs"]),
                        _fmt_mib_pair(row.get("ru_maxrss_mib_max", ""), row.get("ru_maxrss_mib_total", "")),
                        _fmt_optional_gib(row.get("tracked_total_gib_total", "")),
                        _fmt_optional_float(row.get("overlap_owned_ratio", ""), 2),
                    ]
                    for row in supplemental_he_distribution
                ],
            ),
        ],
    )

    write_tablex_blocks(
        "hyperelasticity_pmg_sensitivity.tex",
        [
            (
                "Outcome and solver work",
                "@{}"
                + xspec((1.10, "RaggedRight"), (0.95, "RaggedRight"))
                + r"@{\hspace{0.45em}}c c c c@{}",
                ["Precond.", "Nonlinear work", "Newton", "Krylov", "Solver [s]", "Energy"],
                [
                    [
                        REVIEWER_HE_PMG_LABELS.get(str(row.get("candidate", "")), str(row.get("candidate", "")).replace("_", r"\_")),
                        HE_PMG_WORK_LABELS.get(str(row.get("result", "")), _result_label(row.get("result", ""))),
                        _fmt_optional_count(row.get("newton_iters", "")),
                        _fmt_optional_count(row.get("krylov_iters", "")),
                        _fmt_optional_wall(row.get("solve_time_s", "")),
                        _fmt_optional_energy(row.get("energy", "")),
                    ]
                    for row in supplemental_he_pmg
                ],
            ),
            (
                "Time breakdown and coarse solve",
                "@{}"
                + xspec((1.10, "RaggedRight"), (1.20, "RaggedRight"))
                + r"@{\hspace{0.45em}}c c c@{}",
                ["Precond.", "Coarse", "Assembly [s]", "PC [s]", "Linear [s]"],
                [
                    [
                        REVIEWER_HE_PMG_LABELS.get(str(row.get("candidate", "")), str(row.get("candidate", "")).replace("_", r"\_")),
                        _fmt_he_coarse(row),
                        _fmt_optional_wall(row.get("assemble_time_s", "")),
                        _fmt_optional_wall(row.get("pc_setup_time_s", "")),
                        _fmt_optional_wall(row.get("linear_solve_time_s", "")),
                    ]
                    for row in supplemental_he_pmg
                ],
            ),
        ],
    )

    write_table_star(
        "ginzburg_landau_globalization_fixed_budget.tex",
        fill_spec("l c c c c c c c c"),
        [
            "Method",
            "Outcome",
            "Newton",
            "Krylov",
            "LS evals",
            "TR rejects",
            "Setup [s]",
            "Solve [s]",
            "Energy",
        ],
        [
            [
                GLOBALIZATION_METHOD_LABELS.get(str(row.get("method", "")), str(row.get("method", "")).replace("_", r"\_")),
                _result_label(row.get("result", "")),
                _fmt_optional_count(row.get("newton_iters", "")),
                _fmt_optional_count(row.get("krylov_iters", "")),
                _fmt_optional_count(row.get("line_search_evals", "")),
                _fmt_optional_count(row.get("trust_rejects", "")),
                _fmt_optional_wall(row.get("setup_time_s", "")),
                _fmt_optional_wall(row.get("solve_time_s", "")),
                _fmt_optional_energy(row.get("energy", "")),
            ]
            for row in supplemental_gl
        ],
    )

    write_table_star(
        "topology_rank_consistency.tex",
        fill_spec("c c c c c c c c c"),
        [
            "Ranks",
            "Schedule",
            "Outer",
            "Solve [s]",
            "Compliance",
            "Volume",
            "$p$",
            "$\\Delta C/C_1$",
            "Density rel. $L^2$",
        ],
        [
            [
                fmt_count(row["nprocs"]),
                TOPOLOGY_SCHEDULE_LABELS.get(str(row.get("result", "")), _result_label(row.get("result", ""))),
                _fmt_optional_count(row.get("outer_iterations", "")),
                _fmt_optional_wall(row.get("solve_time_s", "")),
                _fmt_optional_float(row.get("final_compliance", ""), 4),
                _fmt_optional_float(row.get("final_volume_fraction", ""), 4),
                _fmt_optional_float(row.get("final_p", ""), 2),
                _fmt_optional_sci(row.get("compliance_rel_diff_vs_np1", "")),
                _fmt_optional_sci(row.get("density_rel_l2_vs_np1", "")),
            ]
            for row in supplemental_topology
        ],
    )

    write_tablex_blocks(
        "plasticity3d_derivative_degree.tex",
        [
            (
                "Discretization size",
                r"@{}c@{\hspace{0.45em}}"
                + xspec((1.10, "RaggedRight"), (0.90, "RaggedRight"))
                + r"@{\hspace{0.45em}}c c c c@{}",
                ["Element", "Route", "Linearization", "Free DOFs", "Elem DOFs", "Overlap DOFs", "RSS max [GiB]"],
                [
                    [
                        _p3d_discretization_label(row),
                        REVIEWER_P3D_ROUTE_LABELS.get(str(row.get("route", "")), str(row.get("route", "")).replace("_", r"\_")),
                        P3D_DERIVATIVE_DEGREE_WORK_LABELS.get(
                            str(row.get("result", "")),
                            _result_label(row.get("result", "")),
                        ),
                        _fmt_optional_dofs(row.get("free_dofs", "")),
                        _fmt_optional_count(row.get("local_element_dofs", "")),
                        _fmt_optional_count(row.get("local_overlap_dofs", "")),
                        (
                            _fmt_optional_float(float(row.get("ru_maxrss_mib_max", 0.0)) / 1024.0, 1)
                            if str(row.get("ru_maxrss_mib_max", "")).strip()
                            else "--"
                        ),
                    ]
                    for row in supplemental_p3d_degree
                ],
            ),
            (
                "Linearization cost",
                r"@{}c@{\hspace{0.45em}}"
                + xcol(1.0, "RaggedRight")
                + r"@{\hspace{0.45em}}c c c c c@{}",
                ["Element", "Route", "Free DOFs", "Krylov", "Solve [s]", "Hessian [s]", "SFD colors"],
                [
                    [
                        _p3d_discretization_label(row),
                        REVIEWER_P3D_ROUTE_LABELS.get(str(row.get("route", "")), str(row.get("route", "")).replace("_", r"\_")),
                        _fmt_optional_dofs(row.get("free_dofs", "")),
                        _fmt_optional_count(row.get("krylov_iters", "")),
                        _fmt_optional_wall(row.get("solve_time_s", "")),
                        _fmt_optional_wall(row.get("hessian_time_s", "")),
                        _derivative_sfd_colors(row),
                    ]
                    for row in supplemental_p3d_degree
                ],
            ),
        ],
    )

    write_table(
        "plasticity2d_benchmark_summary.tex",
        "@{}l c c c l@{}",
        ["Case", "Free DOFs", "Energy", "Wall time [s]", "Note"],
        [
            [
                str(row["label"]),
                fmt_dofs(row["free_dofs"]),
                fmt_energy(float(row["energy"])),
                fmt_wall_time(float(row["total_time_s"])),
                str(row["note"]),
            ]
            for row in p2d_rows
        ],
    )

    write_table_star(
        "plasticity3d_benchmark_summary.tex",
        fill_spec("l c c c c"),
        ["Element", "Free DOFs", "Energy", "$\\|g\\|_{\\mathrm{final}}$", "Wall time [s]"],
        [
            [
                element_label(row["degree_line"], row["mesh_alias"]),
                fmt_dofs(row["free_dofs"]),
                fmt_energy(float(row["energy"])),
                fmt_sci(float(row["final_grad_norm"])),
                fmt_wall_time(float(row["total_time_s"])),
            ]
            for row in sorted(
                degree_energy_rows,
                key=lambda row: (int(str(row["degree_line"]).replace("P", "")), int(row["free_dofs"])),
            )
        ],
    )

    write_table_star(
        "topology_benchmark_summary.tex",
        fill_spec("l c c c c c"),
        ["Case", "Ranks", "Outer iters", "Compliance", "Volume fraction", "Wall time [s]"],
        [
            [
                row["mesh"].replace("x", r"$\times$"),
                fmt_count(row["ranks"]),
                fmt_count(row["outer_iterations"]),
                fmt_float(float(row["final_compliance"]), 4),
                fmt_float(float(row["final_volume_fraction"]), 4),
                fmt_wall_time(float(row["wall_time_s"])),
            ]
            for row in topo_benchmark_rows
        ],
    )

    write_table_star(
        "plasticity3d_recommended_scaling.tex",
        fill_spec("c c c c c c c"),
        ["Ranks", "Wall time [s]", "Solve time [s]", "Speedup", "Efficiency", "Newton iters", "Krylov iters"],
        [
            [
                fmt_count(row["ranks"]),
                fmt_wall_time(float(row["wall_time_s"])),
                fmt_wall_time(float(row["solve_time_s"])),
                fmt_float(float(local_scaling_rows[0]["wall_time_s"]) / float(row["wall_time_s"])),
                fmt_float((float(local_scaling_rows[0]["wall_time_s"]) / float(row["wall_time_s"])) / float(row["ranks"])),
                fmt_count(row["nit"]),
                fmt_count(row["linear_iterations_total"]),
            ]
            for row in local_scaling_rows
        ],
    )

    write_table_star(
        "plasticity3d_local_karolina_scaling.tex",
        fill_spec("l c c c c c c c c c"),
        [
            "CPU setting",
            "Ranks",
            "Nodes",
            "Solver total [s]",
            "Solve [s]",
            "Newton iters",
            "Krylov iters",
            "Energy",
            "$\\|g\\|$",
            "$\\|g\\|$/target",
        ],
        [
            [
                str(row["source"]),
                fmt_count(row["ranks"]),
                "--" if row["nodes"] is None else fmt_count(row["nodes"]),
                fmt_wall_time(float(row["solver_total_s"])),
                fmt_wall_time(float(row["solve_time_s"])),
                fmt_count(row["newton_iterations"]),
                fmt_count(row["linear_iterations_total"]),
                fmt_energy(float(row["energy"])),
                fmt_float(float(row["grad"]), 3),
                fmt_float(float(row["grad_over_target"]), 3),
            ]
            for row in p3d_local_karolina_rows
        ],
    )

    write_table_star(
        "plasticity3d_local_karolina_partitioning.tex",
        fill_spec("l c c c c"),
        [
            "CPU setting",
            "Ranks",
            "Nodes",
            "Max local DOFs",
            "Overlap / owned DOFs",
        ],
        [
            [
                str(row["source"]),
                fmt_count(row["ranks"]),
                "--" if row["nodes"] is None else fmt_count(row["nodes"]),
                fmt_dofs(row["max_local_dofs"]),
                fmt_float(float(row["replication_ratio"]), 2),
            ]
            for row in p3d_local_karolina_rows
        ],
    )

    write_table_star(
        "plasticity3d_local_vs_source.tex",
        fill_spec("c c c c c c"),
        [
            "Ranks",
            "Constitutive wall [s]",
            "Reference wall [s]",
            "Constitutive solve [s]",
            "Reference solve [s]",
            "Ratio",
        ],
        [
            [
                fmt_count(lrow["ranks"]),
                fmt_wall_time(float(lrow["wall_time_s"])),
                fmt_wall_time(float(srow["wall_time_s"])),
                fmt_wall_time(float(lrow["solve_time_s"])),
                fmt_wall_time(float(srow["solve_time_s"])),
                fmt_float(float(lrow["wall_time_s"]) / float(srow["wall_time_s"])),
            ]
            for lrow, srow in zip(mixed_local_rows, mixed_source_rows, strict=True)
        ],
    )

    sourcefixed_rows = sourcefixed_long_rows(sourcefixed_local_rows, sourcefixed_source_rows)
    write_tablex_blocks(
        "plasticity3d_fixed_source_operator_pmg.tex",
        [
            (
                "Outcome and wall time",
                r"@{}c@{\hspace{0.7em}}"
                + xcol(1.0, "RaggedRight")
                + r"@{\hspace{0.45em}}c c@{}",
                ["Ranks", "Route", "Status", "Wall time [s]"],
                [[row[0], row[1], row[6], row[2]] for row in sourcefixed_rows],
            ),
            (
                "Newton--Krylov work",
                r"@{}c@{\hspace{0.7em}}"
                + xcol(1.0, "RaggedRight")
                + r"@{\hspace{0.45em}}c c c@{}",
                ["Ranks", "Route", "Newton iters", "Krylov iters", final_metric_header(sourcefixed_local_rows + sourcefixed_source_rows)],
                [[row[0], row[1], row[3], row[4], row[5]] for row in sourcefixed_rows],
            ),
        ],
    )

    write_table(
        "plasticity3d_degree_energy_study.tex",
        "@{}l c c c c@{}",
        ["Element", "Free DOFs", "Energy", "Wall time [s]", "Status"],
        [
            [
                element_label(row["degree_line"], row["mesh_alias"]),
                fmt_dofs(row["free_dofs"]),
                fmt_energy(float(row["energy"])),
                fmt_wall_time(float(row["total_time_s"])),
                "reused" if bool(row.get("reused", False)) else str(row["status"]),
            ]
            for row in degree_energy_rows
        ],
    )

    topo_best = [row for row in topo_rows if row["result"] == "completed"]
    topo_best.sort(key=lambda row: int(row["ranks"]))
    write_table_star(
        "topology_summary.tex",
        fill_spec("c c c c c c c"),
        ["Ranks", "Wall time [s]", "Solve time [s]", "Outer iters", "$p$", "Compliance", "Volume fraction"],
        [
            [
                fmt_count(row["ranks"]),
                fmt_wall_time(float(row["wall_time_s"])),
                fmt_wall_time(float(row["solve_time_s"])),
                fmt_count(row["outer_iterations"]),
                fmt_float(float(row["final_p_penal"]), 2),
                fmt_float(float(row["final_compliance"]), 4),
                fmt_float(float(row["final_volume_fraction"]), 4),
            ]
            for row in topo_best
        ],
    )

    write_table_star(
        "source_continuation_compare.tex",
        fill_spec("l c c c c c"),
        [
            "Policy",
            "Ranks",
            "Runtime [s]",
            "Init Krylov iters",
            "Continuation Krylov iters",
            "Final $\\lambda_{\\mathrm{sr}}$",
        ],
        [
            [
                "fixed PMG smoother policy",
                "8",
                fmt_wall_time(float(source8["run_info"]["runtime_seconds"])),
                fmt_count(source8["timings"]["linear"]["init_linear_iterations"]),
                fmt_count(source8["timings"]["linear"]["attempt_linear_iterations_total"]),
                fmt_float(float(source8_progress["lambda_last"]), 6),
            ],
            [
                "fixed PMG smoother policy",
                "32",
                fmt_wall_time(float(source32["run_info"]["runtime_seconds"])),
                fmt_count(source32["timings"]["linear"]["init_linear_iterations"]),
                fmt_count(source32["timings"]["linear"]["attempt_linear_iterations_total"]),
                fmt_float(float(source32_progress["lambda_last"]), 6),
            ],
        ],
    )

    layer1a_metrics = p3d_validation["layer1a"]["final_metrics"]
    layer2_metrics = p3d_validation["layer2"]
    endpoint_dev = layer2_metrics.get("endpoint_deviatoric_strain_relative_l2")
    write_table_star(
        "plasticity3d_validation_summary.tex",
        "@{}c@{\\hspace{1.0em}}" + pcol(r"0.36\textwidth") + r"@{\extracolsep{\fill}}c c@{}",
        ["Layer", "Comparison", "Relative difference", "Status"],
        [
            ["1A", "work", fmt_sci(float(layer1a_metrics["work_relative_difference"])), "--"],
            ["1A", "displacement relative $L^2$", fmt_sci(float(layer1a_metrics["displacement_relative_l2"])), "--"],
            [
                "1A",
                "deviatoric-strain relative $L^2$",
                fmt_sci(float(layer1a_metrics["deviatoric_strain_relative_l2"])),
                "--",
            ],
            [
                "2",
                "highest-successful $\\lambda_{\\mathrm{sr}}$",
                fmt_sci(float(layer2_metrics["critical_lambda_schedule_proxy"]["relative_difference"])),
                _layer2_criterion_status(layer2_metrics, "critical_lambda_pass"),
            ],
            [
                "2",
                "$u_{\\max}(\\lambda_{\\mathrm{sr}})$ relative $L^2$",
                fmt_sci(float(layer2_metrics["umax_curve_relative_l2"])),
                _layer2_criterion_status(layer2_metrics, "umax_curve_pass"),
            ],
            [
                "2",
                "endpoint displacement relative $L^2$",
                fmt_sci(float(layer2_metrics["endpoint_displacement_relative_l2"])),
                _layer2_criterion_status(layer2_metrics, "endpoint_disp_pass"),
            ],
            [
                "2",
                "endpoint deviatoric-strain relative $L^2$",
                fmt_sci(float(endpoint_dev)) if endpoint_dev is not None else "--",
                "diagnostic",
            ],
            [
                "2",
                "boundary profile relative $L^2$",
                fmt_sci(float(layer2_metrics["boundary_profile_relative_l2"])),
                "diagnostic",
            ],
            [
                "2",
                "acceptance criterion",
                "--",
                criterion_status(layer2_metrics["acceptance"]["overall_pass"]),
            ],
        ],
    )

    ablation_rows = [dict(row) for row in p3d_ablation["rows"]]
    write_tablex_blocks(
        "plasticity3d_derivative_ablation.tex",
        [
            (
                "Timing and work",
                "@{}" + xcol(1.0, "RaggedRight") + r"@{\hspace{0.45em}}c c c c@{}",
                ["Route", "Wall time [s]", "Solve time [s]", "Newton iters", "Krylov iters"],
                [
                    [
                        str(row["display_label"]),
                        fmt_wall_time(float(row["median_wall_time_s"])),
                        fmt_wall_time(float(row["median_solve_time_s"])),
                        fmt_count(row["median_nit"]),
                        fmt_count(row["median_linear_iterations_total"]),
                    ]
                    for row in ablation_rows
                ],
            ),
            (
                "Terminal observables",
                "@{}" + xcol(1.0, "RaggedRight") + r"@{\hspace{0.45em}}c c c@{}",
                ["Route", "Energy", "$\\omega$", "$u_{\\max}$"],
                [
                    [
                        str(row["display_label"]),
                        fmt_energy(float(row["median_energy"])),
                        fmt_energy(float(row["median_omega"])),
                        fmt_float(float(row["median_u_max"]), 6),
                    ]
                    for row in ablation_rows
                ],
            ),
        ],
    )

    fairness = dict(jax_fem_baseline["fairness_gate"])
    final_metrics = dict(jax_fem_baseline["final_metrics"])
    timing = dict(jax_fem_baseline["timing_medians_s"])
    fairness_checks = dict(fairness["checks"])
    agreement_pass = all(
        bool(fairness_checks[key])
        for key in (
            "energy_rel_diff_le_5pct",
            "field_relative_l2_le_5pct",
            "centerline_relative_l2_le_5pct",
            "umax_curve_relative_l2_le_5pct",
        )
    )
    write_table_star(
        "jax_fem_hyperelastic_baseline.tex",
        "@{}"
        + pcol(r"0.15\textwidth")
        + "@{\\hspace{1.0em}}"
        + pcol(r"0.30\textwidth")
        + r"@{\extracolsep{\fill}}c c c@{}",
        ["Group", "Quantity", "Relative difference", "Median wall time [s]", "Status"],
        [
            ["Agreement", "final energy", fmt_sci(float(final_metrics["energy_rel_diff"])), "--", "--"],
            ["Agreement", "full-field displacement relative $L^2$", fmt_sci(float(final_metrics["field_relative_l2"])), "--", "--"],
            ["Agreement", "centerline relative $L^2$", fmt_sci(float(final_metrics["centerline_relative_l2"])), "--", "--"],
            ["Agreement", "$u_{\\max}$ curve relative $L^2$", fmt_sci(float(final_metrics["umax_curve_relative_l2"])), "--", "--"],
            r"\addlinespace",
            ["Timing", "this work serial direct", "--", fmt_wall_time(float(timing["repo_serial_direct"])), "--"],
            ["Timing", "JAX-FEM UMFPACK serial", "--", fmt_wall_time(float(timing["jax_fem_umfpack_serial"])), "--"],
            r"\addlinespace",
            ["Condition", "common mesh", "--", "--", criterion_status(fairness_checks["same_mesh_path"])],
            ["Condition", "common displacement schedule", "--", "--", criterion_status(fairness_checks["same_schedule"])],
            ["Condition", "agreement threshold ($5\\%$)", "--", "--", criterion_status(agreement_pass)],
            ["Condition", "energy re-evaluation", "--", "--", "applied"],
        ],
    )

    payload = {
        "plasticity3d_recommended_scaling_rows": [
            {
                "ranks": int(row["ranks"]),
                "wall_time_s": float(row["wall_time_s"]),
                "solve_time_s": float(row["solve_time_s"]),
                "nit": int(row["nit"]),
                "linear_iterations_total": int(row["linear_iterations_total"]),
                "final_metric": float(row["final_metric"]),
                "final_metric_name": str(row.get("final_metric_name", "")),
            }
            for row in local_scaling_rows
        ],
        "plasticity3d_local_vs_source_rows": [
            {
                "ranks": int(lrow["ranks"]),
                "local_wall_time_s": float(lrow["wall_time_s"]),
                "source_wall_time_s": float(srow["wall_time_s"]),
                "local_solve_time_s": float(lrow["solve_time_s"]),
                "source_solve_time_s": float(srow["solve_time_s"]),
            }
            for lrow, srow in zip(mixed_local_rows, mixed_source_rows, strict=True)
        ],
        "plasticity3d_local_karolina_scaling": [
            {
                "source": str(row["source"]),
                "nodes": None if row["nodes"] is None else int(row["nodes"]),
                "ranks": int(row["ranks"]),
                "solver_total_s": float(row["solver_total_s"]),
                "solve_time_s": float(row["solve_time_s"]),
                "newton_iterations": int(row["newton_iterations"]),
                "linear_iterations_total": int(row["linear_iterations_total"]),
                "energy": float(row["energy"]),
                "grad": float(row["grad"]),
                "grad_over_target": float(row["grad_over_target"]),
                "max_local_dofs": int(row["max_local_dofs"]),
                "owned_free_dofs_sum": int(row["owned_free_dofs_sum"]),
                "overlap_total_dofs_sum": int(row["overlap_total_dofs_sum"]),
                "replication_ratio": float(row["replication_ratio"]),
            }
            for row in p3d_local_karolina_rows
        ],
        "globalization_method_compare": [
            {
                "benchmark": str(row["benchmark"]),
                "method": str(row["method"]),
                "nprocs": int(row["nprocs"]),
                "result": str(row["result"]),
                "completed_steps": int(row["completed_steps"]),
                "steps_requested": int(row["steps_requested"]),
                "newton_iters": int(row["newton_iters"]),
                "krylov_iters": int(row["krylov_iters"]),
                "line_search_evals": int(row["line_search_evals"]),
                "trust_rejects": int(row["trust_rejects"]),
                "solve_time_s": None if not str(row.get("solve_time_s", "")).strip() else float(row["solve_time_s"]),
                "wall_time_s": float(row["wall_time_s"]),
                "final_energy": None if not str(row.get("final_energy", "")).strip() else float(row["final_energy"]),
            }
            for row in globalization_rows
        ],
        "derivative_route_compare": [
            {
                "benchmark": str(row["benchmark"]),
                "route": str(row["route"]),
                "nprocs": int(row["nprocs"]),
                "result": str(row["result"]),
                "newton_iters": int(row["newton_iters"]),
                "krylov_iters": int(row["krylov_iters"]),
                "time_s": float(
                    str(row.get("solve_time_s") or row.get("total_time_s") or row.get("wall_time_s"))
                ),
                "hessian_time_s": (
                    None
                    if not str(row.get("hessian_time_s", "")).strip()
                    else float(row["hessian_time_s"])
                ),
                "sfd_colors_min": (
                    None
                    if not str(row.get("sfd_colors_min", "")).strip()
                    else int(float(row["sfd_colors_min"]))
                ),
                "sfd_colors_max": (
                    None
                    if not str(row.get("sfd_colors_max", "")).strip()
                    else int(float(row["sfd_colors_max"]))
                ),
                "final_energy": None if not str(row.get("final_energy", "")).strip() else float(row["final_energy"]),
            }
            for row in derivative_rows
        ],
        "plasticity3d_validation": {
            "layer1a_work_rel": float(layer1a_metrics["work_relative_difference"]),
            "layer2_acceptance": bool(layer2_metrics["acceptance"]["overall_pass"]),
        },
        "jax_fem_hyperelastic_baseline": {
            "fairness_gate_passed": bool(fairness["passed"]),
            "energy_rel_diff": float(final_metrics["energy_rel_diff"]),
        },
        "hyperelasticity_karolina_pmg_scaling": [
            {
                "nodes": int(row["nodes"]),
                "ranks": int(row["ranks"]),
                "coarse_groups": int(row["coarse_groups"]),
                "coarse_group_ranks": int(row["coarse_group_ranks"]),
                "solver_total_s": float(row["solver_total_s"]),
                "first_step_s": float(row["first_step_s"]),
                "newton_iters": int(row["newton_iters"]),
                "linear_iters": int(row["linear_iters"]),
                "energy": float(row["energy"]),
            }
            for row in he_karolina_rows
        ],
        "supplemental_solver_evidence": {
            "he_distribution_rows": len(supplemental_he_distribution),
            "he_pmg_rows": len(supplemental_he_pmg),
            "gl_globalization_rows": len(supplemental_gl),
            "topology_consistency_rows": len(supplemental_topology),
            "p3d_derivative_degree_rows": len(supplemental_p3d_degree),
        },
    }
    write_json(REPO_ROOT / "paper/build/tables_summary.json", payload)


if __name__ == "__main__":
    main()
