#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from common import REPO_ROOT, TABLES_ROOT, ensure_paper_dirs, read_csv_rows, read_json, write_json, write_text


PAPER_SUBMISSION_INPUT_ROOT = REPO_ROOT / "artifacts/reproduction/paper_submission_2026_07_08/inputs"
LOCAL_P3D_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_recommended_scaling/comparison_summary.json"
MIXED_P3D_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_reference_formula/comparison_summary.json"
SOURCEFIXED_P3D_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_fixed_reference_operator/table_summary.json"
P3D_DEGREE_ENERGY_STUDY_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / (
    "plasticity3d_degree_energy_study/comparison_summary.json"
)
P3D_VALIDATION_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_validation/comparison_summary.json"
P3D_DERIVATIVE_ABLATION_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / (
    "plasticity3d_derivative_ablation/comparison_summary.json"
)
JAX_FEM_BASELINE_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / "jax_fem_hyperelastic_baseline/comparison_summary.json"
GLOBALIZATION_METHOD_COMPARE = PAPER_SUBMISSION_INPUT_ROOT / "globalization_method_compare/full_summary.csv"
P3D_GLOBALIZATION_OUTPUTS = (
    PAPER_SUBMISSION_INPUT_ROOT
    / "globalization_method_compare/plasticity3d_p2_l1_np32_lambda155_newton_linesearch/output.json",
    PAPER_SUBMISSION_INPUT_ROOT / "globalization_method_compare/plasticity3d_p2_l1_np32_lambda155_steihaug_trust/output.json",
    PAPER_SUBMISSION_INPUT_ROOT
    / "globalization_method_compare/plasticity3d_p2_l1_np32_lambda155_hybrid_trust_linesearch/output.json",
)
DERIVATIVE_ROUTE_COMPARE = PAPER_SUBMISSION_INPUT_ROOT / "derivative_route_compare/full_summary.csv"
SUPPLEMENTAL_REPORT_ROOT = PAPER_SUBMISSION_INPUT_ROOT / "supplemental_solver_evidence"
SUPPLEMENTAL_HE_DISTRIBUTION = SUPPLEMENTAL_REPORT_ROOT / "full_he_distribution.csv"
SUPPLEMENTAL_HE_PMG = SUPPLEMENTAL_REPORT_ROOT / "full_he_pmg.csv"
SUPPLEMENTAL_TOPOLOGY_CONSISTENCY = SUPPLEMENTAL_REPORT_ROOT / "full_topology_consistency.csv"
SUPPLEMENTAL_GL_GLOBALIZATION = SUPPLEMENTAL_REPORT_ROOT / "full_gl_globalization.csv"
SUPPLEMENTAL_GL_TIMEOUT_ROOT = SUPPLEMENTAL_REPORT_ROOT / "gl_globalization/gl_l10_newton_linesearch_np8"
SUPPLEMENTAL_GL_TIMEOUT_RUN_INFO = SUPPLEMENTAL_GL_TIMEOUT_ROOT / "run_info.json"
SUPPLEMENTAL_GL_TIMEOUT_METADATA = SUPPLEMENTAL_GL_TIMEOUT_ROOT / "case_metadata.json"
SUPPLEMENTAL_P3D_DERIVATIVE_DEGREE = SUPPLEMENTAL_REPORT_ROOT / "full_p3d_derivative_degree.csv"
P3D_LOCAL_LAMBDA155_SCALING = PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_lambda155_scaling/local_solver_total_scaling.csv"
P3D_MULTINODE_LAMBDA155_SCALING = (
    PAPER_SUBMISSION_INPUT_ROOT / "plasticity3d_lambda155_scaling/karolina_rpn16_solver_total_scaling.csv"
)
P3D_LAMBDA155_STOP_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / (
    "plasticity3d_lambda155_scaling/step_grad_convergence_summary.csv"
)

PLAPLACE_PARITY = REPO_ROOT / "experiments/analysis/docs_assets/data/plaplace/parity_showcase.csv"
GL_PARITY = REPO_ROOT / "experiments/analysis/docs_assets/data/ginzburg_landau/parity_showcase.csv"
HE_PARITY = REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/parity_showcase.csv"

PLAPLACE_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/plaplace/strong_scaling.csv"
GL_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/ginzburg_landau/strong_scaling.csv"
HE_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/strong_scaling.csv"
HE_CPU_PMG_SCALING = (
    REPO_ROOT / "experiments/analysis/docs_assets/data/hyperelasticity/karolina_l5_pmg_scaling.csv"
)
TOPO_SCALING = REPO_ROOT / "experiments/analysis/docs_assets/data/topology/strong_scaling.csv"
TOPO_RESOLUTION = REPO_ROOT / "experiments/analysis/docs_assets/data/topology/resolution_objectives.csv"

P2D_SHOWCASE = PAPER_SUBMISSION_INPUT_ROOT / "plasticity2d_resolution/output.json"
P2D_L6_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / "plasticity2d_resolution/slope_stability_l6_p4/summary.json"
P2D_L7_SUMMARY = PAPER_SUBMISSION_INPUT_ROOT / "plasticity2d_resolution/slope_stability_l7_p4/summary.json"
SOURCE_CONT_NP8 = PAPER_SUBMISSION_INPUT_ROOT / "plasticity2d_reference_continuation/np8/run_info.json"
SOURCE_CONT_NP32 = PAPER_SUBMISSION_INPUT_ROOT / "plasticity2d_reference_continuation/np32/run_info.json"
SOURCE_CONT_NP8_PROGRESS = PAPER_SUBMISSION_INPUT_ROOT / (
    "plasticity2d_reference_continuation/np8/progress_latest.json"
)
SOURCE_CONT_NP32_PROGRESS = PAPER_SUBMISSION_INPUT_ROOT / (
    "plasticity2d_reference_continuation/np32/progress_latest.json"
)

TABLE_SOURCE_INPUTS = {
    "implementation_capability_matrix.tex": (),
    "benchmark_specification_matrix.tex": (),
    "reference_availability.tex": (),
    "solver_reporting_protocol.tex": (),
    "numerical_protocol_summary.tex": (),
    "sota_framework_comparison.tex": (),
    "plaplace_benchmark_summary.tex": (PLAPLACE_PARITY,),
    "ginzburg_landau_benchmark_summary.tex": (GL_PARITY,),
    "hyperelasticity_benchmark_summary.tex": (HE_PARITY,),
    "hyperelasticity_cpu_pmg_scaling.tex": (HE_CPU_PMG_SCALING,),
    "globalization_method_compare.tex": (GLOBALIZATION_METHOD_COMPARE, *P3D_GLOBALIZATION_OUTPUTS),
    "derivative_route_compare.tex": (DERIVATIVE_ROUTE_COMPARE,),
    "hyperelasticity_distribution_memory.tex": (SUPPLEMENTAL_HE_DISTRIBUTION,),
    "hyperelasticity_pmg_sensitivity.tex": (SUPPLEMENTAL_HE_PMG,),
    "ginzburg_landau_globalization_fixed_budget.tex": (
        SUPPLEMENTAL_GL_GLOBALIZATION,
        SUPPLEMENTAL_GL_TIMEOUT_RUN_INFO,
        SUPPLEMENTAL_GL_TIMEOUT_METADATA,
    ),
    "topology_rank_consistency.tex": (SUPPLEMENTAL_TOPOLOGY_CONSISTENCY,),
    "plasticity3d_derivative_degree.tex": (SUPPLEMENTAL_P3D_DERIVATIVE_DEGREE,),
    "plasticity2d_benchmark_summary.tex": (P2D_SHOWCASE, P2D_L6_SUMMARY, P2D_L7_SUMMARY),
    "plasticity3d_benchmark_summary.tex": (P3D_DEGREE_ENERGY_STUDY_SUMMARY,),
    "topology_benchmark_summary.tex": (TOPO_RESOLUTION,),
    "plasticity3d_recommended_scaling.tex": (LOCAL_P3D_SUMMARY,),
    "plasticity3d_cpu_scaling.tex": (
        P3D_LOCAL_LAMBDA155_SCALING,
        P3D_MULTINODE_LAMBDA155_SCALING,
        P3D_LAMBDA155_STOP_SUMMARY,
    ),
    "plasticity3d_cpu_partitioning.tex": (
        P3D_LOCAL_LAMBDA155_SCALING,
        P3D_MULTINODE_LAMBDA155_SCALING,
        P3D_LAMBDA155_STOP_SUMMARY,
    ),
    "plasticity3d_constitutive_vs_reference_formula.tex": (MIXED_P3D_SUMMARY,),
    "plasticity3d_fixed_reference_operator_pmg.tex": (SOURCEFIXED_P3D_SUMMARY,),
    "topology_summary.tex": (TOPO_SCALING,),
    "plasticity2d_reference_continuation.tex": (
        SOURCE_CONT_NP8,
        SOURCE_CONT_NP32,
        SOURCE_CONT_NP8_PROGRESS,
        SOURCE_CONT_NP32_PROGRESS,
    ),
    "plasticity3d_validation_summary.tex": (P3D_VALIDATION_SUMMARY,),
    "plasticity3d_derivative_ablation.tex": (P3D_DERIVATIVE_ABLATION_SUMMARY,),
    "jax_fem_hyperelastic_baseline.tex": (JAX_FEM_BASELINE_SUMMARY,),
}

LOCAL_IMPL = "local_constitutiveAD_local_pmg_armijo"
SOURCE_IMPL = "source_local_pmg_armijo"
LOCAL_SOURCEFIXED_IMPL = "constitutive_ad_fixed_reference_pmg"
SOURCE_SOURCEFIXED_IMPL = "reference_formula_fixed_reference_pmg"

IMPLEMENTATION_LABELS = {
    "fenics_custom": "FEniCS Newton reference",
    "jax_petsc_element": "JAX+PETSc element AD",
    "jax_petsc_local_sfd": "JAX+PETSc colored recovery",
    "jax_serial": "pure JAX serial formulation check",
    LOCAL_IMPL: "constitutive-AD PMG solver",
    SOURCE_IMPL: "frozen-preconditioner PMG variant",
    LOCAL_SOURCEFIXED_IMPL: "frozen PMG operator, AD branch tangent",
    SOURCE_SOURCEFIXED_IMPL: "frozen PMG operator, reference-formula branch tangent",
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
    "plasticity3d_p2_l1_np32_lambda155": "\\shortstack[l]{3D plasticity\\\\$P_2(L_1)$}",
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
        "\\shortstack[l]{3D plasticity $P_2(L_1)$\\\\$\\lambda_{\\mathrm{sr}}=\\num{1.55}$}"
    ),
}

DERIVATIVE_ROUTE_LABELS = {
    "element_ad": "Element AD",
    "colored_sfd": "Colored recovery (AD-HVP)",
    "constitutive_ad": "Constitutive AD",
}

REVIEWER_HE_BUILD_LABELS = {
    "replicated": "replicated",
    "rank_local": "rank-local",
}

HE_DISTRIBUTION_PURPOSE_LABELS = {
    "correctness": "fixed-work agreement",
    "memory": "one-linearization memory",
}

HE_DISTRIBUTION_OUTCOME_LABELS = {
    "completed": "completed",
    "fixed_work": "single linearization",
    "fixed_work_completed": "single linearization",
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
    "fixed_work": "one Newton step",
    "fixed_work_completed": "one Newton step",
}

REVIEWER_HE_PMG_LABELS = {
    "gamg": "GAMG",
    "pmg_l2_hypre": "PMG $L_2$ + Hypre",
    "pmg_l2_redundant_mumps": "PMG $L_2$ + MUMPS",
    "pmg_l3_redundant_mumps": "PMG $L_3$ + MUMPS",
}

REVIEWER_P3D_ROUTE_LABELS = {
    "element_ad": "Element AD",
    "colored_sfd": "Colored recovery (AD-HVP)",
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
    if "local_constitutiveAD" in key and "sourcefixed" in key:
        return "frozen PMG operator, AD branch tangent"
    if "local_constitutiveAD" in key and "local_pmg" in key:
        return "constitutive-AD PMG solver"
    if "sourcefixed" in key:
        return "frozen PMG operator, reference-formula branch tangent"
    if key.startswith("source") or "_source" in key:
        return "frozen-preconditioner PMG variant"
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


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Paper table manifest input is outside the repository: {path}") from exc


def _manifest_repo_input(path: Path) -> dict[str, str]:
    return {"kind": "repository_path", "path": _repo_rel(path)}


def _archive_status(name: str, inputs: tuple[Path, ...]) -> str:
    raw_prefixes = ("artifacts/raw_results/", "artifacts/reports/")
    for path in inputs:
        if _repo_rel(path).startswith(raw_prefixes):
            return "needs_final_archive"
    return "archive_neutral"


def _write_table_manifest(out_dir: Path) -> None:
    generated_tables = sorted(TABLE_SOURCE_INPUTS)
    generated_table_sources = {}
    generated_table_inputs = {}
    for name in generated_tables:
        inputs = TABLE_SOURCE_INPUTS[name]
        source = {
            "generator": {
                "path": _repo_rel(Path(__file__)),
                "function": "main",
                "output": name,
            },
            "archive_status": _archive_status(name, inputs),
            "data_inputs": [_manifest_repo_input(path) for path in inputs],
        }
        generated_table_sources[name] = source
        if source["archive_status"] == "archive_neutral":
            generated_table_inputs[name] = source["data_inputs"]
    notes = [
        "Static comparison tables have empty data_inputs because their content is defined directly in paper/scripts/generate_paper_tables.py.",
        "implementation_capability_matrix.tex is a curated static summary of solver components, derivative routes, and reference availability; numeric evidence appears in the result-specific tables.",
    ]
    if any(source["archive_status"] == "needs_final_archive" for source in generated_table_sources.values()):
        notes.insert(
            0,
            "Tables with archive_status=needs_final_archive have explicit source provenance but still depend on raw or report inputs that must be covered by the final durable archive.",
        )
    write_json(
        out_dir / "manifest.json",
        {
            "generated_tables": generated_tables,
            "generated_table_sources": generated_table_sources,
            "generated_table_inputs": generated_table_inputs,
            "notes": notes,
        },
    )


def select_csv_rows(path: Path, implementations: tuple[str, ...]) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    return [row for row in rows if row.get("implementation") in implementations]


def select_topology_rows(labels: tuple[str, ...]) -> list[dict[str, str]]:
    rows = read_csv_rows(TOPO_RESOLUTION)
    return [row for row in rows if row.get("label") in labels]


def plasticity3d_cpu_scaling_rows() -> list[dict[str, object]]:
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
                "nodes": 1,
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
    for row in read_csv_rows(P3D_MULTINODE_LAMBDA155_SCALING):
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
        "failed": "iteration cap",
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
        if groups == "1":
            return f"{factor.upper().replace('_', r'\_')}, one redundant group"
        return f"{factor.upper().replace('_', r'\_')}, {groups} redundant groups"
    if coarse_pc.lower() == "hypre":
        return "Hypre"
    return coarse_pc.replace("_", r"\_")


def _derivative_row_time(row: dict[str, str]) -> str:
    for key in ("solve_time_s", "total_time_s", "wall_time_s"):
        text = str(row.get(key, "")).strip()
        if text:
            return fmt_wall_time(float(text))
    return "--"


def _derivative_row_time_scope(row: dict[str, str]) -> str:
    for key, label in (
        ("solve_time_s", "solve"),
        ("total_time_s", "solver-internal"),
        ("wall_time_s", "wall"),
    ):
        if str(row.get(key, "")).strip():
            return label
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


def _resolve_repo_path(value: object) -> Path | None:
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.exists() else None


def _globalization_grad_over_target(row: dict[str, str]) -> str:
    if "plasticity3d" not in str(row.get("benchmark", "")):
        return "--"
    path = _resolve_repo_path(row.get("json_path", ""))
    if path is None:
        return "--"
    payload = read_json(path)
    grad = payload.get("final_grad_norm")
    target = payload.get("grad_stop_tol")
    if target in (None, ""):
        history = payload.get("history") or []
        if history:
            target = history[-1].get("grad_target")
    if grad in (None, "") or target in (None, ""):
        return "--"
    target_f = float(target)
    if target_f == 0.0:
        return "--"
    return fmt_sig(float(grad) / target_f, sig=3)


def _gl_timeout_run_info() -> dict[str, object]:
    if not SUPPLEMENTAL_GL_TIMEOUT_RUN_INFO.exists():
        return {}
    return read_json(SUPPLEMENTAL_GL_TIMEOUT_RUN_INFO)


def _gl_timeout_metadata() -> dict[str, object]:
    if not SUPPLEMENTAL_GL_TIMEOUT_METADATA.exists():
        return {}
    return read_json(SUPPLEMENTAL_GL_TIMEOUT_METADATA)


def _gl_solve_or_elapsed(row: dict[str, str]) -> str:
    solve = str(row.get("solve_time_s", "")).strip()
    if solve:
        return fmt_wall_time(float(solve))
    if row.get("case") == "gl_l10_newton_linesearch_np8" and row.get("result") == "timeout":
        return _fmt_optional_wall(_gl_timeout_run_info().get("wall_time_s", ""))
    return "--"


def _gl_wall_cap(row: dict[str, str]) -> str:
    if row.get("case") == "gl_l10_newton_linesearch_np8" and row.get("result") == "timeout":
        return _fmt_optional_wall(_gl_timeout_metadata().get("wall_cap_s", ""))
    return "--"


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
                "note": f"20-iteration capped diagnostic at {ranks} ranks",
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
                    _result_label(row["status"]),
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
    he_cpu_rows = [
        row for row in read_csv_rows(HE_CPU_PMG_SCALING) if row.get("result", "completed") == "completed"
    ]
    he_cpu_rows.sort(key=lambda row: int(row["ranks"]))
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
    p3d_cpu_scaling_rows = plasticity3d_cpu_scaling_rows()

    pl_showcase = select_csv_rows(PLAPLACE_PARITY, ("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"))
    gl_showcase = select_csv_rows(GL_PARITY, ("fenics_custom", "jax_petsc_element", "jax_petsc_local_sfd"))
    he_showcase = select_csv_rows(HE_PARITY, ("fenics_custom", "jax_petsc_element", "jax_serial"))
    p2d_rows = plasticity2d_resolution_rows()
    topo_benchmark_rows = select_topology_rows(("serial_reference", "parallel_final"))

    write_table(
        "implementation_capability_matrix.tex",
        "@{}l c c c " + pcol(r"0.40\textwidth") + "@{}",
        ["Family", "FEniCS", "pure JAX", "JAX+PETSc", "Solver and derivative roles"],
        [
            ["$p$-Laplace", "yes", "yes", "yes", "element AD and colored sparse recovery; Newton--CG with Hypre"],
            ["Ginzburg--Landau", "yes", "no", "yes", "element AD and colored sparse recovery; Armijo Newton with GMRES--Hypre"],
            ["Hyperelasticity", "yes", "yes", "yes", "element AD in the primary route, scoped colored sparse-recovery comparison, and trust-region/GAMG or PMG diagnostics"],
            ["2D Mohr--Coulomb", "no", "no", "yes", "endpoint branch-potential derivatives and continuation diagnostics with a same-mesh PMG hierarchy"],
            ["3D Mohr--Coulomb", "no", "no", "yes", "constitutive AD, element AD, and colored sparse-recovery diagnostics; FGMRES with same-mesh PMG and Hypre or LU/MUMPS coarse profiles"],
            ["Topology optimization", "no", "yes", "yes", "distributed design updates with PETSc mechanics and GAMG-preconditioned FGMRES"],
        ],
    )

    write_table(
        "benchmark_specification_matrix.tex",
        "@{}"
        + pcol(r"0.135\textwidth")
        + " "
        + pcol(r"0.185\textwidth")
        + " "
        + pcol(r"0.140\textwidth")
        + " "
        + pcol(r"0.275\textwidth")
        + " "
        + pcol(r"0.170\textwidth")
        + "@{}",
        ["Family", "Reported scope", "Solve policy", "Comparison evidence", "Main difficulty"],
        [
            ["$p$-Laplace", f"{mesh_label('L5')} serial agreement; {mesh_label('L9')} distributed scaling; {mesh_label('L10')} globalization diagnostic", "Newton + line search", "FEniCS and JAX+PETSc on the distributed case; pure JAX in the serial agreement case", "nonlinear elliptic solve with exact sparse Hessians"],
            ["Ginzburg--Landau", f"{mesh_label('L5')} agreement; {mesh_label('L9')} distributed scaling; {mesh_label('L10')} globalization diagnostic", "Newton + line search", "FEniCS and JAX+PETSc comparison", "indefinite local curvature from the double well"],
            ["Hyperelasticity", f"{mesh_label('L1')} serial agreement; {mesh_label('L4')}, 24-step distributed scaling; {mesh_label('L5')} PMG/memory diagnostics", "trust-region solve", "FEniCS and JAX+PETSc on the distributed suite; pure JAX only as a serial formulation check", "nonconvex large-deformation mechanics"],
            ["2D Mohr--Coulomb", f"{element_label('P4', 'L5')}--{element_label('P4', 'L7')}", "endpoint solve and capped fixed work", "JAX+PETSc endpoint and solver-policy evidence", "same-mesh PMG and nonlinear tail behavior"],
            ["3D Mohr--Coulomb", f"{degree_label('P1')}/{degree_label('P2')}/{degree_label('P4')}", "endpoint, scaling, and PMG-policy studies", "constitutive AD, reference-formula assembly diagnostic, and PMG-policy evidence", "heterogeneous 3D Mohr--Coulomb with constitutive AD"],
            ["Topology", "$192\\times96$ serial demonstration; $768\\times384$ parallel benchmark", "adaptive continuation", "pure JAX on the serial demonstration; JAX+PETSc on the fine-grid adaptive MPI timing study and controlled rank-consistency check", "distributed design-mechanics coupling"],
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
            ["$p$-Laplace", "yes", "yes", "All three implementations exist; pure JAX is used in the serial parity case."],
            ["Ginzburg--Landau", "yes", "no", "FEniCS and JAX+PETSc form the reported comparison."],
            ["Hyperelasticity", "yes", "yes", "pure JAX is a serial formulation check only."],
            ["2D Mohr--Coulomb", "no", "no", "The reported realization is JAX+PETSc only."],
            [
                "3D Mohr--Coulomb",
                "no",
                "no",
                "Reference-formula constitutive assembly is used only as a supporting comparison route.",
            ],
            ["Topology", "no", "yes", "Parallel fine-grid realization is JAX+PETSc; pure JAX remains a compact serial design demonstration."],
        ],
    )

    write_tablex(
        "solver_reporting_protocol.tex",
        "@{}" + xspec((0.62, "RaggedRight"), (1.33, "RaggedRight"), (1.05, "RaggedRight")) + "@{}",
        ["Reported term", "Definition in this paper", "Interpretation"],
        [
            [
                "completed",
                "All case-specific stopping tests are met before the nonlinear cap or wall-time cap.",
                "Energy and observables are the values at the stated benchmark endpoint.",
            ],
            [
                "timeout / iteration cap",
                "The wall-time limit or maximum nonlinear-iteration count is reached before the stated stopping tests are met.",
                "Retained as solver-behavior evidence, not as validation evidence.",
            ],
            [
                "fixed work",
                "A prescribed amount of work is performed, such as one Newton step, one linearization, or a fixed continuation schedule.",
                "Used for cost, memory, or policy diagnostics rather than endpoint convergence claims.",
            ],
            [
                "wall time",
                "Elapsed end-to-end time for the stated benchmark evaluation, including setup when that timing scope is specified.",
                "Comparable only for configurations that share the same benchmark contract and timing scope.",
            ],
            [
                "reported solve time",
                "Timer internal to the nonlinear or PETSc solve; setup, state export, and postprocessing may be outside this timer.",
                "Used for within-table solver comparisons, not as a universal replacement for wall time.",
            ],
            [
                "relative correction",
                "Norm of the accepted Newton correction relative to the current iterate scale; paired gradient checks use the absolute or relative target stated with the result.",
                "Defines the stopping metric in 3D plasticity configurations that report correction or gradient targets.",
            ],
        ],
    )

    write_tablex(
        "numerical_protocol_summary.tex",
        "@{}"
        + xspec(
            (0.70, "RaggedRight"),
            (0.75, "RaggedRight"),
            (1.05, "RaggedRight"),
            (0.82, "RaggedRight"),
            (1.00, "RaggedRight"),
            (0.68, "RaggedRight"),
        )
        + "@{}",
        ["Evidence block", "Role", "Solver policy", "KSP rtol / max iters", "Stop or fixed work", "Reported time"],
        [
            [
                "Scalar benchmarks",
                "formulation, derivative-route, globalization",
                "Newton--Armijo or stated trust-region variant; PETSc entries use CG/GMRES with Hypre",
                "$p$-Laplace: $10^{-1}$/30; Ginzburg--Landau: $10^{-3}$/200",
                "case-specific nonlinear tolerances; capped configurations diagnostic",
                "wall or solve/elapsed time",
            ],
            [
                "Hyperelasticity",
                "finite-strain mechanics, JAX-FEM comparison, scaling",
                "trust-region or line-search Newton; GAMG or PMG as stated",
                "GAMG entries: $10^{-1}$/30; PMG tolerances stated in diagnostic entries",
                "per-load stationarity; fixed-step entries diagnostic",
                "wall time; reported solve time for first-step scaling",
            ],
            [
                "2D Mohr--Coulomb",
                "endpoint evidence and fixed-work diagnostics",
                "Armijo continuation with same-mesh PMG",
                "typically $10^{-2}$/15 unless a diagnostic states otherwise",
                "$P_4(L_5)$ endpoint; $P_4(L_6)$--$P_4(L_7)$ fixed work",
                "wall time",
            ],
            [
                "3D Mohr--Coulomb validation",
                "endpoint-surrogate agreement, not path-history validation",
                r"fixed-$\lambda_{\mathrm{sr}}$ comparator diagnostics with matched boundary data",
                "not used for timing comparison",
                r"fixed-$\lambda_{\mathrm{sr}}$ observables; strain/profile entries diagnostic",
                "secondary to agreement metrics",
            ],
            [
                "3D Mohr--Coulomb performance",
                "second-order routes, globalization, parallel scaling",
                "Armijo, residual-bisection, or trust-region Newton with FGMRES/PMG",
                "$10^{-1}$ or $10^{-2}$ with max 100, as stated by configuration",
                "relative correction, gradient target, cap, or fixed work",
                "wall, solve, or solver-total time",
            ],
            [
                "Topology",
                "distributed design-mechanics timing and rank consistency",
                "adaptive reduced-objective continuation with PETSc mechanics and GAMG",
                "mechanics FGMRES/GAMG: $10^{-4}$/100",
                "design/state-change stall criterion or fixed schedule",
                "end-to-end wall time",
            ],
        ],
    )

    write_table_star(
        "sota_framework_comparison.tex",
        fill_spec(
            " ".join(
                [
                    pcol(r"0.23\textwidth"),
                    pcol(r"0.29\textwidth"),
                    pcol(r"0.39\textwidth"),
                ]
            )
        ),
        ["Technical family", "Documented role", "Relation to this work"],
        [
            [
                "FEM and adjoint automation "
                "\\citep{logg2012fenicsbook,baratta2025dolfinx,farrell2013dolfinadjoint,mitusch2019pyadjoint,blauth2023cashocsv2}",
                "High-level variational forms, generated kernels, adjoint-based sensitivities, and PDE-optimization loops.",
                "Provides the reference high-level FEM and optimization context; selected \\fenics{} formulations are scoped comparisons.",
            ],
            [
                "JAX-native differentiable FEM and PDE solvers "
                "\\citep{xue2023jaxfem,xue2026implicit,bode2025autopdex,hu2025jaxcpfem}",
                "Source-specific roles: GPU-oriented differentiable mechanics (JAX-FEM/JAX-CPFEM), implicit differentiation and solver options (AutoPDEx), and Hessian-vector inverse-problem studies (Xue 2026).",
                "Motivates local differentiable modeling; the present study instead couples local JAX derivatives to PETSc sparse MPI solves and compares derivative routes.",
            ],
            [
                "FEM--JAX and JAX--PETSc bridge architectures "
                "\\citep{yashchuk2023bringing,latyshev2025externaloperators,cattaneo2026jetsci}",
                "Source-specific bridges: Firedrake--JAX variational coupling, FEniCSx external operators for constitutive models, and JetSCI-style JAX--PETSc differentiated local discretizations.",
                "Architectural context for the \\jaxpetsc{} realization; this study evaluates derivative construction, globalization, and preconditioning on the stated benchmark suite.",
            ],
            [
                "Slope-stability and elastoplastic implementation context "
                "\\citep{tschuchnigg2015nonassociated,sysala2017returnmapping,sysala2021optimization,sysala2025convexoptimization,sysala2025advancedcontinuation,cermak2019efficient}",
                "Strength-reduction plasticity, nonsmooth return mapping, continuation, and practical 2D/3D elastoplastic implementation.",
                "Sysala-family works provide slope-stability reference-model context; \\v{C}erm{\\'a}k--Sysala--Valdman provides practical elastoplastic implementation context. Numerical claims remain limited to the implemented endpoint surrogate and reported comparison observables.",
            ],
            [
                "Topology-optimization benchmark lineage "
                "\\citep{sigmund2001topology,bendsoe2003topology,ferrari2020top99,bourdin2001filters,jia2024fenitop}",
                "SIMP compliance minimization, density filtering, compact educational implementations, and modern parallel topology software.",
                "Defines the topology problem context; the numerical claims are the reported design-and-mechanics timing and rank-consistency diagnostics.",
            ],
        ],
    )

    write_table_star(
        "plaplace_benchmark_summary.tex",
        fill_spec("l c c c c"),
        ["Solver realization", "Energy", "Newton iters", "Krylov iters", "Wall time [s]"],
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
        ["Solver realization", "Energy", "Newton iters", "Krylov iters", "Wall time [s]"],
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
        ["Solver realization", "Energy", "Steps", "Krylov iters", "Wall time [s]"],
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
        "hyperelasticity_cpu_pmg_scaling.tex",
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
            for row in he_cpu_rows
        ],
    )

    write_tablex_blocks(
        "globalization_method_compare.tex",
        [
            (
                "Outcome and endpoint value",
                "@{}"
                + xspec((1.20, "RaggedRight"), (0.90, "RaggedRight"))
                + r"@{\hspace{0.45em}}c c c c c c@{}",
                [
                    "Benchmark",
                    "Method",
                    "Ranks",
                    "Outcome",
                    r"\shortstack{Completed/requested\\steps}",
                    "Reported time [s]",
                    "Grad/target",
                    "Energy",
                ],
                [
                    [
                        GLOBALIZATION_BENCHMARK_LABELS.get(str(row["benchmark"]), str(row["benchmark_label"])),
                        GLOBALIZATION_METHOD_LABELS.get(str(row["method"]), str(row["method"]).replace("_", r"\_")),
                        fmt_count(row["nprocs"]),
                        GLOBALIZATION_OUTCOME_LABELS.get(str(row["result"]), _result_label(row["result"])),
                        f"{fmt_count(row['completed_steps'])}/{fmt_count(row['steps_requested'])}",
                        _fmt_optional_wall(row.get("solve_time_s") or row.get("wall_time_s")),
                        _globalization_grad_over_target(row),
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
                + r"@{\hspace{0.45em}}c c c c@{}",
                ["Benchmark", "Route", "Outcome", "Time [s]", "Timing scope", "Energy"],
                [
                    [
                        DERIVATIVE_BENCHMARK_LABELS.get(str(row["benchmark"]), str(row["benchmark_label"])),
                        DERIVATIVE_ROUTE_LABELS.get(str(row["route"]), str(row["route_label"])),
                        _result_label(row["result"]),
                        _derivative_row_time(row),
                        _derivative_row_time_scope(row),
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
                ["Benchmark", "Route", "Ranks", "Newton", "Krylov", "Hessian [s]", "Color groups"],
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
                + r"@{\hspace{0.45em}}c c c c c c@{}",
                [
                    "Purpose",
                    "Assembly layout",
                    "Outcome",
                    "Level",
                    "Ranks",
                    "Newton",
                    "Krylov",
                    "Solve [s]",
                    "Energy",
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
                        _fmt_optional_float(row.get("energy", ""), 6),
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
            "Reported time [s]",
            "Cap [s]",
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
                _gl_solve_or_elapsed(row),
                _gl_wall_cap(row),
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
            "Rel. compliance diff.",
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
                ["Element", "Route", "Work", "Free DOFs", "Elem. DOFs", "Overlap DOFs", "RSS [GiB]"],
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
                ["Element", "Route", "Free DOFs", "Krylov", "Solve [s]", "Hessian [s]", "Colors"],
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
        "plasticity3d_cpu_scaling.tex",
        fill_spec("l c c c c c c c"),
        [
            "CPU setting",
            "Ranks",
            "Nodes",
            "Solver total [s]",
            "Newton iters",
            "Krylov iters",
            "Energy",
            "$\\|g\\|_{\\mathrm{stop}}$/target",
        ],
        [
            [
                str(row["source"]),
                fmt_count(row["ranks"]),
                "--" if row["nodes"] is None else fmt_count(row["nodes"]),
                fmt_wall_time(float(row["solver_total_s"])),
                fmt_count(row["newton_iterations"]),
                fmt_count(row["linear_iterations_total"]),
                fmt_energy(float(row["energy"])),
                fmt_float(float(row["grad_over_target"]), 3),
            ]
            for row in p3d_cpu_scaling_rows
        ],
    )

    write_table_star(
        "plasticity3d_cpu_partitioning.tex",
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
            for row in p3d_cpu_scaling_rows
        ],
    )

    write_table_star(
        "plasticity3d_constitutive_vs_reference_formula.tex",
        fill_spec("c c c c c c"),
        [
            "Ranks",
            "AD wall [s]",
            "Reference-formula wall [s]",
            "AD solve [s]",
            "Reference-formula solve [s]",
            "Wall ratio",
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
        "plasticity3d_fixed_reference_operator_pmg.tex",
        [
            (
                "Outcome and wall time",
                r"@{}c@{\hspace{0.7em}}"
                + xcol(1.0, "RaggedRight")
                + r"@{\hspace{0.45em}}c c@{}",
                ["Ranks", "PMG operator and tangent", "Status", "Wall time [s]"],
                [[row[0], row[1], row[6], row[2]] for row in sourcefixed_rows],
            ),
            (
                "Newton--Krylov work",
                r"@{}c@{\hspace{0.7em}}"
                + xcol(1.0, "RaggedRight")
                + r"@{\hspace{0.45em}}c c c@{}",
                [
                    "Ranks",
                    "PMG operator and tangent",
                    "Newton iters",
                    "Krylov iters",
                    final_metric_header(sourcefixed_local_rows + sourcefixed_source_rows),
                ],
                [[row[0], row[1], row[3], row[4], row[5]] for row in sourcefixed_rows],
            ),
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
        "plasticity2d_reference_continuation.tex",
        fill_spec("l c c c c c"),
        [
            "Policy",
            "Ranks",
            "Wall time [s]",
            "Init Krylov iters",
            "Continuation Krylov iters",
            "Final $\\lambda_{\\mathrm{sr}}$",
        ],
        [
            [
                "fixed PMG policy",
                "8",
                fmt_wall_time(float(source8["run_info"]["runtime_seconds"])),
                fmt_count(source8["timings"]["linear"]["init_linear_iterations"]),
                fmt_count(source8["timings"]["linear"]["attempt_linear_iterations_total"]),
                fmt_float(float(source8_progress["lambda_last"]), 6),
            ],
            [
                "fixed PMG policy",
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
        "@{}" + pcol(r"0.25\textwidth") + "@{\\hspace{1.0em}}" + pcol(r"0.36\textwidth") + r"@{\extracolsep{\fill}}c c@{}",
        ["Check", "Comparison", "Relative difference", "Status"],
        [
            ["endpoint observable", "work", fmt_sci(float(layer1a_metrics["work_relative_difference"])), "reported"],
            ["endpoint observable", "displacement relative discrete norm", fmt_sci(float(layer1a_metrics["displacement_relative_l2"])), "reported"],
            [
                "endpoint observable",
                "deviatoric-strain relative discrete norm",
                fmt_sci(float(layer1a_metrics["deviatoric_strain_relative_l2"])),
                "reported",
            ],
            [
                r"fixed-$\lambda_{\mathrm{sr}}$ comparison",
                "relative difference in $\\lambda_{\\max}^{\\mathrm{succ}}$",
                fmt_sci(float(layer2_metrics["critical_lambda_schedule_proxy"]["relative_difference"])),
                _layer2_criterion_status(layer2_metrics, "critical_lambda_pass"),
            ],
            [
                r"fixed-$\lambda_{\mathrm{sr}}$ comparison",
                "$u_{\\max}(\\lambda_{\\mathrm{sr}})$ relative Euclidean curve norm",
                fmt_sci(float(layer2_metrics["umax_curve_relative_l2"])),
                _layer2_criterion_status(layer2_metrics, "umax_curve_pass"),
            ],
            [
                r"fixed-$\lambda_{\mathrm{sr}}$ comparison",
                "endpoint displacement relative discrete norm",
                fmt_sci(float(layer2_metrics["endpoint_displacement_relative_l2"])),
                _layer2_criterion_status(layer2_metrics, "endpoint_disp_pass"),
            ],
            [
                r"fixed-$\lambda_{\mathrm{sr}}$ comparison",
                "endpoint deviatoric-strain relative discrete norm",
                fmt_sci(float(endpoint_dev)) if endpoint_dev is not None else "--",
                "diagnostic",
            ],
            [
                r"fixed-$\lambda_{\mathrm{sr}}$ comparison",
                "upper-slope profile relative Euclidean curve norm",
                fmt_sci(float(layer2_metrics["boundary_profile_relative_l2"])),
                "diagnostic",
            ],
            [
                r"fixed-$\lambda_{\mathrm{sr}}$ comparison",
                r"fixed-$\lambda_{\mathrm{sr}}$ criteria, 3/3 satisfied",
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
                        REVIEWER_P3D_ROUTE_LABELS.get(str(row.get("route", "")), str(row["display_label"])),
                        fmt_wall_time(float(row["median_wall_time_s"])),
                        fmt_wall_time(float(row["median_solve_time_s"])),
                        fmt_count(row["median_nit"]),
                        fmt_count(row["median_linear_iterations_total"]),
                    ]
                    for row in ablation_rows
                ],
            ),
            (
                "Endpoint observables",
                "@{}" + xcol(1.0, "RaggedRight") + r"@{\hspace{0.45em}}c c c@{}",
                ["Route", "Energy", "$\\omega$", "$u_{\\max}$"],
                [
                    [
                        REVIEWER_P3D_ROUTE_LABELS.get(str(row.get("route", "")), str(row["display_label"])),
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
        + r"@{\extracolsep{\fill}}c c@{}",
        ["Group", "Quantity", "Relative difference", "Status"],
        [
            ["Agreement", "final energy", fmt_sci(float(final_metrics["energy_rel_diff"])), "below $5\\%$"],
            ["Agreement", "full-field displacement relative Euclidean", fmt_sci(float(final_metrics["field_relative_l2"])), "below $5\\%$"],
            ["Agreement", "centerline relative Euclidean curve norm", fmt_sci(float(final_metrics["centerline_relative_l2"])), "below $5\\%$"],
            ["Agreement", "$u_{\\max}$ curve relative Euclidean norm", fmt_sci(float(final_metrics["umax_curve_relative_l2"])), "below $5\\%$"],
            r"\addlinespace",
            ["Condition", "common mesh", "--", criterion_status(fairness_checks["same_mesh_path"])],
            ["Condition", "common displacement schedule", "--", criterion_status(fairness_checks["same_schedule"])],
            ["Condition", "agreement threshold ($5\\%$)", "--", criterion_status(agreement_pass)],
            ["Condition", "energy re-evaluation", "--", "applied"],
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
        "plasticity3d_cpu_scaling": [
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
            for row in p3d_cpu_scaling_rows
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
        "hyperelasticity_cpu_pmg_scaling": [
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
            for row in he_cpu_rows
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
    _write_table_manifest(args.out_dir)


if __name__ == "__main__":
    main()
