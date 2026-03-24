"""Assignment-facing metadata and acceptance rules for thesis replication."""

from __future__ import annotations

from dataclasses import dataclass

from .tables import TABLE_5_8_RMPA_BY_LEVEL, TABLE_5_10_OA1_BY_LEVEL


@dataclass(frozen=True)
class AssignmentTarget:
    """Metadata for one thesis benchmark target."""

    table: str
    stage: str
    section: str
    target: str
    family: str
    primary: bool
    reference_kind: str
    caveat: str | None = None


THESIS_PROBLEM_STATEMENT = (
    "Solve the Dirichlet p-Laplacian Lane-Emden problem "
    "-Δ_p u = u^3 in Ω with u = 0 on ∂Ω for p in (4/3, 4)."
)

THESIS_FUNCTIONAL_SUMMARY = (
    "Weak solutions are critical points of J(u) = (1/p)∫|∇u|^p - (1/4)∫u^4, "
    "while OA1/OA2 work with the scale-invariant quotient "
    "I(u) = ||u||_{1,p,0} / ||u||_{L^4(Ω)}."
)

THESIS_GEOMETRY_SUMMARY = (
    "Primary geometry: square Ω = [0, π] x [0, π]. "
    "Secondary geometry: square with centered square hole "
    "Ω = ([0, π]^2) \\ ((π/4, 3π/4)^2)."
)

THESIS_DISCRETIZATION_SUMMARY = (
    "The thesis uses structured uniform right-triangle P1 finite elements with "
    "h = π / 2^L and continuous nodal basis functions on the interior nodes."
)

THESIS_SEED_SUMMARY = (
    "Principal branch seed: sin(x) sin(y). "
    "Square multibranch seeds: sin(x) sin(y), 10 sin(2x) sin(y), "
    "10 sin(x) sin(2y), 4(x-y) sin(x) sin(y). "
    "Square-hole seeds: sin(x) sin(y), 4|sin(x) sin(2y)|, "
    "4(x-y) sin(x) sin(y), |4 sin(3x) sin(3y)|."
)

REPLICATION_LEGEND = {
    "exact": "Direct thesis quantity comparison such as J, I, 1/I, or iteration counts.",
    "low impact": "Completed replication with matched thesis substance but a documented secondary discrepancy such as a moderate iteration-count overrun.",
    "proxy": "Reference-error column computed with a modern proxy reference solve rather than the thesis Section 3.3.1 pipeline.",
    "unmatched": "Current repo output is outside the assignment acceptance rule for that target.",
}

TABLE_5_13_LOW_IMPACT_RATIO = 1.5
TABLE_5_13_EXTREME_RATIO = 10.0
TABLE_5_13_PRINCIPAL_BRANCH_TOL = 2.0e-2

ASSIGNMENT_STAGE_DETAILS = {
    "Calibration": "Internal frozen-constant sweep used to choose thesis reproduction presets.",
    "Optional 1D Harness": "Section 18; cheap stopping and direction sanity check on (0, π).",
    "Stage A": "Section 13 Stage A; principal branch on the square with RMPA.",
    "Stage B": "Section 13 Stage B; cross-check the same branch with OA1.",
    "Stage C": "Section 13 Stage C; compare method behavior and iteration counts.",
    "Stage D": "Section 13 Stage D; multiple branches on the square with OA2.",
    "Stage E": "Section 13 Stage E; multiple branches on the square-with-hole with OA2.",
    "Quick Smoke": "Repository quick smoke cases, not part of the thesis assignment.",
}

METHOD_TO_TABLES = {
    "MPA": ["Table 5.6", "Table 5.7", "Table 5.12"],
    "RMPA": ["Table 5.8", "Table 5.9", "Table 5.12", "Table 5.13"],
    "OA1": ["Table 5.10", "Table 5.11", "Table 5.12", "Table 5.13", "Table 5.14"],
    "OA2": ["Table 5.14", "Figure 5.13"],
}

_TARGETS = {
    "calibration": AssignmentTarget(
        table="calibration",
        stage="Calibration",
        section="internal preset calibration",
        target="Frozen thesis reproduction constant sweep",
        family="calibration",
        primary=False,
        reference_kind="exact",
        caveat="Calibration rows are internal tuning diagnostics, not thesis deliverable rows.",
    ),
    "quick": AssignmentTarget(
        table="quick",
        stage="Quick Smoke",
        section="repository quick checks",
        target="Quick smoke coverage",
        family="smoke",
        primary=False,
        reference_kind="proxy",
    ),
    "table_5_2": AssignmentTarget(
        table="table_5_2",
        stage="Optional 1D Harness",
        section="Section 18 / Table 5.2",
        target="1D direction study with exact d",
        family="1d_harness",
        primary=False,
        reference_kind="proxy",
    ),
    "table_5_2_drn_sanity": AssignmentTarget(
        table="table_5_2_drn_sanity",
        stage="Optional 1D Harness",
        section="Section 18 / d^R_N sanity",
        target="1D d^R_N sanity check",
        family="1d_harness",
        primary=False,
        reference_kind="proxy",
    ),
    "table_5_3": AssignmentTarget(
        table="table_5_3",
        stage="Optional 1D Harness",
        section="Section 18 / Table 5.3",
        target="1D direction study with d^V_h",
        family="1d_harness",
        primary=False,
        reference_kind="proxy",
    ),
    "table_5_6": AssignmentTarget(
        table="table_5_6",
        stage="Stage C",
        section="Section 16.1 / Table 5.6",
        target="MPA principal-branch refinement cross-check",
        family="method_comparison",
        primary=True,
        reference_kind="proxy",
    ),
    "table_5_7": AssignmentTarget(
        table="table_5_7",
        stage="Stage C",
        section="Section 16.1 / Table 5.7",
        target="MPA principal-branch tolerance cross-check",
        family="method_comparison",
        primary=True,
        reference_kind="proxy",
    ),
    "table_5_8": AssignmentTarget(
        table="table_5_8",
        stage="Stage A",
        section="Section 14.1 / Table 5.8",
        target="RMPA principal branch by mesh refinement",
        family="principal_branch",
        primary=True,
        reference_kind="proxy",
    ),
    "table_5_9": AssignmentTarget(
        table="table_5_9",
        stage="Stage A",
        section="Section 14.2 / Table 5.9",
        target="RMPA principal branch by tolerance",
        family="principal_branch",
        primary=True,
        reference_kind="proxy",
    ),
    "table_5_10": AssignmentTarget(
        table="table_5_10",
        stage="Stage B",
        section="Section 15.1 / Table 5.10",
        target="OA1 principal branch by mesh refinement",
        family="principal_branch",
        primary=True,
        reference_kind="proxy",
    ),
    "table_5_11": AssignmentTarget(
        table="table_5_11",
        stage="Stage B",
        section="Section 15.2 / Table 5.11",
        target="OA1 principal branch by tolerance",
        family="principal_branch",
        primary=False,
        reference_kind="proxy",
        caveat="Thesis runbook marks Table 5.11 as internally inconsistent; use Table 5.10 as the primary OA1 target.",
    ),
    "table_5_13": AssignmentTarget(
        table="table_5_13",
        stage="Stage C",
        section="Section 16.2 / Table 5.13",
        target="Direction comparison for RMPA and OA1",
        family="method_comparison",
        primary=True,
        reference_kind="exact",
    ),
    "table_5_14": AssignmentTarget(
        table="table_5_14",
        stage="Stage D",
        section="Section 17.1 / Table 5.14",
        target="Square-domain multibranch OA1/OA2 study",
        family="multibranch_square",
        primary=True,
        reference_kind="proxy",
    ),
    "figure_5_13": AssignmentTarget(
        table="figure_5_13",
        stage="Stage E",
        section="Section 17.2 / Figure 5.13",
        target="Square-hole multibranch OA2 study",
        family="multibranch_hole",
        primary=True,
        reference_kind="exact",
    ),
}


def get_assignment_target(table: str) -> AssignmentTarget:
    """Return assignment metadata for one thesis table/figure key."""
    return _TARGETS[str(table)]


def _delta_j(row: dict[str, object]) -> float | None:
    value = row.get("delta_J")
    return None if value is None else abs(float(value))


def _delta_i(row: dict[str, object]) -> float | None:
    thesis_i = row.get("thesis_I")
    measured = row.get("I")
    if thesis_i is None or measured is None:
        return None
    return abs(float(measured) - float(thesis_i))


def _table_5_13_principal_delta_j(row: dict[str, object]) -> float | None:
    if str(row.get("table", "")) != "table_5_13":
        return None
    measured = row.get("J")
    if measured is None:
        return None
    level = int(row.get("level", 0))
    p_value = float(row.get("p", 0.0))
    method = str(row.get("method", ""))
    if method == "rmpa":
        target = TABLE_5_8_RMPA_BY_LEVEL.get(level, {}).get(p_value, {}).get("J")
    elif method == "oa1":
        target = TABLE_5_10_OA1_BY_LEVEL.get(level, {}).get(p_value, {}).get("J")
    else:
        return None
    if target is None:
        return None
    return abs(float(measured) - float(target))


def _table_5_13_iteration_counts(row: dict[str, object]) -> tuple[int, int] | None:
    thesis_iterations = row.get("thesis_direction_iterations")
    if thesis_iterations is None:
        return None
    measured_iterations = row.get("outer_iterations")
    if measured_iterations is None:
        delta_iterations = row.get("delta_direction_iterations")
        if delta_iterations is None:
            return None
        measured_iterations = int(thesis_iterations) + int(delta_iterations)
    return int(thesis_iterations), int(measured_iterations)


def _table_5_13_iteration_ratio(row: dict[str, object]) -> float | None:
    counts = _table_5_13_iteration_counts(row)
    if counts is None:
        return None
    thesis_iterations, measured_iterations = counts
    if thesis_iterations <= 0:
        return None
    return float(measured_iterations / thesis_iterations)


def _table_5_13_principal_match(row: dict[str, object]) -> bool | None:
    principal_delta_j = _table_5_13_principal_delta_j(row)
    if principal_delta_j is None:
        return None
    return principal_delta_j <= TABLE_5_13_PRINCIPAL_BRANCH_TOL


def _table_5_13_low_impact(row: dict[str, object]) -> bool:
    if str(row.get("table", "")) != "table_5_13":
        return False
    if str(row.get("status", "")) != "completed":
        return False
    counts = _table_5_13_iteration_counts(row)
    ratio = _table_5_13_iteration_ratio(row)
    principal_match = _table_5_13_principal_match(row)
    if counts is None or ratio is None or principal_match is not True:
        return False
    thesis_iterations, measured_iterations = counts
    return (
        measured_iterations > thesis_iterations
        and ratio >= TABLE_5_13_LOW_IMPACT_RATIO
        and ratio < TABLE_5_13_EXTREME_RATIO
    )


def assignment_acceptance_pass(row: dict[str, object]) -> bool | None:
    """Evaluate the primary assignment acceptance rule for one row."""
    table = str(row["table"])
    method = str(row.get("method", ""))

    if table == "table_5_11":
        return None
    if table in {"table_5_6", "table_5_7", "table_5_8", "table_5_9", "table_5_10"}:
        delta_j = _delta_j(row)
        return None if delta_j is None else delta_j <= 2.0e-2
    if table == "table_5_13":
        counts = _table_5_13_iteration_counts(row)
        if counts is None:
            return None
        thesis_iterations, measured_iterations = counts
        if str(row.get("status", "")) != "completed":
            return False
        principal_match = _table_5_13_principal_match(row)
        if principal_match is None:
            return None
        if not principal_match:
            return False
        return measured_iterations < TABLE_5_13_EXTREME_RATIO * thesis_iterations
    if table == "table_5_14":
        delta_j = _delta_j(row)
        if delta_j is None:
            return None
        if method == "oa2":
            delta_i = _delta_i(row)
            return delta_j <= 5.0e-2 and (delta_i is None or delta_i <= 5.0e-2)
        return delta_j <= 2.0e-2
    if table == "figure_5_13":
        delta_j = _delta_j(row)
        delta_i = _delta_i(row)
        if delta_j is None:
            return None
        return delta_j <= 2.0e-1 and (delta_i is None or delta_i <= 5.0e-2)
    if table in {"table_5_2", "table_5_3", "table_5_2_drn_sanity"}:
        delta_j = _delta_j(row)
        return None if delta_j is None else delta_j <= 2.0e-2
    return None


def assignment_acceptance_rule(table: str, method: str) -> str:
    """Return the primary assignment acceptance rule text."""
    table = str(table)
    method = str(method)
    if table == "table_5_11":
        return "Secondary OA1 tolerance table; not used for primary acceptance because the runbook flags it as inconsistent."
    if table in {"table_5_6", "table_5_7", "table_5_8", "table_5_9", "table_5_10"}:
        return "|ΔJ| <= 2e-2"
    if table == "table_5_13":
        return (
            "Completed runs stay official passes when the principal-branch energy "
            "matches and the outer-iteration count stays below 10x the thesis "
            "direction-comparison count; moderate overruns are documented as low impact."
        )
    if table == "table_5_14" and method == "oa2":
        return "|ΔJ| <= 5e-2 and |ΔI| <= 5e-2"
    if table == "table_5_14":
        return "|ΔJ| <= 2e-2"
    if table == "figure_5_13":
        return "|ΔJ| <= 2e-1 and |ΔI| <= 5e-2"
    if table in {"table_5_2", "table_5_3", "table_5_2_drn_sanity"}:
        return "|ΔJ| <= 2e-2"
    return "-"


def assignment_verdict(row: dict[str, object]) -> str:
    """Return the dissemination verdict for one row."""
    target = get_assignment_target(str(row["table"]))
    if not bool(row.get("assignment_primary", target.primary)):
        return "secondary"
    accepted = assignment_acceptance_pass(row)
    if accepted is None:
        return "unknown"
    if accepted is False:
        return "fail"
    if _table_5_13_low_impact(row):
        return "low impact"
    return "pass"


def classify_gap(row: dict[str, object]) -> str | None:
    """Return a short root-cause label for unresolved rows."""
    status = str(row.get("status", ""))
    table = str(row["table"])
    method = str(row.get("method", ""))
    init_mode = str(row.get("init_mode", ""))
    p_value = float(row.get("p", 0.0))
    level = int(row.get("level", 0))
    accepted = assignment_acceptance_pass(row)
    verdict = assignment_verdict(row)

    if verdict == "pass":
        if status != "completed":
            return "Numerically matched, but solver did not report convergence"
        return None
    if table in {"table_5_2", "table_5_3"} and row.get("thesis_J") is None and row.get("thesis_iterations") is None:
        return "Published as unresolved (>500 iterations) in thesis"
    if (
        row.get("thesis_J") is None
        and row.get("thesis_I") is None
        and row.get("thesis_iterations") is None
        and row.get("thesis_direction_iterations") is None
    ):
        return "Secondary / unpublished thesis row"
    if status == "maxit" and method == "mpa":
        return "MPA convergence budget / step robustness"
    if status == "maxit" and table in {"table_5_2", "table_5_3", "table_5_2_drn_sanity"}:
        return "1D harness did not meet thesis stopping tolerance"
    if table == "table_5_14" and method == "oa2":
        return "OA2 square branch-selection mismatch"
    if table == "figure_5_13":
        return "OA2 square-hole branch-selection mismatch"
    if table in {"table_5_8", "table_5_9"} and level >= 7 and p_value <= (10.0 / 6.0):
        return "Low-p RMPA target / branch sensitivity"
    if table == "table_5_13" and verdict == "low impact":
        return "Low-impact direction-count discrepancy with matched principal-branch energy"
    if table == "table_5_13" and accepted is False:
        principal_match = _table_5_13_principal_match(row)
        ratio = _table_5_13_iteration_ratio(row)
        if principal_match is True and ratio is not None and ratio >= TABLE_5_13_EXTREME_RATIO:
            return "Direction-count exceeds 10x the thesis count with matched principal-branch energy"
        if principal_match is True:
            return "Direction comparison did not complete cleanly despite matched principal-branch energy"
        return "Direction comparison did not reproduce the thesis principal-branch energy"
    if table in {"table_5_6", "table_5_7"} and accepted is False:
        return "MPA energy mismatch against thesis table"
    if table in {"table_5_8", "table_5_9", "table_5_10", "table_5_11"} and accepted is False:
        return "Proxy reference or target mismatch"
    return "Unresolved thesis mismatch"


def attach_assignment_metadata(row: dict[str, object]) -> dict[str, object]:
    """Attach assignment-facing metadata to one summary row."""
    target = get_assignment_target(str(row["table"]))
    accepted = assignment_acceptance_pass(row)
    out = dict(row)
    out.update(
        {
            "assignment_stage": str(target.stage),
            "assignment_section": str(target.section),
            "assignment_target": str(target.target),
            "assignment_family": str(target.family),
            "assignment_primary": bool(target.primary),
            "assignment_reference_kind": str(target.reference_kind),
            "assignment_caveat": target.caveat,
            "assignment_acceptance_rule": assignment_acceptance_rule(
                str(row["table"]), str(row.get("method", ""))
            ),
            "assignment_acceptance_pass": accepted,
            "assignment_verdict": assignment_verdict(out),
            "assignment_gap_class": classify_gap(out),
        }
    )
    return out


def summarize_assignment_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize assignment-facing acceptance status across the whole suite."""
    by_stage: dict[str, dict[str, int]] = {}
    by_table: dict[str, dict[str, int]] = {}
    overall = {"pass": 0, "fail": 0, "unknown": 0}

    for row in rows:
        stage = str(row.get("assignment_stage", get_assignment_target(str(row["table"])).stage))
        table = str(row["table"])
        accepted = row.get("assignment_acceptance_pass")
        primary = bool(row.get("assignment_primary", get_assignment_target(table).primary))
        if not primary:
            bucket = "unknown"
        else:
            bucket = "unknown" if accepted is None else ("pass" if bool(accepted) else "fail")
        overall[bucket] += 1
        by_stage.setdefault(stage, {"pass": 0, "fail": 0, "unknown": 0, "total": 0})
        by_stage[stage][bucket] += 1
        by_stage[stage]["total"] += 1
        by_table.setdefault(table, {"pass": 0, "fail": 0, "unknown": 0, "total": 0})
        by_table[table][bucket] += 1
        by_table[table]["total"] += 1

    return {
        "problem_statement": THESIS_PROBLEM_STATEMENT,
        "functional_summary": THESIS_FUNCTIONAL_SUMMARY,
        "geometry_summary": THESIS_GEOMETRY_SUMMARY,
        "discretization_summary": THESIS_DISCRETIZATION_SUMMARY,
        "seed_summary": THESIS_SEED_SUMMARY,
        "legend": dict(REPLICATION_LEGEND),
        "stage_details": dict(ASSIGNMENT_STAGE_DETAILS),
        "method_to_tables": dict(METHOD_TO_TABLES),
        "overall": overall,
        "by_stage": by_stage,
        "by_table": by_table,
    }
