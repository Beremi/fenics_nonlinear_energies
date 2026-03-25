"""Shared rendering helpers for the pLaplace_u3 thesis docs packet."""

from __future__ import annotations

import math
from collections import Counter

from src.problems.plaplace_u3.thesis.tables import (
    TABLE_5_12_ITERATIONS,
    TABLE_5_12_RUNTIME_CONTEXT,
    TABLE_5_12_TIMES,
    TABLE_5_13_RUNTIME_CONTEXT,
    TABLE_5_13_TIMES,
)


def _p_key(value: object) -> float:
    return round(float(value) * 6.0) / 6.0


TABLE_RUNTIME_CONTEXTS = {
    "table_5_12": TABLE_5_12_RUNTIME_CONTEXT,
    "table_5_13": TABLE_5_13_RUNTIME_CONTEXT,
}

CONVERGENCE_DIAGNOSTIC_FAMILIES = (
    "table_5_6",
    "table_5_7",
    "table_5_8",
    "table_5_10",
    "table_5_2",
    "table_5_3",
    "table_5_2_drn_sanity",
)

TIMING_COMPLETE = "timing complete"
TIMING_UNAVAILABLE = "timing unavailable"
TIMING_NON_COMPLETED = "non-completed"
TIMING_BLOCKED = "blocked"


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= 0.0:
        return None
    return out


def _stack_label(row: dict[str, object]) -> str:
    explicit = str(row.get("backend_stack") or row.get("runtime_context") or "").strip()
    if explicit:
        return explicit
    if int(row.get("dimension", 2) or 2) == 1:
        return "JAX + SciPy helper solves"
    method = str(row.get("method", "")).lower()
    if method in {"mpa", "rmpa", "oa1", "oa2"}:
        return "JAX + SciPy + PyAMG helper solves"
    return "JAX + SciPy helper solves"


def runtime_context_text(row: dict[str, object], table: str | None = None) -> str:
    explicit = str(row.get("runtime_context") or "").strip()
    if explicit:
        return explicit
    if table is not None:
        default = TABLE_RUNTIME_CONTEXTS.get(str(table))
        if default is not None and row.get("launcher") is None and row.get("process_count") is None:
            return default
    launcher = str(row.get("launcher") or "serial python")
    process_count = row.get("process_count")
    proc_text = f"{int(process_count)} proc" if process_count is not None else "1 proc"
    return f"{proc_text}, {launcher}, {_stack_label(row)}"


def timing_value(value: object) -> float | None:
    return _coerce_float(value)


def format_timing(value: object, digits: int = 2) -> str:
    coerced = timing_value(value)
    if coerced is None:
        return "-"
    return f"{coerced:.{digits}f}"


def _timing_cap_text(row: dict[str, object]) -> str:
    configured = row.get("configured_maxit")
    if configured is not None:
        return f"maxit={int(configured)}"
    return "the current iteration cap"


def thesis_time_for_row(row: dict[str, object], table: str) -> float | None:
    table = str(table)
    p_value = _p_key(row.get("p", 0.0))
    if table == "table_5_12":
        explicit = _coerce_float(row.get("thesis_table_5_12_time_s"))
        if explicit is None:
            explicit = _coerce_float(row.get("thesis_time_s"))
        if explicit is not None:
            return explicit
        method = str(row.get("method", "")).lower()
        return _coerce_float(TABLE_5_12_TIMES.get(p_value, {}).get(method))
    if table == "table_5_13":
        explicit = _coerce_float(row.get("thesis_time_s"))
        if explicit is not None:
            return explicit
        method = str(row.get("method", "")).lower()
        direction = str(row.get("direction", "")).lower()
        return _coerce_float(TABLE_5_13_TIMES.get(method, {}).get(p_value, {}).get(direction))
    return None


def timing_metadata_for_row(row: dict[str, object], table: str) -> dict[str, object]:
    thesis_time = thesis_time_for_row(row, table)
    local_time = timing_value(row.get("solve_time_s"))
    raw_local_time = row.get("solve_time_s")
    runtime = runtime_context_text(row, table)
    status = str(row.get("status", ""))
    completed = status == "completed"
    outer_iterations = int(row.get("outer_iterations", 0) or 0)

    if thesis_time is None:
        return {
            "timing_status": TIMING_BLOCKED,
            "timing_reason": "thesis timing is unavailable for this mapping",
            "thesis_time_s": None,
            "repo_time_s": None,
            "raw_repo_time_s": local_time,
            "repo_iterations": int(row.get("outer_iterations", 0) or 0) if completed else None,
            "raw_repo_iterations": outer_iterations,
            "runtime_context": runtime,
        }

    if not completed:
        if status == "maxit":
            reason = f"solver stopped at {_timing_cap_text(row)}; local time is diagnostic only"
        elif status == "failed":
            reason = f"solver stopped with `{row.get('message')}`; local time is diagnostic only"
        else:
            reason = f"solver did not finish cleanly (`{status or 'unknown'}`); local time is diagnostic only"
        return {
            "timing_status": TIMING_NON_COMPLETED,
            "timing_reason": reason,
            "thesis_time_s": thesis_time,
            "repo_time_s": None,
            "raw_repo_time_s": local_time,
            "repo_iterations": None,
            "raw_repo_iterations": outer_iterations,
            "runtime_context": runtime,
        }

    if local_time is None:
        if raw_local_time is None:
            reason = "completed rerun selected but solve_time_s is missing; timing propagation bug"
        else:
            reason = (
                f"completed rerun selected but solve_time_s={raw_local_time} is nonpositive; "
                "stale zero-time artifact or timing propagation bug"
            )
        return {
            "timing_status": TIMING_UNAVAILABLE,
            "timing_reason": reason,
            "thesis_time_s": thesis_time,
            "repo_time_s": None,
            "raw_repo_time_s": None,
            "repo_iterations": outer_iterations,
            "raw_repo_iterations": outer_iterations,
            "runtime_context": runtime,
        }

    return {
        "timing_status": TIMING_COMPLETE,
        "timing_reason": "completed rerun with positive local timing",
        "thesis_time_s": thesis_time,
        "repo_time_s": local_time,
        "raw_repo_time_s": local_time,
        "repo_iterations": outer_iterations,
        "raw_repo_iterations": outer_iterations,
        "runtime_context": runtime,
    }


def timing_status_for_row(row: dict[str, object], table: str) -> str:
    return str(timing_metadata_for_row(row, table)["timing_status"])


def timing_note_for_row(row: dict[str, object], table: str) -> str:
    timing = timing_metadata_for_row(row, table)
    thesis_time = timing.get("thesis_time_s")
    if thesis_time is None:
        return "timing note unavailable"
    if timing["timing_status"] == TIMING_COMPLETE:
        return (
            f"timing note: thesis {thesis_time:.2f} s vs local {format_timing(timing.get('repo_time_s'))} s "
            f"on {timing.get('runtime_context')}"
        )
    return (
        f"timing note: thesis {thesis_time:.2f} s; local timing is not publishable because "
        f"{timing['timing_reason']} on {timing.get('runtime_context')}"
    )


def table_legend_lines(table: str) -> list[str]:
    table = str(table)
    if table in {"table_5_2", "table_5_3", "table_5_8", "table_5_9", "table_5_10", "table_5_11"}:
        return [
            "`thesis J`: published thesis energy",
            "`repo J`: reproduced canonical energy",
            "`thesis error` / `repo error`: thesis vs proxy-reference error",
            "`status`: current packet verdict under the assignment policy (`PASS`, `low impact`, `FAIL`, or `secondary`)",
        ]
    if table == "table_5_12":
        return [
            "`thesis it` / `repo it`: published vs reproduced iteration count",
            "`thesis t[s]` / `repo t[s]`: published vs reproduced wall-clock time",
            f"`runtime context`: `{TABLE_5_12_RUNTIME_CONTEXT}`",
            "`timing status`: `timing complete`, `timing unavailable`, `non-completed`, or `blocked`",
            "`timing reason`: concrete publication diagnosis for missing, zero, stale, or diagnostic-only local timing",
        ]
    if table == "table_5_13":
        return [
            "`thesis it` / `repo it`: published vs reproduced iteration count",
            "`thesis dir it`: published exact-direction count used for the low-impact policy",
            "`thesis t[s]` / `repo t[s]`: published vs reproduced wall-clock time",
            f"`runtime context`: `{TABLE_5_13_RUNTIME_CONTEXT}`",
            "`timing status`: `timing complete`, `timing unavailable`, `non-completed`, or `blocked`",
            "`timing reason`: concrete publication diagnosis for missing, zero, stale, or diagnostic-only local timing",
            "`status`: `PASS`, `low impact`, `FAIL`, or `secondary` under the current packet policy",
        ]
    if table in {"table_5_14", "figure_5_13"}:
        return [
            "`thesis J` / `repo J`: published vs reproduced energy",
            "`thesis I` / `repo I`: published vs reproduced quotient-side value",
            "`status`: current packet verdict under the assignment policy (`PASS`, `low impact`, `FAIL`, or `secondary`)",
        ]
    return ["column meanings follow the table header"]


def table_problem_spec_lines(table: str) -> list[str]:
    table = str(table)
    if table == "table_5_2" or table == "table_5_3":
        return [
            "1D harness for $-\\Delta_p u = u^3$ on $(0,\\pi)$.",
            "Domain / mesh: interval seed study from the thesis 1D helper setup.",
            "Method / direction: RMPA and OA1 with `d` / `d^{V_h}`.",
            "Comparison target: thesis energy, proxy error, and iteration count.",
        ]
    if table in {"table_5_8", "table_5_9"}:
        return [
            "Square principal branch for $J(u)$ on $[0,\\pi]^2$.",
            "Domain / mesh: structured $P_1$ right-triangle mesh with $h = \\pi / 2^L$.",
            "Method / direction: RMPA with the approximate direction `d^{V_h}`.",
            "Seed / tolerance: `sin(x)sin(y)` with the table-specific $\\varepsilon$ or level.",
        ]
    if table in {"table_5_10", "table_5_11"}:
        return [
            "Square principal branch for the scale-invariant quotient $I(u)$.",
            "Domain / mesh: structured $P_1$ right-triangle mesh with $h = \\pi / 2^L$.",
            "Method / direction: OA1 with the approximate direction `d^{V_h}`.",
            "Seed / tolerance: `sin(x)sin(y)` with the table-specific $\\varepsilon$ or level.",
        ]
    if table == "table_5_12":
        return [
            "Square cross-method timing table for MPA, RMPA, and OA1.",
            "Domain / mesh: $[0,\\pi]^2$ with $h = \\pi / 2^6$.",
            "Method / direction: the common principal-branch seed `sin(x)sin(y)`.",
            "Seed / tolerance: $\\varepsilon = 10^{-4}$ on the thesis comparison slice.",
            "Comparison target: published iteration counts and timings, with local serial-python timings recorded on 1 proc.",
        ]
    if table == "table_5_13":
        return [
            "Square direction-comparison table for $J(u)$ and the descent counts.",
            "Domain / mesh: $[0,\\pi]^2$ with $h = \\pi / 2^6$.",
            "Method / direction: RMPA exact `d` versus approximate `d^{V_h}`, plus OA1.",
            "Seed / tolerance: `sin(x)sin(y)` with $\\varepsilon = 10^{-4}$.",
            "Comparison target: published direction counts and timings, with principal-branch energy checked against Tables 5.8 / 5.10.",
        ]
    if table == "table_5_14":
        return [
            "Square multi-solution branch-selection table.",
            "Domain / mesh: $[0,\\pi]^2$ with the thesis square seeds.",
            "Method / direction: OA1 and OA2 with the published initialisations.",
            "Comparison target: branch selection via $J$ and $I$.",
        ]
    if table == "figure_5_13":
        return [
            "Square-with-hole multi-solution branch-selection study.",
            "Domain / mesh: nonconvex square-with-hole domain with the thesis hole seeds.",
            "Method / direction: OA2 with the published initialisations.",
            "Comparison target: branch selection via $J$ and $I$.",
        ]
    return ["problem specification follows the section title"]


def table_discrepancy_lines(table: str, rows: list[dict[str, object]]) -> list[str]:
    table = str(table)
    if table == "table_5_12":
        timing_rows = table_5_12_timing_rows(rows)
        if not timing_rows:
            return [
                "timing note: the thesis publishes Table 5.12 wall times, but the current packet does not yet expose the matching fixed-mesh slice.",
            ]
        notes: list[str] = []
        notes.append(
            "timing note: thesis Table 5.12 timings are surfaced alongside the current local timings from the matching serial-python rows; "
            f"the shared runtime context is `{TABLE_5_12_RUNTIME_CONTEXT}`."
        )
        if any(row["timing_status"] == TIMING_UNAVAILABLE for row in timing_rows):
            notes.append(
                "Rows marked `timing unavailable` are completed reruns whose selected artifact carries missing or zero local timing; that is treated as a packet-propagation bug, not as a solver failure."
            )
        if any(row["timing_status"] == TIMING_NON_COMPLETED for row in timing_rows):
            notes.append(
                "Rows marked `non-completed` keep any positive local wall time as diagnostic-only metadata; the public Stage C table suppresses publishable `repo t[s]` until the solver finishes cleanly."
            )
        if any(str(row.get("method")) == "mpa" and row["timing_status"] == TIMING_NON_COMPLETED for row in timing_rows):
            notes.append(
                "Current local MPA rows on the Table 5.12 slice still stop at `maxit=1000`; the packet now says so explicitly instead of folding them into a generic partial status."
            )
        return notes
    if table == "table_5_13":
        notes = [
            f"timing note: thesis Table 5.13 timings are shown beside fresh local serial-python reruns with `{TABLE_5_13_RUNTIME_CONTEXT}`."
        ]
        if any(row["timing_status"] == TIMING_UNAVAILABLE for row in rows):
            notes.append(
                "Rows marked `timing unavailable` are completed direction-comparison reruns whose selected artifact carries missing or zero local timing; those are packet-selection or propagation bugs, not convergence failures."
            )
        for row in rows:
            if str(row.get("table")) != "table_5_13" or str(row.get("method")) != "rmpa" or str(row.get("direction")) != "d":
                continue
            if str(row.get("assignment_verdict")) != "low impact":
                continue
            p_value = _p_key(row.get("p", 0.0))
            thesis_time = thesis_time_for_row(row, "table_5_13")
            timing_note = timing_note_for_row(row, "table_5_13")
            if abs(p_value - _p_key(17.0 / 6.0)) <= 1.0e-12:
                notes.append(
                    "`row`: `RMPA d, p = 17/6`; `impact`: `low impact`; "
                    f"`thesis`: `8 it, {format_timing(thesis_time)} s`; "
                    f"`repo`: `{int(row['outer_iterations'])} outer it, {int(row['direction_solves'])} direction solves, J = {float(row.get('J', 0.0)):.10f}`; "
                    "`meaning`: `principal-branch energy matches Table 5.8`; "
                    "`likely cause`: `late-stage tiny accepted halving steps in the exact-direction run`; "
                    f"`{timing_note}`; `status`: `documented as low impact`."
                )
            elif abs(p_value - _p_key(3.0)) <= 1.0e-12:
                notes.append(
                    "`row`: `RMPA d, p = 3`; `impact`: `low impact`; "
                    f"`thesis`: `19 it, {format_timing(thesis_time)} s`; "
                    f"`repo`: `{int(row['outer_iterations'])} outer it, {int(row['direction_solves'])} direction solves, J = {float(row.get('J', 0.0)):.10f}`; "
                    "`meaning`: `principal-branch energy matches Table 5.8`; "
                    "`likely cause`: `the exact auxiliary direction is not exploited as effectively before the final halving crawl`; "
                    f"`{timing_note}`; `status`: `documented as low impact`."
                )
        return notes or ["no primary mismatch beyond the rows shown above."]
    if table == "table_5_6":
        notes = [
            "Table 5.12 is surfaced separately in the Stage C timing summary, which reuses the fixed-mesh comparison rows from Tables 5.7, 5.9, and 5.11 for the thesis timing slice.",
        ]
        representative = next(
            (
                row
                for row in rows
                if str(row.get("status")) == "maxit"
                and row.get("peak_cycle_detected") is True
                and row.get("best_stop_measure") is not None
            ),
            None,
        )
        if representative is not None:
            notes.append(
                "The remaining level-sweep `maxit` rows are not plain line-search failures: fresh 1000-step probes keep accepting descent steps, "
                f"cycle across peak nodes, and only reach best stop {_fmt_stop(representative.get('best_stop_measure'))} before rebounding."
            )
        return notes
    if table == "table_5_7":
        maxit_rows = [row for row in rows if str(row.get("status")) == "maxit"]
        representative = next(
            (
                row
                for row in maxit_rows
                if row.get("peak_cycle_detected") is True and row.get("best_stop_measure") is not None
            ),
            None,
        )
        if representative is not None:
            return [
                f"Fresh 1000-step reruns still leave {len(maxit_rows)} epsilon-sweep rows at `maxit`; the representative hard case accepts every step, "
                f"flags `peak_cycle_detected=true`, and bottoms out near stop {_fmt_stop(representative.get('best_stop_measure'))} before the tail rebound.",
            ]
        return [
            "The fresh epsilon sweep still contains genuine MPA stop-rule overruns; these are kept explicit in the convergence diagnostics rather than folded into a generic status label.",
        ]
    if table == "table_5_11":
        return [
            "Thesis runbook marks Table 5.11 as internally inconsistent, so this packet keeps it as secondary context rather than a primary OA1 target.",
        ]
    if table == "table_5_8":
        return [
            "The `p = 1.5`, `level = 7` point is a secondary extension row; the primary square-branch rows still pass.",
        ]
    if table == "table_5_2" or table == "table_5_3":
        low_p_row = next(
            (
                row
                for row in rows
                if abs(float(row.get("p", 0.0)) - 1.5) <= 1.0e-12
            ),
            None,
        )
        if low_p_row is not None and str(low_p_row.get("status")) == "failed":
            direction_label = "exact `d`" if table == "table_5_2" else "cheap `d^{V_h}`"
            return [
                "The `p = 1.5` harness row remains a documented hard case rather than a primary match.",
                (
                    f"The refreshed {direction_label} rerun accepts progress for "
                    f"{int(low_p_row.get('accepted_step_count', 0) or 0)} steps, then stops in thesis Step 6 at outer "
                    f"{int(low_p_row.get('outer_iterations', 0) or 0)} with "
                    f"`{low_p_row.get('message')}` and best stop {_fmt_stop(low_p_row.get('best_stop_measure'))}."
                ),
            ]
        return [
            "The `p = 1.5` harness row is the published hard case and is kept as secondary context rather than a primary match.",
        ]
    return ["no material discrepancy in this table family."]


def table_5_12_timing_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    method_order = {"mpa": 0, "rmpa": 1, "oa1": 2}
    direct = [dict(row) for row in rows if str(row.get("table")) == "table_5_12"]
    if direct:
        return [
            {
                **row,
                "assignment_stage": "Stage C",
                "assignment_section": "Section 16.1 / Table 5.12",
                "assignment_target": "Cross-method timing comparison",
                "raw_repo_iterations": int(row.get("outer_iterations", row.get("thesis_iterations", 0)) or 0),
                "thesis_iterations": int(
                    row.get(
                        "thesis_table_5_12_iterations",
                        row.get(
                            "thesis_iterations",
                            TABLE_5_12_ITERATIONS.get(
                                _p_key(row.get("p", 0.0)),
                                {},
                            ).get(str(row.get("method", "")).lower(), 0),
                        ),
                    )
                    or 0
                ),
                **timing_metadata_for_row(row, "table_5_12"),
            }
            for row in sorted(direct, key=lambda item: (method_order.get(str(item.get("method", "")).lower(), 99), float(item.get("p", 0.0))))
        ]

    source_map = {
        "mpa": "table_5_7",
        "rmpa": "table_5_9",
        "oa1": "table_5_11",
    }
    timing_rows: list[dict[str, object]] = []
    for method, source_table in source_map.items():
        for row in rows:
            if str(row.get("table")) != source_table or str(row.get("method")) != method:
                continue
            if int(row.get("level", 0) or 0) != 6:
                continue
            if abs(float(row.get("epsilon", 0.0)) - 1.0e-4) > 1.0e-14:
                continue
            p_value = _p_key(row.get("p", 0.0))
            if p_value not in TABLE_5_12_ITERATIONS:
                continue
            timing_rows.append(
                {
                    **row,
                    "table": "table_5_12",
                    "assignment_stage": "Stage C",
                    "assignment_section": "Section 16.1 / Table 5.12",
                    "assignment_target": "Cross-method timing comparison",
                    "thesis_iterations": int(
                        row.get(
                            "thesis_table_5_12_iterations",
                            row.get(
                                "thesis_iterations",
                                TABLE_5_12_ITERATIONS.get(p_value, {}).get(method, 0),
                            ),
                        )
                        or 0
                    ),
                    **timing_metadata_for_row(row, "table_5_12"),
                    "source_table": source_table,
                }
            )
    return sorted(timing_rows, key=lambda item: (method_order.get(str(item["method"]).lower(), 99), float(item["p"])))


def table_5_13_timing_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    method_order = {"rmpa": 0, "oa1": 1}
    timing_rows: list[dict[str, object]] = []
    for row in rows:
        if str(row.get("table")) != "table_5_13":
            continue
        timing_rows.append(
            {
                **row,
                **timing_metadata_for_row(row, "table_5_13"),
            }
        )
    return sorted(timing_rows, key=lambda item: (method_order.get(str(item["method"]).lower(), 99), str(item["direction"]), float(item["p"])))


def _family_rows(rows: list[dict[str, object]], table: str) -> list[dict[str, object]]:
    return [dict(row) for row in rows if str(row.get("table")) == str(table)]


def _family_problem_rows(rows: list[dict[str, object]], table: str) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in _family_rows(rows, table)
        if str(row.get("status")) != "completed" or str(row.get("assignment_verdict")) == "fail"
    ]


def _problem_status_label(problem_rows: list[dict[str, object]]) -> str:
    if not problem_rows:
        return "resolved in current packet"
    fail_count = sum(1 for row in problem_rows if str(row.get("assignment_verdict")) == "fail")
    maxit_count = sum(1 for row in problem_rows if str(row.get("status")) == "maxit")
    other_count = len(problem_rows) - fail_count - maxit_count
    parts: list[str] = []
    if fail_count:
        parts.append(f"{fail_count} fail")
    if maxit_count:
        parts.append(f"{maxit_count} maxit")
    if other_count:
        parts.append(f"{other_count} non-completed")
    return "unresolved: " + ", ".join(parts)


def _best_stop_text(problem_rows: list[dict[str, object]]) -> str | None:
    values = [
        float(value)
        for value in (_coerce_float(row.get("best_stop_measure")) for row in problem_rows)
        if value is not None
    ]
    if not values:
        return None
    return f"{min(values):.2e}"


def _configured_cap_text(problem_rows: list[dict[str, object]]) -> str | None:
    caps = sorted({int(row.get("configured_maxit")) for row in problem_rows if row.get("configured_maxit") is not None})
    if not caps:
        return None
    return "/".join(str(value) for value in caps)


def _mpa_diagnostic(table: str, rows: list[dict[str, object]]) -> dict[str, str]:
    problem_rows = _family_problem_rows(rows, table)
    if not problem_rows:
        return {
            "family": table,
            "current_status": "resolved in current packet",
            "root_cause_category": "stale lower-budget artifact selection",
            "strongest_evidence": "Fresh packet rows now replace the earlier lower-budget MPA artifacts for this family.",
            "action_taken": "Added merge ranking that prefers budget-matched in-repo reruns and preserved explicit MPA diagnostics for future probes.",
        }

    cycle_count = sum(bool(row.get("peak_cycle_detected")) for row in problem_rows)
    accepted_full = sum(
        1
        for row in problem_rows
        if row.get("accepted_step_count") is not None
        and int(row.get("accepted_step_count", 0)) >= int(row.get("outer_iterations", 0))
    )
    best_stop = _best_stop_text(problem_rows)
    cap_text = _configured_cap_text(problem_rows) or "thesis iteration cap"
    evidence = (
        f"{len(problem_rows)} current rows still stop at {cap_text}; "
        f"{accepted_full} rows record accepted-step counts that reach the outer-iteration count"
    )
    if cycle_count:
        evidence += f", and {cycle_count} rows flag peak-node cycling"
    if best_stop is not None:
        evidence += f"; best recorded stop measure is only {best_stop}"
    evidence += "."
    return {
        "family": table,
        "current_status": _problem_status_label(problem_rows),
        "root_cause_category": "MPA accepted-step cycling / slow stop-measure decay",
        "strongest_evidence": evidence,
        "action_taken": "Added MPA convergence diagnostics and tested a local-path-repair acceptance variant; fresh probes still hit the thesis cap, so the family remains explicitly unresolved.",
    }


def _table_5_8_diagnostic(rows: list[dict[str, object]]) -> dict[str, str]:
    problem_rows = _family_problem_rows(rows, "table_5_8")
    if not problem_rows:
        return {
            "family": "table_5_8",
            "current_status": "resolved in current packet",
            "root_cause_category": "stale lower-budget artifact selection",
            "strongest_evidence": "The low-p square-branch rows now point at refreshed current-budget reruns rather than earlier 200-step artifacts.",
            "action_taken": "Refreshed the family and let merge provenance prefer the newer in-repo rows.",
        }

    best_stop = _best_stop_text(problem_rows)
    evidence = "The hard low-p `p = 1.5` square-branch rows still stop above the thesis tolerance after clean accepted-step progress."
    if best_stop is not None:
        evidence += f" The best recorded stop measure is {best_stop}."
    return {
        "family": "table_5_8",
        "current_status": _problem_status_label(problem_rows),
        "root_cause_category": "hard low-p RMPA stop-rule / budget ceiling",
        "strongest_evidence": evidence,
        "action_taken": "Refreshed the family under the current thesis budget and kept the remaining low-p behavior documented instead of treating it as a packet-selection bug.",
    }


def _table_5_10_diagnostic(rows: list[dict[str, object]]) -> dict[str, str]:
    problem_rows = _family_problem_rows(rows, "table_5_10")
    if not problem_rows:
        return {
            "family": "table_5_10",
            "current_status": "resolved in current packet",
            "root_cause_category": "stale lower-budget artifact selection",
            "strongest_evidence": "The previously problematic OA1 low-p row now resolves to a completed refreshed artifact instead of an older 200-step `maxit` row.",
            "action_taken": "Rebuilt the canonical packet with provenance guards so the fresher completed OA1 row wins.",
        }

    best_stop = _best_stop_text(problem_rows)
    evidence = "The remaining OA1 low-p row progresses with accepted steps but does not cross the thesis tolerance before the cap."
    if best_stop is not None:
        evidence += f" The best recorded stop measure is {best_stop}."
    return {
        "family": "table_5_10",
        "current_status": _problem_status_label(problem_rows),
        "root_cause_category": "hard low-p OA1 stop-rule / budget ceiling",
        "strongest_evidence": evidence,
        "action_taken": "Refreshed the family and kept the residual low-p row explicit rather than misreporting it as a clean convergence match.",
    }


def _table_5_2_diagnostic(rows: list[dict[str, object]]) -> dict[str, str]:
    problem_rows = _family_problem_rows(rows, "table_5_2")
    if not problem_rows:
        return {
            "family": "table_5_2",
            "current_status": "resolved in current packet",
            "root_cause_category": "exact-direction golden-step bug corrected",
            "strongest_evidence": "The exact-direction low-p row no longer fails after the boundary-aware golden-step fix.",
            "action_taken": "Tracked boundary minima in golden-section search and added a tiny-step fallback in thesis RMPA Step 6.",
        }

    if any(str(row.get("status")) == "failed" and str(row.get("step_search")) == "golden" for row in problem_rows):
        root = "golden-step boundary minimizer / near-zero step-search failure"
        evidence = (
            "The exact-direction low-p row still fails in the golden Step 6 branch even though the direction remains descending; "
            "the failure sits at the near-zero step boundary rather than in the nonlinear auxiliary solve."
        )
        action = "Tracked endpoint candidates in golden-section search and added a tiny-step fallback in thesis RMPA Step 6; the family remains explicit until a fresh full-reference rerun confirms the fix."
    else:
        root = "exact-direction low-p stop-rule / budget ceiling"
        evidence = "The exact-direction low-p row still exhausts the thesis iteration budget without satisfying (5.6)."
        action = "Refreshed the row and documented the exact-direction low-p ceiling explicitly."
    return {
        "family": "table_5_2",
        "current_status": _problem_status_label(problem_rows),
        "root_cause_category": root,
        "strongest_evidence": evidence,
        "action_taken": action,
    }


def _table_5_3_diagnostic(rows: list[dict[str, object]]) -> dict[str, str]:
    problem_rows = _family_problem_rows(rows, "table_5_3")
    if not problem_rows:
        return {
            "family": "table_5_3",
            "current_status": "resolved in current packet",
            "root_cause_category": "stale lower-budget artifact selection",
            "strongest_evidence": "The cheap-direction 1D low-p row now points at a completed current-budget rerun.",
            "action_taken": "Refreshed the family and kept merge provenance aligned with the current thesis defaults.",
        }

    best_stop = _best_stop_text(problem_rows)
    evidence = "The 1D cheap-direction low-p row remains a clean accepted-step crawl above the thesis stop rule."
    if best_stop is not None:
        evidence += f" Best recorded stop measure: {best_stop}."
    return {
        "family": "table_5_3",
        "current_status": _problem_status_label(problem_rows),
        "root_cause_category": "cheap-direction low-p stop-rule / budget ceiling",
        "strongest_evidence": evidence,
        "action_taken": "Refreshed the row and kept it documented as a hard low-p case rather than a line-search defect.",
    }


def _table_5_2_drn_sanity_diagnostic(rows: list[dict[str, object]]) -> dict[str, str]:
    problem_rows = _family_problem_rows(rows, "table_5_2_drn_sanity")
    if not problem_rows:
        return {
            "family": "table_5_2_drn_sanity",
            "current_status": "resolved in current packet",
            "root_cause_category": "no current convergence issue",
            "strongest_evidence": "The sanity-check `d_rn` row completes in the current packet.",
            "action_taken": "Retained the row only as a secondary sanity check.",
        }

    return {
        "family": "table_5_2_drn_sanity",
        "current_status": _problem_status_label(problem_rows),
        "root_cause_category": "weak `d_rn` direction under the thesis stop rule",
        "strongest_evidence": "The sanity-check row keeps making tiny descent progress but does not satisfy the thesis stopping rule before the cap.",
        "action_taken": "Kept the row as documented secondary context instead of treating it as a primary solver defect.",
    }


def _legacy_convergence_diagnostic_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    return [
        _mpa_diagnostic("table_5_6", rows),
        _mpa_diagnostic("table_5_7", rows),
        _table_5_8_diagnostic(rows),
        _table_5_10_diagnostic(rows),
        _table_5_2_diagnostic(rows),
        _table_5_3_diagnostic(rows),
        _table_5_2_drn_sanity_diagnostic(rows),
    ]


def _specific_case(
    rows: list[dict[str, object]],
    *,
    table: str,
    level: int,
    p: float,
    epsilon: float,
) -> dict[str, object] | None:
    for row in rows:
        if str(row.get("table")) != table:
            continue
        if int(row.get("level", -1) or -1) != int(level):
            continue
        if abs(float(row.get("p", 0.0)) - float(p)) > 1.0e-12:
            continue
        if abs(float(row.get("epsilon", 0.0)) - float(epsilon)) > 1.0e-14:
            continue
        return row
    return None


def _fmt_stop(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2e}"


def _row_budget_label(row: dict[str, object]) -> str:
    configured = row.get("configured_maxit")
    if configured is None:
        return "legacy/unknown budget"
    return f"maxit={int(configured)}"


def convergence_diagnostic_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    table_56_rows = [row for row in rows if str(row.get("table")) == "table_5_6"]
    historical_fail_cases = [
        _specific_case(rows, table="table_5_6", level=6, p=17.0 / 6.0, epsilon=1.0e-4),
        _specific_case(rows, table="table_5_6", level=7, p=10.0 / 6.0, epsilon=1.0e-4),
        _specific_case(rows, table="table_5_6", level=7, p=17.0 / 6.0, epsilon=1.0e-4),
    ]
    repaired_56 = [
        row
        for row in historical_fail_cases
        if row is not None and row.get("delta_J") is not None and abs(float(row["delta_J"])) <= 0.03
    ]
    if repaired_56:
        maxit_cases = [row for row in repaired_56 if str(row.get("status")) == "maxit"]
        stop_range = ", ".join(_fmt_stop(row.get("best_stop_measure")) for row in maxit_cases[:3]) or "-"
        out.append(
            {
                "family": "table_5_6",
                "current_status": "historical FAIL rows repaired; stop rule unresolved",
                "root_cause_category": "MPA accepted-step peak cycling / slow stop decay",
                "strongest_evidence": (
                    f"{len(repaired_56)}/3 historical FAIL rows are back inside the thesis J tolerance, "
                    f"but the refreshed rows still end at maxit with peak-cycle diagnostics and best stops {stop_range}."
                ),
                "action_taken": "Added MPA convergence diagnostics and promoted fresh 1000-step reruns for the repaired cases.",
            }
        )
    elif table_56_rows:
        fail_count = sum(1 for row in table_56_rows if str(row.get("assignment_verdict")) == "fail")
        out.append(
            {
                "family": "table_5_6",
                "current_status": "unresolved",
                "root_cause_category": "MPA accepted-step peak cycling / slow stop decay",
                "strongest_evidence": f"{fail_count} current FAIL rows remain in the level sweep.",
                "action_taken": "Kept the family under explicit convergence diagnosis instead of treating maxit as the cause.",
            }
        )

    table_57_rows = [row for row in rows if str(row.get("table")) == "table_5_7"]
    if table_57_rows:
        maxit_rows = [row for row in table_57_rows if str(row.get("status")) == "maxit"]
        cycle_rows = [row for row in maxit_rows if row.get("peak_cycle_detected") is True]
        representative = next((row for row in cycle_rows if row.get("best_stop_measure") is not None), None)
        evidence = f"{len(maxit_rows)} refreshed rows still stop at maxit on the epsilon sweep."
        if representative is not None:
            evidence = (
                f"{len(maxit_rows)} rows still stop at maxit; representative best stop "
                f"{_fmt_stop(representative.get('best_stop_measure'))} with `peak_cycle_detected=true`."
            )
        out.append(
            {
                "family": "table_5_7",
                "current_status": "energy stable, stop rule unresolved",
                "root_cause_category": "MPA epsilon-sweep tail stall",
                "strongest_evidence": evidence,
                "action_taken": "Public docs now describe the stall explicitly instead of presenting raw maxit as the explanation.",
            }
        )

    table_58_rows = [
        row
        for row in rows
        if str(row.get("table")) == "table_5_8"
        and abs(float(row.get("p", 0.0)) - 1.5) <= 1.0e-12
        and int(row.get("level", 0) or 0) in {6, 7}
    ]
    if table_58_rows:
        completed = [row for row in table_58_rows if str(row.get("status")) == "completed"]
        if len(completed) == len(table_58_rows):
            it_text = ", ".join(
                f"L{int(row['level'])}: {int(row.get('outer_iterations', 0))} it"
                for row in sorted(completed, key=lambda row: int(row["level"]))
            )
            out.append(
                {
                    "family": "table_5_8",
                    "current_status": "resolved",
                    "root_cause_category": "stale 200-step artifact, not a live RMPA bug",
                    "strongest_evidence": f"The low-p rows now complete under the current thesis budget ({it_text}).",
                    "action_taken": "Promoted fresh 500-step serial reruns so stale maxit rows cannot override them.",
                }
            )
        else:
            counts = Counter(str(row.get("status")) for row in table_58_rows)
            out.append(
                {
                    "family": "table_5_8",
                    "current_status": "partially refreshed",
                    "root_cause_category": "stale/budget-sensitive low-p row",
                    "strongest_evidence": ", ".join(f"{status}: {count}" for status, count in sorted(counts.items())),
                    "action_taken": "Kept the low-p rows under refresh until the canonical packet reflects the completed reruns.",
                }
            )

    table_510_rows = [
        row
        for row in rows
        if str(row.get("table")) == "table_5_10"
        and abs(float(row.get("p", 0.0)) - 1.5) <= 1.0e-12
        and int(row.get("level", 0) or 0) == 7
    ]
    if table_510_rows:
        row = table_510_rows[0]
        if str(row.get("status")) == "completed":
            out.append(
                {
                    "family": "table_5_10",
                    "current_status": "resolved",
                    "root_cause_category": "stale 200-step artifact, not a live OA1 bug",
                    "strongest_evidence": (
                        f"The published low-p row now completes in {int(row.get('outer_iterations', 0))} iterations under "
                        f"{_row_budget_label(row)}."
                    ),
                    "action_taken": "Promoted a fresh 500-step rerun and preserved the OA1 algorithm unchanged.",
                }
            )
        else:
            out.append(
                {
                    "family": "table_5_10",
                    "current_status": "unresolved",
                    "root_cause_category": "low-p OA1 tail still not refreshed cleanly",
                    "strongest_evidence": str(row.get("message") or row.get("status") or "-"),
                    "action_taken": "Kept the case isolated for direct reruns instead of generalizing from the stale packet row.",
                }
            )

    table_52_rows = [
        row
        for row in rows
        if str(row.get("table")) == "table_5_2" and abs(float(row.get("p", 0.0)) - 1.5) <= 1.0e-12
    ]
    if table_52_rows:
        row = table_52_rows[0]
        out.append(
            {
                "family": "table_5_2",
                "current_status": "unresolved",
                "root_cause_category": "exact-direction Step 6 halving failure",
                "strongest_evidence": (
                    f"The refreshed p=1.5 row accepts {int(row.get('accepted_step_count', 0) or 0)} steps, then fails at "
                    f"outer {int(row.get('outer_iterations', 0))} with `{row.get('message')}` and best stop "
                    f"{_fmt_stop(row.get('best_stop_measure'))}."
                ),
                "action_taken": "Reran with the thesis golden-section RMPA path at 500 iterations and documented the Step 6 failure explicitly.",
            }
        )

    table_53_rows = [
        row
        for row in rows
        if str(row.get("table")) == "table_5_3" and abs(float(row.get("p", 0.0)) - 1.5) <= 1.0e-12
    ]
    if table_53_rows:
        row = table_53_rows[0]
        out.append(
            {
                "family": "table_5_3",
                "current_status": "unresolved",
                "root_cause_category": "cheap-direction Step 6 halving failure",
                "strongest_evidence": (
                    f"The refreshed p=1.5 row accepts {int(row.get('accepted_step_count', 0) or 0)} steps, then fails at "
                    f"outer {int(row.get('outer_iterations', 0))} with `{row.get('message')}` and best stop "
                    f"{_fmt_stop(row.get('best_stop_measure'))}."
                ),
                "action_taken": "Reran with the current 500-step thesis budget and recorded the specific Step 6 failure mode.",
            }
        )

    drn_rows = [row for row in rows if str(row.get("table")) == "table_5_2_drn_sanity"]
    if drn_rows:
        row = drn_rows[0]
        if str(row.get("status")) == "completed":
            out.append(
                {
                    "family": "table_5_2_drn_sanity",
                    "current_status": "resolved",
                    "root_cause_category": "stale 200-step artifact, not a d_rn implementation bug",
                    "strongest_evidence": (
                        f"The fresh sanity row completes in {int(row.get('outer_iterations', 0))} iterations with "
                        f"`{row.get('message')}` under {_row_budget_label(row)}."
                    ),
                    "action_taken": "Promoted the refreshed 1D sanity rerun and removed the misleading stale maxit result.",
                }
            )
        else:
            out.append(
                {
                    "family": "table_5_2_drn_sanity",
                    "current_status": "secondary unresolved",
                    "root_cause_category": "weak d_rn behavior under the thesis stop rule",
                    "strongest_evidence": str(row.get("message") or row.get("status") or "-"),
                    "action_taken": "Left the family as secondary context and documented the actual stopping behavior.",
                }
            )

    return out


def convergence_diagnostics_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    return convergence_diagnostic_rows(rows)
