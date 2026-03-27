#!/usr/bin/env python3
"""Generate the `docs/problems/pLaplace_up_arctan.md` page."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "raw_results" / "plaplace_up_arctan_full" / "summary.json"
DEFAULT_PETSC_SUMMARY = REPO_ROOT / "artifacts" / "raw_results" / "plaplace_up_arctan_petsc" / "summary.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "problems" / "pLaplace_up_arctan.md"
DEFAULT_ASSET_DIR = REPO_ROOT / "docs" / "assets" / "plaplace_up_arctan"
TRACK_RAW = "raw"
TRACK_CERTIFIED = "certified"


def _load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_summary(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    return _load_summary(path)


def _select_row(summary: dict[str, object], *, study: str, p: float, method: str, level: int, epsilon: float) -> dict[str, object]:
    matches = [
        row
        for row in summary["rows"]
        if str(row["study"]) == str(study)
        and float(row["p"]) == float(p)
        and str(row["method"]) == str(method)
        and int(row["level"]) == int(level)
        and abs(float(row["epsilon"]) - float(epsilon)) <= 1.0e-14
    ]
    if not matches:
        raise ValueError(f"Missing row for {study=}, {p=}, {method=}, {level=}, {epsilon=}")
    return dict(matches[0])


def _fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _comparison_rows(summary: dict[str, object], *, family: str | None = None, p: float | None = None) -> list[dict[str, object]]:
    rows = [dict(row) for row in summary.get("method_comparison", [])]
    if family is not None:
        rows = [row for row in rows if str(row.get("family")) == str(family)]
    if p is not None:
        rows = [row for row in rows if float(row.get("p")) == float(p)]
    return rows


def _comparison_sort_key(row: dict[str, object]) -> tuple[float, int, int]:
    geometry_order = {
        "0 -> +C seed": 0,
        "-C1 seed -> +C2 seed": 1,
        "ray from 0": 0,
        "line from -C1 seed": 1,
    }
    seed_order = {
        "sine": 0,
        "bubble": 1,
        "tilted": 2,
        "eigenfunction": 3,
    }
    return (
        float(row.get("p")),
        geometry_order.get(str(row.get("geometry_label")), 99),
        seed_order.get(str(row.get("seed_name")), 99),
    )


def _comparison_table_rows(summary: dict[str, object], *, family: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in sorted(_comparison_rows(summary, family=family), key=_comparison_sort_key):
        rows.append(
            [
                str(int(row["p"])),
                str(row.get("geometry_label", "-")),
                str(row.get("seed_name", "-")),
                str(row.get("raw_status", "-")),
                _fmt(row.get("raw_residual_norm"), 6),
                str(int(row.get("raw_outer_iterations", 0))),
                str(row.get("certified_status", "-")),
                _fmt(row.get("certified_residual_norm"), 6),
                str(int(row.get("certified_newton_iters", 0))),
                str(int(row.get("total_nonlinear_iters", 0))),
                _fmt(row.get("solve_time_s"), 3),
            ]
        )
    return rows


def _petsc_rows_for(
    summary: dict[str, object] | None,
    *,
    study: str | None = None,
    p: float | None = None,
) -> list[dict[str, object]]:
    if not summary:
        return []
    rows = [dict(row) for row in summary.get("rows", [])]
    if study is not None:
        rows = [row for row in rows if str(row.get("study")) == str(study)]
    if p is not None:
        rows = [row for row in rows if float(row.get("p")) == float(p)]
    return rows


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _track_value(row: dict[str, object], key: str, track: str | None = None, default: object | None = None) -> object | None:
    prefixes: list[str | None] = []
    if track in {TRACK_RAW, TRACK_CERTIFIED}:
        prefixes.append(track)
    prefixes.extend([TRACK_CERTIFIED, TRACK_RAW, None])
    seen: set[str | None] = set()
    for prefix in prefixes:
        if prefix in seen:
            continue
        seen.add(prefix)
        candidate = key if prefix is None else f"{prefix}_{key}"
        if candidate in row and row[candidate] is not None:
            return row[candidate]
    return row.get(key, default)


def _certified_mpa_iters(row: dict[str, object]) -> int:
    source = str(row.get("certified_handoff_source") or "")
    if source.startswith("mpa:"):
        return int(row.get("raw_outer_iterations", 0))
    return 0


def _certified_newton_iters(row: dict[str, object]) -> int:
    return int(row.get("certified_newton_iters", row.get("certified_outer_iterations", row.get("outer_iterations", 0))))


def _certified_total_nonlinear_iters(row: dict[str, object]) -> int:
    return int(_certified_mpa_iters(row) + _certified_newton_iters(row))


def _handoff_label(source: object | None) -> str:
    value = str(source or "-")
    if value.startswith("mpa:"):
        return "MPA handoff"
    if value == "direct_init":
        return "Continuation direct"
    return value.replace("_", " ")


def _raw_certified_rows(summary: dict[str, object], *, p: float, epsilon: float) -> list[list[str]]:
    rows: list[list[str]] = []
    for method in ("mpa", "rmpa"):
        for level in (4, 5, 6):
            row = _select_row(summary, study="mesh_refinement", p=p, method=method, level=level, epsilon=epsilon)
            rows.append(
                [
                    method.upper(),
                    str(level),
                    str(_track_value(row, "status", TRACK_RAW, default=row["status"])),
                    _fmt(_track_value(row, "residual_norm", TRACK_RAW, default=row["residual_norm"])),
                    str(_track_value(row, "status", TRACK_CERTIFIED, default=row["status"])),
                    _fmt(_track_value(row, "residual_norm", TRACK_CERTIFIED, default=row["residual_norm"])),
                    str(_track_value(row, "start_seed_name", TRACK_RAW, default=row.get("start_seed_name") or "-")),
                    _handoff_label(row.get("certified_handoff_source", row.get("certified_reported_iterate_source"))),
                ]
            )
    return rows


def _main_mpa_rows(
    summary: dict[str, object],
    *,
    p: float,
    epsilon: float,
    include_lambda: bool = False,
    include_handoff_residual: bool = True,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for level in (4, 5, 6):
        row = _select_row(summary, study="mesh_refinement", p=p, method="mpa", level=level, epsilon=epsilon)
        certified_iters = _certified_newton_iters(row)
        entry = [
            str(level),
            _handoff_label(row.get("certified_handoff_source", row.get("certified_reported_iterate_source"))),
            _fmt(_track_value(row, "residual_norm", TRACK_CERTIFIED, default=row["residual_norm"])),
            str(_certified_mpa_iters(row)),
            str(certified_iters),
            str(_certified_total_nonlinear_iters(row)),
            _fmt(_track_value(row, "J", TRACK_CERTIFIED, default=row["J"])),
            str(_track_value(row, "status", TRACK_CERTIFIED, default=row["status"])),
        ]
        if include_handoff_residual:
            entry.insert(2, _fmt(_track_value(row, "residual_norm", TRACK_RAW, default=row["residual_norm"])))
        if include_lambda:
            entry.insert(1, _fmt(row["lambda1"], 6))
        rows.append(entry)
    return rows


def _tolerance_rows(summary: dict[str, object], *, p: float) -> list[list[str]]:
    rows: list[list[str]] = []
    for method in ("mpa", "rmpa"):
        for epsilon in (1.0e-4, 1.0e-5, 1.0e-6):
            row = _select_row(summary, study="tolerance_sweep", p=p, method=method, level=6, epsilon=epsilon)
            rows.append(
                [
                    str(int(p)),
                    method.upper(),
                    f"{epsilon:.0e}",
                    str(_track_value(row, "status", TRACK_RAW, default=row["status"])),
                    _fmt(_track_value(row, "residual_norm", TRACK_RAW, default=row["residual_norm"])),
                    str(_track_value(row, "status", TRACK_CERTIFIED, default=row["status"])),
                    _fmt(_track_value(row, "residual_norm", TRACK_CERTIFIED, default=row["residual_norm"])),
                ]
            )
    return rows


def _rmpa_rows(summary: dict[str, object], *, p: float, epsilon: float) -> list[list[str]]:
    rows: list[list[str]] = []
    for level in (4, 5, 6):
        row = _select_row(summary, study="mesh_refinement", p=p, method="rmpa", level=level, epsilon=epsilon)
        rows.append(
            [
                str(level),
                str(_track_value(row, "status", TRACK_RAW, default=row["status"])),
                _fmt(_track_value(row, "residual_norm", TRACK_RAW, default=row["residual_norm"])),
                str(row.get("ray_best_kind") or "-"),
                "yes" if bool(row.get("ray_stable_interior_extremum", False)) else "no",
                str(row.get("certification_message") or "-"),
            ]
        )
    return rows


def _continuation_rows(payload: dict[str, object]) -> list[list[str]]:
    rows: list[list[str]] = []
    for step in list(payload.get("continuation_steps", [])):
        rows.append(
            [
                f"{float(step['from_p']):.1f}",
                f"{float(step['to_p']):.1f}",
                str(step["path"]),
                _fmt(step["residual_norm"], 6),
                str(step["status"]),
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--petsc-summary", type=Path, default=DEFAULT_PETSC_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    args = parser.parse_args()

    summary = _load_summary(args.summary)
    petsc_summary = _load_optional_summary(args.petsc_summary)
    rel_asset = lambda name: str(Path(os.path.relpath(args.asset_dir / name, start=args.out.parent))).replace("\\", "/")
    summary_dir = args.summary.parent

    def _reference_payload(name: str) -> dict[str, object]:
        path = summary_dir / "references" / name / "output.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    lambda_payloads = []
    for path in sorted(Path(summary["lambda_cache_dir"]).glob("lambda_p3_l*.json")):
        lambda_payloads.append(json.loads(path.read_text(encoding="utf-8")))

    p2_reference = _reference_payload("p2_newton_l7")
    p3_continuation = _reference_payload("p3_certified_l6")
    p3_reference = _reference_payload("p3_certified_l7")
    comparison_level = int(summary.get("comparison_level", 6))
    shifted_rmpa_best = []
    for p_value in (2.0, 3.0):
        candidates = [
            row
            for row in _comparison_rows(summary, family="RMPA", p=p_value)
            if str(row.get("method")) == "rmpa_shifted" and str(row.get("certified_status")) == "completed"
        ]
        if candidates:
            shifted_rmpa_best.append(
                min(
                    candidates,
                    key=lambda row: (
                        0 if str(row.get("raw_status")) == "completed" else 1,
                        float(row.get("certified_residual_norm", float("inf"))),
                    ),
                )
            )

    def _petsc_mesh_rows(p: float) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in sorted(_petsc_rows_for(petsc_summary, study="mesh_ladder", p=p), key=lambda item: int(item["level"])):
            rows.append(
                [
                    str(int(row["level"])),
                    str(int(row.get("free_dofs", 0))),
                    str(row.get("status", "-")),
                    _fmt(row.get("residual_norm"), 6),
                    _fmt(row.get("setup_time_s"), 3),
                    _fmt(row.get("solve_time_s"), 3),
                    _fmt(row.get("total_time_s"), 3),
                    str(int(row.get("outer_iterations", 0))),
                    str(int(row.get("outer_iterations", 0))),
                    str(int(row.get("linear_iterations_total", 0))),
                ]
            )
        return rows

    def _petsc_scaling_rows() -> list[list[str]]:
        rows: list[list[str]] = []
        for row in sorted(_petsc_rows_for(petsc_summary, study="strong_scaling", p=2.0), key=lambda item: int(item["nprocs"])):
            rows.append(
                [
                    str(int(row["nprocs"])),
                    str(row.get("status", "-")),
                    _fmt(row.get("total_time_s"), 3),
                    str(int(row.get("outer_iterations", 0))),
                    str(int(row.get("outer_iterations", 0))),
                    str(int(row.get("linear_iterations_total", 0))),
                    _fmt(row.get("speedup_total"), 3),
                    _fmt(row.get("efficiency_total"), 3),
                    _fmt(row.get("residual_norm"), 6),
                ]
            )
        return rows

    lines = [
        "# p-Laplace Arctan Resonance on the Unit Square",
        "",
        "Source note used for this implementation: `pLaplace_up_arctan.md` in the repo root.",
        "",
        "## Mathematical Specification",
        "",
        "We study the resonant Dirichlet problem",
        "",
        "$$",
        "-\\Delta_p u = \\lambda_1 |u|^{p-2}u + \\arctan(u+1) \\quad \\text{in } \\Omega=(0,1)^2, \\qquad u=0 \\text{ on } \\partial\\Omega.",
        "$$",
        "",
        "with the common nonlinear data",
        "",
        "$$",
        "g(u)=\\arctan(u+1), \\qquad g'(u)=\\frac{1}{1+(u+1)^2},",
        "$$",
        "",
        "$$",
        "G(t) = \\int_0^t g(s)\\,ds = (t+1)\\arctan(t+1) - \\tfrac12\\log(1+(t+1)^2) - \\left(\\tfrac\\pi4 - \\tfrac12\\log 2\\right),",
        "$$",
        "",
        "so that `G(0)=0`. The energy used by the maintained solver is",
        "",
        "$$",
        "J_p(u) = \\frac{1}{p}\\int_\\Omega |\\nabla u|^p\\,dx - \\frac{\\lambda_1}{p}\\int_\\Omega |u|^p\\,dx - \\int_\\Omega G(u)\\,dx.",
        "$$",
        "",
        "The common weak form is",
        "",
        "$$",
        "\\int_\\Omega |\\nabla u|^{p-2}\\nabla u\\cdot\\nabla v\\,dx - \\lambda_1\\int_\\Omega |u|^{p-2}uv\\,dx - \\int_\\Omega \\arctan(u+1)v\\,dx = 0",
        "$$",
        "",
        "for all admissible test functions `v`.",
        "",
        "### p = 2 Validation Problem",
        "",
        "$$",
        "-\\Delta u = 2\\pi^2 u + \\arctan(u+1) \\quad \\text{in } (0,1)^2, \\qquad u=0 \\text{ on } \\partial(0,1)^2.",
        "$$",
        "",
        "Here the first eigenpair is explicit:",
        "",
        "$$",
        "\\lambda_1 = 2\\pi^2, \\qquad \\varphi_1(x,y)=\\sin(\\pi x)\\sin(\\pi y).",
        "$$",
        "",
        "### p = 3 Main Problem",
        "",
        "$$",
        "-\\operatorname{div}\\bigl(|\\nabla u|\\nabla u\\bigr) = \\lambda_1 |u|u + \\arctan(u+1) \\quad \\text{in } (0,1)^2, \\qquad u=0 \\text{ on } \\partial(0,1)^2.",
        "$$",
        "",
        "The first eigenvalue is not explicit on the unit square when `p=3`, so the workflow first computes a discrete positive eigenpair `(\\lambda_{1,h},\\varphi_{1,h})` from",
        "",
        "$$",
        "-\\Delta_3 \\varphi_1 = \\lambda_1 |\\varphi_1|\\varphi_1, \\qquad \\varphi_1|_{\\partial\\Omega}=0, \\qquad \\int_\\Omega |\\varphi_1|^3\\,dx = 1.",
        "$$",
        "",
        "## Solvability And Proof Notes",
        "",
        "For the source-note specialization of Theorem 6, the key helper quantity is",
        "",
        "$$",
        "F(x) = \\frac{p}{x}\\int_0^x g(s)\\,ds - g(x), \\qquad g(x)=\\arctan(x+1).",
        "$$",
        "",
        "The source note gives the asymptotic limits `lim_{x->+∞} F(x) = (p-1)π/2` and `lim_{x->-∞} F(x) = -(p-1)π/2`.",
        "Because `g` is bounded we also have `g(u)/|u|^{p-1} -> 0` as `|u| -> ∞` for both `p=2` and `p=3`.",
        "This page therefore claims existence/solvability for both maintained problems. It does not claim global uniqueness, because the source note proves existence, not a uniqueness theorem for the shifted arctan forcing.",
        "",
        "## Discretization And Algorithm Notes",
        "",
        "- Domain: structured `P1` right-triangle meshes on `(0,1)^2`.",
        "- Serial certified mesh ladder: levels `L4`, `L5`, `L6`; level `L7` is reserved for references and eigen diagnostics. The JAX + PETSc extension later in this page continues the same branch to finer levels.",
        "- `p=2` uses the exact `λ₁ = 2π²`.",
        "- `p=3` uses a level-matched discrete `λ₁,h` computed from a cached first-eigenpair stage with `||φ_{1,h}||_{L^3}=1`.",
        "- The maintained solution path is **certified `MPA + stationary Newton`**.",
        "- Raw `MPA/RMPA` runs are diagnostic only. Certified runs use a physically meaningful handoff state and then solve `J'(u)=0` directly.",
        "- The convergence histories report both the coordinate gradient norm and a Laplace-dual finite-element residual norm `||R_h(u)||_{K^{-1}}`.",
        "",
        "### Certified MPA + Newton",
        "",
        "The public results are organized around the discrete gradient, dual residual, and stationary merit",
        "",
        "$$",
        "g_h(u_h)=\\nabla J_{p,h}(u_h), \\qquad R_h(u_h)=K^{-1}g_h(u_h), \\qquad M(u_h)=\\tfrac12 g_h(u_h)^\\top K^{-1}g_h(u_h).",
        "$$",
        "",
        "The maintained algorithm is:",
        "",
        "1. **Raw mountain-pass branch search.** Build a polygonal path from `0` to a positive endpoint `e_h` with `J_{p,h}(e_h)<J_{p,h}(0)`, and at each outer step identify the current path peak `z_k`.",
        "2. **Auxiliary dissertation direction.** Compute the thesis-style descent direction by solving",
        "",
        "$$",
        "K a_k = g_h(z_k), \\qquad d_k = -\\frac{a_k}{|a_k|_{1,p,0}}.",
        "$$",
        "",
        "3. **Best-iterate handoff.** Update the polygonal path by a halved step from the peak, repair the local chain geometry, and retain the iterate with the smallest `\\|R_h(z_k)\\|_{K^{-1}}` as the certification handoff state.",
        "4. **Certified stationary Newton solve.** Starting from the handoff state, use JAX autodiff to build the discrete gradient and Hessian and solve",
        "",
        "$$",
        "(H_k + \\mu_k K)\\,\\delta_k = -g_h(u_k).",
        "$$",
        "",
        "5. **Merit-based globalization.** Accept a trial step `u_{k+1}=u_k+\\alpha_k\\delta_k` only when",
        "",
        "$$",
        "M(u_{k+1}) < M(u_k),",
        "$$",
        "",
        "with regularization and backtracking applied as needed.",
        "6. **Certification stop.** Declare convergence only when",
        "",
        "$$",
        "\\|R_h(u_k)\\|_{K^{-1}} \\le \\varepsilon_{\\mathrm{cert}}.",
        "$$",
        "",
        "7. **Continuation for `p=3`.** Certify the `p=2` branch first, then continue through `p=2.2, 2.4, 2.6, 2.8, 3.0` on a fixed mesh; each stage uses the previous certified state as the Newton initializer, with raw `MPA` available only as a fallback.",
        "",
        "## p = 2 Validation Study",
        "",
        "The validation problem is reported entirely through the certified workflow. The handoff residual shows the quality of the `MPA` branch-finder state, while the iteration columns show the raw `MPA` work, the Newton work, and their cumulative nonlinear total.",
        "",
        _markdown_table(
            ["level", "certification entry", "MPA handoff residual", "certified residual", "MPA iters", "Newton iters", "total nonlinear", "certified J", "status"],
            _main_mpa_rows(summary, p=2.0, epsilon=1.0e-5),
        ),
        "",
        (
            f"Private `L7` Newton reference residual: `{float(p2_reference['residual_norm']):.3e}`."
            if p2_reference.get("residual_norm") is not None
            else "Private `L7` Newton reference residual: unavailable in this summary."
        ),
        "",
        f"![p=2 certified solution panel]({rel_asset('p2_solution_panel.png')})",
        "",
        f"![p=2 certified Newton convergence history after MPA handoff]({rel_asset('p2_convergence_history.png')})",
        "",
        "## p = 3 Eigenvalue Stage",
        "",
        _markdown_table(
            ["level", "lambda1", "residual", "norm error", "iters", "status"],
            [
                [
                    str(payload["level"]),
                    _fmt(payload["lambda1"], 6),
                    _fmt(payload["residual_norm"], 6),
                    _fmt(payload["normalization_error"], 6),
                    str(payload["outer_iterations"]),
                    str(payload["status"]),
                ]
                for payload in lambda_payloads
            ],
        ),
        "",
        f"![p=3 eigenvalue convergence]({rel_asset('lambda_convergence.png')})",
        "",
        f"![p=3 eigenfunction]({rel_asset('p3_eigenfunction.png')})",
        "",
        "## p = 3 Main Study",
        "",
        "For `p=3`, the main published solution is the certified continuation branch. The continuation step itself is the essential globalization device; once a certified nearby state is available, the stationary Newton solve is short and robust, and the table reports both the `MPA` contribution and the Newton contribution explicitly.",
        "",
        _markdown_table(
            ["level", "lambda1,h", "certification entry", "certified residual", "MPA iters", "Newton iters", "total nonlinear", "certified J", "status"],
            _main_mpa_rows(summary, p=3.0, epsilon=1.0e-5, include_lambda=True, include_handoff_residual=False),
        ),
        "",
        "Representative `L6` continuation path:",
        "",
        _markdown_table(["from p", "to p", "path", "certified residual", "status"], _continuation_rows(p3_continuation)),
        "",
        (
            f"Private `L7` certified reference residual: `{float(p3_reference['certified']['residual_norm']):.3e}`."
            if p3_reference.get("certified", {}).get("residual_norm") is not None
            else "Private `L7` certified reference residual: unavailable in this summary."
        ),
        "",
        f"![p=3 certified solution panel]({rel_asset('p3_solution_panel.png')})",
        "",
        f"![p=3 certified continuation and Newton convergence history]({rel_asset('p3_convergence_history.png')})",
    ]
    if petsc_summary is not None:
        finest_scaling_level = petsc_summary.get("finest_scaling_level")
        lines.extend(
            [
                "",
                "## JAX + PETSc Backend",
                "",
                "The local stationary certification solve has now been rewritten as a JAX + PETSc backend. The branch-finding stage is still the certified `MPA` workflow described above, but the expensive fine-mesh Newton solve now uses PETSc `FGMRES` with a structured PMG preconditioner built from the stiffness matrix.",
                "",
                "The Hessian assembly uses a zero-safe element regularization",
                "",
                "$$",
                "|x|_{\\varepsilon_h}^q = \\bigl(x^2 + \\varepsilon_h^2\\bigr)^{q/2} - \\varepsilon_h^q, \\qquad \\varepsilon_h=10^{-12},",
                "$$",
                "",
                "which keeps the transferred warm-start Hessians finite while remaining visually indistinguishable from the unsmoothed functional at plotting scale.",
                "",
                "**Accepted PMG configuration**",
                "",
                "- Krylov solver: `FGMRES`",
                "- Preconditioner: fixed-stiffness Galerkin PMG",
                "- Multigrid smoother: **Chebyshev + Jacobi**",
                "- Coarse level: PETSc default coarse backend from the solver CLI",
                "- Near the tolerance floor: a small stagnation guard accepts convergence when the residual is already at the target scale and no further regularized step can produce a numerically resolvable merit decrease",
                "",
                "## PETSc Timing And Scaling",
                "",
                "The PETSc timing study extends the maintained ladder beyond the serial publication meshes. For `p=2` the warm-started PMG solve is continued to finer levels, and for `p=3` the PETSc backend is reported as a tuned continuation backend using the finest available certified eigenvalue cache from the serial study. The PETSc tables expose nonlinear outer iterations, Newton iterations, and cumulative Krylov iterations so the PMG workload is visible directly in the published summary. In this backend the nonlinear counter coincides with the Newton outer loop. The timing figure is log-log with an ideal `1:1` triangle, and the strong-scaling speedup panel is also shown on log-log axes against the ideal line.",
                "",
                _markdown_table(
                    ["level", "free dofs", "status", "residual", "setup [s]", "solve [s]", "total [s]", "nonlinear its", "Newton iters", "linear its"],
                    _petsc_mesh_rows(2.0),
                ),
                "",
                _markdown_table(
                    ["level", "free dofs", "status", "residual", "setup [s]", "solve [s]", "total [s]", "nonlinear its", "Newton iters", "linear its"],
                    _petsc_mesh_rows(3.0),
                ),
                "",
                f"![PETSc mesh timing]({rel_asset('petsc_mesh_timing.png')})",
            ]
        )
        if _petsc_scaling_rows():
            lines.extend(
                [
                    "",
                    (
                        f"Strong-scaling rows are reported on the finest successful `p=2` PETSc level `L{int(finest_scaling_level)}`."
                        if finest_scaling_level is not None
                        else "Strong-scaling rows are reported on the finest successful `p=2` PETSc level."
                    ),
                    "",
                    _markdown_table(
                        ["ranks", "status", "total [s]", "nonlinear its", "Newton iters", "linear its", "speedup", "efficiency", "residual"],
                        _petsc_scaling_rows(),
                    ),
                    "",
                    f"![PETSc strong scaling]({rel_asset('petsc_strong_scaling.png')})",
                ]
            )
    lines.extend(
        [
            "",
            "## Alternative Certified Branch: Shifted-Line RMPA + Newton",
            "",
            (
                f"A second certified branch-finding variant is now documented at the representative mesh level `L{comparison_level}`. "
                "Instead of projecting onto the origin-centred ray `t w`, it fixes a negative anchor `-C_1\\phi_h`, numerically maximizes the energy along the affine line through that anchor, and then uses the same stationary Newton certification stage."
            ),
            "This is still presented as an alternative branch finder rather than the maintained headline workflow, because its success is more seed-sensitive than the continuation-guided certified `MPA + Newton` path. The point of the section is to document the geometry change and its measured effect, not to replace the maintained solver story above.",
            "",
        ]
    )
    if shifted_rmpa_best:
        lines.extend(
            [
                _markdown_table(
                    ["p", "best seed", "raw status", "raw residual", "raw its", "certified status", "certified residual", "Newton iters", "time [s]"],
                    [
                        [
                            str(int(row["p"])),
                            str(row.get("seed_name", "-")),
                            str(row.get("raw_status", "-")),
                            _fmt(row.get("raw_residual_norm"), 6),
                            str(int(row.get("raw_outer_iterations", 0))),
                            str(row.get("certified_status", "-")),
                            _fmt(row.get("certified_residual_norm"), 6),
                            str(int(row.get("certified_newton_iters", 0))),
                            _fmt(row.get("solve_time_s"), 3),
                        ]
                        for row in shifted_rmpa_best
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Seed And Endpoint Geometry Comparison",
            "",
            (
                f"These tables keep the mesh fixed at `L{comparison_level}` so only the branch-search geometry and the start seed vary. "
                "For the MPA family, the geometry axis is one-sided `0 -> +C\\phi_h` versus symmetric `-C_1\\phi_h -> +C_2\\phi_h`. "
                "For the RMPA family, the geometry axis is the classical ray from `0` versus the shifted line from `-C_1\\phi_h`."
            ),
            "",
            "### MPA Family",
            "",
            _markdown_table(
                ["p", "geometry", "seed", "raw status", "raw residual", "raw its", "certified status", "certified residual", "Newton iters", "total nonlinear", "time [s]"],
                _comparison_table_rows(summary, family="MPA"),
            ),
            "",
            "### RMPA Family",
            "",
            _markdown_table(
                ["p", "geometry", "seed", "raw status", "raw residual", "raw its", "certified status", "certified residual", "Newton iters", "total nonlinear", "time [s]"],
                _comparison_table_rows(summary, family="RMPA"),
            ),
            "",
            "## Cross-Method Comparison",
            "",
            "The cross-method material is retained for completeness, but it is annexed below so the main narrative stays focused on the certified `MPA + stationary Newton` path.",
        "",
        "## Raw Versus Certified Diagnostics",
        "",
        "The remainder of the page is annex material. It records the raw globalization behavior and the failing `RMPA` variants so the successful certified path above stays readable and unambiguous.",
        "",
        "### Annex A — Cross-Method Diagnostics",
        "",
        "**`p=2` mesh-refinement diagnostics**",
        "",
        _markdown_table(
            ["method", "level", "raw status", "raw residual", "certified status", "certified residual", "raw seed", "certified handoff"],
            _raw_certified_rows(summary, p=2.0, epsilon=1.0e-5),
        ),
        "",
        "**`p=3` mesh-refinement diagnostics**",
        "",
        _markdown_table(
            ["method", "level", "raw status", "raw residual", "certified status", "certified residual", "raw seed", "certified handoff"],
            _raw_certified_rows(summary, p=3.0, epsilon=1.0e-5),
        ),
        "",
        f"![Iteration counts]({rel_asset('iteration_counts.png')})",
        "",
        f"![Reference error refinement]({rel_asset('reference_error_refinement.png')})",
        "",
        "### Annex B — RMPA And Failed Paths",
        "",
        "**`p=2` RMPA diagnostics**",
        "",
        _markdown_table(
            ["level", "raw status", "raw residual", "ray kind", "stable interior ray maximum?", "rationale"],
            _rmpa_rows(summary, p=2.0, epsilon=1.0e-5),
        ),
        "",
        "**`p=3` RMPA diagnostics**",
        "",
        _markdown_table(
            ["level", "raw status", "raw residual", "ray kind", "stable interior ray maximum?", "rationale"],
            _rmpa_rows(summary, p=3.0, epsilon=1.0e-5),
        ),
        "",
        "### Annex C — Tolerance Comparison",
        "",
        _markdown_table(["p", "method", "epsilon", "raw status", "raw residual", "certified status", "certified residual"], _tolerance_rows(summary, p=2.0) + _tolerance_rows(summary, p=3.0)),
        "",
        "Why `RMPA` stays in the annexes:",
        "",
        "- The classical origin-based `RMPA` still does not show the stable interior ray maximum required by its projection logic on the positive arctan branch.",
        "- The shifted-line variant documented above can work well, but only for some seed and geometry combinations, so it is still presented as a secondary certified branch finder rather than the maintained headline method.",
        "- Tightening the raw tolerance does not materially fix the origin-based `RMPA` rows on the published ladder.",
        "- The maintained successful path remains **certified `MPA + stationary Newton`**, with continuation in `p` providing the decisive stabilization for `p=3`.",
        "",
        "## Commands Used",
        "",
        "```bash",
        "./.venv/bin/python -u experiments/runners/run_plaplace_up_arctan_suite.py \\",
        "  --out-dir artifacts/raw_results/plaplace_up_arctan_full \\",
        "  --summary artifacts/raw_results/plaplace_up_arctan_full/summary.json",
        "```",
        "",
        "```bash",
        "./.venv/bin/python -u experiments/runners/run_plaplace_up_arctan_petsc_suite.py \\",
        "  --out-dir artifacts/raw_results/plaplace_up_arctan_petsc \\",
        "  --summary artifacts/raw_results/plaplace_up_arctan_petsc/summary.json",
        "```",
        "",
        "```bash",
        "./.venv/bin/python -u experiments/analysis/generate_plaplace_up_arctan_report.py \\",
        "  --summary artifacts/raw_results/plaplace_up_arctan_full/summary.json \\",
        "  --petsc-summary artifacts/raw_results/plaplace_up_arctan_petsc/summary.json \\",
        "  --out artifacts/reports/plaplace_up_arctan/README.md \\",
        "  --asset-dir docs/assets/plaplace_up_arctan",
        "```",
        "",
        "```bash",
        "./.venv/bin/python -u experiments/analysis/generate_plaplace_up_arctan_problem_page.py \\",
        "  --summary artifacts/raw_results/plaplace_up_arctan_full/summary.json \\",
        "  --petsc-summary artifacts/raw_results/plaplace_up_arctan_petsc/summary.json \\",
        "  --out docs/problems/pLaplace_up_arctan.md \\",
        "  --asset-dir docs/assets/plaplace_up_arctan",
        "```",
        "",
        "## Artifacts",
        "",
        "- Raw study summary: `artifacts/raw_results/plaplace_up_arctan_full/summary.json`",
        "- JAX + PETSc timing summary: `artifacts/raw_results/plaplace_up_arctan_petsc/summary.json`",
        "- Internal report: `artifacts/reports/plaplace_up_arctan/README.md`",
        "- Cached `p=3` eigen stage: `artifacts/raw_results/plaplace_up_arctan_full/lambda_cache/`",
        "- No separate `docs/results/pLaplace_up_arctan.md` page is maintained for this family.",
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
