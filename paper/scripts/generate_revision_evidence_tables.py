#!/usr/bin/env python3
"""Generate the evidence-constrained tables used by the revised manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from admit_revision_publication_evidence import (
    EVIDENCE_SPECS,
    validate_publication_source_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "paper/tables/generated"
DEFAULT_EVIDENCE_ROOT = (
    REPO_ROOT / "artifacts/reproduction/paper_revision_2026_07_10/pilots"
)

INPUT_RELATIVE_PATHS = {spec.key: spec.relative_path for spec in EVIDENCE_SPECS}


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "worktree_clean": not bool(status.strip()),
    }


def _input_paths(evidence_root: Path) -> dict[str, Path]:
    root = evidence_root.resolve()
    return {key: root / relative for key, relative in INPUT_RELATIVE_PATHS.items()}


def _sci(value: float, digits: int = 2) -> str:
    return rf"\num{{{float(value):.{digits}e}}}"


def _fixed(value: float, digits: int = 3) -> str:
    return rf"\num{{{float(value):.{digits}f}}}"


def _distribution_rank_levels(payload: dict[str, Any]) -> list[int]:
    varied_factor = payload.get("varied_factor")
    if not isinstance(varied_factor, dict) or varied_factor.get("name") != "mpi_ranks":
        raise ValueError("distribution evidence must vary mpi_ranks")
    raw_levels = varied_factor.get("levels")
    if not isinstance(raw_levels, list) or not raw_levels:
        raise ValueError("distribution evidence must record nonempty mpi_ranks levels")
    if any(
        isinstance(level, bool) or not isinstance(level, int) or level <= 0
        for level in raw_levels
    ):
        raise ValueError("distribution mpi_ranks levels must be positive integers")
    if len(set(raw_levels)) != len(raw_levels):
        raise ValueError("distribution mpi_ranks levels must be unique")
    return raw_levels


def _distribution_rank_label(levels: list[int]) -> str:
    words = {1: "one", 2: "two", 4: "four"}
    level_label = "/".join(words.get(level, str(level)) for level in levels)
    return f"Hyperelasticity, {level_label} ranks"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _verification_table(data: dict[str, dict[str, Any]]) -> str:
    pl = data["plaplace"]
    gl = data["ginzburg_landau"]
    patch = data["hyperelastic_patch"]
    nonaffine = data["hyperelastic_nonaffine"]
    pl_rate, gl_rate, he_rate = pl["rates"][-1], gl["rates"][-1], nonaffine["rates"][-1]
    pl_fine, gl_fine, he_fine = pl["levels"][-1], gl["levels"][-1], nonaffine["levels"][-1]
    rows = [
        [
            "$p$-Laplace manufactured",
            _fixed(pl_rate["l2_rate"]),
            _fixed(pl_rate["h1_seminorm_rate"]),
            _sci(pl_fine["final_relative_residual"]),
            "smooth P1 weak problem",
        ],
        [
            "Ginzburg--Landau manufactured",
            _fixed(gl_rate["l2_rate"]),
            _fixed(gl_rate["h1_seminorm_rate"]),
            _sci(gl_fine["final_relative_residual"]),
            "controlled positive branch",
        ],
        [
            "Hyperelastic affine patch",
            "--",
            "--",
            _sci(max(patch["metrics"][key] for key in ("energy_relative_error", "residual_relative_error", "hessian_relative_error"))),
            "analytic $W$, $P$, and tangent",
        ],
        [
            "Hyperelastic nonaffine manufactured",
            _fixed(he_rate["l2_displacement_error"]),
            _fixed(he_rate["h1_deformation_error"]),
            _sci(he_fine["final_relative_residual"]),
            "Piola order " + _fixed(he_rate["first_piola_l2_error"]),
        ],
    ]
    body = "\n".join("    " + " & ".join(row) + r" \\" for row in rows)
    return rf"""\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\RaggedRight\arraybackslash}}X *{{3}}{{>{{\Centering\arraybackslash}}p{{0.12\textwidth}}}} >{{\RaggedRight\arraybackslash}}p{{0.25\textwidth}}@{{}}}}
  \toprule
  Verification problem & $L^2$ order & $H^1$ order & \shortstack{{Residual\\or defect}} & Scope \\
  \midrule
{body}
  \bottomrule
\end{{tabularx}}"""


def _derivative_table(data: dict[str, dict[str, Any]]) -> str:
    derivative_rows: list[list[str]] = []
    for key, label in (
        ("smooth_derivatives", "Smooth local energies"),
        ("p1_derivatives", "$P_1(L_1)$ plasticity"),
        ("p2_derivatives", "$P_2(L_1)$ plasticity"),
        ("p4_derivatives", "$P_4(L_1)$ plasticity"),
    ):
        summary = data[key]["summary"]
        state_count = int(summary.get("cases", summary.get("states", 0)))
        route_defect = float(
            summary.get("maximum_hessian_relative_error", 0.0)
        )
        fd_defect = float(
            summary.get(
                "maximum_fd_hvp_error_at_gate",
                summary.get("maximum_centered_fd_hvp_error_at_gate", 0.0),
            )
        )
        assembled = data[key].get("assembled_route_equivalence")
        if assembled is None:
            assembled_defect = "--"
            scope = "fixed element"
        else:
            if not isinstance(assembled, dict) or assembled.get("status") != "passed":
                raise ValueError("assembled derivative evidence must have passed status")
            comparisons = assembled.get("pairwise_comparisons")
            if not isinstance(comparisons, list) or len(comparisons) != 3:
                raise ValueError(
                    "assembled derivative evidence must contain three pairwise comparisons"
                )
            if any(
                not isinstance(row, dict)
                or row.get("hessian_csr_structure_equal") is not True
                or row.get("passed") is not True
                for row in comparisons
            ):
                raise ValueError("assembled derivative CSR comparison did not pass")
            assembled_values = [
                float(row["hessian_relative_error"]) for row in comparisons
            ]
            if any(not math.isfinite(value) or value < 0.0 for value in assembled_values):
                raise ValueError("assembled derivative defects must be finite and nonnegative")
            assembled_defect = _sci(max(assembled_values))
            scope = "five element states; one assembled state"
        derivative_rows.append(
            [
                label,
                str(state_count),
                _sci(route_defect),
                _sci(fd_defect),
                assembled_defect,
                scope,
            ]
        )
    mc = data["material_point"]["summary"]
    derivative_rows.append(
        [
            "Mohr--Coulomb material point",
            str(sum(int(value) for value in mc["branch_interior_counts"].values())),
            _sci(mc["maximum_hessian_symmetry_defect"]),
            _sci(mc["maximum_centered_hvp_error_at_gate"]),
            "--",
            "five branch interiors; switches excluded",
        ]
    )
    distribution = data["distribution"]
    dist = distribution["comparison"]["relative_errors"]
    rank_levels = _distribution_rank_levels(distribution)
    derivative_rows.append(
        [
            _distribution_rank_label(rank_levels),
            str(len(rank_levels)),
            _sci(max(dist["residual_relative"], dist["matrix_relative"])),
            _sci(dist["matrix_action_relative"]),
            "--",
            "fixed state and canonical ordering",
        ]
    )
    body = "\n".join(
        "    " + " & ".join(row) + r" \\" for row in derivative_rows
    )
    return rf"""\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\RaggedRight\arraybackslash}}X >{{\Centering\arraybackslash}}p{{0.07\textwidth}} *{{3}}{{>{{\Centering\arraybackslash}}p{{0.13\textwidth}}}} >{{\RaggedRight\arraybackslash}}p{{0.22\textwidth}}@{{}}}}
  \toprule
  Block & Element states & Element route/symmetry defect & FD/action defect & Assembled CSR defect & Scope \\
  \midrule
{body}
  \bottomrule
\end{{tabularx}}"""


def _quadrature_row(payload: dict[str, Any]) -> list[str]:
    evaluations = payload["evaluations"]
    solve_id = str(payload["solve_quadrature_rule_id"])
    solve = next(row for row in evaluations if str(row["quadrature_rule_id"]) == solve_id)
    reference = next(
        row for row in evaluations if str(row["quadrature_rule_id"]) == str(payload["reference_rule_id"])
    )
    return [
        f"$P_{int(payload['element_degree'])}(L_1)$",
        str(int(solve["quadrature_points_per_element"])),
        _sci(solve["relative_total_potential_difference_from_last_rule"]),
        _sci(solve["free_residual_l2_norm"]),
        _sci(reference["free_residual_l2_norm"]),
        _sci(solve["free_hessian_action_vector_comparison_to_last_rule"]["relative_l2_difference"]),
    ]


def _quadrature_table(data: dict[str, dict[str, Any]]) -> str:
    rows = [
        _quadrature_row(data[key])
        for key in ("p1_quadrature", "p2_quadrature", "p4_quadrature")
    ]
    body = "\n".join("    " + " & ".join(row) + r" \\" for row in rows)
    return rf"""\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\Centering\arraybackslash}}p{{0.10\textwidth}} *{{5}}{{>{{\Centering\arraybackslash}}X}}@{{}}}}
  \toprule
  Space & Degree-rule $n_q$ & Relative energy difference & Degree-rule residual & Reference-rule residual & Relative tangent-action difference \\
  \midrule
{body}
  \bottomrule
\end{{tabularx}}"""


def _evidence_status_table(
    data: dict[str, dict[str, Any]], *, evidence_class: str
) -> str:
    analysis = data["route_analysis"]
    empirical = analysis["empirical_map"]
    cost = analysis["cost_model"]
    admitted = sum(1 for row in empirical if str(row.get("status")) == "admitted")
    total = len(empirical)
    active_rows = sum(
        1 for row in empirical if str(row.get("status")) != "censored"
    )
    terminal = analysis.get("terminal_decision")
    publication_route_evidence = evidence_class == "publication" and terminal in {
        "predictive_selector_admissible",
        "finite_empirical_map_only",
    }
    selector_admitted = (
        publication_route_evidence
        and bool(cost.get("selector_claim_admissible"))
        and terminal == "predictive_selector_admissible"
    )
    finite_map_admitted = publication_route_evidence and total > 0 and admitted == active_rows
    map_decision = (
        "admitted finite map"
        if finite_map_admitted
        else (
            (
                "complete diagnostic map"
                if total > 0 and admitted == active_rows
                else ("partial diagnostic map" if admitted else "no diagnostic map")
            )
            if evidence_class != "publication"
            else ("partially admitted map" if admitted else "no admitted map")
        )
    )
    selector_decision = "admitted" if selector_admitted else "not admitted"
    timing_decision = (
        "admitted descriptive paired timing"
        if finite_map_admitted
        else "not admissible"
    )
    factor = analysis.get("factorized_microbenchmark_gate")
    factor_passed = isinstance(factor, dict) and factor.get("passed") is True
    factor_failures = (
        len(factor.get("failures") or []) if isinstance(factor, dict) else 0
    )
    rows = [
        [
            "Fixed-state route map",
            f"{admitted}/{total}",
            map_decision,
            "contract satisfied"
            if finite_map_admitted
            else "collective-max proof and clean records",
        ],
        [
            "Predictive cost selector",
            (
                f"{int(cost['training_rows'])} train; "
                f"{int(cost['holdout_rows'])} holdout"
            ),
            selector_decision,
            (
                "contract satisfied"
                if selector_admitted and publication_route_evidence
                else (
                    "negative terminal recorded; no predictive claim"
                    if finite_map_admitted
                    else "paired distributed design and held-out validation"
                )
            ),
        ],
        [
            "Descriptive paired route timing",
            (
                f"{admitted} publication rows"
                if finite_map_admitted
                else f"{admitted} diagnostic rows"
            ),
            timing_decision,
            "contract satisfied"
            if finite_map_admitted
            else "equal-accuracy repeated cluster runs",
        ],
        [
            "Post-fit crossover location",
            "0 confirmation rows",
            "not evaluated",
            "separate hash-bound confirmation study",
        ],
        [
            "Synthetic factor diagnostic",
            f"{factor_failures} recorded failures",
            (
                "descriptive diagnostic passed"
                if factor_passed
                else "descriptive diagnostic reported"
            ),
            "not a selector gate",
        ],
    ]
    body = "\n".join("    " + " & ".join(row) + r" \\" for row in rows)
    return rf"""\begin{{tabularx}}{{\textwidth}}{{@{{}}>{{\RaggedRight\arraybackslash}}p{{0.23\textwidth}} >{{\Centering\arraybackslash}}p{{0.15\textwidth}} >{{\RaggedRight\arraybackslash}}X >{{\RaggedRight\arraybackslash}}p{{0.30\textwidth}}@{{}}}}
  \toprule
  Evidence block & Rows & Current decision & Required promotion evidence \\
  \midrule
{body}
  \bottomrule
\end{{tabularx}}"""


def generate(
    output_dir: Path,
    *,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    evidence_class: str = "diagnostic",
    evidence_manifest: Path | None = None,
) -> dict[str, Any]:
    evidence_root = evidence_root.resolve()
    inputs = _input_paths(evidence_root)
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing revision evidence inputs: " + ", ".join(missing))
    if evidence_class not in {"diagnostic", "publication"}:
        raise ValueError("evidence_class must be diagnostic or publication")
    git = _git_metadata()
    source_manifest: dict[str, Any] | None = None
    if evidence_class == "publication":
        if evidence_manifest is None:
            raise ValueError(
                "--evidence-manifest is required for --evidence-class publication"
            )
        source_manifest = validate_publication_source_manifest(
            evidence_manifest.resolve(),
            evidence_root=evidence_root,
            repo_root=REPO_ROOT,
            expected_inputs=inputs,
        )
    elif evidence_manifest is not None:
        raise ValueError(
            "--evidence-manifest is accepted only with --evidence-class publication"
        )

    data = {key: _read(path) for key, path in inputs.items()}
    tables = {
        "revision_verification_summary.tex": _verification_table(data),
        "revision_derivative_checks.tex": _derivative_table(data),
        "revision_quadrature_sensitivity.tex": _quadrature_table(data),
        "revision_evidence_status.tex": _evidence_status_table(
            data, evidence_class=evidence_class
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in tables.items():
        _write(output_dir / name, content)
    manifest = {
        "schema_version": 2,
        "generator": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "evidence_class": evidence_class,
        "evidence_root": _display_path(evidence_root),
        "publication_evidence": evidence_class == "publication",
        "status": (
            "clean_publication_tables"
            if evidence_class == "publication"
            else "diagnostic_tables_not_for_submission"
        ),
        "git": git,
        "source_evidence_manifest": (
            {
                "path": _display_path(evidence_manifest.resolve()),
                "sha256": _sha256(evidence_manifest.resolve()),
                "schema_id": source_manifest["schema_id"],
            }
            if source_manifest is not None and evidence_manifest is not None
            else None
        ),
        "inputs": {
            key: {
                "path": _display_path(path),
                "path_within_evidence_root": INPUT_RELATIVE_PATHS[key].as_posix(),
                "sha256": _sha256(path),
            }
            for key, path in inputs.items()
        },
        "outputs": {
            name: _sha256(output_dir / name) for name in sorted(tables)
        },
        "interpretation": (
            "All input rows were admitted by the clean publication source manifest."
            if evidence_class == "publication"
            else "The generated numbers are diagnostic and must not enter a submission bundle."
        ),
    }
    _write(output_dir / "revision_evidence_manifest.json", json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument(
        "--evidence-class",
        choices=("diagnostic", "publication"),
        default="diagnostic",
    )
    parser.add_argument("--evidence-manifest", type=Path)
    args = parser.parse_args()
    generate(
        Path(args.out_dir).resolve(),
        evidence_root=Path(args.evidence_root).resolve(),
        evidence_class=str(args.evidence_class),
        evidence_manifest=(
            None if args.evidence_manifest is None else Path(args.evidence_manifest).resolve()
        ),
    )


if __name__ == "__main__":
    main()
