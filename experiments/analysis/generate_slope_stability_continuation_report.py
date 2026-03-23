#!/usr/bin/env python3
"""Run a simple adaptive lambda continuation and generate a markdown report."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from src.problems.slope_stability.jax.solve_slope_stability_jax import build_solver_context, solve_lambda_step
from src.problems.slope_stability.support import DEFAULT_CASE


REPO_ROOT = Path(__file__).resolve().parents[2]


def _split_triangles_for_plot(elems_scalar: np.ndarray) -> np.ndarray:
    elem = np.asarray(elems_scalar, dtype=np.int64)
    split = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2], [5, 3, 4]], dtype=np.int64)
    tri = np.empty((4 * elem.shape[0], 3), dtype=np.int64)
    for i in range(elem.shape[0]):
        tri[4 * i : 4 * (i + 1), :] = elem[i][split]
    return tri


def _deviatoric_strain_norm_2d(E: np.ndarray) -> np.ndarray:
    iota = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    dev = np.diag([1.0, 1.0, 0.5]) - np.outer(iota, iota) / 2.0
    dev_e = dev @ E
    return np.sqrt(np.maximum(0.0, np.sum(E * dev_e, axis=0)))


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _element_mean_deviatoric(mesh, u_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords_final = np.asarray(mesh.params["nodes"], dtype=np.float64) + np.asarray(u_full, dtype=np.float64).reshape((-1, 2))
    elems_scalar = np.asarray(mesh.params["elems_scalar"], dtype=np.int64)
    elems_dof = np.asarray(mesh.params["elems"], dtype=np.int64)
    elem_B = np.asarray(mesh.params["elem_B"], dtype=np.float64)
    strain = np.einsum("eqij,ej->eqi", elem_B, np.asarray(u_full, dtype=np.float64)[elems_dof])
    dev_q = _deviatoric_strain_norm_2d(strain.reshape(-1, 3).T).reshape(strain.shape[0], strain.shape[1])
    dev_elem = np.mean(dev_q, axis=1)
    return coords_final, np.asarray(dev_elem, dtype=np.float64)


def _plot_deviatoric_step(mesh, u_full: np.ndarray, out_path: Path, *, title: str) -> None:
    coords_final, dev_elem = _element_mean_deviatoric(mesh, u_full)
    tri = _split_triangles_for_plot(np.asarray(mesh.params["elems_scalar"], dtype=np.int64))
    triangulation = mtri.Triangulation(coords_final[:, 0], coords_final[:, 1], triangles=tri)
    facecolors = np.repeat(dev_elem, 4)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    pc = ax.tripcolor(triangulation, facecolors=facecolors, shading="flat", cmap="magma")
    ax.triplot(triangulation, color="black", linewidth=0.20, alpha=0.16)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(pc, ax=ax, label="deviatoric strain")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_continuation_curve(accepted: list[dict[str, object]], out_path: Path) -> None:
    omega = np.asarray([float(step["omega"]) for step in accepted], dtype=np.float64)
    lam = np.asarray([float(step["lambda"]) for step in accepted], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.5, 5.25), dpi=180)
    ax.plot(omega, lam, marker="o", linewidth=1.6, color="#005F73")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(r"$\omega = f_{ext}^{T} u$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title("Adaptive direct continuation")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_markdown(
    out_path: Path,
    *,
    summary: dict[str, object],
    accepted: list[dict[str, object]],
    rejected: list[dict[str, object]],
    curve_png: str,
) -> None:
    lines = [
        "# Slope Stability P2 JAX Continuation Report",
        "",
        "This run uses a simple direct `lambda` continuation on top of the current pure-JAX prototype.",
        "It uses the corrected Davis-B raw-cohesion interpretation, so the internal Mohr-Coulomb threshold is no longer overstiffened by an extra `2 cos(phi)` factor.",
        "The source repository uses an indirect continuation workflow for this benchmark, but here the controller follows the requested simpler rule:",
        "`lambda <- lambda + d_lambda`, and if the solve fails then `d_lambda <- d_lambda / 2` and the step is retried from the last accepted state.",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Case | `{summary['case']}` |",
        f"| Lambda init | `{summary['lambda_init']}` |",
        f"| Initial increment | `{summary['d_lambda_init']}` |",
        f"| Minimum increment | `{summary['d_lambda_min']}` |",
        f"| Lambda max cap | `{summary['lambda_max']}` |",
        f"| Accepted steps | `{summary['accepted_steps']}` |",
        f"| Rejected attempts | `{summary['rejected_attempts']}` |",
        f"| Final accepted lambda | `{summary['lambda_last']:.9f}` |",
        f"| Final accepted omega | `{summary['omega_last']:.9f}` |",
        f"| Final accepted Umax | `{summary['u_max_last']:.9f}` |",
        f"| Stop reason | `{summary['stop_reason']}` |",
        f"| Runtime [s] | `{summary['runtime_s']:.3f}` |",
        "",
        "## Accepted Continuation Steps",
        "",
        "| Step | Lambda | d_lambda used | Omega | Umax | Newton | Linear | Message |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for step in accepted:
        lines.append(
            "| {step} | {lambda:.9f} | {d_lambda:.9f} | {omega:.9f} | {u_max:.9f} | {newton_iters} | {linear_iters} | `{message}` |".format(
                **step
            )
        )

    lines.extend(
        [
            "",
            "## Rejected Attempts",
            "",
            "| Attempt | Lambda trial | d_lambda trial | Newton | Message |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    if rejected:
        for attempt in rejected:
            lines.append(
                "| {attempt} | {lambda_trial:.9f} | {d_lambda_trial:.9f} | {newton_iters} | `{message}` |".format(
                    **attempt
                )
            )
    else:
        lines.append("| - | - | - | - | No rejected attempts |")

    lines.extend(
        [
            "",
            "## Continuation Curve",
            "",
            f"![Continuation curve]({curve_png})",
            "",
            "## Notes",
            "",
            "- The source repository's 2D homogeneous SSR benchmark reaches a final accepted `lambda` close to `1.2113` on its indirect continuation path.",
            f"- This prototype reached `lambda = {summary['lambda_last']:.9f}` because it still uses the simplified zero-history JAX constitutive path and the requested direct `lambda` stepping rule.",
            "- For the source repo's indirect 2D homogeneous SSR benchmark, `omega` is also the external work `f_ext^T u`; the mismatch is in the constitutive/continuation path, not in the `omega` definition.",
            "",
            "## Deviatoric Strain By Accepted Step",
            "",
            "Rejected steps are not shown below because their nonlinear solve did not converge.",
            "",
        ]
    )

    for step in accepted:
        lines.extend(
            [
                f"### Step {step['step']}",
                "",
                f"- `lambda = {step['lambda']:.9f}`",
                f"- `omega = {step['omega']:.9f}`",
                f"- `Umax = {step['u_max']:.9f}`",
                f"- `Newton iterations = {step['newton_iters']}`",
                "",
                f"![Deviatoric strain step {step['step']}]({step['deviatoric_png']})",
                "",
            ]
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_continuation(
    *,
    case: str,
    lambda_init: float,
    d_lambda_init: float,
    d_lambda_min: float,
    lambda_max: float,
    max_accepted_steps: int,
    maxit: int,
    linesearch_interval: tuple[float, float],
    linesearch_tol: float,
    ksp_rtol: float,
    ksp_max_it: int,
    tolf: float,
    tolg: float,
    tolg_rel: float,
    tolx_rel: float,
    tolx_abs: float,
    require_all_convergence: bool,
    use_trust_region: bool,
    trust_radius_init: float,
    trust_radius_min: float,
    trust_radius_max: float,
    trust_shrink: float,
    trust_expand: float,
    trust_eta_shrink: float,
    trust_eta_expand: float,
    trust_max_reject: int,
    trust_subproblem_line_search: bool,
    reg: float,
    out_dir: Path,
    verbose: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    context = build_solver_context(
        case=case,
        ksp_rtol=ksp_rtol,
        ksp_max_it=ksp_max_it,
        reg=reg,
        verbose=verbose,
    )
    mesh = context["mesh"]

    accepted_steps: list[dict[str, object]] = []
    rejected_attempts: list[dict[str, object]] = []
    full_displacement_hist: list[np.ndarray] = []

    current_lambda = float(lambda_init)
    current_step = float(d_lambda_init)
    current_guess = np.asarray(context["u_init_default"], dtype=np.float64)

    def _solve(candidate_lambda: float, guess: np.ndarray) -> dict[str, object]:
        return solve_lambda_step(
            context,
            lambda_target=float(candidate_lambda),
            u_guess=guess,
            maxit=maxit,
            linesearch_interval=linesearch_interval,
            linesearch_tol=linesearch_tol,
            tolf=tolf,
            tolg=tolg,
            tolg_rel=tolg_rel,
            tolx_rel=tolx_rel,
            tolx_abs=tolx_abs,
            require_all_convergence=require_all_convergence,
            use_trust_region=use_trust_region,
            trust_radius_init=trust_radius_init,
            trust_radius_min=trust_radius_min,
            trust_radius_max=trust_radius_max,
            trust_shrink=trust_shrink,
            trust_expand=trust_expand,
            trust_eta_shrink=trust_eta_shrink,
            trust_eta_expand=trust_eta_expand,
            trust_max_reject=trust_max_reject,
            trust_subproblem_line_search=trust_subproblem_line_search,
            verbose=verbose,
        )

    first = _solve(current_lambda, current_guess)
    if not bool(first["success"]):
        raise RuntimeError(f"Initial lambda {current_lambda:.6f} failed: {first['result']['message']}")

    accepted_steps.append(
        {
            "step": 1,
            "lambda": float(current_lambda),
            "d_lambda": 0.0,
            "omega": float(first["omega"]),
            "u_max": float(first["result"]["u_max"]),
            "newton_iters": int(first["result"]["newton_iters"]),
            "linear_iters": int(first["result"]["linear_iters"]),
            "message": str(first["result"]["message"]),
        }
    )
    current_guess = np.asarray(first["u_free"], dtype=np.float64)
    full_displacement_hist.append(np.asarray(first["displacement"], dtype=np.float64))

    stop_reason = "unknown"
    attempt_counter = 0
    while True:
        if len(accepted_steps) >= int(max_accepted_steps):
            stop_reason = "max_accepted_steps_reached"
            break
        if current_step < d_lambda_min:
            stop_reason = "d_lambda_min_reached"
            break
        candidate_lambda = current_lambda + current_step
        if candidate_lambda > lambda_max + 1.0e-12:
            stop_reason = "lambda_max_cap_reached"
            break

        attempt_counter += 1
        result = _solve(candidate_lambda, current_guess)
        if bool(result["success"]):
            current_lambda = float(candidate_lambda)
            current_guess = np.asarray(result["u_free"], dtype=np.float64)
            accepted_steps.append(
                {
                    "step": len(accepted_steps) + 1,
                    "lambda": float(current_lambda),
                    "d_lambda": float(current_step),
                    "omega": float(result["omega"]),
                    "u_max": float(result["result"]["u_max"]),
                    "newton_iters": int(result["result"]["newton_iters"]),
                    "linear_iters": int(result["result"]["linear_iters"]),
                    "message": str(result["result"]["message"]),
                }
            )
            full_displacement_hist.append(np.asarray(result["displacement"], dtype=np.float64))
            continue

        rejected_attempts.append(
            {
                "attempt": int(attempt_counter),
                "lambda_trial": float(candidate_lambda),
                "d_lambda_trial": float(current_step),
                "newton_iters": int(result["result"]["newton_iters"]),
                "message": str(result["result"]["message"]),
            }
        )
        current_step *= 0.5

    curve_png = out_dir / "continuation_curve.png"
    _plot_continuation_curve(accepted_steps, curve_png)

    for step, displacement in zip(accepted_steps, full_displacement_hist, strict=False):
        u_full = np.asarray(mesh.params["u_0"], dtype=np.float64).copy()
        u_full += displacement.reshape(-1)
        png_name = f"deviatoric_step_{int(step['step']):02d}.png"
        _plot_deviatoric_step(
            mesh,
            u_full,
            out_dir / png_name,
            title=(
                f"Accepted step {int(step['step'])}: "
                f"lambda={float(step['lambda']):.6f}, omega={float(step['omega']):.6f}"
            ),
        )
        step["deviatoric_png"] = png_name

    np.savez(
        out_dir / "continuation_states.npz",
        coords_ref=np.asarray(mesh.params["nodes"], dtype=np.float64),
        triangles=np.asarray(mesh.params["elems_scalar"], dtype=np.int32),
        lambda_hist=np.asarray([float(step["lambda"]) for step in accepted_steps], dtype=np.float64),
        omega_hist=np.asarray([float(step["omega"]) for step in accepted_steps], dtype=np.float64),
        u_max_hist=np.asarray([float(step["u_max"]) for step in accepted_steps], dtype=np.float64),
        displacement_hist=np.asarray(full_displacement_hist, dtype=np.float64),
    )

    summary = {
        "case": str(case),
        "lambda_init": float(lambda_init),
        "d_lambda_init": float(d_lambda_init),
        "d_lambda_min": float(d_lambda_min),
        "lambda_max": float(lambda_max),
        "accepted_steps": int(len(accepted_steps)),
        "rejected_attempts": int(len(rejected_attempts)),
        "lambda_last": float(accepted_steps[-1]["lambda"]),
        "omega_last": float(accepted_steps[-1]["omega"]),
        "u_max_last": float(accepted_steps[-1]["u_max"]),
        "stop_reason": str(stop_reason),
        "runtime_s": float(time.perf_counter() - start),
    }

    summary_path = out_dir / "continuation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "summary": _jsonify(summary),
                "accepted_steps": _jsonify(accepted_steps),
                "rejected_attempts": _jsonify(rejected_attempts),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    md_path = out_dir / "report.md"
    _write_markdown(
        md_path,
        summary=summary,
        accepted=accepted_steps,
        rejected=rejected_attempts,
        curve_png=curve_png.name,
    )
    return md_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=str, default=DEFAULT_CASE)
    parser.add_argument("--lambda-init", type=float, default=0.9)
    parser.add_argument("--d-lambda-init", type=float, default=0.1)
    parser.add_argument("--d-lambda-min", type=float, default=1e-4)
    parser.add_argument("--lambda-max", type=float, default=2.0)
    parser.add_argument("--max-accepted-steps", type=int, default=24)
    parser.add_argument("--out-dir", type=str, default="artifacts/reports/slope_stability_p2_jax_continuation")
    parser.add_argument("--maxit", type=int, default=50)
    parser.add_argument("--linesearch-a", type=float, default=-0.5)
    parser.add_argument("--linesearch-b", type=float, default=2.0)
    parser.add_argument("--linesearch-tol", type=float, default=1e-1)
    parser.add_argument("--ksp-rtol", type=float, default=1e-1)
    parser.add_argument("--ksp-max-it", type=int, default=30)
    parser.add_argument("--tolf", type=float, default=1e-4)
    parser.add_argument("--tolg", type=float, default=1e-3)
    parser.add_argument("--tolg-rel", type=float, default=1e-3)
    parser.add_argument("--tolx-rel", type=float, default=1e-3)
    parser.add_argument("--tolx-abs", type=float, default=1e-10)
    parser.add_argument("--require-all-convergence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-trust-region", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-radius-init", type=float, default=0.5)
    parser.add_argument("--trust-radius-min", type=float, default=1e-8)
    parser.add_argument("--trust-radius-max", type=float, default=1e6)
    parser.add_argument("--trust-shrink", type=float, default=0.5)
    parser.add_argument("--trust-expand", type=float, default=1.5)
    parser.add_argument("--trust-eta-shrink", type=float, default=0.05)
    parser.add_argument("--trust-eta-expand", type=float, default=0.75)
    parser.add_argument("--trust-max-reject", type=int, default=6)
    parser.add_argument("--trust-subproblem-line-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reg", type=float, default=1.0e-12)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_dir_arg = Path(args.out_dir)
    out_dir = out_dir_arg.resolve() if out_dir_arg.is_absolute() else (REPO_ROOT / out_dir_arg).resolve()
    md_path = run_continuation(
        case=args.case,
        lambda_init=args.lambda_init,
        d_lambda_init=args.d_lambda_init,
        d_lambda_min=args.d_lambda_min,
        lambda_max=args.lambda_max,
        max_accepted_steps=args.max_accepted_steps,
        maxit=args.maxit,
        linesearch_interval=(float(args.linesearch_a), float(args.linesearch_b)),
        linesearch_tol=args.linesearch_tol,
        ksp_rtol=args.ksp_rtol,
        ksp_max_it=args.ksp_max_it,
        tolf=args.tolf,
        tolg=args.tolg,
        tolg_rel=args.tolg_rel,
        tolx_rel=args.tolx_rel,
        tolx_abs=args.tolx_abs,
        require_all_convergence=args.require_all_convergence,
        use_trust_region=args.use_trust_region,
        trust_radius_init=args.trust_radius_init,
        trust_radius_min=args.trust_radius_min,
        trust_radius_max=args.trust_radius_max,
        trust_shrink=args.trust_shrink,
        trust_expand=args.trust_expand,
        trust_eta_shrink=args.trust_eta_shrink,
        trust_eta_expand=args.trust_eta_expand,
        trust_max_reject=args.trust_max_reject,
        trust_subproblem_line_search=args.trust_subproblem_line_search,
        reg=args.reg,
        out_dir=out_dir,
        verbose=not args.quiet,
    )
    print(str(md_path))


if __name__ == "__main__":
    main()
