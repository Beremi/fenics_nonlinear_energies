from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _iter_arrays(history: list[dict]) -> dict[str, np.ndarray]:
    it = np.array([int(row["it"]) for row in history], dtype=int)
    energy = np.array([float(row["energy"]) for row in history], dtype=float)
    grad = np.array([float(row["grad_norm"]) for row in history], dtype=float)
    alpha = np.array([float(row["alpha"]) for row in history], dtype=float)
    t_grad = np.array([float(row["t_grad"]) for row in history], dtype=float)
    t_hess = np.array([float(row["t_hess"]) for row in history], dtype=float)
    t_ls = np.array([float(row["t_ls"]) for row in history], dtype=float)
    t_update = np.array([float(row["t_update"]) for row in history], dtype=float)
    t_iter = np.array([float(row["t_iter"]) for row in history], dtype=float)
    ksp_max = np.array(
        [int(row.get("linear_iteration", {}).get("ksp_its_max", 0)) for row in history],
        dtype=int,
    )
    ksp_mean = np.array(
        [
            float(row.get("linear_iteration", {}).get("ksp_its_mean", 0.0))
            for row in history
        ],
        dtype=float,
    )
    ksp_sum = np.array(
        [int(row.get("linear_iteration", {}).get("ksp_its_sum", 0)) for row in history],
        dtype=int,
    )
    t_lin_assemble = np.array(
        [
            float(row.get("linear_iteration", {}).get("t_assemble_max", 0.0))
            for row in history
        ],
        dtype=float,
    )
    t_lin_setup = np.array(
        [
            float(row.get("linear_iteration", {}).get("t_setup_max", 0.0))
            for row in history
        ],
        dtype=float,
    )
    t_lin_solve = np.array(
        [
            float(row.get("linear_iteration", {}).get("t_solve_max", 0.0))
            for row in history
        ],
        dtype=float,
    )
    lin_true_rel = np.array(
        [
            float(
                row.get("linear_iteration", {}).get("true_relative_residual_max", np.nan)
            )
            for row in history
        ],
        dtype=float,
    )
    rss_current_max = np.array(
        [
            float(row.get("memory_profile", {}).get("rss_current_max_gib", np.nan))
            for row in history
        ],
        dtype=float,
    )
    rss_current_mean = np.array(
        [
            float(row.get("memory_profile", {}).get("rss_current_mean_gib", np.nan))
            for row in history
        ],
        dtype=float,
    )
    rss_hwm_max = np.array(
        [float(row.get("memory_profile", {}).get("rss_hwm_max_gib", np.nan)) for row in history],
        dtype=float,
    )
    hess_other = np.maximum(t_hess - (t_lin_assemble + t_lin_setup + t_lin_solve), 0.0)
    return {
        "it": it,
        "energy": energy,
        "grad": grad,
        "alpha": alpha,
        "t_grad": t_grad,
        "t_hess": t_hess,
        "t_ls": t_ls,
        "t_update": t_update,
        "t_iter": t_iter,
        "ksp_max": ksp_max,
        "ksp_mean": ksp_mean,
        "ksp_sum": ksp_sum,
        "t_lin_assemble": t_lin_assemble,
        "t_lin_setup": t_lin_setup,
        "t_lin_solve": t_lin_solve,
        "lin_true_rel": lin_true_rel,
        "rss_current_max": rss_current_max,
        "rss_current_mean": rss_current_mean,
        "rss_hwm_max": rss_hwm_max,
        "hess_other": hess_other,
    }


def plot_convergence(arr: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    ax.plot(arr["it"], arr["energy"], marker="o", linewidth=2.0)
    ax.set_title("Energy by Newton Iteration")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.semilogy(arr["it"], arr["grad"], marker="o", linewidth=2.0, label="||g||")
    ax.set_title("Gradient Norm by Newton Iteration")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("Gradient norm")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out)


def plot_timing(arr: dict[str, np.ndarray], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    width = 0.72
    bottom = np.zeros_like(arr["it"], dtype=float)
    parts = (
        ("t_grad", "Gradient"),
        ("t_lin_assemble", "Linear assemble"),
        ("t_lin_setup", "Linear KSP setup"),
        ("t_lin_solve", "Linear KSP solve"),
        ("hess_other", "Other Hessian"),
        ("t_ls", "Line search"),
        ("t_update", "Update"),
    )
    for key, label in parts:
        vals = arr[key]
        ax.bar(arr["it"], vals, width=width, bottom=bottom, label=label)
        bottom += vals
    ax.plot(arr["it"], arr["t_iter"], color="black", linewidth=1.8, marker="o", label="Total iter")
    ax.set_title("Per-Iteration Timing Breakdown")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("Time [s]")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4)
    _save(fig, out)


def plot_linear(arr: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True)

    ax = axes[0]
    ax.plot(arr["it"], arr["ksp_max"], marker="o", linewidth=2.0, label="KSP its max")
    ax.plot(arr["it"], arr["ksp_mean"], marker="s", linewidth=2.0, label="KSP its mean")
    ax.plot(arr["it"], arr["ksp_sum"], marker="^", linewidth=2.0, label="KSP its sum")
    ax.set_title("Linear Iteration Counts by Newton Step")
    ax.set_ylabel("Iterations")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=3)

    ax = axes[1]
    ax.semilogy(
        arr["it"],
        arr["lin_true_rel"],
        marker="o",
        linewidth=2.0,
        label="True relative residual max",
    )
    ax2 = ax.twinx()
    ax2.plot(arr["it"], arr["alpha"], color="tab:red", marker="s", linewidth=1.8, label="alpha")
    ax.set_title("Linear Residual / Step Acceptance")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("True rel residual")
    ax2.set_ylabel("alpha")
    ax.grid(True, which="both", alpha=0.25)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")
    _save(fig, out)


def plot_memory(arr: dict[str, np.ndarray], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.plot(arr["it"], arr["rss_current_max"], marker="o", linewidth=2.0, label="RSS current max")
    ax.plot(arr["it"], arr["rss_current_mean"], marker="s", linewidth=2.0, label="RSS current mean")
    ax.plot(arr["it"], arr["rss_hwm_max"], marker="^", linewidth=2.0, label="RSS HWM max")
    ax.set_title("Per-Iteration Memory Profile")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("Memory [GiB]")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out)


def _fmt(val: float, fmt: str) -> str:
    return format(float(val), fmt)


def build_report(obj: dict, history: list[dict], report_path: Path, asset_dir: Path) -> None:
    lin = obj["linear_solver"]
    mesh = obj["mesh"]
    stage = obj["stage_timings"]
    init_guess = obj["initial_guess"]
    mg = obj["mg_hierarchy"]
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# `P4(L1_2), lambda = 1.5, np = 32` Newton Diagnostic")
    lines.append("")
    lines.append("## Outcome")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("| --- | ---: |")
    lines.append(f"| status | `{obj['status']}` |")
    lines.append(f"| message | `{obj['message']}` |")
    lines.append(f"| nonlinear iterations | `{obj['nit']}` |")
    lines.append(f"| final gradient norm | `{obj['final_grad_norm']:.6e}` |")
    lines.append(f"| energy | `{obj['energy']:.10e}` |")
    lines.append(f"| `u_max` | `{obj['u_max']:.10e}` |")
    lines.append(f"| `omega` | `{obj['omega']:.10e}` |")
    lines.append(f"| linear iterations total | `{obj['linear_iterations_total']}` |")
    lines.append(f"| solve time | `{obj['solve_time']:.3f} s` |")
    lines.append(f"| end-to-end wall time | `{obj['total_time']:.3f} s` |")
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    lines.append("| knob | value |")
    lines.append("| --- | --- |")
    lines.append(f"| mesh / space | `hetero_ssr_L1_2`, `P4` |")
    lines.append(f"| MPI ranks | `32` |")
    lines.append(f"| lambda | `{obj['lambda_target']}` |")
    lines.append(f"| hierarchy | `P4(L1_2) -> P2(L1_2) -> P1(L1_2) -> P1(L1)` |")
    lines.append(f"| nonlinear method | `Newton + {history[0].get('line_search', 'armijo')}` |")
    lines.append(f"| initial guess | `elastic_initial_guess = {bool(init_guess.get('enabled', False))}` |")
    lines.append(f"| linear method | `{lin['ksp_type']} + {lin['pc_type']}` |")
    lines.append(f"| `ksp_rtol / ksp_max_it` | `{lin['ksp_rtol']} / {lin['ksp_max_it']}` |")
    lines.append(f"| distribution | `{lin['distribution_strategy']}` |")
    lines.append(f"| build modes | `{lin['problem_build_mode']}`, `{lin['mg_level_build_mode']}`, `{lin['mg_transfer_build_mode']}` |")
    lines.append(f"| reorder | `{lin['element_reorder_mode']}` |")
    lines.append(f"| coarse solve | `{lin['mg_coarse_ksp_type']} + {lin['mg_coarse_pc_type']}` |")
    lines.append(f"| Hypre nodal / interp / strong | `{lin['mg_coarse_hypre_nodal_coarsen']}` / `{lin['mg_coarse_hypre_vec_interp_variant']}` / `{lin['mg_coarse_hypre_strong_threshold']}` |")
    lines.append(f"| smoothers | `P1 {lin['mg_p1_smoother_ksp_type']}+{lin['mg_p1_smoother_pc_type']} x{lin['mg_p1_smoother_steps']}`, `P2 {lin['mg_p2_smoother_ksp_type']}+{lin['mg_p2_smoother_pc_type']} x{lin['mg_p2_smoother_steps']}`, `P4 {lin['mg_p4_smoother_ksp_type']}+{lin['mg_p4_smoother_pc_type']} x{lin['mg_p4_smoother_steps']}` |")
    lines.append(f"| near-nullspace | `{lin['use_near_nullspace']}` |")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("| stage | time |")
    lines.append("| --- | ---: |")
    lines.append(f"| problem load | `{stage['problem_load']:.3f} s` |")
    lines.append(f"| assembler create | `{stage['assembler_create']:.3f} s` |")
    lines.append(f"| MG hierarchy build | `{stage['mg_hierarchy_build']:.3f} s` |")
    lines.append(f"| elastic initial guess | `{stage['initial_guess_total']:.3f} s` |")
    lines.append(f"| elastic initial-guess solve | `{float(init_guess.get('solve_time', 0.0)):.3f} s` |")
    lines.append(f"| elastic initial-guess KSP its | `{int(init_guess.get('ksp_iterations', 0))}` |")
    lines.append(f"| MG transfer build time | `{float(mg.get('transfer_build_time', 0.0)):.3f} s` |")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append(f"![convergence]({asset_dir.name}/plasticity3d_p4_l1_2_np32_maxit20_convergence.png)")
    lines.append("")
    lines.append(f"![timing]({asset_dir.name}/plasticity3d_p4_l1_2_np32_maxit20_newton_timing.png)")
    lines.append("")
    lines.append(f"![linear]({asset_dir.name}/plasticity3d_p4_l1_2_np32_maxit20_linear_diagnostics.png)")
    lines.append("")
    lines.append(f"![memory]({asset_dir.name}/plasticity3d_p4_l1_2_np32_maxit20_memory_profile.png)")
    lines.append("")
    lines.append("## Newton Table")
    lines.append("")
    lines.append("| it | energy | `||g||` | `alpha` | KSP max | lin solve max [s] | lin assemble max [s] | lin setup max [s] | `t_grad` [s] | `t_hess` [s] | `t_ls` [s] | `t_iter` [s] | RSS max [GiB] | HWM max [GiB] |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in history:
        lin_row = row.get("linear_iteration", {})
        mem_row = row.get("memory_profile", {})
        lines.append(
            "| "
            f"`{int(row['it'])}` | "
            f"`{float(row['energy']):.6e}` | "
            f"`{float(row['grad_norm']):.6e}` | "
            f"`{float(row['alpha']):.5f}` | "
            f"`{int(lin_row.get('ksp_its_max', 0))}` | "
            f"`{float(lin_row.get('t_solve_max', 0.0)):.3f}` | "
            f"`{float(lin_row.get('t_assemble_max', 0.0)):.3f}` | "
            f"`{float(lin_row.get('t_setup_max', 0.0)):.3f}` | "
            f"`{float(row['t_grad']):.3f}` | "
            f"`{float(row['t_hess']):.3f}` | "
            f"`{float(row['t_ls']):.3f}` | "
            f"`{float(row['t_iter']):.3f}` | "
            f"`{float(mem_row.get('rss_current_max_gib', 0.0)):.3f}` | "
            f"`{float(mem_row.get('rss_hwm_max_gib', 0.0)):.3f}` |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args()

    output_json = Path(args.output_json)
    asset_dir = Path(args.asset_dir)
    report_path = Path(args.report_path)

    obj = _load_json(output_json)
    history = list(obj.get("history", []))
    if not history:
        raise RuntimeError("Output JSON does not contain Newton history")

    arr = _iter_arrays(history)
    asset_dir.mkdir(parents=True, exist_ok=True)

    plot_convergence(arr, asset_dir / "plasticity3d_p4_l1_2_np32_maxit20_convergence.png")
    plot_timing(arr, asset_dir / "plasticity3d_p4_l1_2_np32_maxit20_newton_timing.png")
    plot_linear(
        arr,
        asset_dir / "plasticity3d_p4_l1_2_np32_maxit20_linear_diagnostics.png",
    )
    plot_memory(
        arr,
        asset_dir / "plasticity3d_p4_l1_2_np32_maxit20_memory_profile.png",
    )

    summary = {
        "status": str(obj["status"]),
        "message": str(obj["message"]),
        "nit": int(obj["nit"]),
        "final_grad_norm": float(obj["final_grad_norm"]),
        "energy": float(obj["energy"]),
        "solve_time": float(obj["solve_time"]),
        "total_time": float(obj["total_time"]),
        "linear_iterations_total": int(obj["linear_iterations_total"]),
        "max_rss_gib": float(np.nanmax(arr["rss_current_max"])),
        "max_hwm_gib": float(np.nanmax(arr["rss_hwm_max"])),
        "max_ksp_its": int(np.max(arr["ksp_max"])),
    }
    (asset_dir / "plasticity3d_p4_l1_2_np32_maxit20_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    build_report(obj, history, report_path, asset_dir)


if __name__ == "__main__":
    main()
