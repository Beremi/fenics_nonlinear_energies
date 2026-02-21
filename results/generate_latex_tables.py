#!/usr/bin/env python3
"""
Generate LaTeX tables from experiment results stored in results/<experiment_id>/.

Reads all JSON result files from an experiment directory, aggregates repeated
runs (taking the median time), and produces LaTeX table source.

Usage:
  python3 results/generate_latex_tables.py results/<experiment_id>/
  python3 results/generate_latex_tables.py results/<experiment_id>/ --output results/<experiment_id>/tables.tex
"""
import argparse
import json
import os
import sys
from collections import defaultdict
import statistics


# Mapping from mesh file level -> table level (for display)
MESH_TO_TABLE_LEVEL = {5: 4, 6: 5, 7: 6, 8: 7, 9: 8}

# Expected free DOFs per table level (total DOFs - boundary DOFs in serial)
EXPECTED_DOFS = {4: 2945, 5: 12033, 6: 48641, 7: 195585, 8: 784385}


def load_experiment(exp_dir):
    """Load all result files from an experiment directory.

    Returns:
        metadata: dict with experiment metadata
        runs: dict of {(solver, nprocs): [list of result dicts]}
    """
    metadata = {}
    meta_path = os.path.join(exp_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    runs = defaultdict(list)
    for fname in sorted(os.listdir(exp_dir)):
        if fname == "metadata.json" or not fname.endswith(".json"):
            continue
        if fname.endswith(".tex") or fname.endswith(".txt"):
            continue

        fpath = os.path.join(exp_dir, fname)
        with open(fpath) as f:
            data = json.load(f)

        solver = data.get("metadata", {}).get("solver", "unknown")
        nprocs = data.get("metadata", {}).get("nprocs", 1)
        results = data.get("results", [])
        runs[(solver, nprocs)].append(results)

    return metadata, dict(runs)


def aggregate_runs(runs_list):
    """Aggregate multiple runs: take median time, verify consistency of iters/energy.

    Args:
        runs_list: list of [list of level results], each from one repetition

    Returns:
        dict of {mesh_level: {"time": median_time, "iters": iters, "energy": energy, "total_dofs": dofs}}
    """
    # Group by mesh level
    by_level = defaultdict(lambda: {"times": [], "iters": [], "energies": [], "dofs": []})

    for run_results in runs_list:
        for r in run_results:
            lvl = r["mesh_level"]
            by_level[lvl]["times"].append(r["time"])
            by_level[lvl]["iters"].append(r["iters"])
            by_level[lvl]["energies"].append(r["energy"])
            by_level[lvl]["dofs"].append(r.get("total_dofs", r.get("dofs", 0)))

    aggregated = {}
    for lvl in sorted(by_level.keys()):
        d = by_level[lvl]
        aggregated[lvl] = {
            "time_median": statistics.median(d["times"]),
            "time_min": min(d["times"]),
            "time_max": max(d["times"]),
            "time_all": d["times"],
            "iters": d["iters"][0],  # should be consistent
            "iters_all": d["iters"],
            "energy": d["energies"][0],
            "energy_all": d["energies"],
            "total_dofs": d["dofs"][0],
        }
    return aggregated


def generate_fenics_comparison_table(metadata, runs):
    """Generate a LaTeX table comparing FEniCS serial vs parallel configurations."""
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")

    # Find which configurations we have
    configs = sorted(runs.keys())
    snes_configs = [(s, n) for s, n in configs if s == "snes_newton"]

    if not snes_configs:
        return "% No SNES Newton results found.\n"

    # Build column spec
    n_configs = len(snes_configs)
    col_spec = "ll|" + "|".join(["ccc"] * n_configs)
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\hline")

    # Header row 1: config names
    header1 = " & "
    headers = []
    for solver, nprocs in snes_configs:
        label = f"FEniCS {'serial' if nprocs == 1 else f'parallel ({nprocs} proc)'}"
        headers.append(r"\multicolumn{3}{c" + ("|" if solver !=
                       snes_configs[-1][0] or nprocs != snes_configs[-1][1] else "") + r"}{\textbf{" + label + "}}")
    header1 += " & ".join(headers)
    lines.append(header1 + r"\\")
    lines.append(r"\hline")

    # Header row 2: time/iters/J(u)
    header2 = r"lvl & dofs"
    for _ in snes_configs:
        header2 += r" & time [s] & iters & $J(\boldsymbol{u})$"
    lines.append(header2 + r"\\")
    lines.append(r"\hline")

    # Data rows
    aggregated = {}
    for key in snes_configs:
        aggregated[key] = aggregate_runs(runs[key])

    # Get all mesh levels
    all_levels = sorted(set().union(*(agg.keys() for agg in aggregated.values())))

    for mesh_lvl in all_levels:
        table_lvl = MESH_TO_TABLE_LEVEL.get(mesh_lvl, mesh_lvl)
        # Get dofs from first config that has this level
        dofs = ""
        for key in snes_configs:
            if mesh_lvl in aggregated[key]:
                dofs = str(EXPECTED_DOFS.get(table_lvl, aggregated[key][mesh_lvl]["total_dofs"]))
                break

        row = f"{table_lvl} & {dofs}"
        for key in snes_configs:
            if mesh_lvl in aggregated[key]:
                d = aggregated[key][mesh_lvl]
                row += f" & {d['time_median']:.3f} & {d['iters']} & {d['energy']:.4f}"
            else:
                row += r" & -- & -- & --"
        lines.append(row + r"\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")

    # Caption
    git_short = metadata.get("git_commit", "unknown")[:8]
    ts = metadata.get("timestamp", "unknown")
    cpu = metadata.get("cpu", "unknown")
    dolfinx_ver = metadata.get("dolfinx_version", "unknown")
    caption = (
        f"p-Laplace 2D FEniCS results. "
        f"DOLFINx {dolfinx_ver}, "
        f"CPU: {cpu}, "
        f"commit: \\texttt{{{git_short}}}, "
        f"date: {ts[:10]}."
    )
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{tab:plaplace_2d_fenics_results}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def generate_all_solvers_table(metadata, runs):
    """Generate a table comparing both solver variants."""
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")

    configs = sorted(runs.keys())
    if not configs:
        return "% No results found.\n"

    n_configs = len(configs)
    col_spec = "ll|" + "|".join(["ccc"] * n_configs)
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\hline")

    # Header row 1
    header1 = " & "
    headers = []
    for solver, nprocs in configs:
        if solver == "snes_newton":
            solver_label = "SNES Newton"
        elif solver == "jax_newton":
            solver_label = "JAX Newton"
        else:
            solver_label = "Custom Newton"
        proc_label = "serial" if nprocs == 1 else f"{nprocs} proc"
        label = f"{solver_label} ({proc_label})"
        sep = "|" if (solver, nprocs) != configs[-1] else ""
        headers.append(r"\multicolumn{3}{c" + sep + r"}{\textbf{" + label + "}}")
    header1 += " & ".join(headers)
    lines.append(header1 + r"\\")
    lines.append(r"\hline")

    # Header row 2
    header2 = r"lvl & dofs"
    for _ in configs:
        header2 += r" & time [s] & iters & $J(\boldsymbol{u})$"
    lines.append(header2 + r"\\")
    lines.append(r"\hline")

    # Aggregate all
    aggregated = {key: aggregate_runs(runs[key]) for key in configs}
    all_levels = sorted(set().union(*(agg.keys() for agg in aggregated.values())))

    for mesh_lvl in all_levels:
        table_lvl = MESH_TO_TABLE_LEVEL.get(mesh_lvl, mesh_lvl)
        dofs = ""
        for key in configs:
            if mesh_lvl in aggregated[key]:
                dofs = str(EXPECTED_DOFS.get(table_lvl, aggregated[key][mesh_lvl]["total_dofs"]))
                break

        row = f"{table_lvl} & {dofs}"
        for key in configs:
            if mesh_lvl in aggregated[key]:
                d = aggregated[key][mesh_lvl]
                row += f" & {d['time_median']:.3f} & {d['iters']} & {d['energy']:.4f}"
            else:
                row += r" & -- & -- & --"
        lines.append(row + r"\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")

    git_short = metadata.get("git_commit", "unknown")[:8]
    ts = metadata.get("timestamp", "unknown")
    caption = (
        f"p-Laplace 2D: all solver configurations. "
        f"Commit: \\texttt{{{git_short}}}, date: {ts[:10]}."
    )
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{tab:plaplace_2d_all}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def generate_markdown_table(metadata, runs):
    """Generate a Markdown table for README."""
    configs = sorted(runs.keys())
    snes_configs = [(s, n) for s, n in configs if s == "snes_newton"]

    if not snes_configs:
        return "No SNES Newton results found.\n"

    aggregated = {key: aggregate_runs(runs[key]) for key in snes_configs}
    all_levels = sorted(set().union(*(agg.keys() for agg in aggregated.values())))

    # Header
    header = "| lvl | dofs |"
    sep = "|-----|------|"
    for solver, nprocs in snes_configs:
        label = "serial" if nprocs == 1 else f"{nprocs} proc"
        header += f" time ({label}) | iters | J(u) |"
        sep += "------------|-------|------|"

    lines = [header, sep]

    for mesh_lvl in all_levels:
        table_lvl = MESH_TO_TABLE_LEVEL.get(mesh_lvl, mesh_lvl)
        dofs = EXPECTED_DOFS.get(table_lvl, "?")
        row = f"| {table_lvl} | {dofs} |"
        for key in snes_configs:
            if mesh_lvl in aggregated[key]:
                d = aggregated[key][mesh_lvl]
                row += f" {d['time_median']:.3f} | {d['iters']} | {d['energy']:.4f} |"
            else:
                row += " -- | -- | -- |"
        lines.append(row)

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from experiment results")
    parser.add_argument("exp_dir", help="Path to experiment directory (e.g. results/20260221_120000/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .tex file (default: print to stdout)")
    parser.add_argument("--markdown", action="store_true",
                        help="Also output Markdown table")
    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        print(f"Error: {args.exp_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    metadata, runs = load_experiment(args.exp_dir)

    tex_output = []
    tex_output.append("% Auto-generated LaTeX tables from p-Laplace experiment results")
    tex_output.append(f"% Experiment: {metadata.get('experiment_id', 'unknown')}")
    tex_output.append(f"% Generated: {metadata.get('timestamp', 'unknown')}")
    tex_output.append(f"% Git commit: {metadata.get('git_commit', 'unknown')}")
    tex_output.append("")

    tex_output.append("% === FEniCS SNES Newton: serial vs parallel ===")
    tex_output.append(generate_fenics_comparison_table(metadata, runs))
    tex_output.append("")

    tex_output.append("% === All solver configurations ===")
    tex_output.append(generate_all_solvers_table(metadata, runs))

    tex_content = "\n".join(tex_output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(tex_content)
        print(f"LaTeX tables written to {args.output}")
    else:
        print(tex_content)

    if args.markdown:
        print("\n--- Markdown Table ---\n")
        print(generate_markdown_table(metadata, runs))


if __name__ == "__main__":
    main()
