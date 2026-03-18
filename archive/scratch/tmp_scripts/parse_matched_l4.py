import json

def get_stats(filename, is_fenics=False):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return None
    
    step_data = data.get("steps", [{}])[0]
    linear_timing = step_data.get("linear_timing", [])
    
    if not linear_timing:
        return None
        
    t = linear_timing[0]
    
    if is_fenics:
        assembly = t.get("assemble_time", 0)
        pc_setup = t.get("pc_setup_time", 0)
        solve = t.get("solve_time", 0)
    else:
        assembly = t.get("assemble_total_time", 0)
        pc_setup = t.get("pc_setup_time", 0)
        solve = t.get("solve_time", 0)
            
    # Also grab total KSP Iterations
    ksp_its = [x.get("ksp_its", 0) for x in linear_timing]
    
    # Check for empty array to prevent division by zero
    if len(linear_timing) == 0:
        return None
        
    solves_avg = sum(x.get("solve_time", 0) for x in linear_timing)/len(linear_timing)

    nit = step_data.get("nit", step_data.get("iters", len(linear_timing)))
    energy = step_data.get("energy", "N/A")
    
    safe_ksp_len = len(ksp_its) if len(ksp_its) > 0 else 1
    
    return {
        "assembly": assembly,
        "pc_setup": pc_setup,
        "solve_1": solve,
        "solve_avg": solves_avg,
        "ksp_avg": sum(ksp_its)/safe_ksp_len,
        "nit": nit,
        "ksp_history": ksp_its
    }

configs_gamg = [
    ("FEniCS GAMG (32, L4)", "l4_fenics_exact_gamg.json", True),
    ("New JAX GAMG (32, L4)", "l4_jax_exact_gamg.json", False),
]

configs_hypre = [
    ("FEniCS HYPRE (32, L4)", "l4_fenics_exact_hypre.json", True),
    ("New JAX HYPRE (32, L4)", "l4_jax_exact_hypre.json", False),
]

for title, configs in [("== EXACT MATCH `rtol=1e-3, max_it=10000`, LEVEL 4, 32 CORES (CG + GAMG) ==", configs_gamg), ("\n== EXACT MATCH `rtol=1e-3, max_it=10000`, LEVEL 4, 32 CORES (GMRES + HYPRE) ==", configs_hypre)]:
    print(title)
    print(f"| {'Configuration':<25} | {'Assy (1)':<10} | {'PC Set (1)':<10} | {'Solve (1)':<10} | {'Solve/it':<10} | {'Avg KSP / line_iter':<20} |")
    print("-" * 102)
    for name, fname, is_fenics in configs:
        stats = get_stats(fname, is_fenics)
        if stats:
            print(f"| {name:<25} | {stats['assembly']:9.3f}s | {stats['pc_setup']:9.3f}s | {stats['solve_1']:9.3f}s | {stats['solve_avg']:9.3f}s | {stats['ksp_avg']:<20.1f} |")
            print(f"       -> KSP per line iteration: {stats['ksp_history']}")
