import json

def get_stats(filename, is_fenics=False):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
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
            
    nit = step_data.get("nit", step_data.get("iters", len(linear_timing)))
    energy = step_data.get("energy", "N/A")
    ksp_its = t.get("ksp_its", "N/A")
    total_ksp_its = sum([x.get("ksp_its", 0) for x in linear_timing])
    
    return {
        "assembly": assembly,
        "pc_setup": pc_setup,
        "solve": solve,
        "nit": nit,
        "total_ksp": total_ksp_its,
        "energy": energy
    }

configs = [
    ("Old JAX (16)", "old_detailed_16.json", False),
    ("New JAX (16)", "new_detailed_16.json", False),
    ("FEniCS (16)", "fenics_detailed_16.json", True),
    ("Old JAX (32)", "old_detailed_32.json", False),
    ("New JAX (32)", "new_detailed_32.json", False),
    ("FEniCS (32)", "fenics_detailed_32.json", True)
]

print(f"| Configuration        | Assy (Iter 1) | PC Setup (1) | Solve (1) | Newton Iters | Energy         |")
print(f"|----------------------|---------------|--------------|-----------|--------------|----------------|")

for name, fname, is_fenics in configs:
    stats = get_stats(fname, is_fenics)
    if stats:
        energy_str = f"{stats['energy']:.4e}" if isinstance(stats['energy'], float) else str(stats['energy'])
        print(f"| {name:<20} | {stats['assembly']:12.3f}s | {stats['pc_setup']:11.3f}s | {stats['solve']:8.3f}s | {stats['nit']:>12} | {energy_str:<14} |")

