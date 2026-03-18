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
        
    ksp_its = [x.get("ksp_its", 0) for x in linear_timing]
    nit = step_data.get("iters", len(linear_timing))
    energy = step_data.get("energy", "N/A")
    
    return {
        "nit": nit,
        "energy": energy,
        "ksp_history": ksp_its
    }

configs_gamg_l4 = [
    ("FEniCS GAMG (L4, 32)", "l4_fenics_exact_gamg.json", True),
    ("New JAX GAMG (L4, 32)", "l4_jax_exact_gamg.json", False),
]

print("== EXACT MATCH `rtol=1e-3, max_it=10000`, LEVEL 4, 32 CORES (CG + GAMG) ==")
print(f"| {'Configuration':<25} | {'Newton Iters':<12} | {'Energy':<15} |")
print("-" * 60)
for name, fname, is_fenics in configs_gamg_l4:
    stats = get_stats(fname, is_fenics)
    if stats:
        energy_str = f"{stats['energy']:.6e}" if isinstance(stats['energy'], float) else str(stats['energy'])
        print(f"| {name:<25} | {stats['nit']:<12} | {energy_str:<15} |")
        print(f"  -> KSP iters per line iter: {stats['ksp_history']}")
