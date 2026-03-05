import json

def parse_file(fname):
    try:
        with open(fname) as f:
            d = json.load(f)
        
        if "steps" in d and len(d["steps"]) > 0:
            step = d["steps"][0]
            total_time = step.get("time", 0.0)
            
            nit = step.get("nit")
            if nit is None: nit = step.get("iters", "N/A")
            
            total_ksp = step.get("linear_iters", None)
            if total_ksp is None and "linear_timing" in step:
                total_ksp = sum(t.get("ksp_its", 0) for t in step["linear_timing"])
            
            final_egy = step.get("energy", "N/A")
            msg = step.get("message", "N/A")
            if "maximum number of iterations" in msg.lower() or "max newton" in msg.lower():
                nit = f"{nit} (DNC)"
        else:
            solve_stage = next((s for s in d.get("stages", []) if "solve" in s["name"].lower()), None)
            total_time = solve_stage["duration"] if solve_stage else 0.0
            nit = d.get("nit") or d.get("metrics", {}).get("nit", "N/A")
            ksp_its = d.get("ksp_its") or d.get("metrics", {}).get("ksp_its", [])
            total_ksp = sum(ksp_its)
            final_egy = d.get("final_energy") or d.get("metrics", {}).get("final_energy", "N/A")
            
        return {
            "Time": f"{total_time:.2f}s",
            "Newton Iters": nit,
            "Linear Iters": total_ksp,
            "Final Energy": f"{final_egy:.6e}" if isinstance(final_egy, float) else final_egy
        }
    except Exception as e:
        return {"Error": str(e)}

files = [
    ("FEniCS (1e-1)", "l4_fenics_gamg_1e-1.json"), 
    ("JAX SFD (1e-1)", "l4_jax_gamg_1e-1.json"),
    ("FEniCS (1e-3)", "l4_fenics_exact_gamg.json"), 
    ("JAX SFD (1e-3)", "l4_jax_exact_gamg.json"),
    ("JAX Elem (1e-3)", "l4_jax_element_gamg_1e-3.json"),
    ("FEniCS (1e-6)", "l4_fenics_gamg_1e-6.json"), 
    ("JAX SFD (1e-6)", "l4_jax_gamg_1e-6.json")
]

print(f"{'Implementation':<16} | {'Final Energy':<15} | {'Newton Iter':<17} | {'Total KSP Iter':<15} | {'Solve Time'}")
print("-" * 84)
for label, f in files:
    res = parse_file(f)
    if "Error" in res:
        print(f"{label:<16} | Running... ({f})")
    else:
        print(f"{label:<16} | {res['Final Energy']:<15} | {str(res['Newton Iters']):<17} | {str(res['Linear Iters']):<15} | {res['Time']}")
