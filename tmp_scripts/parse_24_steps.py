import json
import sys

def parse_24_steps(fname):
    with open(fname) as f:
        d = json.load(f)
    if "steps" not in d:
        print("Data format not supported or no steps found.")
        sys.exit(0)
    
    steps = d["steps"]
    
    print(f"Total DOFs: {d.get('total_dofs')}")
    print("\n| Step | Angle (rad) | Newton Iters | Total KSP Iters | Time/Newton (s) | Step Time (s) | Final Energy |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    total_time = 0.0
    total_nit = 0
    total_ksp = 0
    
    for s in steps:
        step_idx = s["step"]
        angle = s.get("angle", 0.0)
        time_s = s.get("time", 0.0)
        
        nit = s.get("nit", 0)
        
        lin_iters = s.get("linear_iters", 0)
        if lin_iters == 0 and "linear_timing" in s:
            lin_iters = sum(t.get("ksp_its", 0) for t in s["linear_timing"])
            
        egy = s.get("energy", 0.0)
        
        time_per_nit = time_s / nit if nit > 0 else 0.0
        
        total_time += time_s
        total_nit += nit
        total_ksp += lin_iters
        
        # If it retried
        attempt_mark = "*" if s.get("attempt", "primary") != "primary" else ""
        
        print(f"| {step_idx:2d}{attempt_mark} |  {angle:8.4f} | {nit:12d} | {lin_iters:15d} | {time_per_nit:15.3f} | {time_s:13.3f} | {egy:12.6f} |")
        
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    avg_ksp_per_nit = total_ksp / total_nit if total_nit > 0 else 0.0
    print(f"| **SUM** | | **{total_nit}** | **{total_ksp}** | | **{total_time:.3f} s** | |")

if __name__ == "__main__":
    import os
    if os.path.exists("l4_jax_element_gamg_1e-3_24steps_total24.json"):
        parse_24_steps("l4_jax_element_gamg_1e-3_24steps_total24.json")
