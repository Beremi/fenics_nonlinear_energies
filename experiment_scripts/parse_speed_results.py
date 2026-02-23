"""Parse and summarize he_speed_custom/snes experiment results."""
import json

with open("experiment_scripts/he_speed_custom_l3np16.json") as f:
    c = json.load(f)

step = c["steps"][0]
print(f"=== Custom Newton (level {c['total_dofs']} DOFs, np=16) ===")
print(f"Time: {step['time']}s | Newton: {step['iters']} | Energy: {step['energy']}")

lt = step["linear_timing"]
n = len(lt)
total_assemble = sum(x["assemble_time"] for x in lt)
total_setop = sum(x["setop_time"] for x in lt)
total_setup = sum(x["pc_setup_time"] for x in lt)
total_solve = sum(x["solve_time"] for x in lt)
total_linear = sum(x["linear_total_time"] for x in lt)
n_setup = sum(1 for x in lt if x["pc_setup_time"] > 0.01)

print(f"Newton steps with timing: {n}")
print(f"PC setups (>10ms): {n_setup}")
print(f"Totals: assemble={total_assemble:.4f}s  setop={total_setop:.4f}s  "
      f"setup={total_setup:.4f}s  solve={total_solve:.4f}s  linear_total={total_linear:.4f}s")
print(f"Total linear % of wall: {total_linear/step['time']*100:.1f}%")
print()
print("Per-Newton breakdown:")
print(f"{'i':>3}  {'assemble':>10}  {'setop':>8}  {'setup':>8}  {'solve':>8}  {'total':>8}")
for i, x in enumerate(lt):
    print(f"{i:>3}  {x['assemble_time']:>10.4f}  {x['setop_time']:>8.4f}  "
          f"{x['pc_setup_time']:>8.4f}  {x['solve_time']:>8.4f}  {x['linear_total_time']:>8.4f}")

h = step.get("history", [])
print(f"\nNewton history ({len(h)} entries):")
for i, r in enumerate(h):
    print(f"  [{i}] {r}")

print()
with open("experiment_scripts/he_speed_snes_l3np16.json") as f:
    s = json.load(f)
step_s = s["steps"][0]
print(f"=== SNES Newton (level {s['total_dofs']} DOFs, np=16) ===")
print(f"Time: {step_s['time']}s | Newton: {step_s['iters']} | "
      f"Linear: {step_s['linear_iters']} | Energy: {step_s['energy']} | reason: {step_s['reason']}")
print(f"Avg KSP/Newton: {step_s['linear_iters']/step_s['iters']:.1f}")
