import json

obj = json.load(open("experiment_scripts/he_snes_phase2_l1.json"))
runs = obj["runs"]
ok = [r for r in runs if r["status"] == "ok"]
other = [r for r in runs if r["status"] != "ok"]
ok = sorted(ok, key=lambda r: r["wall_time"])

print("OK", len(ok))
for r in ok:
    s = r["result"]["steps"][0]
    print(r["case"]["name"], r["wall_time"], s["energy"], s["iters"], s["reason"])

print("DIV", len(other))
for r in other:
    s = (r.get("result") or {"steps": [{}]})["steps"][0]
    print(r["case"]["name"], r["status"], s.get("reason"), s.get("energy"))
