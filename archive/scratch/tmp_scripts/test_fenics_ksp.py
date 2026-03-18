import json

with open('fenics_gamg_16.json') as f:
    d = json.load(f)
    print("FEniCS GAMG 16 KSP its:")
    print([x.get("ksp_its", None) for x in d["steps"][0]["linear_timing"]])
