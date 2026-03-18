import json

with open("tmp_work/jax_sfd_n32.log") as f:
    text = f.read()

import re
m = re.search(r'\{.*"level".*\}', text, re.DOTALL)
if m:
    j = json.loads(m.group(0))
    s = j['steps'][0]
    hist = s.get('linear_timing', [])
    times = [h.get('assemble_coo_assembly', 0) for h in hist]
    print(f"COO times per step (first 5): {times[:5]}")
