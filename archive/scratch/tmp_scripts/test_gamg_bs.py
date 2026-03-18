file = "run_view.py"
with open(file, 'r') as f:
    text = f.read()
import re
text = re.sub(r'sys\.argv = \[sys\.argv\[0\]\] \+ petsc_options', r'sys.argv = [sys.argv[0]] + petsc_options + ["-ksp_view"]', text)
with open(file, 'w') as f:
    f.write(text)
