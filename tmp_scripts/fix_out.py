import re
with open("out_fix.json", "r") as f:
    text = f.read()

m = re.search(r'(\{.*?\})', text, flags=re.DOTALL)
if m:
    with open("cleaned_out.json", "w") as fw:
        fw.write(m.group(1))
