"""Fix broken print f-strings in notebook 04 cell 7."""
import json
from pathlib import Path

nb_path = Path("notebooks/04_texte_modelisation_baseline.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src_list = cell.get("source", [])
    src = "".join(src_list)
    if "print_label_stats" not in src:
        continue

    # Fix: the source array has split lines - when joined we get print(f"\n" + newline + "rest"
    # So we have literal \n\n (backslash-n + newline) - replace by merging
    fixes = [
        ('    print(f"\\n\n📊', '    print(f"\\n📊'),
        ('    print(f"\\n\n✅ Superclasse', '    print(f"\\n✅ Superclasse'),
        ('print(f"\\n\n✅ Label encoder', 'print(f"\\n✅ Label encoder'),
        ('print(f"\\n\n✅ Scénarios', 'print(f"\\n✅ Scénarios'),
    ]

    for old, new in fixes:
        if old in src:
            src = src.replace(old, new)
            print(f"Applied fix in cell {i}")

    # Rebuild source as list of lines (each ending with \n except possibly last)
    lines = src.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        cell["source"].append(lines[-1] + "\n")
    break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Done")
