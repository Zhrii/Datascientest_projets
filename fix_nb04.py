# -*- coding: utf-8 -*-
import json

path = 'notebooks/04_texte_modelisation_baseline.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c.get('source', []))
    if 'Preparation des labels' in src or 'Superclasse' in src:
        fixes = [
            ('print(f"\\n",\n"Preparation des labels', 'print(f"\\nPreparation des labels'),
            ('print(f"\\n",\n"Superclasse chargee', 'print(f"\\nSuperclasse chargee'),
            ('print(f"\\n",\n"Label encoder sauvegarde', 'print(f"\\nLabel encoder sauvegarde'),
            ('print(f"\\n",\n"Scenarios a tester', 'print(f"\\nScenarios a tester'),
        ]
        for old, new in fixes:
            src = src.replace(old, new)
        lines = src.split('\n')
        c['source'] = [line + '\n' for line in lines[:-1]]
        if lines[-1]:
            c['source'].append(lines[-1] + '\n')
        print('Fixed cell', i)
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print('Done')
