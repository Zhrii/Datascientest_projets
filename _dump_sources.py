import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

for nb_name in ['07_image_exploration_donnees', '08_image_traitement_donnees']:
    path = f'notebooks/{nb_name}.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"\n{'#'*80}")
    print(f"# {nb_name}")
    print(f"{'#'*80}")
    
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        ctype = cell['cell_type']
        print(f"\n--- Cell {i} ({ctype}) ---")
        print(src[:2000])  
        if len(src) > 2000:
            print(f"... [truncated, total {len(src)} chars]")
