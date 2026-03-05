import sys, io, pickle, traceback
from pathlib import Path
from PIL import Image
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path('.').resolve()

# Load image_df exactly like NB11
with open(ROOT / 'models' / 'image_df.pkl', 'rb') as f:
    image_df = pickle.load(f)

with open(ROOT / 'models' / 'label_encoder_image.pkl', 'rb') as f:
    le = pickle.load(f)

image_df['label'] = le.transform(image_df['prdtypecode'])

print(f"Columns: {list(image_df.columns)}")
print(f"Shape: {image_df.shape}")

# Check which path column to use
for col in ['image_path_clean', 'image_path']:
    if col in image_df.columns:
        sample = image_df[col].iloc[0]
        print(f"\n{col} sample: {sample}")
        print(f"  is_absolute: {Path(sample).is_absolute()}")

# Try to open an image EXACTLY as NB11 Cell 4 does it
print("\n" + "=" * 60)
print("Simulating NB11 extract_clip_features...")
print("=" * 60)

# Use image_path_clean (what the fixed code uses)
path_col = 'image_path_clean' if 'image_path_clean' in image_df.columns else 'image_path'
paths = image_df[path_col].tolist()[:5]

for i, p in enumerate(paths):
    print(f"\n[{i}] Path: {p}")
    
    # Try direct open (what the old code does)
    try:
        img = Image.open(p).convert('RGB')
        print(f"  Direct open: OK ({img.size})")
    except Exception as e:
        print(f"  Direct open: FAILED - {e}")
    
    # Try with ROOT prefix
    try:
        full = ROOT / p
        img = Image.open(full).convert('RGB')
        print(f"  ROOT/p open: OK ({img.size})")
    except Exception as e:
        print(f"  ROOT/p open: FAILED - {e}")

# Now try with CLIP preprocess
print("\n" + "=" * 60)
print("Testing with CLIP model...")
print("=" * 60)
try:
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    
    p = paths[0]
    full = ROOT / p
    img = Image.open(full).convert('RGB')
    processed = clip_preprocess(img)
    print(f"CLIP preprocess OK: tensor shape = {processed.shape}")
except Exception as e:
    print(f"CLIP test FAILED: {e}")
    traceback.print_exc()
