import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import open_clip
import torch
from PIL import Image
import numpy as np
import pickle
import hashlib

def file_hash(path):
    """Compute MD5 hash of file content."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

print("Loading CLIP...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32-quickgelu', pretrained='openai'
)
model.eval()

assets_folder = "assets"
embeddings_file = "embeddings.pkl"

# Map current files to their hashes
current_files = [
    f for f in os.listdir(assets_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]
current_hash_to_filename = {file_hash(os.path.join(assets_folder, f)): f for f in current_files}

# Load existing data: now keyed by hash, not filename
if os.path.exists(embeddings_file):
    with open(embeddings_file, "rb") as f:
        data = pickle.load(f)
    existing = data  # dict: hash -> {"embedding": ..., "filename": ...}
    print(f"Loaded {len(existing)} existing embeddings.")
else:
    existing = {}
    print("No existing embeddings found. Starting fresh.")

existing_hashes = set(existing.keys())
current_hashes = set(current_hash_to_filename.keys())

new_hashes = current_hashes - existing_hashes
deleted_hashes = existing_hashes - current_hashes
renamed_count = 0

# Remove truly deleted images
for h in deleted_hashes:
    print(f"  Removed: {existing[h]['filename']}")
    del existing[h]

# For hashes that already exist, check if the filename changed (rename detection)
for h in current_hashes & existing_hashes:
    old_filename = existing[h]["filename"]
    new_filename = current_hash_to_filename[h]
    if old_filename != new_filename:
        print(f"  Renamed: {old_filename} → {new_filename} (embedding reused, no recompute)")
        existing[h]["filename"] = new_filename
        renamed_count += 1

# Embed only genuinely new images
for h in new_hashes:
    filename = current_hash_to_filename[h]
    path = os.path.join(assets_folder, filename)
    try:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    except Exception as e:
        print(f"  Skipping {filename}: {e}")
        continue
    with torch.no_grad():
        features = model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)
    existing[h] = {"embedding": features.numpy().flatten(), "filename": filename}
    print(f"  Embedded: {filename}")

with open(embeddings_file, "wb") as f:
    pickle.dump(existing, f)

print(f"\nDone. Total: {len(existing)} | New: {len(new_hashes)} | Deleted: {len(deleted_hashes)} | Renamed: {renamed_count}")

def keyword_search(query, tag_data, top_k=5):
    query_words = set(query.lower().split())
    matches = []
    for filename, info in tag_data.items():
        caption_words = set(info['caption'].lower().split())
        overlap = query_words & caption_words
        if overlap:
            matches.append((filename, len(overlap)))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_k]
