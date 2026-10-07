import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import open_clip
import torch
import numpy as np
import pickle
import faiss

print("Loading CLIP...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32-quickgelu', pretrained='openai'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
model.eval()

# Load saved embeddings (hash-based format)
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

hashes = list(data.keys())
embeddings = np.array([data[h]["embedding"] for h in hashes]).astype('float32')
filenames = [data[h]["filename"] for h in hashes]

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

def search(query_text, top_k=5, min_score=0.2):
    tokens = tokenizer([query_text])
    with torch.no_grad():
        query_vec = model.encode_text(tokens)
        query_vec = query_vec / query_vec.norm(dim=-1, keepdim=True)
    query_vec = query_vec.numpy().astype('float32')

    scores, indices = index.search(query_vec, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if score >= min_score:
            results.append((filenames[idx], score))

    return results

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
