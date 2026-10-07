import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import open_clip
import torch
import numpy as np
import pickle
import faiss
from PIL import Image

st.title("Multimodal DAM Search Engine")

# Fixed internal threshold — not exposed to the user
MIN_SCORE = 0.2

with st.sidebar:
    st.subheader("Add a new asset")
    new_upload = st.file_uploader(
        "Upload to library", type=["jpg", "jpeg", "png"], key="asset_upload"
    )
    if new_upload:
        save_path = os.path.join("assets", new_upload.name)
        with open(save_path, "wb") as f:
            f.write(new_upload.getbuffer())
        st.success(
            f"Saved {new_upload.name} to assets/. "
            f"Run ingest.py and auto_tag.py, then restart the app, to make it searchable."
        )


@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32-quickgelu', pretrained='openai'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
    model.eval()
    return model, tokenizer, preprocess


@st.cache_resource
def load_index():
    with open("embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    hashes = list(data.keys())
    embeddings = np.array([data[h]["embedding"] for h in hashes]).astype('float32')
    filenames = [data[h]["filename"] for h in hashes]

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, filenames


@st.cache_resource
def load_tags():
    if os.path.exists("tags.pkl"):
        with open("tags.pkl", "rb") as f:
            return pickle.load(f)
    return {}


model, tokenizer, preprocess = load_model()
index, filenames = load_index()
tag_data = load_tags()

query = st.text_input("Search your assets:", "a car")
top_k = st.number_input("Number of results", min_value=1, max_value=100, value=5, step=1)

if st.button("Search"):
    tokens = tokenizer([query])
    with torch.no_grad():
        query_vec = model.encode_text(tokens)
        query_vec = query_vec / query_vec.norm(dim=-1, keepdim=True)
    query_vec = query_vec.numpy().astype('float32')

    scores, indices = index.search(query_vec, top_k)

    results = [
        (idx, score)
        for idx, score in zip(indices[0], scores[0])
        if score >= MIN_SCORE
    ]

    if not results:
        st.warning("No relevant results found for this query.")
    else:
        st.caption(f"Found {len(results)} relevant result(s) for '{query}'")

        cols = st.columns(min(len(results), 5))
        for i, (idx, score) in enumerate(results):
            col = cols[i % 5]
            filename = filenames[idx]
            img_path = os.path.join("assets", filename)

            if not os.path.exists(img_path):
                col.warning(f"Missing: {filename}")
                continue

            col.image(img_path, caption=f"{filename}\nScore: {score:.3f}")

            if filename in tag_data:
                info = tag_data[filename]
                col.caption(f"📝 {info['caption']}")
                if info.get('tags'):
                    col.caption(f"🏷️ {', '.join(info['tags'][:5])}")


st.divider()
st.subheader("Or search by uploading an image")

uploaded_file = st.file_uploader(
    "Upload an image to find similar assets", type=["jpg", "jpeg", "png"], key="image_search"
)

if uploaded_file:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Your uploaded query image", width=200)

    img_tensor = preprocess(query_image).unsqueeze(0)
    with torch.no_grad():
        img_query_vec = model.encode_image(img_tensor)
        img_query_vec = img_query_vec / img_query_vec.norm(dim=-1, keepdim=True)
    img_query_vec = img_query_vec.numpy().astype('float32')

    img_scores, img_indices = index.search(img_query_vec, top_k)

    img_results = [
        (idx, score)
        for idx, score in zip(img_indices[0], img_scores[0])
        if score >= MIN_SCORE
    ]

    if not img_results:
        st.warning("No similar images found.")
    else:
        st.caption(f"Found {len(img_results)} similar image(s)")
        img_cols = st.columns(min(len(img_results), 5))
        for i, (idx, score) in enumerate(img_results):
            col = img_cols[i % 5]
            filename = filenames[idx]
            img_path = os.path.join("assets", filename)

            if not os.path.exists(img_path):
                col.warning(f"Missing: {filename}")
                continue

            col.image(img_path, caption=f"{filename}\nScore: {score:.3f}")

            if filename in tag_data:
                info = tag_data[filename]
                col.caption(f"📝 {info['caption']}")

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
