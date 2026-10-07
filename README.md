# Multimodal DAM Search Engine

A Digital Asset Management (DAM) search engine that lets users find images using natural language queries or by uploading a similar image. Powered by **CLIP** for joint text-image embeddings, **FAISS** for fast similarity search, and **spaCy** for automatic caption tagging (NER + noun-chunk extraction).

## Features

- **Text-based semantic search** — search assets using natural language, not just exact keywords
- **Search by image** — upload an image to find visually/semantically similar assets
- **Automatic captioning and tagging** — each asset gets a caption (from dataset or auto-generated) and extracted tags via spaCy NER
- **Incremental indexing** — handles added, deleted, and renamed files without a full rebuild
- **New asset upload** — add images directly through the web interface
- **Relevance filtering** — low-similarity results are automatically excluded

## Tech Stack

- **CLIP** (`open_clip_torch`) — joint image-text embeddings, run locally (no API calls)
- **FAISS** — vector similarity search
- **spaCy** — Named Entity Recognition and tagging
- **Streamlit** — web interface
- **PyTorch** — backend for CLIP

## Architecture

```
Assets (images)
      |
      v
CLIP --------------------> embeddings.pkl (FAISS index)
      |
Captions + spaCy NER -----> tags.pkl

Streamlit app queries both at search time
(text query or uploaded image -> CLIP -> FAISS -> ranked results + tags)
```

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

On macOS, if you hit an OpenMP error (`OMP: Error #15`), set:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 4. Add your image dataset

This repo does not include image data. To set it up:

1. Create an `assets/` folder in the project root
2. Add your own images, or download [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) and copy a subset of images + `captions.txt` into the project root

### 5. Build the search index
```bash
python ingest.py
python auto_tag.py
```

### 6. Run the app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

## Project Structure

```
.
├── app.py              # Streamlit web interface
├── ingest.py            # Builds/updates CLIP embeddings (incremental)
├── auto_tag.py           # Generates captions + NER tags per image
├── search.py             # Standalone search script (CLI testing)
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

1. **Ingestion**: Every image is converted into a 512-dimensional vector using CLIP's image encoder, and stored in `embeddings.pkl`, indexed by a content hash (so renames don't trigger re-embedding).
2. **Tagging**: Each image's caption (from the dataset, or generated automatically) is processed with spaCy to extract named entities and noun-chunk tags, stored in `tags.pkl`.
3. **Search**: A text query or uploaded image is converted into the same type of vector using CLIP, then compared against all stored vectors using FAISS (cosine similarity via inner product on normalized vectors). Results below a relevance threshold are filtered out.

## Possible Extensions

- Hybrid search (combine semantic similarity with keyword/tag matching)
- Precision@K evaluation against a labeled test set
- Video support (keyframe extraction + embedding)
- Metadata filters (file type, upload date)

## License

This project is for educational purposes. If using the Flickr8k dataset, refer to its own license terms for the images and captions.
