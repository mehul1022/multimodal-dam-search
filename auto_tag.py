import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import spacy
import pickle
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("Loading spaCy NER model...")
nlp = spacy.load("en_core_web_sm")

print("Loading BLIP captioning model (fallback for uncaptioned images)...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

assets_folder = "assets"
captions_file = "captions.txt"
tags_file = "tags.pkl"

# Load dataset captions if the file exists
captions_map = {}
if os.path.exists(captions_file):
    with open(captions_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            filename, caption = row[0].strip(), row[1].strip()
            if filename not in captions_map:
                captions_map[filename] = caption
    print(f"Loaded {len(captions_map)} dataset captions.")

# Load existing tags (incremental)
if os.path.exists(tags_file):
    with open(tags_file, "rb") as f:
        tag_data = pickle.load(f)
else:
    tag_data = {}

current_files = [
    f for f in os.listdir(assets_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

for filename in current_files:
    if filename in tag_data:
        continue

    path = os.path.join(assets_folder, filename)

    if filename in captions_map:
        caption = captions_map[filename]
        source = "dataset"
    else:
        try:
            image = Image.open(path).convert("RGB")
            inputs = blip_processor(image, return_tensors="pt")
            out = blip_model.generate(**inputs, max_new_tokens=30)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            source = "blip"
        except Exception as e:
            print(f"  Skipping {filename}: {e}")
            continue

    doc = nlp(caption)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    noun_tags = list(set(chunk.text.lower() for chunk in doc.noun_chunks))

    tag_data[filename] = {
        "caption": caption,
        "source": source,
        "entities": entities,
        "tags": noun_tags
    }

    print(f"  [{source}] {filename}: \"{caption}\" → {noun_tags}")

with open(tags_file, "wb") as f:
    pickle.dump(tag_data, f)

print(f"\nDone. Tagged {len(tag_data)} images total.")