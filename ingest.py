import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import shutil

# --- Configuration ---
DATA_PATH = "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
IMAGES_DIR = "Food Images"
DB_PATH = "data/lancedb"
TABLE_NAME = "recipes"

# --- Models ---
print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_name):
    image_path = os.path.join(IMAGES_DIR, image_name)
    if not os.path.exists(image_path):
        # Gracefully handle missing images by returning a zero vector or None
        # For simplicity in this demo, we'll just print a warning and return a zero vector
        print(f"Warning: Image not found: {image_name}")
        return [0.0] * 512 
    
    try:
        image = Image.open(image_path)
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        # Normalize and convert to list
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0].tolist()
    except Exception as e:
        print(f"Error processing image {image_name}: {e}")
        return [0.0] * 512

def process_data():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH) # Reset DB for fresh ingestion

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Cleaning
    initial_len = len(df)
    df = df.dropna(subset=['Title', 'Instructions'])
    print(f"Dropped {initial_len - len(df)} rows with missing Title or Instructions.")
    
    # Reset index to create integer IDs
    df = df.reset_index(drop=True)
    df['id'] = df.index

    # Connect to LanceDB
    db = lancedb.connect(DB_PATH)
    
    data = []
    print(f"Processing {len(df)} recipes...")
    
    # Batch processing could be done here for speed, but row-by-row is clearer for this requirement
    for index, row in df.iterrows():
        # Text Embedding
        combined_text = f"{row['Title']} {row['Ingredients']} {row['Instructions']}"
        text_embedding = text_model.encode(combined_text).tolist()
        
        # Image Embedding
        image_name = row['Image_Name']
        # Handle case where Image_Name might be NaN
        if pd.isna(image_name):
             image_name = "placeholder.jpg" # Should ideally handle this better, but consistent with 'missing file' check
        
        image_embedding = get_image_embedding(image_name)
        
        data.append({
            "id": index,
            "title": row['Title'],
            "ingredients": row['Ingredients'],
            "instructions": row['Instructions'],
            "image_name": str(image_name),
            "text_vector": text_embedding,
            "image_vector": image_embedding
        })
        
        if index > 0 and index % 100 == 0:
            print(f"Processed {index} records...")

    print("Creating LanceDB table...")
    # Create table with data directly to infer schema
    tbl = db.create_table(TABLE_NAME, data=data)
    print("Ingestion complete!")

if __name__ == "__main__":
    process_data()
