import lancedb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# --- Configuration ---
DB_PATH = "data/lancedb"
TABLE_NAME = "recipes"

def update_embeddings_with_enrichment():
    """
    Re-generate text embeddings after enrichment (tags + visual_description).
    
    This script assumes you already have 'tags' and 'visual_description' fields
    populated in your database, but the text_vector embeddings don't include them yet.
    
    Steps:
    1. Load all recipes from database
    2. Rebuild search_text with tags + visual_description
    3. Generate new text embeddings with MiniLM
    4. Update database with new text_vector
    """
    print("="*60)
    print("Updating Text Embeddings with Enrichment Data")
    print("="*60)
    
    # Connect to database
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)
    
    # Load all recipes
    df = tbl.to_pandas()
    print(f"\n[1/4] Loaded {len(df)} recipes from database")
    
    # Initialize embedding model
    print("[2/4] Loading MiniLM embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Rebuild search_text with enrichment
    print("[3/4] Rebuilding search_text with tags + visual_description...")
    updated_count = 0
    
    for idx, row in df.iterrows():
        # Get tags as string
        tags = row.get('tags', [])
        tags_str = " ".join(tags) if isinstance(tags, list) else ""
        
        # Get visual description
        visual_desc = row.get('visual_description', '')
        
        # Rebuild search_text
        new_search_text = f"{row['title']} {row['ingredients']} {row.get('instructions', '')} {visual_desc} {tags_str}"
        df.at[idx, 'search_text'] = new_search_text
        
        if visual_desc or tags_str:
            updated_count += 1
    
    print(f"  → Updated search_text for {updated_count} enriched recipes")
    
    # Generate new embeddings
    print("[4/4] Generating new text embeddings...")
    text_embeddings = model.encode(df['search_text'].tolist(), show_progress_bar=True)
    df['text_vector'] = text_embeddings.tolist()
    
    # Update database
    print("\n[DB] Updating database with new embeddings...")
    
    # Drop old table and create new one with updated embeddings
    db.drop_table(TABLE_NAME)
    db.create_table(TABLE_NAME, df)
    
    print(f"[DONE] Successfully updated {len(df)} recipes with new embeddings!")
    print(f"       Tags and visual descriptions are now searchable.")
    print("="*60)

if __name__ == "__main__":
    update_embeddings_with_enrichment()
