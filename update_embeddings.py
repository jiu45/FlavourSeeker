import lancedb
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DB_PATH = "data/lancedb"
TABLE_NAME = "recipes"

def update_search_text_with_enrichment():
    """
    Update search_text to include tags and visual_description.
    
    LanceDB will automatically re-embed the updated search_text into text_vector
    thanks to the embedding registry defined in ingest.py.
    
    This script assumes you already have 'tags' and 'visual_description' fields
    populated in your database.
    """
    print("="*60)
    print("Updating search_text with Enrichment Data")
    print("="*60)
    
    # Connect to database
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)
    
    print(f"\n[1/2] Processing {tbl.count_rows()} recipes...")
    
    updated_count = 0
    
    # Iterate through all recipes
    for recipe in tbl.to_pandas().itertuples():
        try:
            # Get tags as string
            tags = recipe.tags if hasattr(recipe, 'tags') else []
            tags_str = " ".join(tags) if isinstance(tags, list) else ""
            
            # Get visual description
            visual_desc = recipe.visual_description if hasattr(recipe, 'visual_description') else ""
            
            # Only update if we have enrichment data
            if visual_desc or tags_str:
                # Rebuild search_text
                new_search_text = f"{recipe.title} {recipe.ingredients} {recipe.instructions} {visual_desc} {tags_str}"
                
                # Update the record
                tbl.update(
                    where=f"id = {recipe.id}",
                    values={"search_text": new_search_text}
                )
                
                updated_count += 1
                
                if updated_count % 10 == 0:
                    print(f"  Updated {updated_count} recipes...")
        
        except Exception as e:
            print(f"  [WARN] Failed to update recipe {recipe.id}: {e}")
    
    print(f"\n[2/2] Updated {updated_count} recipes with enrichment data")
    print(f"\n[DONE] LanceDB will auto-embed the updated search_text!")
    print(f"       Tags and visual descriptions are now searchable.")
    print("="*60)

if __name__ == "__main__":
    update_search_text_with_enrichment()
