"""
Resumable LLM Enrichment Script

This script enriches recipe records with visual descriptions and tags.
It is RESUMABLE - it only processes recipes that haven't been enriched yet.

Usage:
    python enrich_recipes.py           # Enrich all remaining recipes
    python enrich_recipes.py --limit 50  # Enrich only 50 recipes per run

Run this script multiple times to gradually enrich the entire dataset,
especially when dealing with API rate limits.
"""

import argparse
import lancedb
from llm_enrichment import batch_enrich_recipes

# --- Configuration ---
DB_PATH = "data/lancedb"
TABLE_NAME = "recipes"
IMAGES_DIR = "Food Images"


def get_unenriched_recipes(limit: int = None) -> list[dict]:
    """
    Query the database for recipes that haven't been enriched yet.
    
    A recipe is considered "unenriched" if visual_description is empty.
    
    Args:
        limit: Maximum number of recipes to return (None for all)
        
    Returns:
        List of recipe dicts with id, title, ingredients, image_name
    """
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)
    
    # Query for recipes with empty visual_description
    # Using SQL filter to find empty strings
    df = tbl.to_pandas()
    
    # Filter for unenriched recipes (empty visual_description)
    unenriched = df[df['visual_description'] == '']
    
    if limit:
        unenriched = unenriched.head(limit)
    
    # Convert to list of dicts
    recipes = []
    for _, row in unenriched.iterrows():
        recipes.append({
            "id": int(row['id']),
            "title": row['title'],
            "ingredients": row['ingredients'],
            "image_name": row['image_name']
        })
    
    return recipes


def update_enriched_recipes(enriched_recipes: list[dict]):
    """
    Update the database with enriched data.
    
    Args:
        enriched_recipes: List of recipe dicts with visual_description and tags
    """
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)
    
    print(f"\n[DB] Updating {len(enriched_recipes)} records...")
    
    for i, enriched in enumerate(enriched_recipes):
        try:
            tbl.update(
                where=f"id = {enriched['id']}",
                values={
                    "visual_description": enriched.get("visual_description", ""),
                    "tags": enriched.get("tags", [])
                }
            )
        except Exception as e:
            print(f"  [WARN] Failed to update recipe {enriched['id']}: {e}")
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Updated {i + 1}/{len(enriched_recipes)} records...")
    
    print(f"[DONE] Database updated successfully!")


def get_enrichment_progress() -> dict:
    """
    Get statistics on enrichment progress.
    
    Returns:
        dict with total, enriched, remaining counts
    """
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)
    df = tbl.to_pandas()
    
    total = len(df)
    
    # Check if visual_description column exists
    if 'visual_description' not in df.columns:
        print("[WARN] Database schema is outdated. visual_description column not found.")
        print("       Please re-run 'python ingest.py' to update the schema.")
        return {
            "total": total,
            "enriched": 0,
            "remaining": total,
            "percent_complete": 0,
            "schema_outdated": True
        }
    
    enriched = len(df[df['visual_description'] != ''])
    remaining = total - enriched
    
    return {
        "total": total,
        "enriched": enriched,
        "remaining": remaining,
        "percent_complete": round(enriched / total * 100, 1) if total > 0 else 0,
        "schema_outdated": False
    }


def main():
    parser = argparse.ArgumentParser(
        description="Resumable LLM enrichment for recipe database"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Maximum number of recipes to enrich per run (default: all remaining)"
    )
    parser.add_argument(
        "--status", 
        action="store_true",
        help="Show enrichment progress and exit"
    )
    args = parser.parse_args()
    
    # Show progress
    progress = get_enrichment_progress()
    print("\n" + "="*50)
    print("Recipe Enrichment Progress")
    print("="*50)
    print(f"  Total recipes:    {progress['total']}")
    print(f"  Enriched:         {progress['enriched']}")
    print(f"  Remaining:        {progress['remaining']}")
    print(f"  Progress:         {progress['percent_complete']}%")
    print("="*50)
    
    if args.status:
        return
    
    if progress['remaining'] == 0:
        print("\n[DONE] All recipes have been enriched!")
        return
    
    # Get unenriched recipes
    recipes = get_unenriched_recipes(limit=args.limit)
    
    if not recipes:
        print("\n[DONE] No recipes to enrich.")
        return
    
    print(f"\nStarting enrichment for {len(recipes)} recipes...")
    
    try:
        # Run LLM enrichment
        enriched = batch_enrich_recipes(recipes, IMAGES_DIR)
        
        # Update database
        update_enriched_recipes(enriched)
        
        # Show updated progress
        progress = get_enrichment_progress()
        print("\n" + "="*50)
        print("Updated Progress")
        print("="*50)
        print(f"  Enriched:   {progress['enriched']}/{progress['total']}")
        print(f"  Remaining:  {progress['remaining']}")
        print(f"  Progress:   {progress['percent_complete']}%")
        print("="*50)
        
        if progress['remaining'] > 0:
            print(f"\nRun this script again to continue enriching remaining {progress['remaining']} recipes.")
        else:
            print("\n[DONE] All recipes have been enriched!")
            
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Progress has been saved. Run again to continue.")
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            print(f"\n[RATE LIMITED] {e}")
            print("Progress has been saved. Wait and run again to continue.")
        else:
            print(f"\n[ERROR] {e}")
            print("Progress has been saved up to the last successful batch.")


if __name__ == "__main__":
    main()
