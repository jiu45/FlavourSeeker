import lancedb
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import ast
import pandas as pd

# --- Configuration ---
DB_PATH = "data/lancedb"
TABLE_NAME = "recipes"

class RecipeSearchEngine:
    def __init__(self):
        self.db = lancedb.connect(DB_PATH)
        self.table = self.db.open_table(TABLE_NAME)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load models - in production, might want to load these once or serve them separately
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def search_by_text(self, query, top_k=5):
        query_vector = self.text_model.encode(query).tolist()
        results = self.table.search(query_vector, vector_column_name="text_vector") \
            .limit(top_k) \
            .to_pandas()
        return results

    def search_by_image(self, image_file, top_k=5):
        image = Image.open(image_file)
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        query_vector = image_features.cpu().numpy()[0].tolist()
        
        results = self.table.search(query_vector, vector_column_name="image_vector") \
            .limit(top_k) \
            .to_pandas()
        return results

    def search_by_ingredients(self, ingredients_list, top_k=5, strict=False):
        """
        Search for recipes based on ingredients.
        
        Args:
            ingredients_list (list or str): List of available ingredients.
            top_k (int): Number of results to return.
            strict (bool): 
                If True (Strict/Subset), returns recipes that require *only* the provided ingredients (Recipe <= User).
                If False (Partial/Any), returns recipes that share *at least one* ingredient, 
                sorted by match count (descending) and then missing count (ascending).
        """
        if isinstance(ingredients_list, str):
            ingredients_list = [i.strip().lower() for i in ingredients_list.split(',')]
        else:
            ingredients_list = [i.strip().lower() for i in ingredients_list]

        user_ingredients = set(ingredients_list)
        
        # 1. Fetch all recipes (optimized: could fetch only needed columns)
        # In a real large DB, we'd use an Inverted Index or FTS for pre-filtering.
        # Since we have ~13k, fetching all ingredients is okay for a demo, 
        # but to be slightly better, we can assume we only want recipes that have *at least one* match if partial.
        # However, for pure python flexibility we iterate.
        # Let's fetch the whole frame for valid logic.
        df = self.table.to_pandas() 

        # 2. Logic
        results = []
        
        for idx, row in df.iterrows():
            # Parse recipe ingredients. They are stored as string representation of list "['a', 'b']"
            try:
                # Safer eval
                recipe_ing_list = ast.literal_eval(row['ingredients'])
                # cleanup recipe ingredients for display/logic
                # recipe_ing_list = ["2 cups milk", "1 egg"]
            except:
                continue
                
            # Fuzzy match logic
            # Check which recipe ingredients are covered by user ingredients
            # We assume user inputs simple terms like "milk", "eggs"
            # Recipe terms are "2 cups milk", "large eggs"
            
            matched_indices = set()
            
            # Check for matches
            # For each recipe ingredient, does it contain ANY user ingredient?
            current_missing = []
            current_matches = 0
            
            for ring in recipe_ing_list:
                ring_lower = ring.lower()
                is_match = False
                for uing in user_ingredients:
                    # check if user ingredient (e.g. 'milk') is in recipe string (e.g. '2 cups milk')
                    # Simple substring match
                    if uing in ring_lower:
                        is_match = True
                        break
                
                if is_match:
                    current_matches += 1
                else:
                    current_missing.append(ring)
            
            match_count = current_matches
            missing_count = len(current_missing)
            
            if strict:
                # Strict: Recipe must be a subset of User (i.e., NO missing ingredients)
                if missing_count == 0 and match_count > 0:
                     results.append({
                        **row, 
                        'match_count': match_count,
                        'missing_count': 0,
                        'missing_ingredients': []
                    })
            else:
                # Partial: Must have at least one match
                if match_count > 0:
                    results.append({
                        **row,
                        'match_count': match_count,
                        'missing_count': missing_count,
                        'missing_ingredients': current_missing
                    })
        
        # 3. Create DataFrame and Sort
        if not results:
            return pd.DataFrame()
            
        results_df = pd.DataFrame(results)
        
        if strict:
            # Sort by match count primarily (maximize usage)
            results_df = results_df.sort_values(by=['match_count'], ascending=[False]).head(top_k)
        else:
            # Sort by match count (desc), then missing count (asc)
            results_df = results_df.sort_values(by=['match_count', 'missing_count'], ascending=[False, True]).head(top_k)
            
        return results_df

