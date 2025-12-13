from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from search_engine import RecipeSearchEngine
import os

# --- Configuration ---
# Set GROQ_API_KEY environment variable before running
GROQ_API_KEY = os.environ.get(\"GROQ_API_KEY\")

model = GroqModel('llama-3.3-70b-versatile')
engine = RecipeSearchEngine()

class RecipeSearchTools:
    """Tools for the PydanticAI agent"""

    def __init__(self, engine: RecipeSearchEngine):
        self.engine = engine

    def search_recipes(self, ctx: RunContext, query: str, filter_str: str = None) -> str:
        """
        Search for recipes by text with optional filtering.
        
        Args:
            query: The search keywords (e.g., "chicken", "comfort food").
            filter_str: SQL-like where clause (e.g., "ingredients NOT LIKE '%beef%'").
        """
        try:
            results = self.engine.search_by_text(query, top_k=5, where=filter_str)
            if results.empty:
                return "No recipes found matching your query."

            response_parts = []
            response_parts.append(f"Here are {len(results)} recipes that match your query:\\n")

            for _, row in results.iterrows():
                response_parts.append(f"## {row['title']}")
                
                # Ingredients
                ingredients = row['ingredients']
                response_parts.append(f"**Ingredients:** {ingredients}")
                
                response_parts.append("---\\n")

            return "\\n".join(response_parts)

        except Exception as e:
            return f"Error searching recipes: {str(e)}"

# Initialize tools
tools_instance = RecipeSearchTools(engine)

# --- Comprehensive System Prompt (8 Rules) ---
SYSTEM_PROMPT = """You are the "Culinary Compass AI", a specialized recipe assistant.
Your goal is to help users find recipes using the database tool.

**RULES - FOLLOW EXACTLY:**

1. **ALWAYS use the `search_recipes` tool**: Never invent or generate recipes manually. Only answer based on tool results.

2. **Extract Keywords**: If the user talks about feelings ("I am sad"), ignore the emotion and extract the food intent (e.g., "comfort food") to search.

3. **Recipe Modification**: If a user asks for "Lamb Pho" but only "Beef Pho" exists:
   - Search for "Pho".
   - Return the closest standard recipe (Beef Pho).
   - Explain how to modify it (e.g., "Use lamb instead of beef").

4. **Multiple Dishes**: If user asks for "Chicken and Cake":
   - Call the search tool for "Chicken".
   - Call the search tool for "Cake".
   - Combine and present both results.

5. **No Independent Searching**: Do NOT combine unrelated terms (e.g., don't search "Chicken Cake").

6. **Clarification**: If query is too vague (e.g., "Tortilla"), ask 1-2 clarifying questions (e.g., "Spanish or Mexican?") before searching.

7. **Filtering (CRITICAL FORMAT)**:
   - Parse constraints like "no beef", "without salt", "don't want salty" into SQL-like filters.
   - **ONLY use this exact format**: `ingredients NOT LIKE '%[ingredient]%'`
   - Map adjectives: "salty" → "salt", "sweet" → "sugar", "spicy" → "chili"
   - Example: "no beef" → filter_str = "ingredients NOT LIKE '%beef%'"
   - Example: "don't want salty" → filter_str = "ingredients NOT LIKE '%salt%'"
   - **DO NOT use columns like sodium_content, calories, etc. - they do not exist!**
   - **ONLY valid columns: id, title, ingredients, instructions**

8. **Response Style**: Be helpful, concise, and appetizing.

**TOOL CALL FORMAT**:
- Do NOT use XML tags like `<function=...>`.
- Do NOT use Python code blocks.
- Just call the tool with valid JSON arguments."""

agent = Agent(
    model,
    tools=[tools_instance.search_recipes],
    system_prompt=SYSTEM_PROMPT,
    retries=1
)

import re
import json

# --- Ingredient Mapping for Exclusion Parsing ---
INGREDIENT_MAPPING = {
    'salty': 'salt',
    'sweet': 'sugar',
    'spicy': 'chili',
    'fatty': 'fat',
    'meaty': 'meat',
    'creamy': 'cream',
    'buttery': 'butter',
    'oily': 'oil',
    'cheesy': 'cheese',
}

def parse_user_exclusions(user_message: str) -> str:
    """
    Parse user message for exclusion patterns and build valid LanceDB filter.
    
    Patterns matched:
    - "no [X]", "without [X]", "don't want [X]", "exclude [X]"
    - Maps adjectives like "salty" -> "salt"
    
    Returns:
        A valid filter string like "ingredients NOT LIKE '%salt%'" or None.
    """
    text_lower = user_message.lower()
    exclusions = []
    
    # Pattern: "no [X]", "without [X]", "don't want [something] [X]", "exclude [X]"
    # Examples: "no beef", "without salt", "don't want something salty", "exclude cheese"
    patterns = [
        r"(?:no|without|exclude)\s+(?:something\s+)?(\w+)",
        r"don'?t\s+want\s+(?:something\s+)?(\w+)",
        r"i\s+don'?t\s+like\s+(\w+)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Skip common stop words that aren't ingredients
            if match in ['the', 'a', 'an', 'any', 'it', 'that', 'this', 'meal', 'dish', 'food', 'recipe']:
                continue
            # Map adjectives to ingredients
            ingredient = INGREDIENT_MAPPING.get(match, match)
            if ingredient not in exclusions:
                exclusions.append(ingredient)
    
    if not exclusions:
        return None
    
    # Build filter string
    filters = [f"ingredients NOT LIKE '%{exc}%'" for exc in exclusions]
    return " AND ".join(filters)

def chat_with_agent(user_message: str, history: list):
    """
    Wrapper to run the agent.
    
    WORKAROUND: Llama-3.3 on Groq often returns invalid XML/Pseudo-code 
    instead of real tool calls (e.g., `<function=...>`).
    
    This wrapper intercepts those strings, parses them, executes the tool manualy,
    and returns the result, effectively fixing the model's behavior transparently.
    """
    try:
        # PydanticAI run_sync
        result = agent.run_sync(user_message)
        output = result.output
        
        # --- Interceptor Logic ---
        # Detects: <function=search_recipes...> or {function=search_recipes}...
        if "function=search_recipes" in output or "search_recipes(" in output:
            print(f"Intercepted Invalid Tool Call: {output}")
            
            # 1. Extract Query
            # Look for "query": "something" OR query="something"
            query_match = re.search(r'query"?\s*[:=]\s*["\'](.*?)["\']', output, re.IGNORECASE)
            
            # 2. Extract Filter
            # Look for "filter_str": "something" OR filter_str="something"
            filter_match = re.search(r'filter_str"?\s*[:=]\s*["\'](.*?)["\']', output, re.IGNORECASE)
            
            if query_match:
                query_val = query_match.group(1)
                filter_val = filter_match.group(1) if filter_match else None
                
                # Validate Filter - Only allow known columns
                VALID_FILTER_COLUMNS = ['id', 'title', 'ingredients', 'instructions', 'image_name', 'search_text']
                if filter_val:
                    has_valid_column = any(col in filter_val.lower() for col in VALID_FILTER_COLUMNS)
                    if not has_valid_column:
                        print(f"Invalid LLM filter '{filter_val}' - using client-side parser.")
                        # FALLBACK: Parse original user message for exclusions
                        filter_val = parse_user_exclusions(user_message)
                
                # If still no filter, try parsing user message anyway
                if not filter_val:
                    filter_val = parse_user_exclusions(user_message)
                
                print(f"Manual Execution -> Query: {query_val}, Filter: {filter_val}")
                
                # Manual Tool Execution
                tool_result = tools_instance.search_recipes(None, query_val, filter_val)
                return tool_result
        
        # Return normal output if no interception needed
        return output

    except Exception as e:
        error_str = str(e)
        
        # --- Fallback: Parse the error message itself ---
        # If PydanticAI throws a 400 error, the failed_generation is in the message.
        # e.g., "body: {'error': {..., 'failed_generation': '<function=search_recipes,{"query": "..."}>'...}}"
        
        if "failed_generation" in error_str or "function=search_recipes" in error_str:
            print(f"Caught 400 Error, Attempting Recovery: {error_str[:200]}...")
            
            # 1. Extract Query from error message
            query_match = re.search(r'query"?\s*[:=]\s*["\'](.*?)["\']', error_str, re.IGNORECASE)
            
            # 2. Extract Filter (optional)
            filter_match = re.search(r'filter_str"?\s*[:=]\s*["\'](.*?)["\']', error_str, re.IGNORECASE)
            
            if query_match:
                query_val = query_match.group(1)
                filter_val = filter_match.group(1) if filter_match else None
                
                # 3. Validate Filter - Only allow known columns
                # LanceDB schema: id, title, ingredients, instructions, image_name, search_text
                VALID_FILTER_COLUMNS = ['id', 'title', 'ingredients', 'instructions', 'image_name', 'search_text']
                if filter_val:
                    # Check if ANY valid column is mentioned in the filter
                    has_valid_column = any(col in filter_val.lower() for col in VALID_FILTER_COLUMNS)
                    if not has_valid_column:
                        print(f"Invalid LLM filter '{filter_val}' - using client-side parser.")
                        # FALLBACK: Parse original user message for exclusions
                        filter_val = parse_user_exclusions(user_message)
                
                # If still no filter, try parsing user message anyway
                if not filter_val:
                    filter_val = parse_user_exclusions(user_message)
                
                print(f"Recovery Execution -> Query: {query_val}, Filter: {filter_val}")
                
                # Manual Tool Execution
                tool_result = tools_instance.search_recipes(None, query_val, filter_val)
                return tool_result
        
        # If we couldn't parse it, return a helpful error
        return f"Agent Error: {error_str}"
