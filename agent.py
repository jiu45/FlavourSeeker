import os
import re
import lancedb
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# ----- Configuration -----
DB_PATH = "data/lancedb"
TABLE_NAME = "recipes"

# Import search engine
from search_engine import RecipeSearchEngine

# Initialize engine
engine = RecipeSearchEngine()

# Initialize Groq client
client = Groq()  # Reads GROQ_API_KEY from environment

# --- System Prompt for Structured Output ---
SYSTEM_PROMPT = """You are "Culinary Compass AI", a recipe search assistant.

Your job is to understand user requests and output a structured search plan.

**DATABASE SCHEMA:**
- `title`: Recipe name
- `ingredients`: List of ingredients (use for "no X" exclusions)
- `instructions`: Cooking steps (use for "no frying" etc.)
- `tags`: AI tags like Cuisine, Diet, Course

**OUTPUT FORMAT (ALWAYS use this exact format):**
```
QUERY: <search keywords>
FILTER: <SQL filter or NONE>
FULL: <YES or NO>
RESPONSE: <brief explanation to user>
```

- Set FULL: YES when user asks "how to make", "recipe for", "how do I cook" (show full instructions)
- Set FULL: NO for browsing/searching (show summary only)

**FILTER SYNTAX:**
- **Include** ingredient: `ingredients LIKE '%cheese%'`
- **Exclude** ingredient: `ingredients NOT LIKE '%cheese%'`
- **Include** cooking method: `instructions LIKE '%bake%'`
- **Exclude** cooking method: `instructions NOT LIKE '%fry%'`
- **Filter by tag**: `tags LIKE '%Vegetarian%'` ✅ (Now supported!)
- **Exclude by tag**: `tags NOT LIKE '%Meat%'`
- **Multiple conditions**: `ingredients LIKE '%chicken%' AND tags LIKE '%Healthy%'`
- No filter needed: `NONE`

**RULES:**

1. **Extract Keywords**: "I'm sad, need comfort" → QUERY: comfort food soup stew

2. **Smart Inclusions**: "I want cheese" → Just put "cheese" in QUERY (simpler and works with semantic search)
   - But if user says "must have cheese" → FILTER: ingredients LIKE '%cheese%'

3. **Smart Exclusions**: "no cheese" → FILTER: ingredients NOT LIKE '%cheese%'

4. **Missing/Indirect Data** (nutritional concepts, metrics we don't have):
   - **Strategy**: Combine BOTH ingredients AND dish names for comprehensive coverage
   - **Always include disclaimer** in RESPONSE explaining the limitation
   
   - **Nutritional Concepts** (vitamins, minerals, macros):
     - "Vitamin A" → QUERY: carrot sweet potato spinach pumpkin squash
     - "High fiber" → QUERY: vegetables salad greens quinoa beans lentils whole grain oats broccoli
     - "High protein" → QUERY: chicken beef fish tofu eggs meat legumes
     - "Iron rich" → QUERY: spinach red meat lentils chickpeas beans
   
   - **Missing Metrics** (calories, time, serving size):
     - "Low calorie" → QUERY: salad grilled vegetables steamed fish chicken broccoli greens
     - RESPONSE: Note: We don't have calorie data. Here are dishes typically low in calories...
     - "Quick recipes" → QUERY: quick easy stir-fry wrap sandwich chicken vegetables
     - RESPONSE: Note: We don't have cooking times. Here are typically quick dishes...
   
   - **Special Diets** (when not using tag filters):
     - "Low histamine" → QUERY: fresh chicken rice vegetables | FILTER: ingredients NOT LIKE '%tomato%' AND ingredients NOT LIKE '%fermented%'

5. **Recipe Modification**: If a user asks for a variation (e.g., "Lamb Pho", "Sugar Potatoes") but only the standard exists:
   - Search for the closest standard dish (e.g., "Pho", "Crispy Potatoes")
   - **CRITICAL**: In RESPONSE, matches found will be shown below. You must explicitly explain how to modify the standard recipe to match their request.
   - Example: "Lamb Pho" → QUERY: pho | RESPONSE: We have a Beef Pho recipe. You can use this as a base and simply substitute lamb for the beef.
   - Example: "Sugar and Pepper Potatoes" → QUERY: salt pepper potatoes | RESPONSE: We have a recipe for Salt and Pepper Potatoes. You can modify this by using sugar instead of salt to get the sweet-savory flavor you want.

6. **Clarification**: If too vague, set QUERY: CLARIFY and ask in RESPONSE.

7. **Multiple Dishes**: "Chicken and Cake" → Make TWO separate outputs.

8. **Double Negatives**: Convert to positive before filtering.
   - "I don't want anything without cheese" = "I want cheese" → FILTER: ingredients LIKE '%cheese%'
   - "No recipes that aren't vegan" = "Only vegan" → QUERY: vegan | FILTER: tags LIKE '%Vegan%'

9. **Equipment/Method Filters**: Be flexible with phrasing.
    - "No oven needed" → FILTER: instructions NOT LIKE '%bake%' AND instructions NOT LIKE '%oven%'
    - "Slow cooker recipes" → QUERY: slow cooker crock pot | FILTER: instructions LIKE '%slow cooker%'

10. **Knowledge/Reasoning Questions** (no search needed):
    - If user asks about substitutions, nutrition facts, or comparisons, set QUERY: KNOWLEDGE
    - Provide the answer directly in RESPONSE without searching
    - Example: "Can I replace butter with margarine?" → QUERY: KNOWLEDGE | RESPONSE: Yes, you can substitute...
    - Example: "Is salmon healthy?" → QUERY: KNOWLEDGE | RESPONSE: Salmon is rich in omega-3...

**EXAMPLES:**

User: "I need something with no cheese"
```
QUERY: recipe dinner lunch
FILTER: ingredients NOT LIKE '%cheese%'
RESPONSE: Here are recipes without cheese.
```

User: "spicy dinner but no beef"  
```
QUERY: spicy dinner
FILTER: ingredients NOT LIKE '%beef%'
RESPONSE: Spicy dinner options without beef.
```

User: "I don't want to fry anything"
```
QUERY: healthy baked steamed grilled
FILTER: instructions NOT LIKE '%fry%'
RESPONSE: Recipes that don't require frying.
```

User: "Tortilla"
```
QUERY: CLARIFY
FILTER: NONE
RESPONSE: Do you mean Spanish tortilla (egg omelette) or Mexican tortilla (flatbread)?
```

User: "I want something with cheese"
```
QUERY: cheese recipe cheesy
FILTER: NONE
RESPONSE: Here are delicious cheesy recipes.
```

User: "must have chicken, no dairy"
```
QUERY: chicken
FILTER: ingredients LIKE '%chicken%' AND ingredients NOT LIKE '%milk%' AND ingredients NOT LIKE '%cheese%' AND ingredients NOT LIKE '%cream%'
RESPONSE: Chicken recipes without dairy products.
```

User: "vegetarian pasta"
```
QUERY: vegetarian pasta
FILTER: tags LIKE '%Vegetarian%'
FULL: NO
RESPONSE: Vegetarian pasta dishes.
```

User: "How do I make mac and cheese?"
```
QUERY: mac and cheese
FILTER: NONE
FULL: YES
RESPONSE: Here's how to make mac and cheese:
```

User: "recipe for chicken soup"
```
QUERY: chicken soup
FILTER: NONE
FULL: YES
RESPONSE: Here's a delicious chicken soup recipe:
```

IMPORTANT: Always output the structured format. Never output code or function calls.
"""


def parse_agent_output(output: str) -> dict:
    """
    Parse the structured output from the LLM.
    
    Returns:
        dict with 'query', 'filter', 'full', 'response' keys
    """
    result = {
        'query': None,
        'filter': None,
        'full': False,
        'response': None,
        'raw': output
    }
    
    # Extract QUERY
    query_match = re.search(r'QUERY:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    if query_match:
        result['query'] = query_match.group(1).strip()
    
    # Extract FILTER
    filter_match = re.search(r'FILTER:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    if filter_match:
        filter_val = filter_match.group(1).strip()
        if filter_val.upper() != 'NONE':
            result['filter'] = filter_val
    
    # Extract FULL
    full_match = re.search(r'FULL:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    if full_match:
        result['full'] = full_match.group(1).strip().upper() == 'YES'
    
    # Extract RESPONSE
    response_match = re.search(r'RESPONSE:\s*(.+?)(?:\n```|$)', output, re.IGNORECASE | re.DOTALL)
    if response_match:
        result['response'] = response_match.group(1).strip()
    
    return result


def convert_tag_filters(filter_str: str) -> str:
    """
    Convert tag LIKE filters to array operations.
    
    Example: tags LIKE '%Vegetarian%' → array_has_any(tags, ['Vegetarian'])
    """
    if not filter_str:
        return filter_str
    
    def replace_tag_like(match):
        is_not = match.group(1)
        value = match.group(2)
        array_expr = f"array_has_any(tags, ['{value}'])"
        if is_not:
            return f"NOT {array_expr}"
        return array_expr
    
    # Match: tags LIKE '%Value%' or tags NOT LIKE '%Value%'
    converted = re.sub(
        r"tags\s+(NOT\s+)?LIKE\s+'%(\w+)%'",
        replace_tag_like,
        filter_str,
        flags=re.IGNORECASE
    )
    
    return converted


def search_with_filter(query: str, filter_str: str = None, show_full: bool = False) -> str:
    """
    Execute search with validated filter.
    
    Args:
        query: Search keywords
        filter_str: SQL filter
        show_full: If True, show full recipe with instructions (for "how to" questions)
    """
    try:
        # Convert tag filters to array operations
        if filter_str:
            filter_str = convert_tag_filters(filter_str)
        
        # Validate filter before use
        if filter_str:
            # Check for obviously broken filters
            if "LIKE" in filter_str.upper() and "'%"not in filter_str:
                print(f"[WARN] Malformed filter: {filter_str}")
                filter_str = None
        
        # Use relevance score threshold to filter out irrelevant results
        # Based on testing with 8 recipes:
        #   0.030+: Excellent match (direct query like "mac and cheese" → Mac and Cheese)
        #   0.016-0.020: Weak match (generic queries return all recipes with similar scores)
        #   <0.015: Poor match
        # Setting to 0.025 means: Only show excellent/direct matches, filter weak ones
        min_score = 0.025  # Strict: Only excellent matches
        
        results = engine.search_by_text(
            query, 
            top_k=5 if not show_full else 1, 
            where=filter_str,
            min_score=min_score
        )
        
        if results.empty:
            return "No recipes found matching your criteria. Try different keywords or remove some filters."
        
        response_parts = []
        for _, row in results.iterrows():
            response_parts.append(f"## {row['title']}")
            
            if 'visual_description' in row and row['visual_description']:
                response_parts.append(f"_{row['visual_description']}_")
            
            response_parts.append(f"\n**Ingredients:**\n{row['ingredients']}")
            
            # Show full instructions if requested
            if show_full and 'instructions' in row:
                response_parts.append(f"\n**Instructions:**\n{row['instructions']}")
            
            response_parts.append("\n---")
        
        return "\n".join(response_parts)
    
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        # Retry without filter
        if filter_str:
            return search_with_filter(query, None, show_full) + "\n\n(Note: Filter was invalid, showing unfiltered results)"
        return f"Search error: {str(e)}"


def chat_with_agent(user_message: str, history: list = None) -> str:
    """
    Main chat interface for the agent.
    
    Args:
        user_message: User's query
        history: Chat history (optional, for context)
    
    Returns:
        Agent's response
    """
    if history is None:
        history = []
    
    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add history
    for h in history:
        messages.append({"role": "user", "content": h['user']})
        messages.append({"role": "assistant", "content": h['assistant']})
    
    # Add current message
    messages.append({"role": "user", "content": user_message})
    
    try:
        # Call LLM
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )
        
        llm_output = response.choices[0].message.content.strip()
        print(f"[LLM Output]\n```\n{llm_output}\n```\n")
        
        # Parse output
        parsed = parse_agent_output(llm_output)
        print(f"[Parsed] Query: {parsed['query']}, Filter: {parsed['filter']}, Full: {parsed['full']}")
        
        # Handle clarification
        if parsed['query'] and parsed['query'].upper() == 'CLARIFY':
            return parsed['response'] or "Could you please be more specific about what you're looking for?"
        
        # Handle knowledge/reasoning questions (no search needed)
        if parsed['query'] and parsed['query'].upper() == 'KNOWLEDGE':
            return parsed['response'] or "I can help with that question."
            
        # Execute search if we have a query
        if parsed['query']:
            search_results = search_with_filter(parsed['query'], parsed['filter'], parsed['full'])
            
            # Combine agent response with search results
            if parsed['response']:
                return f"{parsed['response']}\n\n{search_results}"
            return search_results
        
        # Fallback if no query
        return parsed['response'] or "I couldn't understand your request. Could you rephrase?"
        
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Interactive test
    print("Culinary Compass AI - Recipe Search Agent")
    print("=" * 50)
    print("Try queries like:")
    print("  - 'I want something with chicken but no dairy'")
    print("  - 'vegetarian pasta'")
    print("  - 'How do I make mac and cheese?'")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        response = chat_with_agent(user_input)
        print(f"\nAgent: {response}")
