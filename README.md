# 🧭 FlavourSeeker (Culinary Compass)

**FlavourSeeker** is a multimodal AI-powered recipe search engine. It helps you find the perfect dish using text descriptions, food images, or by checking what's in your pantry.

## 🌟 Features

*   **Search by Name**: Find recipes by title or description using semantic search (MiniLM).
*   **Visual Search**: Upload a photo of a dish to find visually similar recipes (OpenAI CLIP).
*   **Pantry Search (Smart)**:
    *   **Strict Mode**: Find recipes you can make *right now* with your current ingredients.
    *   **Flexible Mode**: Find recipes you can *partially* make, prioritizing the ones closest to completion.
*   **AI Chat**: Chat with a "Recipe Bot" to get tips or substitutions (placeholder).

## 🛠️ Setup

### 1. Prerequisites
*   Python 3.10+
*   Git

### 2. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/jiu45/FlavourSeeker.git
cd FlavourSeeker
pip install -r requirements.txt
```

### 3. Data Preparation
You need to place your dataset in the root directory:
1.  **CSV File**: `Food Ingredients and Recipe Dataset with Image Name Mapping.csv`
2.  **Images Folder**: `Food Images/` (containing recipe JPG/PNG files)

*(Note: The repository contains a mock script to generate dummy data if you don't have the full dataset yet.)*

### 4. Database Ingestion
Before running the app, you must process the data and build the Vector Database (LanceDB).

```bash
python ingest.py
```
*Note: This may take a few minutes on the first run to download the AI models.*

## 🚀 Running the App

Start the Streamlit interface:

```bash
streamlit run app.py
```

Or on Windows, simply double-click:
`run_app.bat`

## 🧠 Technologies
*   **Frontend**: Streamlit
*   **Vector DB**: LanceDB
*   **Embeddings**:
    *   Text: `all-MiniLM-L6-v2`
    *   Image: `openai/clip-vit-base-patch32`
