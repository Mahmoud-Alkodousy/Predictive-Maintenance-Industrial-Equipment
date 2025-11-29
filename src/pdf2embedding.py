"""
Supabase Embeddings Upload Script
Loads structured problems from JSONL, generates embeddings, and uploads to Supabase
for semantic search and RAG (Retrieval-Augmented Generation) applications
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from supabase import create_client
from sentence_transformers import SentenceTransformer
import numpy as np

# ============================================
# ENVIRONMENT CONFIGURATION
# ============================================

# Load environment variables from .env file
load_dotenv()

# Required: Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Optional: Configuration parameters with defaults
JSONL_PATH = os.getenv("JSONL_PATH", "/mnt/data/user_pasted_problems_structured.jsonl")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # Sentence transformer model
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))  # Number of items per batch
TABLE_NAME = os.getenv("TABLE_NAME", "problems")  # Supabase table name

# Validate required environment variables
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit(
        "❌ Missing required environment variables:\n"
        "   - SUPABASE_URL\n"
        "   - SUPABASE_KEY\n"
        "Please set them in .env file or environment"
    )

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================
# DATA LOADING FUNCTIONS
# ============================================

def load_jsonl(path: str) -> list:
    """
    Load structured problems from JSONL file
    
    JSONL format: One JSON object per line
    Expected structure:
    {
        "id": "problem_1",
        "title": "Motor overheating",
        "description": "...",
        "summary": "...",
        "symptoms": ["high temperature", "noise"],
        "causes": ["worn bearings", "insufficient lubrication"],
        "solutions": ["replace bearings", "add lubrication"],
        "notes": "...",
        "raw_text": "..."
    }
    
    Args:
        path (str): Path to JSONL file
        
    Returns:
        list: List of problem dictionaries
    """
    items = []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                try:
                    # Parse JSON object
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️ Warning: Invalid JSON on line {line_num}: {e}")
                    continue
        
        print(f"✅ Loaded {len(items)} items from {path}")
        return items
    
    except FileNotFoundError:
        raise SystemExit(f"❌ File not found: {path}")


def make_text_for_embedding(item: dict) -> str:
    """
    Create composite text for embedding generation
    
    Combines multiple fields into a single text optimized for semantic search:
    - Title (most important)
    - Summary
    - Description
    - Top 3 solutions
    
    Args:
        item (dict): Problem dictionary
        
    Returns:
        str: Combined text for embedding
    """
    parts = []
    
    # Add title (highest semantic weight)
    if item.get("title"):
        parts.append(item["title"])
    
    # Add summary (concise overview)
    if item.get("summary"):
        parts.append(item["summary"])
    
    # Add full description
    if item.get("description"):
        parts.append(item["description"])
    
    # Add top 3 solutions (actionable information)
    solutions = item.get("solutions") or []
    if len(solutions) > 0:
        parts.append("Solution: " + " | ".join(solutions[:3]))
    
    # Join with newlines for better semantic separation
    return " \n ".join(parts)


# ============================================
# UTILITY FUNCTIONS
# ============================================

def to_list(embedding_array) -> list:
    """
    Convert numpy array or tensor to Python list for JSON serialization
    
    Args:
        embedding_array: NumPy array or PyTorch tensor
        
    Returns:
        list: Python list of floats
    """
    return [float(v) for v in np.asarray(embedding_array).tolist()]


def chunk_iterable(lst: list, n: int):
    """
    Split list into chunks of size n
    
    Args:
        lst (list): Input list
        n (int): Chunk size
        
    Yields:
        list: Chunks of size n (last chunk may be smaller)
    """
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def upsert_batch(rows: list):
    """
    Upload batch of rows to Supabase using upsert
    
    Upsert = Insert or Update
    - If row with same ID exists → Update
    - If row doesn't exist → Insert
    
    Args:
        rows (list): List of row dictionaries
        
    Returns:
        Supabase response object
    """
    try:
        res = supabase.table(TABLE_NAME).upsert(rows).execute()
        return res
    except Exception as e:
        print(f"❌ Batch upload failed: {e}")
        raise


# ============================================
# MAIN UPLOAD PIPELINE
# ============================================

def main():
    """
    Main pipeline for generating and uploading embeddings
    
    Process:
    1. Load problems from JSONL file
    2. Generate text representations for embedding
    3. Load sentence transformer model
    4. Generate embeddings in batches
    5. Upload to Supabase with progress tracking
    """
    
    print("🚀 Starting embedding upload pipeline...\n")
    
    # ============================================
    # STEP 1: Load data
    # ============================================
    
    print(f"📂 Loading data from: {JSONL_PATH}")
    items = load_jsonl(JSONL_PATH)
    
    if len(items) == 0:
        raise SystemExit("❌ No items loaded. Check your JSONL file.")
    
    # ============================================
    # STEP 2: Prepare texts for embedding
    # ============================================
    
    print("📝 Preparing texts for embedding...")
    texts = [make_text_for_embedding(item) for item in items]
    print(f"✅ Prepared {len(texts)} texts")
    
    # ============================================
    # STEP 3: Load embedding model
    # ============================================
    
    print(f"\n🤖 Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"✅ Model loaded | Embedding dimension: {embedding_dim}")
    
    # ============================================
    # STEP 4: Generate and upload embeddings in batches
    # ============================================
    
    total = len(items)
    print(f"\n📤 Uploading {total} items in batches of {BATCH_SIZE}...")
    
    # Progress tracking
    uploaded_count = 0
    failed_count = 0
    
    # Process in batches for efficiency
    for batch_indices in chunk_iterable(list(range(total)), BATCH_SIZE):
        
        # Get texts for current batch
        batch_texts = [texts[i] for i in batch_indices]
        
        # Generate embeddings for batch
        embeddings = model.encode(
            batch_texts, 
            show_progress_bar=False,  # Disable internal progress bar
            convert_to_numpy=True
        )
        
        # Prepare payload for Supabase
        payload = []
        for idx, embedding in zip(batch_indices, embeddings):
            item = items[idx]
            
            # Construct row with all fields
            row = {
                "id": item.get("id") or f"problem_{idx+1}",
                "title": item.get("title") or "",
                "description": item.get("description") or "",
                "summary": item.get("summary") or "",
                "symptoms": item.get("symptoms") or [],
                "causes": item.get("causes") or [],
                "solutions": item.get("solutions") or [],
                "notes": item.get("notes") or "",
                "raw_text": item.get("raw_text") or "",
                "embedding": to_list(embedding)  # Convert numpy array to list
            }
            payload.append(row)
        
        # Upload batch to Supabase
        try:
            res = upsert_batch(payload)
            uploaded_count += len(batch_indices)
            
            # Display progress
            print(
                f"✅ Batch {uploaded_count}/{total} | "
                f"Items {batch_indices[0]+1}..{batch_indices[-1]+1} | "
                f"Status: {getattr(res, 'status_code', 'ok')}"
            )
        
        except Exception as e:
            failed_count += len(batch_indices)
            print(f"❌ Failed to upload batch {batch_indices[0]}..{batch_indices[-1]}: {e}")
        
        # Rate limiting - avoid overwhelming Supabase
        time.sleep(0.1)
    
    # ============================================
    # STEP 5: Summary
    # ============================================
    
    print("\n" + "="*60)
    print("📊 Upload Summary:")
    print(f"   Total items: {total}")
    print(f"   Uploaded: {uploaded_count}")
    print(f"   Failed: {failed_count}")
    print(f"   Success rate: {uploaded_count/total*100:.1f}%")
    print("="*60)
    
    if failed_count == 0:
        print("\n✅ All embeddings uploaded successfully!")
    else:
        print(f"\n⚠️ {failed_count} items failed to upload. Check errors above.")


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Upload interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        raise