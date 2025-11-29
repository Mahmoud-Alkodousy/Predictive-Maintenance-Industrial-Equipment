"""
PDF Document Chunking and Embedding Upload to Supabase
Extracts text from PDF, splits into chunks, generates embeddings, and uploads to Supabase
for document-based semantic search and RAG applications
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
from supabase import create_client
from sentence_transformers import SentenceTransformer
import PyPDF2
from dotenv import load_dotenv
from typing import Tuple, List, Optional

# ============================================
# ENVIRONMENT CONFIGURATION
# ============================================

# Load environment variables from .env file
load_dotenv()

# Required: Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Optional: Configuration parameters with defaults
PDF_PATH = os.getenv("PDF_PATH", "")
FILENAME = os.getenv("FILENAME", None)  # Optional filename override
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))  # Characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # Overlap between chunks

# Validate required environment variables
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "❌ Missing required environment variables:\n"
        "   - SUPABASE_URL\n"
        "   - SUPABASE_KEY\n"
        "Please set them in .env file or environment"
    )

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================
# PDF TEXT EXTRACTION
# ============================================

def read_pdf_text(path: str) -> str:
    """
    Extract all text from PDF file
    
    Processes all pages and concatenates text with newlines
    Handles extraction errors gracefully (skips problematic pages)
    
    Args:
        path (str): Path to PDF file
        
    Returns:
        str: Extracted text from all pages
    """
    try:
        reader = PyPDF2.PdfReader(path)
        texts = []
        
        # Process each page
        for page_num, page in enumerate(reader.pages, 1):
            try:
                # Extract text from page
                page_text = page.extract_text() or ""
                texts.append(page_text)
            except Exception as e:
                print(f"⚠️ Warning: Failed to extract text from page {page_num}: {e}")
                texts.append("")  # Add empty string to maintain page order
        
        # Concatenate all pages with newlines
        full_text = "\n".join(texts)
        return full_text
    
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ PDF file not found: {path}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to read PDF: {e}")


# ============================================
# TEXT CHUNKING
# ============================================

def chunk_text(
    text: str, 
    chunk_size: int = 1000, 
    overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation
    
    Chunking strategy:
    - Fixed-size chunks with overlap
    - Overlap ensures context continuity across chunks
    - Maintains semantic coherence for embeddings
    
    Example:
        text = "ABCDEFGHIJ"
        chunk_size = 4, overlap = 2
        
        Chunks:
        1. "ABCD"
        2. "CDEF"  (2-char overlap with chunk 1)
        3. "EFGH"  (2-char overlap with chunk 2)
        4. "GHIJ"  (2-char overlap with chunk 3)
    
    Args:
        text (str): Input text to chunk
        chunk_size (int): Maximum characters per chunk
        overlap (int): Number of overlapping characters between chunks
        
    Returns:
        list: List of text chunks
    """
    chunks = []
    start = 0
    length = len(text)
    
    while start < length:
        # Calculate end position for this chunk
        end = start + chunk_size
        
        # Extract chunk
        chunk = text[start:end]
        
        # Add to list (strip whitespace)
        chunks.append(chunk.strip())
        
        # Check if we've reached the end
        if end >= length:
            break
        
        # Move start position (with overlap)
        start = end - overlap
    
    return chunks


# ============================================
# EMBEDDING MODEL
# ============================================

def load_embed_model() -> SentenceTransformer:
    """
    Load sentence transformer model for generating embeddings
    
    Model: all-MiniLM-L6-v2
    - Dimensions: 384
    - Speed: Fast
    - Quality: Good for semantic search
    
    Returns:
        SentenceTransformer: Loaded model
    """
    print("🧠 Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"✅ Model loaded | Dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_texts(model: SentenceTransformer, texts: List[str]):
    """
    Generate embeddings for list of text chunks
    
    Args:
        model: Loaded SentenceTransformer model
        texts (list): List of text chunks
        
    Returns:
        numpy.ndarray: Array of embeddings
    """
    print("📈 Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


# ============================================
# ERROR HANDLING UTILITY
# ============================================

def _is_error_response(res) -> Tuple[bool, str]:
    """
    Robust error checking for Supabase response objects
    
    Handles different versions of supabase-py client library
    Checks multiple response formats for error indicators
    
    Args:
        res: Supabase response object
        
    Returns:
        tuple: (is_error: bool, error_message: str)
    """
    try:
        # Check 1: response.status_code (newer versions)
        if hasattr(res, "response") and getattr(res, "response") is not None:
            status = getattr(res.response, "status_code", None)
            text = getattr(res.response, "text", None)
            
            if status is not None and status >= 400:
                return True, f"HTTP {status}: {text}"
        
        # Check 2: Direct status_code attribute
        if hasattr(res, "status_code"):
            status = getattr(res, "status_code")
            
            if status is not None and int(status) >= 400:
                body = getattr(res, "text", None) or getattr(res, "body", None) or ""
                return True, f"HTTP {status}: {body}"
        
        # Check 3: Data field exists (no error if present)
        if hasattr(res, "data"):
            return False, ""
        
        # Default: assume success (compatibility mode)
        return False, ""
    
    except Exception as e:
        return True, f"Exception checking response: {e}"


# ============================================
# SUPABASE UPLOAD
# ============================================

def upsert_chunk_json(
    filename: str,
    chunk_index: int,
    content: str,
    embedding: List[float],
    source: Optional[str] = None
) -> Tuple[bool, any]:
    """
    Upload single chunk with embedding to Supabase
    
    Uploads to 'documents_json' table with schema:
    - filename: Source document name
    - chunk_index: Sequential chunk number
    - content: Text content of chunk
    - embedding: Vector embedding (384 dimensions)
    - source: Optional source identifier
    
    Args:
        filename (str): Document filename
        chunk_index (int): Index of this chunk
        content (str): Text content
        embedding (list): Embedding vector as list
        source (str, optional): Source identifier
        
    Returns:
        tuple: (success: bool, response: object)
    """
    # Prepare payload
    payload = {
        "filename": filename,
        "chunk_index": chunk_index,
        "content": content,
        "embedding": embedding,
        "source": source or f"chunk_{chunk_index}"
    }
    
    try:
        # Insert into Supabase
        res = supabase.table("documents_json").insert(payload).execute()
        
        # Check for errors
        is_err, msg = _is_error_response(res)
        
        if is_err:
            print(f"⚠️ Supabase insert error at chunk {chunk_index}: {msg}")
            return False, res
        else:
            print(f"✅ Chunk {chunk_index} uploaded successfully")
            return True, res
    
    except Exception as e:
        print(f"❌ Exception uploading chunk {chunk_index}: {e}")
        return False, None


# ============================================
# MAIN PROCESSING PIPELINE
# ============================================

def process_pdf(path: str, filename_override: Optional[str] = None):
    """
    Complete pipeline for processing PDF and uploading to Supabase
    
    Pipeline:
    1. Validate PDF exists
    2. Extract text from all pages
    3. Split text into overlapping chunks
    4. Generate embeddings for all chunks
    5. Upload chunks with embeddings to Supabase
    
    Args:
        path (str): Path to PDF file
        filename_override (str, optional): Override filename in database
    """
    
    # ============================================
    # STEP 1: Validate file
    # ============================================
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"❌ PDF not found: {path}")

    filename = filename_override if filename_override else path.name
    print(f"\n{'='*60}")
    print(f"📄 Processing PDF: {filename}")
    print(f"{'='*60}\n")

    # ============================================
    # STEP 2: Extract text
    # ============================================
    
    print("📖 Extracting text from PDF...")
    text = read_pdf_text(str(path))
    
    if not text or not text.strip():
        print("❌ No text extracted from PDF. File may be empty or image-based.")
        return

    word_count = len(text.split())
    char_count = len(text)
    print(f"✅ Extracted {word_count:,} words ({char_count:,} characters)")

    # ============================================
    # STEP 3: Split into chunks
    # ============================================
    
    print(f"\n✂️ Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Show sample chunk
    if chunks:
        print(f"\n📝 Sample chunk (first 200 chars):")
        print(f"   {chunks[0][:200]}...")

    # ============================================
    # STEP 4: Generate embeddings
    # ============================================
    
    model = load_embed_model()
    embeddings = embed_texts(model, chunks)
    print(f"✅ Generated {len(embeddings)} embeddings")

    # ============================================
    # STEP 5: Upload to Supabase
    # ============================================
    
    print(f"\n🚀 Uploading {len(chunks)} chunks to Supabase...")
    success_count = 0
    failed_chunks = []
    
    for idx, (chunk, emb) in enumerate(tqdm(zip(chunks, embeddings), total=len(chunks))):
        # Convert embedding to Python list (safe conversion)
        try:
            if hasattr(emb, "tolist"):
                emb_list = emb.tolist()
            else:
                emb_list = list(map(float, emb))
        except Exception as e:
            print(f"⚠️ Warning: Failed to convert embedding {idx}: {e}")
            emb_list = [float(x) for x in emb]
        
        # Upload chunk
        ok, res = upsert_chunk_json(filename, idx, chunk, emb_list, f"chunk_{idx}")
        
        if ok:
            success_count += 1
        else:
            failed_chunks.append(idx)
            
            # Show detailed error info if available
            try:
                if hasattr(res, "response") and getattr(res.response, "text", None):
                    print(f"Response body: {res.response.text[:200]}")
            except Exception:
                pass

    # ============================================
    # STEP 6: Summary
    # ============================================
    
    print(f"\n{'='*60}")
    print("📊 Upload Summary:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(failed_chunks)}")
    print(f"   Success rate: {success_count/len(chunks)*100:.1f}%")
    print(f"   Document: {filename}")
    
    if failed_chunks:
        print(f"\n⚠️ Failed chunks: {failed_chunks[:10]}" + 
              (f" ... and {len(failed_chunks)-10} more" if len(failed_chunks) > 10 else ""))
    
    print(f"{'='*60}\n")
    
    if success_count == len(chunks):
        print("✅ All chunks uploaded successfully!")
    else:
        print(f"⚠️ {len(failed_chunks)} chunks failed. Check errors above.")


# ============================================
# ENTRY POINT
# ============================================

def main():
    """
    Main entry point - processes PDF from environment or user input
    """
    # Get PDF path from environment or prompt user
    pdf_path = PDF_PATH or input("Enter path to PDF file: ").strip()
    
    # Get optional filename override
    filename_override = FILENAME
    
    try:
        process_pdf(pdf_path, filename_override)
    except FileNotFoundError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Error processing PDF: {e}")
        raise


if __name__ == "__main__":
    main()