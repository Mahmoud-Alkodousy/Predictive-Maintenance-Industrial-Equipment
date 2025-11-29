"""
===============================================================================
FACTORY MAINTENANCE AI ASSISTANT - ENHANCED VERSION
===============================================================================
✅ Fixed RAG response quality
✅ Better casual conversation handling  
✅ More flexible system prompts
✅ Improved context utilization

Developer: Eng. Mahmoud Khalid Alkodousy
===============================================================================
"""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

import os
import sys

# Prevent library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['USE_TF'] = 'NO'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

model_display_names = {
    "openai/gpt-4o": "GPT-4o 🧠",
    "openai/gpt-4o-mini": "GPT-4o Mini 🚀",
    "google/gemini-pro-1.5": "Gemini Pro 1.5 💎",
    "anthropic/claude-3-haiku": "Claude 3 Haiku ⚡",
    "anthropic/claude-3.5-sonnet": "Claude 3.5 Sonnet 🎭",
    "meta-llama/llama-3.1-70b-instruct": "Llama 3.1 70B 🦙",
    "mistralai/mixtral-8x7b-instruct": "Mixtral 8x7B 🌟"
}

# ============================================================================
# IMPORTS
# ============================================================================

import json
import requests
import traceback
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import time
from collections import OrderedDict, defaultdict
from functools import wraps
from dataclasses import dataclass
from typing import Generic, TypeVar, Union
import logging
import re

import streamlit as st
from dotenv import load_dotenv

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

_SUPABASE_CLIENT = None
_EMBEDDING_MODEL = None
_EMBEDDINGS_LOADED = False

load_dotenv()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# GENERIC RESULT TYPE
# ============================================================================

T = TypeVar('T')

@dataclass
class Result(Generic[T]):
    """
    Generic result wrapper for operations that can succeed or fail
    """
    success: bool
    data: Union[T, None] = None
    error: Union[str, None] = None
    metadata: dict = None

    @classmethod
    def ok(cls, data: T, metadata: dict = None) -> 'Result[T]':
        """Create successful result"""
        return cls(success=True, data=data, metadata=metadata or {})
    
    @classmethod
    def fail(cls, error: str, metadata: dict = None) -> 'Result[T]':
        """Create failed result"""
        return cls(success=False, error=error, metadata=metadata or {})

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_supabase():
    """
    Get or create Supabase client connection
    Uses singleton pattern to avoid multiple connections
    
    Returns:
        Supabase client or None if connection fails
    """
    global _SUPABASE_CLIENT
    
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT
    
    try:
        from supabase import create_client
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            logger.error("Missing Supabase credentials in .env")
            return None
        
        _SUPABASE_CLIENT = create_client(url, key)
        
        # Test connection
        _SUPABASE_CLIENT.table("machines").select("id").limit(1).execute()
        logger.info("✅ Supabase connected successfully")
        return _SUPABASE_CLIENT
        
    except Exception as e:
        logger.error(f"Supabase connection failed: {e}")
        return None

# ============================================================================
# EMBEDDING MODEL
# ============================================================================

def get_embedding_model():
    """
    Load sentence transformer model for text embeddings
    Uses singleton pattern with lazy loading
    
    Returns:
        SentenceTransformer model or None if loading fails
    """
    global _EMBEDDING_MODEL, _EMBEDDINGS_LOADED
    
    if _EMBEDDINGS_LOADED:
        return _EMBEDDING_MODEL
    
    _EMBEDDINGS_LOADED = True
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        _EMBEDDING_MODEL = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        logger.info("✅ Embedding model loaded successfully")
        return _EMBEDDING_MODEL
        
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        _EMBEDDING_MODEL = None
        return None

def embed_text(text: str, use_cache: bool = True):
    """
    Generate embedding vector for text
    
    Args:
        text: Input text to embed
        use_cache: Whether to use cached embeddings
    
    Returns:
        numpy array of embedding or None if fails
    """
    model = get_embedding_model()
    if model is None:
        logger.warning("Embedding model unavailable")
        return None
    
    try:
        import numpy as np
        
        # Create cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache
        if use_cache:
            cached = embedding_cache.get(cache_key)
            if cached is not None:
                return np.array(cached, dtype=np.float32)
        
        # Generate embedding
        emb = model.encode(text, show_progress_bar=False)
        arr = np.array(emb, dtype=np.float32)
        
        # Cache result
        embedding_cache.set(cache_key, arr)
        return arr
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None

# ============================================================================
# METRICS TRACKING
# ============================================================================

class MetricsTracker:
    """
    Track performance metrics and API costs
    """
    def __init__(self):
        self.metrics = defaultdict(list)
        self.enabled = True
        self.api_costs = defaultdict(float)
    
    def record(self, metric_name: str, value: float):
        """Record a metric value"""
        if not self.enabled:
            return
        
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.now()
        })
    
    def record_cost(self, tokens: int, model: str):
        """Record API cost based on tokens used"""
        cost_per_1k = 0.003 if "claude" in model.lower() else 0.001
        cost = (tokens / 1000) * cost_per_1k
        self.api_costs[model] += cost
    
    def get_stats(self, metric_name: str) -> Dict[str, Any]:
        """Get statistics for a specific metric"""
        data = self.metrics.get(metric_name, [])
        if not data:
            return {
                "count": 0,
                "avg": 0,
                "min": 0,
                "max": 0,
                "total": 0
            }
        
        values = [d['value'] for d in data]
        return {
            "count": len(values),
            "avg": round(sum(values) / len(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "total": round(sum(values), 3)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked statistics"""
        all_stats = {}
        
        for metric_name in self.metrics.keys():
            if '_duration' in metric_name:
                func_name = metric_name.replace('_duration', '')
                if func_name not in all_stats:
                    all_stats[func_name] = {}
                all_stats[func_name]['duration'] = self.get_stats(metric_name)
            elif '_success' in metric_name:
                func_name = metric_name.replace('_success', '')
                if func_name not in all_stats:
                    all_stats[func_name] = {}
                all_stats[func_name]['success_count'] = len(self.metrics[metric_name])
            elif '_errors' in metric_name:
                func_name = metric_name.replace('_errors', '')
                if func_name not in all_stats:
                    all_stats[func_name] = {}
                all_stats[func_name]['error_count'] = len(self.metrics[metric_name])
        
        return all_stats
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        all_stats = self.get_all_stats()
        
        total_calls = sum(
            stats.get('success_count', 0) + stats.get('error_count', 0)
            for stats in all_stats.values()
        )
        
        total_errors = sum(
            stats.get('error_count', 0)
            for stats in all_stats.values()
        )
        
        total_cost = sum(self.api_costs.values())
        
        return {
            "total_function_calls": total_calls,
            "total_errors": total_errors,
            "success_rate": f"{((total_calls - total_errors) / total_calls * 100):.1f}%" if total_calls > 0 else "N/A",
            "tracked_functions": len(all_stats),
            "metrics_count": sum(len(v) for v in self.metrics.values()),
            "total_api_cost": f"${total_cost:.4f}"
        }
    
    def clear(self):
        """Clear all metrics"""
        self.metrics.clear()
        self.api_costs.clear()
        logger.info("📊 All metrics cleared")
    
    def toggle(self, enabled: bool = None):
        """Toggle metrics tracking on/off"""
        if enabled is None:
            self.enabled = not self.enabled
        else:
            self.enabled = enabled
        logger.info(f"📊 Metrics tracking: {'enabled' if self.enabled else 'disabled'}")

# Global metrics tracker instance
metrics_tracker = MetricsTracker()

# ============================================================================
# PERFORMANCE MONITORING DECORATOR
# ============================================================================

def monitor_performance(func):
    """
    Decorator to track function performance
    Records execution time, success/error counts
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not metrics_tracker.enabled:
            return func(*args, **kwargs)
        
        func_name = func.__name__
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            metrics_tracker.record(f"{func_name}_duration", duration)
            metrics_tracker.record(f"{func_name}_success", 1)
            
            if duration > 1.0:
                logger.info(f"⚡ {func_name} completed in {duration:.3f}s")
            else:
                logger.debug(f"✓ {func_name} completed in {duration:.3f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            metrics_tracker.record(f"{func_name}_duration", duration)
            metrics_tracker.record(f"{func_name}_errors", 1)
            
            logger.error(f"✗ {func_name} failed after {duration:.3f}s: {str(e)[:100]}")
            
            raise
    
    return wrapper

# ============================================================================
# CONFIGURATION
# ============================================================================

try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    
    class Config(BaseModel):
        """
        Application configuration with validation
        """
        model_config = ConfigDict(env_prefix='')
        
        supabase_url: str = Field(..., min_length=1)
        supabase_key: str = Field(..., min_length=1)
        openrouter_key: str = Field(..., min_length=1)
        openrouter_model: str = Field(default="openai/gpt-4o-mini")
        pdf_filename: str = Field(default="Industrial_Maintenance_Guide.pdf")
        top_k: int = Field(default=5, ge=1, le=20)
        timeout: int = Field(default=60, ge=10, le=300)
        max_requests_per_minute: int = Field(default=30, ge=10, le=100)

        @field_validator('supabase_url')
        @classmethod
        def validate_url(cls, v: str) -> str:
            """Validate Supabase URL format"""
            if not v.startswith('https://'):
                raise ValueError('URL must start with https://')
            return v.rstrip('/')

        @classmethod
        def from_env(cls) -> 'Config':
            """Load configuration from environment variables"""
            return cls(
                supabase_url=os.getenv("SUPABASE_URL"),
                supabase_key=os.getenv("SUPABASE_KEY"),
                openrouter_key=os.getenv("OPENROUTER_KEY"),
                openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                pdf_filename=os.getenv("RAG_PDF_FILENAME", "Industrial_Maintenance_Guide.pdf"),
                top_k=int(os.getenv("RAG_TOP_K", "5")),
                max_requests_per_minute=int(os.getenv("MAX_REQUESTS_PER_MINUTE", "30"))
            )
    
    config = Config.from_env()
    logger.info("✅ Configuration loaded successfully")
    
except Exception as e:
    logger.error(f"Configuration error: {e}")
    config = None

# ============================================================================
# SMART CACHING
# ============================================================================

class SmartCache:
    """
    LRU cache with TTL (Time To Live)
    """
    def __init__(self, max_size: int = 200, ttl: int = 3600):
        self._cache = OrderedDict()
        self._timestamps = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if exists and not expired"""
        if key in self._cache:
            # Check if not expired
            if time.time() - self._timestamps.get(key, 0) < self.ttl:
                self.hits += 1
                self._cache.move_to_end(key)  # Move to end (most recent)
                return self._cache[key]
            else:
                # Remove expired entry
                del self._cache[key]
                del self._timestamps[key]
        
        self.misses += 1
        return None

    def set(self, key: str, value: Any):
        """Set cache value with current timestamp"""
        # Remove oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            del self._timestamps[oldest]
        
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def clear(self):
        """Clear all cached data"""
        self._cache.clear()
        self._timestamps.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{(self.hits / total * 100):.1f}%" if total > 0 else "0%"
        }

# Global cache instances
embedding_cache = SmartCache(max_size=500, ttl=7200)  # 2 hours
query_cache = SmartCache(max_size=200, ttl=3600)      # 1 hour
pdf_cache = SmartCache(max_size=50, ttl=7200)         # 2 hours

# ============================================================================
# QUERY ANALYSIS
# ============================================================================

class QueryAnalyzer:
    """
    Analyze user queries for language and machine IDs
    """
    ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF]')
    
    # Casual greetings detection
    CASUAL_PATTERNS_AR = [
        r'^\s*(ازيك|إزيك|ازاى|إزاى|ايه|أيه|يا|هاى|سلام|مرحبا|صباح|مساء)\s*',
        r'^\s*(اخبارك|أخبارك|عامل|حالك)\s*'
    ]
    
    CASUAL_PATTERNS_EN = [
        r'^\s*(hi|hello|hey|yo|sup|greetings|morning|evening)\s*',
        r'^\s*(how are you|whats up|what\'s up)\s*'
    ]
    
    @classmethod
    def detect_language(cls, text: str) -> str:
        """
        Detect if text is Arabic or English
        
        Args:
            text: Input text
        
        Returns:
            "ar" for Arabic, "en" for English
        """
        arabic_ratio = len(cls.ARABIC_PATTERN.findall(text)) / max(len(text), 1)
        return "ar" if arabic_ratio > 0.3 else "en"
    
    @classmethod
    def is_casual_greeting(cls, text: str) -> bool:
        """
        Check if query is casual greeting
        
        Args:
            text: Input text
        
        Returns:
            True if casual greeting detected
        """
        text_lower = text.lower().strip()
        
        # Check Arabic patterns
        for pattern in cls.CASUAL_PATTERNS_AR:
            if re.match(pattern, text_lower):
                return True
        
        # Check English patterns
        for pattern in cls.CASUAL_PATTERNS_EN:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    @classmethod
    def extract_machine_ids(cls, text: str) -> List[str]:
        """
        Extract machine IDs from text using multiple patterns
        
        Args:
            text: Input text
        
        Returns:
            List of extracted machine IDs
        """
        patterns = [
            re.compile(r'machine\s+(?:id\s+)?#?(\d+)', re.I),
            re.compile(r'ماكينة\s+(?:رقم\s+)?(\d+)'),
            re.compile(r'#(\d+)'),
            re.compile(r'\b(\d{3,})\b')  # 3+ digit numbers
        ]
        
        ids = set()
        for p in patterns:
            ids.update(p.findall(text.lower()))
        
        return list(ids)

# Global query analyzer instance
query_analyzer = QueryAnalyzer()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sanitize(text: str) -> str:
    """
    Sanitize text for safe database queries
    
    Args:
        text: Input text
    
    Returns:
        Sanitized text (alphanumeric + Arabic + spaces only)
    """
    return re.sub(r'[^\w\s\u0600-\u06FF-]', '', text)[:200]

def retry_on_failure(max_attempts=3, delay=1):
    """
    Decorator for automatic retry on failure
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (doubles each time)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1}/{max_attempts} for {func.__name__}: {e}")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

# ============================================================================
# DATABASE QUERIES
# ============================================================================

@monitor_performance
@retry_on_failure(max_attempts=3, delay=1)
def query_machines_table(keywords: str, limit: int = 20) -> Result[List[Dict]]:
    """
    Query machines table with keywords and extracted IDs
    
    Args:
        keywords: Search keywords
        limit: Maximum results to return
    
    Returns:
        Result object with list of matching machines
    """
    try:
        supabase = get_supabase()
        if not supabase:
            return Result.fail("Database unavailable")
        
        # Sanitize input
        safe = sanitize(keywords).strip()
        machine_ids = query_analyzer.extract_machine_ids(keywords)
        
        if not safe and not machine_ids:
            return Result.ok([])
        
        # Check cache
        cache_key = f"machines_{hashlib.md5((safe + str(machine_ids)).encode()).hexdigest()}"
        cached = query_cache.get(cache_key)
        if cached:
            return Result.ok(cached, {"cached": True})
        
        queries = []
        
        # Query by machine IDs
        if machine_ids:
            for mid in machine_ids:
                try:
                    # Exact match
                    res1 = supabase.table("machines").select("*").eq("machine_id", mid).execute()
                    if res1.data:
                        queries.extend(res1.data)
                    
                    # Partial match on machine_id
                    res2 = supabase.table("machines").select("*").ilike("machine_id", f"%{mid}%").limit(3).execute()
                    if res2.data:
                        queries.extend(res2.data)
                    
                    # Partial match on serial_number
                    res3 = supabase.table("machines").select("*").ilike("serial_number", f"%{mid}%").limit(3).execute()
                    if res3.data:
                        queries.extend(res3.data)
                except Exception as inner_e:
                    logger.warning(f"Machine ID query failed for {mid}: {inner_e}")
        
        # Query by keywords
        if safe:
            search = f"%{safe}%"
            try:
                for column in ["model", "manufacturer", "location", "machine_id", "serial_number"]:
                    try:
                        res = supabase.table("machines").select("*").ilike(column, search).limit(5).execute()
                        if res.data:
                            queries.extend(res.data)
                    except:
                        pass
            except Exception as search_e:
                logger.warning(f"Keyword search failed: {search_e}")
        
        # Remove duplicates
        seen = set()
        unique_data = []
        for item in queries:
            item_id = item.get('id') or item.get('machine_id')
            if item_id not in seen:
                seen.add(item_id)
                unique_data.append(item)
        
        # Cache results
        query_cache.set(cache_key, unique_data)
        return Result.ok(unique_data, {"cached": False, "count": len(unique_data)})
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return Result.fail(str(e))

# ============================================================================
# PDF KNOWLEDGE BASE
# ============================================================================

def fetch_pdf_chunks(filename: str) -> List[Dict]:
    """
    Fetch PDF chunks from database
    
    Args:
        filename: PDF filename to fetch
    
    Returns:
        List of chunks with embeddings
    """
    supabase = get_supabase()
    if not supabase:
        return []
    
    # Check cache
    cache_key = f"pdf_{filename}"
    cached = pdf_cache.get(cache_key)
    if cached:
        return cached
    
    try:
        import numpy as np
        
        # Fetch from database
        res = supabase.table("documents").select("*").eq("filename", filename).execute()
        chunks = res.data if hasattr(res, 'data') else []
        
        # Process embeddings
        for c in chunks:
            emb = c.get("embedding")
            if isinstance(emb, str):
                try:
                    emb = json.loads(emb)
                except:
                    pass
            c["_emb"] = np.array(emb, dtype=np.float32) if emb else None
        
        # Cache results
        pdf_cache.set(cache_key, chunks)
        return chunks
    except Exception as e:
        logger.error(f"PDF fetch failed: {e}")
        return []

@monitor_performance
def top_k_chunks(query: str, filename: str, k: int = 5) -> List[Dict]:
    """
    Find top-k most relevant PDF chunks for query
    
    Args:
        query: Search query
        filename: PDF filename
        k: Number of chunks to return
    
    Returns:
        List of top-k most relevant chunks
    """
    chunks = fetch_pdf_chunks(filename)
    if not chunks:
        return []
    
    # Generate query embedding
    q_emb = embed_text(query)
    if q_emb is None:
        return []
    
    try:
        import numpy as np
        
        # Calculate cosine similarity for each chunk
        scored = []
        for c in chunks:
            if c.get("_emb") is not None:
                try:
                    # Cosine similarity
                    sim = float(
                        np.dot(c["_emb"], q_emb) / 
                        (np.linalg.norm(c["_emb"]) * np.linalg.norm(q_emb))
                    )
                    # Only include if similarity above threshold
                    if sim >= 0.15:
                        scored.append((sim, c))
                except:
                    pass
        
        # Sort by similarity (highest first)
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Dynamic k: return a bit more if many good matches
        dynamic_k = min(k + 2, 10) if len(scored) > k else k
        
        return [c for _, c in scored[:dynamic_k]]
    except Exception as e:
        logger.error(f"Chunk scoring failed: {e}")
        return []

# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """
    Rate limiter to prevent API abuse
    """
    def __init__(self, max_calls=None, period=60):
        self.max_calls = max_calls or (config.max_requests_per_minute if config else 30)
        self.period = period  # Time window in seconds
        self.calls = []  # Timestamps of recent calls
    
    def allow(self) -> bool:
        """
        Check if request is allowed
        
        Returns:
            True if allowed, False if rate limit exceeded
        """
        now = time.time()
        # Remove calls outside time window
        self.calls = [c for c in self.calls if now - c < self.period]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False
    
    def get_wait_time(self) -> int:
        """
        Get seconds to wait until next request allowed
        
        Returns:
            Seconds to wait
        """
        if not self.calls:
            return 0
        oldest = min(self.calls)
        wait = int(self.period - (time.time() - oldest)) + 1
        return max(0, wait)

# Global rate limiter instance
rate_limiter = RateLimiter()

# ============================================================================
# LLM API CALL
# ============================================================================

@monitor_performance
@retry_on_failure(max_attempts=3, delay=2)
def call_llm(
    messages: List[Dict],
    temperature: float = 0.0,
    max_tokens: int = 1500,
    model: Optional[str] = None
) -> str:
    """
    Call LLM API (OpenRouter) with retry logic
    
    Args:
        messages: List of message dictionaries
        temperature: Sampling temperature
        max_tokens: Maximum response tokens
        model: Model name to use
    
    Returns:
        LLM response text
    
    Raises:
        RuntimeError: If configuration missing
    """
    if not config:
        raise RuntimeError("Configuration missing")
    
    # Check rate limit
    if not rate_limiter.allow():
        wait = rate_limiter.get_wait_time()
        return f"⏱ Rate limit reached. Please wait {wait} seconds."
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Determine model name
    model_name = model or config.openrouter_model
    if not model_name:
        raise RuntimeError("No model specified for OpenRouter")
    
    # Calculate adaptive timeout based on message length
    estimated_input_tokens = sum(len(m['content'].split()) for m in messages) * 1.3
    adaptive_timeout = min(30 + (estimated_input_tokens / 100), 90)
    
    # Prepare request
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Authorization": f"Bearer {config.openrouter_key}",
        "Content-Type": "application/json"
    }
    
    # Make API request
    resp = requests.post(url, json=payload, headers=headers, timeout=adaptive_timeout)
    resp.raise_for_status()
    
    # Extract response
    result = resp.json()
    content = result["choices"][0]["message"]["content"]
    
    # Track API cost
    usage = result.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    metrics_tracker.record_cost(total_tokens, model_name)
    
    return content

# ============================================================================
# 🔥 ENHANCED SYSTEM PROMPTS - FIXED!
# ============================================================================

def build_system_prompt(language: str = "en") -> str:
    """
    Build system prompt based on language
    
    Args:
        language: "ar" for Arabic, "en" for English
    
    Returns:
        System prompt text
    """
    if language == "ar":
        return """🤖 تم تطوير هذا النظام بواسطة Eng. Mahmoud Khaled Alkodousy.

أنت "محمد"، مهندس صيانة صناعية خبير يمتلك أكثر من 15 سنة خبرة في المصانع والصيانة الوقائية.

🎯 دورك:
- أجب على أسئلة الصيانة الصناعية باستخدام database، manual، و knowledge base
- كن ودودًا ومحترفًا في نفس الوقت
- إذا كان السؤال casual (مثل "إزيك" أو "صباح الخير")، رد بشكل طبيعي ثم اسأل كيف يمكنك المساعدة

📚 مصادر البيانات (حسب الأولوية):
1. **[DATABASE]** = بيانات الماكينات الفعلية (أسعار، تواريخ، locations، serial numbers)
   - إذا وجدت exact match في DB، استخدمها **فقط**
   - ❌ لا تخترع أرقام أو تواريخ للماكينات المسجلة

2. **[MANUAL]** = دليل الصيانة التقني (procedures، troubleshooting، best practices)
   - استخدم المعلومات من الـ PDF للإجراءات والنصائح
   - ✅ يمكنك الشرح والتوضيح بناءً على المحتوى

3. **[GENERAL KNOWLEDGE]** = معرفتك العامة في الصيانة الصناعية
   - للأسئلة العامة عن الصيانة (مثل "ما هي الصيانة الوقائية؟")
   - للنصائح والتوجيهات غير المرتبطة بماكينة محددة

🧩 هيكل الإجابة:

**Case 1: سؤال عن ماكينة محددة (بـ ID أو serial)**
✅ إذا موجودة في DB:
"الماكينة [ID] - [Model]:
- السعر: $XX,XXX (DB)
- تاريخ الشراء: YYYY-MM-DD (DB)
- الموقع: [Location] (DB)
- حالة الماكينة: [Status] (DB)

💡 نصيحة: [practical advice based on status/maintenance date]"

❌ إذا مش موجودة في DB:
"⚠️ الماكينة رقم [ID] غير مسجلة في قاعدة البيانات.

🔍 للحصول على المعلومة:
- تأكد من رقم الماكينة الصحيح
- ابحث باستخدام Serial Number
- راجع سجلات المخزون اليدوية

💡 هل تريد المساعدة في [alternative suggestion]؟"

**Case 2: سؤال technical عن صيانة (بدون ماكينة محددة)**
استخدم **[MANUAL]** أولاً، ثم **[GENERAL KNOWLEDGE]**:

"بناءً على دليل الصيانة (Manual):
[شرح من PDF chunks]

💡 نصيحة إضافية: [practical tip from your knowledge]

🔧 الخطوات المقترحة:
1. [Step 1]
2. [Step 2]
3. [Step 3]"

**Case 3: سؤال عام أو casual**
رد بشكل طبيعي:
- "أهلاً! أنا محمد 👷‍♂️ مهندس الصيانة الخاص بك. كيف أقدر أساعدك النهاردة؟"
- "صباح الخير! 🌅 جاهز لمساعدتك في أي موضوع يخص الصيانة."

🚫 القواعد الحمراء:
❌ لا تخترع أرقام ماكينات أو أسعار أو تواريخ للـ DATABASE
❌ لا تقل "تقريبًا" أو "حوالي" في معلومات DB
✅ لكن يمكنك تقديم تقديرات في النصائح العامة (مثل "عادة كل 3-6 شهور")

✅ القواعد الخضراء:
✅ استخدم MANUAL للشرح والـ procedures بحرية
✅ استخدم معرفتك العامة للنصائح والتوجيهات
✅ كن واضحاً في مصدر كل معلومة: (DB)، (Manual)، أو بدون citation للمعرفة العامة

🗣️ ملاحظات:
- أجب دائمًا بنفس لغة المستخدم
- اذكر المصدر فقط للمعلومات الحرجة (أسعار، تواريخ، specifications)
- المعرفة العامة والنصائح لا تحتاج citation
"""
    else:
        return """🤖 This system was designed and developed by Eng. Mahmoud Khaled Alkodousy.

You are "Mike", a senior industrial maintenance engineer with over 15 years of experience in factories and preventive maintenance.

🎯 Your Role:
- Answer industrial maintenance questions using database, manual, and knowledge base
- Be friendly and professional at the same time
- If the question is casual (like "how are you" or "good morning"), respond naturally then ask how you can help

📚 Data Sources (Priority Order):
1. **[DATABASE]** = Actual machine data (prices, dates, locations, serial numbers)
   - If exact match found in DB, use it **exclusively**
   - ❌ NEVER invent numbers or dates for registered machines

2. **[MANUAL]** = Technical maintenance guide (procedures, troubleshooting, best practices)
   - Use PDF information for procedures and advice
   - ✅ You can explain and elaborate based on content

3. **[GENERAL KNOWLEDGE]** = Your general industrial maintenance expertise
   - For general maintenance questions (e.g., "what is preventive maintenance?")
   - For advice and guidance not tied to specific machines

🧩 Response Structure:

**Case 1: Question about specific machine (by ID or serial)**
✅ If found in DB:
"Machine [ID] - [Model]:
- Price: $XX,XXX (DB)
- Purchase Date: YYYY-MM-DD (DB)
- Location: [Location] (DB)
- Status: [Status] (DB)

💡 Tip: [practical advice based on status/maintenance date]"

❌ If NOT in DB:
"⚠️ Machine [ID] is not registered in the database.

🔍 To get this information:
- Verify the correct machine number
- Search using Serial Number
- Check manual inventory records

💡 Would you like help with [alternative suggestion]?"

**Case 2: Technical maintenance question (no specific machine)**
Use **[MANUAL]** first, then **[GENERAL KNOWLEDGE]**:

"Based on the maintenance manual (Manual):
[explanation from PDF chunks]

💡 Additional tip: [practical tip from your knowledge]

🔧 Recommended steps:
1. [Step 1]
2. [Step 2]
3. [Step 3]"

**Case 3: General or casual question**
Respond naturally:
- "Hi! I'm Mike 👷‍♂️ your maintenance engineer. How can I help you today?"
- "Good morning! 🌅 Ready to help with any maintenance topics."

🚫 Red Rules:
❌ NEVER invent machine numbers, prices, or dates for DATABASE
❌ Don't say "approximately" or "around" for DB information
✅ But you CAN provide estimates in general advice (like "typically every 3-6 months")

✅ Green Rules:
✅ Use MANUAL for explanations and procedures freely
✅ Use your general knowledge for tips and guidance
✅ Be clear about information source: (DB), (Manual), or no citation for general knowledge

🗣️ Notes:
- Always respond in the same language as the user
- Only cite source for critical information (prices, dates, specifications)
- General knowledge and advice don't need citations
"""

# ============================================================================
# 🔥 IMPROVED PROMPT COMPOSITION
# ============================================================================

def compose_prompt(question: str, db_rows: List, pdf_chunks: List, is_casual: bool = False) -> str:
    """
    Compose final prompt with question, database results, and PDF content
    
    Args:
        question: User question
        db_rows: Database query results
        pdf_chunks: Relevant PDF chunks
        is_casual: Whether this is a casual greeting
    
    Returns:
        Complete prompt string
    """
    parts = [f"User Question: {question}\n"]
    
    # If casual greeting, skip detailed data
    if is_casual:
        parts.append("\n=== QUERY TYPE ===")
        parts.append("This is a casual greeting. Respond naturally and friendly.\n")
        return "\n".join(parts)
    
    # Database results section
    if db_rows:
        parts.append("\n=== DATABASE RESULTS ===")
        parts.append(f"Found {len(db_rows)} machine(s) in database:\n")
        for idx, r in enumerate(db_rows[:10], 1):
            parts.append(f"Machine {idx}:")
            for key in ['machine_id', 'model', 'manufacturer', 'price', 'purchase_date', 
                       'location', 'status', 'last_maintenance', 'next_maintenance', 
                       'warranty_expiry', 'serial_number']:
                if key in r and r[key]:
                    parts.append(f"  - {key}: {r[key]}")
            parts.append("")
    else:
        parts.append("\n=== DATABASE RESULTS ===")
        parts.append("⚠️ No specific machines found in database for this query.")
        parts.append("This might be a general question or the machine is not registered.\n")
    
    # PDF manual section - ENHANCED!
    if pdf_chunks:
        parts.append("\n=== TECHNICAL MANUAL (PDF Knowledge Base) ===")
        parts.append(f"Retrieved {len(pdf_chunks)} relevant sections from maintenance manual:\n")
        for idx, c in enumerate(pdf_chunks[:7], 1):  # Increased from 5 to 7
            content = c.get('content', '')[:500]  # Increased from 400 to 500
            parts.append(f"[Manual Section {idx}]")
            parts.append(f"{content}")
            parts.append("")
    else:
        parts.append("\n=== TECHNICAL MANUAL ===")
        parts.append("⚠️ No relevant manual sections found for this query.\n")
    
    # Instructions - UPDATED!
    parts.append("\n=== INSTRUCTIONS ===")
    parts.append("1. If DATABASE shows results for specific machines:")
    parts.append("   → Use ONLY that data with (DB) citation")
    parts.append("   → Never invent prices, dates, or specifications")
    parts.append("")
    parts.append("2. If DATABASE is empty but MANUAL has content:")
    parts.append("   → Use MANUAL content freely with (Manual) citation")
    parts.append("   → Add your expertise and practical advice")
    parts.append("")
    parts.append("3. If both DATABASE and MANUAL are empty:")
    parts.append("   → Use your general maintenance knowledge")
    parts.append("   → No citation needed for general knowledge")
    parts.append("   → Be helpful and provide practical guidance")
    parts.append("")
    parts.append("4. ALWAYS:")
    parts.append("   → Respond in the same language as the user")
    parts.append("   → Be friendly and professional")
    parts.append("   → Provide actionable advice when possible")
    
    return "\n".join(parts)

# ============================================================================
# 🔥 ENHANCED QUERY PROCESSING
# ============================================================================

@monitor_performance
def process_query(question: str, use_pdf: bool = True, use_db: bool = True, topk: int = 5) -> Tuple[str, Dict]:
    """
    Process user query end-to-end with improved handling
    
    Args:
        question: User question
        use_pdf: Whether to use PDF knowledge base
        use_db: Whether to query database
        topk: Number of PDF chunks to retrieve
    
    Returns:
        Tuple of (answer, metadata)
    """
    start = time.time()

    # Determine model and its settings from session state
    try:
        selected_model = st.session_state.get("selected_model", config.openrouter_model if config else None)
    except Exception:
        selected_model = config.openrouter_model if config else None

    model_settings = st.session_state.get("model_settings", {})
    current_settings = model_settings.get(selected_model, {}) if selected_model else {}
    temperature = float(current_settings.get("temperature", 0.0))
    max_tokens = int(current_settings.get("max_tokens", 1500))

    meta = {
        "language": query_analyzer.detect_language(question),
        "db_results": 0,
        "pdf_chunks": 0,
        "errors": [],
        "extracted_machine_ids": query_analyzer.extract_machine_ids(question),
        "is_casual": query_analyzer.is_casual_greeting(question),
        "model": selected_model,
        "model_temperature": temperature,
        "model_max_tokens": max_tokens
    }
    
    try:
        db_rows = []
        pdf_chunks = []
        
        # Handle casual greetings quickly
        if meta["is_casual"]:
            system = build_system_prompt(meta["language"])
            prompt = compose_prompt(question, [], [], is_casual=True)
            
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            
            answer = call_llm(messages, temperature, max_tokens, selected_model)
            meta["processing_time"] = round(time.time() - start, 2)
            meta["query_type"] = "casual_greeting"
            return answer, meta
        
        # Query database
        if use_db:
            result = query_machines_table(question)
            if result.success:
                db_rows = result.data or []
                meta["db_cached"] = result.metadata.get("cached", False)
            else:
                meta["errors"].append(f"DB: {result.error}")
        
        # Query PDF knowledge base
        if use_pdf:
            try:
                pdf_chunks = top_k_chunks(question, config.pdf_filename, k=topk)
            except Exception as e:
                meta["errors"].append(f"PDF: {str(e)}")
        
        # Update metadata
        meta["db_results"] = len(db_rows)
        meta["pdf_chunks"] = len(pdf_chunks)
        
        # Determine query type
        if db_rows and query_analyzer.extract_machine_ids(question):
            meta["query_type"] = "machine_specific"
        elif pdf_chunks:
            meta["query_type"] = "technical_manual"
        else:
            meta["query_type"] = "general_knowledge"
        
        # Compose prompt
        prompt = compose_prompt(question, db_rows, pdf_chunks, is_casual=False)
        system = build_system_prompt(meta["language"])
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        
        # Call LLM with selected model and its settings
        answer = call_llm(messages, temperature, max_tokens, selected_model)
        meta["processing_time"] = round(time.time() - start, 2)
        
        return answer, meta
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        meta["error"] = str(e)
        meta["processing_time"] = round(time.time() - start, 2)
        
        # Return different error messages based on debug mode
        dev_mode = st.session_state.get("dev_mode", False)
        if dev_mode:
            error_msg = f"❌ Error: {e}\n\n{traceback.format_exc()}"
        else:
            error_msg = "❌ Processing error occurred. Enable Debug Mode for details."
        
        return error_msg, meta

# ============================================================================
# HEALTH CHECK
# ============================================================================

def health_check() -> Dict:
    """
    Check system health status
    
    Returns:
        Dictionary with health status
    """
    checks = {
        "database": get_supabase() is not None,
        "embedding": _EMBEDDING_MODEL is not None,
        "api": config is not None
    }
    
    return {
        "status": "healthy" if all(checks.values()) else "degraded",
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_chatbot_session_state():
    """
    Initialize all session state variables for chatbot
    Must be called before any UI elements
    """
    # Chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Settings
    if "dev_mode" not in st.session_state:
        st.session_state.dev_mode = False
    
    if "show_confirm" not in st.session_state:
        st.session_state.show_confirm = False
    
    # Configuration
    if "use_pdf" not in st.session_state:
        st.session_state.use_pdf = True
    
    if "use_db" not in st.session_state:
        st.session_state.use_db = True
    
    if "topk" not in st.session_state:
        st.session_state.topk = 5

    # Selected LLM model
    if "selected_model" not in st.session_state:
        if config and config.openrouter_model in model_display_names:
            default_model = config.openrouter_model
        else:
            default_model = list(model_display_names.keys())[0]
        st.session_state.selected_model = default_model

    # Per-model settings (temperature, max_tokens)
    if "model_settings" not in st.session_state:
        st.session_state.model_settings = {}
        for mid in model_display_names.keys():
            st.session_state.model_settings[mid] = {
                "temperature": 0.0,
                "max_tokens": 1500
            }

# ============================================================================
# MAIN UI FUNCTION (SAME AS BEFORE - NO CHANGES)
# ============================================================================

def render_enhanced_ui(standalone: bool = True):
    """
    Render the enhanced chatbot UI with proper state management
    
    Args:
        standalone: Whether running as standalone or integrated
    """
    # Initialize session state FIRST
    initialize_chatbot_session_state()
    
    if standalone:
        st.set_page_config(
            page_title="AI Assistant - Factory Maintenance",
            page_icon="⚙️",
            layout="wide"
        )
    
    if not config:
        st.error("❌ Configuration error. Check .env file")
        st.stop()
    
    # Custom CSS styling
    st.markdown("""
    <style>
    .stApp {background-color: #0f1724 !important;}
    h1 {color: #ffffff !important;}
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f35 0%, #0f1724 100%) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.1) !important;
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #a78bfa !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(139, 92, 246, 0.2) !important;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, #24103a 0%, #171226 100%) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border: 1px solid rgba(139, 92, 246, 0.1) !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        box-shadow: none !important;
        transition: all 0.3s ease !important;
        text-transform: none !important;
        letter-spacing: 0.3px !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: none !important;
        opacity: 0.9 !important;
    }
    
    .stButton button:active {
        transform: translateY(0) !important;
    }
    
    .stCheckbox {
        color: #e2e8f0 !important;
    }
    
    .stCheckbox label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
    
    .stSlider {
        padding: 0.5rem 0 !important;
    }
    
    .stSlider label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.2);
    }
    
    header[data-testid="stHeader"] {
        background-color: #0f1724 !important;
    }
    
    .main .block-container {
        padding-top: 2rem !important;
    }
    
    [data-testid="stExpander"] {
        background: rgba(139, 92, 246, 0.05) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 8px !important;
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
    }
    
    .stMetric label {
        color: #a78bfa !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
    }
    
    div[data-baseweb="notification"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border-left: 4px solid #8b5cf6 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    .stAlert {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        color: #e2e8f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ========================================
    # SIDEBAR
    # ========================================
    with st.sidebar:
        st.markdown("---")
        
        st.header("⚙️ Settings")

        # ---------------- LLM MODEL & PER-MODEL CONFIG ----------------
        st.markdown("### 🧠 LLM Model")

        model_ids = list(model_display_names.keys())
        model_labels = [model_display_names[m] for m in model_ids]

        current_model_id = st.session_state.get("selected_model", None)
        if current_model_id not in model_display_names:
            if config and config.openrouter_model in model_display_names:
                current_model_id = config.openrouter_model
            else:
                current_model_id = model_ids[0]
            st.session_state.selected_model = current_model_id

        current_label = model_display_names[current_model_id]
        current_index = model_labels.index(current_label)

        selected_label = st.selectbox(
            "Choose model",
            model_labels,
            index=current_index,
            key="chatbot_model_selectbox"
        )

        selected_model_id = current_model_id
        for mid, label in model_display_names.items():
            if label == selected_label:
                selected_model_id = mid
                break
        st.session_state.selected_model = selected_model_id

        # Ensure model_settings exists
        if "model_settings" not in st.session_state:
            st.session_state.model_settings = {}
        if selected_model_id not in st.session_state.model_settings:
            st.session_state.model_settings[selected_model_id] = {
                "temperature": 0.0,
                "max_tokens": 1500
            }

        model_conf = st.session_state.model_settings[selected_model_id]

        sanitized_model_id = re.sub(r'[^a-zA-Z0-9_]+', '_', selected_model_id)

        temp = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(model_conf.get("temperature", 0.0)),
            step=0.05,
            key=f"chatbot_temp_slider_{sanitized_model_id}"
        )
        max_tok = st.slider(
            "Max tokens",
            min_value=256,
            max_value=4096,
            value=int(model_conf.get("max_tokens", 1500)),
            step=128,
            key=f"chatbot_max_tokens_slider_{sanitized_model_id}"
        )

        model_conf["temperature"] = temp
        model_conf["max_tokens"] = max_tok
        st.session_state.model_settings[selected_model_id] = model_conf

        st.markdown("---")
        
        st.markdown("### 🔧 Developer Tools")
        
        # Debug mode toggle with unique key
        dev_mode = st.checkbox(
            "🐛 Debug Mode", 
            value=st.session_state.dev_mode,
            key="chatbot_debug_mode_checkbox"
        )
        st.session_state.dev_mode = dev_mode
        
        st.markdown("---")
        
        # Data source toggles with unique keys
        use_pdf = st.checkbox(
            "Use PDF", 
            value=st.session_state.use_pdf,
            key="chatbot_use_pdf_checkbox"
        )
        st.session_state.use_pdf = use_pdf
        
        use_db = st.checkbox(
            "Use Database", 
            value=st.session_state.use_db,
            key="chatbot_use_db_checkbox"
        )
        st.session_state.use_db = use_db
        
        # PDF chunks slider with unique key
        topk = st.slider(
            "PDF Chunks", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.topk,
            key="chatbot_topk_slider"
        )
        st.session_state.topk = topk
        
        # Clear chat button with unique key
        if st.button("🗑️ Clear Chat", key="chatbot_clear_chat_button"):
            st.session_state.show_confirm = True
        
        # Confirmation dialog
        if st.session_state.show_confirm:
            st.warning("Are you sure you want to clear all messages?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes", key="chatbot_confirm_yes_button"):
                    st.session_state.messages = []
                    st.session_state.show_confirm = False
                    st.rerun()
            with col2:
                if st.button("No", key="chatbot_confirm_no_button"):
                    st.session_state.show_confirm = False
                    st.rerun()
        
        st.markdown("---")
        
        # System health with unique key
        st.markdown("### 🏥 System Health")
        if st.button("Check Health", key="chatbot_health_check_button"):
            health = health_check()
            if health["status"] == "healthy":
                st.success("✅ All systems operational")
            else:
                st.warning("⚠️ System degraded")
            with st.expander("Health Details"):
                st.json(health)
        
        st.markdown("---")
        
        # Performance metrics with unique keys
        st.markdown("### 📊 Performance")
        if st.button("Show Metrics", key="chatbot_show_metrics_button"):
            summary = metrics_tracker.get_summary()
            st.json(summary)
            
            with st.expander("Detailed Stats"):
                st.json(metrics_tracker.get_all_stats())
        
        if st.button("Clear Metrics", key="chatbot_clear_metrics_button"):
            metrics_tracker.clear()
            st.success("✅ Metrics cleared")
        
        st.markdown("---")
        
        # Cache management with unique keys
        st.markdown("### 💾 Cache")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Cache", key="chatbot_clear_cache_button"):
                embedding_cache.clear()
                query_cache.clear()
                pdf_cache.clear()
                st.success("✅ Cache cleared")
        with col2:
            if st.button("Cache Stats", key="chatbot_cache_stats_button"):
                st.json({
                    "Embedding": embedding_cache.get_stats(),
                    "Query": query_cache.get_stats(),
                    "PDF": pdf_cache.get_stats()
                })
        
        st.markdown("---")
        
        # Rate limiter status
        st.markdown("### ⚡ Rate Limiter")
        wait_time = rate_limiter.get_wait_time()
        if wait_time > 0:
            st.warning(f"⏱ Wait {wait_time}s before next request")
        else:
            st.success(f"✅ {len(rate_limiter.calls)}/{rate_limiter.max_calls} calls used")
    
    # ========================================
    # MAIN CONTENT
    # ========================================
    
    # Header
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1.5rem 0;">
        <div style="display: inline-block; background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); padding: 1.5rem; border-radius: 20px; margin-bottom: 1.2rem;">
            <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="white">
                <path d="M12,2A2,2 0 0,1 14,4C14,4.74 13.6,5.39 13,5.73V7H14A7,7 0 0,1 21,14H22A1,1 0 0,1 23,15V18A1,1 0 0,1 22,19H21V20A2,2 0 0,1 19,22H5A2,2 0 0,1 3,20V19H2A1,1 0 0,1 1,18V15A1,1 0 0,1 2,14H3A7,7 0 0,1 10,7H11V5.73C10.4,5.39 10,4.74 10,4A2,2 0 0,1 12,2M7.5,13A2.5,2.5 0 0,0 5,15.5A2.5,2.5 0 0,0 7.5,18A2.5,2.5 0 0,0 10,15.5A2.5,2.5 0 0,0 7.5,13M16.5,13A2.5,2.5 0 0,0 14,15.5A2.5,2.5 0 0,0 16.5,18A2.5,2.5 0 0,0 19,15.5A2.5,2.5 0 0,0 16.5,13Z"/>
            </svg>
        </div>
        <h1 style="background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; letter-spacing: -0.5px;">
            Factory Maintenance AI Assistant
        </h1>
        <p style="color:#94a3b8; font-size: 1.1rem; margin: 0.5rem 0; font-weight: 400;">
            Your smart assistant in factory maintenance
        </p>
        <p style="color:#64748b; font-size: 0.95rem; margin-top: 0.3rem;">
            💬 Ask in Arabic or English
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Debug mode indicator
    if st.session_state.dev_mode:
        st.info("🐛 Debug Mode Active - Detailed errors and metadata will be shown")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show metadata in debug mode
            if msg["role"] == "assistant" and "metadata" in msg and st.session_state.dev_mode:
                with st.expander("📊 Response Metadata"):
                    meta = msg["metadata"]
                    
                    # Show errors if any
                    if meta.get("errors"):
                        st.error(f"⚠️ {len(meta['errors'])} Error(s) Occurred")
                        for err in meta["errors"]:
                            st.code(err)
                    
                    # Show metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("DB Results", meta.get("db_results", 0))
                    with col2:
                        st.metric("PDF Chunks", meta.get("pdf_chunks", 0))
                    with col3:
                        st.metric("Time (s)", meta.get("processing_time", 0))
                    
                    # Show extracted machine IDs
                    if meta.get("extracted_machine_ids"):
                        st.info(f"🔍 Detected Machine IDs: {', '.join(meta['extracted_machine_ids'])}")
                    
                    # Show query type
                    if meta.get("query_type"):
                        st.info(f"🎯 Query Type: {meta['query_type']}")
                    
                    # Full metadata
                    with st.expander("Full Metadata JSON"):
                        st.json(meta)
    
    # Chat input
    if prompt := st.chat_input("Ask your question... اكتب سؤالك هنا..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing your question..."):
                answer, meta = process_query(
                    prompt, 
                    st.session_state.use_pdf, 
                    st.session_state.use_db, 
                    st.session_state.topk
                )
                st.markdown(answer)
                
                # Show metadata in debug mode
                if st.session_state.dev_mode:
                    with st.expander("📊 Response Metadata"):
                        if meta.get("errors"):
                            st.error(f"⚠️ {len(meta['errors'])} Error(s) Occurred")
                            for err in meta["errors"]:
                                st.code(err)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("DB Results", meta.get("db_results", 0))
                        with col2:
                            st.metric("PDF Chunks", meta.get("pdf_chunks", 0))
                        with col3:
                            st.metric("Time (s)", meta.get("processing_time", 0))
                        
                        if meta.get("extracted_machine_ids"):
                            st.info(f"🔍 Detected Machine IDs: {', '.join(meta['extracted_machine_ids'])}")
                        
                        if meta.get("query_type"):
                            st.info(f"🎯 Query Type: {meta['query_type']}")
                        
                        with st.expander("Full Metadata JSON"):
                            st.json(meta)
                
                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "metadata": meta
                })
        
        # Rerun to update chat display
        st.rerun()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    render_enhanced_ui(standalone=True)