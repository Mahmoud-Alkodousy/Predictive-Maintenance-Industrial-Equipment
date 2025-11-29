# src/config.py
"""
Central configuration for Predictive Maintenance Project.
Single-file config: paths, constants, enums, and helpers.
"""

from enum import Enum
from pathlib import Path
import os

# ---------------------
# Project root detection
# ---------------------
def find_project_root():
    """
    Try to find project root where data/models exists.
    Returns a Path.
    """
    current = Path(__file__).parent.absolute()
    parent = current.parent
    if (parent / "data" / "models").exists():
        return parent
    if (current / "data" / "models").exists():
        return current
    grandparent = parent.parent
    if (grandparent / "data" / "models").exists():
        return grandparent
    return parent


PROJECT_ROOT = Path(__file__).parent.absolute()

# ---------------------
# Directory structure
# ---------------------
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GRAPHS_DIR = DATA_DIR / "graphs"

for d in (DATA_DIR, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, GRAPHS_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ---------------------
# Model files mapping
# ---------------------
MODEL_FILES = {
    "LR_24h": MODELS_DIR / "model_24h_LR.joblib",
    "RF_24h": MODELS_DIR / "model_24h_RF_fast.joblib",
    "LR_48h": MODELS_DIR / "model_48h_LR.joblib",
    "RF_48h": MODELS_DIR / "model_48h_RF_fast.joblib",
}

# Image inspection model paths
VGG_MODEL_PATH = MODELS_DIR / "industrial_defect_detection_model.h5"
YOLO_WEIGHTS_PATH = MODELS_DIR / "yolo_best.pt"
YOLOV5_DIR = PROJECT_ROOT / "yolov5"

# ---------------------
# App defaults & constants
# ---------------------
DEFAULT_THRESHOLD = 0.5
RANDOM_STATE = 42

# ---------------------
# Feature engineering params
# ---------------------
SENSORS = ["volt", "rotate", "pressure", "vibration"]
LAGS = [1, 3, 6, 12, 24]
ROLL_MEANS = [3, 6, 12, 24, 48]
ROLL_STDS = [6, 24, 48]
SLOPES_K = [3, 6, 12]

# ---------------------
# Enums and AppConstants
# ---------------------
class CacheType(Enum):
    EMBEDDING = "embedding"
    QUERY = "query"
    PDF = "pdf"


class IntentType(Enum):
    PRICE = "price_inquiry"
    MAINTENANCE = "maintenance_inquiry"
    LUBRICATION = "lubrication_inquiry"
    TROUBLESHOOTING = "troubleshooting"
    LIST = "list_query"
    GENERAL = "general_inquiry"


class AppConstants:
    # Cache settings
    EMBEDDING_CACHE_SIZE = 500
    EMBEDDING_CACHE_TTL = 7200
    QUERY_CACHE_SIZE = 200
    QUERY_CACHE_TTL = 1800
    PDF_CACHE_SIZE = 100
    PDF_CACHE_TTL = 3600

    # Rate limiting
    MAX_API_CALLS = 30
    RATE_LIMIT_PERIOD = 60

    # Retry settings
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 2
    REQUEST_TIMEOUT = 60

    # Similarity threshold
    MIN_SIMILARITY = 0.2

    # Query settings
    DEFAULT_TOP_K = 5
    MAX_TOP_K = 20
    DB_QUERY_LIMIT = 20


# ---------------------
# Helper functions
# ---------------------
def get_available_models():
    """Return dict of model name -> Path for models that exist."""
    return {name: path for name, path in MODEL_FILES.items() if path.exists()}


def ensure_model_paths_exist():
    """Optional helper: returns dict of booleans for image model existence."""
    return {
        "vgg_exists": VGG_MODEL_PATH.exists(),
        "yolo_exists": YOLO_WEIGHTS_PATH.exists(),
        "yolov5_dir_exists": YOLOV5_DIR.exists(),
    }


# ---------------------
# Export list
# ---------------------
__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "GRAPHS_DIR",
    "MODEL_FILES",
    "VGG_MODEL_PATH",
    "YOLO_WEIGHTS_PATH",
    "YOLOV5_DIR",
    "DEFAULT_THRESHOLD",
    "RANDOM_STATE",
    "SENSORS",
    "LAGS",
    "ROLL_MEANS",
    "ROLL_STDS",
    "SLOPES_K",
    "CacheType",
    "IntentType",
    "AppConstants",
    "get_available_models",
    "ensure_model_paths_exist",
]
