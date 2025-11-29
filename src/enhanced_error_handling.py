"""
Enhanced Error Handling for ModelExplainer Import
Replace the import section in streamlit_app.py with this code

Developer: Eng. Mahmoud Khalid Alkodousy
"""

import streamlit as st
import sys
import traceback
from pathlib import Path

# ============================================
# ENHANCED IMPORT WITH DETAILED ERROR LOGGING
# ============================================

def try_import_explainer():
    """
    Attempt to import ModelExplainer with detailed error logging
    
    Returns:
        tuple: (ModelExplainer class or None, list of import errors/messages)
    """
    import_errors = []
    
    # 1️⃣ Check if explainability.py file exists in current directory
    explainability_path = Path("explainability.py")
    if not explainability_path.exists():
        import_errors.append({
            "type": "FILE_NOT_FOUND",
            "message": "explainability.py file not found in current directory",
            "path": str(explainability_path.absolute())
        })
        return None, import_errors
    
    # 2️⃣ Verify SHAP library installation
    try:
        import shap
        shap_version = shap.__version__
    except ImportError as e:
        import_errors.append({
            "type": "DEPENDENCY_MISSING",
            "message": "SHAP library is not installed",
            "error": str(e),
            "solution": "pip install shap"
        })
        return None, import_errors
    
    # 3️⃣ Attempt to import ModelExplainer class
    try:
        from explainability import ModelExplainer
        import_errors.append({
            "type": "SUCCESS",
            "message": "✅ ModelExplainer loaded successfully",
            "shap_version": shap_version
        })
        return ModelExplainer, import_errors
        
    except ImportError as e:
        import_errors.append({
            "type": "IMPORT_ERROR",
            "message": "Failed to import ModelExplainer from explainability.py",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        return None, import_errors
        
    except Exception as e:
        import_errors.append({
            "type": "UNKNOWN_ERROR",
            "message": "Unexpected error during import",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        return None, import_errors


def display_import_diagnostics(import_errors):
    """
    Display detailed import error diagnostics in Streamlit sidebar
    
    Args:
        import_errors (list): List of error dictionaries from try_import_explainer()
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Explainer Diagnostics")
    
    for error in import_errors:
        # Success case - show confirmation message
        if error["type"] == "SUCCESS":
            st.sidebar.success(error["message"])
            st.sidebar.info(f"SHAP Version: {error.get('shap_version', 'Unknown')}")
        
        # File not found - show expected path
        elif error["type"] == "FILE_NOT_FOUND":
            st.sidebar.error("❌ " + error["message"])
            with st.sidebar.expander("📂 File Path"):
                st.code(error["path"])
        
        # Missing dependency - show installation instructions
        elif error["type"] == "DEPENDENCY_MISSING":
            st.sidebar.error("❌ " + error["message"])
            with st.sidebar.expander("💡 Solution"):
                st.code(error["solution"])
                st.markdown("""
                **Installation Steps:**
```bash
                pip install shap
                # or specify version
                pip install shap==0.42.1
```
                """)
        
        # Import error - show error details and traceback
        elif error["type"] == "IMPORT_ERROR":
            st.sidebar.error("❌ " + error["message"])
            with st.sidebar.expander("🐛 Error Details"):
                st.code(error["error"])
                st.code(error["traceback"])
        
        # Unknown error - show full traceback
        elif error["type"] == "UNKNOWN_ERROR":
            st.sidebar.error("❌ " + error["message"])
            with st.sidebar.expander("🐛 Full Traceback"):
                st.code(error["traceback"])


# ============================================
# EXECUTE IMPORT AND STORE RESULTS IN SESSION STATE
# ============================================

# Perform the import attempt
ModelExplainer, import_errors = try_import_explainer()

# Store results in Streamlit session state for access across entire app
if 'explainer_import_errors' not in st.session_state:
    st.session_state.explainer_import_errors = import_errors
    st.session_state.ModelExplainer = ModelExplainer


# ============================================
# ENHANCED EXPLAINER LOADING FUNCTION
# ============================================

@st.cache_resource
def load_explainers(_models):
    """
    Load SHAP explainers for all models with comprehensive error handling
    
    Args:
        _models (dict): Dictionary of model_name -> model object pairs
        
    Returns:
        dict or None: Dictionary of model_name -> explainer object pairs, or None if loading fails
    """
    explainers = {}
    loading_errors = []

    # ✅ VALIDATION 1: Check if ModelExplainer is available
    if ModelExplainer is None:
        st.warning("⚠️ ModelExplainer is not available - check errors in Sidebar")
        display_import_diagnostics(st.session_state.explainer_import_errors)
        return None

    # ✅ VALIDATION 2: Verify processed data files exist
    try:
        from config import PROCESSED_DATA_DIR
        telemetry_path = PROCESSED_DATA_DIR / "telemetry.csv"
        
        if not telemetry_path.exists():
            st.warning(f"⚠️ telemetry.csv file not found at: {PROCESSED_DATA_DIR}")
            st.info("""
            **Required Solution:**
            1. Run `data_preprocessing.py` first
            2. Or place telemetry.csv file in the correct directory
            
            **Expected Directory:**
```
            {path}
```
            """.format(path=str(PROCESSED_DATA_DIR.absolute())))
            return None
            
    except ImportError as e:
        st.error(f"❌ Failed to import config.py: {e}")
        return None

    # ✅ VALIDATION 3: Load and validate telemetry data
    try:
        import pandas as pd
        telemetry = pd.read_csv(telemetry_path, parse_dates=["datetime"])
        st.info(f"✅ Loaded {len(telemetry)} rows from telemetry.csv")
        
        # Extract feature columns (exclude metadata and target columns)
        exclude_cols = {"datetime", "machineID", "will_fail_in_24h", "will_fail_in_48h"}
        features_cols = [c for c in telemetry.columns if c not in exclude_cols]
        
        if not features_cols:
            st.error("❌ No feature columns found in telemetry.csv")
            return None
        
        # Sample data for explainer training (using first 1000 rows for efficiency)
        X_sample = telemetry[features_cols].head(1000)
        st.success(f"✅ Selected {len(X_sample)} samples for training ({len(features_cols)} features)")
        
    except Exception as e:
        st.error(f"❌ Error reading telemetry.csv: {e}")
        with st.expander("🐛 Error Details"):
            st.code(traceback.format_exc())
        return None

    # ✅ STAGE 4: Fit SHAP explainers for each model
    progress_text = "⏳ Loading explainers..."
    progress_bar = st.progress(0, text=progress_text)

    total = max(1, len(_models))
    
    for idx, (model_name, model) in enumerate(_models.items()):
        try:
            # Update progress bar
            progress_bar.progress(
                int((idx + 1) / total * 100),
                text=f"⏳ Loading {model_name} explainer... ({idx+1}/{total})"
            )
            
            # Determine model type for appropriate explainer selection
            # Tree-based models (Random Forest) vs Linear models
            model_type = 'tree' if 'RF' in model_name or 'rf' in model_name else 'linear'
            
            # Create and fit explainer
            explainer = ModelExplainer(model, model_type=model_type)
            explainer.fit(X_sample)
            
            explainers[model_name] = explainer
            st.success(f"✅ {model_name} explainer ready")
            
        except Exception as e:
            error_msg = f"❌ Failed to load explainer for {model_name}: {str(e)}"
            loading_errors.append(error_msg)
            st.warning(error_msg)
            
            # Show detailed error in expandable section
            with st.expander(f"🐛 {model_name} Error Details"):
                st.code(traceback.format_exc())
            
            explainers[model_name] = None

    # Clear progress bar after completion
    progress_bar.empty()
    
    # ✅ FINAL STATUS: Check if any explainers loaded successfully
    if not explainers or all(v is None for v in explainers.values()):
        st.error("❌ Failed to load all explainers")
        return None
    
    successful = sum(1 for v in explainers.values() if v is not None)
    st.success(f"✅ Successfully loaded {successful}/{len(_models)} explainers")
    
    return explainers


# ============================================
# UTILITY FUNCTION FOR SIDEBAR STATUS DISPLAY
# ============================================

def show_explainer_status_in_sidebar():
    """
    Display current explainer status in Streamlit sidebar
    Shows either error diagnostics or success confirmation
    """
    if ModelExplainer is None:
        display_import_diagnostics(st.session_state.explainer_import_errors)
    else:
        st.sidebar.success("✅ ModelExplainer available")
        st.sidebar.info("SHAP explainability active")


# ============================================
# INTEGRATION GUIDE
# ============================================

"""
USAGE IN MAIN streamlit_app.py:

Replace the old import:

    try:
        from explainability import ModelExplainer
    except:
        ModelExplainer = None

With the enhanced import:

    # Import enhanced error handling components
    from enhanced_error_handling import (
        ModelExplainer, 
        load_explainers,
        show_explainer_status_in_sidebar
    )

In your sidebar section, add:

    show_explainer_status_in_sidebar()

When loading explainers, use:

    explainers = load_explainers(models)
    
This provides:
- Detailed error diagnostics
- Step-by-step validation
- User-friendly error messages
- Progress tracking
- Comprehensive logging
"""