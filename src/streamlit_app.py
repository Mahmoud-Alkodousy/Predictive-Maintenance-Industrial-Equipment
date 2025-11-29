"""
===============================================================================
PREDICTIVE MAINTENANCE SYSTEM
===============================================================================

KEY FIXES:
1. ✅ Proper session state initialization
2. ✅ Unique keys for all interactive widgets
3. ✅ Tab state persistence
4. ✅ No more unwanted tab switching
5. ✅ Comprehensive error handling
6. ✅ All comments in English

Version: 2.1.0 (Fixed Streamlit State Issues)
===============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import time

# Custom modules
from enhanced_error_handling import (
    ModelExplainer, 
    load_explainers,
    show_explainer_status_in_sidebar
)
from feature_engineering import process_telemetry
from analysis_tab import render_analysis_tab
from sample_data import create_sample_data
from explainability import ModelExplainer
from database_management import render_machine_management_tab
from config import MODELS_DIR, MODEL_FILES, DEFAULT_THRESHOLD, get_available_models

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SAFE MODULE IMPORTS
# ============================================================================

def import_chatbot_safely():
    """
    Safe import of chatbot module with comprehensive error handling
    Returns chatbot module or None if import fails
    """
    try:
        import chatbot
        return chatbot
    except ImportError as e:
        st.error(f"❌ Chatbot module not found: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Failed to load chatbot: {e}")
        import traceback
        with st.expander("🐛 Error Details"):
            st.code(traceback.format_exc())
        return None

# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================

@st.cache_resource
def load_models():
    """
    Load all available ML models with caching
    Silent loading to prevent UI flickering
    
    Returns:
        dict: Dictionary of loaded models or None if all fail
    """
    available_models = get_available_models()
    
    if not available_models:
        st.error(f"❌ No model files found in: {MODELS_DIR}")
        st.info("""
        **Expected model files:**
        - model_24h_LR.joblib
        - model_24h_RF_fast.joblib
        - model_48h_LR.joblib
        - model_48h_RF_fast.joblib
        
        **Please ensure models are in the correct directory.**
        """)
        return None
    
    models = {}
    loading_errors = []
    
    # Silent loading - no progress bars to prevent rerun issues
    for model_name, model_path in available_models.items():
        try:
            models[model_name] = joblib.load(str(model_path))
        except Exception as e:
            error_msg = f"Failed to load {model_name}: {str(e)}"
            loading_errors.append(error_msg)
    
    # Only show errors if something went wrong
    if loading_errors:
        for error in loading_errors:
            st.error(f"❌ {error}")
    
    if not models:
        st.error("❌ All models failed to load")
        return None
    
    return models

@st.cache_resource
def load_explainers(_models):
    """
    Load SHAP explainers for each model with safe error handling
    
    Args:
        _models: Dictionary of loaded models (underscore prefix for caching)
    
    Returns:
        dict: Dictionary of explainers or None if loading fails
    """
    explainers = {}

    if ModelExplainer is None:
        st.warning("⚠️ explainability.ModelExplainer not available. Skipping explainer loading.")
        return None

    try:
        from config import PROCESSED_DATA_DIR
        telemetry_path = PROCESSED_DATA_DIR / "telemetry.csv"
        
        if not telemetry_path.exists():
            st.warning("⚠️ Processed data not found. Run data_preprocessing.py first.")
            return None

        # Load sample data for explainer fitting
        telemetry = pd.read_csv(telemetry_path, parse_dates=["datetime"])
        exclude_cols = {"datetime", "machineID", "will_fail_in_24h", "will_fail_in_48h"}
        features_cols = [c for c in telemetry.columns if c not in exclude_cols]
        X_sample = telemetry[features_cols].head(1000)

        # Load explainers for each model
        for model_name, model in _models.items():
            model_type = 'tree' if 'RF' in model_name or 'rf' in model_name else 'linear'
            try:
                explainer = ModelExplainer(model, model_type=model_type)
                explainer.fit(X_sample)
                explainers[model_name] = explainer
            except Exception as e:
                st.warning(f"Failed to create explainer for {model_name}: {e}")
                explainers[model_name] = None

        return explainers

    except Exception as e:
        st.warning(f"Could not load explainers: {e}")
        return None

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """
    Initialize ALL session state variables
    MUST be called BEFORE any UI elements to prevent rerun issues
    
    This is the KEY FIX for tab switching problems
    """
    # Tab state persistence
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Data state
    if 'df_input' not in st.session_state:
        st.session_state.df_input = None
    
    # Prediction state
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'predictions_proba' not in st.session_state:
        st.session_state.predictions_proba = None
    if 'last_prediction_time' not in st.session_state:
        st.session_state.last_prediction_time = None
    
    # Explanation state
    if 'current_explanation' not in st.session_state:
        st.session_state.current_explanation = None
    if 'current_explainer' not in st.session_state:
        st.session_state.current_explainer = None
    if 'current_machine_id' not in st.session_state:
        st.session_state.current_machine_id = None
    if 'explain_requested' not in st.session_state:
        st.session_state.explain_requested = False
    if 'explained_machine_id' not in st.session_state:
        st.session_state.explained_machine_id = None
    if 'explainers_cache' not in st.session_state:
        st.session_state.explainers_cache = None
    if 'simple_explanation_cache' not in st.session_state:
        st.session_state.simple_explanation_cache = None
    
    # Configuration state
    if 'current_threshold' not in st.session_state:
        st.session_state.current_threshold = DEFAULT_THRESHOLD
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None

# ============================================================================
# PREDICTION TAB
# ============================================================================

def render_prediction_tab(models, selected_model, threshold):
    """
    Render the main prediction tab with fixed state management
    
    Args:
        models: Dictionary of loaded models
        selected_model: Currently selected model name
        threshold: Prediction threshold value
    """
    st.header("Machine Failure Prediction")
    
    # Input method selector with unique key
    input_method = st.radio(
        "Data Input Method:",
        ["Upload CSV File", "Use Sample Data", "Manual Input"],
        horizontal=True,
        key="input_method_radio"  # ✅ FIXED: Unique key
    )
    
    df_input = None
    
    # Handle different input methods
    if input_method == "Upload CSV File":
        df_input = handle_file_upload()
    elif input_method == "Use Sample Data":
        df_input = create_sample_data()
        st.info("Using sample data for demonstration")
    elif input_method == "Manual Input":
        df_input = handle_manual_input()
    
    # Store in session state for sharing between tabs
    if df_input is not None:
        st.session_state.df_input = df_input
    
    # Process predictions if data is available
    if df_input is not None:
        st.subheader("Input Data Preview")
        st.dataframe(df_input.head(), key="input_preview_df")  # ✅ FIXED: Unique key
        
        # Predict button with unique key
        if st.button("🔮 Make Predictions", type="primary", key="predict_button"):
            make_predictions(df_input, models, selected_model, threshold)
    
    # Display previous results if they exist
    if 'prediction_done' in st.session_state and st.session_state.prediction_done:
        st.subheader("🎯 Prediction Results")
        display_prediction_results(st.session_state.results_df, threshold)
        
        # Show explanation section
        generate_explanation(
            st.session_state.results_df, 
            models, 
            selected_model, 
            st.session_state.predictions_proba, 
            threshold
        )

# ============================================================================
# FILE UPLOAD HANDLER
# ============================================================================

def handle_file_upload():
    """
    Handle CSV file upload with error handling
    
    Returns:
        DataFrame: Uploaded data or None
    """
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload telemetry data with columns: machineID, datetime, volt, rotate, pressure, vibration",
        key="csv_file_uploader"  # ✅ FIXED: Unique key
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! {len(df)} rows loaded.")
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
    return None

# ============================================================================
# MANUAL INPUT HANDLER
# ============================================================================

def handle_manual_input():
    """
    Handle manual data entry with form
    
    Returns:
        DataFrame: Manually entered data
    """
    st.subheader("Enter Machine Data")
    
    col1, col2 = st.columns(2)

    with col1:
        machine_id = st.number_input(
            "Machine ID", 
            min_value=1, 
            value=1, 
            key="manual_machine_id"  # ✅ FIXED: Unique key
        )
        volt = st.number_input(
            "Voltage", 
            min_value=0.0, 
            value=170.0, 
            key="manual_volt"  # ✅ FIXED: Unique key
        )
        rotate = st.number_input(
            "Rotation", 
            min_value=0.0, 
            value=420.0, 
            key="manual_rotate"  # ✅ FIXED: Unique key
        )

    with col2:
        pressure = st.number_input(
            "Pressure", 
            min_value=0.0, 
            value=100.0, 
            key="manual_pressure"  # ✅ FIXED: Unique key
        )
        vibration = st.number_input(
            "Vibration", 
            min_value=0.0, 
            value=40.0, 
            key="manual_vibration"  # ✅ FIXED: Unique key
        )
        model_type = st.selectbox(
            "Model", 
            ["model1", "model2", "model3", "model4"], 
            key="manual_model_type"  # ✅ FIXED: Unique key
        )

    # Create DataFrame from manual input
    df = pd.DataFrame({
        'machineID': [machine_id],
        'datetime': ['2015-12-01 10:00:00'],
        'volt': [volt],
        'rotate': [rotate],
        'pressure': [pressure],
        'vibration': [vibration],
        'model': [model_type],
        'age': [10]
    })

    st.write("Current Input Data:")
    st.dataframe(df, key="manual_input_preview_df")  # ✅ FIXED: Unique key
    
    return df

# ============================================================================
# PREDICTION PROCESSING
# ============================================================================

def make_predictions(df_input, models, selected_model, threshold):
    """
    Process data and make predictions with proper state management
    
    Args:
        df_input: Input DataFrame
        models: Dictionary of models
        selected_model: Selected model name
        threshold: Prediction threshold
    """
    try:
        with st.spinner("Processing data and making predictions..."):
            model = models[selected_model]
            
            # Process each row separately
            all_predictions = []
            for idx in range(len(df_input)):
                single_row = df_input.iloc[[idx]]
                processed_single = process_telemetry(single_row)
                pred_proba = model.predict_proba(processed_single)[0, 1]
                all_predictions.append(pred_proba)
            
            # Convert to arrays
            predictions_proba = np.array(all_predictions)
            predictions_binary = (predictions_proba >= threshold).astype(int)
            
            # Create results DataFrame
            results_df = df_input.copy()
            results_df['failure_probability'] = predictions_proba
            results_df['failure_prediction'] = predictions_binary
            results_df['risk_level'] = pd.cut(
                predictions_proba, 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['Low', 'Medium', 'High']
            )
        
        # Save results to session state
        st.session_state.results_df = results_df
        st.session_state.predictions_proba = predictions_proba
        st.session_state.prediction_done = True
        st.session_state.last_prediction_time = time.time()
        
        # Clear any previous explanation to prevent conflicts
        keys_to_clear = [
            'current_explanation', 'current_explainer', 'current_machine_id', 
            'explain_requested', 'explained_machine_id'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("✅ Predictions completed!")
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        with st.expander("🐛 Error Details"):
            st.code(traceback.format_exc())

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def display_prediction_results(results_df, threshold):
    """
    Display prediction results with metrics and visualizations
    
    Args:
        results_df: Results DataFrame with predictions
        threshold: Prediction threshold used
    """
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Machines", len(results_df))
    with col2:
        high_risk = (results_df['failure_probability'] >= threshold).sum()
        st.metric("High Risk Machines", high_risk)
    with col3:
        avg_risk = results_df['failure_probability'].mean()
        st.metric("Average Risk", f"{avg_risk:.3f}")
    with col4:
        max_risk = results_df['failure_probability'].max()
        st.metric("Maximum Risk", f"{max_risk:.3f}")
    
    # Risk distribution histogram
    if len(results_df) > 1:
        fig = px.histogram(
            results_df, 
            x='failure_probability', 
            title="Risk Distribution",
            nbins=20
        )
        fig.add_vline(
            x=threshold, 
            line_dash="dash", 
            line_color="red", 
            annotation_text="Threshold"
        )
        # ✅ FIXED: Unique key with timestamp to prevent caching issues
        unique_key = f"risk_hist_{int(time.time() * 1000000)}"
        st.plotly_chart(fig, use_container_width=True, key=unique_key)
    
    # Detailed results table
    st.subheader("Detailed Results")
    display_results_table(results_df)
    
    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download Results",
        csv,
        "prediction_results.csv",
        "text/csv",
        key="download_results_button"  # ✅ FIXED: Unique key
    )

def display_results_table(results_df):
    """
    Format and display results table with proper column selection
    
    Args:
        results_df: Results DataFrame
    """
    # Define display columns
    base_cols = ['machineID', 'model', 'failure_probability', 'failure_prediction', 'risk_level']
    display_cols = []
    
    # Add datetime if exists
    if 'datetime' in results_df.columns:
        display_cols.append('datetime')
    
    # Add base columns
    for col in base_cols:
        if col not in display_cols:
            display_cols.append(col)

    # Filter available columns
    available_cols = [col for col in display_cols if col in results_df.columns]
    display_df = results_df[available_cols].copy()
    
    # Add status column
    if 'failure_prediction' in display_df.columns:
        display_df['Status'] = display_df['failure_prediction'].map({
            0: '✅ Normal', 
            1: '⚠ High Risk'
        })
    
    # Round probability
    if 'failure_probability' in display_df.columns:
        display_df['failure_probability'] = display_df['failure_probability'].round(4)

    # Display with unique key
    st.dataframe(display_df, use_container_width=True, key="detailed_results_table")

# ============================================================================
# EXPLANATION GENERATION (FIXED STATE MANAGEMENT)
# ============================================================================

def generate_explanation(results_df, models, selected_model, predictions_proba, threshold):
    """
    Generate explanation with FIXED state management to prevent tab switching
    
    Args:
        results_df: Results DataFrame
        models: Dictionary of models
        selected_model: Selected model name
        predictions_proba: Prediction probabilities
        threshold: Prediction threshold
    """
    st.markdown("---")
    st.header("🔍 Explain This Prediction")
    st.write("**Select Machine to Explain:**")
    
    # Machine selector with control buttons
    col_select, col_explain_btn, col_clear_btn = st.columns([5, 1, 1])
    
    with col_select:
        if len(results_df) > 1:
            machine_options = results_df['machineID'].tolist()
            selected_machine_idx = st.selectbox(
                "Select Machine to Explain:",
                range(len(machine_options)),
                format_func=lambda x: f"Machine {machine_options[x]} - Risk: {predictions_proba[x]:.1%}",
                key="machine_selector",  # ✅ FIXED: Unique key
                label_visibility="collapsed"
            )
        else:
            selected_machine_idx = 0
            st.info(f"Machine {results_df.iloc[0]['machineID']} - Risk: {predictions_proba[0]:.1%}")
    
    with col_explain_btn:
        explain_button = st.button(
            "🔍 Explain", 
            type="primary", 
            use_container_width=True, 
            key="explain_button"  # ✅ FIXED: Unique key
        )
    
    with col_clear_btn:
        clear_button = st.button(
            "🗑️ Clear", 
            use_container_width=True, 
            key="clear_explanation_button"  # ✅ FIXED: Unique key
        )
    
    # Clear button handler
    if clear_button:
        keys_to_clear = [
            'current_explanation', 'current_explainer', 'current_machine_id', 
            'explain_requested', 'explained_machine_id', 'explainers_cache',
            'simple_explanation_cache'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()  # ✅ FIXED: Proper rerun without tab switch
    
    # Get selected machine info
    selected_machine_id = results_df.iloc[selected_machine_idx]['machineID']
    selected_prediction = predictions_proba[selected_machine_idx]
    
    machine_info = {
        'machineID': selected_machine_id,
        'datetime': results_df.iloc[selected_machine_idx].get('datetime', 'N/A'),
        'model': results_df.iloc[selected_machine_idx].get('model', 'N/A')
    }
    
    # Track machine selection changes
    if 'current_machine_id' in st.session_state:
        if st.session_state.current_machine_id != selected_machine_id:
            # Clear old explanation when machine changes
            for key in ['current_explanation', 'current_explainer', 'simple_explanation_cache']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_machine_id = None
            st.session_state.explain_requested = False
    
    # Determine if we should display explanation
    should_display = False
    
    if explain_button:
        st.session_state.explain_requested = True
        st.session_state.explained_machine_id = selected_machine_id
        should_display = True
    elif (
        'explain_requested' in st.session_state 
        and st.session_state.explain_requested 
        and 'explained_machine_id' in st.session_state 
        and st.session_state.explained_machine_id == selected_machine_id
    ):
        should_display = True
    
    # Display cached explanation if available
    if should_display and (
        'current_explanation' in st.session_state 
        and 'current_machine_id' in st.session_state 
        and st.session_state.current_machine_id == selected_machine_id
    ):
        explainer = st.session_state.get('current_explainer', None)
        display_explanation(
            st.session_state.current_explanation, 
            explainer, 
            machine_info
        )
        return
    
    # Generate new explanation
    if should_display:
        try:
            # Get machine input data
            machine_input_row = results_df.iloc[[selected_machine_idx]].copy()
            
            # Process the data
            with st.spinner("⏳ Processing machine data..."):
                processed_row = process_telemetry(machine_input_row)
            
            # Try SHAP explainers first
            explanation = None
            explainer_used = None
            
            try:
                # Load explainers with caching
                with st.spinner("🔧 Loading explainer models... (may take 30-60 seconds first time)"):
                    if 'explainers_cache' not in st.session_state:
                        explainers = load_explainers(models)
                        st.session_state['explainers_cache'] = explainers
                    else:
                        explainers = st.session_state['explainers_cache']
                
                # Check if explainers loaded successfully
                if explainers is not None and selected_model in explainers:
                    explainer_obj = explainers[selected_model]
                    
                    if explainer_obj is not None:
                        try:
                            explanation_result = explainer_obj.explain_instance(processed_row)
                            
                            if explanation_result and 'feature_impacts' in explanation_result:
                                explanation = {
                                    'prediction': 1 if selected_prediction >= threshold else 0,
                                    'probability': selected_prediction,
                                    'feature_impacts': explanation_result['feature_impacts'],
                                    'method': explanation_result.get('method', 'SHAP')
                                }
                                explainer_used = explainer_obj
                                st.success(f"✅ Explanation generated using {explanation['method']}")
                                
                        except Exception as e:
                            st.warning(f"⚠️ SHAP explainer failed: {e}")
                            st.info("Falling back to simple method...")
            
            except Exception as e:
                st.warning(f"⚠️ Could not load explainers: {e}")
            
            # Fallback: Use simple feature-based explanation
            if explanation is None:
                with st.spinner("🔧 Generating simple explanation..."):
                    explanation = generate_simple_explanation(
                        processed_row, 
                        models[selected_model], 
                        selected_prediction,
                        threshold
                    )
                    explainer_used = None
                    st.info("ℹ️ Using simplified explanation (Feature Importance)")
            
            # Cache the explanation
            st.session_state.current_explanation = explanation
            st.session_state.current_explainer = explainer_used
            st.session_state.current_machine_id = selected_machine_id
            
            # Display the explanation
            display_explanation(explanation, explainer_used, machine_info)
            
        except Exception as e:
            st.error(f"❌ Could not generate explanation: {e}")
            
            with st.expander("🐛 Error Details"):
                import traceback
                st.code(traceback.format_exc())
            
            st.info("""
            **Possible solutions:**
            - Ensure model is properly trained
            - Check that input data has all required features
            - Verify preprocessing pipeline works correctly
            """)
    else:
        st.info("👆 Select a machine and click 'Explain' to see detailed analysis")

# ============================================================================
# SIMPLE EXPLANATION GENERATOR (FALLBACK)
# ============================================================================

def generate_simple_explanation(processed_row, model, prediction_proba, threshold):
    """
    Generate explanation using model coefficients or feature importance
    Fallback method when SHAP is unavailable
    
    Args:
        processed_row: Processed input data
        model: ML model (pipeline)
        prediction_proba: Prediction probability
        threshold: Decision threshold
    
    Returns:
        dict: Explanation dictionary
    """
    try:
        # Extract preprocessor and classifier from pipeline
        prep = model.named_steps['prep']
        clf = model.named_steps['clf']
        
        # Transform the data
        X_transformed = prep.transform(processed_row)
        feature_names = prep.get_feature_names_out()
        
        # Extract feature values
        if hasattr(X_transformed, 'toarray'):
            feature_values = X_transformed.toarray()[0]
        else:
            feature_values = X_transformed[0] if X_transformed.ndim > 1 else X_transformed
        
        # Calculate impacts based on model type
        
        # Method 1: Logistic Regression (exact coefficients)
        if hasattr(clf, 'coef_') and hasattr(clf, 'intercept_'):
            coefficients = clf.coef_[0]
            intercept = clf.intercept_[0]
            
            # Impact = coefficient × feature_value
            impacts = coefficients * feature_values
            
            # Base value (probability with intercept only)
            from scipy.special import expit
            base_value = expit(intercept)
            
            method = "Logistic Regression Coefficients"
            
            st.info(f"""
            📊 **Calculation Method:**
            - Using exact Logistic Regression coefficients
            - Impact = coefficient × feature_value
            - Sum of impacts ≈ log-odds of prediction
            """)
        
        # Method 2: Random Forest (approximate importance)
        elif hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
            
            # Calculate normalized contribution
            feature_scale = np.abs(feature_values).max() + 1e-10
            normalized_values = feature_values / feature_scale
            prediction_deviation = (prediction_proba - 0.5) * 2  # Scale -1 to 1
            
            impacts = importance * normalized_values * prediction_deviation
            
            base_value = 0.5
            method = "Random Forest Feature Importance"
            
            st.info(f"""
            📊 **Calculation Method:**
            - Using Random Forest feature importance
            - Impact = importance × normalized_value × prediction_deviation
            - This is an approximation (not exact like SHAP)
            """)
        
        else:
            # Fallback: minimal impact
            impacts = np.zeros(len(feature_values))
            base_value = 0.5
            method = "Fallback (No Model Info)"
            st.warning("⚠️ Could not extract model coefficients or importance")
        
        # Create feature impacts DataFrame
        feature_impacts = pd.DataFrame({
            'feature': feature_names,
            'value': feature_values,
            'impact': impacts
        })
        
        # Sort by absolute impact
        feature_impacts = feature_impacts.assign(
            abs_impact=feature_impacts['impact'].abs()
        ).sort_values('abs_impact', ascending=False).drop(columns=['abs_impact']).reset_index(drop=True)
        
        # Verify impacts are reasonable
        max_impact = feature_impacts['impact'].abs().max()
        if max_impact > 10:
            st.warning(f"⚠️ Large impacts detected (max: {max_impact:.2f}) - this might indicate a calculation issue")
        
        # Return explanation
        return {
            'prediction': 1 if prediction_proba >= threshold else 0,
            'probability': prediction_proba,
            'feature_impacts': feature_impacts,
            'method': method,
            'base_value': base_value
        }
        
    except Exception as e:
        st.error(f"❌ Failed to generate explanation: {e}")
        
        import traceback
        with st.expander("🐛 Full Error Traceback"):
            st.code(traceback.format_exc())
        
        # Return minimal safe explanation
        return {
            'prediction': 1 if prediction_proba >= threshold else 0,
            'probability': prediction_proba,
            'feature_impacts': pd.DataFrame({
                'feature': ['Error'],
                'value': [0],
                'impact': [0]
            }),
            'method': 'Error - Could not calculate',
            'base_value': 0.5
        }

# ============================================================================
# EXPLANATION DISPLAY
# ============================================================================

def display_explanation(explanation, explainer, machine_info):
    """
    Display explanation results with visualizations
    
    Args:
        explanation: Explanation dictionary
        explainer: Explainer object (or None)
        machine_info: Machine information dictionary
    """
    if explanation is None:
        st.error("❌ Missing explanation data")
        return
    
    st.success(f"✅ Explanation Generated ({explanation.get('method', 'Unknown Method')})")
    
    # Display machine information
    st.subheader(f"🔧 Machine {machine_info['machineID']} Analysis")
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.info(f"**Machine ID:** {machine_info['machineID']}")
    with info_col2:
        st.info(f"**Timestamp:** {machine_info['datetime']}")
    with info_col3:
        st.info(f"**Model:** {machine_info['model']}")
    
    st.markdown("---")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Prediction",
            "⚠️ FAILURE" if explanation['prediction'] == 1 else "✅ NORMAL"
        )
    
    with col2:
        st.metric(
            "Failure Probability",
            f"{explanation['probability']:.1%}"
        )
    
    with col3:
        prob = explanation['probability']
        risk = "🔴 HIGH" if prob > 0.7 else "🟡 MEDIUM" if prob > 0.3 else "🟢 LOW"
        st.metric("Risk Level", risk)
    
    # Feature impacts analysis
    impacts_df = explanation['feature_impacts'].copy()
    
    if len(impacts_df) == 0:
        st.warning("⚠️ No feature impacts available")
        return
    
    # Calculate cumulative impact
    impacts_df['abs_impact'] = impacts_df['impact'].abs()
    impacts_df = impacts_df.sort_values('abs_impact', ascending=False)
    
    total_impact = impacts_df['abs_impact'].sum()
    
    if total_impact > 0:
        impacts_df['cumulative_impact'] = impacts_df['abs_impact'].cumsum()
        impacts_df['cumulative_pct'] = (impacts_df['cumulative_impact'] / total_impact) * 100
        
        # Calculate features for 90% impact
        features_for_90 = (impacts_df['cumulative_pct'] <= 90).sum()
        num_features = max(5, min(features_for_90, 20))
    else:
        num_features = min(10, len(impacts_df))
    
    # Waterfall plot
    st.subheader("📊 Feature Contribution Waterfall")
    st.markdown(f"*Showing top {num_features} features*")
    st.markdown("*🔴 Red = Increases risk | 🟢 Green = Decreases risk*")
    
    # Create waterfall plot
    if explainer is not None and hasattr(explainer, 'plot_waterfall'):
        try:
            fig_waterfall = explainer.plot_waterfall(explanation, num_features)
            # ✅ FIXED: Unique key with timestamp
            unique_key = f"waterfall_{machine_info['machineID']}_{int(time.time() * 1000000)}"
            st.plotly_chart(fig_waterfall, use_container_width=True, key=unique_key)
        except Exception as e:
            st.warning(f"Could not generate waterfall plot: {e}")
            create_simple_waterfall(impacts_df, num_features, explanation, machine_info)
    else:
        create_simple_waterfall(impacts_df, num_features, explanation, machine_info)
    
    # Feature impacts table
    st.subheader("📋 Top Feature Impacts")
    
    display_impacts = impacts_df.head(15).copy()
    display_impacts['impact'] = display_impacts['impact'].round(4)
    display_impacts['value'] = display_impacts['value'].round(2)
    
    if 'cumulative_pct' in display_impacts.columns:
        display_impacts['cumulative_pct'] = display_impacts['cumulative_pct'].round(1)
    
    display_impacts['effect'] = display_impacts['impact'].apply(
        lambda x: "⬆️ Increases Risk" if x > 0 else "⬇️ Decreases Risk"
    )
    
    cols_to_show = ['feature', 'value', 'impact', 'effect']
    if 'cumulative_pct' in display_impacts.columns:
        cols_to_show.append('cumulative_pct')
    
    rename_dict = {'cumulative_pct': 'Cumulative %'} if 'cumulative_pct' in cols_to_show else {}
    
    st.dataframe(
        display_impacts[cols_to_show].rename(columns=rename_dict),
        use_container_width=True,
        hide_index=True,
        key="feature_impacts_table"  # ✅ FIXED: Unique key
    )
    
    # Summary recommendation
    display_recommendation(explanation, machine_info)

def create_simple_waterfall(impacts_df, num_features, explanation, machine_info):
    """
    Create a simple waterfall plot using Plotly
    
    Args:
        impacts_df: Feature impacts DataFrame
        num_features: Number of features to show
        explanation: Explanation dictionary
        machine_info: Machine information
    """
    top_impacts = impacts_df.head(num_features)
    
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(top_impacts),
        y=top_impacts['feature'].tolist()[::-1],
        x=top_impacts['impact'].tolist()[::-1],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "red"}},
        decreasing={"marker": {"color": "green"}},
        text=[f"{v:.3f}" for v in top_impacts['impact'].tolist()[::-1]],
        textposition="outside"
    ))
    
    fig.update_layout(
        title=f"🔍 Top {num_features} Feature Contributions<br>"
              f"<sub>Prediction: {explanation['probability']:.1%} failure probability</sub>",
        xaxis_title="Impact on Prediction",
        yaxis_title="Feature",
        height=400 + (num_features * 20),
        showlegend=False
    )
    
    # ✅ FIXED: Unique key with timestamp
    unique_key = f"simple_waterfall_{machine_info['machineID']}_{int(time.time() * 1000000)}"
    st.plotly_chart(fig, use_container_width=True, key=unique_key)

def display_recommendation(explanation, machine_info):
    """
    Display actionable recommendation based on prediction
    
    Args:
        explanation: Explanation dictionary
        machine_info: Machine information
    """
    top_feature = explanation['feature_impacts'].iloc[0]
    
    if explanation['prediction'] == 1:
        st.error(f"""
        **⚠️ High Failure Risk Detected for Machine {machine_info['machineID']}**
        
        Main driver: **{top_feature['feature']}** (impact: {top_feature['impact']:.3f})
        
        **Recommended Actions:**
        - Schedule immediate inspection for Machine {machine_info['machineID']}
        - Monitor {top_feature['feature']} closely
        - Review maintenance logs for this machine
        - Consider preventive maintenance
        """)
    else:
        st.success(f"""
        **✅ Machine {machine_info['machineID']} Operating Normally**
        
        All indicators within acceptable ranges.
        
        **Continue monitoring:** {top_feature['feature']} (value: {top_feature['value']:.2f})
        """)

# ============================================================================
# CHATBOT TAB
# ============================================================================

def render_chatbot_tab():
    """
    Render the integrated chatbot tab with safe import
    Handles missing dependencies gracefully
    """
    try:
        # Safe import
        chatbot_module = import_chatbot_safely()
        
        if chatbot_module is None:
            st.error("❌ Chatbot module not available")
            st.info("""
            **Required Setup:**
            
            1. Ensure `chatbot.py` exists in the same directory
            2. Install dependencies:
               ```bash
               pip install supabase sentence-transformers pydantic requests python-dotenv
               ```
            3. Configure `.env` file with:
               - SUPABASE_URL
               - SUPABASE_KEY
               - OPENROUTER_KEY
            4. Restart the application
            """)
            return
        
        # Call the UI function
        chatbot_module.render_enhanced_ui(standalone=False)
        
    except AttributeError as e:
        st.error(f"❌ Chatbot function not found: {e}")
        st.info("Make sure `render_enhanced_ui(standalone=False)` exists in chatbot.py")
        
    except Exception as e:
        st.error(f"❌ Chatbot failed to load: {str(e)}")
        
        with st.expander("🐛 Error Details"):
            import traceback
            st.code(traceback.format_exc())
        
        st.info("""
        **Troubleshooting:**
        
        1. Check that chatbot.py is in the correct location
        2. Verify all dependencies are installed
        3. Check .env configuration
        4. Look at the error details above
        5. Try running chatbot.py standalone to test
        """)

# ============================================================================
# ABOUT TAB
# ============================================================================

def render_about_tab():
    """
    Render the about/information tab
    """
    st.header("ℹ️ About This System")
    st.markdown("""
    ### Predictive Maintenance System with AI Assistant
    
    This comprehensive system combines machine learning predictions with AI-powered assistance
    for industrial maintenance operations.
    
    **Core Features:**
    - 🔮 **ML Predictions**: 24h/48h failure forecasting
    - 🤖 **AI Chatbot**: Natural language Q&A in Arabic/English
    - 🔍 **Image Inspection**: Visual defect detection
    - 📊 **Analytics**: Advanced data visualization
    - 💡 **Explainability**: SHAP-based model interpretation
    
    **Available Models:**
    - **LR_24h**: Logistic Regression for 24-hour prediction
    - **RF_24h**: Random Forest for 24-hour prediction  
    - **LR_48h**: Logistic Regression for 48-hour prediction
    - **RF_48h**: Random Forest for 48-hour prediction
    
    **Explainability (XAI):**
    - 🌊 Waterfall charts showing feature contributions
    - 📊 Feature impact analysis
    - 🎯 Risk level assessment
    - 💡 Actionable recommendations
    
    **AI Chatbot Capabilities:**
    - 💬 Natural language understanding (Arabic/English)
    - 📚 RAG-based answers from maintenance manuals
    - 🔍 Semantic search across documentation
    - 🗄️ Database queries for machine information
    - 💰 Price and parts information
    
    **Input Data Requirements:**
    - `machineID`: Unique machine identifier
    - `datetime`: Measurement timestamp
    - `volt`: Voltage reading
    - `rotate`: Rotation speed
    - `pressure`: Pressure measurement  
    - `vibration`: Vibration level
    - `model`: Machine model (optional, defaults to model1)
    - `age`: Machine age in years (optional, defaults to 18)
    
    **How It Works:**
    1. Upload telemetry data or use manual input
    2. System engineers features from raw sensor data
    3. ML models predict failure probability
    4. SHAP explainer shows feature contributions
    5. AI chatbot provides additional guidance
    6. Get actionable insights and recommendations
    
    **Technology Stack:**
    - 🤖 **ML**: Scikit-learn (RF, LR)
    - 🔍 **XAI**: SHAP (SHapley Additive exPlanations)
    - 💬 **AI**: Anthropic Claude (via OpenRouter)
    - 🗄️ **Database**: Supabase (PostgreSQL + pgvector)
    - 🔎 **Embeddings**: Sentence Transformers
    - 📊 **Frontend**: Streamlit + Plotly
    - 🖼️ **Vision**: TensorFlow + YOLOv5
    
    **Version:** 2.1.0 (Fixed State Management)
    """)

# ============================================================================
# SAMPLE DATA TAB
# ============================================================================

def render_sample_data_tab():
    """
    Render the sample data tab with downloadable template
    """
    st.header("🔧 Sample Data")
    st.write("Here's a sample of the expected data format:")
    
    sample_df = create_sample_data()
    st.dataframe(sample_df, key="sample_data_display")  # ✅ FIXED: Unique key
    
    csv_sample = sample_df.to_csv(index=False)
    st.download_button(
        "📥 Download Sample CSV",
        csv_sample,
        "sample_telemetry.csv",
        "text/csv",
        help="Download this sample to test the application",
        key="download_sample_button"  # ✅ FIXED: Unique key
    )

# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """
    Main application entry point with FIXED state management
    This prevents tab switching and UI flickering on button clicks
    """
    # ✅ CRITICAL: Initialize session state FIRST
    initialize_session_state()
    
    st.title("⚙️ Predictive Maintenance System")
    st.markdown("---")

    # Load models with caching
    models = load_models()
    if models is None:
        st.error("Failed to load models. Please ensure model files exist.")
        return

    # ========================================
    # SIDEBAR CONFIGURATION
    # ========================================
    st.sidebar.header("Configuration")
    
    available_models = get_available_models()

    if not available_models:
        st.sidebar.error("No available models found.")
        return

    # Model selector with persistent state
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(available_models.keys()),
        key="model_selector_sidebar",  # ✅ FIXED: Unique key
        help="Choose prediction horizon and algorithm"
    )
    st.session_state.selected_model = selected_model
    
    # Threshold slider with persistent state
    threshold = st.sidebar.slider(
        "Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.current_threshold,  # ✅ FIXED: Use session state
        step=0.01,
        key="threshold_slider_sidebar",  # ✅ FIXED: Unique key
        help="Threshold for failure prediction (higher = more conservative)"
    )
    st.session_state.current_threshold = threshold
    
    # ========================================
    # TAB NAVIGATION WITH STATE PERSISTENCE
    # ========================================
    
    # Tab labels
    tab_labels = [
        "📊 Prediction", 
        "📈 Data Analysis",
        "💬 AI Assistant",
        "🏭 Machine Management",
        "🔍 Image Inspection",
        "📡 Live Monitor",
        "🎨 3D Visualizations",
        "📄 Reports",
        "🔔 Alerts",
        "🔧 Sample Data", 
        "ℹ️ About"
    ]
    
    # Create tabs
    tabs = st.tabs(tab_labels)
    
    # ========================================
    # TAB 0: PREDICTION
    # ========================================
    with tabs[0]:
        render_prediction_tab(models, selected_model, threshold)
    
    # ========================================
    # TAB 1: DATA ANALYSIS
    # ========================================
    with tabs[1]:
        render_analysis_tab(st.session_state.df_input)
    
    # ========================================
    # TAB 2: AI ASSISTANT (CHATBOT)
    # ========================================
    with tabs[2]:
        render_chatbot_tab()

    # ========================================
    # TAB 3: MACHINE MANAGEMENT
    # ========================================
    with tabs[3]:
        try:
            render_machine_management_tab()
        except Exception as e:
            import traceback
            st.error("⚠️ Failed to load Machine Management module.")
            with st.expander("🐛 Error Details"):
                st.code(traceback.format_exc())
            st.info("Check database_management.py for errors or missing dependencies.")
    
    # ========================================
    # TAB 4: IMAGE INSPECTION
    # ========================================
    with tabs[4]:
        try:
            from image_inspection import render_image_inspection_tab
            render_image_inspection_tab()
        except Exception as e:
            import traceback
            st.error("⚠️ Failed to load Image Inspection module.")
            with st.expander("🐛 Error Details"):
                st.code(traceback.format_exc())
            st.info("Check image_inspection.py for errors or missing models.")

    # ========================================
    # TAB 5: LIVE MONITOR
    # ========================================
    with tabs[5]:
        try:
            from live_dashboard import main as live_main
            live_main()
        except Exception as e:
            st.error(f"⚠️ Live Monitor failed: {e}")

    # ========================================
    # TAB 6: 3D VISUALIZATIONS
    # ========================================
    with tabs[6]:
        try:
            from visualization_3d import show_3d_visualizations
            show_3d_visualizations()
        except Exception as e:
            st.error(f"⚠️ 3D Visualizations failed: {e}")

    # ========================================
    # TAB 7: REPORTS
    # ========================================
    with tabs[7]:
        try:
            from report_generator import show_report_generator
            show_report_generator()
        except Exception as e:
            st.error(f"⚠️ Report Generator failed: {e}")

    # ========================================
    # TAB 8: ALERTS
    # ========================================
    with tabs[8]:
        try:
            from smart_alerts import show_alert_system
            show_alert_system()
        except Exception as e:
            st.error(f"⚠️ Alert System failed: {e}")
    
    # ========================================
    # TAB 9: SAMPLE DATA
    # ========================================
    with tabs[9]:
        render_sample_data_tab()
    
    # ========================================
    # TAB 10: ABOUT
    # ========================================
    with tabs[10]:
        render_about_tab()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()