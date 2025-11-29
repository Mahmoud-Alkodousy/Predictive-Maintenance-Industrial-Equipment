"""
Feature Engineering Module
Handles sensor data transformation and feature extraction for predictive maintenance
"""

import pandas as pd
import numpy as np
from config import SENSORS, LAGS, ROLL_MEANS, ROLL_STDS, SLOPES_K


# ============================================
# SENSOR FEATURE ENGINEERING
# ============================================

def add_sensor_features(df, sensors, lags, rmeans, rstds, slopes):
    """
    Create advanced time-series features from raw sensor data
    
    Generates 4 types of features:
    1. Lag features: Previous values at specific time intervals
    2. Rolling means: Moving averages over time windows
    3. Rolling std: Moving standard deviations (volatility)
    4. Slope features: Rate of change over time intervals
    
    Args:
        df (pd.DataFrame): Input telemetry data with datetime and machineID
        sensors (list): List of sensor column names to process
        lags (list): List of lag periods in hours (e.g., [3, 6, 12])
        rmeans (list): List of window sizes for rolling means
        rstds (list): List of window sizes for rolling standard deviations
        slopes (list): List of periods for slope calculation
        
    Returns:
        pd.DataFrame: Original data with added engineered features
    """
    out = df.copy()
    
    # Group by machineID to calculate features per machine
    g = out.groupby("machineID", group_keys=False)

    # ============================================
    # FEATURE TYPE 1: Lag Features
    # Captures historical values at specific time points
    # ============================================
    for c in sensors:
        for k in lags:
            # Create lagged column: sensor_lag_Xh
            # Example: volt_lag_3h = voltage value from 3 hours ago
            out[f"{c}_lag_{k}h"] = g[c].shift(k)

    # ============================================
    # FEATURE TYPE 2: Rolling Means
    # Captures recent average trends
    # ============================================
    for c in sensors:
        for k in rmeans:
            # Calculate rolling average over k-hour window
            # Example: volt_mean_24h = average voltage over last 24 hours
            out[f"{c}_mean_{k}h"] = (
                g[c].rolling(window=k, min_periods=k).mean()
                 .reset_index(level=0, drop=True)
            )

    # ============================================
    # FEATURE TYPE 3: Rolling Standard Deviations
    # Captures recent volatility/stability
    # ============================================
    for c in sensors:
        for k in rstds:
            # Calculate rolling standard deviation over k-hour window
            # Example: volt_std_24h = voltage volatility over last 24 hours
            # ddof=0 for population std (consistent with training)
            out[f"{c}_std_{k}h"] = (
                g[c].rolling(window=k, min_periods=k).std(ddof=0)
                 .reset_index(level=0, drop=True)
            )

    # ============================================
    # FEATURE TYPE 4: Slope Features
    # Captures rate of change (trend direction)
    # ============================================
    for c in sensors:
        for k in slopes:
            lag_col = f"{c}_lag_{k}h"
            
            # Ensure lag column exists (may already be created above)
            if lag_col not in out.columns:
                out[lag_col] = g[c].shift(k)
            
            # Calculate slope: (current_value - lagged_value) / time_period
            # Example: volt_slope_12h = (volt_now - volt_12h_ago) / 12
            # Positive slope = increasing trend, Negative = decreasing trend
            out[f"{c}_slope_{k}h"] = (out[c] - out[lag_col]) / k

    return out


# ============================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================

def process_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for real-time telemetry data
    
    Handles edge cases for production deployment:
    - Single-row predictions (creates synthetic history)
    - Missing columns (adds defaults)
    - Missing values (imputation)
    - Categorical encoding
    
    Args:
        df (pd.DataFrame): Raw telemetry data (can be single row)
        
    Returns:
        pd.DataFrame: Fully processed feature set ready for model prediction
    """
    out = df.copy()

    # ============================================
    # STEP 1: Parse datetime column
    # ============================================
    if 'datetime' in out.columns and out['datetime'].dtype == 'object':
        out['datetime'] = pd.to_datetime(out['datetime'])

    # ============================================
    # STEP 2: Add default values for missing metadata
    # ============================================
    # Machine model (if not provided)
    if 'model' not in out.columns:
        out['model'] = 'model1'
    
    # Machine age in months (if not provided)
    if 'age' not in out.columns:
        out['age'] = 18  # Default: 18 months old

    # ============================================
    # STEP 3: Handle single-row predictions
    # Create synthetic historical data for rolling features
    # ============================================
    if len(out) < 50:
        original_row = out.iloc[0].copy()
        dummy_rows = []
        
        # Generate 49 dummy rows with same values but earlier timestamps
        for i in range(49):
            row = original_row.copy()
            
            # Backdate timestamp if datetime exists
            if 'datetime' in out.columns:
                row['datetime'] = pd.to_datetime(row['datetime']) - pd.Timedelta(hours=49-i)
            
            dummy_rows.append(row)
        
        # Prepend dummy history to actual data
        dummy_df = pd.DataFrame(dummy_rows)
        out = pd.concat([dummy_df, out], ignore_index=True)

    # ============================================
    # STEP 4: Add dummy target columns (not used in prediction)
    # ============================================
    out["will_fail_in_24h"] = 0
    out["will_fail_in_48h"] = 0

    # ============================================
    # STEP 5: Add dummy error count features
    # In production, these would come from error log aggregation
    # ============================================
    for e in ["error1", "error2", "error3", "error4", "error5"]:
        out[f"{e}_last_24h"] = 0  # Error count in last 24 hours
        out[f"{e}_last_48h"] = 0  # Error count in last 48 hours

    # ============================================
    # STEP 6: Add dummy maintenance history features
    # In production, these would come from maintenance log
    # ============================================
    for c in ["comp1", "comp2", "comp3", "comp4"]:
        # Time since last maintenance for each component
        out[f"time_since_maint_{c}_h"] = 48.0   # Hours since maintenance
        out[f"time_since_maint_{c}_d"] = 2.0    # Days since maintenance

    # ============================================
    # STEP 7: Generate sensor-based engineered features
    # This is the core feature engineering step
    # ============================================
    out = add_sensor_features(out, SENSORS, LAGS, ROLL_MEANS, ROLL_STDS, SLOPES_K)

    # ============================================
    # STEP 8: Handle missing values (imputation)
    # ============================================
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if out[col].isna().any():
            # Fill with median value (robust to outliers)
            out[col] = out[col].fillna(out[col].median())

    # ============================================
    # STEP 9: Encode categorical features
    # ============================================
    if 'model' in out.columns:
        out['model'] = out['model'].astype('category')

    # ============================================
    # STEP 10: Select only feature columns (exclude metadata/targets)
    # ============================================
    exclude = {"datetime", "machineID", "will_fail_in_24h", "will_fail_in_48h"}
    features = [c for c in out.columns if c not in exclude]
    
    # Return only the last row (actual prediction row) with feature columns
    return out.loc[:, features].tail(1).copy()