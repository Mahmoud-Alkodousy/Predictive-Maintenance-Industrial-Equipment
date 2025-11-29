"""
Utility Functions for Data Processing and Model Training
Provides helper functions for preprocessing, feature engineering, and model evaluation
Used by preprocessing and training scripts only
"""

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    roc_curve, roc_auc_score, classification_report, confusion_matrix,
    average_precision_score, precision_recall_curve, fbeta_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from uuid import uuid4

# ============================================
# REPRODUCIBILITY CONFIGURATION
# ============================================

# Set global seed for reproducible results across runs
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
RANDOM_STATE = 42


# ============================================
# CUSTOM SKLEARN TRANSFORMERS
# ============================================

class ToFloat32(BaseEstimator, TransformerMixin):
    """
    Custom transformer to cast data to float32 for memory efficiency
    
    Reduces memory usage by ~50% compared to float64 while maintaining
    sufficient precision for most ML tasks
    """
    def fit(self, X, y=None):
        """Fitting does nothing - stateless transformer"""
        return self
    
    def transform(self, X):
        """Convert input data to float32 dtype"""
        return X.astype(np.float32)


# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def plot_hist(df, feature_name, log=False, bins=100):
    """
    Plot histogram of a feature distribution
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_name (str): Column name to plot
        log (bool): Use log scale on y-axis
        bins (int): Number of histogram bins
    """
    plt.figure(figsize=(10, 6))
    
    if log:
        plt.hist(df[feature_name].dropna(), bins=bins, log=True)
    else:
        plt.hist(df[feature_name].dropna(), bins=bins)
    
    plt.title(f"Distribution of {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(f"data/graphs/histogram-{feature_name}.png")


def plot_barh(df, feature_name, log=False, title="", xlabel=""):
    """
    Plot horizontal bar chart for categorical feature value counts
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_name (str): Column name to plot
        log (bool): Use log scale on x-axis
        title (str): Chart title
        xlabel (str): X-axis label
    """
    plt.figure(figsize=(10, 6))
    
    if log:
        df[feature_name].value_counts().plot(kind="barh", logx=True)
    else:
        df[feature_name].value_counts().plot(kind="barh")
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(feature_name)
    plt.savefig(f"data/graphs/barh-{feature_name}-{title}.png")


def plot_grouped_bar(df, index, columns, values, title=None, xlabel="", ylabel=""):
    """
    Plot stacked grouped bar chart for multi-dimensional categorical data
    
    Args:
        df (pd.DataFrame): Input dataframe
        index (str): Column for x-axis groups
        columns (str): Column for stacked bars
        values (str): Column name for bar heights
        title (str): Chart title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
    """
    # Aggregate data by groups
    df_temp = df.groupby([index, columns]).size().reset_index()
    df_temp.columns = [index, columns, values]
    
    # Pivot to format for stacked bar chart
    df_pivot = pd.pivot(df_temp, index=index, columns=columns, values=values).rename_axis(None, axis=1)
    
    # Create stacked bar plot
    df_pivot.plot.bar(stacked=True, figsize=(20, 6), title=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"data/graphs/{title}-{xlabel}-{ylabel}-{uuid4()}.png")


def plot_roc(y_true, y_prob, title):
    """
    Plot ROC (Receiver Operating Characteristic) curve
    
    Shows trade-off between True Positive Rate and False Positive Rate
    
    Args:
        y_true (array): True binary labels
        y_prob (array): Predicted probabilities
        title (str): Chart title
    """
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")  # Diagonal reference line (random classifier)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"data/graphs/roc-curve-{title}-{uuid4()}.png")


# ============================================
# DATA PROCESSING FUNCTIONS
# ============================================

def check_future_failure(current_time, failure_times, window_hours):
    """
    Check if machine will fail within specified time window
    
    Used to create target labels for predictive maintenance
    
    Args:
        current_time (pd.Timestamp): Current timestamp
        failure_times (list): List of failure timestamps for machine
        window_hours (int): Prediction window in hours (e.g., 24 or 48)
        
    Returns:
        int: 1 if failure occurs within window, 0 otherwise
    """
    if not failure_times:
        return 0
    
    window = pd.Timedelta(hours=window_hours)
    
    # Check if any failure falls within the prediction window
    for ft in failure_times:
        dt = ft - current_time
        if pd.Timedelta(0) <= dt <= window:
            return 1
    
    return 0


def add_error_flags_per_machine(telemetry_df, errors_df, error_label):
    """
    Add binary flags indicating if error occurred in last 24h/48h
    
    Uses efficient merge_asof for time-based matching
    
    Args:
        telemetry_df (pd.DataFrame): Telemetry data with datetime and machineID
        errors_df (pd.DataFrame): Error logs with datetime, machineID, errorID
        error_label (str): Error type (e.g., 'error1', 'error2')
        
    Returns:
        pd.DataFrame: Telemetry with added error flag columns
    """
    timecol = f"{error_label}_time"
    
    # Initialize flag columns
    telemetry_df[f"{error_label}_last_24h"] = 0
    telemetry_df[f"{error_label}_last_48h"] = 0
    
    # Filter errors for this error type and rename datetime column
    e = errors_df[errors_df["errorID"] == error_label].rename(columns={"datetime": timecol})
    
    machine_ids = telemetry_df["machineID"].unique().tolist()

    # Process each machine separately for efficiency
    for m in machine_ids:
        # Get telemetry records for this machine
        left_mask = telemetry_df["machineID"] == m
        left = telemetry_df.loc[left_mask, ["datetime"]].sort_values("datetime")
        
        # Get error records for this machine
        right = e.loc[e["machineID"] == m, [timecol]].sort_values(timecol)
        
        if right.empty:
            continue

        # Find most recent error before each telemetry timestamp
        merged = pd.merge_asof(
            left, right, 
            left_on="datetime", 
            right_on=timecol,
            direction="backward",  # Look backward in time
            allow_exact_matches=True
        )
        
        # Calculate time since last error in hours
        delta_h = (merged["datetime"] - merged[timecol]).dt.total_seconds() / 3600.0
        has_prev = merged[timecol].notna()
        
        # Set flags based on time windows
        telemetry_df.loc[left.index, f"{error_label}_last_24h"] = (
            ((delta_h <= 24) & has_prev).astype(int).values
        )
        telemetry_df.loc[left.index, f"{error_label}_last_48h"] = (
            ((delta_h <= 48) & has_prev).astype(int).values
        )

    return telemetry_df


def add_time_since_maint(telemetry_df, maint_df, comp_label):
    """
    Add features for time elapsed since last maintenance of component
    
    Args:
        telemetry_df (pd.DataFrame): Telemetry data
        maint_df (pd.DataFrame): Maintenance logs with datetime, machineID, comp
        comp_label (str): Component name (e.g., 'comp1', 'comp2')
        
    Returns:
        pd.DataFrame: Telemetry with added time-since-maintenance columns
    """
    timecol = f"{comp_label}_maint_time"
    
    # Initialize columns with NA
    telemetry_df[f"time_since_maint_{comp_label}_h"] = pd.NA
    telemetry_df[f"time_since_maint_{comp_label}_d"] = pd.NA

    # Filter maintenance records for this component
    m = maint_df[maint_df["comp"] == comp_label].rename(columns={"datetime": timecol})
    machine_ids = telemetry_df["machineID"].unique().tolist()

    # Process each machine separately
    for m_id in machine_ids:
        # Get telemetry records for this machine
        left_mask = telemetry_df["machineID"] == m_id
        left = telemetry_df.loc[left_mask, ["datetime"]].sort_values("datetime")
        
        # Get maintenance records for this machine
        right = m.loc[m["machineID"] == m_id, [timecol]].sort_values(timecol)

        if right.empty:
            continue

        # Find most recent maintenance before each telemetry timestamp
        merged = pd.merge_asof(
            left, right, 
            left_on="datetime", 
            right_on=timecol,
            direction="backward",
            allow_exact_matches=True
        )
        
        # Calculate time since maintenance in hours and days
        delta_h = (merged["datetime"] - merged[timecol]).dt.total_seconds() / 3600.0
        telemetry_df.loc[left.index, f"time_since_maint_{comp_label}_h"] = delta_h.values
        telemetry_df.loc[left.index, f"time_since_maint_{comp_label}_d"] = (delta_h / 24).values

    return telemetry_df


def add_sensor_features(df, sensors, lags, rmeans, rstds, slopes):
    """
    Add time-series features from sensor data
    
    Creates lag, rolling statistics, and slope features for each sensor
    
    Args:
        df (pd.DataFrame): Input telemetry data
        sensors (list): List of sensor column names
        lags (list): Lag periods in hours
        rmeans (list): Rolling mean window sizes
        rstds (list): Rolling std window sizes
        slopes (list): Slope calculation periods
        
    Returns:
        pd.DataFrame: Data with added sensor features
    """
    # Group by machine to calculate features per machine
    g = df.groupby("machineID", group_keys=False)
    
    # LAG FEATURES: Previous values at specific time points
    for c in sensors:
        for k in lags:
            df[f"{c}_lag_{k}h"] = g[c].shift(k)
    
    # ROLLING MEANS: Average over time windows
    for c in sensors:
        for k in rmeans:
            df[f"{c}_mean_{k}h"] = (
                g[c].rolling(window=k, min_periods=k).mean()
                .reset_index(level=0, drop=True)
            )
    
    # ROLLING STD: Volatility over time windows
    for c in sensors:
        for k in rstds:
            df[f"{c}_std_{k}h"] = (
                g[c].rolling(window=k, min_periods=k).std(ddof=0)
                .reset_index(level=0, drop=True)
            )
    
    # SLOPE FEATURES: Rate of change
    for c in sensors:
        for k in slopes:
            lag_col = f"{c}_lag_{k}h"
            if lag_col not in df.columns:
                df[lag_col] = g[c].shift(k)
            df[f"{c}_slope_{k}h"] = (df[c] - df[lag_col]) / k
    
    return df


# ============================================
# MODEL TRAINING UTILITIES
# ============================================

def build_split(X, y, train_mask, val_mask, test_mask, label, timestamps):
    """
    Build train/validation/test splits and print statistics
    
    Args:
        X (array): Feature matrix
        y (array): Target labels
        train_mask (array): Boolean mask for training set
        val_mask (array): Boolean mask for validation set
        test_mask (array): Boolean mask for test set
        label (str): Dataset label (e.g., '24h', '48h')
        timestamps (array): Timestamps for date range tracking
        
    Returns:
        OrderedDict: Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test
    """
    ds = OrderedDict()
    
    # Create splits
    ds[f"X_train_{label}"] = X[train_mask]
    ds[f"y_train_{label}"] = y[train_mask]
    ds[f"X_val_{label}"] = X[val_mask]
    ds[f"y_val_{label}"] = y[val_mask]
    ds[f"X_test_{label}"] = X[test_mask]
    ds[f"y_test_{label}"] = y[test_mask]

    # Print split statistics
    print(f"\n== {label.upper()} SPLIT STATISTICS ===")
    for split in ["train", "val", "test"]:
        y_split = ds[f"y_{split}_{label}"]
        pos = int(y_split.sum())
        n = int(y_split.shape[0])
        prop = pos / max(n, 1)
        print(f"{split:>5}: n={n:,} | positives={pos:,} ({prop: .4%})")

    # Print date ranges for each split
    for split, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        print(f"{split:>5} range: {timestamps[mask].min()} → {timestamps[mask].max()}")

    return ds


def best_threshold_by_fbeta(y_true, y_prob, beta=2.0):
    """
    Find optimal classification threshold by maximizing F-beta score
    
    F-beta score weights recall higher than precision when beta > 1
    For predictive maintenance, we prefer high recall (catch all failures)
    
    Args:
        y_true (array): True binary labels
        y_prob (array): Predicted probabilities
        beta (float): Beta parameter for F-beta score (default 2.0)
        
    Returns:
        tuple: (optimal_threshold, best_fbeta_score)
    """
    # Get precision-recall curve
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.r_[t, 1.0]  # Add 1.0 threshold at end
    
    # Calculate F-beta score for each threshold
    fbeta = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)
    
    # Find threshold with maximum F-beta
    idx = np.nanargmax(fbeta)
    
    return float(t[idx]), float(fbeta[idx])


def metrics_block(y_true, y_prob, threshold):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true (array): True binary labels
        y_prob (array): Predicted probabilities
        threshold (float): Classification threshold
        
    Returns:
        dict: Dictionary containing all metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate specificity (True Negative Rate)
    specificity = tn / (tn + fp + 1e-12)
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred),
        "f2": fbeta_score(y_true, y_pred, beta=2.0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tp": tp, 
        "fp": fp, 
        "tn": tn, 
        "fn": fn,
    }


def print_confmat(y_true, y_prob, threshold, title):
    """
    Print confusion matrix and detailed classification report
    
    Args:
        y_true (array): True binary labels
        y_prob (array): Predicted probabilities
        threshold (float): Classification threshold
        title (str): Report title
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{title} | threshold={threshold:.3f}")
    print(cm)
    print(classification_report(y_true, y_pred, digits=4))