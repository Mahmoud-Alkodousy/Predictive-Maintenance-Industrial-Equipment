"""
===============================================================================
DATA PREPROCESSING SCRIPT
===============================================================================
Creates features and targets for predictive maintenance model training
Run after data_acquisition.py to prepare data for modeling

KEY FEATURES:
1. ✅ Feature engineering (lags, rolling stats, slopes)
2. ✅ Target variable creation (24h, 48h failure prediction)
3. ✅ Error handling and validation
4. ✅ Progress tracking
5. ✅ All comments in English

Developer: Eng. Mahmoud Khalid Alkodousy
===============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# Custom modules
from data_acquisition import load_data
from config import (
    SENSORS, 
    LAGS, 
    ROLL_MEANS, 
    ROLL_STDS, 
    SLOPES_K, 
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    GRAPHS_DIR
)
from functions import (
    check_future_failure,
    add_error_flags_per_machine,
    add_time_since_maint,
    add_sensor_features,
)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directory_exists(directory: Path) -> None:
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path object or string path to create
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Directory ready: {directory}")

def validate_dataframe(
    df: pd.DataFrame, 
    name: str, 
    required_columns: List[str] = None
) -> bool:
    """
    Validate DataFrame structure and contents
    
    Args:
        df: DataFrame to validate
        name: Name for logging
        required_columns: List of required column names
    
    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        logger.error(f"❌ {name}: DataFrame is empty or None")
        return False
    
    logger.info(f"✅ {name}: {len(df):,} rows, {len(df.columns)} columns")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.error(f"❌ {name}: Missing required columns: {missing}")
            return False
    
    return True

def calculate_class_imbalance(
    series: pd.Series, 
    name: str
) -> Dict[str, float]:
    """
    Calculate and log class imbalance statistics
    
    Args:
        series: Binary target variable
        name: Name for logging
    
    Returns:
        Dictionary with class distribution
    """
    value_counts = series.value_counts()
    proportions = series.value_counts(normalize=True)
    
    logger.info(f"\n{name} class distribution:")
    for value in sorted(value_counts.index):
        count = value_counts[value]
        pct = proportions[value] * 100
        logger.info(f"  Class {value}: {count:,} samples ({pct:.2f}%)")
    
    return proportions.to_dict()

# ============================================================================
# TARGET VARIABLE CREATION
# ============================================================================

def create_failure_targets(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    horizons: List[int] = [24, 48]
) -> pd.DataFrame:
    """
    Create binary target variables for failure prediction
    
    Args:
        telemetry: Telemetry data with datetime and machineID
        failures: Failure data with datetime and machineID
        horizons: List of prediction horizons in hours
    
    Returns:
        Telemetry DataFrame with added target columns
    """
    logger.info("\n🎯 Creating failure target variables...")
    
    # Group failures by machine for efficient lookup
    failures_by_machine = failures.groupby("machineID")["datetime"].apply(list).to_dict()
    logger.info(f"  • Grouped failures for {len(failures_by_machine)} machines")
    
    # Create target for each horizon
    for horizon in horizons:
        target_col = f"will_fail_in_{horizon}h"
        logger.info(f"\n  📊 Processing {horizon}h prediction horizon...")
        
        # Initialize target column
        telemetry[target_col] = 0
        
        # Use tqdm for progress tracking
        tqdm.pandas(desc=f"    Assigning {horizon}h targets")
        
        telemetry[target_col] = telemetry.progress_apply(
            lambda row: check_future_failure(
                row["datetime"], 
                failures_by_machine.get(row["machineID"], []), 
                horizon
            ),
            axis=1
        ).astype("int8")
        
        # Calculate class distribution
        positive_samples = telemetry[target_col].sum()
        total_samples = len(telemetry)
        positive_pct = (positive_samples / total_samples) * 100
        
        logger.info(f"    ✓ {target_col}: {positive_samples:,} positive samples ({positive_pct:.2f}%)")
    
    return telemetry

# ============================================================================
# MACHINE INFORMATION MERGE
# ============================================================================

def merge_machine_info(
    telemetry: pd.DataFrame,
    machines: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge machine metadata into telemetry data
    
    Args:
        telemetry: Telemetry data
        machines: Machine metadata
    
    Returns:
        Merged DataFrame
    """
    logger.info("\n🔗 Merging machine information...")
    
    original_len = len(telemetry)
    
    telemetry = telemetry.merge(machines, on="machineID", how="left")
    
    if len(telemetry) != original_len:
        logger.warning(f"⚠️ Row count changed after merge: {original_len} → {len(telemetry)}")
    else:
        logger.info(f"  ✓ Merged successfully, {len(telemetry):,} rows maintained")
    
    # Sort by machine and time
    telemetry = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    logger.info(f"  ✓ Sorted by machineID and datetime")
    
    return telemetry

# ============================================================================
# ERROR FLAGS
# ============================================================================

def add_error_features(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame
) -> pd.DataFrame:
    """
    Add error occurrence flags within time windows
    
    Args:
        telemetry: Telemetry data
        errors: Error data
    
    Returns:
        Telemetry with error flag columns
    """
    logger.info("\n🚨 Adding error flags...")
    
    error_types = errors["errorID"].unique().tolist()
    logger.info(f"  • Processing {len(error_types)} error types")
    
    for error_type in tqdm(error_types, desc="  Processing error types"):
        telemetry = add_error_flags_per_machine(telemetry, errors, error_type)
    
    # Count generated error columns
    error_cols = [c for c in telemetry.columns if "error" in c.lower()]
    logger.info(f"  ✓ Generated {len(error_cols)} error flag columns")
    
    # Show summary of error occurrences
    if error_cols:
        error_counts = telemetry[error_cols].sum().sort_values(ascending=False)
        logger.info("\n  Top 5 error types by frequency:")
        for col, count in error_counts.head(5).items():
            logger.info(f"    • {col}: {count:,} occurrences")
    
    return telemetry

# ============================================================================
# MAINTENANCE FEATURES
# ============================================================================

def add_maintenance_features(
    telemetry: pd.DataFrame,
    maintenance: pd.DataFrame
) -> pd.DataFrame:
    """
    Add time since last maintenance for each component
    
    Args:
        telemetry: Telemetry data
        maintenance: Maintenance data
    
    Returns:
        Telemetry with maintenance time features
    """
    logger.info("\n🔧 Adding maintenance features...")
    
    # Sort maintenance records
    maintenance = maintenance.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    
    components = maintenance["comp"].unique().tolist()
    logger.info(f"  • Processing {len(components)} components")
    
    for component in tqdm(components, desc="  Processing components"):
        telemetry = add_time_since_maint(telemetry, maintenance, component)
    
    # Count generated maintenance columns
    maint_cols = [c for c in telemetry.columns if c.startswith("time_since_maint_")]
    logger.info(f"  ✓ Generated {len(maint_cols)} maintenance time columns")
    
    return telemetry

# ============================================================================
# SENSOR FEATURES
# ============================================================================

def add_sensor_feature_engineering(
    telemetry: pd.DataFrame,
    sensors: List[str],
    lags: List[int],
    roll_means: List[int],
    roll_stds: List[int],
    slopes_k: int
) -> pd.DataFrame:
    """
    Add engineered features from sensor data
    
    Creates:
    - Lag features: Previous values at different time steps
    - Rolling means: Moving average over windows
    - Rolling stds: Moving standard deviation over windows
    - Slopes: Trend indicators
    
    Args:
        telemetry: Telemetry data
        sensors: List of sensor column names
        lags: List of lag values to create
        roll_means: List of window sizes for rolling means
        roll_stds: List of window sizes for rolling stds
        slopes_k: Number of points for slope calculation
    
    Returns:
        Telemetry with engineered features
    """
    logger.info("\n⚙️ Engineering sensor features...")
    logger.info(f"  • Sensors: {len(sensors)}")
    logger.info(f"  • Lags: {lags}")
    logger.info(f"  • Rolling mean windows: {roll_means}")
    logger.info(f"  • Rolling std windows: {roll_stds}")
    logger.info(f"  • Slope points: {slopes_k}")
    
    original_cols = len(telemetry.columns)
    
    telemetry = add_sensor_features(
        telemetry, 
        sensors, 
        lags, 
        roll_means, 
        roll_stds, 
        slopes_k
    )
    
    new_cols = len(telemetry.columns) - original_cols
    logger.info(f"  ✓ Generated {new_cols} new sensor features")
    
    # Categorize features
    feature_types = {
        'lag': [c for c in telemetry.columns if '_lag_' in c],
        'mean': [c for c in telemetry.columns if '_mean_' in c],
        'std': [c for c in telemetry.columns if '_std_' in c],
        'slope': [c for c in telemetry.columns if '_slope_' in c]
    }
    
    logger.info("\n  Feature breakdown:")
    for feat_type, cols in feature_types.items():
        logger.info(f"    • {feat_type.capitalize()}: {len(cols)} features")
    
    return telemetry

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_telemetry(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    machines: pd.DataFrame,
    errors: pd.DataFrame,
    maintenance: pd.DataFrame
) -> pd.DataFrame:
    """
    Complete telemetry processing pipeline
    
    Orchestrates all preprocessing steps:
    1. Create target variables
    2. Merge machine information
    3. Add error flags
    4. Add maintenance features
    5. Engineer sensor features
    
    Args:
        telemetry: Raw telemetry data
        failures: Failure records
        machines: Machine metadata
        errors: Error records
        maintenance: Maintenance records
    
    Returns:
        Fully processed telemetry DataFrame
    """
    logger.info("=" * 70)
    logger.info("🚀 STARTING TELEMETRY PROCESSING PIPELINE")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Create target variables
        telemetry = create_failure_targets(telemetry, failures, horizons=[24, 48])
        
        # Step 2: Merge machine information
        telemetry = merge_machine_info(telemetry, machines)
        
        # Step 3: Add error flags
        telemetry = add_error_features(telemetry, errors)
        
        # Step 4: Add maintenance features
        telemetry = add_maintenance_features(telemetry, maintenance)
        
        # Step 5: Engineer sensor features
        telemetry = add_sensor_feature_engineering(
            telemetry,
            SENSORS,
            LAGS,
            ROLL_MEANS,
            ROLL_STDS,
            SLOPES_K
        )
        
        # Calculate processing time
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✅ Processing completed in {elapsed_time:.2f} seconds")
        
        return telemetry
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(
    telemetry: pd.DataFrame,
    sensors: List[str],
    output_dir: Path
) -> None:
    """
    Create and save diagnostic visualizations
    
    Args:
        telemetry: Processed telemetry data
        sensors: List of sensor columns
        output_dir: Directory to save plots
    """
    logger.info("\n📊 Creating visualizations...")
    
    ensure_directory_exists(output_dir)
    
    try:
        # 1. Sensor distributions
        logger.info("  • Creating sensor distribution histograms...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Sensor Value Distributions", fontsize=16)
        
        for idx, sensor in enumerate(sensors):
            ax = axes[idx // 2, idx % 2]
            telemetry[sensor].hist(bins=50, ax=ax, edgecolor='black')
            ax.set_title(f"{sensor.capitalize()} Distribution")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(output_dir / "sensor_distributions.png", dpi=150)
        plt.close()
        logger.info("    ✓ Saved: sensor_distributions.png")
        
        # 2. Failure rate over time (24h)
        logger.info("  • Creating 24h failure rate timeline...")
        plt.figure(figsize=(14, 5))
        
        failure_rate_24h = telemetry.groupby("datetime")["will_fail_in_24h"].mean()
        failure_rate_24h.plot(linewidth=1.5, color='#e74c3c')
        
        plt.title("Failure Rate Over Time (24h Horizon)", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Failure Rate", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "failure_rate_24h.png", dpi=150)
        plt.close()
        logger.info("    ✓ Saved: failure_rate_24h.png")
        
        # 3. Failure rate over time (48h)
        logger.info("  • Creating 48h failure rate timeline...")
        plt.figure(figsize=(14, 5))
        
        failure_rate_48h = telemetry.groupby("datetime")["will_fail_in_48h"].mean()
        failure_rate_48h.plot(linewidth=1.5, color='#3498db')
        
        plt.title("Failure Rate Over Time (48h Horizon)", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Failure Rate", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "failure_rate_48h.png", dpi=150)
        plt.close()
        logger.info("    ✓ Saved: failure_rate_48h.png")
        
        # 4. Class imbalance visualization
        logger.info("  • Creating class imbalance chart...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 24h
        counts_24h = telemetry["will_fail_in_24h"].value_counts()
        ax1.bar(['No Failure', 'Failure'], counts_24h.values, color=['#2ecc71', '#e74c3c'])
        ax1.set_title("24h Prediction Target", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Count")
        for i, v in enumerate(counts_24h.values):
            ax1.text(i, v, f'{v:,}', ha='center', va='bottom')
        
        # 48h
        counts_48h = telemetry["will_fail_in_48h"].value_counts()
        ax2.bar(['No Failure', 'Failure'], counts_48h.values, color=['#2ecc71', '#e74c3c'])
        ax2.set_title("48h Prediction Target", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Count")
        for i, v in enumerate(counts_48h.values):
            ax2.text(i, v, f'{v:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "class_imbalance.png", dpi=150)
        plt.close()
        logger.info("    ✓ Saved: class_imbalance.png")
        
        logger.info("✅ All visualizations created successfully")
        
    except Exception as e:
        logger.warning(f"⚠️ Visualization creation failed: {e}")
        logger.warning("Continuing without visualizations...")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_preprocessing_summary(telemetry: pd.DataFrame) -> None:
    """
    Print comprehensive summary of preprocessing results
    
    Args:
        telemetry: Processed telemetry DataFrame
    """
    logger.info("\n" + "=" * 70)
    logger.info("📋 PREPROCESSING SUMMARY")
    logger.info("=" * 70)
    
    # Basic stats
    logger.info(f"\n📊 Dataset Dimensions:")
    logger.info(f"  • Total rows: {len(telemetry):,}")
    logger.info(f"  • Total columns: {len(telemetry.columns):,}")
    logger.info(f"  • Memory usage: {telemetry.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Target variables
    logger.info(f"\n🎯 Target Variables:")
    
    for target in ["will_fail_in_24h", "will_fail_in_48h"]:
        if target in telemetry.columns:
            proportions = telemetry[target].value_counts(normalize=True)
            positive_pct = proportions.get(1, 0) * 100
            negative_pct = proportions.get(0, 0) * 100
            
            logger.info(f"\n  {target}:")
            logger.info(f"    • Negative (0): {negative_pct:.2f}%")
            logger.info(f"    • Positive (1): {positive_pct:.2f}%")
            logger.info(f"    • Imbalance ratio: 1:{negative_pct/positive_pct:.1f}")
    
    # Feature categories
    logger.info(f"\n🔧 Feature Categories:")
    
    feature_categories = {
        'Sensors': [c for c in telemetry.columns if c in SENSORS],
        'Lag features': [c for c in telemetry.columns if '_lag_' in c],
        'Rolling means': [c for c in telemetry.columns if '_mean_' in c],
        'Rolling stds': [c for c in telemetry.columns if '_std_' in c],
        'Slopes': [c for c in telemetry.columns if '_slope_' in c],
        'Error flags': [c for c in telemetry.columns if 'error' in c.lower()],
        'Maintenance': [c for c in telemetry.columns if c.startswith('time_since_maint_')]
    }
    
    for category, columns in feature_categories.items():
        logger.info(f"  • {category}: {len(columns)} features")
    
    # Missing values
    logger.info(f"\n❓ Missing Values:")
    missing_counts = telemetry.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) > 0:
        logger.info(f"  • Columns with missing values: {len(cols_with_missing)}")
        logger.info(f"  • Total missing cells: {missing_counts.sum():,}")
        logger.info("\n  Top 5 columns with most missing values:")
        for col, count in cols_with_missing.nlargest(5).items():
            pct = (count / len(telemetry)) * 100
            logger.info(f"    • {col}: {count:,} ({pct:.2f}%)")
    else:
        logger.info("  ✅ No missing values found")
    
    logger.info("\n" + "=" * 70)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main preprocessing pipeline
    
    Orchestrates the complete preprocessing workflow:
    1. Load raw data
    2. Process telemetry with all feature engineering
    3. Save processed data
    4. Create visualizations
    5. Print summary
    """
    logger.info("=" * 70)
    logger.info("🎬 STARTING DATA PREPROCESSING")
    logger.info("=" * 70)
    
    overall_start = datetime.now()
    
    try:
        # Step 1: Load raw data
        logger.info("\n📂 Step 1/5: Loading raw data...")
        telemetry, errors, maintenance, failures, machines = load_data()
        
        # Validate loaded data
        validate_dataframe(telemetry, "Telemetry", ["machineID", "datetime"])
        validate_dataframe(failures, "Failures", ["machineID", "datetime"])
        validate_dataframe(machines, "Machines", ["machineID"])
        validate_dataframe(errors, "Errors", ["machineID", "datetime", "errorID"])
        validate_dataframe(maintenance, "Maintenance", ["machineID", "datetime", "comp"])
        
        # Step 2: Process telemetry
        logger.info("\n⚙️ Step 2/5: Processing telemetry...")
        telemetry = process_telemetry(telemetry, failures, machines, errors, maintenance)
        
        # Step 3: Save processed data
        logger.info("\n💾 Step 3/5: Saving processed data...")
        ensure_directory_exists(PROCESSED_DATA_DIR)
        
        output_path = PROCESSED_DATA_DIR / "telemetry.csv"
        telemetry.to_csv(output_path, index=False)
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"  ✓ Saved to: {output_path}")
        logger.info(f"  ✓ File size: {file_size:.2f} MB")
        
        # Step 4: Create visualizations
        logger.info("\n📊 Step 4/5: Creating visualizations...")
        create_visualizations(telemetry, SENSORS, GRAPHS_DIR)
        
        # Step 5: Print summary
        logger.info("\n📋 Step 5/5: Generating summary...")
        print_preprocessing_summary(telemetry)
        
        # Final timing
        total_time = (datetime.now() - overall_start).total_seconds()
        logger.info(f"\n🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️ Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info(f"\n🎯 Next step: Run model training script")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ PREPROCESSING FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point when script is run directly
    """
    import sys
    
    success = main()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)