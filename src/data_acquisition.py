"""
===============================================================================
DATA ACQUISITION SCRIPT
===============================================================================
Downloads predictive maintenance dataset from Kaggle and saves locally
Run once to get the raw data before preprocessing

KEY FEATURES:
1. ✅ Downloads from Kaggle API
2. ✅ Proper error handling
3. ✅ Progress tracking
4. ✅ Data validation
5. ✅ All comments in English

Developer: Eng. Mahmoud Khalid Alkodousy
===============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import kagglehub
import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Optional
import logging
from datetime import datetime

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Kaggle dataset identifier
KAGGLE_DATASET = "arnabbiswas1/microsoft-azure-predictive-maintenance"

# Expected dataset files
EXPECTED_FILES = [
    "PdM_telemetry.csv",
    "PdM_errors.csv",
    "PdM_maint.csv",
    "PdM_failures.csv",
    "PdM_machines.csv"
]

# Output filenames
OUTPUT_FILES = {
    "PdM_telemetry.csv": "telemetry.csv",
    "PdM_errors.csv": "errors.csv",
    "PdM_maint.csv": "maint.csv",
    "PdM_failures.csv": "failures.csv",
    "PdM_machines.csv": "machines.csv"
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directory_exists(path: str) -> Path:
    """
    Create directory if it doesn't exist
    
    Args:
        path: Directory path to create
    
    Returns:
        Path object of created/existing directory
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Directory ready: {directory.absolute()}")
    return directory

def validate_dataframe(df: pd.DataFrame, name: str, required_columns: list = None) -> bool:
    """
    Validate DataFrame has expected structure
    
    Args:
        df: DataFrame to validate
        name: Name of dataset for logging
        required_columns: List of required column names
    
    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        logger.error(f"❌ {name}: DataFrame is empty or None")
        return False
    
    logger.info(f"✅ {name}: {len(df)} rows, {len(df.columns)} columns")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.warning(f"⚠️ {name}: Missing columns: {missing}")
            return False
    
    # Check for null values
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        null_pct = (null_count / (len(df) * len(df.columns))) * 100
        logger.info(f"ℹ️ {name}: {null_count} null values ({null_pct:.2f}%)")
    
    return True

# ============================================================================
# MAIN DOWNLOAD FUNCTION
# ============================================================================

def download_from_kaggle(dataset_id: str = KAGGLE_DATASET) -> Optional[str]:
    """
    Download dataset from Kaggle
    
    Args:
        dataset_id: Kaggle dataset identifier
    
    Returns:
        Path to downloaded dataset or None if failed
    """
    try:
        logger.info(f"📥 Downloading dataset: {dataset_id}")
        logger.info("⏳ This may take a few minutes...")
        
        path = kagglehub.dataset_download(dataset_id)
        
        logger.info(f"✅ Download complete!")
        logger.info(f"📂 Dataset location: {path}")
        
        # Verify all expected files exist
        for file in EXPECTED_FILES:
            file_path = Path(path) / file
            if not file_path.exists():
                logger.error(f"❌ Missing file: {file}")
                return None
            else:
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"  ✓ {file} ({file_size:.2f} MB)")
        
        return path
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        logger.info("💡 Make sure you have:")
        logger.info("   1. Kaggle API installed: pip install kagglehub")
        logger.info("   2. Kaggle API credentials configured")
        logger.info("   3. Internet connection")
        return None

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_dataset_file(
    source_path: str, 
    filename: str, 
    parse_dates: list = None
) -> Optional[pd.DataFrame]:
    """
    Load a single dataset file with error handling
    
    Args:
        source_path: Directory containing the file
        filename: Name of file to load
        parse_dates: List of columns to parse as dates
    
    Returns:
        DataFrame or None if loading fails
    """
    try:
        file_path = Path(source_path) / filename
        
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            return None
        
        logger.info(f"📖 Loading: {filename}")
        
        df = pd.read_csv(
            file_path, 
            parse_dates=parse_dates if parse_dates else []
        )
        
        logger.info(f"  ✓ Loaded {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to load {filename}: {e}")
        return None

def load_all_datasets(source_path: str) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame]
]:
    """
    Load all dataset files from source directory
    
    Args:
        source_path: Directory containing dataset files
    
    Returns:
        Tuple of (telemetry, errors, maintenance, failures, machines) DataFrames
    """
    logger.info("📚 Loading all datasets...")
    
    # Load each dataset with appropriate date parsing
    telemetry_df = load_dataset_file(
        source_path, 
        "PdM_telemetry.csv", 
        parse_dates=["datetime"]
    )
    
    errors_df = load_dataset_file(
        source_path, 
        "PdM_errors.csv", 
        parse_dates=["datetime"]
    )
    
    maint_df = load_dataset_file(
        source_path, 
        "PdM_maint.csv", 
        parse_dates=["datetime"]
    )
    
    failures_df = load_dataset_file(
        source_path, 
        "PdM_failures.csv", 
        parse_dates=["datetime"]
    )
    
    machines_df = load_dataset_file(
        source_path, 
        "PdM_machines.csv"
    )
    
    # Check if all loaded successfully
    all_loaded = all([
        telemetry_df is not None,
        errors_df is not None,
        maint_df is not None,
        failures_df is not None,
        machines_df is not None
    ])
    
    if all_loaded:
        logger.info("✅ All datasets loaded successfully")
    else:
        logger.error("❌ Some datasets failed to load")
    
    return telemetry_df, errors_df, maint_df, failures_df, machines_df

# ============================================================================
# DATA PROCESSING
# ============================================================================

def sort_and_clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Sort DataFrame by machineID and datetime, remove duplicates
    
    Args:
        df: DataFrame to process
        name: Name for logging
    
    Returns:
        Processed DataFrame
    """
    try:
        original_len = len(df)
        
        # Sort if columns exist
        if 'machineID' in df.columns and 'datetime' in df.columns:
            df = df.sort_values(['machineID', 'datetime'])
            logger.info(f"  ✓ {name}: Sorted by machineID and datetime")
        elif 'machineID' in df.columns:
            df = df.sort_values('machineID')
            logger.info(f"  ✓ {name}: Sorted by machineID")
        
        # Remove duplicates
        df = df.drop_duplicates()
        duplicates_removed = original_len - len(df)
        
        if duplicates_removed > 0:
            logger.info(f"  ✓ {name}: Removed {duplicates_removed} duplicates")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        logger.warning(f"⚠️ {name}: Sorting/cleaning failed: {e}")
        return df

# ============================================================================
# DATA SAVING
# ============================================================================

def save_dataframe(
    df: pd.DataFrame, 
    destination_folder: str, 
    filename: str
) -> bool:
    """
    Save DataFrame to CSV file
    
    Args:
        df: DataFrame to save
        destination_folder: Destination directory
        filename: Output filename
    
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(destination_folder) / filename
        
        df.to_csv(output_path, index=False)
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"  ✓ Saved: {filename} ({file_size:.2f} MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save {filename}: {e}")
        return False

# ============================================================================
# MAIN SAVE FUNCTION
# ============================================================================

def save_data(destination_folder: str = "src/data/raw") -> bool:
    """
    Download dataset from Kaggle and save locally
    
    This is the main function that orchestrates the entire process:
    1. Downloads data from Kaggle
    2. Loads all datasets
    3. Validates data quality
    4. Sorts and cleans data
    5. Saves to destination folder
    
    Args:
        destination_folder: Where to save the processed data
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 70)
    logger.info("🚀 STARTING DATA ACQUISITION PROCESS")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Ensure destination directory exists
        logger.info("\n📁 Step 1/5: Creating destination directory...")
        dest_path = ensure_directory_exists(destination_folder)
        
        # Step 2: Download from Kaggle
        logger.info("\n📥 Step 2/5: Downloading from Kaggle...")
        source_path = download_from_kaggle()
        
        if source_path is None:
            logger.error("❌ Download failed. Aborting.")
            return False
        
        # Step 3: Load all datasets
        logger.info("\n📚 Step 3/5: Loading datasets...")
        telemetry_df, errors_df, maint_df, failures_df, machines_df = load_all_datasets(source_path)
        
        if any(df is None for df in [telemetry_df, errors_df, maint_df, failures_df, machines_df]):
            logger.error("❌ Some datasets failed to load. Aborting.")
            return False
        
        # Step 4: Validate and process
        logger.info("\n🔍 Step 4/5: Validating and processing...")
        
        datasets = {
            "Telemetry": telemetry_df,
            "Errors": errors_df,
            "Maintenance": maint_df,
            "Failures": failures_df,
            "Machines": machines_df
        }
        
        for name, df in datasets.items():
            if not validate_dataframe(df, name):
                logger.warning(f"⚠️ {name} validation issues detected")
        
        # Sort and clean datasets with temporal data
        logger.info("\n🔄 Sorting and cleaning...")
        telemetry_df = sort_and_clean(telemetry_df, "Telemetry")
        errors_df = sort_and_clean(errors_df, "Errors")
        maint_df = sort_and_clean(maint_df, "Maintenance")
        failures_df = sort_and_clean(failures_df, "Failures")
        machines_df = sort_and_clean(machines_df, "Machines")
        
        # Step 5: Save to destination
        logger.info("\n💾 Step 5/5: Saving to disk...")
        
        save_tasks = [
            (telemetry_df, "telemetry.csv"),
            (errors_df, "errors.csv"),
            (maint_df, "maint.csv"),
            (failures_df, "failures.csv"),
            (machines_df, "machines.csv")
        ]
        
        all_saved = True
        for df, filename in save_tasks:
            if not save_dataframe(df, destination_folder, filename):
                all_saved = False
        
        # Summary
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "=" * 70)
        if all_saved:
            logger.info("✅ DATA ACQUISITION COMPLETED SUCCESSFULLY!")
            logger.info(f"⏱️ Total time: {elapsed_time:.2f} seconds")
            logger.info(f"📂 Data saved to: {dest_path.absolute()}")
            logger.info("\n📊 Dataset Summary:")
            logger.info(f"   • Telemetry: {len(telemetry_df):,} rows")
            logger.info(f"   • Errors: {len(errors_df):,} rows")
            logger.info(f"   • Maintenance: {len(maint_df):,} rows")
            logger.info(f"   • Failures: {len(failures_df):,} rows")
            logger.info(f"   • Machines: {len(machines_df):,} rows")
            logger.info("\n🎯 Next step: Run data_preprocessing.py")
        else:
            logger.error("❌ SOME FILES FAILED TO SAVE")
        logger.info("=" * 70)
        
        return all_saved
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# ============================================================================
# DATA LOADING FUNCTION (FOR LATER USE)
# ============================================================================

def load_data(data_folder: str = "src/data/raw") -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame
]:
    """
    Load previously saved datasets from CSV files
    
    Use this function after running save_data() to load the processed data
    for further analysis or model training.
    
    Args:
        data_folder: Directory containing the CSV files
    
    Returns:
        Tuple of (telemetry, errors, maintenance, failures, machines) DataFrames
    
    Example:
        >>> telemetry, errors, maint, failures, machines = load_data()
        >>> print(f"Loaded {len(telemetry)} telemetry records")
    """
    logger.info("📖 Loading saved datasets...")
    
    try:
        telemetry_df = pd.read_csv(
            f"{data_folder}/telemetry.csv", 
            parse_dates=["datetime"]
        )
        logger.info(f"  ✓ Telemetry: {len(telemetry_df):,} rows")
        
        errors_df = pd.read_csv(
            f"{data_folder}/errors.csv", 
            parse_dates=["datetime"]
        )
        logger.info(f"  ✓ Errors: {len(errors_df):,} rows")
        
        maint_df = pd.read_csv(
            f"{data_folder}/maint.csv", 
            parse_dates=["datetime"]
        )
        logger.info(f"  ✓ Maintenance: {len(maint_df):,} rows")
        
        failures_df = pd.read_csv(
            f"{data_folder}/failures.csv", 
            parse_dates=["datetime"]
        )
        logger.info(f"  ✓ Failures: {len(failures_df):,} rows")
        
        machines_df = pd.read_csv(
            f"{data_folder}/machines.csv"
        )
        logger.info(f"  ✓ Machines: {len(machines_df):,} rows")
        
        logger.info("✅ All datasets loaded successfully")
        
        return telemetry_df, errors_df, maint_df, failures_df, machines_df
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        logger.info("💡 Run save_data() first to download and save the dataset")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        raise

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point when script is run directly
    Downloads and saves the dataset to src/data/raw directory
    """
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        destination = sys.argv[1]
        logger.info(f"Using custom destination: {destination}")
    else:
        destination = "src/data/raw"
        logger.info(f"Using default destination: {destination}")
    
    # Run data acquisition
    success = save_data(destination)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)