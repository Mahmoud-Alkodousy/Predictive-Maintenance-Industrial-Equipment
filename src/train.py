"""
Complete Training Pipeline
Trains all models (LR and RF) for both horizons (24h and 48h)
Uses model_definitions.py and mlflow_integration_clean.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import argparse
from datetime import datetime

# Import project modules
from config import (
    PROCESSED_DATA_DIR, 
    MODELS_DIR, 
    MODEL_FILES,
    RANDOM_STATE
)
from functions import build_split
from model_definitions import get_model, list_available_models
from mlflow_integration import MLflowTracker, train_with_mlflow


def load_processed_data():
    """
    Load preprocessed telemetry data
    
    Returns:
        DataFrame with all features and targets
    """
    data_path = PROCESSED_DATA_DIR / "telemetry.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"❌ Processed data not found at {data_path}\n"
            "Please run: python data_preprocessing.py"
        )
    
    print(f"📂 Loading data from {data_path}...")
    telemetry = pd.read_csv(data_path, parse_dates=["datetime"])
    print(f"✅ Loaded {len(telemetry):,} rows × {len(telemetry.columns)} columns")
    
    return telemetry


def prepare_features_and_targets(telemetry):
    """
    Separate features and targets
    
    Args:
        telemetry: Full dataframe
    
    Returns:
        (X, y_24h, y_48h, feature_names)
    """
    # Define columns to exclude
    exclude_cols = {"datetime", "machineID", "will_fail_in_24h", "will_fail_in_48h"}
    
    # Get feature columns
    feature_cols = [c for c in telemetry.columns if c not in exclude_cols]
    
    print(f"\n📊 Feature Summary:")
    print(f"   Total features: {len(feature_cols)}")
    
    # Count feature types
    sensor_features = [c for c in feature_cols if any(s in c for s in ['volt', 'rotate', 'pressure', 'vibration'])]
    error_features = [c for c in feature_cols if 'error' in c]
    maint_features = [c for c in feature_cols if 'maint' in c]
    
    print(f"   - Sensor features: {len(sensor_features)}")
    print(f"   - Error features: {len(error_features)}")
    print(f"   - Maintenance features: {len(maint_features)}")
    print(f"   - Other features: {len(feature_cols) - len(sensor_features) - len(error_features) - len(maint_features)}")
    
    # Prepare X and y
    X = telemetry[feature_cols]
    y_24 = telemetry["will_fail_in_24h"].astype(int)
    y_48 = telemetry["will_fail_in_48h"].astype(int)
    
    # Check class balance
    print(f"\n⚖️  Class Balance:")
    print(f"   24h target: {y_24.mean():.2%} positive")
    print(f"   48h target: {y_48.mean():.2%} positive")
    
    return X, y_24, y_48, feature_cols


def create_time_based_splits(telemetry, X, y_24, y_48):
    """
    Create time-based train/val/test splits
    
    Args:
        telemetry: Full dataframe with datetime
        X: Features
        y_24: 24h target
        y_48: 48h target
    
    Returns:
        (splits_24h, splits_48h)
    """
    # Define cutoff dates
    cut_train = pd.Timestamp("2015-09-30 23:59:59")
    cut_val = pd.Timestamp("2015-11-15 23:59:59")
    
    # Create masks
    train_mask = telemetry["datetime"] <= cut_train
    val_mask = (telemetry["datetime"] > cut_train) & (telemetry["datetime"] <= cut_val)
    test_mask = telemetry["datetime"] > cut_val
    
    print(f"\n📅 Time-based Split:")
    print(f"   Train: {telemetry['datetime'].min()} → {cut_train}")
    print(f"   Val:   {cut_train} → {cut_val}")
    print(f"   Test:  {cut_val} → {telemetry['datetime'].max()}")
    
    # Build splits
    splits_24 = build_split(
        X, y_24,
        train_mask, val_mask, test_mask,
        "24h", telemetry["datetime"]
    )
    
    splits_48 = build_split(
        X, y_48,
        train_mask, val_mask, test_mask,
        "48h", telemetry["datetime"]
    )
    
    return splits_24, splits_48


def train_single_model(
    model_name,
    horizon,
    splits,
    feature_names,
    tracker,
    use_smote=True
):
    """
    Train a single model
    
    Args:
        model_name: Model identifier (LR, RF_fast, etc.)
        horizon: 24h or 48h
        splits: Dictionary from build_split()
        feature_names: List of feature names
        tracker: MLflowTracker instance
        use_smote: Whether to use SMOTE
    
    Returns:
        (trained_model, metrics)
    """
    print(f"\n{'='*60}")
    print(f"🚀 Training: {model_name} - {horizon}")
    print(f"{'='*60}")
    
    # Extract data from splits
    X_train = splits[f"X_train_{horizon}"]
    y_train = splits[f"y_train_{horizon}"]
    X_val = splits[f"X_val_{horizon}"]
    y_val = splits[f"y_val_{horizon}"]
    X_test = splits[f"X_test_{horizon}"]
    y_test = splits[f"y_test_{horizon}"]
    
    # Get model
    print(f"🔧 Building {model_name} pipeline...")
    pipeline = get_model(model_name, X_train, use_smote=use_smote)
    
    # Train with MLflow
    model, metrics = train_with_mlflow(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        horizon=horizon,
        tracker=tracker,
        feature_names=feature_names,
        save_joblib=True
    )
    
    # Print results
    print(f"\n📊 Results for {model_name}_{horizon}:")
    print(f"   Val  - F1: {metrics['val']['val_f1']:.3f} | ROC-AUC: {metrics['val']['val_roc_auc']:.3f}")
    print(f"   Test - F1: {metrics['test']['test_f1']:.3f} | ROC-AUC: {metrics['test']['test_roc_auc']:.3f}")
    
    return model, metrics


def save_model_to_file(model, model_name, horizon):
    """
    Save model to standard location
    
    Args:
        model: Trained pipeline
        model_name: Model identifier
        horizon: 24h or 48h
    """
    # Determine save path based on naming convention
    if model_name == "LR":
        save_path = MODEL_FILES[f"LR_{horizon}"]
    elif model_name == "RF_fast":
        save_path = MODEL_FILES[f"RF_{horizon}"]
    else:
        # Custom path for other models
        save_path = MODELS_DIR / f"model_{horizon}_{model_name}.joblib"
    
    # Save
    joblib.dump(model, save_path)
    print(f"💾 Model saved: {save_path}")
    
    return save_path


def main(
    models_to_train=None,
    horizons=None,
    use_smote=True,
    experiment_name="predictive_maintenance"
):
    """
    Main training pipeline
    
    Args:
        models_to_train: List of model names (default: ['LR', 'RF_fast'])
        horizons: List of horizons (default: ['24h', '48h'])
        use_smote: Whether to use SMOTE
        experiment_name: MLflow experiment name
    """
    print("\n" + "="*60)
    print("🚀 PREDICTIVE MAINTENANCE - TRAINING PIPELINE")
    print("="*60)
    
    # Set defaults
    if models_to_train is None:
        models_to_train = ['LR', 'RF_fast']
    
    if horizons is None:
        horizons = ['24h', '48h']
    
    print(f"\n⚙️  Configuration:")
    print(f"   Models: {models_to_train}")
    print(f"   Horizons: {horizons}")
    print(f"   SMOTE: {use_smote}")
    print(f"   Experiment: {experiment_name}")
    
    # Initialize MLflow
    tracker = MLflowTracker(experiment_name)
    
    # Load data
    telemetry = load_processed_data()
    
    # Prepare features and targets
    X, y_24, y_48, feature_names = prepare_features_and_targets(telemetry)
    
    # Create splits
    splits_24, splits_48 = create_time_based_splits(telemetry, X, y_24, y_48)
    
    # Training loop
    results = {}
    start_time = datetime.now()
    
    for model_name in models_to_train:
        for horizon in horizons:
            try:
                # Select appropriate splits
                splits = splits_24 if horizon == '24h' else splits_48
                
                # Train model
                model, metrics = train_single_model(
                    model_name=model_name,
                    horizon=horizon,
                    splits=splits,
                    feature_names=feature_names,
                    tracker=tracker,
                    use_smote=use_smote
                )
                
                # Save to standard location
                save_path = save_model_to_file(model, model_name, horizon)
                
                # Store results
                key = f"{model_name}_{horizon}"
                results[key] = {
                    'model': model,
                    'metrics': metrics,
                    'path': save_path
                }
                
            except Exception as e:
                print(f"\n❌ Error training {model_name}_{horizon}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Training complete
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"⏱️  Total time: {duration/60:.1f} minutes")
    print(f"📦 Models trained: {len(results)}")
    
    # Summary table
    print("\n📊 Results Summary:")
    print(f"{'Model':<20} {'Horizon':<10} {'Val F1':<10} {'Test F1':<10} {'Test ROC-AUC':<15}")
    print("-" * 65)
    
    for key, result in results.items():
        model_name, horizon = key.rsplit('_', 1)
        val_f1 = result['metrics']['val']['val_f1']
        test_f1 = result['metrics']['test']['test_f1']
        test_auc = result['metrics']['test']['test_roc_auc']
        
        print(f"{model_name:<20} {horizon:<10} {val_f1:<10.3f} {test_f1:<10.3f} {test_auc:<15.3f}")
    
    # Compare runs in MLflow
    print("\n🔍 Comparing all runs in MLflow...")
    comparison = tracker.compare_runs()
    
    if not comparison.empty:
        print(f"\n📈 Top 5 models by Test F1:")
        top_models = comparison.nlargest(5, 'metrics.test_f1')[
            ['tags.model_type', 'tags.horizon', 'metrics.test_f1', 'metrics.test_roc_auc']
        ]
        print(top_models.to_string(index=False))
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train predictive maintenance models')
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['LR', 'RF_fast'],
        choices=list_available_models(),
        help='Models to train'
    )
    
    parser.add_argument(
        '--horizons',
        nargs='+',
        default=['24h', '48h'],
        choices=['24h', '48h'],
        help='Time horizons to train'
    )
    
    parser.add_argument(
        '--no-smote',
        action='store_true',
        help='Disable SMOTE oversampling'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default='predictive_maintenance',
        help='MLflow experiment name'
    )
    
    args = parser.parse_args()
    
    # Run training
    results = main(
        models_to_train=args.models,
        horizons=args.horizons,
        use_smote=not args.no_smote,
        experiment_name=args.experiment
    )
    
    print("\n✅ Training pipeline finished successfully!")
