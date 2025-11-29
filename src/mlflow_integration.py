"""
MLflow Integration Module - Experiment Tracking and Model Registry
Provides comprehensive experiment tracking, model versioning, and artifact management
Clean implementation without code duplication - uses model_definitions.py
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, 
    precision_score, accuracy_score, confusion_matrix
)
import joblib
import json
from typing import Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


# ============================================
# MLFLOW TRACKER CLASS
# ============================================

class MLflowTracker:
    """
    MLflow Manager for comprehensive experiment tracking and model registry
    
    Features:
    - Automatic experiment creation and management
    - Comprehensive metric and parameter logging
    - Model versioning and registration
    - Artifact management (models, plots, feature importance)
    - Run comparison utilities
    """
    
    def __init__(self, experiment_name: str = "predictive_maintenance"):
        """
        Initialize MLflow tracker with experiment
        
        Args:
            experiment_name (str): Name of the MLflow experiment
        """
        # Set or create experiment
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        
        print(f"✅ MLflow Experiment: {experiment_name}")
        print(f"📁 Artifact Location: {self.experiment.artifact_location}")
    
    def start_run(self, run_name: str, tags: Optional[Dict] = None):
        """
        Start a new MLflow run with optional metadata tags
        
        Args:
            run_name (str): Descriptive name for this run (e.g., "LR_24h_v1")
            tags (dict, optional): Additional metadata tags
        
        Returns:
            MLflow run context manager
        """
        return mlflow.start_run(run_name=run_name, tags=tags or {})
    
    def log_data_info(self, data: pd.DataFrame, prefix: str = "data"):
        """
        Log comprehensive dataset information and statistics
        
        Args:
            data (pd.DataFrame): Input dataset
            prefix (str): Prefix for parameter names
        """
        # Log basic dataset statistics
        mlflow.log_param(f"{prefix}_total_samples", len(data))
        mlflow.log_param(f"{prefix}_num_features", len(data.columns))
        
        # Log temporal information if available
        if 'datetime' in data.columns:
            mlflow.log_param(f"{prefix}_date_start", str(data['datetime'].min()))
            mlflow.log_param(f"{prefix}_date_end", str(data['datetime'].max()))
        
        # Log machine coverage if available
        if 'machineID' in data.columns:
            mlflow.log_param(f"{prefix}_num_machines", data['machineID'].nunique())
        
        print(f"✅ {prefix.capitalize()} info logged")
    
    def log_split_info(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Log detailed information about train/validation/test splits
        
        Logs for each split:
        - Total number of samples
        - Number of positive samples
        - Class imbalance ratio
        
        Args:
            X_train, y_train: Training data and labels
            X_val, y_val: Validation data and labels
            X_test, y_test: Test data and labels
        """
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        for split_name, (X, y) in splits.items():
            mlflow.log_metric(f"{split_name}_samples", len(y))
            mlflow.log_metric(f"{split_name}_positive", int(y.sum()))
            mlflow.log_metric(f"{split_name}_positive_ratio", float(y.mean()))
        
        print("✅ Split info logged")
    
    def log_model_params(self, pipeline, model_type: str):
        """
        Extract and log all model hyperparameters
        
        Args:
            pipeline: sklearn pipeline containing preprocessor and classifier
            model_type (str): Model identifier (LR, RF, XGB, etc.)
        """
        # Extract classifier from pipeline
        clf = pipeline.named_steps.get('clf')
        if clf is None:
            print("⚠️ No classifier found in pipeline")
            return
        
        # Get all hyperparameters
        params = clf.get_params()
        mlflow.log_param("model_type", model_type)
        
        # Log each parameter (filter out complex objects)
        for key, value in params.items():
            if value is None or isinstance(value, (int, float, str, bool)):
                mlflow.log_param(f"model_{key}", value)
            else:
                # Convert complex objects to string representation
                mlflow.log_param(f"model_{key}", str(value))
        
        print(f"✅ Model params logged ({model_type})")
    
    def log_evaluation_metrics(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        threshold: float, 
        prefix: str = "val"
    ) -> Dict[str, float]:
        """
        Calculate and log comprehensive evaluation metrics
        
        Metrics logged:
        - Accuracy, Precision, Recall, F1-score
        - ROC-AUC
        - Specificity (True Negative Rate)
        - Confusion Matrix
        
        Args:
            y_true (array): True binary labels
            y_prob (array): Predicted probabilities
            threshold (float): Decision threshold for binary classification
            prefix (str): Metric prefix ("val" or "test")
        
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        # Convert probabilities to predictions using threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate core metrics
        metrics = {
            f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
            f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
            f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
            f"{prefix}_roc_auc": roc_auc_score(y_true, y_prob),
            f"{prefix}_threshold": threshold,
        }
        
        # Log all metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Calculate and save confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_dict = {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1])
        }
        
        # Calculate specificity (True Negative Rate)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        mlflow.log_metric(f"{prefix}_specificity", specificity)
        
        # Save confusion matrix as JSON artifact
        import tempfile
        import os
        
        # Create temp file in system temp directory (works on Windows/Linux/Mac)
        temp_dir = tempfile.gettempdir()
        cm_path = os.path.join(temp_dir, f"{prefix}_confusion_matrix.json")
        
        with open(cm_path, 'w') as f:
            json.dump(cm_dict, f, indent=2)
        mlflow.log_artifact(cm_path, f"{prefix}_metrics")
        
        print(f"✅ {prefix.capitalize()} metrics logged")
        return metrics
    
    def log_feature_importance(
        self, 
        pipeline, 
        feature_names: list, 
        top_n: int = 20
    ):
        """
        Extract and log feature importance/coefficients
        
        Supports:
        - Tree-based models: feature_importances_
        - Linear models: coefficients
        
        Args:
            pipeline: Trained sklearn pipeline
            feature_names (list): List of feature names
            top_n (int): Number of top features to save
        """
        clf = pipeline.named_steps.get('clf')
        if clf is None:
            return
        
        # Get the actual feature names after preprocessing
        try:
            # Try to get feature names from the preprocessor
            prep = pipeline.named_steps.get('prep')
            if prep and hasattr(prep, 'get_feature_names_out'):
                transformed_feature_names = prep.get_feature_names_out()
            else:
                # Fallback to original feature names
                transformed_feature_names = feature_names
        except:
            transformed_feature_names = feature_names
        
        # ============================================
        # TREE-BASED MODELS: Feature Importances
        # ============================================
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            
            # Match number of features
            n_features = min(len(transformed_feature_names), len(importances))
            
            # Create importance DataFrame
            fi_df = pd.DataFrame({
                'feature': list(transformed_feature_names[:n_features]),
                'importance': importances[:n_features]
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Save as CSV artifact
            import tempfile
            import os
            temp_dir = tempfile.gettempdir()
            fi_path = os.path.join(temp_dir, "feature_importance.csv")
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path, "feature_importance")
            
            # Log top 10 features as individual metrics
            for idx, row in fi_df.head(10).iterrows():
                # Clean feature name for MLflow (no spaces, max 50 chars)
                safe_name = str(row['feature']).replace(' ', '_')[:50]
                mlflow.log_metric(f"fi_{safe_name}", row['importance'])
            
            print("✅ Feature importance logged")
        
        # ============================================
        # LINEAR MODELS: Coefficients
        # ============================================
        elif hasattr(clf, 'coef_'):
            coef = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
            
            # Match number of features
            n_features = min(len(transformed_feature_names), len(coef))
            
            # Create coefficients DataFrame
            coef_df = pd.DataFrame({
                'feature': list(transformed_feature_names[:n_features]),
                'coefficient': coef[:n_features],
                'abs_coefficient': np.abs(coef[:n_features])
            }).sort_values('abs_coefficient', ascending=False).head(top_n)
            
            # Save as CSV artifact
            import tempfile
            import os
            temp_dir = tempfile.gettempdir()
            coef_path = os.path.join(temp_dir, "coefficients.csv")
            coef_df.to_csv(coef_path, index=False)
            mlflow.log_artifact(coef_path, "coefficients")
            
            print("✅ Coefficients logged")
    
    def log_model(
        self, 
        pipeline, 
        model_name: str, 
        signature = None
    ):
        """
        Register trained model in MLflow Model Registry
        
        Args:
            pipeline: Trained sklearn pipeline
            model_name (str): Name for model registry
            signature: MLflow model signature (input/output schema)
        """
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature
        )
        print(f"✅ Model registered: {model_name}")
    
    def compare_runs(self, experiment_name: Optional[str] = None) -> pd.DataFrame:
        """
        Compare all runs within an experiment
        
        Args:
            experiment_name (str, optional): Experiment name (uses current if None)
        
        Returns:
            pd.DataFrame: DataFrame with all runs and their metrics
        """
        # Get experiment ID
        if experiment_name:
            exp = mlflow.get_experiment_by_name(experiment_name)
            exp_id = exp.experiment_id
        else:
            exp_id = self.experiment.experiment_id
        
        # Search and return all runs
        runs = mlflow.search_runs(experiment_ids=[exp_id])
        return runs


# ============================================
# COMPLETE TRAINING WORKFLOW
# ============================================

def train_with_mlflow(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    horizon: str,
    tracker: MLflowTracker,
    feature_names: Optional[list] = None,
    save_joblib: bool = True
) -> Tuple[Any, Dict]:
    """
    Complete end-to-end training workflow with MLflow integration
    
    Workflow:
    1. Start MLflow run with metadata
    2. Log all parameters (model, data splits)
    3. Train model
    4. Find optimal threshold on validation set
    5. Evaluate on validation and test sets
    6. Log feature importance/coefficients
    7. Register model in Model Registry
    8. Save joblib file (optional)
    
    Args:
        pipeline: sklearn pipeline to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        model_name (str): Model identifier (LR, RF, XGB)
        horizon (str): Time horizon (24h, 48h)
        tracker (MLflowTracker): Tracker instance
        feature_names (list, optional): Feature names for importance logging
        save_joblib (bool): Whether to save joblib artifact
    
    Returns:
        tuple: (trained_pipeline, metrics_dict)
    """
    # Create descriptive run name
    run_name = f"{model_name}_{horizon}"
    
    with tracker.start_run(
        run_name=run_name,
        tags={
            "model_type": model_name,
            "horizon": horizon,
            "framework": "sklearn"
        }
    ):
        # ============================================
        # STEP 1: Log Parameters
        # ============================================
        
        tracker.log_model_params(pipeline, model_name)
        mlflow.log_param("horizon", horizon)
        tracker.log_split_info(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # ============================================
        # STEP 2: Train Model
        # ============================================
        
        print(f"\n🔄 Training {run_name}...")
        pipeline.fit(X_train, y_train)
        print("✅ Training complete")
        
        # ============================================
        # STEP 3: Generate Predictions
        # ============================================
        
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # ============================================
        # STEP 4: Find Optimal Threshold (F2-score)
        # ============================================
        
        from functions import best_threshold_by_fbeta
        threshold, f2_val = best_threshold_by_fbeta(y_val.values, val_prob, beta=2.0)
        
        mlflow.log_metric("optimal_threshold", threshold)
        mlflow.log_metric("val_f2_at_threshold", f2_val)
        
        # ============================================
        # STEP 5: Evaluate and Log Metrics
        # ============================================
        
        val_metrics = tracker.log_evaluation_metrics(
            y_val.values, val_prob, threshold, prefix="val"
        )
        
        test_metrics = tracker.log_evaluation_metrics(
            y_test.values, test_prob, threshold, prefix="test"
        )
        
        # ============================================
        # STEP 6: Log Feature Importance
        # ============================================
        
        if feature_names:
            tracker.log_feature_importance(pipeline, feature_names)
        
        # ============================================
        # STEP 7: Register Model
        # ============================================
        
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, pipeline.predict_proba(X_train))
        
        tracker.log_model(
            pipeline,
            model_name=f"pm_{model_name}_{horizon}",
            signature=signature
        )
        
        # ============================================
        # STEP 8: Save Joblib (Optional)
        # ============================================
        
        if save_joblib:
            import tempfile
            import os
            temp_dir = tempfile.gettempdir()
            model_path = os.path.join(temp_dir, f"model_{horizon}_{model_name}.joblib")
            joblib.dump(pipeline, model_path)
            mlflow.log_artifact(model_path, "joblib_model")
            print(f"✅ Joblib saved: {model_path}")
        
        print(f"✅ {run_name} completed successfully")
        
        return pipeline, {"val": val_metrics, "test": test_metrics}


# ============================================
# EXAMPLE USAGE
# ============================================

def example_usage():
    """
    Complete example demonstrating MLflow integration workflow
    
    Shows:
    - Tracker initialization
    - Data loading and splitting
    - Model training with MLflow
    - Results comparison
    """
    from model_definitions import get_model
    from config import PROCESSED_DATA_DIR
    
    # ============================================
    # STEP 1: Initialize Tracker
    # ============================================
    
    tracker = MLflowTracker("predictive_maintenance_clean")
    
    # ============================================
    # STEP 2: Load Data
    # ============================================
    
    telemetry = pd.read_csv(
        PROCESSED_DATA_DIR / "telemetry.csv",
        parse_dates=["datetime"]
    )
    
    # ============================================
    # STEP 3: Prepare Features and Targets
    # ============================================
    
    exclude_cols = {"datetime", "machineID", "will_fail_in_24h", "will_fail_in_48h"}
    feature_cols = [c for c in telemetry.columns if c not in exclude_cols]
    
    X = telemetry[feature_cols]
    y_24 = telemetry["will_fail_in_24h"].astype(int)
    
    # ============================================
    # STEP 4: Time-Based Split
    # ============================================
    
    cut_train = pd.Timestamp("2015-09-30 23:59:59")
    cut_val = pd.Timestamp("2015-11-15 23:59:59")
    
    train_mask = telemetry["datetime"] <= cut_train
    val_mask = (telemetry["datetime"] > cut_train) & (telemetry["datetime"] <= cut_val)
    test_mask = telemetry["datetime"] > cut_val
    
    X_train, y_train = X[train_mask], y_24[train_mask]
    X_val, y_val = X[val_mask], y_24[val_mask]
    X_test, y_test = X[test_mask], y_24[test_mask]
    
    # ============================================
    # STEP 5: Get Model from Definitions
    # ============================================
    
    pipeline = get_model('LR', X_train, use_smote=True)
    
    # ============================================
    # STEP 6: Train with MLflow
    # ============================================
    
    model, metrics = train_with_mlflow(
        pipeline=pipeline,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        model_name="LR",
        horizon="24h",
        tracker=tracker,
        feature_names=feature_cols
    )
    
    # ============================================
    # STEP 7: Display Results
    # ============================================
    
    print("\n📊 Results:")
    print(f"Val F1: {metrics['val']['val_f1']:.3f}")
    print(f"Test F1: {metrics['test']['test_f1']:.3f}")


if __name__ == "__main__":
    example_usage()