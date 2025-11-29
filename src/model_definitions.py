"""
Model Definitions Module
Contains all model architectures and pipeline builders
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from config import RANDOM_STATE


# ============================================
# PREPROCESSING TRANSFORMERS
# ============================================

def build_preprocessor(X, strategy='standard'):
    """
    Build preprocessing pipeline based on data characteristics
    
    FIXED: Added SimpleImputer to handle NaN values before scaling
    This fixes the SMOTE error with missing values
    
    Args:
        X: Training features
        strategy: 'standard', 'robust', or 'minmax'
    
    Returns:
        ColumnTransformer
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    transformers = []
    
    # Numeric features pipeline: Impute NaN → Scale
    if strategy == 'standard':
        scaler = StandardScaler()
    elif strategy == 'robust':
        scaler = RobustScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    # Create pipeline: Imputer → Scaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Fill NaN with mean
        ('scaler', scaler)
    ])
    
    transformers.append(('num', numeric_transformer, numeric_features))
    
    # Categorical features (if any)
    if categorical_features:
        from sklearn.preprocessing import OneHotEncoder
        # Categorical pipeline: Impute NaN → Encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    return ColumnTransformer(transformers=transformers)


# ============================================
# LOGISTIC REGRESSION MODELS
# ============================================

def make_lr(X, use_smote=False, class_weight='balanced', **kwargs):
    """
    Create Logistic Regression pipeline
    
    Args:
        X: Training features (for preprocessing setup)
        use_smote: Whether to use SMOTE for class imbalance
        class_weight: 'balanced', None, or custom dict
        **kwargs: Additional LogisticRegression parameters
    
    Returns:
        sklearn Pipeline or imblearn Pipeline
    """
    default_params = {
        'max_iter': 1000,
        'class_weight': class_weight,
        'random_state': RANDOM_STATE,
        'solver': 'lbfgs',
        'n_jobs': -1
    }
    default_params.update(kwargs)
    
    prep = build_preprocessor(X, strategy='standard')
    clf = LogisticRegression(**default_params)
    
    if use_smote:
        pipeline = ImbPipeline([
            ('prep', prep),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('clf', clf)
        ])
    else:
        pipeline = Pipeline([
            ('prep', prep),
            ('clf', clf)
        ])
    
    return pipeline


def make_lr_tuned(X, use_smote=False):
    """
    Logistic Regression with tuned hyperparameters
    Based on best practices for imbalanced time-series
    """
    return make_lr(
        X,
        use_smote=use_smote,
        C=0.1,
        penalty='l2',
        max_iter=2000
    )


# ============================================
# RANDOM FOREST MODELS
# ============================================

def make_rf_fast(X, use_smote=False, **kwargs):
    """
    Fast Random Forest for quick experimentation
    
    Args:
        X: Training features
        use_smote: Whether to use SMOTE
        **kwargs: Additional RandomForestClassifier parameters
    """
    default_params = {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 100,
        'min_samples_leaf': 50,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': 0
    }
    default_params.update(kwargs)
    
    prep = build_preprocessor(X, strategy='standard')
    clf = RandomForestClassifier(**default_params)
    
    if use_smote:
        pipeline = ImbPipeline([
            ('prep', prep),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('clf', clf)
        ])
    else:
        pipeline = Pipeline([
            ('prep', prep),
            ('clf', clf)
        ])
    
    return pipeline


def make_rf_full(X, use_smote=False):
    """
    Full Random Forest with more estimators for better performance
    """
    return make_rf_fast(
        X,
        use_smote=use_smote,
        n_estimators=500,
        max_depth=30,
        min_samples_split=50,
        min_samples_leaf=20
    )


# ============================================
# GRADIENT BOOSTING MODELS
# ============================================

def make_gb(X, use_smote=False, **kwargs):
    """
    Gradient Boosting Classifier
    
    Args:
        X: Training features
        use_smote: Whether to use SMOTE
        **kwargs: Additional GradientBoostingClassifier parameters
    """
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 100,
        'min_samples_leaf': 50,
        'subsample': 0.8,
        'random_state': RANDOM_STATE,
        'verbose': 0
    }
    default_params.update(kwargs)
    
    prep = build_preprocessor(X, strategy='standard')
    clf = GradientBoostingClassifier(**default_params)
    
    if use_smote:
        pipeline = ImbPipeline([
            ('prep', prep),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('clf', clf)
        ])
    else:
        pipeline = Pipeline([
            ('prep', prep),
            ('clf', clf)
        ])
    
    return pipeline


# ============================================
# ENSEMBLE MODELS
# ============================================

def make_stacking(X, use_smote=False):
    """
    Stacking ensemble of multiple models
    """
    from sklearn.ensemble import StackingClassifier
    
    prep = build_preprocessor(X, strategy='standard')
    
    # Base estimators
    estimators = [
        ('lr', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE
        ))
    ]
    
    # Meta-learner
    final_estimator = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        n_jobs=-1
    )
    
    if use_smote:
        pipeline = ImbPipeline([
            ('prep', prep),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('clf', clf)
        ])
    else:
        pipeline = Pipeline([
            ('prep', prep),
            ('clf', clf)
        ])
    
    return pipeline


# ============================================
# MODEL REGISTRY
# ============================================

MODEL_REGISTRY = {
    'LR': make_lr,
    'LR_tuned': make_lr_tuned,
    'RF_fast': make_rf_fast,
    'RF_full': make_rf_full,
    'GB': make_gb,
    'Stacking': make_stacking
}


def get_model(model_name, X, use_smote=False, **kwargs):
    """
    Factory function to get model by name
    
    Args:
        model_name: Name from MODEL_REGISTRY
        X: Training features
        use_smote: Whether to use SMOTE
        **kwargs: Additional model parameters
    
    Returns:
        Model pipeline
    
    Example:
        >>> model = get_model('LR', X_train, use_smote=True)
        >>> model.fit(X_train, y_train)
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    
    builder = MODEL_REGISTRY[model_name]
    return builder(X, use_smote=use_smote, **kwargs)


# ============================================
# UTILITY FUNCTIONS
# ============================================

def list_available_models():
    """List all available model architectures"""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name):
    """Get information about a model"""
    if model_name not in MODEL_REGISTRY:
        return None
    
    builder = MODEL_REGISTRY[model_name]
    return {
        'name': model_name,
        'function': builder.__name__,
        'docstring': builder.__doc__
    }


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import make_classification
    
    print("🧪 Testing Model Definitions...")
    
    # Create synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        weights=[0.95, 0.05],  # Imbalanced
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    
    print(f"\n✅ Available models: {list_available_models()}")
    
    # Test each model
    for model_name in ['LR', 'RF_fast', 'GB']:
        print(f"\n🔄 Testing {model_name}...")
        model = get_model(model_name, X, use_smote=True)
        model.fit(X[:800], y[:800])
        score = model.score(X[800:], y[800:])
        print(f"✅ {model_name} accuracy: {score:.3f}")
    
    print("\n✅ All tests passed!")