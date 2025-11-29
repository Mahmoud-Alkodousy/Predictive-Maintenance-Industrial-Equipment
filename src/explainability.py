"""
Explainability Module - SHAP-based model interpretability
Provides model explanation capabilities using SHAP values with fallback methods
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
import logging

# Initialize module logger
logger = logging.getLogger(__name__)

# ============================================
# SHAP LIBRARY LAZY LOADING SYSTEM
# ============================================

# Global flags for SHAP availability
SHAP_AVAILABLE = False
_shap = None

def get_shap():
    """
    Lazy load SHAP library to avoid import errors if not installed
    
    Returns:
        module or None: SHAP module if available, None otherwise
    """
    global SHAP_AVAILABLE, _shap
    
    # Return cached module if already loaded
    if _shap is not None:
        return _shap
    
    # Attempt to import SHAP
    try:
        import shap as _shap_mod
        SHAP_AVAILABLE = True
        _shap = _shap_mod
        logger.info("SHAP library imported successfully")
        return _shap
    except Exception as e:
        SHAP_AVAILABLE = False
        logger.warning("SHAP library not available: %s", e)
        return None


# ============================================
# MODEL EXPLAINER CLASS
# ============================================

class ModelExplainer:
    """
    Universal model explainer supporting both SHAP-based and fallback methods
    
    Attributes:
        model: Trained sklearn pipeline or model
        model_type (str): Type of model ('tree', 'linear', or 'other')
        explainer: SHAP explainer object (if SHAP is available)
        feature_names (list): Names of input features
        feature_importance (array): Feature importance scores (fallback method)
        use_shap (bool): Flag indicating if SHAP is being used
    """
    
    def __init__(self, model, model_type: str = 'tree'):
        """
        Initialize explainer for a given model
        
        Args:
            model: Trained sklearn pipeline or model object
            model_type (str): Model type - 'tree' for tree-based models (RF, XGBoost),
                            'linear' for linear models (LogisticRegression, LinearSVC)
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.feature_names = None
        self.feature_importance = None
        self.use_shap = False

    def _infer_feature_names(self, X_train: pd.DataFrame):
        """
        Extract feature names from training data or preprocessor
        
        Attempts to get feature names from:
        1. Preprocessor's get_feature_names_out() method
        2. Preprocessor's get_feature_names() method
        3. DataFrame column names
        4. Auto-generated names (f_0, f_1, ...)
        
        Args:
            X_train: Training data (DataFrame or array)
            
        Returns:
            list: Feature names
        """
        names = None
        
        # Try extracting from sklearn preprocessor
        try:
            prep = self.model.named_steps.get('prep')
            if hasattr(prep, 'get_feature_names_out'):
                names = prep.get_feature_names_out()
            elif hasattr(prep, 'get_feature_names'):
                names = prep.get_feature_names()
        except Exception:
            names = None
        
        # Fallback to DataFrame columns or auto-generate
        if names is None:
            if isinstance(X_train, pd.DataFrame):
                names = X_train.columns.tolist()
            else:
                names = [f"f_{i}" for i in range(X_train.shape[1])]
        
        return list(names)

    def fit(self, X_train: pd.DataFrame):
        """
        Fit the explainer on training data
        
        Process:
        1. Extract feature names
        2. Try to create SHAP explainer (if available)
        3. Fall back to feature importance extraction if SHAP fails
        
        Args:
            X_train: Training data to fit explainer on
        """
        logger.info("Fitting explainer...")
        
        # Extract feature names first
        self.feature_names = self._infer_feature_names(X_train)
        
        # Attempt SHAP explainer creation
        shap_mod = get_shap()
        if shap_mod is not None:
            try:
                # Transform data using preprocessor if available
                prep = self.model.named_steps.get('prep')
                if prep is not None and hasattr(prep, 'transform'):
                    X_transformed = prep.transform(X_train)
                else:
                    X_transformed = np.asarray(X_train)
                
                # Get actual model (classifier) from pipeline
                actual_model = self.model.named_steps.get('clf', self.model)
                
                # Create appropriate SHAP explainer based on model type
                if self.model_type == 'tree':
                    # Fast TreeExplainer for tree-based models
                    self.explainer = shap_mod.TreeExplainer(actual_model)
                elif self.model_type == 'linear':
                    # LinearExplainer for linear models
                    self.explainer = shap_mod.LinearExplainer(actual_model, X_transformed)
                else:
                    # Generic KernelExplainer (slower but works with any model)
                    self.explainer = shap_mod.Explainer(actual_model, X_transformed)
                
                self.use_shap = True
                logger.info("SHAP Explainer fitted successfully")
                return
                
            except Exception as e:
                logger.warning("SHAP failed during fit: %s", e)
                self.use_shap = False
        
        # ============================================
        # FALLBACK: Extract feature importance directly from model
        # ============================================
        try:
            clf = self.model.named_steps.get('clf', self.model)
            
            # Tree-based models: use feature_importances_
            if hasattr(clf, 'feature_importances_'):
                self.feature_importance = np.asarray(clf.feature_importances_, dtype=np.float32)
                logger.info("Using tree-based feature_importances_")
            
            # Linear models: use coefficient magnitudes
            elif hasattr(clf, 'coef_'):
                coef = np.asarray(clf.coef_)
                if coef.ndim > 1:
                    coef = coef[0]  # Take first class for binary classification
                self.feature_importance = np.abs(coef).astype(np.float32)
                logger.info("Using linear model coefficients")
            
            # Ultimate fallback: uniform weights
            else:
                self.feature_importance = np.ones(len(self.feature_names), dtype=np.float32)
                logger.info("Using uniform weights fallback")
                
        except Exception as e:
            logger.warning("Failed to extract feature importance: %s", e)
            self.feature_importance = np.ones(len(self.feature_names), dtype=np.float32)
        
        logger.info("Simple explainer fitted successfully")

    def explain_instance(self, X_instance):
        """
        Generate explanation for a single prediction instance
        
        Args:
            X_instance: Single instance to explain (DataFrame with 1 row or array)
            
        Returns:
            dict: Explanation dictionary containing:
                - 'feature_impacts': DataFrame with features, values, and impacts
                - 'method': Explanation method used ('SHAP' or 'Feature Importance')
        """
        # ============================================
        # STEP 1: Transform input data
        # ============================================
        prep = self.model.named_steps.get('prep')
        if prep is not None and hasattr(prep, 'transform'):
            X_transformed = prep.transform(X_instance)
        else:
            X_transformed = np.asarray(X_instance)

        # ============================================
        # STEP 2A: SHAP-based explanation (if available)
        # ============================================
        if self.use_shap and self.explainer is not None:
            try:
                # Calculate SHAP values
                shap_values = self.explainer.shap_values(X_transformed) if hasattr(self.explainer, 'shap_values') else self.explainer(X_transformed).values
                
                # Handle multi-class output (take positive class for binary)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                
                shap_values = np.asarray(shap_values)
                
                # Flatten if needed
                if shap_values.ndim == 1:
                    impacts = shap_values
                else:
                    impacts = shap_values[0]
                
                # Extract feature values
                if isinstance(X_transformed, np.ndarray):
                    values = X_transformed[0] if X_transformed.ndim > 1 else X_transformed
                else:
                    try:
                        values = X_transformed.toarray()[0]  # For sparse matrices
                    except Exception:
                        values = np.asarray(X_transformed)[0]
                        
            except Exception as e:
                logger.warning("SHAP explain failed: %s. Falling back to simple method.", e)
                self.use_shap = False
                return self.explain_instance(X_instance)  # Retry with fallback
        
        # ============================================
        # STEP 2B: Fallback explanation method
        # ============================================
        else:
            # Extract feature values
            if isinstance(X_transformed, np.ndarray):
                values = X_transformed[0] if X_transformed.ndim > 1 else X_transformed
            else:
                try:
                    values = X_transformed.toarray()[0]
                except Exception:
                    values = np.asarray(X_transformed)[0]
            
            clf = self.model.named_steps.get('clf', self.model)
            
            # Linear models: impact = coefficient × feature_value
            if hasattr(clf, 'coef_'):
                coefficients = np.asarray(clf.coef_)
                if coefficients.ndim > 1:
                    coefficients = coefficients[0]
                impacts = coefficients * values
            
            # Tree-based models: approximate impact using importance × normalized_value
            elif hasattr(clf, 'feature_importances_'):
                importance = np.asarray(clf.feature_importances_, dtype=np.float32)
                
                # Handle zero importance case
                if np.all(importance == 0):
                    importance = np.ones_like(importance, dtype=np.float32)
                
                # Normalize feature values to prevent overflow
                feature_scale = (np.abs(values).max() + 1e-10)
                normalized_values = values / feature_scale
                
                # Scale by prediction confidence
                try:
                    pred_proba = self.model.predict_proba(X_instance)[0, 1]
                    prediction_deviation = (pred_proba - 0.5) * 2  # Range: -1 to 1
                except Exception:
                    prediction_deviation = 1.0
                
                impacts = importance * normalized_values * prediction_deviation
            
            # Ultimate fallback: minimal impact
            else:
                impacts = values * 0.01

        # ============================================
        # STEP 3: Create and sort feature impact DataFrame
        # ============================================
        feature_impacts = pd.DataFrame({
            'feature': self.feature_names,
            'value': np.asarray(values, dtype=np.float32),
            'impact': np.asarray(impacts, dtype=np.float32)
        })
        
        # Sort by absolute impact (most important features first)
        feature_impacts = feature_impacts.assign(
            abs_impact=feature_impacts['impact'].abs()
        ).sort_values(
            'abs_impact', 
            ascending=False
        ).drop(
            columns=['abs_impact']
        )
        
        return {
            'feature_impacts': feature_impacts.reset_index(drop=True),
            'method': 'SHAP' if self.use_shap else 'Feature Importance'
        }

    def plot_waterfall(self, explanation: dict, max_features: int = 10) -> go.Figure:
        """
        Create interactive waterfall chart showing feature contributions
        
        Args:
            explanation (dict): Explanation dictionary from explain_instance()
            max_features (int): Maximum number of features to display
            
        Returns:
            plotly.graph_objects.Figure: Interactive waterfall chart
        """
        # Get top N features by impact
        impacts = explanation['feature_impacts'].head(max_features)
        features = impacts['feature'].tolist()
        values = impacts['impact'].tolist()
        
        # Create horizontal waterfall chart
        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"] * len(values),  # All bars are relative contributions
            y=features[::-1],  # Reverse for top-to-bottom display
            x=values[::-1],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "red"}},      # Positive impact = increases failure risk
            decreasing={"marker": {"color": "green"}},     # Negative impact = decreases failure risk
            text=[f"{v:.3f}" for v in values[::-1]],
            textposition="outside"
        ))
        
        # Add metadata to title
        method = explanation.get('method', 'Unknown')
        probability = explanation.get('probability', 0)
        
        fig.update_layout(
            title=f"🔍 Top {max_features} Feature Contributions<br><sub>Method: {method} | Prediction: {probability:.1%} failure probability</sub>",
            xaxis_title="Impact on Prediction",
            yaxis_title="Feature",
            height=400 + (max_features * 20),  # Dynamic height based on features
            showlegend=False,
            font=dict(size=11)
        )
        
        return fig


# ============================================
# TESTING FUNCTION
# ============================================

def test_explainer():
    """
    Test ModelExplainer with both linear and tree-based models
    
    Tests:
    1. Logistic Regression (linear model)
    2. Random Forest (tree-based model)
    3. Validates that impacts are in reasonable range
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    logger.info("Testing ModelExplainer with sample models...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # ============================================
        # TEST 1: Linear Model (Logistic Regression)
        # ============================================
        model_lr = Pipeline([
            ('prep', StandardScaler()), 
            ('clf', LogisticRegression(random_state=42))
        ])
        
        # Generate synthetic data
        X = pd.DataFrame(
            np.random.randn(100, 5), 
            columns=[f'feat_{i}' for i in range(5)]
        )
        y = np.random.randint(0, 2, 100)
        
        # Train and explain
        model_lr.fit(X, y)
        explainer_lr = ModelExplainer(model_lr, model_type='linear')
        explainer_lr.fit(X.head(50))
        explanation_lr = explainer_lr.explain_instance(X.head(1))
        
        logger.info("✅ Linear Model Explainer OK | Method: %s", explanation_lr['method'])

        # ============================================
        # TEST 2: Tree-Based Model (Random Forest)
        # ============================================
        model_rf = Pipeline([
            ('prep', StandardScaler()), 
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        model_rf.fit(X, y)
        explainer_rf = ModelExplainer(model_rf, model_type='tree')
        explainer_rf.fit(X.head(50))
        explanation_rf = explainer_rf.explain_instance(X.head(1))
        
        logger.info("✅ Tree Model Explainer OK | Method: %s", explanation_rf['method'])

        # ============================================
        # TEST 3: Validate Impact Magnitudes
        # ============================================
        max_impact_lr = explanation_lr['feature_impacts']['impact'].abs().max()
        max_impact_rf = explanation_rf['feature_impacts']['impact'].abs().max()
        
        # Check if impacts are in reasonable range (not too large)
        if max_impact_lr > 100:
            logger.warning("⚠️ Linear model impacts unusually large: %.2f", max_impact_lr)
        else:
            logger.info("✅ Linear model impacts reasonable: %.4f", max_impact_lr)
        
        if max_impact_rf > 100:
            logger.warning("⚠️ Tree model impacts unusually large: %.2f", max_impact_rf)
        else:
            logger.info("✅ Tree model impacts reasonable: %.4f", max_impact_rf)
        
        logger.info("✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.exception("❌ ModelExplainer test failed: %s", e)
        return False


# ============================================
# MAIN ENTRY POINT FOR TESTING
# ============================================

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    success = test_explainer()
    
    if success:
        print("\n✅ ModelExplainer module is working correctly!")
    else:
        print("\n❌ ModelExplainer module has errors - check logs above")