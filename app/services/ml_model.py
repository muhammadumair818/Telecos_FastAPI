import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from typing import Dict, Any, Tuple, Optional, List
import warnings
import joblib
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Directory to save trained models
SAVED_MODELS_DIR = Path("saved_models")
SAVED_MODELS_DIR.mkdir(exist_ok=True)

# ============================================
# DYNAMIC ML MODEL FUNCTIONS
# ============================================

def detect_prediction_target(df: pd.DataFrame, schema: Dict) -> Optional[str]:
    """
    Automatically detect the best target column for prediction.
    Prioritizes: revenue, profit, sales, price, cost, count.
    """
    metrics = schema.get("metrics", [])
    
    # Priority keywords for target
    priority_keywords = ['revenue', 'profit', 'sales', 'price', 'cost', 'count', 'amount', 'total']
    
    for keyword in priority_keywords:
        for metric in metrics:
            if keyword in metric.lower():
                return metric
    
    # If no priority found, use the first numeric column
    return metrics[0] if metrics else None

def detect_features(df: pd.DataFrame, target: str, schema: Dict) -> List[str]:
    """
    Automatically detect feature columns for prediction.
    Excludes target column and ID columns.
    """
    metrics = schema.get("metrics", [])
    categories = schema.get("categories", [])
    
    # Combine numeric metrics and categorical columns
    features = metrics.copy()
    features.extend(categories)
    
    # Remove target column from features
    if target in features:
        features.remove(target)
    
    # Remove ID columns
    for id_col in schema.get("id_columns", []):
        if id_col in features:
            features.remove(id_col)
    
    return features[:10]  # Limit to top 10 features for simplicity

def prepare_ml_data(df: pd.DataFrame, target: str, features: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for ML model training.
    Handles categorical variables automatically.
    """
    X = df[features].copy()
    y = df[target].copy()
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Handle missing values
    X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
    y = y.fillna(y.mean())
    
    feature_names = X.columns.tolist()
    
    return X.values, y.values, feature_names

def train_regression_models(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """
    Train multiple regression models and return best one.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = -np.inf
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {
                'r2': float(r2),
                'rmse': float(rmse),
                'model': model
            }
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = name
        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = {'error': str(e)}
    
    return {
        'results': results,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_score': float(best_score)
    }

def train_classification_model(df: pd.DataFrame, target: str, features: List[str]) -> Dict[str, Any]:
    """
    Train classification model for categorical target.
    """
    X = df[features].copy()
    y = df[target].copy()
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    X = X.fillna(0)
    y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 10 else None)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'model': model,
        'accuracy': float(accuracy),
        'features': X.columns.tolist(),
        'target_classes': y.unique().tolist()
    }

def train_models_dynamic(df: pd.DataFrame, schema: Dict) -> Dict[str, Any]:
    """
    Main function to train ML models dynamically based on dataset.
    """
    results = {
        'regression': {},
        'classification': {},
        'target_detected': None,
        'features_used': []
    }
    
    # Detect target for regression
    target = detect_prediction_target(df, schema)
    
    if target:
        results['target_detected'] = target
        features = detect_features(df, target, schema)
        results['features_used'] = features
        
        if len(features) >= 1 and len(df) >= 10:
            X, y, feature_names = prepare_ml_data(df, target, features)
            
            if len(X) > 0 and len(np.unique(y)) > 1:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Check if regression or classification
                if y.dtype == 'object' or len(np.unique(y)) < 10:
                    # Classification
                    class_result = train_classification_model(df, target, features)
                    results['classification'] = {
                        'accuracy': class_result['accuracy'],
                        'target_classes': class_result['target_classes'],
                        'model_saved': False
                    }
                else:
                    # Regression
                    reg_results = train_regression_models(X_train, y_train, X_test, y_test)
                    results['regression'] = {
                        'best_model': reg_results['best_model_name'],
                        'best_r2': reg_results['best_score'],
                        'all_results': {k: {'r2': v.get('r2'), 'rmse': v.get('rmse')} 
                                       for k, v in reg_results['results'].items() if 'r2' in v}
                    }
                    
                    # Save best model
                    if reg_results['best_model']:
                        model_path = SAVED_MODELS_DIR / f"best_model_{target}.pkl"
                        joblib.dump(reg_results['best_model'], model_path)
                        results['regression']['model_saved'] = str(model_path)
    
    return results

def make_prediction(df: pd.DataFrame, session_data: Dict, input_values: Dict) -> Dict[str, Any]:
    """
    Make prediction using trained model.
    """
    try:
        target = session_data.get('ml_results', {}).get('target_detected')
        features = session_data.get('ml_results', {}).get('features_used', [])
        
        if not target or not features:
            return {'error': 'No trained model available'}
        
        # Build input array
        X_input = []
        for feature in features:
            if feature in input_values:
                X_input.append(input_values[feature])
            elif feature in df.columns:
                X_input.append(df[feature].mean())
            else:
                X_input.append(0)
        
        # Load model
        model_path = SAVED_MODELS_DIR / f"best_model_{target}.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            prediction = model.predict([X_input])[0]
            return {
                'target': target,
                'prediction': float(prediction),
                'features_used': features,
                'input_values': input_values
            }
        else:
            return {'error': 'Model file not found'}
    except Exception as e:
        return {'error': str(e)}

def get_available_models_info() -> Dict[str, Any]:
    """
    Get information about saved models.
    """
    models = []
    for model_file in SAVED_MODELS_DIR.glob("*.pkl"):
        models.append({
            'name': model_file.stem,
            'path': str(model_file),
            'modified': model_file.stat().st_mtime
        })
    
    return {
        'saved_models': models,
        'count': len(models)
    }