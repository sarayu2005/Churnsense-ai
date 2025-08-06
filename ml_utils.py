# ml_utils.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import xgboost as xgb
import shap
import lime
import lime.lime_tabular
import os
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

def save_plot(fig, filename):
    """Save plot and return path"""
    path = f"plots/{filename}"
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return path

def preprocess_data(df, churn_col, test_size=0.2, random_state=42):
    """
    Preprocess data for ML models
    """
    # Separate target and features
    if churn_col not in df.columns:
        raise ValueError(f"Churn column '{churn_col}' not found in dataset")
    
    y = df[churn_col].copy()
    X = df.drop(columns=[churn_col]).copy()

    # --- FIX 1: Drop identifier columns to prevent data leakage ---
    for col in X.columns:
        if 'id' in col.lower():
            X.drop(columns=[col], inplace=True)
            print(f"Dropped identifier column: {col}")
            
    # --- FIX 2: Drop other leaky features ---
    leaky_cols = ['last_login_days']
    for col in leaky_cols:
        if col in X.columns:
            X.drop(columns=[col], inplace=True)
            print(f"Dropped leaky feature: {col}")
    
    # Handle missing values
    # Numerical columns - fill with median
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        X[col].fillna(X[col].median(), inplace=True)
    
    # Categorical columns - fill with mode
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Use single brackets to ensure a 1D array is passed
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    # Create a new list of numerical columns after potential dropping of ID
    numerical_cols_to_scale = X.select_dtypes(include=[np.number]).columns
    X_train[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
    X_test[numerical_cols_to_scale] = scaler.transform(X_test[numerical_cols_to_scale])
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns)
    }

def train_models(data_dict):
    """
    Train multiple models and return performance metrics
    """
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    
    models = {}
    results = {}
    
    # Define models
    model_definitions = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    # Evaluate each model
    for name, model in model_definitions.items():
        model.fit(X_train, y_train)
        models[name] = model
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist(),
            'y_test': y_test.tolist()
        }
    
    # Save models
    for name, model in models.items():
        model_filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
    
    return models, results

def plot_model_comparison(results):
    """
    Create comparison plots for all models
    """
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        ax.bar(x + i * width, values, width, label=metric.upper())
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * (len(metrics)-1)/2)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    return save_plot(fig, "model_comparison.png")

def plot_roc_curves(results):
    """
    Plot ROC curves for all models
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, result in results.items():
        y_test = result['y_test']
        y_pred_proba = result['y_pred_proba']
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = result['roc_auc']
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return save_plot(fig, "roc_curves.png")

def get_best_model(results):
    """
    Determine the best model based on ROC-AUC score
    """
    if not results:
        return None, None
    best_model = max(results.items(), key=lambda item: item[1]['roc_auc'])
    return best_model[0], best_model[1]

def run_complete_ml_pipeline(df, churn_col):
    """
    Run the complete ML pipeline
    """
    try:
        data_dict = preprocess_data(df, churn_col)
        models, results = train_models(data_dict)
        
        comparison_plot = plot_model_comparison(results)
        roc_plot = plot_roc_curves(results)
        
        best_model_name, best_model_metrics = get_best_model(results)
        
        return {
            'results': results,
            'best_model': {
                'name': best_model_name,
                'metrics': best_model_metrics
            },
            'plots': {
                'comparison': comparison_plot,
                'roc_curves': roc_plot,
            },
            'data_info': {
                'n_features': len(data_dict['feature_names']),
                'feature_names': data_dict['feature_names'],
                'train_size': len(data_dict['X_train']),
                'test_size': len(data_dict['X_test'])
            }
        }
    
    except Exception as e:
        # Provide a more specific error message
        raise Exception(f"ML Pipeline failed: {str(e)}")
