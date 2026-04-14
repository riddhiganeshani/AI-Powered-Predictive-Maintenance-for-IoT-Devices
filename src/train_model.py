# ============================================================
# FILE: src/train_model.py
# PURPOSE: Train Random Forest and Logistic Regression models
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)

def train_test_split_time(df, feature_cols, target_col='failure_label', test_ratio=0.2):
    """
    Split data by engine units (not random).
    
    WHY? For time-series, random split would leak future data into training.
    Instead: train on some engines, test on different engines.
    """
    all_units = df['unit_id'].unique()
    n_test = int(len(all_units) * test_ratio)
    
    # Last 20% of engines go to test set
    test_units = all_units[-n_test:]
    train_units = all_units[:-n_test]
    
    train_df = df[df['unit_id'].isin(train_units)]
    test_df  = df[df['unit_id'].isin(test_units)]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test  = test_df[feature_cols]
    y_test  = test_df[target_col]
    
    print(f"✅ Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"   Train engines: {len(train_units)} | Test engines: {len(test_units)}")
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Classifier.
    
    WHY Random Forest?
    - Handles noisy sensor data well
    - Gives feature importance (which sensors matter most?)
    - Robust to overfitting
    - No need for feature scaling
    """
    print("\n🌲 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,      # 100 decision trees
        max_depth=10,          # Prevent overfitting
        min_samples_split=10,  # Minimum samples to split a node
        random_state=42,       # For reproducibility
        n_jobs=-1,             # Use all CPU cores
        class_weight='balanced' # Handle class imbalance
    )
    model.fit(X_train, y_train)
    print("✅ Random Forest trained!")
    return model

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression as a baseline model."""
    print("\n📈 Training Logistic Regression (baseline)...")
    model = LogisticRegression(
        max_iter=500,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    print("✅ Logistic Regression trained!")
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model with full metrics.
    
    Key metrics for predictive maintenance:
    - Recall: Catch as many ACTUAL failures as possible (avoid missing a failure!)
    - Precision: Don't raise too many false alarms
    - F1: Balance of both
    - AUC-ROC: Overall discrimination ability
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*50}")
    print(f"📊 {model_name} Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}  ← Most important!")
    print(f"  F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\nDetailed Report:\n{classification_report(y_test, y_pred, target_names=['Normal','Failure'])}")
    
    return y_pred, y_proba, confusion_matrix(y_test, y_pred)

def save_model(model, filepath='models/random_forest_model.pkl'):
    """Save trained model to disk for reuse."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✅ Model saved to: {filepath}")

def load_model(filepath='models/random_forest_model.pkl'):
    """Load a previously saved model."""
    model = joblib.load(filepath)
    print(f"✅ Model loaded from: {filepath}")
    return model