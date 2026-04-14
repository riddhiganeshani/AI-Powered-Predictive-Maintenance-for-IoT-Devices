# ============================================================
# FILE: main.py
# PURPOSE: Run the complete predictive maintenance pipeline
# Run with: python main.py
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import load_data, add_rul, create_failure_label, drop_useless_sensors, scale_features
from features import add_rolling_features, add_lag_features, add_cycle_features, get_feature_columns
from train_model import train_test_split_time, train_random_forest, train_logistic_regression, evaluate_model, save_model
from predict import predict_failure, run_alert_system
from visualize import (plot_sensor_over_time, plot_confusion_matrix, 
                       plot_feature_importance, plot_rul_prediction,
                       plot_failure_distribution)

def main():
    print("\n" + "="*65)
    print("   🤖 AI PREDICTIVE MAINTENANCE SYSTEM — STARTING PIPELINE")
    print("="*65)
    
    # ──────────────────────────────────────────────
    # PHASE 1: LOAD DATA
    # ──────────────────────────────────────────────
    print("\n📂 PHASE 1: Loading Dataset...")
    DATA_PATH = 'data/raw/train_FD001.txt'
    
    # Check if dataset exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at {DATA_PATH}")
        print("📥 Please download NASA CMAPSS dataset:")
        print("   https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6")
        print("   OR run: python download_data.py")
        return
    
    df = load_data(DATA_PATH)
    
    # ──────────────────────────────────────────────
    # PHASE 2: PREPROCESSING
    # ──────────────────────────────────────────────
    print("\n🔧 PHASE 2: Preprocessing...")
    df = add_rul(df)
    df = create_failure_label(df, threshold=30)
    df = drop_useless_sensors(df)
    
    # Get sensor columns for feature engineering
    sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
    
    # ──────────────────────────────────────────────
    # PHASE 3: FEATURE ENGINEERING
    # ──────────────────────────────────────────────
    print("\n⚙️  PHASE 3: Feature Engineering...")
    df = add_rolling_features(df, sensor_cols, window=5)
    df = add_lag_features(df, sensor_cols, lag=3)
    df = add_cycle_features(df)
    
    # Get all feature columns (for model input)
    feature_cols = get_feature_columns(df)
    
    # Scale features
    df, scaler = scale_features(df, feature_cols)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/processed_data.csv', index=False)
    print("✅ Processed data saved to data/processed/processed_data.csv")
    
    # ──────────────────────────────────────────────
    # PHASE 4: VISUALIZATION (EDA)
    # ──────────────────────────────────────────────
    print("\n📊 PHASE 4: Generating EDA Visualizations...")
    plot_failure_distribution(df)
    plot_sensor_over_time(df, 'sensor_2', unit_ids=[1, 2, 3, 4, 5])
    
    # ──────────────────────────────────────────────
    # PHASE 5: MODEL TRAINING
    # ──────────────────────────────────────────────
    print("\n🧠 PHASE 5: Training Models...")
    X_train, X_test, y_train, y_test = train_test_split_time(df, feature_cols)
    
    # Train both models
    rf_model = train_random_forest(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)
    
    # ──────────────────────────────────────────────
    # PHASE 6: EVALUATION
    # ──────────────────────────────────────────────
    print("\n📈 PHASE 6: Evaluating Models...")
    y_pred_rf, y_proba_rf, cm_rf = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    y_pred_lr, y_proba_lr, cm_lr = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # Plot confusion matrix for best model (Random Forest)
    plot_confusion_matrix(cm_rf)
    plot_feature_importance(rf_model, feature_cols)
    
    # ──────────────────────────────────────────────
    # PHASE 7: FAILURE PREDICTION DEMO
    # ──────────────────────────────────────────────
    print("\n🔮 PHASE 7: Running Failure Predictions...")
    # Visualize one engine's degradation + prediction
    plot_rul_prediction(df, rf_model, feature_cols, unit_id=1)
    plot_rul_prediction(df, rf_model, feature_cols, unit_id=5)
    
    # ──────────────────────────────────────────────
    # PHASE 8: ALERT SYSTEM DEMO
    # ──────────────────────────────────────────────
    print("\n🚨 PHASE 8: Alert System Demo...")
    alerts_df = run_alert_system(df, rf_model, feature_cols, sample_engines=10)
    alerts_df.to_csv('outputs/alert_report.csv', index=False)
    print("\n✅ Alert report saved to outputs/alert_report.csv")
    
    # ──────────────────────────────────────────────
    # PHASE 9: SAVE MODEL
    # ──────────────────────────────────────────────
    print("\n💾 PHASE 9: Saving Model...")
    save_model(rf_model)
    
    print("\n" + "="*65)
    print("   ✅ PIPELINE COMPLETE! Check the outputs/ folder for results.")
    print("="*65)
    print("\n📁 Generated files:")
    for f in os.listdir('outputs'):
        print(f"   ├── outputs/{f}")

if __name__ == '__main__':
    main()