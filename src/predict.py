# ============================================================
# FILE: src/predict.py
# PURPOSE: Run predictions and generate failure alerts
# ============================================================

import pandas as pd
import numpy as np

def predict_failure(model, X, threshold=0.5):
    """
    Predict failure probability for each data point.
    
    Returns:
    - predictions: 0 (normal) or 1 (failure)
    - probabilities: 0.0 to 1.0 (confidence of failure)
    """
    proba = model.predict_proba(X)[:, 1]  # Probability of class=1 (failure)
    predictions = (proba >= threshold).astype(int)
    return predictions, proba

def generate_alert(unit_id, cycle, failure_prob, rul):
    """
    Generate human-readable alert based on failure probability.
    
    Alert levels:
    - CRITICAL (≥80%): Immediate shutdown needed
    - WARNING  (≥50%): Schedule maintenance soon  
    - NORMAL   (<50%): Machine is healthy
    """
    if failure_prob >= 0.80:
        level = "🔴 CRITICAL"
        action = "IMMEDIATE MAINTENANCE REQUIRED — Risk of failure!"
    elif failure_prob >= 0.50:
        level = "🟡 WARNING"
        action = f"Schedule maintenance within {max(1, int(rul))} cycles"
    else:
        level = "🟢 NORMAL"
        action = f"Machine healthy — estimated {int(rul)} cycles remaining"
    
    return {
        'unit_id': unit_id,
        'cycle': cycle,
        'failure_probability': round(failure_prob, 3),
        'estimated_RUL': int(rul),
        'alert_level': level,
        'recommended_action': action
    }

def run_alert_system(df, model, feature_cols, sample_engines=5):
    """
    Simulate real-time alert system for a set of engines.
    Shows the latest reading for each engine and its alert status.
    """
    print("\n" + "="*65)
    print("   🏭 PREDICTIVE MAINTENANCE ALERT SYSTEM — LIVE STATUS")
    print("="*65)
    
    alerts = []
    # Get last known state for each engine (most recent cycle)
    latest = df.sort_values('cycle').groupby('unit_id').last().reset_index()
    latest = latest.head(sample_engines)
    
    for _, row in latest.iterrows():
        X_single = row[feature_cols].values.reshape(1, -1)
        _, proba = predict_failure(model, X_single)
        alert = generate_alert(
            unit_id=int(row['unit_id']),
            cycle=int(row['cycle']),
            failure_prob=proba[0],
            rul=row['RUL']
        )
        alerts.append(alert)
        print(f"\n  Engine #{alert['unit_id']:>3} | Cycle {alert['cycle']:>4} | "
              f"Fail Prob: {alert['failure_probability']*100:>5.1f}% | "
              f"RUL: {alert['estimated_RUL']:>4} cycles")
        print(f"  {alert['alert_level']} → {alert['recommended_action']}")
    
    print("\n" + "="*65)
    return pd.DataFrame(alerts)