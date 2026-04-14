# ============================================================
# FILE: src/features.py
# PURPOSE: Create engineered features from raw sensor data
# ============================================================

import pandas as pd
import numpy as np

def add_rolling_features(df, sensor_cols, window=5):
    """
    Add rolling (moving) statistics for each sensor.
    
    WHY? A single sensor reading might be noisy.
    Rolling mean/std captures the TREND over recent cycles.
    
    Example: If sensor_2 is usually 500, but the last 5 readings 
    averaged 650, that's a red flag — the trend is worsening.
    """
    for sensor in sensor_cols:
        # Rolling mean: average of last 'window' readings per engine
        df[f'{sensor}_rolling_mean'] = (
            df.groupby('unit_id')[sensor]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        # Rolling std: how much variation in last 'window' readings
        df[f'{sensor}_rolling_std'] = (
            df.groupby('unit_id')[sensor]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
        )
    
    print(f"✅ Rolling features added (window={window}) for {len(sensor_cols)} sensors")
    return df

def add_lag_features(df, sensor_cols, lag=3):
    """
    Add lagged sensor values.
    
    WHY? The reading from 3 cycles ago can help predict current state.
    If sensor_4 was 400 three cycles ago and is now 750, 
    that rapid increase may signal failure.
    """
    for sensor in sensor_cols:
        df[f'{sensor}_lag_{lag}'] = (
            df.groupby('unit_id')[sensor]
            .transform(lambda x: x.shift(lag).bfill())
        )
    
    print(f"✅ Lag features added (lag={lag}) for {len(sensor_cols)} sensors")
    return df

def add_cycle_features(df):
    """
    Add time-based features from the cycle number.
    
    WHY? Machines degrade over time. A machine at cycle 200 
    is more likely to fail than one at cycle 10.
    """
    # Normalized cycle position per engine
    max_cycles = df.groupby('unit_id')['cycle'].transform('max')
    df['cycle_normalized'] = df['cycle'] / max_cycles
    
    print("✅ Cycle features added")
    return df

def get_feature_columns(df):
    """
    Return list of all feature columns (exclude metadata and target).
    These are the columns we'll feed into our ML model.
    """
    exclude_cols = ['unit_id', 'cycle', 'RUL', 'failure_label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"✅ Total features for model: {len(feature_cols)}")
    return feature_cols