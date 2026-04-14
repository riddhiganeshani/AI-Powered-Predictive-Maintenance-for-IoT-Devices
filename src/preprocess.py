# ============================================================
# FILE: src/preprocess.py
# PURPOSE: Load the NASA CMAPSS dataset and clean it
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """
    Load NASA CMAPSS dataset.
    The file has no header - we define column names manually.
    Columns: unit_id, cycle, 3 operational settings, 21 sensors
    """
    # Column names: unit=engine id, cycle=time step
    col_names = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    sensor_names = [f'sensor_{i}' for i in range(1, 22)]  # 21 sensors
    col_names += sensor_names

    # NASA dataset is space-separated with no header
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=col_names)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def add_rul(df):
    """
    Add Remaining Useful Life (RUL) column.
    RUL = max_cycle_for_this_engine - current_cycle
    This tells us: how many cycles left before this engine fails?
    """
    # Find the last recorded cycle for each engine
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']

    # Merge and compute RUL
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns=['max_cycle'], inplace=True)

    print(f"✅ RUL column added. Max RUL: {df['RUL'].max()}, Min RUL: {df['RUL'].min()}")
    return df

def create_failure_label(df, threshold=30):
    """
    Create binary classification target:
    - label = 1  → engine will fail within 'threshold' cycles (DANGER)
    - label = 0  → engine is healthy (SAFE)
    
    threshold=30 means: if RUL <= 30 cycles, predict as 'about to fail'
    """
    df['failure_label'] = (df['RUL'] <= threshold).astype(int)
    fail_count = df['failure_label'].sum()
    total = len(df)
    print(f"✅ Labels created | Failure: {fail_count} ({fail_count/total*100:.1f}%) | Normal: {total-fail_count}")
    return df

def drop_useless_sensors(df):
    """
    Some sensors have near-zero variance (they don't change much).
    These sensors add noise without useful information.
    We drop them to improve model quality.
    """
    # Sensors known to be nearly constant in CMAPSS FD001
    cols_to_drop = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                    'sensor_16', 'sensor_18', 'sensor_19']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    print(f"✅ Dropped low-variance sensors. Remaining columns: {df.shape[1]}")
    return df

def scale_features(df, feature_cols):
    """
    Normalize sensor readings to 0-1 range using MinMaxScaler.
    This prevents one sensor (e.g. pressure in PSI) from dominating
    another (e.g. temperature in °C) just because of scale difference.
    """
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"✅ Features scaled (MinMax normalization applied)")
    return df, scaler