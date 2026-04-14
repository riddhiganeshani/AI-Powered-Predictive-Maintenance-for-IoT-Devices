# ============================================================
# FILE: download_data.py
# PURPOSE: Download the NASA CMAPSS dataset automatically
# ============================================================

import urllib.request
import os
import zipfile

def download_cmapss():
    """
    Downloads NASA CMAPSS dataset from a mirror.
    Original: https://data.nasa.gov/dataset/CMAPSS/ff5v-kuh6
    """
    os.makedirs('data/raw', exist_ok=True)
    
    # Alternative: Use this direct link or download manually
    print("📥 NASA CMAPSS Dataset Download Instructions:")
    print("="*50)
    print("1. Go to: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6")
    print("2. Click 'Export' → Download as CSV")
    print("   OR")
    print("3. Direct file: search for 'CMAPSSData.zip' on Google")
    print("   Extract train_FD001.txt to data/raw/")
    print("\nAlternatively, use this synthetic generator:")
    generate_synthetic_data()

def generate_synthetic_data():
    """
    Generate synthetic data that mimics NASA CMAPSS structure.
    This is your VIRTUAL SIMULATION — realistic IoT sensor data!
    
    This is PERFECT if you can't download the NASA dataset.
    """
    import numpy as np
    import pandas as pd
    
    print("\n🔄 Generating synthetic CMAPSS-like data...")
    np.random.seed(42)
    
    n_engines = 100
    rows = []
    
    for engine_id in range(1, n_engines + 1):
        # Each engine runs for 150-300 cycles before failing
        max_cycle = np.random.randint(150, 300)
        
        for cycle in range(1, max_cycle + 1):
            # Degradation factor: increases as engine approaches end of life
            deg = cycle / max_cycle  # 0 at start, 1 at failure
            
            row = [engine_id, cycle,
                   # Operational settings (vary randomly)
                   np.random.choice([0, 20, 25, 42, 100]),  # op_setting_1
                   np.random.choice([0, 0.7, 0.84]),         # op_setting_2
                   100,                                        # op_setting_3
                   
                   # ── SENSORS (21 total, realistic degradation) ──
                   # Sensor 1: nearly constant (useless)
                   518.67 + np.random.normal(0, 0.5),
                   # Sensor 2: increases with degradation (compressor outlet temp)
                   642.0 + 15 * deg + np.random.normal(0, 1),
                   # Sensor 3: increases (total temperature at HPC outlet)
                   1590.0 + 30 * deg + np.random.normal(0, 3),
                   # Sensor 4: increases (total temperature at LPT outlet)
                   1400.0 + 20 * deg + np.random.normal(0, 5),
                   # Sensor 5: constant
                   14.62 + np.random.normal(0, 0.02),
                   # Sensor 6: constant (useless)
                   21.61 + np.random.normal(0, 0.1),
                   # Sensor 7: decreases (fan speed)
                   554.0 - 8 * deg + np.random.normal(0, 1),
                   # Sensor 8: pressure ratio (complex pattern)
                   2388.0 + 10 * np.sin(cycle * 0.1) + np.random.normal(0, 2),
                   # Sensor 9: decreases (physical core speed)
                   9044.0 - 50 * deg + np.random.normal(0, 10),
                   # Sensor 10: constant (useless)
                   1.3 + np.random.normal(0, 0.001),
                   # Sensor 11: increases (burner fuel-air ratio)
                   47.0 + 2 * deg + np.random.normal(0, 0.3),
                   # Sensor 12: decreases (corrected fan speed)
                   521.0 - 5 * deg + np.random.normal(0, 1),
                   # Sensor 13: increases (corrected core speed)
                   2388.0 + 8 * deg + np.random.normal(0, 2),
                   # Sensor 14: increases (ratio of fuel flow)
                   8138.0 + 25 * deg + np.random.normal(0, 5),
                   # Sensor 15: decreases (bypass ratio)
                   8.42 - 0.05 * deg + np.random.normal(0, 0.01),
                   # Sensor 16: constant (useless)
                   0.03 + np.random.normal(0, 0.0001),
                   # Sensor 17: increases (bleed enthalpy)
                   391.0 + 3 * deg + np.random.normal(0, 0.5),
                   # Sensor 18: constant
                   2388.0 + np.random.normal(0, 0.5),
                   # Sensor 19: constant
                   100.0 + np.random.normal(0, 0.01),
                   # Sensor 20: varies
                   39.0 + 1.5 * deg + np.random.normal(0, 0.2),
                   # Sensor 21: varies
                   23.4 + deg + np.random.normal(0, 0.1),
                  ]
            rows.append(row)
    
    col_names = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    col_names += [f'sensor_{i}' for i in range(1, 22)]
    
    df = pd.DataFrame(rows, columns=col_names)
    
    # Save as space-separated (same format as NASA original)
    df.to_csv('data/raw/train_FD001.txt', sep=' ', index=False, header=False)
    print(f"✅ Synthetic data generated: {len(df)} rows, {len(df.columns)} columns")
    print(f"   Saved to: data/raw/train_FD001.txt")
    print(f"   Engines: {n_engines} | Avg cycles: {df.groupby('unit_id')['cycle'].max().mean():.0f}")
    return df

if __name__ == '__main__':
    download_cmapss()