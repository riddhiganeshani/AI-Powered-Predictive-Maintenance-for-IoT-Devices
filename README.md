# Data Directory

## raw/
- `train_FD001.txt` — Generated synthetic NASA CMAPSS-style turbofan engine dataset
  - 22,025 rows × 26 columns
  - 100 engines, 150–300 cycles each
  - 21 sensor readings per cycle

## processed/
- `processed_data.csv` — Cleaned dataset with engineered features
  - RUL column added
  - Binary failure_label added (threshold = 30 cycles)
  - Rolling mean/std features added (window=5)
  - Lag features added (lag=3)
  - MinMax scaled

## Dataset Column Reference

| Column | Description |
|--------|-------------|
| unit_id | Engine ID (1–100) |
| cycle | Operating cycle number |
| op_setting_1/2/3 | Operational conditions |
| sensor_1 to sensor_21 | Raw sensor readings |
| RUL | Remaining Useful Life (computed) |
| failure_label | 0 = Normal, 1 = About to fail |

## Source
Synthetic simulation based on NASA CMAPSS structure.
Original: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
