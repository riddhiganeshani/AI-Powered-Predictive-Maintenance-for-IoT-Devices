# Model Card — Random Forest Failure Classifier

## Model Details
- **Type:** Random Forest Classifier
- **Library:** scikit-learn 1.3.2
- **File:** `random_forest_model.pkl`
- **Trained:** NASA CMAPSS synthetic turbofan engine data

## Hyperparameters
| Parameter | Value | Reason |
|-----------|-------|--------|
| n_estimators | 100 | Balance speed vs accuracy |
| max_depth | 10 | Prevent overfitting |
| min_samples_split | 10 | Regularization |
| class_weight | balanced | Handle class imbalance |
| random_state | 42 | Reproducibility |

## Performance
| Metric | Score |
|--------|-------|
| Accuracy | ~93% |
| Precision | ~89% |
| Recall | ~91% |
| F1 Score | ~90% |
| AUC-ROC | ~97% |

## Input Features
- 14 sensor readings (after dropping low-variance sensors)
- Rolling mean and std (window=5) for each sensor
- Lag features (lag=3) for each sensor
- Normalized cycle position
- **Total: 60 features**

## Output
- `0` = Normal operation (no failure expected within 30 cycles)
- `1` = Failure imminent (failure expected within 30 cycles)
- `predict_proba()` returns probability 0.0–1.0

## Limitations
- Trained on synthetic data; real deployment needs real sensor calibration
- Threshold of 30 cycles is tunable based on maintenance lead time
- Does not distinguish between failure modes (only binary: fail/no fail)

## How to Load and Use
```python
import joblib
model = joblib.load('models/random_forest_model.pkl')
feature_cols = joblib.load('models/feature_columns.pkl')
# prediction = model.predict(X[feature_cols])
# probability = model.predict_proba(X[feature_cols])[:, 1]
```