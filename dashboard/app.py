# ============================================================
# FILE: dashboard/app.py  (FIXED VERSION)
# PURPOSE: Flask web server — serves the live dashboard
# HOW TO RUN: python dashboard/app.py
# THEN OPEN:  http://127.0.0.1:5000  in your browser
# ============================================================

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)

# ── Paths ──
BASE          = os.path.dirname(__file__)
MODEL_PATH    = os.path.join(BASE, '..', 'models', 'random_forest_model.pkl')
FEATURES_PATH = os.path.join(BASE, '..', 'models', 'feature_columns.pkl')
DATA_PATH     = os.path.join(BASE, '..', 'data', 'processed', 'processed_data.csv')

model         = None
feature_cols  = None
df            = None
df_snapshot   = None   # mid-life snapshot — gives realistic mix of statuses


def load_assets():
    global model, feature_cols, df, df_snapshot
    try:
        model        = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURES_PATH)
        df           = pd.read_csv(DATA_PATH)
        print(f"✅ Model loaded   | Features: {len(feature_cols)}")
        print(f"✅ Data loaded    | Rows: {len(df):,} | Engines: {df['unit_id'].nunique()}")

        # ── KEY FIX: Build a realistic "current reading" snapshot ──
        # The bug before: using the LAST cycle of each engine (RUL=0)
        # means every engine looks like it's about to fail RIGHT NOW.
        # Fix: pick each engine at a random point between 40%-90% of life.
        # This gives us a natural mix of Normal, Warning, and Critical.
        np.random.seed(42)
        rows = []
        for uid, grp in df.groupby('unit_id'):
            grp = grp.sort_values('cycle').reset_index(drop=True)
            n   = len(grp)
            # Random position between 40% and 90% through engine lifetime
            idx = int(np.random.uniform(0.40, 0.90) * n)
            idx = min(max(idx, 0), n - 1)
            rows.append(grp.iloc[idx])

        df_snapshot = pd.DataFrame(rows).reset_index(drop=True)

        # Log the distribution for confirmation
        proba = model.predict_proba(df_snapshot[feature_cols])[:, 1]
        print(f"✅ Snapshot built | Critical:{int((proba>=0.80).sum())}  "
              f"Warning:{int(((proba>=0.50)&(proba<0.80)).sum())}  "
              f"Normal:{int((proba<0.50).sum())}")

    except FileNotFoundError as e:
        print(f"\n❌  File not found: {e}")
        print("    Please run  python main.py  first!\n")
    except Exception as e:
        print(f"\n❌  Unexpected error: {e}\n")
        import traceback; traceback.print_exc()


# ────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/summary')
def api_summary():
    if df_snapshot is None or model is None:
        return jsonify({'error': 'Run main.py first to generate the model.'}), 500

    proba      = model.predict_proba(df_snapshot[feature_cols])[:, 1]
    n_engines  = int(df_snapshot['unit_id'].nunique())
    n_critical = int((proba >= 0.80).sum())
    n_warning  = int(((proba >= 0.50) & (proba < 0.80)).sum())
    avg_prob   = round(float(proba.mean()) * 100, 1)

    # Overall accuracy on the full labelled dataset
    y_true   = df['failure_label']
    y_pred   = model.predict(df[feature_cols])
    accuracy = round(float((y_true == y_pred).mean()) * 100, 1)

    return jsonify({
        'n_engines' : n_engines,
        'n_critical': n_critical,
        'n_warning' : n_warning,
        'avg_prob'  : avg_prob,
        'accuracy'  : accuracy,
    })


@app.route('/api/fleet')
def api_fleet():
    if df_snapshot is None or model is None:
        return jsonify([])

    proba = model.predict_proba(df_snapshot[feature_cols])[:, 1]
    rows  = []
    for i, row in df_snapshot.iterrows():
        p   = float(proba[i])
        pct = round(p * 100, 1)
        rul = int(row['RUL'])
        if pct >= 80:
            status, label = 'critical', 'Critical'
        elif pct >= 50:
            status, label = 'warning',  'Warning'
        else:
            status, label = 'normal',   'Normal'
        rows.append({
            'unit_id'  : int(row['unit_id']),
            'cycle'    : int(row['cycle']),
            'fail_prob': pct,
            'rul'      : rul,
            'status'   : status,
            'label'    : label,
        })

    rows.sort(key=lambda x: x['fail_prob'], reverse=True)
    return jsonify(rows)


@app.route('/api/rul/<int:engine_id>')
def api_rul(engine_id):
    if df is None or model is None:
        return jsonify({'error': 'Assets not loaded'}), 500

    eng = df[df['unit_id'] == engine_id].sort_values('cycle').copy()
    if eng.empty:
        return jsonify({'error': f'Engine {engine_id} not found'}), 404

    proba = model.predict_proba(eng[feature_cols])[:, 1]
    return jsonify({
        'engine_id': engine_id,
        'cycles'   : eng['cycle'].tolist(),
        'ruls'     : eng['RUL'].tolist(),
        'probs'    : [round(float(p), 4) for p in proba],
        'max_cycle': int(eng['cycle'].max()),
    })


@app.route('/api/sensors/<int:engine_id>/<int:cycle>')
def api_sensors(engine_id, cycle):
    if df is None or model is None:
        return jsonify({'error': 'Assets not loaded'}), 500

    eng = df[df['unit_id'] == engine_id].sort_values('cycle')
    if eng.empty:
        return jsonify({'error': 'Engine not found'}), 404

    # Find closest available cycle
    idx          = (eng['cycle'] - cycle).abs().idxmin()
    row          = eng.loc[idx]
    actual_cycle = int(row['cycle'])

    # Raw sensor columns only (no rolling/lag suffixes)
    sensor_raw = [c for c in feature_cols
                  if c.startswith('sensor_')
                  and '_rolling' not in c
                  and '_lag'     not in c][:8]

    sensors = [{'name': s.replace('_', ' ').title(),
                'value': round(float(row[s]), 4)}
               for s in sensor_raw if s in row.index]

    X_row     = eng.loc[[idx]][feature_cols]
    fail_prob = round(float(model.predict_proba(X_row)[0, 1]) * 100, 1)

    return jsonify({
        'engine_id': engine_id,
        'cycle'    : actual_cycle,
        'rul'      : int(row['RUL']),
        'fail_prob': fail_prob,
        'sensors'  : sensors,
    })


@app.route('/api/model_metrics')
def api_model_metrics():
    if df is None or model is None:
        return jsonify({'error': 'Assets not loaded'}), 500

    from sklearn.metrics import (confusion_matrix, accuracy_score,
                                  precision_score, recall_score, f1_score)

    all_units  = df['unit_id'].unique()
    test_units = all_units[int(len(all_units) * 0.8):]
    test_df    = df[df['unit_id'].isin(test_units)]

    X_test = test_df[feature_cols]
    y_test = test_df['failure_label']
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred).tolist()

    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:10]
    fi = [{'feature': feature_cols[i].replace('_', ' '),
           'score'  : round(float(importances[i]), 4)}
          for i in top_idx]

    return jsonify({
        'confusion_matrix': cm,
        'accuracy' : round(accuracy_score(y_test, y_pred) * 100, 1),
        'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
        'recall'   : round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
        'f1'       : round(f1_score(y_test, y_pred, zero_division=0) * 100, 1),
        'feature_importance': fi,
    })


@app.route('/api/class_dist')
def api_class_dist():
    if df is None:
        return jsonify({'normal': 0, 'failure': 0})
    counts = df['failure_label'].value_counts().to_dict()
    return jsonify({
        'normal' : int(counts.get(0, 0)),
        'failure': int(counts.get(1, 0)),
    })


# ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_assets()
    print("\n" + "="*55)
    print("  Dashboard →  http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)