# ============================================================
# FILE: src/visualize.py
# PURPOSE: Generate all plots and save to outputs/
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

os.makedirs('outputs', exist_ok=True)

# Set clean, professional style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.color': '#dee2e6',
    'grid.linewidth': 0.5,
    'font.family': 'sans-serif',
    'font.size': 11
})

def plot_sensor_over_time(df, sensor_col, unit_ids=None, save=True):
    """
    Plot sensor readings over machine lifetime for selected engines.
    Shows how sensor values degrade as engine approaches failure.
    """
    if unit_ids is None:
        unit_ids = df['unit_id'].unique()[:5]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unit_ids)))
    
    for uid, color in zip(unit_ids, colors):
        engine_data = df[df['unit_id'] == uid].sort_values('cycle')
        ax.plot(engine_data['cycle'], engine_data[sensor_col],
                color=color, linewidth=1.5, label=f'Engine {uid}', alpha=0.8)
    
    ax.set_xlabel('Cycle (Time)', fontsize=12)
    ax.set_ylabel(sensor_col.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Sensor Degradation Over Time — {sensor_col}', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    
    if save:
        path = f'outputs/sensor_{sensor_col}_trend.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {path}")
    plt.show()
    plt.close()

def plot_confusion_matrix(cm, save=True):
    """
    Plot confusion matrix with color coding.
    
    TN (top-left): Correctly predicted normal — GOOD
    TP (bottom-right): Correctly predicted failure — GREAT
    FN (bottom-left): Missed a failure — VERY BAD (dangerous!)
    FP (top-right): False alarm — Annoying but safe
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Custom colors: True predictions in blue, False in red
    colors = np.array([[0.2, 0.8, 0.4, 1.0],   # TN - green
                        [0.9, 0.3, 0.3, 1.0],   # FP - red
                        [1.0, 0.5, 0.0, 1.0],   # FN - orange (worst!)
                        [0.2, 0.5, 0.9, 1.0]])  # TP - blue
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal (0)', 'Failure (1)'],
                yticklabels=['Normal (0)', 'Failure (1)'],
                linewidths=2, linecolor='white',
                annot_kws={'size': 16, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, labelpad=10)
    ax.set_title('Confusion Matrix — Failure Detection', fontsize=14, fontweight='bold')
    
    # Add annotations
    ax.text(0.5, -0.12, 'FN = Missed Failure (Most Dangerous!)',
            transform=ax.transAxes, ha='center', fontsize=10, color='darkorange')
    
    plt.tight_layout()
    if save:
        path = 'outputs/confusion_matrix.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {path}")
    plt.show()
    plt.close()

def plot_feature_importance(model, feature_cols, top_n=15, save=True):
    """
    Plot which sensors/features are most important for prediction.
    This is a KEY visualization for GitHub — shows your understanding!
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n),
                   importances[indices][::-1],
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, top_n)))
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in indices[::-1]], fontsize=10)
    ax.set_xlabel('Feature Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features — Random Forest', 
                 fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, val in zip(bars, importances[indices][::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    if save:
        path = 'outputs/feature_importance.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {path}")
    plt.show()
    plt.close()

def plot_rul_prediction(df, model, feature_cols, unit_id=1, save=True):
    """
    Plot predicted failure probability vs actual RUL for one engine.
    This is your BEST demo visualization — shows the model working!
    """
    engine_df = df[df['unit_id'] == unit_id].sort_values('cycle').copy()
    X = engine_df[feature_cols]
    _, proba = model.predict_proba(X)[:, 0], model.predict_proba(X)[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top: Actual RUL
    ax1.plot(engine_df['cycle'], engine_df['RUL'],
             color='#2196F3', linewidth=2, label='Actual RUL')
    ax1.axhline(y=30, color='red', linestyle='--', linewidth=1.5, label='Failure threshold (30 cycles)')
    ax1.fill_between(engine_df['cycle'], engine_df['RUL'], 30,
                     where=(engine_df['RUL'] <= 30), alpha=0.2, color='red', label='Danger zone')
    ax1.set_ylabel('Remaining Useful Life (cycles)', fontsize=11)
    ax1.set_title(f'Engine #{unit_id} — Degradation & Failure Probability', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_ylim(bottom=0)
    
    # Bottom: Model's failure probability
    ax2.plot(engine_df['cycle'], proba,
             color='#F44336', linewidth=2, label='Predicted Failure Probability')
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, label='Alert threshold (50%)')
    ax2.fill_between(engine_df['cycle'], proba, 0.5,
                     where=(proba >= 0.5), alpha=0.2, color='red', label='Alert zone')
    ax2.set_xlabel('Cycle (Operating Time)', fontsize=11)
    ax2.set_ylabel('Failure Probability', fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.legend()
    
    plt.tight_layout()
    if save:
        path = f'outputs/rul_prediction_engine_{unit_id}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {path}")
    plt.show()
    plt.close()

def plot_failure_distribution(df, save=True):
    """Plot class distribution — normal vs failure samples."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    counts = df['failure_label'].value_counts()
    axes[0].bar(['Normal (0)', 'Failure (1)'], counts.values,
                color=['#4CAF50', '#F44336'], edgecolor='white', linewidth=2)
    axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # RUL distribution by class
    df[df['failure_label']==0]['RUL'].hist(ax=axes[1], bins=40, alpha=0.6,
                                           color='#4CAF50', label='Normal')
    df[df['failure_label']==1]['RUL'].hist(ax=axes[1], bins=40, alpha=0.6,
                                           color='#F44336', label='Failure')
    axes[1].set_title('RUL Distribution by Class', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Remaining Useful Life')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    
    plt.suptitle('Dataset Overview', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        path = 'outputs/failure_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {path}")
    plt.show()
    plt.close()