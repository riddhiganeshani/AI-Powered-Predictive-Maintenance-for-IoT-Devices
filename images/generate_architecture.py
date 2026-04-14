# ============================================================
# Run this once: python images/generate_architecture.py
# Generates the architecture diagram for README.md
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

os.makedirs('images', exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

def draw_box(ax, x, y, w, h, text, subtext='', color='#3b82f6', textcolor='white'):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(box)
    if subtext:
        ax.text(x, y + 0.15, text, ha='center', va='center',
                fontsize=10, fontweight='bold', color=textcolor)
        ax.text(x, y - 0.2, subtext, ha='center', va='center',
                fontsize=8, color=textcolor, alpha=0.85)
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=10, fontweight='bold', color=textcolor)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#6b7280', lw=2))

# ── Title ──
ax.text(8, 9.5, 'AI-Powered Predictive Maintenance System — Architecture',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#1e293b')

# ── Layer 1: Sensors ──
sensor_info = [
    (1.6, 'Temperature\nSensor', '°C'),
    (4.0, 'Vibration\nSensor', 'g-force'),
    (6.4, 'Pressure\nSensor', 'PSI'),
    (8.8, 'RPM\nSensor', 'rot/min'),
    (11.2, 'Current\nSensor', 'Amps'),
]
for x, name, unit in sensor_info:
    draw_box(ax, x, 8.2, 2.0, 1.0, name, unit, color='#0ea5e9')

ax.text(13.5, 8.2, 'Layer 1:\nSensor Input', ha='left', va='center',
        fontsize=9, color='#64748b')

# ── Arrow down ──
draw_arrow(ax, 6.4, 7.65, 6.4, 7.05)

# ── Layer 2: Preprocessing ──
proc_info = [(2.5, 'Data Cleaning', 'nulls, outliers'),
             (6.4, 'Feature Engineering', 'rolling, lag, RUL'),
             (10.3, 'Train/Test Split', '80/20 engine-based')]
for x, name, sub in proc_info:
    draw_box(ax, x, 6.5, 3.4, 0.95, name, sub, color='#8b5cf6')

ax.text(13.5, 6.5, 'Layer 2:\nPreprocessing', ha='left', va='center',
        fontsize=9, color='#64748b')

draw_arrow(ax, 6.4, 6.0, 6.4, 5.4)

# ── Layer 3: Models ──
model_info = [(2.5, 'Random Forest', '100 estimators'), 
              (6.4, 'Logistic Regression', 'Baseline model'),
              (10.3, 'Cross-Validation', '5-fold CV')]
for x, name, sub in model_info:
    draw_box(ax, x, 4.85, 3.4, 0.95, name, sub, color='#f59e0b', textcolor='#1e293b')

ax.text(13.5, 4.85, 'Layer 3:\nML Model', ha='left', va='center',
        fontsize=9, color='#64748b')

draw_arrow(ax, 6.4, 4.35, 6.4, 3.75)

# ── Layer 4: Output ──
out_info = [(2.0, 'Failure\nPrediction', '0 or 1'),
            (4.8, 'RUL Curve', 'time to fail'),
            (7.6, 'Confusion\nMatrix', 'TP/FP/FN/TN'),
            (10.4, 'Alert\nSystem', 'CRITICAL/WARN')]
for x, name, sub in out_info:
    draw_box(ax, x, 3.2, 2.4, 0.95, name, sub, color='#10b981')

ax.text(13.5, 3.2, 'Layer 4:\nOutput', ha='left', va='center',
        fontsize=9, color='#64748b')

draw_arrow(ax, 6.4, 2.72, 6.4, 2.15)

# ── Layer 5: GitHub ──
draw_box(ax, 6.4, 1.7, 8.0, 0.85,
         'GitHub Repository — Proof of Work',
         'notebooks/ | src/ | outputs/ | README | models/',
         color='#1e293b')

ax.text(13.5, 1.7, 'Layer 5:\nDeployment', ha='left', va='center',
        fontsize=9, color='#64748b')

plt.tight_layout()
plt.savefig('images/architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print("✅ Architecture image saved to images/architecture.png")
plt.show()