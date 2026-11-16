#!/usr/bin/env python3
"""
Generates high-resolution (300 DPI) Milestone-2 charts from GazeConnect benchmark metrics.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.8

# -------------------------------------------------------------------------
# Figure 1: Model Ablation (Error in px vs Throughput in FPS)
# -------------------------------------------------------------------------
models = [
    "M1: Pure Geom\n(Fixed Ridge)",
    "M2: Pure Geom\n(Grid Ridge)",
    "M3: CNN-Only\n(L2CS 7x3)",
    "M4: Hybrid\n(BasicCNN+PSO)",
    "M5: Hybrid L2CS\n+ PSO [Proposed]",
]
median_err = [97.1, 94.9, 631.6, 86.1, 26.78]
fps_vals = [94.1, 94.1, 46.1, 31.5, 46.1]
colors = ["#64748B", "#475569", "#EF4444", "#3B82F6", "#10B981"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

bars1 = ax1.bar(models, median_err, color=colors, width=0.55)
ax1.set_yscale("log")
ax1.set_ylabel("Median Euclidean Gaze Error (px, log scale)", fontsize=10, fontweight="bold")
ax1.set_title("(a) Localization Error Across Pipeline Variants (Lower is Better)", fontsize=11, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.set_ylim(15, 1100)
for bar, val in zip(bars1, median_err):
    ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.12, f"{val:.1f} px",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.tick_params(axis="x", labelsize=8.5)

bars2 = ax2.bar(models, fps_vals, color=colors, width=0.55)
ax2.axhline(30.0, color="#F59E0B", linestyle="--", linewidth=1.3, label="Real-Time Threshold (30 FPS)")
ax2.set_ylabel("End-to-End Inference Speed (FPS on Apple MPS)", fontsize=10, fontweight="bold")
ax2.set_title("(b) Real-Time Throughput on MacBook Air MPS (Higher is Better)", fontsize=11, fontweight="bold")
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.set_ylim(0, 112)
ax2.legend(loc="upper right", fontsize=8.5)
for bar, val in zip(bars2, fps_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 2.2, f"{val:.1f} FPS",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.tick_params(axis="x", labelsize=8.5)

plt.tight_layout()
plt.savefig("fig1_model_ablation.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 2: Discrete Key Selection Classification Metrics
# -------------------------------------------------------------------------
layouts = [
    "Needs Board\n(4x2 = 8 Keys)",
    "Large Alphabet / Varnamala\n(5x3 = 15 Keys)",
    "Dense QWERTY\n(10x3 = 30 Keys)",
]
acc = [85.61, 99.65, 63.51]
prec = [98.25, 99.68, 77.78]
rec = [83.68, 99.35, 59.80]
f1 = [88.43, 99.50, 62.50]

x = np.arange(len(layouts))
w = 0.19

fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)
b1 = ax.bar(x - 1.5 * w, acc, width=w, label="Accuracy (%)", color="#2563EB")
b2 = ax.bar(x - 0.5 * w, prec, width=w, label="Macro Precision (%)", color="#10B981")
b3 = ax.bar(x + 0.5 * w, rec, width=w, label="Macro Recall (%)", color="#F59E0B")
b4 = ax.bar(x + 1.5 * w, f1, width=w, label="Macro F1-Score (%)", color="#8B5CF6")

ax.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
ax.set_title("GazeConnect Discrete Key-Selection Classification Performance (Hybrid-L2CS + PSO)",
             fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(layouts, fontsize=9.5, fontweight="bold")
ax.set_ylim(0, 113)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(ncol=4, loc="upper right", fontsize=8.5)

for bars in (b1, b2, b3, b4):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", rotation=90)

plt.tight_layout()
plt.savefig("fig2_layout_classification.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 3: PSO-Discovered Modality Weights & Hyperparameters
# -------------------------------------------------------------------------
params = ["5-Point Iris\nGeometry ($w_{geom}$)", "Nose-to-Eye\nHead Pose ($w_{head}$)",
          "L2CS ResNet-18\nAppearance ($w_{cnn}$)", "Ridge Penalty\n($\\lambda$)"]
vals = [1.541, 0.800, 0.622, 1.000]
p_colors = ["#10B981", "#3B82F6", "#8B5CF6", "#F59E0B"]

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
bars = ax.barh(params[::-1], vals[::-1], color=p_colors[::-1], height=0.5)
ax.set_xlabel("PSO-Optimized Parameter Value (LOTO Cross-Validated)", fontsize=10, fontweight="bold")
ax.set_title("Optimal Hyperparameters Discovered by Particle Swarm Optimization (PSO)",
             fontsize=11, fontweight="bold")
ax.set_xlim(0, 1.85)
ax.grid(axis="x", linestyle="--", alpha=0.4)
for bar, val in zip(bars, vals[::-1]):
    ax.text(val + 0.04, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
            va="center", ha="left", fontsize=9.5, fontweight="bold")

plt.tight_layout()
plt.savefig("fig3_pso_modality_weights.png", dpi=300)
plt.close()

print("Saved fig1_model_ablation.png, fig2_layout_classification.png, and fig3_pso_modality_weights.png")