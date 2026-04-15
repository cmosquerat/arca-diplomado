"""Genera figuras ilustrativas para la presentacion de clase 16."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
ARCA_RED = "#C82B40"
ARCA_DARK = "#6B1525"
GREEN = "#16A34A"
BLUE = "#2563EB"
GRAY = "#9CA3AF"

rng = np.random.default_rng(42)
n_pos, n_neg = 250, 4800
scores_pos = rng.beta(2.2, 2.0, n_pos)
scores_neg = rng.beta(1.5, 4.5, n_neg)
y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
scores = np.concatenate([scores_pos, scores_neg])

fpr, tpr, _ = roc_curve(y, scores)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6.5, 5.2))
ax.fill_between(fpr, tpr, alpha=0.15, color=ARCA_RED)
ax.plot(fpr, tpr, color=ARCA_RED, lw=2.8, label=f"Nuestro modelo (AUC = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], color=GRAY, lw=1.5, linestyle="--", label="Azar (AUC = 0.50)")
ax.plot([0, 0, 1], [0, 1, 1], color=GREEN, lw=1.5, linestyle=":", label="Perfecto (AUC = 1.00)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("FPR — tasa de falsos positivos")
ax.set_ylabel("TPR — recall (verdaderos positivos)")
ax.set_title("Curva ROC", color=ARCA_DARK, fontweight="bold", pad=10)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", frameon=True, fontsize=10)
ax.annotate("Cuanto más pegada\narriba-izquierda,\nmejor el modelo",
            xy=(0.18, 0.78), xytext=(0.45, 0.35),
            fontsize=10, color=ARCA_DARK, ha="center",
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.2))
plt.tight_layout()
plt.savefig("fig_roc.png", dpi=180, bbox_inches="tight")
plt.close()

prec, rec, _ = precision_recall_curve(y, scores)
ap = average_precision_score(y, scores)

fig, ax = plt.subplots(figsize=(6.5, 5.2))
ax.fill_between(rec, prec, alpha=0.15, color=ARCA_RED)
ax.plot(rec, prec, color=ARCA_RED, lw=2.8, label=f"Nuestro modelo (AP = {ap:.2f})")
baseline = y.mean()
ax.axhline(baseline, color=GRAY, lw=1.5, linestyle="--",
           label=f"Baseline (% positivos = {baseline:.2f})")
ax.plot([0, 1, 1], [1, 1, 0], color=GREEN, lw=1.5, linestyle=":", label="Perfecto (AP = 1.00)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("Recall — de los enfermos, ¿cuántos detecto?")
ax.set_ylabel("Precision — de los marcados, ¿cuántos aciertan?")
ax.set_title("Curva Precision-Recall", color=ARCA_DARK, fontweight="bold", pad=10)
ax.grid(alpha=0.3)
ax.legend(loc="lower left", frameon=True, fontsize=10)
ax.annotate("Subir recall\n$\\rightarrow$ baja precision",
            xy=(0.72, 0.30), xytext=(0.45, 0.65),
            fontsize=10, color=ARCA_DARK, ha="center",
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.2))
plt.tight_layout()
plt.savefig("fig_pr.png", dpi=180, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(9, 3.2))
for s in scores_neg[:90]:
    ax.plot([s], [0], "o", color=BLUE, alpha=0.35, markersize=7)
for s in scores_pos[:70]:
    ax.plot([s], [0.35], "o", color=ARCA_RED, alpha=0.6, markersize=7)

threshold = 0.5
ax.axvline(threshold, color=ARCA_DARK, lw=2.3, linestyle="--")
ax.text(threshold, 0.72, f"umbral = {threshold}",
        ha="center", fontsize=11, color=ARCA_DARK, fontweight="bold")
ax.annotate("", xy=(0.96, -0.25), xytext=(0.52, -0.25),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.3))
ax.text(0.74, -0.38, "predicción = 1 (ACV)", ha="center",
        fontsize=10, color=ARCA_DARK, fontweight="bold")
ax.annotate("", xy=(0.04, -0.25), xytext=(0.48, -0.25),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.3))
ax.text(0.26, -0.38, "predicción = 0 (sano)", ha="center",
        fontsize=10, color=ARCA_DARK, fontweight="bold")

ax.text(-0.05, 0.35, "Positivos reales", ha="right", fontsize=10,
        color=ARCA_RED, va="center", fontweight="bold")
ax.text(-0.05, 0.0, "Negativos reales", ha="right", fontsize=10,
        color=BLUE, va="center", fontweight="bold")

ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.55, 0.90)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([])
ax.set_xlabel("Probabilidad predicha por el modelo", fontsize=11)
ax.set_title("Cada predicción es una probabilidad. El umbral decide el corte.",
             color=ARCA_DARK, fontweight="bold", fontsize=12)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("fig_umbral.png", dpi=180, bbox_inches="tight")
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

sizes_total = [95.1, 4.9]
colors_total = [BLUE, ARCA_RED]
labels = ["Sanos (0)", "ACV (1)"]

for ax, title, ratios in [
    (ax1, "Split SIN stratify\n(al azar)", [93.4, 6.6]),
    (ax2, "Split CON stratify\n(mantiene proporción)", [95.0, 5.0]),
]:
    total_bar = ax.barh([2], [100], color=GRAY, alpha=0.3, height=0.5)
    ax.barh([2], [sizes_total[0]], color=BLUE, alpha=0.75, height=0.5, label="Sanos")
    ax.barh([2], [sizes_total[1]], left=sizes_total[0], color=ARCA_RED,
            alpha=0.85, height=0.5, label="ACV")
    ax.text(50, 2.55, f"TOTAL: 95.1% / 4.9%",
            ha="center", fontsize=10, color=ARCA_DARK, fontweight="bold")

    ax.barh([0.8], [ratios[0]], color=BLUE, alpha=0.75, height=0.5)
    ax.barh([0.8], [ratios[1]], left=ratios[0], color=ARCA_RED, alpha=0.85, height=0.5)
    ax.text(50, 1.35, f"TEST: {ratios[0]:.1f}% / {ratios[1]:.1f}%",
            ha="center", fontsize=10,
            color=ARCA_DARK if title.startswith("Split CON") else ARCA_RED,
            fontweight="bold")

    ax.set_xlim(0, 100); ax.set_ylim(0, 3.3)
    ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% de ejemplos")
    ax.set_title(title, color=ARCA_DARK, fontweight="bold", fontsize=12)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

ax1.text(50, 0.15, "¡El test quedó desbalanceado distinto!",
         ha="center", color=ARCA_RED, fontsize=9, fontweight="bold")
ax2.text(50, 0.15, "Misma proporción que el total ✓",
         ha="center", color=GREEN, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("fig_stratify.png", dpi=180, bbox_inches="tight")
plt.close()

print("Figuras:", "fig_roc.png", "fig_pr.png", "fig_umbral.png", "fig_stratify.png")
