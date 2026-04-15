"""Figuras para construccion punto-a-punto de ROC/PR + calculo de AUC/AP."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"

rng = np.random.default_rng(42)
n_pos, n_neg = 250, 4800
scores_pos = rng.beta(2.2, 2.0, n_pos)
scores_neg = rng.beta(1.5, 4.5, n_neg)
y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
scores = np.concatenate([scores_pos, scores_neg])

def metrics_at(thr):
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, tpr=tpr, fpr=fpr, prec=prec, rec=tpr)

fpr_full, tpr_full, _ = roc_curve(y, scores)
prec_full, rec_full, _ = precision_recall_curve(y, scores)

thresholds_demo = [0.2, 0.45, 0.70]
colors_demo = [ORANGE, PURPLE, GREEN]
labels_demo = ["Umbral BAJO (0.20)", "Umbral MEDIO (0.45)", "Umbral ALTO (0.70)"]
pts = [metrics_at(t) for t in thresholds_demo]

fig, (axc, axm) = plt.subplots(1, 2, figsize=(11, 5.2),
                                gridspec_kw=dict(width_ratios=[1.1, 1]))

axc.fill_between(fpr_full, tpr_full, alpha=0.12, color=ARCA_RED)
axc.plot(fpr_full, tpr_full, color=ARCA_RED, lw=2.5, label="Curva ROC completa")
axc.plot([0, 1], [0, 1], color=GRAY, lw=1.3, linestyle="--", label="Azar")

for pt, col, lbl in zip(pts, colors_demo, labels_demo):
    axc.plot(pt["fpr"], pt["tpr"], "o", color=col, markersize=14, markeredgecolor="white", markeredgewidth=2, zorder=5)
    axc.annotate(f"{lbl}\n(FPR={pt['fpr']:.2f}, TPR={pt['tpr']:.2f})",
                 xy=(pt["fpr"], pt["tpr"]),
                 xytext=(pt["fpr"] + 0.15, pt["tpr"] - 0.10),
                 fontsize=9, color=col, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=col, lw=1))

axc.set_xlim(-0.02, 1.02); axc.set_ylim(-0.02, 1.05)
axc.set_xlabel("FPR — falsos positivos / sanos totales")
axc.set_ylabel("TPR — verdaderos positivos / enfermos totales")
axc.set_title("Cada umbral = UN punto en la curva", color=ARCA_DARK, fontweight="bold")
axc.grid(alpha=0.3); axc.legend(loc="lower right", fontsize=9)

axm.axis("off")
axm.set_title("Paso a paso por cada umbral", color=ARCA_DARK, fontweight="bold", pad=10)

y0 = 0.95
for pt, col, lbl in zip(pts, colors_demo, labels_demo):
    axm.add_patch(mpatches.Rectangle((0.02, y0 - 0.26), 0.96, 0.25,
                                      facecolor=col, alpha=0.12, edgecolor=col, linewidth=1.5))
    axm.text(0.05, y0 - 0.04, lbl, fontsize=11, fontweight="bold", color=col)
    axm.text(0.05, y0 - 0.10, f"VP = {pt['tp']}    FN = {pt['fn']}    FP = {pt['fp']}    VN = {pt['tn']}",
             fontsize=10, family="monospace", color=ARCA_DARK)
    axm.text(0.05, y0 - 0.16, f"TPR = VP/(VP+FN) = {pt['tp']}/{pt['tp']+pt['fn']} = {pt['tpr']:.2f}",
             fontsize=10, family="monospace", color=ARCA_DARK)
    axm.text(0.05, y0 - 0.22, f"FPR = FP/(FP+VN) = {pt['fp']}/{pt['fp']+pt['tn']} = {pt['fpr']:.2f}",
             fontsize=10, family="monospace", color=ARCA_DARK)
    y0 -= 0.33

plt.tight_layout()
plt.savefig("fig_roc_puntos.png", dpi=180, bbox_inches="tight")
plt.close()

fig, (axc, axm) = plt.subplots(1, 2, figsize=(11, 5.2),
                                gridspec_kw=dict(width_ratios=[1.1, 1]))

axc.fill_between(rec_full, prec_full, alpha=0.12, color=ARCA_RED)
axc.plot(rec_full, prec_full, color=ARCA_RED, lw=2.5, label="Curva PR completa")
baseline = y.mean()
axc.axhline(baseline, color=GRAY, lw=1.3, linestyle="--", label=f"Baseline = {baseline:.2f}")

for pt, col, lbl in zip(pts, colors_demo, labels_demo):
    axc.plot(pt["rec"], pt["prec"], "o", color=col, markersize=14, markeredgecolor="white", markeredgewidth=2, zorder=5)
    axc.annotate(f"{lbl}\n(Rec={pt['rec']:.2f}, Prec={pt['prec']:.2f})",
                 xy=(pt["rec"], pt["prec"]),
                 xytext=(pt["rec"] - 0.35, pt["prec"] + 0.08),
                 fontsize=9, color=col, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=col, lw=1))

axc.set_xlim(-0.02, 1.02); axc.set_ylim(-0.02, 1.05)
axc.set_xlabel("Recall — enfermos detectados / enfermos totales")
axc.set_ylabel("Precision — aciertos / total de alertas")
axc.set_title("Cada umbral = UN punto en la curva", color=ARCA_DARK, fontweight="bold")
axc.grid(alpha=0.3); axc.legend(loc="upper right", fontsize=9)

axm.axis("off")
axm.set_title("Paso a paso por cada umbral", color=ARCA_DARK, fontweight="bold", pad=10)

y0 = 0.95
for pt, col, lbl in zip(pts, colors_demo, labels_demo):
    axm.add_patch(mpatches.Rectangle((0.02, y0 - 0.26), 0.96, 0.25,
                                      facecolor=col, alpha=0.12, edgecolor=col, linewidth=1.5))
    axm.text(0.05, y0 - 0.04, lbl, fontsize=11, fontweight="bold", color=col)
    axm.text(0.05, y0 - 0.10, f"VP = {pt['tp']}    FN = {pt['fn']}    FP = {pt['fp']}    VN = {pt['tn']}",
             fontsize=10, family="monospace", color=ARCA_DARK)
    axm.text(0.05, y0 - 0.16, f"Precision = VP/(VP+FP) = {pt['tp']}/{pt['tp']+pt['fp']} = {pt['prec']:.2f}",
             fontsize=10, family="monospace", color=ARCA_DARK)
    axm.text(0.05, y0 - 0.22, f"Recall    = VP/(VP+FN) = {pt['tp']}/{pt['tp']+pt['fn']} = {pt['rec']:.2f}",
             fontsize=10, family="monospace", color=ARCA_DARK)
    y0 -= 0.33

plt.tight_layout()
plt.savefig("fig_pr_puntos.png", dpi=180, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))

n_sample = 8
idx = np.linspace(0, len(fpr_full)-1, n_sample).astype(int)
xs = fpr_full[idx]; ys = tpr_full[idx]

for i in range(len(xs)-1):
    x_pair = [xs[i], xs[i], xs[i+1], xs[i+1]]
    y_pair = [0, ys[i], ys[i+1], 0]
    ax.fill(x_pair, y_pair, alpha=0.35, color=ARCA_RED, edgecolor=ARCA_DARK, linewidth=0.8)

ax.plot(fpr_full, tpr_full, color=ARCA_DARK, lw=2.2, label="Curva ROC", zorder=4)
ax.plot(xs, ys, "o", color=ARCA_DARK, markersize=9, zorder=5,
        markerfacecolor="white", markeredgewidth=2)
ax.plot([0, 1], [0, 1], color=GRAY, lw=1.2, linestyle="--")

total_auc = auc(fpr_full, tpr_full)
ax.text(0.55, 0.25, f"AUC = {total_auc:.3f}\n(suma de áreas\nde los trapecios)",
        fontsize=13, color=ARCA_DARK, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=ARCA_RED, linewidth=1.5))

ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("AUC = integral numérica bajo la curva (regla del trapecio)",
             color=ARCA_DARK, fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_auc_trapecios.png", dpi=180, bbox_inches="tight")
plt.close()

print("Figuras:", "fig_roc_puntos.png", "fig_pr_puntos.png", "fig_auc_trapecios.png")
