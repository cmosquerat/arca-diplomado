"""Figuras para Clase 22 — Proyecto MNIST: Imágenes como Datos."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"

# ─────────────────────────────────────────────────────────────────────────────
# Load MNIST (subset for speed)
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando MNIST...")
X_full, y_full = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
rng = np.random.default_rng(42)
idx = rng.choice(len(X_full), 15000, replace=False)
X, y = X_full[idx], y_full[idx]
y_int = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Sample digits — 2 rows of 10
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 10, figsize=(14, 3.5))
for digit in range(10):
    idxs = np.where(y_int == digit)[0]
    for row in range(2):
        ax = axes[row, digit]
        ax.imshow(X[idxs[row]].reshape(28, 28), cmap="gray_r", interpolation="nearest")
        if row == 0:
            ax.set_title(str(digit), fontweight="bold", color=ARCA_DARK, fontsize=14)
        ax.axis("off")
fig.suptitle("MNIST: digitos escritos a mano (28x28 pixeles = 784 variables)",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_mnist_samples.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_mnist_samples.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Average digit per class — what does the "average 3" look like?
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 10, figsize=(14, 2))
for digit in range(10):
    mask = y_int == digit
    avg_img = X[mask].mean(axis=0).reshape(28, 28)
    ax = axes[digit]
    ax.imshow(avg_img, cmap="hot", interpolation="nearest")
    ax.set_title(str(digit), fontweight="bold", color=ARCA_DARK, fontsize=13)
    ax.axis("off")
fig.suptitle("Imagen PROMEDIO de cada digito — los pixeles que siempre se activan",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.05)
plt.tight_layout()
fig.savefig("fig_mnist_promedios.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_mnist_promedios.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Pixel variance — which pixels carry information?
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel 1: variance heatmap
var_map = X.var(axis=0).reshape(28, 28)
ax = axes[0]
im = ax.imshow(var_map, cmap="Reds", interpolation="nearest")
ax.set_title("Varianza por pixel\n(rojo = mucha variacion)", fontweight="bold", color=ARCA_DARK)
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046)

# Panel 2: pixels that are always zero (useless)
zero_pct = (X == 0).mean(axis=0).reshape(28, 28)
ax = axes[1]
im2 = ax.imshow(zero_pct, cmap="Blues", interpolation="nearest")
ax.set_title("% de veces que el pixel es 0\n(azul = siempre vacio)", fontweight="bold", color=ARCA_DARK)
ax.axis("off")
plt.colorbar(im2, ax=ax, fraction=0.046)

fig.suptitle("No todos los pixeles son utiles: las esquinas siempre estan vacias",
             fontweight="bold", color=ARCA_DARK, fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig("fig_mnist_varianza.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_mnist_varianza.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Confusion matrix — Logística
# ─────────────────────────────────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train / 255.0, y_train)
y_pred_lr = lr.predict(X_test / 255.0)

cm = confusion_matrix(y_test, y_pred_lr, labels=[str(i) for i in range(10)])
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=range(10), yticklabels=range(10))
ax.set_xlabel("Predicho", fontsize=12)
ax.set_ylabel("Real", fontsize=12)
acc = accuracy_score(y_test, y_pred_lr)
ax.set_title(f"Confusion Matrix — Logistica (Accuracy: {acc:.1%})",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_confusion_lr.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_confusion_lr.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Misclassified images — visual error analysis
# ─────────────────────────────────────────────────────────────────────────────
wrong_mask = y_pred_lr != y_test
wrong_idx = np.where(wrong_mask)[0]

fig, axes = plt.subplots(2, 10, figsize=(14, 3.5))
for i in range(min(20, len(wrong_idx))):
    ax = axes[i // 10, i % 10]
    idx_w = wrong_idx[i]
    ax.imshow(X_test[idx_w].reshape(28, 28), cmap="gray_r", interpolation="nearest")
    real = y_test[idx_w]
    pred = y_pred_lr[idx_w]
    ax.set_title(f"{real}→{pred}", fontsize=10, color=ARCA_RED, fontweight="bold")
    ax.axis("off")
fig.suptitle("Errores de la logistica: real → predicho (se entiende por que se confunde)",
             fontweight="bold", color=ARCA_DARK, fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig("fig_errores.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_errores.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Confusion matrix — RF
# ─────────────────────────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf, labels=[str(i) for i in range(10)])
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=ax,
            xticklabels=range(10), yticklabels=range(10))
ax.set_xlabel("Predicho", fontsize=12)
ax.set_ylabel("Real", fontsize=12)
acc_rf = accuracy_score(y_test, y_pred_rf)
ax.set_title(f"Confusion Matrix — Random Forest (Accuracy: {acc_rf:.1%})",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_confusion_rf.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_confusion_rf.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Fashion MNIST samples + comparison
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando Fashion MNIST...")
X_f, y_f = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="auto")
y_f_int = y_f.astype(int)
fashion_labels = ["Camiseta", "Pantalon", "Sueter", "Vestido", "Abrigo",
                  "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota"]

fig, axes = plt.subplots(2, 10, figsize=(14, 4))
for cls in range(10):
    idxs_f = np.where(y_f_int == cls)[0]
    for row in range(2):
        ax = axes[row, cls]
        ax.imshow(X_f[idxs_f[row]].reshape(28, 28), cmap="gray_r", interpolation="nearest")
        if row == 0:
            ax.set_title(fashion_labels[cls], fontweight="bold", color=ARCA_DARK, fontsize=9)
        ax.axis("off")
fig.suptitle("Fashion MNIST: misma estructura que MNIST, pero MUCHO mas dificil",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_fashion_samples.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_fashion_samples.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Comparison bar chart — MNIST vs Fashion
# ─────────────────────────────────────────────────────────────────────────────
idx_f = rng.choice(len(X_f), 15000, replace=False)
X_f_sub, y_f_sub = X_f[idx_f], y_f[idx_f]
X_f_tr, X_f_te, y_f_tr, y_f_te = train_test_split(
    X_f_sub, y_f_sub, test_size=0.2, random_state=42, stratify=y_f_sub)

lr_f = LogisticRegression(max_iter=1000, random_state=42)
lr_f.fit(X_f_tr / 255.0, y_f_tr)
acc_lr_f = accuracy_score(y_f_te, lr_f.predict(X_f_te / 255.0))

rf_f = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_f.fit(X_f_tr, y_f_tr)
acc_rf_f = accuracy_score(y_f_te, rf_f.predict(X_f_te))

fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(2)
width = 0.3
bars1 = ax.bar(x_pos - width/2, [acc, acc_lr_f], width, color=BLUE, label="Logistica", alpha=0.8)
bars2 = ax.bar(x_pos + width/2, [acc_rf, acc_rf_f], width, color=GREEN, label="Random Forest", alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(["MNIST\n(digitos)", "Fashion MNIST\n(ropa)"], fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_ylim(0.75, 1.0)
ax.legend(fontsize=10)
for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.1%}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("MNIST vs Fashion MNIST: con ropa nuestros modelos llegan al techo",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
plt.tight_layout()
fig.savefig("fig_comparacion_final.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_comparacion_final.png")

print("\n¡Todas las figuras generadas!")
