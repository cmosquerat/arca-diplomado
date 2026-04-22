"""Figuras estáticas para la presentación de Clase 17 — Árboles de Decisión."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Frontera de decisión — 3 paneles con make_moons (no-lineal)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(42)
X_toy, y_toy = make_moons(n_samples=400, noise=0.25, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: solo los datos
ax = axes[0]
ax.scatter(X_toy[y_toy==0, 0], X_toy[y_toy==0, 1], c=BLUE, s=20,
           edgecolors="white", linewidths=0.4, label="Sano")
ax.scatter(X_toy[y_toy==1, 0], X_toy[y_toy==1, 1], c=ARCA_RED, s=20,
           edgecolors="white", linewidths=0.4, label="ACV")
ax.set_title("Los datos", fontweight="bold", color=ARCA_DARK)
ax.legend(fontsize=8, loc="lower right")
ax.set_xlabel("Variable 1"); ax.set_ylabel("Variable 2")

# Panels 2-3: modelos
models = [
    ("Logística (línea recta)", LogisticRegression()),
    ("Árbol (cortes rectangulares)", DecisionTreeClassifier(max_depth=5, random_state=42)),
]

for ax, (name, model) in zip(axes[1:], models):
    model.fit(X_toy, y_toy)
    acc = accuracy_score(y_toy, model.predict(X_toy))
    xx, yy = np.meshgrid(
        np.linspace(X_toy[:, 0].min()-0.5, X_toy[:, 0].max()+0.5, 300),
        np.linspace(X_toy[:, 1].min()-0.5, X_toy[:, 1].max()+0.5, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.22, levels=[-.5, .5, 1.5],
                colors=[BLUE, ARCA_RED])
    ax.contour(xx, yy, Z, levels=[0.5], colors=[ARCA_DARK], linewidths=1.5)
    ax.scatter(X_toy[y_toy==0, 0], X_toy[y_toy==0, 1], c=BLUE, s=16,
               edgecolors="white", linewidths=0.4)
    ax.scatter(X_toy[y_toy==1, 0], X_toy[y_toy==1, 1], c=ARCA_RED, s=16,
               edgecolors="white", linewidths=0.4)
    ax.set_title(f"{name}\nAccuracy: {acc:.0%}", fontweight="bold", color=ARCA_DARK)
    ax.set_xlabel("Variable 1"); ax.set_ylabel("Variable 2")

fig.suptitle("Problema NO lineal: la logística no puede, el árbol sí",
             fontweight="bold", color=ARCA_DARK, fontsize=14)
plt.tight_layout()
fig.savefig("fig_frontera.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_frontera.png")

# ─────────────────────────────────────────────────────────────────────────────
# Cargar stroke dataset (se usa para Fig 2, 3 y 5)
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd

df = pd.read_csv("stroke.csv")
df = df.drop(columns=["id"])
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df_enc = pd.get_dummies(df, drop_first=True, dtype=int)
X_s = df_enc.drop(columns=["stroke"])
y_s = df_enc["stroke"]
Xtr, Xte, ytr, yte = train_test_split(X_s, y_s, test_size=0.2,
                                       random_state=42, stratify=y_s)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Árbol pequeño visualizado con plot_tree (datos reales de stroke)
# ─────────────────────────────────────────────────────────────────────────────
tree_small = DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                    random_state=42)
tree_small.fit(Xtr, ytr)

fig, ax = plt.subplots(figsize=(16, 7))
plot_tree(tree_small, ax=ax, filled=True, rounded=True, fontsize=9,
          feature_names=X_s.columns,
          class_names=["No ACV", "ACV"],
          impurity=True, proportion=False)
ax.set_title("Árbol de decisión — Stroke (max_depth=3, balanced)",
             fontweight="bold", color=ARCA_DARK, fontsize=14)
plt.tight_layout()
fig.savefig("fig_arbol_visual.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_arbol_visual.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Overfitting — train vs test accuracy por max_depth
# ─────────────────────────────────────────────────────────────────────────────

depths = list(range(1, 21))
acc_train, acc_test = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    t.fit(Xtr, ytr)
    acc_train.append(accuracy_score(ytr, t.predict(Xtr)))
    acc_test.append(accuracy_score(yte, t.predict(Xte)))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(depths, acc_train, "o-", color=BLUE, lw=2, markersize=5, label="Train")
ax.plot(depths, acc_test, "s-", color=ARCA_RED, lw=2, markersize=5, label="Test")
ax.axvline(depths[np.argmax(acc_test)], color=GREEN, linestyle="--", lw=1.3,
           label=f"Mejor test (depth={depths[np.argmax(acc_test)]})")
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.set_title("Overfitting: Train vs Test por profundidad del árbol",
             fontweight="bold", color=ARCA_DARK)
ax.legend()
ax.set_xticks(depths)
plt.tight_layout()
fig.savefig("fig_overfitting.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_overfitting.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Gini intuición — 3 escenarios
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
scenarios = [
    ("Puro (Gini=0)", [1.0, 0.0], [BLUE, ARCA_RED]),
    ("Mezclado (Gini=0.48)", [0.6, 0.4], [BLUE, ARCA_RED]),
    ("Peor caso (Gini=0.50)", [0.5, 0.5], [BLUE, ARCA_RED]),
]
for ax, (title, fracs, colors) in zip(axes, scenarios):
    wedges, _ = ax.pie(fracs, colors=colors, startangle=90,
                       wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title(title, fontweight="bold", color=ARCA_DARK, fontsize=11)
    gini = 1 - sum(p**2 for p in fracs)
    ax.text(0, -1.3, f"Gini = 1 - ({fracs[0]:.1f}² + {fracs[1]:.1f}²) = {gini:.2f}",
            ha="center", fontsize=9, color=GRAY)

fig.suptitle("Índice Gini — Mide la impureza de un nodo",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_gini.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_gini.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Feature importances — árbol vs logística (stroke)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(Xtr_s, ytr)

tree_best = DecisionTreeClassifier(max_depth=5, random_state=42,
                                   class_weight="balanced")
tree_best.fit(Xtr, ytr)

feat_names = X_s.columns
imp_tree = tree_best.feature_importances_
coef_lr = np.abs(lr.coef_[0])
coef_lr_norm = coef_lr / coef_lr.max()
imp_tree_norm = imp_tree / imp_tree.max() if imp_tree.max() > 0 else imp_tree

# Top 10 por importancia del árbol
top_idx = np.argsort(imp_tree)[-10:]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].barh(range(len(top_idx)), imp_tree_norm[top_idx], color=ARCA_RED, height=0.6)
axes[0].set_yticks(range(len(top_idx)))
axes[0].set_yticklabels(feat_names[top_idx], fontsize=9)
axes[0].set_title("Árbol: feature_importances_", fontweight="bold", color=ARCA_DARK)
axes[0].set_xlabel("Importancia (normalizada)")

axes[1].barh(range(len(top_idx)), coef_lr_norm[top_idx], color=BLUE, height=0.6)
axes[1].set_yticks(range(len(top_idx)))
axes[1].set_yticklabels(feat_names[top_idx], fontsize=9)
axes[1].set_title("Logística: |coeficientes|", fontweight="bold", color=ARCA_DARK)
axes[1].set_xlabel("Peso absoluto (normalizado)")

fig.suptitle("¿Qué variables importan? — Árbol vs Logística",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_importances.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_importances.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Analogía "20 preguntas" — diagrama de flujo simple
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.axis("off")

def draw_box(ax, xy, text, color=ARCA_RED, w=2.6, h=0.7):
    rect = mpatches.FancyBboxPatch(
        (xy[0]-w/2, xy[1]-h/2), w, h,
        boxstyle="round,pad=0.12", facecolor=color, edgecolor="white",
        linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")

def draw_leaf(ax, xy, text, color):
    rect = mpatches.FancyBboxPatch(
        (xy[0]-1.1, xy[1]-0.35), 2.2, 0.7,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="white",
        linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], text, ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

def arrow(ax, start, end, label="", side="left"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->, head_width=0.15",
                                color=ARCA_DARK, lw=1.5))
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    offset = -0.3 if side == "left" else 0.3
    ax.text(mid_x + offset, mid_y, label, fontsize=8, color=ARCA_DARK,
            ha="center", fontweight="bold")

# Nodos
draw_box(ax, (5, 5.2), "Edad > 60?")
draw_box(ax, (2.5, 3.5), "Hipertensión = Sí?")
draw_box(ax, (7.5, 3.5), "Glucosa > 200?")

draw_leaf(ax, (1.2, 1.8), "ALTO riesgo\n(72% ACV)", ARCA_RED)
draw_leaf(ax, (3.8, 1.8), "Riesgo MEDIO\n(15% ACV)", ORANGE)
draw_leaf(ax, (6.2, 1.8), "ALTO riesgo\n(45% ACV)", ARCA_RED)
draw_leaf(ax, (8.8, 1.8), "Riesgo BAJO\n(3% ACV)", GREEN)

arrow(ax, (5, 4.85), (2.5, 3.85), "Sí", "left")
arrow(ax, (5, 4.85), (7.5, 3.85), "No", "right")
arrow(ax, (2.5, 3.15), (1.2, 2.15), "Sí", "left")
arrow(ax, (2.5, 3.15), (3.8, 2.15), "No", "right")
arrow(ax, (7.5, 3.15), (6.2, 2.15), "Sí", "left")
arrow(ax, (7.5, 3.15), (8.8, 2.15), "No", "right")

ax.set_title("Un árbol de decisión es un diagrama de flujo",
             fontweight="bold", color=ARCA_DARK, fontsize=13, pad=10)

plt.tight_layout()
fig.savefig("fig_20preguntas.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_20preguntas.png")

print("\n¡Todas las figuras generadas!")
