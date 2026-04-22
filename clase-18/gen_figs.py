"""Figuras para Clase 18 — Random Forest (dataset Pima Diabetes)."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import make_moons, fetch_openml

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"

# ─────────────────────────────────────────────────────────────────────────────
# Cargar dataset Pima Diabetes
# ─────────────────────────────────────────────────────────────────────────────
d = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
df = d.frame.copy()
df["target"] = (df["class"] == "tested_positive").astype(int)
df = df.drop(columns=["class"])

feat_names_es = {
    "preg": "Embarazos", "plas": "Glucosa", "pres": "Presión",
    "skin": "Pliegue cutáneo", "insu": "Insulina", "mass": "BMI",
    "pedi": "Func. pedigrí", "age": "Edad"
}
X = df.drop(columns=["target"])
y = df["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                       random_state=42, stratify=y)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Inestabilidad — 3 muestras distintas → árboles distintos
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
X_moons, y_moons = make_moons(n_samples=120, noise=0.3, random_state=42)
X_ref = X_moons.copy()
xx, yy = np.meshgrid(
    np.linspace(X_ref[:, 0].min()-0.5, X_ref[:, 0].max()+0.5, 300),
    np.linspace(X_ref[:, 1].min()-0.5, X_ref[:, 1].max()+0.5, 300))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
seeds_data = [42, 43, 44]
titles = ["Muestra A (seed=42)", "Muestra B (seed=43)", "Muestra C (seed=44)"]
for ax, seed, title in zip(axes, seeds_data, titles):
    X_m, y_m = make_moons(n_samples=120, noise=0.3, random_state=seed)
    t = DecisionTreeClassifier(max_depth=None, random_state=0)
    t.fit(X_m, y_m)
    Z = t.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.22, levels=[-.5, .5, 1.5], colors=[BLUE, ARCA_RED])
    ax.contour(xx, yy, Z, levels=[0.5], colors=[ARCA_DARK], linewidths=1.5)
    ax.scatter(X_m[y_m==0, 0], X_m[y_m==0, 1], c=BLUE, s=18,
               edgecolors="white", linewidths=0.4)
    ax.scatter(X_m[y_m==1, 0], X_m[y_m==1, 1], c=ARCA_RED, s=18,
               edgecolors="white", linewidths=0.4)
    ax.set_title(f"{title}\nHojas: {t.get_n_leaves()}", fontweight="bold",
                 color=ARCA_DARK, fontsize=10)
    ax.set_xlabel("Variable 1"); ax.set_ylabel("Variable 2")
fig.suptitle("Inestabilidad: datos ligeramente distintos → fronteras MUY distintas",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_inestabilidad.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_inestabilidad.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Bagging visual — 3 árboles bootstrap + voto
# ─────────────────────────────────────────────────────────────────────────────
X_moons, y_moons = make_moons(n_samples=150, noise=0.25, random_state=42)
xx2, yy2 = np.meshgrid(
    np.linspace(X_moons[:, 0].min()-0.5, X_moons[:, 0].max()+0.5, 300),
    np.linspace(X_moons[:, 1].min()-0.5, X_moons[:, 1].max()+0.5, 300))

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for i, ax in enumerate(axes[:3]):
    idx = rng.choice(len(X_moons), size=len(X_moons), replace=True)
    X_boot, y_boot = X_moons[idx], y_moons[idx]
    t = DecisionTreeClassifier(max_depth=4, random_state=i)
    t.fit(X_boot, y_boot)
    Z = t.predict(np.c_[xx2.ravel(), yy2.ravel()]).reshape(xx2.shape)
    ax.contourf(xx2, yy2, Z, alpha=0.22, levels=[-.5, .5, 1.5], colors=[BLUE, ARCA_RED])
    ax.contour(xx2, yy2, Z, levels=[0.5], colors=[ARCA_DARK], linewidths=1.2)
    ax.scatter(X_boot[y_boot==0, 0], X_boot[y_boot==0, 1], c=BLUE, s=10,
               edgecolors="white", linewidths=0.3, alpha=0.6)
    ax.scatter(X_boot[y_boot==1, 0], X_boot[y_boot==1, 1], c=ARCA_RED, s=10,
               edgecolors="white", linewidths=0.3, alpha=0.6)
    ax.set_title(f"Árbol {i+1}\n(muestra bootstrap)", fontweight="bold",
                 color=ARCA_DARK, fontsize=10)

rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf.fit(X_moons, y_moons)
Z_rf = rf.predict(np.c_[xx2.ravel(), yy2.ravel()]).reshape(xx2.shape)
ax = axes[3]
ax.contourf(xx2, yy2, Z_rf, alpha=0.22, levels=[-.5, .5, 1.5], colors=[BLUE, ARCA_RED])
ax.contour(xx2, yy2, Z_rf, levels=[0.5], colors=[GREEN], linewidths=2.5)
ax.scatter(X_moons[y_moons==0, 0], X_moons[y_moons==0, 1], c=BLUE, s=10,
           edgecolors="white", linewidths=0.3)
ax.scatter(X_moons[y_moons==1, 0], X_moons[y_moons==1, 1], c=ARCA_RED, s=10,
           edgecolors="white", linewidths=0.3)
ax.set_title("Random Forest (100 árboles)\nVoto de la mayoría",
             fontweight="bold", color=GREEN, fontsize=10)
fig.suptitle("Bagging: cada árbol ve datos distintos → el voto suaviza la frontera",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_bagging.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_bagging.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Árbol vs RF — frontera comparada
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
models = [
    ("1 Árbol (max_depth=4)", DecisionTreeClassifier(max_depth=4, random_state=42)),
    ("Random Forest (100 árboles)", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)),
]
for ax, (name, model), bc in zip(axes, models, [ARCA_RED, GREEN]):
    model.fit(X_moons, y_moons)
    Z = model.predict(np.c_[xx2.ravel(), yy2.ravel()]).reshape(xx2.shape)
    ax.contourf(xx2, yy2, Z, alpha=0.22, levels=[-.5, .5, 1.5], colors=[BLUE, ARCA_RED])
    ax.contour(xx2, yy2, Z, levels=[0.5], colors=[bc], linewidths=2)
    ax.scatter(X_moons[y_moons==0, 0], X_moons[y_moons==0, 1], c=BLUE, s=18,
               edgecolors="white", linewidths=0.4)
    ax.scatter(X_moons[y_moons==1, 0], X_moons[y_moons==1, 1], c=ARCA_RED, s=18,
               edgecolors="white", linewidths=0.4)
    ax.set_title(name, fontweight="bold", color=ARCA_DARK)
fig.suptitle("1 Árbol vs Random Forest: el bosque suaviza la frontera",
             fontweight="bold", color=ARCA_DARK, fontsize=14)
plt.tight_layout()
fig.savefig("fig_arbol_vs_rf.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_arbol_vs_rf.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Feature importances — RF vs Árbol (diabetes)
# ─────────────────────────────────────────────────────────────────────────────
tree_single = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
tree_single.fit(Xtr, ytr)
rf_diab = RandomForestClassifier(n_estimators=200, max_depth=8,
                                  class_weight="balanced", random_state=42)
rf_diab.fit(Xtr, ytr)

feat_labels = [feat_names_es.get(c, c) for c in X.columns]
imp_tree = tree_single.feature_importances_
imp_rf = rf_diab.feature_importances_
idx_sorted = np.argsort(imp_rf)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].barh(range(len(idx_sorted)), imp_tree[idx_sorted], color=ARCA_RED, height=0.6)
axes[0].set_yticks(range(len(idx_sorted)))
axes[0].set_yticklabels([feat_labels[i] for i in idx_sorted], fontsize=9)
axes[0].set_title("1 Árbol", fontweight="bold", color=ARCA_DARK)
axes[0].set_xlabel("Importancia")

axes[1].barh(range(len(idx_sorted)), imp_rf[idx_sorted], color=GREEN, height=0.6)
axes[1].set_yticks(range(len(idx_sorted)))
axes[1].set_yticklabels([feat_labels[i] for i in idx_sorted], fontsize=9)
axes[1].set_title("Random Forest (200 árboles)", fontweight="bold", color=GREEN)
axes[1].set_xlabel("Importancia")

fig.suptitle("¿Qué variables importan? — Árbol vs Random Forest",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_importances_rf.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_importances_rf.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: n_estimators vs AUC
# ─────────────────────────────────────────────────────────────────────────────
n_trees_list = [1, 5, 10, 25, 50, 100, 200, 300, 500]
aucs = []
for n in n_trees_list:
    rf_n = RandomForestClassifier(n_estimators=n, max_depth=8,
                                  class_weight="balanced", random_state=42)
    rf_n.fit(Xtr, ytr)
    prob = rf_n.predict_proba(Xte)[:, 1]
    aucs.append(roc_auc_score(yte, prob))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(n_trees_list, aucs, "o-", color=GREEN, lw=2, markersize=7)
ax.axhline(y=max(aucs), color=GRAY, linestyle="--", lw=1)
ax.set_xlabel("n_estimators (número de árboles)")
ax.set_ylabel("AUC en test")
ax.set_title("¿Cuántos árboles necesito?",
             fontweight="bold", color=ARCA_DARK)
ax.set_xscale("log")
ax.set_xticks(n_trees_list)
ax.set_xticklabels(n_trees_list)
plt.tight_layout()
fig.savefig("fig_n_estimators.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_n_estimators.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: K-Fold CV diagram
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
K = 5
for fold in range(K):
    for block in range(K):
        x = block * 1.6
        y_pos = (K - 1 - fold) * 0.7
        color = ARCA_RED if block == fold else BLUE
        alpha = 1.0 if block == fold else 0.35
        rect = mpatches.FancyBboxPatch(
            (x, y_pos), 1.4, 0.5, boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white", linewidth=2, alpha=alpha)
        ax.add_patch(rect)
        label = "Val" if block == fold else "Train"
        ax.text(x + 0.7, y_pos + 0.25, label, ha="center", va="center",
                fontsize=9 if block == fold else 8,
                fontweight="bold" if block == fold else "normal",
                color="white", alpha=1.0 if block == fold else 0.9)
    ax.text(K * 1.6 + 0.3, (K - 1 - fold) * 0.7 + 0.25,
            f"Fold {fold+1}", ha="left", va="center",
            fontsize=10, fontweight="bold", color=ARCA_DARK)

ax.set_xlim(-0.3, K * 1.6 + 1.5)
ax.set_ylim(-0.3, K * 0.7 + 0.3)
ax.axis("off")
ax.set_title("5-Fold Cross-Validation: cada bloque rota como validación",
             fontweight="bold", color=ARCA_DARK, fontsize=13, pad=10)
train_patch = mpatches.Patch(facecolor=BLUE, alpha=0.35, label="Train (80%)")
test_patch = mpatches.Patch(facecolor=ARCA_RED, label="Validación (20%)")
ax.legend(handles=[train_patch, test_patch], loc="lower right", fontsize=9)
plt.tight_layout()
fig.savefig("fig_cv.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_cv.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Comparación triple — AUC de 3 modelos con CV (diabetes)
# ─────────────────────────────────────────────────────────────────────────────
modelos = {
    "Logística": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Árbol (d=5)": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
}

fig, ax = plt.subplots(figsize=(8, 4.5))
positions = []
for i, (nombre, mod) in enumerate(modelos.items()):
    X_cv = Xtr_s if "Logís" in nombre else Xtr
    scores = cross_val_score(mod, X_cv, ytr, cv=5, scoring="roc_auc")
    bp = ax.boxplot(scores, positions=[i], widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor=[BLUE, ARCA_RED, GREEN][i], alpha=0.4),
                    medianprops=dict(color=ARCA_DARK, lw=2))
    ax.text(i, scores.mean() + 0.015, f"{scores.mean():.3f}",
            ha="center", fontsize=10, fontweight="bold", color=ARCA_DARK)

ax.set_xticks(range(len(modelos)))
ax.set_xticklabels(modelos.keys(), fontsize=11)
ax.set_ylabel("AUC (5-fold CV)")
ax.set_title("Comparación: AUC por modelo (5-fold CV sobre train)",
             fontweight="bold", color=ARCA_DARK)
plt.tight_layout()
fig.savefig("fig_comparacion_cv.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_comparacion_cv.png")

print("\n¡Todas las figuras generadas!")
