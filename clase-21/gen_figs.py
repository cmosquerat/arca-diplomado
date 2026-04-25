"""Figuras para Clase 21 — PCA + K-Means (World Happiness Report)."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"
COLORS_K = [BLUE, ARCA_RED, GREEN, ORANGE, PURPLE, "#06B6D4"]

# ─────────────────────────────────────────────────────────────────────────────
# Load and prepare
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("world_happiness.csv")
features = ["Economy", "Family", "Health", "Freedom", "Generosity", "Corruption", "Job Satisfaction"]
df_clean = df.dropna(subset=features).copy()
X = df_clean[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
df_clean["PC1"] = X_pca[:, 0]
df_clean["PC2"] = X_pca[:, 1]

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: The curse of dimensionality — 7 variables, can't visualize
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

pairs = [("Economy", "Health"), ("Freedom", "Generosity"), ("Family", "Corruption")]
for ax, (v1, v2) in zip(axes, pairs):
    ax.scatter(df_clean[v1], df_clean[v2], c=GRAY, s=20, edgecolors="white", linewidths=0.4)
    ax.set_xlabel(v1); ax.set_ylabel(v2)
    ax.set_title(f"{v1} vs {v2}", fontweight="bold", color=ARCA_DARK, fontsize=10)

fig.suptitle("7 variables = 21 combinaciones de scatter. Imposible ver todo. PCA lo resuelve.",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_dimension_problem.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_dimension_problem.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Scree plot — varianza explicada por componente
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Individual
ax = axes[0]
ax.bar(range(1, 8), pca.explained_variance_ratio_, color=ARCA_RED, alpha=0.8)
ax.set_xlabel("Componente Principal")
ax.set_ylabel("Varianza explicada (%)")
ax.set_title("Varianza por componente", fontweight="bold", color=ARCA_DARK)
ax.set_xticks(range(1, 8))
for i, v in enumerate(pca.explained_variance_ratio_):
    ax.text(i + 1, v + 0.01, f"{v:.1%}", ha="center", fontsize=9, color=ARCA_DARK)

# Cumulative
ax = axes[1]
cum_var = np.cumsum(pca.explained_variance_ratio_)
ax.plot(range(1, 8), cum_var, "o-", color=BLUE, lw=2.5, markersize=8)
ax.axhline(0.7, color=GREEN, linestyle="--", lw=1.5, label="70%")
ax.axhline(0.8, color=ORANGE, linestyle="--", lw=1.5, label="80%")
ax.fill_between(range(1, 8), cum_var, alpha=0.1, color=BLUE)
ax.set_xlabel("Numero de componentes")
ax.set_ylabel("Varianza acumulada")
ax.set_title("Varianza acumulada", fontweight="bold", color=ARCA_DARK)
ax.set_xticks(range(1, 8))
ax.legend(fontsize=9)

fig.suptitle("Scree Plot: 2 componentes capturan el 71% de la informacion",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_scree.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_scree.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: PCA biplot — countries in 2D with loadings
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

# Plot countries colored by region (simplified)
region_colors = {
    "Western Europe": BLUE, "North America": BLUE,
    "Australia and New Zealand": BLUE,
    "Latin America": GREEN, "Central America": GREEN,
    "Eastern Europe": ORANGE, "CIS": ORANGE,
    "East Asia": PURPLE, "Southeast Asia": PURPLE, "South Asia": PURPLE,
    "Middle East": ARCA_RED, "Africa": ARCA_RED,
}
for _, row in df_clean.iterrows():
    color = region_colors.get(row["Region"], GRAY)
    ax.scatter(row["PC1"], row["PC2"], c=color, s=20, edgecolors="white",
               linewidths=0.3, alpha=0.7)

# Loadings (arrows)
loadings = pca.components_[:2].T
scale = 3.5
for i, feat in enumerate(features):
    ax.annotate("", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.8))
    ax.text(loadings[i, 0] * scale * 1.12, loadings[i, 1] * scale * 1.12,
            feat, fontsize=8, fontweight="bold", color=ARCA_DARK, ha="center")

# Legend
from matplotlib.lines import Line2D
legend_items = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=8, label="Occidente"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markersize=8, label="Latinoamerica"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=8, label="Europa del Este"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=PURPLE, markersize=8, label="Asia"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ARCA_RED, markersize=8, label="Medio Oriente / Africa"),
]
ax.legend(handles=legend_items, fontsize=8, loc="lower left")

ax.set_xlabel("PC1 — Desarrollo y calidad de vida (51.7%)", fontsize=11)
ax.set_ylabel("PC2 — Libertad y generosidad (19.2%)", fontsize=11)
ax.set_title("151 paises en 2 dimensiones (PCA)",
             fontweight="bold", color=ARCA_DARK, fontsize=14)
ax.axhline(0, color=GRAY, lw=0.5); ax.axvline(0, color=GRAY, lw=0.5)
plt.tight_layout()
fig.savefig("fig_biplot.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_biplot.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Loadings heatmap — what each PC means
# ─────────────────────────────────────────────────────────────────────────────
loadings_df = pd.DataFrame(pca.components_[:3].T, index=features,
                            columns=["PC1", "PC2", "PC3"])
fig, ax = plt.subplots(figsize=(6, 5))
import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list("", [BLUE, "white", ARCA_RED])
im = ax.imshow(loadings_df.values, cmap=cmap, aspect="auto", vmin=-0.6, vmax=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels(["PC1\n(51.7%)", "PC2\n(19.2%)", "PC3\n(10.0%)"])
ax.set_yticks(range(len(features))); ax.set_yticklabels(features, fontsize=10)
for i in range(len(features)):
    for j in range(3):
        ax.text(j, i, f"{loadings_df.values[i, j]:.2f}", ha="center", va="center",
                fontsize=9, color="black" if abs(loadings_df.values[i, j]) < 0.35 else "white")
plt.colorbar(im, ax=ax, label="Peso (loading)")
ax.set_title("Loadings: que significa cada componente",
             fontweight="bold", color=ARCA_DARK)
plt.tight_layout()
fig.savefig("fig_loadings.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_loadings.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: PCA + K-Means — clusters in PCA space
# ─────────────────────────────────────────────────────────────────────────────
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean["cluster"] = km3.fit_predict(X_pca[:, :2])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel 1: clusters
ax = axes[0]
for k in range(3):
    mask = df_clean["cluster"] == k
    ax.scatter(df_clean.loc[mask, "PC1"], df_clean.loc[mask, "PC2"],
               c=COLORS_K[k], s=25, edgecolors="white", linewidths=0.4,
               label=f"Cluster {k} ({mask.sum()} paises)")
for i, c in enumerate(km3.cluster_centers_):
    ax.scatter(c[0], c[1], c=COLORS_K[i], s=250, marker="X",
               edgecolors="black", linewidths=2, zorder=5)
ax.set_xlabel("PC1 — Desarrollo"); ax.set_ylabel("PC2 — Libertad/generosidad")
ax.set_title("K-Means (K=3) en espacio PCA", fontweight="bold", color=ARCA_DARK)
ax.legend(fontsize=8)

# Panel 2: same but colored by Happiness Score
ax = axes[1]
scatter = ax.scatter(df_clean["PC1"], df_clean["PC2"],
                     c=df_clean["Happiness Score"], cmap="RdYlGn",
                     s=25, edgecolors="white", linewidths=0.4)
plt.colorbar(scatter, ax=ax, label="Happiness Score")
ax.set_xlabel("PC1 — Desarrollo"); ax.set_ylabel("PC2 — Libertad/generosidad")
ax.set_title("Mismos datos, coloreados por Happiness Score",
             fontweight="bold", color=ARCA_DARK)

fig.suptitle("PCA + K-Means: los clusters coinciden con el nivel de felicidad",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_pca_kmeans.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_pca_kmeans.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Before and after PCA — scatter comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(df_clean["Economy"], df_clean["Health"], c=GRAY, s=20,
           edgecolors="white", linewidths=0.4)
ax.set_xlabel("Economy"); ax.set_ylabel("Health")
ax.set_title("Solo 2 de 7 variables\n(perdemos informacion)", fontweight="bold",
             color=ARCA_RED, fontsize=11)

ax = axes[1]
for k in range(3):
    mask = df_clean["cluster"] == k
    ax.scatter(df_clean.loc[mask, "PC1"], df_clean.loc[mask, "PC2"],
               c=COLORS_K[k], s=20, edgecolors="white", linewidths=0.4,
               label=f"Cluster {k}")
ax.set_xlabel("PC1 (51.7%)"); ax.set_ylabel("PC2 (19.2%)")
ax.set_title("PCA: las 7 variables comprimidas en 2\n(71% de la info)",
             fontweight="bold", color=GREEN, fontsize=11)
ax.legend(fontsize=8)

fig.suptitle("Sin PCA elegimos 2 variables y perdemos. Con PCA comprimimos 7 en 2.",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
plt.tight_layout()
fig.savefig("fig_antes_despues.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_antes_despues.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: PCA geometric intuition — 2D rotation of axes
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
# Correlated 2D data
cov = [[2.5, 1.8], [1.8, 1.5]]
X_2d = rng.multivariate_normal([5, 4], cov, 150)

pca_2d = PCA(n_components=2)
pca_2d.fit(StandardScaler().fit_transform(X_2d))
X_2d_c = X_2d - X_2d.mean(axis=0)  # center

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: original axes
ax = axes[0]
ax.scatter(X_2d_c[:, 0], X_2d_c[:, 1], c=GRAY, s=20, edgecolors="white", linewidths=0.4)
ax.axhline(0, color=GRAY, lw=0.5); ax.axvline(0, color=GRAY, lw=0.5)
ax.set_xlabel("Variable original X1", fontsize=10)
ax.set_ylabel("Variable original X2", fontsize=10)
ax.set_title("Ejes originales\n(X1 y X2 correlacionados)", fontweight="bold", color=ARCA_DARK)
ax.set_aspect("equal")
ax.set_xlim(-5, 5); ax.set_ylim(-4, 4)

# Panel 2: PCA axes (rotated)
ax = axes[1]
ax.scatter(X_2d_c[:, 0], X_2d_c[:, 1], c=GRAY, s=20, edgecolors="white", linewidths=0.4, alpha=0.4)

# Draw PCA axes
origin = [0, 0]
sc = StandardScaler().fit(X_2d)
pca_vis = PCA().fit(sc.transform(X_2d))
comps = pca_vis.components_
scale = 3.5
for i, (comp, var, color, label) in enumerate(zip(
    comps, pca_vis.explained_variance_ratio_,
    [ARCA_RED, BLUE], ["PC1", "PC2"])):
    # Need to transform back to original scale
    ax.annotate("", xy=(comp[0]*scale*sc.scale_[0], comp[1]*scale*sc.scale_[1]),
                xytext=origin,
                arrowprops=dict(arrowstyle="->, head_width=0.3", color=color, lw=3))
    ax.text(comp[0]*scale*sc.scale_[0]*1.15, comp[1]*scale*sc.scale_[1]*1.15,
            f"{label} ({var:.0%})", fontsize=11, fontweight="bold", color=color)

ax.axhline(0, color=GRAY, lw=0.5); ax.axvline(0, color=GRAY, lw=0.5)
ax.set_xlabel("Variable original X1", fontsize=10)
ax.set_ylabel("Variable original X2", fontsize=10)
ax.set_title("Ejes de PCA\n(rotados hacia la maxima varianza)", fontweight="bold", color=ARCA_DARK)
ax.set_aspect("equal")
ax.set_xlim(-5, 5); ax.set_ylim(-4, 4)

fig.suptitle("PCA rota los ejes para alinearlos con la direccion de mayor variacion",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_pca_rotacion.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_pca_rotacion.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Curse of dimensionality — distances in high dimensions
# ─────────────────────────────────────────────────────────────────────────────
dims = [2, 5, 10, 50, 100, 500, 1000]
ratio_max_min = []
for d in dims:
    pts = rng.standard_normal((100, d))
    dists = np.array([np.linalg.norm(pts[i] - pts[j])
                      for i in range(100) for j in range(i+1, 100)])
    ratio_max_min.append(dists.max() / dists.min())

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(dims, ratio_max_min, "o-", color=ARCA_RED, lw=2.5, markersize=8)
ax.set_xlabel("Numero de dimensiones")
ax.set_ylabel("Ratio distancia maxima / minima")
ax.set_title("La maldicion de la dimensionalidad:\nlas distancias pierden significado",
             fontweight="bold", color=ARCA_DARK)
ax.set_xscale("log")
plt.tight_layout()
fig.savefig("fig_maldicion.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_maldicion.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: Selecting 2 variables vs PCA — information lost comparison
# ─────────────────────────────────────────────────────────────────────────────
from itertools import combinations

# Variance retained by selecting any pair of original variables
pair_vars = []
for v1, v2 in combinations(features, 2):
    var_kept = X_scaled[:, [features.index(v1), features.index(v2)]].var(axis=0).sum()
    total_var = X_scaled.var(axis=0).sum()
    pair_vars.append({"pair": f"{v1[:4]}+{v2[:4]}", "pct": var_kept / total_var})

pair_df = pd.DataFrame(pair_vars).sort_values("pct", ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(pair_df)), pair_df["pct"], color=GRAY, height=0.7, alpha=0.6)
ax.axvline(cum_var[1], color=GREEN, lw=3, linestyle="-",
           label=f"PCA 2 comp: {cum_var[1]:.1%}")
ax.barh(len(pair_df), cum_var[1], color=GREEN, height=0.7, alpha=0.9)

yticks = list(pair_df["pair"]) + ["PCA (2 comp)"]
ax.set_yticks(range(len(yticks)))
ax.set_yticklabels(yticks, fontsize=7)
ax.set_xlabel("Varianza retenida (%)", fontsize=10)
ax.set_title("Elegir 2 variables a mano vs PCA:\nPCA siempre retiene mas informacion",
             fontweight="bold", color=ARCA_DARK)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig("fig_pca_vs_manual.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_pca_vs_manual.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 10: PCA vs t-SNE — same data, two projections
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Simplified region color mapping
region_colors_map = {
    "Western Europe": BLUE, "North America": BLUE,
    "Australia and New Zealand": BLUE,
    "Latin America": GREEN, "Central America": GREEN,
    "Eastern Europe": ORANGE, "CIS": ORANGE,
    "East Asia": PURPLE, "Southeast Asia": PURPLE, "South Asia": PURPLE,
    "Middle East": ARCA_RED, "Africa": ARCA_RED,
}
colors_per_point = [region_colors_map.get(r, GRAY) for r in df_clean["Region"]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel 1: PCA
ax = axes[0]
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_per_point, s=25,
           edgecolors="white", linewidths=0.4, alpha=0.8)
ax.set_xlabel("PC1 (51.7%)", fontsize=10)
ax.set_ylabel("PC2 (19.2%)", fontsize=10)
ax.set_title("PCA\n(lineal, preserva estructura global)", fontweight="bold",
             color=ARCA_DARK, fontsize=12)

# Panel 2: t-SNE
ax = axes[1]
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors_per_point, s=25,
           edgecolors="white", linewidths=0.4, alpha=0.8)
ax.set_xlabel("t-SNE 1", fontsize=10)
ax.set_ylabel("t-SNE 2", fontsize=10)
ax.set_title("t-SNE\n(no lineal, preserva vecindarios)", fontweight="bold",
             color=ARCA_DARK, fontsize=12)

# Legend
from matplotlib.lines import Line2D
legend_items = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=8, label="Occidente"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markersize=8, label="Latinoamerica"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=8, label="Europa del Este"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=PURPLE, markersize=8, label="Asia"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=ARCA_RED, markersize=8, label="Medio Oriente/Africa"),
]
axes[1].legend(handles=legend_items, fontsize=7, loc="lower right")

fig.suptitle("PCA vs t-SNE: mismos datos, dos formas de comprimir",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_pca_vs_tsne.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_pca_vs_tsne.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 11: PCA vs t-SNE on DIGITS — dramatic difference
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE as TSNE2

digits = load_digits()
X_dig = digits.data
y_dig = digits.target

X_dig_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X_dig))
X_dig_tsne = TSNE2(n_components=2, random_state=42, perplexity=30).fit_transform(X_dig)

DIGIT_COLORS = ["#2563EB", "#C82B40", "#16A34A", "#EA580C", "#7C3AED",
                "#06B6D4", "#D946EF", "#F59E0B", "#6B7280", "#1E293B"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for d in range(10):
    mask = y_dig == d
    ax.scatter(X_dig_pca[mask, 0], X_dig_pca[mask, 1], c=DIGIT_COLORS[d],
               s=8, alpha=0.6, label=str(d))
ax.set_title("PCA (2 componentes)\nLos digitos se solapan", fontweight="bold",
             color=ARCA_RED, fontsize=12)
ax.legend(fontsize=7, ncol=5, loc="upper right", title="Digito")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

ax = axes[1]
for d in range(10):
    mask = y_dig == d
    ax.scatter(X_dig_tsne[mask, 0], X_dig_tsne[mask, 1], c=DIGIT_COLORS[d],
               s=8, alpha=0.6, label=str(d))
ax.set_title("t-SNE\n10 islas claras, una por digito", fontweight="bold",
             color=GREEN, fontsize=12)
ax.legend(fontsize=7, ncol=5, loc="upper right", title="Digito")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

fig.suptitle("64 dimensiones (8x8 pixels): PCA no alcanza, t-SNE revela la estructura",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_tsne_digits.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_tsne_digits.png")

print("\n¡Todas las figuras generadas!")
