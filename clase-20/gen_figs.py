"""Figuras para Clase 20 — Interpretacion de Clusters, K-Means 3D y Pipelines."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"; AMBER = "#D97706"
COLORS_K = [BLUE, ARCA_RED, GREEN, ORANGE, PURPLE, AMBER, "#06B6D4"]

# ─────────────────────────────────────────────────────────────────────────────
# Cargar Mall Customers
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("https://raw.githubusercontent.com/cmosquerat/arca-diplomado/main/clase-19/mall_customers.csv")
df = df.rename(columns={
    "Annual Income (k$)": "income",
    "Spending Score (1-100)": "spending",
})

X_2d = df[["income", "spending"]].values
km_2d = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = km_2d.fit_predict(X_2d)

# Nombres de segmento
perfil = df.groupby("cluster")[["income", "spending"]].mean()
nombres = {}
for c in perfil.index:
    inc, sp = perfil.loc[c, "income"], perfil.loc[c, "spending"]
    if inc > 70 and sp > 60: nombres[c] = "Premium"
    elif inc > 70 and sp < 40: nombres[c] = "Ahorradores VIP"
    elif inc < 40 and sp > 60: nombres[c] = "Entusiastas"
    elif inc < 40 and sp < 40: nombres[c] = "Cautelosos"
    else: nombres[c] = "Mainstream"
df["segmento"] = df["cluster"].map(nombres)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Segmentos 2D con nombres de negocio
# ─────────────────────────────────────────────────────────────────────────────
seg_colors = {"Premium": GREEN, "Ahorradores VIP": BLUE,
              "Entusiastas": ORANGE, "Cautelosos": PURPLE, "Mainstream": GRAY}

fig, ax = plt.subplots(figsize=(9, 6))
for seg, color in seg_colors.items():
    mask = df["segmento"] == seg
    ax.scatter(df.loc[mask, "income"], df.loc[mask, "spending"],
               c=color, s=40, edgecolors="white", linewidths=0.5,
               label=seg, alpha=0.8)

for i, c in enumerate(km_2d.cluster_centers_):
    ax.scatter(c[0], c[1], c=COLORS_K[i], s=200, marker="X",
               edgecolors="black", linewidths=1.5, zorder=5)

ax.set_xlabel("Ingreso anual (k$)", fontsize=12)
ax.set_ylabel("Spending Score (1-100)", fontsize=12)
ax.set_title("Segmentos con nombres de negocio",
             fontweight="bold", color=ARCA_DARK, fontsize=14)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig("fig_segmentos_nombres.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_segmentos_nombres.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Barras agrupadas ingreso vs gasto por segmento
# ─────────────────────────────────────────────────────────────────────────────
perfil_seg = df.groupby("segmento")[["income", "spending"]].mean().round(1)
perfil_seg = perfil_seg.loc[["Premium", "Ahorradores VIP", "Mainstream",
                              "Entusiastas", "Cautelosos"]]

x = np.arange(len(perfil_seg))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, perfil_seg["income"], w, label="Ingreso (k$)", color=BLUE)
ax.bar(x + w/2, perfil_seg["spending"], w, label="Spending Score", color=ARCA_RED)
ax.set_xticks(x)
ax.set_xticklabels(perfil_seg.index, fontsize=10)
ax.set_ylabel("Promedio")
ax.set_title("Ingreso vs Gasto por segmento",
             fontweight="bold", color=ARCA_DARK, fontsize=14)
ax.legend()
plt.tight_layout()
fig.savefig("fig_barras_segmento.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_barras_segmento.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Radar chart de perfiles
# ─────────────────────────────────────────────────────────────────────────────
perfil_radar = df.groupby("segmento")[["Age", "income", "spending"]].mean()
perfil_norm = (perfil_radar - perfil_radar.min()) / (perfil_radar.max() - perfil_radar.min())

categorias = list(perfil_norm.columns)
N = len(categorias)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for seg in perfil_norm.index:
    vals = list(perfil_norm.loc[seg]) + [perfil_norm.loc[seg].iloc[0]]
    ax.plot(angles, vals, "o-", linewidth=2, label=seg, color=seg_colors[seg])
    ax.fill(angles, vals, alpha=0.15, color=seg_colors[seg])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(["Edad", "Ingreso", "Gasto"], fontsize=12)
ax.set_ylim(0, 1)
ax.set_title("Perfil de cada segmento (normalizado 0-1)",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
fig.savefig("fig_radar.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_radar.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Codo + Silhouette para 3D
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_3d = scaler.fit_transform(df[["income", "spending", "Age"]])

inertias = []
sils = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labs = km.fit_predict(X_3d)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_3d, labs))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(list(K_range), inertias, "o-", color=BLUE, lw=2.5, markersize=8)
ax.set_xlabel("K"); ax.set_ylabel("Inercia")
ax.set_title("Metodo del Codo (3 variables)", fontweight="bold", color=ARCA_DARK)
ax.set_xticks(list(K_range))

ax = axes[1]
ax.plot(list(K_range), sils, "o-", color=ARCA_RED, lw=2.5, markersize=8)
ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Score (3 variables)", fontweight="bold", color=ARCA_DARK)
ax.set_xticks(list(K_range))

plt.tight_layout()
fig.savefig("fig_codo_silhouette_3d.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_codo_silhouette_3d.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Heatmap de perfiles 3D
# ─────────────────────────────────────────────────────────────────────────────
import seaborn as sns

km_3d = KMeans(n_clusters=6, random_state=42, n_init=10)
df["cluster_3d"] = km_3d.fit_predict(X_3d)

perfil_3d = df.groupby("cluster_3d")[["Age", "income", "spending"]].mean().round(1)
perfil_3d.columns = ["Edad", "Ingreso (k$)", "Spending Score"]
perfil_3d_norm = (perfil_3d - perfil_3d.min()) / (perfil_3d.max() - perfil_3d.min())

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(perfil_3d_norm, annot=perfil_3d.values, fmt=".1f",
            cmap="RdYlGn", linewidths=1, ax=ax)
ax.set_title("Perfil de cada cluster 3D\n(color = normalizado, numero = real)",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
ax.set_ylabel("Cluster")
plt.tight_layout()
fig.savefig("fig_heatmap_3d.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_heatmap_3d.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Pipeline diagram (conceptual)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 3))
ax.set_xlim(0, 10); ax.set_ylim(0, 2)
ax.axis("off")

# Sin pipeline
y_top = 1.5
boxes_top = [("X_train", 0.5), ("StandardScaler\nfit_transform", 2.5),
             ("X_train_s", 4.5), ("Modelo\n.fit()", 6.5), ("Prediccion", 8.5)]
for label, x in boxes_top:
    color = GRAY if "X_" in label or "Pred" in label else ARCA_RED
    ax.add_patch(plt.Rectangle((x-0.7, y_top-0.3), 1.4, 0.6,
                 facecolor=color if "X_" not in label and "Pred" not in label else "#E5E7EB",
                 edgecolor=ARCA_DARK, linewidth=1.5, zorder=2, alpha=0.3))
    ax.text(x, y_top, label, ha="center", va="center", fontsize=8,
            fontweight="bold", color=ARCA_DARK, zorder=3)

for i in range(len(boxes_top)-1):
    ax.annotate("", xy=(boxes_top[i+1][1]-0.7, y_top),
                xytext=(boxes_top[i][1]+0.7, y_top),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))

ax.text(0.5, y_top+0.45, "SIN Pipeline (manual, propenso a errores)",
        fontsize=10, fontweight="bold", color=ARCA_RED, ha="left")

# Con pipeline
y_bot = 0.4
ax.add_patch(plt.Rectangle((1.3, y_bot-0.35), 6.4, 0.7,
             facecolor=GREEN, edgecolor=ARCA_DARK, linewidth=2,
             alpha=0.15, zorder=1))
ax.text(4.5, y_bot+0.5, "Pipeline", fontsize=11, fontweight="bold",
        color=GREEN, ha="center")

boxes_bot = [("X_train", 0.5), ("StandardScaler", 3.0), ("Modelo", 5.5),
             ("Prediccion", 8.5)]
for label, x in boxes_bot:
    inside = x > 1.3 and x < 7.7
    fc = GREEN if inside else "#E5E7EB"
    ax.add_patch(plt.Rectangle((x-0.7, y_bot-0.25), 1.4, 0.5,
                 facecolor=fc, edgecolor=ARCA_DARK, linewidth=1.5,
                 zorder=2, alpha=0.3))
    ax.text(x, y_bot, label, ha="center", va="center", fontsize=9,
            fontweight="bold", color=ARCA_DARK, zorder=3)

for i in range(len(boxes_bot)-1):
    ax.annotate("", xy=(boxes_bot[i+1][1]-0.7, y_bot),
                xytext=(boxes_bot[i][1]+0.7, y_bot),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))

ax.text(0.5, y_bot-0.55, "CON Pipeline (automatico, seguro, sin data leakage)",
        fontsize=10, fontweight="bold", color=GREEN, ha="left")

fig.suptitle("Pipeline encadena preprocesamiento + modelo",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig("fig_pipeline.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_pipeline.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Data leakage diagram
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

# MAL: escalar todo antes de split
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
ax.set_title("MAL: escalar antes de dividir", fontweight="bold", color=ARCA_RED, fontsize=12)

# Bloque datos completos
ax.add_patch(plt.Rectangle((0.5, 2.0), 3, 0.6, facecolor="#FEE2E2",
             edgecolor=ARCA_RED, linewidth=2))
ax.text(2, 2.3, "Todos los datos", ha="center", fontsize=9, fontweight="bold")

ax.annotate("", xy=(2, 1.6), xytext=(2, 2.0),
            arrowprops=dict(arrowstyle="->", color=ARCA_RED, lw=2))
ax.text(4.5, 1.8, "fit_transform(X)", fontsize=8, color=ARCA_RED, fontstyle="italic")

ax.add_patch(plt.Rectangle((0.5, 1.0), 3, 0.6, facecolor="#FEE2E2",
             edgecolor=ARCA_RED, linewidth=2))
ax.text(2, 1.3, "X escalado (CONTAMINADO)", ha="center", fontsize=8, fontweight="bold",
        color=ARCA_RED)

ax.annotate("", xy=(1, 0.5), xytext=(1, 1.0),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))
ax.annotate("", xy=(3, 0.5), xytext=(3, 1.0),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))

ax.add_patch(plt.Rectangle((0, 0), 2, 0.5, facecolor="#DBEAFE", edgecolor=BLUE, linewidth=1.5))
ax.text(1, 0.25, "Train", ha="center", fontsize=9, fontweight="bold", color=BLUE)
ax.add_patch(plt.Rectangle((2.2, 0), 2, 0.5, facecolor="#FEF3C7", edgecolor=ORANGE, linewidth=1.5))
ax.text(3.2, 0.25, "Test", ha="center", fontsize=9, fontweight="bold", color=ORANGE)

ax.text(5, 0.8, "El scaler VIO\nlos datos de test!", fontsize=9,
        color=ARCA_RED, fontweight="bold")

# BIEN: Pipeline dentro de CV
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
ax.set_title("BIEN: Pipeline dentro de CV", fontweight="bold", color=GREEN, fontsize=12)

ax.add_patch(plt.Rectangle((0.5, 2.0), 3, 0.6, facecolor="#DCFCE7",
             edgecolor=GREEN, linewidth=2))
ax.text(2, 2.3, "Todos los datos", ha="center", fontsize=9, fontweight="bold")

ax.annotate("", xy=(1, 1.6), xytext=(1, 2.0),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))
ax.annotate("", xy=(3, 1.6), xytext=(3, 2.0),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))

ax.add_patch(plt.Rectangle((0, 1.0), 2, 0.6, facecolor="#DBEAFE", edgecolor=BLUE, linewidth=1.5))
ax.text(1, 1.3, "Train fold", ha="center", fontsize=9, fontweight="bold", color=BLUE)
ax.add_patch(plt.Rectangle((2.2, 1.0), 2, 0.6, facecolor="#FEF3C7", edgecolor=ORANGE, linewidth=1.5))
ax.text(3.2, 1.3, "Val fold", ha="center", fontsize=9, fontweight="bold", color=ORANGE)

ax.text(1, 1.75, "split primero", fontsize=8, ha="center", color=GRAY, fontstyle="italic")

ax.annotate("", xy=(1, 0.5), xytext=(1, 1.0),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
ax.text(2.5, 0.7, "fit_transform\n(solo train)", fontsize=8, color=GREEN, fontstyle="italic")

ax.add_patch(plt.Rectangle((0, 0), 2, 0.5, facecolor="#DCFCE7", edgecolor=GREEN, linewidth=2))
ax.text(1, 0.25, "Scaler + Modelo", ha="center", fontsize=8, fontweight="bold", color=GREEN)

ax.annotate("", xy=(3.2, 0.5), xytext=(3.2, 1.0),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
ax.text(4.5, 0.7, "transform\n(solo val)", fontsize=8, color=GREEN, fontstyle="italic")

ax.text(5.5, 0.1, "El scaler NUNCA\nve los datos de val", fontsize=9,
        color=GREEN, fontweight="bold")

plt.tight_layout()
fig.savefig("fig_data_leakage.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_data_leakage.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: GridSearchCV con Pipeline (notacion)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")

# Pipeline box
ax.add_patch(plt.Rectangle((0.5, 1.5), 9, 2, facecolor=GREEN,
             edgecolor=ARCA_DARK, linewidth=2, alpha=0.1))
ax.text(5, 3.7, "Pipeline", fontsize=14, fontweight="bold",
        color=GREEN, ha="center")

# Scaler box
ax.add_patch(plt.Rectangle((1, 2.0), 3, 1, facecolor=BLUE,
             edgecolor=ARCA_DARK, linewidth=1.5, alpha=0.2))
ax.text(2.5, 2.5, '"scaler"\nStandardScaler()', ha="center",
        fontsize=10, fontweight="bold", color=ARCA_DARK)

# Arrow
ax.annotate("", xy=(5, 2.5), xytext=(4, 2.5),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))

# Model box
ax.add_patch(plt.Rectangle((5.5, 2.0), 3.5, 1, facecolor=ARCA_RED,
             edgecolor=ARCA_DARK, linewidth=1.5, alpha=0.2))
ax.text(7.25, 2.5, '"modelo"\nRandomForestClassifier()', ha="center",
        fontsize=10, fontweight="bold", color=ARCA_DARK)

# GridSearchCV notation
ax.text(1, 0.8, 'param_grid = {', fontsize=10, fontfamily="monospace", color=ARCA_DARK)
ax.text(1.5, 0.4, '"modelo__n_estimators": [100, 200],', fontsize=10,
        fontfamily="monospace", color=PURPLE)
ax.text(1.5, 0.05, '"modelo__max_depth": [5, 8, None]', fontsize=10,
        fontfamily="monospace", color=PURPLE)
ax.text(1, -0.3, '}', fontsize=10, fontfamily="monospace", color=ARCA_DARK)

ax.annotate("", xy=(5.5, 2.0), xytext=(3.5, 0.8),
            arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5,
                          connectionstyle="arc3,rad=-0.3"))
ax.text(5, 1.2, 'nombre_paso__parametro', fontsize=9, fontstyle="italic",
        color=PURPLE, fontweight="bold")

fig.suptitle("GridSearchCV con Pipeline: notacion doble guion bajo",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=1.0)
plt.tight_layout()
fig.savefig("fig_gridsearch_pipeline.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_gridsearch_pipeline.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: K-Means FALLA en formas no esfericas (moons + circles)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import DBSCAN

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# --- Fila 1: Moons ---
X_moons, y_moons = make_moons(n_samples=300, noise=0.08, random_state=42)

# Datos reales
ax = axes[0, 0]
for k in [0, 1]:
    mask = y_moons == k
    ax.scatter(X_moons[mask, 0], X_moons[mask, 1], c=COLORS_K[k], s=20,
               edgecolors="white", linewidths=0.4)
ax.set_title("Datos reales\n(2 lunas)", fontweight="bold", color=ARCA_DARK)

# K-Means en moons
ax = axes[0, 1]
km_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km = km_moons.fit_predict(X_moons)
for k in [0, 1]:
    mask = labels_km == k
    ax.scatter(X_moons[mask, 0], X_moons[mask, 1], c=COLORS_K[k], s=20,
               edgecolors="white", linewidths=0.4)
ax.set_title("K-Means (K=2)\nFALLA", fontweight="bold", color=ARCA_RED)

# DBSCAN en moons
ax = axes[0, 2]
db_moons = DBSCAN(eps=0.2, min_samples=5)
labels_db = db_moons.fit_predict(X_moons)
for k in sorted(set(labels_db)):
    mask = labels_db == k
    if k == -1:
        ax.scatter(X_moons[mask, 0], X_moons[mask, 1], c="gray", s=15,
                   marker="x", alpha=0.5, label="Ruido")
    else:
        ax.scatter(X_moons[mask, 0], X_moons[mask, 1], c=COLORS_K[k], s=20,
                   edgecolors="white", linewidths=0.4)
ax.set_title("DBSCAN\nACIERTA", fontweight="bold", color=GREEN)

# --- Fila 2: Circles ---
X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

# Datos reales
ax = axes[1, 0]
for k in [0, 1]:
    mask = y_circles == k
    ax.scatter(X_circles[mask, 0], X_circles[mask, 1], c=COLORS_K[k], s=20,
               edgecolors="white", linewidths=0.4)
ax.set_title("Datos reales\n(2 circulos)", fontweight="bold", color=ARCA_DARK)

# K-Means en circles
ax = axes[1, 1]
km_circles = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km_c = km_circles.fit_predict(X_circles)
for k in [0, 1]:
    mask = labels_km_c == k
    ax.scatter(X_circles[mask, 0], X_circles[mask, 1], c=COLORS_K[k], s=20,
               edgecolors="white", linewidths=0.4)
ax.set_title("K-Means (K=2)\nFALLA", fontweight="bold", color=ARCA_RED)

# DBSCAN en circles
ax = axes[1, 2]
db_circles = DBSCAN(eps=0.2, min_samples=5)
labels_db_c = db_circles.fit_predict(X_circles)
for k in sorted(set(labels_db_c)):
    mask = labels_db_c == k
    if k == -1:
        ax.scatter(X_circles[mask, 0], X_circles[mask, 1], c="gray", s=15,
                   marker="x", alpha=0.5, label="Ruido")
    else:
        ax.scatter(X_circles[mask, 0], X_circles[mask, 1], c=COLORS_K[k], s=20,
                   edgecolors="white", linewidths=0.4)
ax.set_title("DBSCAN\nACIERTA", fontweight="bold", color=GREEN)

for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("K-Means asume clusters esfericos — DBSCAN no",
             fontweight="bold", color=ARCA_DARK, fontsize=14)
plt.tight_layout()
fig.savefig("fig_kmeans_falla.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_kmeans_falla.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 10: DBSCAN — como funciona (eps y min_samples visual)
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
X_demo = rng.normal(size=(40, 2)) * 0.8
X_demo = np.vstack([X_demo, [[3.5, 3.0], [3.8, 2.7], [-3.0, 2.5]]])  # outliers

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_demo[:40, 0], X_demo[:40, 1], c=BLUE, s=50,
           edgecolors="white", linewidths=0.5, zorder=3)
ax.scatter(X_demo[40:, 0], X_demo[40:, 1], c=GRAY, s=50, marker="x",
           linewidths=2, zorder=3, label="Puntos aislados (ruido)")

# Circulo eps alrededor de un punto central
center_idx = 20
cx, cy = X_demo[center_idx]
eps_val = 0.8
circle = plt.Circle((cx, cy), eps_val, fill=False, color=ARCA_RED,
                     linewidth=2, linestyle="--", zorder=2)
ax.add_patch(circle)
ax.scatter(cx, cy, c=ARCA_RED, s=100, edgecolors="black", linewidths=1.5,
           zorder=4, label=f"Punto P (eps={eps_val})")

# Contar vecinos dentro del circulo
dists_demo = np.linalg.norm(X_demo[:40] - X_demo[center_idx], axis=1)
neighbors = np.sum(dists_demo <= eps_val)
ax.text(cx + 0.9, cy + 0.1, f"{neighbors} vecinos\ndentro de eps",
        fontsize=10, color=ARCA_RED, fontweight="bold")

ax.set_title("DBSCAN: eps define el radio, min_samples el minimo de vecinos",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
ax.legend(fontsize=10, loc="upper left")
ax.set_aspect("equal")
plt.tight_layout()
fig.savefig("fig_dbscan_concepto.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_dbscan_concepto.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 11: K-Means vs DBSCAN en Mall Customers
# ─────────────────────────────────────────────────────────────────────────────
X_mall_sc = StandardScaler().fit_transform(df[["income", "spending"]])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# K-Means
ax = axes[0]
km_mall = KMeans(n_clusters=5, random_state=42, n_init=10)
labs_km = km_mall.fit_predict(X_mall_sc)
for k in range(5):
    mask = labs_km == k
    ax.scatter(df.loc[mask, "income"], df.loc[mask, "spending"],
               c=COLORS_K[k], s=30, edgecolors="white", linewidths=0.4)
ax.set_xlabel("Ingreso (k$)"); ax.set_ylabel("Spending Score")
ax.set_title("K-Means (K=5)\nTodos los puntos asignados",
             fontweight="bold", color=BLUE, fontsize=12)

# DBSCAN
ax = axes[1]
db_mall = DBSCAN(eps=0.45, min_samples=5)
labs_db = db_mall.fit_predict(X_mall_sc)
n_clusters_db = len(set(labs_db) - {-1})
n_noise = (labs_db == -1).sum()

for k in sorted(set(labs_db)):
    mask = labs_db == k
    if k == -1:
        ax.scatter(df.loc[mask, "income"], df.loc[mask, "spending"],
                   c="gray", s=20, marker="x", alpha=0.6, label=f"Ruido ({n_noise})")
    else:
        ax.scatter(df.loc[mask, "income"], df.loc[mask, "spending"],
                   c=COLORS_K[k % len(COLORS_K)], s=30,
                   edgecolors="white", linewidths=0.4)
ax.set_xlabel("Ingreso (k$)"); ax.set_ylabel("Spending Score")
ax.set_title(f"DBSCAN ({n_clusters_db} clusters + ruido)\nDetecta outliers",
             fontweight="bold", color=GREEN, fontsize=12)
ax.legend(fontsize=9)

fig.suptitle("Mismo dataset, dos algoritmos distintos",
             fontweight="bold", color=ARCA_DARK, fontsize=14)
plt.tight_layout()
fig.savefig("fig_kmeans_vs_dbscan.png", dpi=200, bbox_inches="tight")
plt.close()
print("fig_kmeans_vs_dbscan.png")

print("\nTodas las figuras generadas!")
