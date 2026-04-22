"""Figuras para Clase 19 — Gradio + K-Means (Mall Customers)."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"
COLORS_K = [BLUE, ARCA_RED, GREEN, ORANGE, PURPLE, "#06B6D4", "#D946EF"]

# ─────────────────────────────────────────────────────────────────────────────
# Cargar Mall Customers
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("mall_customers.csv")
df = df.rename(columns={
    "Annual Income (k$)": "income",
    "Spending Score (1-100)": "spending",
    "Gender": "gender",
    "Age": "age",
    "CustomerID": "id"
})

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Supervisado vs No Supervisado
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
X_blobs, y_blobs = make_blobs(n_samples=200, centers=3, cluster_std=1.2, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
for k in range(3):
    mask = y_blobs == k
    ax.scatter(X_blobs[mask, 0], X_blobs[mask, 1], c=COLORS_K[k], s=30,
               edgecolors="white", linewidths=0.4, label=f"Clase {k}")
ax.set_title("Supervisado\n(tenemos las etiquetas)", fontweight="bold", color=ARCA_DARK)
ax.legend(fontsize=8)

ax = axes[1]
ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=GRAY, s=30,
           edgecolors="white", linewidths=0.4)
ax.set_title("No supervisado\n(solo datos, sin etiquetas)", fontweight="bold", color=ARCA_DARK)

fig.suptitle("La diferencia: con etiquetas vs sin etiquetas",
             fontweight="bold", color=ARCA_DARK, fontsize=14)
plt.tight_layout()
fig.savefig("fig_sup_vs_nosup.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_sup_vs_nosup.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: K-Means paso a paso (4 paneles) — con Mall Customers
# ─────────────────────────────────────────────────────────────────────────────
X_mall = df[["income", "spending"]].values
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_mall)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))

# Panel 1: datos sin color
ax = axes[0]
ax.scatter(df["income"], df["spending"], c=GRAY, s=20, edgecolors="white", linewidths=0.4)
ax.set_xlabel("Ingreso (k$)"); ax.set_ylabel("Gasto (1-100)")
ax.set_title("Paso 1:\n200 clientes", fontweight="bold", color=ARCA_DARK, fontsize=10)

# Panel 2: centroides aleatorios
init_idx = rng.choice(len(X_mall), 5, replace=False)
centroids_init = X_mall[init_idx]
ax = axes[1]
ax.scatter(df["income"], df["spending"], c=GRAY, s=20, edgecolors="white", linewidths=0.4)
for i, c in enumerate(centroids_init):
    ax.scatter(c[0], c[1], c=COLORS_K[i], s=200, marker="X",
               edgecolors="black", linewidths=1.5, zorder=5)
ax.set_xlabel("Ingreso (k$)"); ax.set_ylabel("Gasto (1-100)")
ax.set_title("Paso 2:\n5 centroides aleatorios", fontweight="bold", color=ARCA_DARK, fontsize=10)

# Panel 3: primera asignación
dists = np.array([np.linalg.norm(X_mall - c, axis=1) for c in centroids_init])
labels_init = dists.argmin(axis=0)
ax = axes[2]
for k in range(5):
    mask = labels_init == k
    ax.scatter(df.loc[mask, "income"], df.loc[mask, "spending"],
               c=COLORS_K[k], s=20, edgecolors="white", linewidths=0.4)
for i, c in enumerate(centroids_init):
    ax.scatter(c[0], c[1], c=COLORS_K[i], s=200, marker="X",
               edgecolors="black", linewidths=1.5, zorder=5)
ax.set_xlabel("Ingreso (k$)"); ax.set_ylabel("Gasto (1-100)")
ax.set_title("Paso 3:\nAsignar al mas cercano", fontweight="bold", color=ARCA_DARK, fontsize=10)

# Panel 4: resultado final
km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
km5.fit(X_mall)
ax = axes[3]
for k in range(5):
    mask = km5.labels_ == k
    ax.scatter(df.loc[mask, "income"], df.loc[mask, "spending"],
               c=COLORS_K[k], s=20, edgecolors="white", linewidths=0.4)
for i, c in enumerate(km5.cluster_centers_):
    ax.scatter(c[0], c[1], c=COLORS_K[i], s=200, marker="X",
               edgecolors="black", linewidths=1.5, zorder=5)
ax.set_xlabel("Ingreso (k$)"); ax.set_ylabel("Gasto (1-100)")
ax.set_title("Resultado final:\n5 segmentos", fontweight="bold", color=GREEN, fontsize=10)

fig.suptitle("K-Means paso a paso: asignar → recalcular centroides → repetir",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_kmeans_pasos.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_kmeans_pasos.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Método del codo
# ─────────────────────────────────────────────────────────────────────────────
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_mall)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(list(K_range), inertias, "o-", color=ARCA_RED, lw=2.5, markersize=8)
ax.axvline(x=5, color=GREEN, linestyle="--", lw=1.5, label="Codo (K=5)")
ax.set_xlabel("K (numero de clusters)")
ax.set_ylabel("Inercia")
ax.set_title("Metodo del Codo: Mall Customers",
             fontweight="bold", color=ARCA_DARK)
ax.set_xticks(list(K_range))
ax.legend()
plt.tight_layout()
fig.savefig("fig_codo.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_codo.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: K=3 vs K=5 vs K=7
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, k in zip(axes, [3, 5, 7]):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_mall)
    for c_idx in range(k):
        mask = km.labels_ == c_idx
        ax.scatter(df.loc[mask, "income"], df.loc[mask, "spending"],
                   c=COLORS_K[c_idx % len(COLORS_K)], s=20,
                   edgecolors="white", linewidths=0.4)
    for i, c in enumerate(km.cluster_centers_):
        ax.scatter(c[0], c[1], c=COLORS_K[i % len(COLORS_K)], s=120, marker="X",
                   edgecolors="black", linewidths=1.5, zorder=5)
    ax.set_xlabel("Ingreso (k$)"); ax.set_ylabel("Gasto (1-100)")
    ax.set_title(f"K = {k}", fontweight="bold", color=ARCA_DARK, fontsize=12)

fig.suptitle("Efecto de K: pocos clusters mezclan, muchos fragmentan",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_k_comparacion.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_k_comparacion.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Los 5 segmentos con nombres de negocio
# ─────────────────────────────────────────────────────────────────────────────
km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = km5.fit_predict(X_mall)

# Perfilar para dar nombres
perfil = df.groupby("cluster")[["income", "spending", "age"]].mean().round(0)
# Asignar nombres según perfil
nombres = {}
for c in range(5):
    inc = perfil.loc[c, "income"]
    sp = perfil.loc[c, "spending"]
    if inc > 70 and sp > 60:
        nombres[c] = "Alto ingreso,\nalto gasto"
    elif inc > 70 and sp < 40:
        nombres[c] = "Alto ingreso,\nbajo gasto"
    elif inc < 40 and sp > 60:
        nombres[c] = "Bajo ingreso,\nalto gasto"
    elif inc < 40 and sp < 40:
        nombres[c] = "Bajo ingreso,\nbajo gasto"
    else:
        nombres[c] = "Ingreso medio,\ngasto medio"

fig, ax = plt.subplots(figsize=(9, 6))
for k in range(5):
    mask = df["cluster"] == k
    ax.scatter(df.loc[mask, "income"], df.loc[mask, "spending"],
               c=COLORS_K[k], s=40, edgecolors="white", linewidths=0.5,
               label=nombres[k], alpha=0.8)
    # Anotar centroide
    cx, cy = km5.cluster_centers_[k]
    ax.scatter(cx, cy, c=COLORS_K[k], s=250, marker="X",
               edgecolors="black", linewidths=2, zorder=5)

ax.set_xlabel("Ingreso anual (k$)", fontsize=12)
ax.set_ylabel("Spending Score (1-100)", fontsize=12)
ax.set_title("5 Segmentos de Clientes — Mall Customers",
             fontweight="bold", color=ARCA_DARK, fontsize=14)
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
fig.savefig("fig_segmentos_mall.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_segmentos_mall.png")

print("\n¡Todas las figuras generadas!")
