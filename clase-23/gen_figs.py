"""Figuras para Clase 23 — Redes Neuronales (MLP) — explicación detallada."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"

def plot_decision(ax, model, X, y, title, color, show_acc=True):
    """Helper: plot decision boundary."""
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 300),
        np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.22, levels=[-.5, .5, 1.5], colors=[BLUE, ARCA_RED])
    ax.contour(xx, yy, Z, levels=[0.5], colors=[color], linewidths=2)
    ax.scatter(X[y==0, 0], X[y==0, 1], c=BLUE, s=12, edgecolors="white", linewidths=0.3)
    ax.scatter(X[y==1, 0], X[y==1, 1], c=ARCA_RED, s=12, edgecolors="white", linewidths=0.3)
    if show_acc:
        acc = accuracy_score(y, model.predict(X))
        ax.set_title(f"{title}\n{acc:.0%}", fontweight="bold", color=color, fontsize=10)
    else:
        ax.set_title(title, fontweight="bold", color=color, fontsize=10)

X_moons, y_moons = make_moons(n_samples=400, noise=0.2, random_state=42)
X_circles, y_circles = make_circles(n_samples=400, noise=0.1, factor=0.4, random_state=42)

# XOR dataset
rng = np.random.default_rng(42)
X_xor = rng.standard_normal((400, 2))
y_xor = ((X_xor[:, 0] > 0) ^ (X_xor[:, 1] > 0)).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: El problema — logística vs RF (SIN MLP todavía)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

for row, (name, X, y) in enumerate([
    ("Medialunas", X_moons, y_moons),
    ("Circulos", X_circles, y_circles),
]):
    lr = LogisticRegression().fit(X, y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    plot_decision(axes[row, 0], lr, X, y, f"{name}: Logistica", ARCA_RED)
    plot_decision(axes[row, 1], rf, X, y, f"{name}: Random Forest", ORANGE)

fig.suptitle("La logistica traza lineas rectas y falla. RF usa cortes rectangulares (funciona, pero tosco).\n"
             "Pregunta: hay algo que dibuje fronteras CURVAS y SUAVES?",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
plt.tight_layout()
fig.savefig("fig_problema.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_problema.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: 1 neurona = 1 línea recta (limitación)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, (name, X, y) in zip(axes, [
    ("Medialunas", X_moons, y_moons),
    ("Circulos", X_circles, y_circles),
    ("XOR", X_xor, y_xor),
]):
    neuron = MLPClassifier(hidden_layer_sizes=(1,), activation="logistic",
                           max_iter=500, random_state=42)
    neuron.fit(X, y)
    plot_decision(ax, neuron, X, y, f"1 neurona en {name}", ARCA_RED)

fig.suptitle("UNA sola neurona = una linea recta. No importa el dataset — siempre es lineal.",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
plt.tight_layout()
fig.savefig("fig_1neurona.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_1neurona.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Funciones de activación detalladas
# ─────────────────────────────────────────────────────────────────────────────
x = np.linspace(-5, 5, 300)
sigmoid = 1 / (1 + np.exp(-x))
relu = np.maximum(0, x)
tanh_v = np.tanh(x)
linear = x
softmax_raw = np.exp(x) / (np.exp(x) + np.exp(-x) + np.exp(0))

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Sigmoid
ax = axes[0, 0]
ax.plot(x, sigmoid, color=BLUE, lw=3)
ax.axhline(0, color=GRAY, lw=0.5); ax.axvline(0, color=GRAY, lw=0.5)
ax.axhline(0.5, color=ARCA_RED, lw=1, ls="--", alpha=0.5)
ax.set_title("Sigmoid", fontweight="bold", color=ARCA_DARK)
ax.set_ylabel("Output")
ax.annotate("Rango: [0, 1]\nUsar: capa de SALIDA\n(clasificacion binaria)",
            xy=(2, 0.3), fontsize=9, color=ARCA_DARK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRAY))

# ReLU
ax = axes[0, 1]
ax.plot(x, relu, color=GREEN, lw=3)
ax.axhline(0, color=GRAY, lw=0.5); ax.axvline(0, color=GRAY, lw=0.5)
ax.set_title("ReLU (Rectified Linear Unit)", fontweight="bold", color=ARCA_DARK)
ax.annotate("Rango: [0, inf)\nUsar: capas OCULTAS\n(la mas usada hoy)",
            xy=(1, 3), fontsize=9, color=ARCA_DARK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRAY))

# Tanh
ax = axes[1, 0]
ax.plot(x, tanh_v, color=PURPLE, lw=3)
ax.axhline(0, color=GRAY, lw=0.5); ax.axvline(0, color=GRAY, lw=0.5)
ax.set_title("Tanh", fontweight="bold", color=ARCA_DARK)
ax.set_ylabel("Output"); ax.set_xlabel("Input (z)")
ax.annotate("Rango: [-1, 1]\nUsar: capas ocultas\n(centrada en cero)",
            xy=(1.5, -0.5), fontsize=9, color=ARCA_DARK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRAY))

# Lineal (identidad)
ax = axes[1, 1]
ax.plot(x, linear, color=GRAY, lw=3)
ax.axhline(0, color=GRAY, lw=0.5); ax.axvline(0, color=GRAY, lw=0.5)
ax.set_title("Lineal (sin activacion)", fontweight="bold", color=ARCA_DARK)
ax.set_xlabel("Input (z)")
ax.annotate("Rango: (-inf, inf)\nNO USAR en capas ocultas\n(anula el efecto de las capas)",
            xy=(-4, 2), fontsize=9, color=ARCA_RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ARCA_RED))

fig.suptitle("Funciones de activacion: cada una tiene su uso especifico",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig("fig_activaciones.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_activaciones.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Activación importa — con vs sin (identity vs relu vs sigmoid)
# ─────────────────────────────────────────────────────────────────────────────
X, y = X_moons, y_moons
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 300),
    np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 300))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
activations = [
    ("Sin activacion (identity)\n= siempre lineal", "identity", ARCA_RED),
    ("Sigmoid\n= curva suave", "logistic", BLUE),
    ("ReLU\n= curva con esquinas", "relu", GREEN),
]
for ax, (name, act, color) in zip(axes, activations):
    m = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42, activation=act)
    m.fit(X, y)
    plot_decision(ax, m, X, y, name, color)

fig.suptitle("Misma red (2 capas, 16+8 neuronas). La activacion cambia TODO.",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
plt.tight_layout()
fig.savefig("fig_activacion_importa.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_activacion_importa.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Transición de 1 neurona a red — frontera evoluciona
# ─────────────────────────────────────────────────────────────────────────────
architectures = [
    ("1 neurona", (1,)),
    ("4 neuronas\n(1 capa)", (4,)),
    ("16 neuronas\n(1 capa)", (16,)),
    ("2 capas\n(16, 8)", (16, 8)),
    ("3 capas\n(32, 16, 8)", (32, 16, 8)),
]

fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
for ax, (name, layers) in zip(axes, architectures):
    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=500, random_state=42, activation="relu")
    mlp.fit(X, y)
    plot_decision(ax, mlp, X, y, name, GREEN)

fig.suptitle("De 1 neurona (linea recta) a una red (curva compleja): cada paso agrega capacidad",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
plt.tight_layout()
fig.savefig("fig_capas.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_capas.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Loss curves — cómo interpretar
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: loss normal (todo bien)
mlp_ok = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
mlp_ok.fit(X, y)
ax = axes[0]
ax.plot(mlp_ok.loss_curve_, color=GREEN, lw=2.5)
ax.set_title("Baja suavemente\n= TODO BIEN", fontweight="bold", color=GREEN)
ax.set_xlabel("Epoca"); ax.set_ylabel("Loss")

# Panel 2: LR muy alto (oscila)
mlp_high = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42,
                         learning_rate_init=0.5)
mlp_high.fit(X, y)
ax = axes[1]
ax.plot(mlp_high.loss_curve_, color=ARCA_RED, lw=2.5)
ax.set_title("Oscila sin bajar\n= LEARNING RATE MUY ALTO", fontweight="bold", color=ARCA_RED)
ax.set_xlabel("Epoca"); ax.set_ylabel("Loss")

# Panel 3: LR muy bajo (lento)
mlp_low = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42,
                        learning_rate_init=0.00005)
mlp_low.fit(X, y)
ax = axes[2]
ax.plot(mlp_low.loss_curve_, color=BLUE, lw=2.5)
ax.set_title("Baja pero MUY lento\n= LEARNING RATE MUY BAJO", fontweight="bold", color=BLUE)
ax.set_xlabel("Epoca"); ax.set_ylabel("Loss")

fig.suptitle("Como interpretar la curva de loss y que hacer en cada caso",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_loss_interpretar.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_loss_interpretar.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Funciones de costo — visual
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Binary cross-entropy
y_true = 1
p = np.linspace(0.01, 0.99, 200)
bce = -y_true * np.log(p)
ax = axes[0]
ax.plot(p, bce, color=ARCA_RED, lw=3)
ax.set_xlabel("Probabilidad predicha (p)", fontsize=10)
ax.set_ylabel("Loss", fontsize=10)
ax.set_title("Binary Cross-Entropy\n(clasificacion binaria)", fontweight="bold", color=ARCA_DARK)
ax.annotate("Si real=1 y predices p=0.01\n→ loss MUY alto (penaliza mucho)",
            xy=(0.05, 4), fontsize=9, color=ARCA_DARK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRAY))
ax.annotate("Si real=1 y predices p=0.99\n→ loss bajo (bien hecho)",
            xy=(0.55, 0.5), fontsize=9, color=GREEN,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GREEN))

# MSE
y_pred_range = np.linspace(-2, 4, 200)
y_real = 1.0
mse = (y_pred_range - y_real) ** 2
ax = axes[1]
ax.plot(y_pred_range, mse, color=BLUE, lw=3)
ax.axvline(y_real, color=GREEN, ls="--", lw=1.5, label=f"Valor real = {y_real}")
ax.set_xlabel("Prediccion", fontsize=10)
ax.set_ylabel("Loss (MSE)", fontsize=10)
ax.set_title("Mean Squared Error\n(regresion)", fontweight="bold", color=ARCA_DARK)
ax.legend(fontsize=9)

fig.suptitle("Funciones de costo: miden QUE TAN MAL predijo el modelo",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_costos.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_costos.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: MLP en los 3 datasets sintéticos — resultado final
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, Xd, yd) in zip(axes, [
    ("Medialunas", X_moons, y_moons),
    ("Circulos", X_circles, y_circles),
    ("XOR", X_xor, y_xor),
]):
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    mlp.fit(Xd, yd)
    plot_decision(ax, mlp, Xd, yd, f"MLP en {name}", GREEN)

fig.suptitle("MLP resuelve los 3 problemas no lineales que la logistica no podia",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
plt.tight_layout()
fig.savefig("fig_mlp_resuelve.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_mlp_resuelve.png")

print("\n¡Todas las figuras generadas!")
