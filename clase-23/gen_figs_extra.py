"""Figuras adicionales para Clase 23 — NN detallada."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"

# ─────────────────────────────────────────────────────────────────────────────
# Fig: Gradient descent visual — 2D landscape with path
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Create a simple loss surface
x = np.linspace(-3, 3, 200)
y_loss = x**2 + 0.5 * np.sin(3*x)  # loss landscape with local minimum

for ax, (lr_name, lr_val, color, steps) in zip(axes, [
    ("LR muy grande (0.5)", 0.5, ARCA_RED, []),
    ("LR bueno (0.1)", 0.1, GREEN, []),
    ("LR muy chico (0.01)", 0.01, BLUE, []),
]):
    ax.plot(x, y_loss, color=GRAY, lw=2)
    ax.fill_between(x, y_loss, alpha=0.05, color=GRAY)

    # Simulate gradient descent
    w = 2.5  # starting point
    path_w = [w]
    path_loss = [w**2 + 0.5 * np.sin(3*w)]
    for _ in range(15):
        grad = 2*w + 1.5 * np.cos(3*w)
        w = w - lr_val * grad
        w = np.clip(w, -3, 3)
        path_w.append(w)
        path_loss.append(w**2 + 0.5 * np.sin(3*w))

    # Plot path
    ax.plot(path_w, path_loss, "o-", color=color, lw=2, markersize=6, alpha=0.8)
    ax.scatter(path_w[0], path_loss[0], color=color, s=100, zorder=5,
               edgecolors="black", linewidths=1.5, label="Inicio")
    ax.scatter(path_w[-1], path_loss[-1], color=color, s=100, zorder=5,
               marker="*", edgecolors="black", linewidths=1.5, label="Final")
    ax.set_xlabel("Peso (w)", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title(lr_name, fontweight="bold", color=color, fontsize=11)
    ax.set_ylim(-1, 10)
    ax.legend(fontsize=8)

fig.suptitle("Gradient Descent: el learning rate controla el tamano del paso cuesta abajo",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("fig_gradient_descent.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_gradient_descent.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: Batches visual — full dataset split into mini-batches
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 3.5))

n_total = 1000
batch_size = 200
n_batches = n_total // batch_size

colors_batch = [BLUE, ARCA_RED, GREEN, ORANGE, PURPLE]

for i in range(n_batches):
    x_start = i * 2.2
    rect = mpatches.FancyBboxPatch(
        (x_start, 0.5), 1.8, 1.5, boxstyle="round,pad=0.1",
        facecolor=colors_batch[i], edgecolor="white", linewidth=2, alpha=0.7)
    ax.add_patch(rect)
    ax.text(x_start + 0.9, 1.25, f"Batch {i+1}\n{batch_size} datos",
            ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    if i < n_batches - 1:
        ax.annotate("", xy=(x_start + 2.1, 1.25), xytext=(x_start + 1.9, 1.25),
                    arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))

# Epoch bracket
ax.annotate("", xy=(0, 0.2), xytext=(n_batches * 2.2 - 0.4, 0.2),
            arrowprops=dict(arrowstyle="<->", color=ARCA_DARK, lw=2))
ax.text(n_batches * 1.1 - 0.2, -0.1, f"1 EPOCA = {n_batches} batches = {n_total} datos",
        ha="center", fontsize=12, fontweight="bold", color=ARCA_DARK)

ax.set_xlim(-0.3, n_batches * 2.2 + 0.3)
ax.set_ylim(-0.5, 2.5)
ax.axis("off")
ax.set_title(f"Mini-batches: dividir {n_total} datos en grupos de {batch_size}.\n"
             "Los pesos se actualizan despues de CADA batch, no al final.",
             fontweight="bold", color=ARCA_DARK, fontsize=12, pad=10)
plt.tight_layout()
fig.savefig("fig_batches.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_batches.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: Overfitting vs good fit — train loss vs validation loss
# ─────────────────────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: Overfit (too many neurons, no early stopping)
mlp_overfit = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                             random_state=42, validation_fraction=0.2)
mlp_overfit.fit(X_train, y_train)

# Simulate train vs val loss (sklearn only gives train loss)
train_losses = mlp_overfit.loss_curve_
# Fake val loss that diverges (for illustration)
np.random.seed(42)
val_losses = [l + 0.02 for l in train_losses[:30]]
for i in range(30, len(train_losses)):
    val_losses.append(val_losses[-1] + np.random.uniform(0, 0.01))

ax = axes[0]
ax.plot(train_losses[:len(val_losses)], color=BLUE, lw=2.5, label="Train loss")
ax.plot(val_losses, color=ARCA_RED, lw=2.5, label="Validation loss")
ax.axvline(30, color=GREEN, ls="--", lw=1.5, label="Aqui debia parar")
ax.set_xlabel("Epoca", fontsize=10)
ax.set_ylabel("Loss", fontsize=10)
ax.set_title("OVERFITTING\nTrain baja, validation SUBE", fontweight="bold", color=ARCA_RED, fontsize=11)
ax.legend(fontsize=9)
ax.annotate("Brecha = overfitting",
            xy=(len(val_losses)-10, val_losses[-10]), fontsize=9, color=ARCA_RED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ARCA_RED))

# Panel 2: Good fit (early stopping)
mlp_good = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200,
                          random_state=42, early_stopping=True, validation_fraction=0.2)
mlp_good.fit(X_train, y_train)

ax = axes[1]
ax.plot(mlp_good.loss_curve_, color=BLUE, lw=2.5, label="Train loss")
# Simulate val loss that follows train
val_good = [l + np.random.uniform(0.01, 0.03) for l in mlp_good.loss_curve_]
ax.plot(val_good, color=ARCA_RED, lw=2.5, label="Validation loss")
ax.set_xlabel("Epoca", fontsize=10)
ax.set_ylabel("Loss", fontsize=10)
ax.set_title("BUEN AJUSTE (early stopping)\nAmbas bajan juntas", fontweight="bold", color=GREEN, fontsize=11)
ax.legend(fontsize=9)
ax.annotate("early_stopping=True\nparo automatico",
            xy=(len(mlp_good.loss_curve_)-5, mlp_good.loss_curve_[-5]+0.05),
            fontsize=9, color=GREEN,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GREEN))

fig.suptitle("Overfitting: cuando la red memoriza en vez de aprender. Solucion: early stopping.",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
plt.tight_layout()
fig.savefig("fig_overfitting_nn.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_overfitting_nn.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: Biological vs artificial neuron
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

# Left side: biological description
ax.text(0.02, 0.95, "Neurona BIOLOGICA", fontsize=14, fontweight="bold",
        color=ARCA_DARK, transform=ax.transAxes)
ax.text(0.02, 0.82, "Dendritas reciben senales\nde otras neuronas",
        fontsize=10, color=ARCA_DARK, transform=ax.transAxes)
ax.text(0.02, 0.65, "El soma (cuerpo) integra\nlas senales recibidas",
        fontsize=10, color=ARCA_DARK, transform=ax.transAxes)
ax.text(0.02, 0.48, "Si la senal supera un\numbral, se activa (dispara)",
        fontsize=10, color=ARCA_DARK, transform=ax.transAxes)
ax.text(0.02, 0.31, "El axon transmite la senal\na las neuronas siguientes",
        fontsize=10, color=ARCA_DARK, transform=ax.transAxes)

# Arrow
ax.annotate("", xy=(0.5, 0.5), xytext=(0.42, 0.5),
            arrowprops=dict(arrowstyle="->", color=ARCA_RED, lw=3),
            transform=ax.transAxes)

# Right side: artificial
ax.text(0.55, 0.95, "Neurona ARTIFICIAL", fontsize=14, fontweight="bold",
        color=ARCA_RED, transform=ax.transAxes)
ax.text(0.55, 0.82, "Inputs (x) reciben datos\nde la capa anterior",
        fontsize=10, color=ARCA_RED, transform=ax.transAxes)
ax.text(0.55, 0.65, "Suma ponderada (z) integra\nlos inputs con pesos",
        fontsize=10, color=ARCA_RED, transform=ax.transAxes)
ax.text(0.55, 0.48, "Funcion de activacion\ndecide si 'dispara' o no",
        fontsize=10, color=ARCA_RED, transform=ax.transAxes)
ax.text(0.55, 0.31, "Output se envia a las\nneuronas de la capa siguiente",
        fontsize=10, color=ARCA_RED, transform=ax.transAxes)

# Connections
for y_pos, label in [(0.85, "Dendritas = Inputs"),
                      (0.68, "Soma = Suma ponderada"),
                      (0.51, "Umbral = Activacion"),
                      (0.34, "Axon = Output")]:
    ax.plot([0.38, 0.52], [y_pos, y_pos], color=GRAY, lw=1, ls="--",
            transform=ax.transAxes)
    ax.text(0.45, y_pos + 0.04, label, fontsize=7, color=GRAY,
            ha="center", transform=ax.transAxes, style="italic")

ax.text(0.5, 0.12, "La neurona artificial es una simplificacion matematica de la biologica.\n"
        "No es una replica exacta, pero captura la idea esencial:\n"
        "recibir, integrar, decidir, transmitir.",
        fontsize=10, color=ARCA_DARK, ha="center", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=GRAY, alpha=0.1, edgecolor=GRAY))

fig.suptitle("Por que se llama 'neurona': la inspiracion biologica",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_neurona_bio.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_neurona_bio.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: Convergence problems — local minima, saddle points
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel 1: local minimum
x = np.linspace(-3, 4, 300)
y_complex = 0.3*x**4 - 2*x**3 + 3*x**2 + 0.5*x + 2
ax = axes[0]
ax.plot(x, y_complex, color=ARCA_DARK, lw=2.5)
ax.fill_between(x, y_complex, alpha=0.05, color=ARCA_DARK)

# Mark local and global minimum
local_min_x, local_min_y = 0.5, 0.3*0.5**4 - 2*0.5**3 + 3*0.5**2 + 0.5*0.5 + 2
global_min_x, global_min_y = 3.2, 0.3*3.2**4 - 2*3.2**3 + 3*3.2**2 + 0.5*3.2 + 2
ax.scatter(local_min_x, local_min_y, color=ORANGE, s=150, zorder=5, edgecolors="black", linewidths=2)
ax.scatter(global_min_x, global_min_y, color=GREEN, s=150, zorder=5, edgecolors="black", linewidths=2)
ax.annotate("Minimo LOCAL\n(se puede quedar aqui)", xy=(local_min_x, local_min_y-0.5),
            fontsize=9, color=ORANGE, fontweight="bold", ha="center")
ax.annotate("Minimo GLOBAL\n(donde queremos llegar)", xy=(global_min_x, global_min_y-0.5),
            fontsize=9, color=GREEN, fontweight="bold", ha="center")
ax.set_title("Minimos locales\nEl optimizador puede quedarse atrapado",
             fontweight="bold", color=ARCA_DARK)
ax.set_xlabel("Pesos"); ax.set_ylabel("Loss")

# Panel 2: solutions
ax = axes[1]
ax.axis("off")
ax.text(0.05, 0.9, "Problemas de convergencia y soluciones:", fontsize=12,
        fontweight="bold", color=ARCA_DARK, transform=ax.transAxes)

problems = [
    ("Minimo local", "La red se queda en un valle\nque no es el mejor.",
     "Usar 'adam' optimizer (default\nen sklearn). Reiniciar con\ndiferentes random_state.", ORANGE),
    ("No converge", "El loss no baja. La red\nes muy pequena.",
     "Agregar neuronas o capas.", ARCA_RED),
    ("Overfitting", "Train loss baja pero\ntest loss sube.",
     "early_stopping=True.\nReducir tamano de red.", PURPLE),
]

for i, (name, desc, sol, color) in enumerate(problems):
    y = 0.7 - i * 0.28
    ax.text(0.05, y, f"{name}:", fontsize=10, fontweight="bold", color=color,
            transform=ax.transAxes)
    ax.text(0.05, y - 0.08, desc, fontsize=9, color=ARCA_DARK,
            transform=ax.transAxes)
    ax.text(0.55, y, "Solucion:", fontsize=10, fontweight="bold", color=GREEN,
            transform=ax.transAxes)
    ax.text(0.55, y - 0.08, sol, fontsize=9, color=ARCA_DARK,
            transform=ax.transAxes)

fig.suptitle("Problemas de convergencia: la red no siempre encuentra el mejor resultado",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
plt.tight_layout()
fig.savefig("fig_convergencia.png", dpi=200, bbox_inches="tight")
plt.close()
print("✓ fig_convergencia.png")

print("\n¡Todas las figuras extra generadas!")
