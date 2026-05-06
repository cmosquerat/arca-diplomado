"""Figuras v3 para Clase 24 — explicaciones pedagogicas detalladas."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ConnectionPatch

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"; LIGHT = "#F5F5F5"

# ─────────────────────────────────────────────────────────────────────────────
# Fig: Que es un parametro — concepto basico
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4),
                         gridspec_kw={"width_ratios": [1, 1, 1]})

# Panel 1: Logistic regression — pocos parametros
ax = axes[0]
ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.axis("off")
ax.text(2, 3.7, "Logistic", ha="center", fontweight="bold", fontsize=12, color=BLUE)
# 3 inputs -> 1 output
for i, y in enumerate([2.5, 1.8, 1.1]):
    ax.add_patch(Rectangle((0.3, y), 0.4, 0.4, fc=BLUE, ec="white"))
    ax.text(0.5, y+0.2, f"x{i}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax.add_patch(plt.Circle((3, 1.8), 0.35, fc=ARCA_RED))
# Show weights as labels on lines
for i, (y, w) in enumerate(zip([2.5, 1.8, 1.1], ["w0", "w1", "w2"])):
    ax.plot([0.7, 2.65], [y+0.2, 1.8], color=ARCA_RED, lw=1.0)
    ax.text(1.6, (y+0.2 + 1.8)/2 + 0.05, w, fontsize=9, color=ARCA_RED, fontweight="bold")
ax.text(2, 0.3, "3 pesos + 1 bias\n= 4 parametros",
        ha="center", fontsize=10, fontweight="bold", color=ARCA_DARK)

# Panel 2: MLP — mas parametros
ax = axes[1]
ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.axis("off")
ax.text(2, 3.7, "MLP (3-4-2)", ha="center", fontweight="bold", fontsize=12, color=ORANGE)
# 3 inputs
for i, y in enumerate([2.6, 1.9, 1.2]):
    ax.add_patch(Rectangle((0.2, y), 0.3, 0.3, fc=BLUE, ec="white"))
# 4 hidden
for i, y in enumerate([2.9, 2.1, 1.4, 0.7]):
    ax.add_patch(plt.Circle((1.8, y+0.15), 0.18, fc=ORANGE))
# 2 output
for i, y in enumerate([2.2, 1.4]):
    ax.add_patch(plt.Circle((3.2, y+0.15), 0.18, fc=ARCA_RED))
# Connections (sparse for visual clarity)
for ya in [2.6, 1.9, 1.2]:
    for yb in [2.9, 2.1, 1.4, 0.7]:
        ax.plot([0.5, 1.62], [ya+0.15, yb+0.15], color=GRAY, lw=0.4)
for ya in [2.9, 2.1, 1.4, 0.7]:
    for yb in [2.2, 1.4]:
        ax.plot([1.98, 3.02], [ya+0.15, yb+0.15], color=GRAY, lw=0.4)
ax.text(2, 0.0, "(3x4)+(4x2) + 6 bias\n= 26 parametros",
        ha="center", fontsize=10, fontweight="bold", color=ARCA_DARK)

# Panel 3: training process
ax = axes[2]
ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.axis("off")
ax.text(2, 3.7, "Entrenar = ajustar pesos",
        ha="center", fontweight="bold", fontsize=12, color=GREEN)
# Loss curve
xs = np.linspace(0.3, 3.7, 50)
ys = 2.5 * np.exp(-xs * 0.6) + 0.7
ax.plot(xs, ys, color=GREEN, lw=2.5)
ax.text(0.3, 0.4, "epoch", fontsize=9, color=ARCA_DARK)
ax.text(0.3, 3.0, "loss", fontsize=9, color=ARCA_DARK)
ax.annotate("", xy=(3.7, 0.6), xytext=(0.3, 0.6),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK))
ax.annotate("", xy=(0.3, 3.4), xytext=(0.3, 0.6),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK))
ax.text(2, 1.3, "los pesos\ncambian de\nvalor en cada\nepoch",
        ha="center", fontsize=9, color=ARCA_DARK, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=GREEN))

fig.suptitle("Un PARAMETRO es un peso que se aprende. Mas capas = mas parametros.",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("/root/Arca/clase-24/fig_parametros.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_parametros.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: convolution — sliding window animation (3 frames, big and clear)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

inp = np.array([
    [3, 1, 2, 7, 5],
    [0, 1, 1, 4, 8],
    [1, 2, 2, 0, 1],
    [3, 9, 8, 0, 0],
    [4, 8, 1, 5, 4]
])
filt = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

# Compute output beforehand
out = np.zeros((3, 3), dtype=int)
for i in range(3):
    for j in range(3):
        out[i, j] = (inp[i:i+3, j:j+3] * filt).sum()

# Draw 3 frames showing sliding
positions = [(0, 0), (0, 1), (1, 0)]  # 3 different window positions
titles = ["Paso 1: ventana arriba-izq", "Paso 2: deslizar 1 a la derecha", "Paso 3: bajar 1 fila"]
for ax_idx, ((i0, j0), title) in enumerate(zip(positions, titles)):
    ax = axes[ax_idx]
    ax.set_xlim(-1, 7); ax.set_ylim(-1, 6); ax.axis("off")
    # Input matrix
    ax.imshow(inp, cmap="Blues", vmin=0, vmax=10, extent=(-0.5, 4.5, -0.5, 4.5),
              origin="upper")
    for ii in range(5):
        for jj in range(5):
            ax.text(jj, ii, str(inp[ii, jj]), ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if inp[ii, jj] > 5 else ARCA_DARK)
    # Highlight the 3x3 window
    ax.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), 3, 3,
                           fill=False, ec=ARCA_RED, lw=4))
    # Output value at this position
    val = out[i0, j0]
    ax.text(6.0, 2.0, f"= {val}", ha="center", va="center",
            fontsize=24, fontweight="bold", color=ARCA_RED)
    ax.text(6.0, 3.0, "salida\nen esta\nposicion",
            ha="center", va="center", fontsize=9, color=ARCA_DARK)
    ax.set_title(title, fontweight="bold", color=ARCA_DARK, fontsize=11)

fig.suptitle("La ventana 3x3 se desliza. En cada parada: 9 multiplicaciones, 1 suma -> 1 numero",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("/root/Arca/clase-24/fig_conv_deslizar.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_conv_deslizar.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: padding visual — valid vs same
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Valid: output is smaller
ax = axes[0]
ax.set_xlim(-1, 8); ax.set_ylim(-1, 7); ax.axis("off")
# Input 5x5
ax.add_patch(Rectangle((0, 0), 5, 5, fc=BLUE, ec="white", alpha=0.6))
ax.text(2.5, 2.5, "INPUT\n5x5", ha="center", va="center", fontsize=11,
        fontweight="bold", color="white")
# Output 3x3
ax.add_patch(Rectangle((6, 1), 3, 3, fc=ARCA_RED, ec="white", alpha=0.85))
ax.text(7.5, 2.5, "OUTPUT\n3x3", ha="center", va="center", fontsize=10,
        fontweight="bold", color="white")
ax.annotate("", xy=(5.8, 2.5), xytext=(5.2, 2.5),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))
ax.text(2.5, 6.0, "padding=\"valid\"",
        ha="center", fontweight="bold", fontsize=14, color=ARCA_RED)
ax.text(2.5, 5.4, "(sin padding)", ha="center", fontsize=10, color=ARCA_DARK)
ax.text(2.5, -0.5, "Output mas chico que input.\nLas esquinas se 'pierden'.",
        ha="center", fontsize=9.5, color=ARCA_DARK)

# Same: output keeps size
ax = axes[1]
ax.set_xlim(-1, 8); ax.set_ylim(-1, 7); ax.axis("off")
# Padding ring
ax.add_patch(Rectangle((-0.5, -0.5), 6, 6, fc=GRAY, ec="white", alpha=0.4))
# Input 5x5
ax.add_patch(Rectangle((0, 0), 5, 5, fc=BLUE, ec="white", alpha=0.85))
ax.text(2.5, 2.5, "INPUT\n5x5", ha="center", va="center", fontsize=11,
        fontweight="bold", color="white")
# Pad labels
ax.text(-0.25, 2.5, "0", fontsize=9, color=ARCA_DARK, rotation=90)
ax.text(5.25, 2.5, "0", fontsize=9, color=ARCA_DARK, rotation=90)
ax.text(2.5, -0.25, "ceros", fontsize=8, color=ARCA_DARK, ha="center")
ax.text(2.5, 5.25, "ceros", fontsize=8, color=ARCA_DARK, ha="center")
# Output 5x5
ax.add_patch(Rectangle((6, 0), 5, 5, fc=ARCA_RED, ec="white", alpha=0.85))
ax.text(8.5, 2.5, "OUTPUT\n5x5", ha="center", va="center", fontsize=11,
        fontweight="bold", color="white")
ax.set_xlim(-1, 11)
ax.annotate("", xy=(5.8, 2.5), xytext=(5.7, 2.5),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))
ax.text(2.5, 6.0, "padding=\"same\"",
        ha="center", fontweight="bold", fontsize=14, color=GREEN)
ax.text(2.5, 5.7, "(rodeamos con ceros)", ha="center", fontsize=10, color=ARCA_DARK)
ax.text(5, -0.5, "Output del MISMO tamano que el input.\nNada se pierde en los bordes.",
        ha="center", fontsize=9.5, color=ARCA_DARK)

fig.suptitle("padding: que pasa con los bordes de la imagen",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("/root/Arca/clase-24/fig_padding.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_padding.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: why pool — without pooling parameters explode
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")

# Without pooling
ax.text(3.25, 4.7, "SIN POOLING", ha="center", fontweight="bold",
        fontsize=12, color=ARCA_RED)
# Block diagram
boxes_no_pool = [
    (0.3, 32, "32x32x3", BLUE),
    (1.4, 32, "32x32x32", ARCA_RED),
    (2.5, 32, "32x32x64", ARCA_RED),
    (3.6, 32, "32x32x128", ARCA_RED),
    (4.7, "Flatten", "131,072!", PURPLE),
]
for x, h_or_label, shape, color in boxes_no_pool:
    if isinstance(h_or_label, int):
        ax.add_patch(Rectangle((x, 2.0), 0.7, 1.4, fc=color, ec="white"))
    else:
        ax.add_patch(Rectangle((x, 2.0), 0.5, 1.4, fc=color, ec="white"))
    ax.text(x + 0.35, 1.6, shape, ha="center", va="top", fontsize=8,
            color=ARCA_DARK, fontweight="bold")
ax.text(3.25, 0.9, "Flatten produce 131K features\nDense de 64: ~8.4 MILLONES de parametros",
        ha="center", fontsize=9.5, color=ARCA_RED, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=ARCA_RED))

# Vertical separator
ax.plot([6.5, 6.5], [0.3, 4.5], color=GRAY, lw=1, linestyle="--")

# With pooling
ax.text(9.5, 4.7, "CON POOLING (2x2 cada bloque)",
        ha="center", fontweight="bold", fontsize=12, color=GREEN)
boxes_pool = [
    (7.0, 32, "32x32x3", BLUE),
    (8.0, 32, "32x32x32", ARCA_RED),
    (9.0, 16, "16x16x32", GREEN),
    (9.8, 16, "16x16x64", ARCA_RED),
    (10.6, 8, "8x8x64", GREEN),
    (11.3, 8, "8x8x128", ARCA_RED),
    (12.0, 4, "4x4x128", GREEN),
]
heights = [1.5, 1.5, 1.0, 1.0, 0.7, 0.7, 0.5]
for (x, _, shape, color), h in zip(boxes_pool, heights):
    ax.add_patch(Rectangle((x, 2.5 - h/2), 0.5, h, fc=color, ec="white"))
    ax.text(x + 0.25, 2.5 - h/2 - 0.25, shape, ha="center", va="top",
            fontsize=7, color=ARCA_DARK, fontweight="bold", rotation=0)
ax.text(9.5, 0.9, "Flatten produce 2,048 features\nDense de 64: ~131 mil parametros",
        ha="center", fontsize=9.5, color=GREEN, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=GREEN))

fig.suptitle("Por que necesitamos POOLING: sin reducir tamano, los Dense finales explotan",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.0)
plt.tight_layout()
fig.savefig("/root/Arca/clase-24/fig_por_que_pool.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_por_que_pool.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: flatten visualization
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4),
                         gridspec_kw={"width_ratios": [1, 2]})

# Left: 3D tensor (4x4x3 illustration)
ax = axes[0]
ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.axis("off")
# Three stacked layers
for k, (offset, color) in enumerate([(0.0, BLUE), (0.3, GREEN), (0.6, ARCA_RED)]):
    ax.add_patch(Rectangle((0.5 + offset, 1.5 + offset), 3, 2,
                           fc=color, ec="white", alpha=0.7))
    ax.text(0.5 + offset + 1.5, 1.5 + offset + 1.0, f"canal {k}",
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")
ax.text(2.2, 4.3, "Tensor 3D\n(4 x 4 x 3 = 48 numeros)",
        ha="center", fontweight="bold", fontsize=11, color=ARCA_DARK)

# Right: flat vector
ax = axes[1]
ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
# 48 cells in a row
for i in range(48):
    color_idx = i // 16   # which canal (4 from each = 16 cells per canal)
    color = [BLUE, GREEN, ARCA_RED][color_idx]
    ax.add_patch(Rectangle((0.2 + i*0.22, 1.5), 0.20, 0.6,
                           fc=color, ec="white", alpha=0.7))
ax.text(6, 3.0, "Vector 1D\n(48 numeros, una sola fila)",
        ha="center", fontweight="bold", fontsize=11, color=ARCA_DARK)
ax.text(0.5, 1.0, "0", fontsize=8, color=ARCA_DARK)
ax.text(11, 1.0, "47", fontsize=8, color=ARCA_DARK)

# Arrow between
fig.text(0.36, 0.5, "Flatten\n(.flatten())", fontsize=11,
         ha="center", va="center", fontweight="bold", color=ARCA_RED,
         bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=ARCA_RED))

fig.suptitle("FLATTEN: simplemente 'desenrolla' el tensor 3D en un vector 1D",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("/root/Arca/clase-24/fig_flatten.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_flatten.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: 3 stages of CNN
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")

stages = [
    (0.5, 4.5, "ETAPA 1: extraer features", BLUE,
     "Conv2D + ReLU + MaxPool\n(repetir 2-4 veces)\n\nDe pixeles -> bordes ->\ntexturas -> partes",
     "Filtros 3x3 detectan patrones.\nMaxPool reduce tamano."),
    (5.5, 4.5, "ETAPA 2: aplanar", PURPLE,
     "Flatten\n\nTensor 3D -> vector 1D",
     "Solo cambia la forma.\nSin parametros."),
    (8.5, 4.5, "ETAPA 3: clasificar", ARCA_RED,
     "Dense + ReLU\nDense + Softmax\n\nEl 'cerebro' que decide",
     "MLP clasico al final.\nUna sola decision por\ncada clase."),
]

for x, y, title, color, body, note in stages:
    # Header bar
    ax.add_patch(FancyBboxPatch((x, y - 0.6), 4.5, 0.6,
                                boxstyle="round,pad=0.05",
                                fc=color, ec="white", lw=1.5))
    ax.text(x + 2.25, y - 0.3, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    # Body box
    ax.add_patch(FancyBboxPatch((x, 0.5), 4.5, 3.2,
                                boxstyle="round,pad=0.05",
                                fc=LIGHT, ec=color, lw=1.5))
    ax.text(x + 2.25, 2.6, body, ha="center", va="center",
            fontsize=10, color=ARCA_DARK, fontweight="bold")
    ax.text(x + 2.25, 1.0, note, ha="center", va="center",
            fontsize=9, color=ARCA_DARK, style="italic")

fig.suptitle("Toda CNN tiene 3 ETAPAS bien diferenciadas",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=1.0)
plt.tight_layout()
fig.savefig("/root/Arca/clase-24/fig_3etapas.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_3etapas.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: ImageNet — what VGG was trained on
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")

# Header
ax.add_patch(FancyBboxPatch((0.5, 4.0), 12, 0.7,
                            boxstyle="round,pad=0.05",
                            fc=ARCA_DARK, ec="white", lw=1.5))
ax.text(6.5, 4.35, "ImageNet: el dataset que entreno a las CNN modernas",
        ha="center", va="center", fontsize=13, fontweight="bold", color="white")

# 4 fact boxes
facts = [
    (0.5, "1.4 millones\nde imagenes", BLUE,
     "Etiquetadas\na mano por\nhumanos"),
    (3.6, "1\\,000 categorias", ORANGE,
     "Perros, autos,\nfrutas, herramientas,\nanimales..."),
    (6.7, "Resoluci\\'on alta\n(224 x 224)", PURPLE,
     "Fotos reales\ndel mundo\n(no MNIST)"),
    (9.8, "Concurso\n2010-2017", GREEN,
     "ImageNet\nChallenge:\ndonde nacieron\nVGG, ResNet..."),
]

for x, title, color, body in facts:
    ax.add_patch(FancyBboxPatch((x, 0.6), 2.7, 3.0,
                                boxstyle="round,pad=0.05",
                                fc=color, ec="white", lw=1.5, alpha=0.92))
    ax.text(x + 1.35, 2.8, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(x + 1.35, 1.5, body, ha="center", va="center",
            fontsize=9.5, color="white")

ax.text(6.5, 0.1,
        "VGG16 'vio' 1.4M de fotos. Sus filtros aprendieron lo que es un BORDE, una TEXTURA, un OJO.",
        ha="center", fontsize=10, fontweight="bold", color=ARCA_DARK,
        style="italic")

plt.tight_layout()
fig.savefig("/root/Arca/clase-24/fig_imagenet.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_imagenet.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: VGG16 vs our CNN — comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")

categories = ["Datos de entrenamiento", "Parametros", "Capas convolucionales",
              "Tiempo entrenamiento\n(con GPU)"]

y_positions = [3.5, 2.7, 1.9, 1.1]
our_values = ["4,000 imagenes\n(cats vs dogs)", "~120 mil",
              "3 (Conv-Pool x3)", "~2 minutos"]
vgg_values = ["1.4 millones\n(ImageNet)", "138 millones",
              "13 (5 bloques)", "~3 semanas\n(en 2014)"]

ax.text(2.5, 4.5, "NUESTRA CNN\n(la que entrenamos hoy)",
        ha="center", fontweight="bold", fontsize=12, color=ARCA_RED)
ax.text(10.5, 4.5, "VGG16\n(la que reusamos)",
        ha="center", fontweight="bold", fontsize=12, color=GREEN)

# Categories
for cat, y, our, vgg in zip(categories, y_positions, our_values, vgg_values):
    # Category label (center)
    ax.text(6.5, y, cat, ha="center", va="center", fontsize=10,
            fontweight="bold", color=ARCA_DARK,
            bbox=dict(boxstyle="round,pad=0.2", fc=LIGHT, ec=GRAY))
    # Our value
    ax.text(2.5, y, our, ha="center", va="center", fontsize=9, color=ARCA_RED,
            fontweight="bold")
    # VGG value
    ax.text(10.5, y, vgg, ha="center", va="center", fontsize=9, color=GREEN,
            fontweight="bold")

ax.text(6.5, 0.4,
        "Imposible que nuestra CNN compita desde cero. Por eso REUSAMOS.",
        ha="center", fontsize=10.5, fontweight="bold", color=ARCA_DARK,
        bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec=ARCA_RED, lw=1.5))

fig.suptitle("Comparacion: nuestra CNN vs VGG16",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.0)
plt.tight_layout()
fig.savefig("/root/Arca/clase-24/fig_vgg_vs_nuestra.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_vgg_vs_nuestra.png")

print("\nFiguras v3 listas.")
