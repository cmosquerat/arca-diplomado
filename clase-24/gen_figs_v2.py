"""Figuras adicionales para Clase 24 v2 — andropedagogia mejorada."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from scipy.ndimage import convolve
from sklearn.datasets import load_sample_images

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})

ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"; LIGHT = "#F5F5F5"

# ─────────────────────────────────────────────────────────────────────────────
# Fig: convolution origin — sliding stamp intuition
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                         gridspec_kw={"width_ratios": [1.4, 1]})

# Left: 1D signal convolution illustration
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(-0.5, 3.5); ax.axis("off")

# Signal (input)
xs = np.arange(0, 10, 0.05)
sig = 1.5 * np.exp(-((xs - 4) ** 2) / 1.5) + 0.5 * np.sin(xs * 1.5)
ax.plot(xs, sig + 1.7, color=BLUE, lw=2)
ax.text(0.2, 3.0, "SENAL (input)", fontsize=10, fontweight="bold", color=BLUE)

# Kernel sliding  (small bell curve)
kx = np.arange(-1, 1, 0.05)
kernel_curve = np.exp(-(kx ** 2) / 0.15)
positions = [2.0, 4.0, 6.0]
colors_k = [ORANGE, ARCA_RED, PURPLE]
for pos, color in zip(positions, colors_k):
    ax.plot(kx + pos, kernel_curve * 0.5 + 0.2, color=color, lw=1.8, alpha=0.85)
    ax.fill_between(kx + pos, 0.2, kernel_curve * 0.5 + 0.2, color=color, alpha=0.15)
ax.text(0.2, 0.85, "KERNEL\n(se desliza)", fontsize=10, fontweight="bold", color=ARCA_RED)

# Arrow showing sliding
ax.annotate("", xy=(7, 0.5), xytext=(1.5, 0.5),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))
ax.text(4.0, 0.0, "deslizar", fontsize=10, color=ARCA_DARK,
        fontweight="bold", ha="center")

# Output convolution
out = np.convolve(sig, kernel_curve, mode="same")
out = out / out.max() * 1.0
ax.plot(xs, out - 0.3, color=GREEN, lw=2)
# Don't actually plot below 0 (out of frame), just label
ax.text(0.2, -0.25, "RESULTADO (cada punto = patron presente alli?)",
        fontsize=9, fontweight="bold", color=GREEN)

ax.set_title("Convolucion 1D: deslizar un patron por una senal",
             fontweight="bold", color=ARCA_DARK, fontsize=12)

# Right: 2D — same idea applied as a "stamp"
ax = axes[1]
ax.set_xlim(0, 7); ax.set_ylim(0, 5); ax.axis("off")

# Input image (random texture)
np.random.seed(3)
img2d = np.random.rand(6, 6) * 0.5
img2d[2:4, 2:4] = 1.0   # bright spot
ax.imshow(img2d, extent=(0.5, 4.0, 0.5, 4.0), cmap="gray", alpha=0.9)
ax.add_patch(Rectangle((0.5, 0.5), 3.5, 3.5, fill=False, ec=BLUE, lw=2))
ax.text(2.25, 4.2, "Imagen (input)", ha="center", fontsize=10,
        fontweight="bold", color=BLUE)

# Show stamp at 3 positions
for (x0, y0), color in zip([(1.0, 3.0), (2.5, 2.5), (1.5, 1.0)], colors_k):
    ax.add_patch(Rectangle((x0, y0), 0.7, 0.7, fill=False, ec=color, lw=2.5,
                           linestyle="--"))
ax.text(2.5, 0.05, "El kernel 'se estampa' en cada posicion",
        ha="center", fontsize=9, fontweight="bold", color=ARCA_DARK)

# Arrow to output
ax.annotate("", xy=(5.5, 2.25), xytext=(4.2, 2.25),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))

# Output
out2d = convolve(img2d, np.ones((2, 2))/4)
ax.imshow(out2d, extent=(5.5, 6.8, 1.6, 2.9), cmap="viridis")
ax.text(6.15, 3.05, "Mapa\nrespuesta", ha="center", fontsize=9,
        fontweight="bold", color=GREEN)

ax.set_title("Igual en imagenes:\nrespuesta en cada posicion",
             fontweight="bold", color=ARCA_DARK, fontsize=11)

fig.suptitle("La palabra CONVOLUCION viene de signal processing: deslizar y combinar",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_conv_origen.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_conv_origen.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: stamp/match intuition — pattern matching response
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 4.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.3, 1])

# Pattern (small filter)
ax_p = fig.add_subplot(gs[0])
pattern = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]])
ax_p.imshow(pattern, cmap="gray_r", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax_p.text(j, i, str(pattern[i, j]), ha="center", va="center",
                  fontsize=14, fontweight="bold",
                  color="white" if pattern[i, j] == 1 else "black")
ax_p.set_title("FILTRO\n(patron de '+')", fontweight="bold", color=ARCA_RED, fontsize=12)
ax_p.set_xticks([]); ax_p.set_yticks([])

# Image with embedded pattern
ax_i = fig.add_subplot(gs[1])
np.random.seed(7)
img = (np.random.rand(8, 8) * 0.3).astype(float)
# Embed two crosses
for cy, cx in [(1, 1), (5, 5)]:
    img[cy, cx-1:cx+2] = 1.0
    img[cy-1:cy+2, cx] = 1.0
ax_i.imshow(img, cmap="gray_r")
ax_i.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, ec=GREEN, lw=2))
ax_i.add_patch(Rectangle((3.5, 3.5), 3, 3, fill=False, ec=GREEN, lw=2))
ax_i.add_patch(Rectangle((4.5, 0.5), 3, 3, fill=False, ec=ARCA_RED, lw=2,
                         linestyle="--"))
ax_i.set_title("IMAGEN\n(2 cruces, otras cosas)", fontweight="bold",
               color=BLUE, fontsize=12)
ax_i.set_xticks([]); ax_i.set_yticks([])

# Arrow column
ax_arr = fig.add_subplot(gs[2])
ax_arr.axis("off")
ax_arr.annotate("convolucion", xy=(0.5, 0.5), ha="center", va="center",
                fontsize=10, color=ARCA_DARK, fontweight="bold",
                xytext=(0.5, 0.5))
ax_arr.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5),
                arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2),
                xycoords=ax_arr.transAxes)

# Response map
ax_o = fig.add_subplot(gs[3])
response = convolve(img, pattern)
ax_o.imshow(response, cmap="hot")
# Mark high-response positions
peaks = np.unravel_index(np.argsort(response.ravel())[-2:], response.shape)
for py, px in zip(*peaks):
    ax_o.add_patch(Circle((px, py), 0.5, fill=False, ec=GREEN, lw=2.5))
ax_o.set_title("MAPA DE RESPUESTA\n(brillante = match)", fontweight="bold",
               color=GREEN, fontsize=12)
ax_o.set_xticks([]); ax_o.set_yticks([])

fig.suptitle("Filtro = patron buscado.  Salida brilla donde el patron aparece.",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_conv_intuicion.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_conv_intuicion.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: CNN design criteria — visual decision tree
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")

def box(x, y, w, h, text, fc, tc="white", fontsize=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.05",
                                fc=fc, ec="white", lw=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=tc)

# Q1
box(4.5, 5.0, 4, 0.7,
    "?Que tamano tiene tu imagen?", ARCA_DARK, fontsize=11)

# Branches
ax.annotate("", xy=(2, 4.3), xytext=(5.5, 4.95),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.3))
ax.text(3.5, 4.6, "32x32", fontsize=9, color=BLUE, fontweight="bold")
ax.annotate("", xy=(7, 4.3), xytext=(6.5, 4.95),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.3))
ax.text(6.4, 4.6, "96x96", fontsize=9, color=ORANGE, fontweight="bold")
ax.annotate("", xy=(11, 4.3), xytext=(7.5, 4.95),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.3))
ax.text(9.5, 4.6, "224x224", fontsize=9, color=ARCA_RED, fontweight="bold")

# Recipes
box(0.5, 3.5, 3.0, 0.8,
    "3 bloques\nConv-Pool\n(32-64-128)", BLUE, fontsize=9)
box(5.0, 3.5, 3.0, 0.8,
    "4 bloques\nConv-Pool\n(32-64-128-256)", ORANGE, fontsize=9)
box(9.5, 3.5, 3.0, 0.8,
    "5 bloques\nConv-Pool\no transfer learning", ARCA_RED, fontsize=9)

# Common header for next section
box(2.5, 2.4, 8, 0.6, "Reglas comunes para todas las CNN", ARCA_DARK,
    fontsize=11)

# Common rules
rules = [
    (0.3, 1.4, "kernel = 3x3\n(siempre)", GREEN),
    (3.0, 1.4, "padding =\n'same'", GREEN),
    (5.6, 1.4, "Doblar filtros\ncada bloque", GREEN),
    (8.2, 1.4, "Dropout 0.2-0.5\nantes del Dense final", GREEN),
    (10.8, 1.4, "Adam + LR=1e-3\n(default casi siempre)", GREEN),
]
for x, y, t, c in rules:
    box(x, y, 2.2, 0.85, t, c, fontsize=9)

# Bottom note
ax.text(6.5, 0.4,
        "Empezar SIMPLE. Si no funciona, doblar filtros o agregar un bloque.",
        ha="center", fontsize=10, color=ARCA_DARK, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec=ARCA_RED, lw=1))

fig.suptitle("Como elegir la arquitectura de tu CNN",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=0.98)
plt.tight_layout()
fig.savefig("fig_cnn_diseno.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_cnn_diseno.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: GPU vs CPU operation count
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

# Left: ops per image
ax = axes[0]
layers = ["1 Conv2D\n(32 filtros, 3x3)\n+ activacion",
          "5 capas\n(CNN simple)",
          "50 capas\n(CNN seria)"]
ops_per_image = [3e7, 5e8, 1e10]
bars = ax.bar(layers, ops_per_image, color=[BLUE, ORANGE, ARCA_RED])
ax.set_yscale("log")
ax.set_ylabel("Operaciones por imagen (log)")
ax.set_title("Una sola imagen requiere MILLONES de multiplicaciones",
             fontweight="bold", color=ARCA_DARK, fontsize=11)
for b, v in zip(bars, ops_per_image):
    ax.text(b.get_x() + b.get_width()/2, v * 1.5, f"{v:,.0e}",
            ha="center", va="bottom", fontweight="bold", fontsize=10)

# Right: time for 1 epoch with 50K images
ax = axes[1]
hardware = ["CPU\n(laptop)", "GPU T4\n(Colab gratis)", "GPU A100\n(servidor)"]
times_minutes = [25, 0.8, 0.15]
bars = ax.bar(hardware, times_minutes, color=[GRAY, GREEN, PURPLE])
ax.set_yscale("log")
ax.set_ylabel("Minutos por epoch (log)")
ax.set_title("Tiempo para 1 epoch en MNIST CNN (60K imagenes)",
             fontweight="bold", color=ARCA_DARK, fontsize=11)
for b, v in zip(bars, times_minutes):
    if v >= 1:
        s = f"{v:.0f} min"
    else:
        s = f"{v*60:.0f} seg"
    ax.text(b.get_x() + b.get_width()/2, v * 1.4, s,
            ha="center", va="bottom", fontweight="bold", fontsize=11)

fig.suptitle("Por que las CNN necesitan GPU: 30x-100x mas rapida que CPU",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_gpu_vs_cpu.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_gpu_vs_cpu.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: VGG16 architecture — pure Conv + MaxPool + Dense
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off")

# Just blocks: Conv x2, Pool, Conv x2, Pool, Conv x3, Pool, Conv x3, Pool, Conv x3, Pool, Flatten, Dense, Dense, Dense
blocks = [
    (0.2, "224x224x3\nINPUT", "#374151", 1.5),
    (1.4, "Conv2D 64\nx2", ARCA_RED, 1.3),
    (2.6, "MaxPool", GREEN, 0.7),
    (3.5, "Conv2D 128\nx2", ARCA_RED, 1.1),
    (4.6, "MaxPool", GREEN, 0.6),
    (5.4, "Conv2D 256\nx3", ARCA_RED, 1.0),
    (6.4, "MaxPool", GREEN, 0.5),
    (7.0, "Conv2D 512\nx3", ARCA_RED, 0.9),
    (7.9, "MaxPool", GREEN, 0.4),
    (8.4, "Conv2D 512\nx3", ARCA_RED, 0.7),
    (9.2, "MaxPool", GREEN, 0.3),
    (9.6, "Flatten", PURPLE, 1.2),
    (10.9, "Dense\n4096", ORANGE, 0.6),
    (11.6, "Dense\n4096", ORANGE, 0.6),
    (12.3, "Dense\n1000\n(softmax)", ARCA_DARK, 0.6),
]

prev_x = None
for x, label, color, h in blocks:
    width = 0.9 if "Conv" in label else 0.5 if "Pool" in label else 0.6
    ax.add_patch(Rectangle((x, 2 - h/2), width, h, fc=color, ec="white", lw=0.7))
    ax.text(x + width/2, 2 - h/2 - 0.3, label, ha="center", va="top",
            fontsize=7, color=ARCA_DARK, fontweight="bold")

ax.text(7, 3.5, "VGG16: solo Conv2D + MaxPool + Dense (lo que ya conocemos)",
        ha="center", fontsize=12, fontweight="bold", color=ARCA_DARK)
ax.text(7, 0.5,
        "13 capas Conv2D 3x3 + 5 MaxPool + 3 Dense.  138M parametros (vs 100K de la nuestra).",
        ha="center", fontsize=10, color=ARCA_DARK)
ax.text(7, 0.05,
        "Entrenada por Oxford en ImageNet (1.4M imagenes, 1000 clases) en 2014.",
        ha="center", fontsize=9.5, color="#6B7280", style="italic")

plt.tight_layout()
fig.savefig("fig_vgg16.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_vgg16.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: Project pipeline — collect → train → deploy
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off")

steps = [
    (0.3, 2.0, "1. ELEGIR\ntu tarea", BLUE,
     "Ej: Coca vs Pepsi\nFruta madura/verde\nTu mascota\n2-3 clases"),
    (3.0, 2.0, "2. RECOLECTAR\ndataset", ORANGE,
     "Webcam con\nGradio en Colab\n30-60 fotos\npor clase"),
    (5.7, 2.0, "3. PREPROCESAR\ny augmentar", PURPLE,
     "Resize 96x96\nNormalizar /255\nFlip + rotacion\n+ brillo"),
    (8.4, 2.0, "4. ENTRENAR\nCNN", ARCA_RED,
     "Transfer learning\ncon VGG16\n5-10 epochs\nGPU activada"),
    (11.1, 2.0, "5. DESPLEGAR\ncon Gradio", GREEN,
     "App web\npredice fotos\nnuevas en\nvivo"),
]

for x, y, title, color, body in steps:
    ax.add_patch(FancyBboxPatch((x, y - 1.0), 2.5, 1.8,
                                boxstyle="round,pad=0.05",
                                fc=color, ec="white", lw=1.5))
    ax.text(x + 1.25, y + 0.5, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(x + 1.25, y - 0.4, body, ha="center", va="center",
            fontsize=8.5, color="white")

# Arrows between
for i in range(4):
    x_start = 0.3 + 2.5 + i * 2.7
    x_end = 0.3 + i * 2.7 + 2.7
    ax.annotate("", xy=(x_end + 0.05, 2.0), xytext=(x_start + 0.05, 2.0),
                arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))

fig.suptitle("Proyecto Final: tu propio clasificador de imagenes en 5 pasos",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=0.95)
plt.tight_layout()
fig.savefig("fig_proyecto_pipeline.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_proyecto_pipeline.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig: Transfer learning — same network, new task
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Left: trained on task A
ax = axes[0]
ax.set_xlim(0, 8); ax.set_ylim(0, 5); ax.axis("off")
ax.text(4, 4.6, "1. ENTRENAMOS en Cats vs Dogs",
        ha="center", fontweight="bold", color=BLUE, fontsize=12)
# Conv layers
for i, (x, color, h) in enumerate([(0.5, ARCA_RED, 1.5), (1.7, ARCA_RED, 1.3),
                                     (2.9, ARCA_RED, 1.1)]):
    ax.add_patch(Rectangle((x, 2.5 - h/2), 0.8, h, fc=color, ec="white", lw=0.7))
    ax.text(x + 0.4, 2.5 - h/2 - 0.3, f"Conv\n+Pool", ha="center", va="top",
            fontsize=8, fontweight="bold", color=ARCA_DARK)
# Dense head
ax.add_patch(Rectangle((4.5, 1.8), 0.5, 1.4, fc=ORANGE, ec="white"))
ax.text(4.75, 1.5, "Dense", ha="center", fontsize=8, fontweight="bold")
ax.add_patch(Rectangle((5.5, 2.2), 0.4, 0.6, fc=ARCA_DARK, ec="white"))
ax.text(5.7, 1.9, "Cat\nDog", ha="center", fontsize=7, fontweight="bold",
        color=ARCA_DARK)

# Right: same backbone, new head
ax = axes[1]
ax.set_xlim(0, 8); ax.set_ylim(0, 5); ax.axis("off")
ax.text(4, 4.6, "2. REUSAMOS el backbone para Horse vs Deer",
        ha="center", fontweight="bold", color=GREEN, fontsize=12)
# Same conv layers — but greyed out (frozen)
for i, (x, h) in enumerate([(0.5, 1.5), (1.7, 1.3), (2.9, 1.1)]):
    ax.add_patch(Rectangle((x, 2.5 - h/2), 0.8, h, fc=GRAY, ec="white", lw=0.7))
    ax.text(x + 0.4, 2.5 - h/2 - 0.3, "Conv\n+Pool\nCONGELADO", ha="center", va="top",
            fontsize=7, fontweight="bold", color=ARCA_DARK)
# New head
ax.add_patch(Rectangle((4.5, 1.8), 0.5, 1.4, fc=GREEN, ec="white"))
ax.text(4.75, 1.5, "Dense\nNUEVO", ha="center", fontsize=7,
        fontweight="bold", color=GREEN)
ax.add_patch(Rectangle((5.5, 2.2), 0.4, 0.6, fc=GREEN, ec="white"))
ax.text(5.7, 1.9, "Horse\nDeer", ha="center", fontsize=7, fontweight="bold",
        color=ARCA_DARK)

# Arrow between panels
fig.text(0.51, 0.5, "→", fontsize=30, ha="center", va="center", color=ARCA_RED)

fig.suptitle("Transfer learning intuitivo: la MISMA red, NUEVA cabeza",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_transfer_misma_red.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_transfer_misma_red.png")

print("\nTodas las figuras nuevas generadas.")
