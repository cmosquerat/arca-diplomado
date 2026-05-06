"""Figuras para Clase 24 — CNN: De MLP a Convoluciones."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from scipy.ndimage import convolve, shift
from sklearn.datasets import load_digits, load_sample_images

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})

ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"; LIGHT = "#F5F5F5"

# Load data once
digits = load_digits()
samples = load_sample_images()
NPZ = "../clase-22/cifar_cats_dogs_4k.npz"
if os.path.exists(NPZ):
    cd = np.load(NPZ)
    X_cd, y_cd = cd["X"], cd["y"]
else:
    X_cd, y_cd = None, None

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: TF Playground — illustration of the interface
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(0, 13); ax.set_ylim(0, 5)
ax.axis("off")

# Left panel: input data (circles dataset)
np.random.seed(42)
n_pts = 80
theta = np.random.uniform(0, 2*np.pi, n_pts)
r1 = np.random.uniform(0.2, 0.5, n_pts//2)
r2 = np.random.uniform(0.7, 1.0, n_pts//2)
xs1 = 1.5 + r1 * np.cos(theta[:n_pts//2])
ys1 = 1.5 + r1 * np.sin(theta[:n_pts//2])
xs2 = 1.5 + r2 * np.cos(theta[n_pts//2:])
ys2 = 1.5 + r2 * np.sin(theta[n_pts//2:])
ax.scatter(xs1, ys1, c=ORANGE, s=15, alpha=0.8)
ax.scatter(xs2, ys2, c=BLUE, s=15, alpha=0.8)
ax.add_patch(Rectangle((0.3, 0.3), 2.4, 2.4, fill=False, ec=ARCA_DARK, lw=1.5))
ax.text(1.5, 3.0, "DATA\n(forma circular)", ha="center", fontsize=10,
        fontweight="bold", color=ARCA_DARK)

# Middle: features (4 small thumbnails)
for j, (label, mark) in enumerate(zip([r"$x_1$", r"$x_2$", r"$x_1^2$", r"$x_2^2$"], range(4))):
    ax.add_patch(Rectangle((3.5, 2.5 - j*0.65), 0.5, 0.5,
                           fill=True, fc=LIGHT, ec=ARCA_DARK, lw=0.8))
    ax.text(3.75, 2.75 - j*0.65, label, ha="center", va="center",
            fontsize=9, color=ARCA_DARK)
ax.text(3.75, 3.4, "FEATURES", ha="center", fontsize=10,
        fontweight="bold", color=ARCA_DARK)

# Network: 3 hidden layers
layers_x = [5.5, 7.5, 9.5]
layer_n  = [4, 4, 2]
positions = []
for x, n in zip(layers_x, layer_n):
    ys = np.linspace(0.7, 4.3, n)
    pos = [(x, y) for y in ys]
    positions.append(pos)
    for (xi, yi) in pos:
        circ = plt.Circle((xi, yi), 0.22, color=BLUE, alpha=0.85)
        ax.add_patch(circ)

# Connections (sparse, just for illustration)
prev_features = [(4.0, 2.75 - j*0.65) for j in range(4)]
for (xa, ya) in prev_features:
    for (xb, yb) in positions[0]:
        ax.plot([xa, xb], [ya, yb], color=GRAY, lw=0.4, alpha=0.6)
for a in range(2):
    for (xa, ya) in positions[a]:
        for (xb, yb) in positions[a+1]:
            ax.plot([xa, xb], [ya, yb], color=GRAY, lw=0.4, alpha=0.6)

ax.text(7.5, 4.6, "RED NEURONAL  (capas + activacion)",
        ha="center", fontsize=10, fontweight="bold", color=ARCA_DARK)

# Output panel
ax.add_patch(Rectangle((10.7, 0.3), 2.0, 2.4, fill=False, ec=ARCA_DARK, lw=1.5))
# Draw decision boundary sketch (concentric)
theta2 = np.linspace(0, 2*np.pi, 60)
ax.fill(11.7 + 0.85*np.cos(theta2), 1.5 + 0.85*np.sin(theta2),
        color=BLUE, alpha=0.15)
ax.fill(11.7 + 0.45*np.cos(theta2), 1.5 + 0.45*np.sin(theta2),
        color=ORANGE, alpha=0.25)
ax.scatter(xs1*0.6 + 11.7 - 1.5*0.6, ys1*0.6 + 1.5 - 1.5*0.6, c=ORANGE, s=8)
ax.scatter(xs2*0.6 + 11.7 - 1.5*0.6, ys2*0.6 + 1.5 - 1.5*0.6, c=BLUE, s=8)
ax.text(11.7, 3.0, "SALIDA\n(frontera aprendida)", ha="center", fontsize=10,
        fontweight="bold", color=ARCA_DARK)

# Loss line (top right)
ax.text(6.5, 0.2, "LOSS:  Train  0.041   Test  0.057",
        ha="center", fontsize=9, color=ARCA_DARK,
        bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=ARCA_DARK, lw=0.5))

fig.suptitle("TensorFlow Playground: red neuronal interactiva en el navegador",
             fontweight="bold", color=ARCA_DARK, fontsize=14)
plt.tight_layout()
fig.savefig("fig_playground.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_playground.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: MLP no es invariante a traslación — predict same digit shifted
# (Without training, simulate prediction degradation visually)
# ─────────────────────────────────────────────────────────────────────────────
img3 = digits.images[np.where(digits.target == 3)[0][0]]    # 8x8 '3'
fig, axes = plt.subplots(2, 5, figsize=(13, 5),
                         gridspec_kw={"height_ratios": [1, 1]})

shifts = [0, 1, 2, 3, 4]
# Top row: shifted images
for i, s in enumerate(shifts):
    ax = axes[0, i]
    shifted = shift(img3, [0, s], mode="constant", cval=0)
    ax.imshow(shifted, cmap="gray_r")
    ax.set_title(f"'3' desplazado +{s}px", fontsize=10, fontweight="bold")
    ax.axis("off")

# Bottom row: representation as flat vectors (visualize how DIFFERENT the vector is)
for i, s in enumerate(shifts):
    ax = axes[1, i]
    shifted = shift(img3, [0, s], mode="constant", cval=0)
    flat = shifted.flatten()
    ax.imshow(flat.reshape(1, -1), cmap="gray_r", aspect="auto")
    diff = np.linalg.norm(flat - img3.flatten())
    color = GREEN if diff == 0 else (ORANGE if diff < 30 else ARCA_RED)
    ax.set_title(f"vector aplanado | dist={diff:.0f}",
                 fontsize=9, color=color, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("MLP ve un vector totalmente distinto cuando movemos el dígito 1 pixel",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.0)
plt.tight_layout()
fig.savefig("fig_mlp_traslacion.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_mlp_traslacion.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Parameter explosion — MLP first layer for different image sizes
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
sizes  = [(28, 28, 1, "MNIST\n28x28 grises"),
          (32, 32, 3, "CIFAR\n32x32 RGB"),
          (96, 96, 3, "Foto pequena\n96x96 RGB"),
          (224, 224, 3, "Foto web\n224x224 RGB")]
hidden_neurons = 128
labels = []; params = []
for h, w, c, name in sizes:
    n = h * w * c * hidden_neurons + hidden_neurons   # weights + bias
    labels.append(name)
    params.append(n)
colors = [BLUE, ORANGE, ARCA_RED, ARCA_DARK]
bars = ax.bar(labels, params, color=colors)
ax.set_yscale("log")
ax.set_ylabel("Parametros en la 1ra capa (log)", fontsize=11)
ax.set_title(f"MLP -> 1 capa oculta de {hidden_neurons} neuronas: parametros explotan con la resolucion",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
for bar, n in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width()/2, n*1.2, f"{n:,}",
            ha="center", va="bottom", fontweight="bold", fontsize=10)
plt.tight_layout()
fig.savefig("fig_mlp_parametros.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_mlp_parametros.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Edge detector — handcrafted Sobel filter on real image
# ─────────────────────────────────────────────────────────────────────────────
img_color = samples.images[1]    # flower (427, 640, 3)
img_gray = img_color.mean(axis=2)
# Crop a reasonable square
H, W = 280, 280
cy, cx = img_gray.shape[0]//2, img_gray.shape[1]//2
img_gray = img_gray[cy-H//2:cy+H//2, cx-W//2:cx+W//2]
img_rgb_crop = img_color[cy-H//2:cy+H//2, cx-W//2:cx+W//2]

# Sobel filters (horizontal and vertical edges)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = sobel_x.T
edges_x = convolve(img_gray, sobel_x)
edges_y = convolve(img_gray, sobel_y)
edges_mag = np.sqrt(edges_x**2 + edges_y**2)

# Blur filter
blur = np.ones((5, 5)) / 25
img_blur = convolve(img_gray, blur)

fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
axes[0].imshow(img_rgb_crop)
axes[0].set_title("Imagen original (RGB)", fontweight="bold", fontsize=11)
axes[0].axis("off")

axes[1].imshow(img_gray, cmap="gray")
axes[1].set_title("Grises", fontweight="bold", fontsize=11)
axes[1].axis("off")

axes[2].imshow(edges_mag, cmap="gray")
axes[2].set_title("Filtro Sobel\n(detecta bordes)", fontweight="bold",
                  fontsize=11, color=ARCA_RED)
axes[2].axis("off")

axes[3].imshow(img_blur, cmap="gray")
axes[3].set_title("Filtro promedio\n(desenfoque)", fontweight="bold",
                  fontsize=11, color=BLUE)
axes[3].axis("off")

fig.suptitle("Un FILTRO 3x3 transforma la imagen. Esto es la operacion central de una CNN.",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_filtro_edge.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_filtro_edge.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Convolution operation step by step
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                         gridspec_kw={"width_ratios": [1.2, 0.6, 1.2]})

# Input image (5x5) with values
np.random.seed(1)
in_img = np.array([
    [3, 1, 2, 7, 5],
    [0, 1, 1, 4, 8],
    [1, 2, 2, 0, 1],
    [3, 9, 8, 0, 0],
    [4, 8, 1, 5, 4]
])
axes[0].imshow(in_img, cmap="Blues", vmin=0, vmax=10)
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, str(in_img[i, j]), ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if in_img[i, j] > 5 else ARCA_DARK)
# Highlight a 3x3 window
axes[0].add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, ec=ARCA_RED, lw=3))
axes[0].set_title(f"INPUT: {in_img.shape}\n(ventana 3x3 en rojo)",
                  fontweight="bold", color=ARCA_DARK, fontsize=11)
axes[0].set_xticks([]); axes[0].set_yticks([])

# Kernel
kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])    # vertical edge
axes[1].imshow(kernel, cmap="RdBu", vmin=-1, vmax=1)
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, str(kernel[i, j]), ha="center", va="center",
                     fontsize=14, fontweight="bold", color=ARCA_DARK)
axes[1].set_title("FILTRO 3x3\n(borde vertical)",
                  fontweight="bold", color=ARCA_RED, fontsize=11)
axes[1].set_xticks([]); axes[1].set_yticks([])

# Output (3x3): full 2D convolution result
out = np.zeros((3, 3), dtype=int)
for i in range(3):
    for j in range(3):
        out[i, j] = (in_img[i:i+3, j:j+3] * kernel).sum()
axes[2].imshow(out, cmap="RdBu_r", vmin=-15, vmax=15)
for i in range(3):
    for j in range(3):
        axes[2].text(j, i, str(out[i, j]), ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if abs(out[i,j]) > 8 else ARCA_DARK)
axes[2].add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, ec=ARCA_RED, lw=3))
axes[2].set_title(f"OUTPUT: {out.shape}\n(esquina = 1ra ventana)",
                  fontweight="bold", color=ARCA_DARK, fontsize=11)
axes[2].set_xticks([]); axes[2].set_yticks([])

# Annotation
fig.text(0.5, 0.02,
         "Multiplicar elemento a elemento + sumar = 1 numero. Mover ventana, repetir.",
         ha="center", fontsize=11, color=ARCA_DARK, fontweight="bold")
fig.suptitle("Operacion CONVOLUCION: ventana 3x3 recorre la imagen",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig("fig_conv_paso.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_conv_paso.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Max Pooling 2x2
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                         gridspec_kw={"width_ratios": [1.3, 1]})

inp = np.array([
    [1, 3, 2, 4],
    [5, 6, 8, 1],
    [7, 2, 0, 3],
    [4, 9, 5, 2]
])
axes[0].imshow(inp, cmap="Blues", vmin=0, vmax=10)
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, str(inp[i, j]), ha="center", va="center",
                     fontsize=14, fontweight="bold",
                     color="white" if inp[i, j] > 5 else ARCA_DARK)
# Color the 4 quadrants with a thin border
for (yi, xi), color in zip([(0,0), (0,2), (2,0), (2,2)],
                           [ARCA_RED, BLUE, GREEN, ORANGE]):
    axes[0].add_patch(Rectangle((xi-0.5, yi-0.5), 2, 2, fill=False,
                                ec=color, lw=3))
axes[0].set_title(f"INPUT: {inp.shape}\n(4 ventanas de 2x2)",
                  fontweight="bold", color=ARCA_DARK, fontsize=11)
axes[0].set_xticks([]); axes[0].set_yticks([])

# Output: max in each 2x2 block
out2 = np.array([[6, 8], [9, 5]])
axes[1].imshow(out2, cmap="Blues", vmin=0, vmax=10)
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(out2[i, j]), ha="center", va="center",
                     fontsize=18, fontweight="bold", color="white")
border_colors = [[ARCA_RED, BLUE], [GREEN, ORANGE]]
for i in range(2):
    for j in range(2):
        axes[1].add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                    ec=border_colors[i][j], lw=3))
axes[1].set_title(f"OUTPUT: {out2.shape}\n(maximo de cada bloque)",
                  fontweight="bold", color=ARCA_DARK, fontsize=11)
axes[1].set_xticks([]); axes[1].set_yticks([])

fig.suptitle("MaxPooling 2x2: reduce a la mitad cada dimension, conserva lo mas activo",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_pooling.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_pooling.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Full CNN architecture
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5)
ax.axis("off")

stages = [
    (0.2, "Input\n32x32x3", "32x32\nx3", BLUE,    1.4, 1.4, 0.3),
    (1.8, "Conv2D 32\n3x3 + ReLU", "32x32\nx32", ARCA_RED, 1.4, 1.4, 0.6),
    (3.4, "MaxPool 2x2", "16x16\nx32",          GREEN,   1.0, 1.0, 0.6),
    (4.7, "Conv2D 64\n3x3 + ReLU", "16x16\nx64", ARCA_RED, 1.0, 1.0, 0.9),
    (6.0, "MaxPool 2x2", "8x8\nx64",            GREEN,   0.7, 0.7, 0.9),
    (7.0, "Flatten", "4096",                      PURPLE,  0.3, 1.3, 0.0),
    (8.0, "Dense 64\n+ ReLU", "64",              ORANGE,  0.2, 0.8, 0.0),
    (8.9, "Dense 2\n+ Softmax", "2",             ARCA_DARK, 0.2, 0.4, 0.0),
]
prev_x = None
for x, label, shape, color, h, w, depth in stages:
    if prev_x is not None:
        ax.annotate("", xy=(x, 2.5), xytext=(prev_x + 0.5, 2.5),
                    arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.2))
    if depth > 0:
        # Draw a stack of squares
        for k in range(3):
            offset = k * 0.10
            rect = Rectangle((x + offset, 2.5 - h/2 + offset),
                             w, h, fc=color, ec="white", lw=0.5, alpha=0.75)
            ax.add_patch(rect)
    else:
        rect = Rectangle((x, 2.5 - h/2), w, h, fc=color, ec=ARCA_DARK, lw=0.7)
        ax.add_patch(rect)
    ax.text(x + w/2, 2.5 - h/2 - 0.4, shape, ha="center", va="top",
            fontsize=8, color=ARCA_DARK, fontweight="bold")
    ax.text(x + w/2, 2.5 + h/2 + 0.4, label, ha="center", va="bottom",
            fontsize=9, color=color, fontweight="bold")
    prev_x = x

ax.text(7.5, 4.8,
        "CONV/POOL extraen FEATURES (bordes -> texturas -> partes)   |   DENSE clasifica",
        ha="center", fontsize=11, fontweight="bold", color=ARCA_DARK)
fig.suptitle("Arquitectura tipica de CNN: bloques convolucionales -> aplanar -> clasificar",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.0)
plt.tight_layout()
fig.savefig("fig_cnn_arquitectura.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_cnn_arquitectura.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Multiple filters → multiple feature maps
# ─────────────────────────────────────────────────────────────────────────────
img_demo = img_gray.copy()

filters = {
    "Borde vertical":   np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
    "Borde horizontal": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    "Esquina /":        np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]),
    "Detalle fino":     np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
}

fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
axes[0].imshow(img_demo, cmap="gray")
axes[0].set_title("Input", fontweight="bold", fontsize=11)
axes[0].axis("off")

colors_feat = [ARCA_RED, BLUE, GREEN, PURPLE]
for ax, (name, kernel), color in zip(axes[1:], filters.items(), colors_feat):
    out = convolve(img_demo, kernel)
    out = np.abs(out)
    ax.imshow(out, cmap="gray")
    ax.set_title(name, fontweight="bold", color=color, fontsize=11)
    ax.axis("off")

fig.suptitle("Cada filtro produce un MAPA DE CARACTERISTICAS distinto",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_feature_maps.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_feature_maps.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: Parameter sharing (MLP vs CNN comparison)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: MLP — every input pixel has its OWN weight per neuron
ax = axes[0]
ax.set_xlim(0, 6); ax.set_ylim(0, 5); ax.axis("off")
# 5 input pixels
for i in range(5):
    ax.add_patch(Rectangle((0.5, 4-i*0.7), 0.4, 0.4, fc=BLUE, ec="white"))
    ax.text(0.7, 4.2 - i*0.7, f"p{i}", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")
# Hidden neuron
ax.add_patch(plt.Circle((4, 2.4), 0.4, fc=ORANGE, alpha=0.85))
ax.text(4, 2.4, "h", ha="center", va="center", fontsize=12,
        fontweight="bold", color="white")
# Connections — each unique
weights = [r"$w_0$", r"$w_1$", r"$w_2$", r"$w_3$", r"$w_4$"]
for i in range(5):
    ax.plot([0.9, 3.6], [4.2 - i*0.7, 2.4], color=ARCA_RED, lw=1.2)
    mid_x = 0.9 + (3.6 - 0.9) * 0.5
    mid_y = (4.2 - i*0.7) * 0.5 + 2.4 * 0.5
    ax.text(mid_x, mid_y + 0.05, weights[i], fontsize=10, color=ARCA_RED,
            fontweight="bold")
ax.set_title("MLP: cada pixel tiene un peso PROPIO\n(N pixeles -> N pesos por neurona)",
             fontweight="bold", color=ARCA_DARK, fontsize=12)

# Right: CNN — same 3 weights are reused across windows
ax = axes[1]
ax.set_xlim(0, 6); ax.set_ylim(0, 5); ax.axis("off")
for i in range(5):
    ax.add_patch(Rectangle((0.5, 4-i*0.7), 0.4, 0.4, fc=BLUE, ec="white"))
    ax.text(0.7, 4.2 - i*0.7, f"p{i}", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")
# Filter window slides — show 3 hidden neurons but ALL share same 3 weights
for k, (yh, label) in enumerate(zip([3.2, 2.0, 0.8], ["h0", "h1", "h2"])):
    ax.add_patch(plt.Circle((4, yh), 0.32, fc=GREEN, alpha=0.85))
    ax.text(4, yh, label, ha="center", va="center", fontsize=10,
            fontweight="bold", color="white")
# 3 colors for the 3 SHARED weights
weight_colors = [ARCA_RED, ORANGE, PURPLE]
weight_names  = [r"$w_a$", r"$w_b$", r"$w_c$"]
# Connections: h0 sees p0,p1,p2; h1 sees p1,p2,p3; h2 sees p2,p3,p4
connections = [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
hidden_y    = [3.2, 2.0, 0.8]
for hy, conn in zip(hidden_y, connections):
    for k, p in enumerate(conn):
        ax.plot([0.9, 3.7], [4.2 - p*0.7, hy],
                color=weight_colors[k], lw=1.0, alpha=0.85)

# Legend for shared weights
for k in range(3):
    ax.plot([5.0, 5.4], [3.5 - k*0.4, 3.5 - k*0.4],
            color=weight_colors[k], lw=2.5)
    ax.text(5.5, 3.5 - k*0.4, weight_names[k], fontsize=10,
            color=weight_colors[k], fontweight="bold")
ax.text(5.2, 4.0, "los MISMOS\n3 pesos", ha="center", fontsize=9,
        color=ARCA_DARK, fontweight="bold")

ax.set_title("CNN: 3 pesos COMPARTIDOS para todas las posiciones\n(parameter sharing -> mucho mas eficiente)",
             fontweight="bold", color=ARCA_DARK, fontsize=12)

fig.suptitle("La idea clave: compartir los pesos del filtro a lo largo de la imagen",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_parameter_sharing.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_parameter_sharing.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 10: Keras ecosystem stack diagram
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

layers = [
    (0.3, "TU CODIGO (Python)\nmodel = Sequential([Conv2D, MaxPool, Dense])",
     ARCA_DARK, "white"),
    (1.4, "KERAS\nAPI de alto nivel: capas, compile, fit, evaluate",
     ARCA_RED, "white"),
    (2.5, "BACKEND: TensorFlow / JAX / PyTorch\n(quien hace los gradientes y la GPU)",
     BLUE, "white"),
    (3.6, "GPU / CPU / TPU\n(quien ejecuta las matrices)",
     "#1F2937", "white"),
]
for y, label, fc, tc in layers:
    box = FancyBboxPatch((0.5, y), 11, 0.95, boxstyle="round,pad=0.04",
                         fc=fc, ec="white", lw=2)
    ax.add_patch(box)
    ax.text(6, y + 0.475, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color=tc)
# Arrows
for y in [1.3, 2.4, 3.5]:
    ax.annotate("", xy=(6, y - 0.1), xytext=(6, y + 0.1),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=2))

fig.suptitle("La pila: tu codigo -> Keras -> TensorFlow -> GPU",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=0.98)
plt.tight_layout()
fig.savefig("fig_keras_stack.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_keras_stack.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 11: Data augmentation samples (using cats)
# ─────────────────────────────────────────────────────────────────────────────
if X_cd is not None:
    cat_idx = np.where(y_cd == 0)[0][7]
    cat = X_cd[cat_idx]
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    axes[0].imshow(cat); axes[0].set_title("Original", fontweight="bold"); axes[0].axis("off")
    # Flip
    axes[1].imshow(cat[:, ::-1, :]); axes[1].set_title("Flip horizontal", fontweight="bold"); axes[1].axis("off")
    # Rotate (using scipy)
    from scipy.ndimage import rotate as sci_rotate
    axes[2].imshow(np.clip(sci_rotate(cat, 12, reshape=False), 0, 255).astype(np.uint8))
    axes[2].set_title("Rotacion 12°", fontweight="bold"); axes[2].axis("off")
    # Shift
    axes[3].imshow(np.clip(shift(cat, [2, -3, 0], mode="nearest"), 0, 255).astype(np.uint8))
    axes[3].set_title("Desplazamiento", fontweight="bold"); axes[3].axis("off")
    # Brightness
    axes[4].imshow(np.clip(cat.astype(int) + 35, 0, 255).astype(np.uint8))
    axes[4].set_title("+ Brillo", fontweight="bold"); axes[4].axis("off")
    # Zoom (crop + resize)
    crop = cat[4:28, 4:28]
    from PIL import Image
    zoomed = np.array(Image.fromarray(crop).resize((32, 32)))
    axes[5].imshow(zoomed); axes[5].set_title("Zoom in", fontweight="bold"); axes[5].axis("off")
    fig.suptitle("Data augmentation: el mismo gato visto de muchas formas (mas datos gratis)",
                 fontweight="bold", color=ARCA_DARK, fontsize=12, y=1.05)
    plt.tight_layout()
    fig.savefig("fig_data_aug.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("OK fig_data_aug.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 12: Results comparison (typical numbers)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# MNIST: classic
ax = axes[0]
models = ["Logistic", "Random\nForest", "MLP\n(128, 64)", "CNN\nsimple"]
mnist_acc = [0.92, 0.95, 0.97, 0.99]
bars = ax.bar(models, mnist_acc, color=[BLUE, GREEN, ORANGE, ARCA_RED])
for b, v in zip(bars, mnist_acc):
    ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.0%}",
            ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.set_ylim(0.85, 1.02)
ax.set_title("MNIST (28x28 grises)", fontweight="bold", color=ARCA_DARK, fontsize=12)
ax.set_ylabel("Accuracy")

# Cats vs Dogs RGB
ax = axes[1]
cd_acc = [0.55, 0.62, 0.60, 0.85]
bars = ax.bar(models, cd_acc, color=[BLUE, GREEN, ORANGE, ARCA_RED])
for b, v in zip(bars, cd_acc):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.0%}",
            ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.axhline(0.5, ls="--", color="gray", lw=1)
ax.text(3.4, 0.51, "azar", color="gray", fontsize=9)
ax.set_ylim(0.4, 1.0)
ax.set_title("Cats vs Dogs (32x32 RGB)", fontweight="bold", color=ARCA_DARK, fontsize=12)
ax.set_ylabel("Accuracy")

fig.suptitle("CNN cierra el gap: en imagenes, las convoluciones cambian todo",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("fig_resultados_cnn.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_resultados_cnn.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 13: Transfer learning concept
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")

# Pretrained model block
pretrained = FancyBboxPatch((0.5, 1), 6.5, 3,
                            boxstyle="round,pad=0.05", fc=BLUE, ec="white",
                            alpha=0.85, lw=2)
ax.add_patch(pretrained)
ax.text(3.75, 4.2, "MODELO PREENTRENADO (ej. ResNet50, MobileNet)",
        ha="center", fontsize=11, fontweight="bold", color=ARCA_DARK)
ax.text(3.75, 3.4, "Entrenado por Google con\n1.4M de imagenes (ImageNet)",
        ha="center", fontsize=10, color="white")
# Lock icon
ax.text(3.75, 2.3, r"$\bigodot$ pesos congelados",
        ha="center", fontsize=11, color="white", fontweight="bold")
ax.text(3.75, 1.7, "Sus filtros ya saben detectar\nbordes, texturas, formas, ojos, narices",
        ha="center", fontsize=9, color="white")

# Arrow
ax.annotate("", xy=(8.0, 2.5), xytext=(7.2, 2.5),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))

# Your head
your_head = FancyBboxPatch((8.5, 1.5), 4.0, 2,
                           boxstyle="round,pad=0.05", fc=ARCA_RED, ec="white",
                           alpha=0.95, lw=2)
ax.add_patch(your_head)
ax.text(10.5, 4.2, "TU CABEZAL (Dense)",
        ha="center", fontsize=11, fontweight="bold", color=ARCA_DARK)
ax.text(10.5, 2.8, "Solo entrenas las\nultimas 1-2 capas",
        ha="center", fontsize=10, color="white", fontweight="bold")
ax.text(10.5, 2.0, "Con 500 fotos basta",
        ha="center", fontsize=10, color="white")

fig.suptitle("Transfer Learning: usar lo que otros ya entrenaron",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=1.0)
plt.tight_layout()
fig.savefig("fig_transfer_learning.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_transfer_learning.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 14: When to use CNN — decision flow chart
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")

# Q1
q1 = FancyBboxPatch((4.5, 4), 4, 0.7, boxstyle="round,pad=0.05",
                    fc=ARCA_DARK, ec="white", lw=1.5)
ax.add_patch(q1)
ax.text(6.5, 4.35, "¿Tus datos son IMAGENES o cuadricula 2D/3D?",
        ha="center", va="center", fontsize=11, fontweight="bold", color="white")

# Branches
ax.annotate("", xy=(2.5, 3.5), xytext=(5.5, 3.95),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.5))
ax.text(3.5, 3.7, "NO", fontsize=10, color=ARCA_RED, fontweight="bold")
ax.annotate("", xy=(10.5, 3.5), xytext=(7.5, 3.95),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.5))
ax.text(9.5, 3.7, "SI", fontsize=10, color=GREEN, fontweight="bold")

# Left branch (no): no CNN
no_cnn = FancyBboxPatch((0.5, 2.6), 4, 0.7, boxstyle="round,pad=0.05",
                        fc=GRAY, ec="white", lw=1.2)
ax.add_patch(no_cnn)
ax.text(2.5, 2.95, "Tabular -> RF/XGBoost\nTexto -> Transformers",
        ha="center", va="center", fontsize=9, color=ARCA_DARK, fontweight="bold")

# Right Q2
q2 = FancyBboxPatch((8.5, 2.6), 4, 0.7, boxstyle="round,pad=0.05",
                    fc=ARCA_DARK, ec="white", lw=1.5)
ax.add_patch(q2)
ax.text(10.5, 2.95, "¿Tienes muchos datos\n(>10K imagenes etiquetadas)?",
        ha="center", va="center", fontsize=10, fontweight="bold", color="white")

# Branch from Q2
ax.annotate("", xy=(7.0, 2.0), xytext=(9.5, 2.55),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.5))
ax.text(8.0, 2.2, "NO", fontsize=10, color=ORANGE, fontweight="bold")
ax.annotate("", xy=(12.0, 2.0), xytext=(11.5, 2.55),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.5))
ax.text(11.7, 2.2, "SI", fontsize=10, color=GREEN, fontweight="bold")

# Use transfer learning
xfer = FancyBboxPatch((4.5, 1.0), 5, 0.9, boxstyle="round,pad=0.05",
                      fc=ORANGE, ec="white", lw=1.5)
ax.add_patch(xfer)
ax.text(7, 1.45, "TRANSFER LEARNING\n(usa CNN preentrenada + tu cabezal)",
        ha="center", va="center", fontsize=10, fontweight="bold", color="white")

# Train CNN from scratch
scratch = FancyBboxPatch((10.5, 1.0), 2.3, 0.9, boxstyle="round,pad=0.05",
                         fc=GREEN, ec="white", lw=1.5)
ax.add_patch(scratch)
ax.text(11.65, 1.45, "CNN desde\ncero",
        ha="center", va="center", fontsize=10, fontweight="bold", color="white")

# Bottom note
ax.text(6.5, 0.3,
        "Para imagenes pequenas / pocos datos: SIEMPRE empezar con transfer learning",
        ha="center", fontsize=10, color=ARCA_DARK, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=ARCA_RED, lw=1))

fig.suptitle("Cuando usar (y NO usar) una CNN",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=1.0)
plt.tight_layout()
fig.savefig("fig_cuando_cnn.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_cuando_cnn.png")

print("\nTodas las figuras de clase-24 generadas.")
