"""Figuras nuevas para Clase 25 v2 — proyecto placas + OCR."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from PIL import Image

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})

ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"; LIGHT = "#F5F5F5"

# Use the user's actual plate hero image as base
hero = np.array(Image.open("/root/Arca/clase-25/fig_hero_plate.png").convert("RGB"))
H, W = hero.shape[:2]
print(f"Hero image: {hero.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG: YOLO grid sobre la imagen del carro
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.imshow(hero)
n_cols, n_rows = 12, 7
for i in range(n_rows + 1):
    ax.axhline(y=i * H / n_rows, color="white", lw=0.8, alpha=0.65)
for j in range(n_cols + 1):
    ax.axvline(x=j * W / n_cols, color="white", lw=0.8, alpha=0.65)

# Highlight the cell where the plate center likely is
# The plate in imagen1 is around the middle-bottom of the image
cx, cy = W * 0.42, H * 0.65
col = int(cx / (W / n_cols))
row = int(cy / (H / n_rows))
ax.add_patch(Rectangle((col * W/n_cols, row * H/n_rows),
                       W/n_cols, H/n_rows,
                       fill=False, ec=GREEN, lw=4))
ax.plot(cx, cy, 'o', color=GREEN, markersize=14,
        markeredgecolor="white", markeredgewidth=2)
ax.axis("off")
ax.set_title("YOLO divide la imagen en una rejilla. La celda que contiene el centro de la placa es 'responsable' de detectarla",
             fontweight="bold", color=ARCA_DARK, fontsize=11)
plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_yolo_grid.png", dpi=160, bbox_inches="tight")
plt.close()
print("OK fig_yolo_grid.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG: NMS — antes vs después sobre la imagen real
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

# Plate location estimate (from inspection of the image)
plate_x = W * 0.32
plate_y = H * 0.55
plate_w = W * 0.20
plate_h = H * 0.10

# Left: many noisy boxes
ax = axes[0]
ax.imshow(hero); ax.axis("off")
np.random.seed(7)
for _ in range(8):
    dx = np.random.randint(-30, 30); dy = np.random.randint(-20, 20)
    dw = np.random.randint(-15, 15); dh = np.random.randint(-10, 10)
    ax.add_patch(Rectangle((plate_x+dx, plate_y+dy), plate_w+dw, plate_h+dh,
                           fill=False, ec=ARCA_RED, lw=1.5, alpha=0.55))
ax.set_title("ANTES de NMS\n(YOLO produce muchas detecciones por objeto)",
             fontweight="bold", color=ARCA_RED, fontsize=11)

# Right: 1 clean box
ax = axes[1]
ax.imshow(hero); ax.axis("off")
ax.add_patch(Rectangle((plate_x, plate_y), plate_w, plate_h,
                       fill=False, ec=GREEN, lw=3))
ax.text(plate_x, plate_y - 8, "placa  0.94", fontsize=10, color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc=GREEN, ec="none"))
ax.set_title("DESPUES de NMS\n(1 caja final, descarta las solapadas)",
             fontweight="bold", color=GREEN, fontsize=11)

fig.suptitle("NMS (Non-Maximum Suppression): se queda con la caja de mayor confianza, descarta las que solapan mucho con ella",
             fontweight="bold", color=ARCA_DARK, fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_nms.png", dpi=160, bbox_inches="tight")
plt.close()
print("OK fig_nms.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG: pipeline del proyecto plates+OCR (5 pasos)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4.0))
ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off")

steps = [
    (0.3, "1. DESCARGAR\ndataset", BLUE,
     "ZIP en el repo:\n100 fotos de carros\nsin etiquetar"),
    (3.0, "2. ETIQUETAR\nen CVAT.ai", ORANGE,
     "Dibujar bbox\nsobre cada placa\nclase: 'placa'"),
    (5.7, "3. EXPORTAR\nformato YOLO", PURPLE,
     "1 .txt por imagen\nsubir ZIP\na Drive comun"),
    (8.4, "4. FINE-TUNE\nYOLO26", ARCA_RED,
     "Carlos consolida\n+ entrena en\nColab GPU"),
    (11.1, "5. OCR + APP\nGradio", GREEN,
     "Pipeline cascada:\ndetectar -> leer\ntexto + demo"),
]

for x, title, color, body in steps:
    ax.add_patch(FancyBboxPatch((x, 1.0), 2.5, 1.8,
                                boxstyle="round,pad=0.05",
                                fc=color, ec="white", lw=1.5))
    ax.text(x + 1.25, 2.4, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(x + 1.25, 1.4, body, ha="center", va="center",
            fontsize=8.5, color="white")

# Arrows between
for i in range(4):
    x_start = 0.3 + 2.5 + i * 2.7
    x_end = 0.3 + i * 2.7 + 2.7
    ax.annotate("", xy=(x_end + 0.05, 1.9), xytext=(x_start + 0.05, 1.9),
                arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))

fig.suptitle("Pipeline del proyecto plates+OCR: del dataset crudo a la app desplegada",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=0.95)
plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_pipeline_proyecto.png", dpi=160, bbox_inches="tight")
plt.close()
print("OK fig_pipeline_proyecto.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG: cascada OCR (alternativa al imagen10.png si no quedó bien)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")

# Stage 1: input
ax.add_patch(FancyBboxPatch((0.3, 1.2), 2.6, 2.4,
                            boxstyle="round,pad=0.05",
                            fc=BLUE, ec="white", lw=1.5))
ax.text(1.6, 4.3, "Input", ha="center", fontweight="bold",
        fontsize=12, color=ARCA_DARK)
ax.text(1.6, 2.4, "Foto de carro\n(cualquier resolución)",
        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
ax.text(1.6, 0.7, "(input)", ha="center", fontsize=9, color=BLUE)

# Stage 2: detect
ax.add_patch(FancyBboxPatch((3.6, 1.2), 2.8, 2.4,
                            boxstyle="round,pad=0.05",
                            fc=ARCA_RED, ec="white", lw=1.5))
ax.text(5.0, 4.3, "1. Detectar", ha="center", fontweight="bold",
        fontsize=12, color=ARCA_DARK)
ax.text(5.0, 2.4, "YOLO encuentra\nla placa\n(bbox + score)",
        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
ax.text(5.0, 0.7, "(detección)", ha="center", fontsize=9, color=ARCA_RED)

# Stage 3: crop
ax.add_patch(FancyBboxPatch((7.1, 1.2), 2.6, 2.4,
                            boxstyle="round,pad=0.05",
                            fc=ORANGE, ec="white", lw=1.5))
ax.text(8.4, 4.3, "2. Recortar", ha="center", fontweight="bold",
        fontsize=12, color=ARCA_DARK)
ax.text(8.4, 2.4, "Cortar la región\nde la placa\nde la foto",
        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
ax.text(8.4, 0.7, "(crop)", ha="center", fontsize=9, color=ORANGE)

# Stage 4: read
ax.add_patch(FancyBboxPatch((10.4, 1.2), 3.2, 2.4,
                            boxstyle="round,pad=0.05",
                            fc=GREEN, ec="white", lw=1.5))
ax.text(12.0, 4.3, "3. Leer texto", ha="center", fontweight="bold",
        fontsize=12, color=ARCA_DARK)
ax.text(12.0, 2.4, "OCR:\n\"PCJ-3421\"",
        ha="center", va="center", fontsize=11, color="white", fontweight="bold")
ax.text(12.0, 0.7, "(reconocimiento)", ha="center", fontsize=9, color=GREEN)

# Arrows
for x_start, x_end in [(2.95, 3.55), (6.45, 7.05), (9.75, 10.35)]:
    ax.annotate("", xy=(x_end, 2.4), xytext=(x_start, 2.4),
                arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2.5))

fig.suptitle("Pipeline en CASCADA: detección + OCR son DOS modelos distintos encadenados",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_ocr_cascade_diagram.png", dpi=160, bbox_inches="tight")
plt.close()
print("OK fig_ocr_cascade_diagram.png")

print("\nFiguras técnicas v2 listas.")
