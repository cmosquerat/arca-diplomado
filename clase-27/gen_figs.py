"""Figuras nuevas para Clase 27 — YOLO folder, multi-class, segmentation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"; LIGHT = "#F5F5F5"

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: YOLO folder structure
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 11); ax.set_ylim(0, 8); ax.axis("off")

# Tree as boxes with connectors
def folder_box(x, y, label, color=BLUE, w=2.4, h=0.5, fontsize=10, italic=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                fc=color, ec="white", lw=1))
    style = "italic" if italic else "normal"
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", style=style)

# Root
folder_box(4.0, 7.0, "dataset/", ARCA_DARK, w=3.0)

# Top-level branches
branches = [
    (0.5, 5.5, "images/"),
    (3.5, 5.5, "labels/"),
    (6.5, 5.5, "data.yaml"),
]
for x, y, name in branches:
    folder_box(x, y, name, ORANGE if "yaml" in name else BLUE)

# images/ children
folder_box(0.0, 4.0, "train/", GREEN, w=1.7)
folder_box(0.0, 3.0, "val/",   GREEN, w=1.7)
folder_box(0.0, 2.0, "test/",  GREEN, w=1.7, italic=True)

# labels/ children
folder_box(3.0, 4.0, "train/", GREEN, w=1.7)
folder_box(3.0, 3.0, "val/",   GREEN, w=1.7)
folder_box(3.0, 2.0, "test/",  GREEN, w=1.7, italic=True)

# Sample files inside train/
ax.text(0.85, 1.3, ".jpg / .png", ha="center", fontsize=8, color=ARCA_DARK, style="italic")
ax.text(3.85, 1.3, ".txt (1 por imagen)", ha="center", fontsize=8, color=ARCA_DARK, style="italic")

# Connectors
def line(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=GRAY, lw=1.2)

line(5.5, 7.0, 1.7, 6.0)
line(5.5, 7.0, 4.7, 6.0)
line(5.5, 7.0, 7.7, 6.0)
line(1.7, 5.5, 0.85, 4.5)
line(1.7, 5.5, 0.85, 3.5)
line(1.7, 5.5, 0.85, 2.5)
line(4.7, 5.5, 3.85, 4.5)
line(4.7, 5.5, 3.85, 3.5)
line(4.7, 5.5, 3.85, 2.5)

# data.yaml content preview
yaml_lines = [
    "path: /content/dataset",
    "train: images/train",
    "val:   images/val",
    "",
    "names:",
    "  0: coca",
    "  1: pepsi",
    "  2: sprite",
]
yaml_text = "\n".join(yaml_lines)
ax.add_patch(FancyBboxPatch((6.8, 0.5), 4.0, 3.5,
                            boxstyle="round,pad=0.05", fc="#1F2937",
                            ec=ORANGE, lw=1.5))
ax.text(7.0, 3.7, "data.yaml", ha="left", fontweight="bold",
        fontsize=10, color=ORANGE)
ax.text(7.0, 0.7, yaml_text, ha="left", va="bottom",
        fontfamily="monospace", fontsize=9, color="white",
        linespacing=1.3)
line(7.7, 5.5, 8.5, 4.0)

ax.text(5.5, 0.1, "* test/ es opcional. Las imágenes y labels deben tener mismo nombre (img_001.jpg + img_001.txt).",
        ha="center", fontsize=8.5, color=ARCA_DARK, style="italic")

plt.tight_layout()
fig.savefig("/root/Arca/clase-27/fig_yolo_folder.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_yolo_folder.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Multi-class vs single-class
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: single class (placa)
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("1 clase: 'placa'", fontweight="bold", color=BLUE, fontsize=12)
# Cars with same color box
for x, y in [(1.5, 2), (5.5, 3), (3.0, 4.5)]:
    ax.add_patch(Rectangle((x, y), 2.5, 1.5, fc=GRAY, ec="white", lw=1))
    ax.add_patch(Rectangle((x+0.7, y+0.2), 1.1, 0.3, fc="white", ec=GREEN, lw=2))
    ax.text(x+1.25, y+0.35, "placa", ha="center", va="center", fontsize=7, color=GREEN, fontweight="bold")
ax.text(5, 0.5, "Mismo color, mismo label\n→ data.yaml: names: {0: placa}",
        ha="center", fontsize=10, color=ARCA_DARK)

# Right: multi-class (cold drinks)
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("Multi-clase: coca, pepsi, sprite, ...", fontweight="bold", color=ARCA_RED, fontsize=12)
products = [
    (0.5, 4.5, "coca",   ARCA_RED),
    (2.3, 4.5, "pepsi",  BLUE),
    (4.1, 4.5, "sprite", GREEN),
    (5.9, 4.5, "coca",   ARCA_RED),
    (7.7, 4.5, "fanta",  ORANGE),
    (0.5, 2.5, "pepsi",  BLUE),
    (2.3, 2.5, "coca",   ARCA_RED),
    (4.1, 2.5, "sprite", GREEN),
    (5.9, 2.5, "coca",   ARCA_RED),
    (7.7, 2.5, "fanta",  ORANGE),
]
for x, y, name, color in products:
    ax.add_patch(Rectangle((x, y), 1.5, 1.6, fc=color, ec="white", alpha=0.8))
    ax.text(x + 0.75, y + 0.8, name, ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")
ax.text(5, 0.7, "Cada producto su color y class_id\n→ data.yaml: names: {0:coca, 1:pepsi, 2:sprite, 3:fanta}",
        ha="center", fontsize=10, color=ARCA_DARK)

fig.suptitle("Single-class vs multi-class detection: la diferencia está en data.yaml",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("/root/Arca/clase-27/fig_multiclass.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_multiclass.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Bbox failure on dense overlap
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: bbox detection fails when objects overlap
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("Detección por bbox: productos densos", fontweight="bold",
             color=ARCA_RED, fontsize=12)
# 8 bottles touching each other
for i in range(8):
    x = 1 + i * 1.0
    ax.add_patch(Rectangle((x, 1), 0.9, 4, fc=ARCA_RED, alpha=0.7, ec="white"))
    # Bbox over each
    ax.add_patch(Rectangle((x-0.05, 0.95), 1.0, 4.1, fill=False, ec=GREEN, lw=2))
ax.text(5, 0.3,
        "8 botellas pegadas: bboxes solapan mucho.\nNMS borra cajas válidas → conteo malo",
        ha="center", fontsize=10, color=ARCA_DARK)

# Right: pixel-level segmentation
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("Segmentación: máscara por pixel", fontweight="bold",
             color=GREEN, fontsize=12)
colors_seg = [ARCA_RED, BLUE, GREEN, ORANGE, PURPLE,
              "#06B6D4", "#D946EF", "#F59E0B"]
for i in range(8):
    x = 1 + i * 1.0
    # Body
    ax.add_patch(Rectangle((x, 1), 0.9, 4, fc=colors_seg[i], alpha=0.9, ec="white", lw=1.5))
ax.text(5, 0.3,
        "Cada botella es una máscara distinta a nivel pixel.\nConteo exacto y no se afectan entre sí.",
        ha="center", fontsize=10, color=ARCA_DARK)

fig.suptitle("Por qué necesitamos segmentación: productos densos rompen los bboxes",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("/root/Arca/clase-27/fig_seg_motivacion.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_seg_motivacion.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Semantic vs Instance segmentation
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
np.random.seed(7)

# Generate base "scene"
def draw_shapes(ax, mode):
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    shapes = [(1.5, 2, "circle"), (3.5, 2.5, "circle"), (5.5, 2, "circle"),
              (7.5, 2.5, "rect"),  (2.0, 4, "rect")]
    # Background
    ax.add_patch(Rectangle((0, 0), 10, 5, fc="#F3F4F6"))
    for i, (cx, cy, kind) in enumerate(shapes):
        if mode == "image":
            color = ARCA_RED if kind == "circle" else BLUE
        elif mode == "semantic":
            # Same color per class
            color = ARCA_RED if kind == "circle" else BLUE
        else:  # instance
            color = ["#C82B40", "#EA580C", "#7C3AED",
                    "#2563EB", "#0EA5E9"][i]
        if kind == "circle":
            ax.add_patch(Circle((cx, cy), 0.7, fc=color, ec="white", lw=1.5,
                                 alpha=0.9))
        else:
            ax.add_patch(Rectangle((cx-0.7, cy-0.6), 1.4, 1.2, fc=color,
                                    ec="white", lw=1.5, alpha=0.9))
    if mode != "image":
        for cx, cy, kind in shapes:
            ax.text(cx, cy, "P" if kind=="rect" else "C",
                    ha="center", va="center", color="white",
                    fontweight="bold", fontsize=10)

draw_shapes(axes[0], "image")
axes[0].set_title("(a) Imagen original", fontweight="bold", color=ARCA_DARK, fontsize=11)

draw_shapes(axes[1], "semantic")
axes[1].set_title("(b) Segmentación SEMÁNTICA\n(cada pixel = clase, sin instancias)",
                  fontweight="bold", color=BLUE, fontsize=11)

draw_shapes(axes[2], "instance")
axes[2].set_title("(c) Segmentación INSTANCE\n(cada objeto = id único)",
                  fontweight="bold", color=GREEN, fontsize=11)

fig.suptitle("Dos tipos de segmentación. YOLO segmentation hace (c).",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("/root/Arca/clase-27/fig_seg_types.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_seg_types.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Mask IoU visualization
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

shapes_iou = [
    ("IoU = 0.15  (mal)", 0.15, ARCA_RED),
    ("IoU = 0.55  (umbral típico)", 0.55, ORANGE),
    ("IoU = 0.88  (excelente)", 0.88, GREEN),
]

for ax, (title, iou, color) in zip(axes, shapes_iou):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    # GT polygon (irregular)
    if iou == 0.15:
        gt = [(2, 2), (5, 1.5), (6, 3), (5, 4.5), (2.5, 4.5), (1.8, 3)]
        pred = [(5, 2.5), (8, 2), (8.5, 4), (7, 5), (5, 4.5), (4.5, 3.5)]
    elif iou == 0.55:
        gt = [(2, 2), (5, 1.5), (6, 3), (5, 4.5), (2.5, 4.5), (1.8, 3)]
        pred = [(3, 2.3), (6, 2), (7, 3.2), (6, 4.7), (3.2, 4.5), (2.5, 3.2)]
    else:
        gt = [(2, 2), (5, 1.5), (6, 3), (5, 4.5), (2.5, 4.5), (1.8, 3)]
        pred = [(2.1, 2.05), (5.05, 1.55), (6.05, 3.05), (4.95, 4.45),
                (2.55, 4.5), (1.85, 3.0)]
    ax.add_patch(Polygon(gt, fc=GREEN, alpha=0.4, ec=GREEN, lw=2))
    ax.add_patch(Polygon(pred, fc=BLUE, alpha=0.4, ec=BLUE, lw=2))
    ax.text(0.5, 5.7, "Real (GT)", color=GREEN, fontweight="bold", fontsize=9)
    ax.text(0.5, 5.3, "Predicha", color=BLUE, fontweight="bold", fontsize=9)
    ax.text(5, 0.3, f"Mask IoU = {iou:.2f}", ha="center",
            fontsize=14, fontweight="bold", color=color)
    ax.set_title(title, fontweight="bold", color=color, fontsize=11)

fig.suptitle("Mask IoU: intersección / unión de polígonos, no de rectángulos",
             fontweight="bold", color=ARCA_DARK, fontsize=13)
plt.tight_layout()
fig.savefig("/root/Arca/clase-27/fig_mask_iou.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_mask_iou.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: EasyOCR pipeline diagram (CRAFT + CRNN)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")

stages = [
    (0.3, "Imagen\nde entrada", BLUE,
     "Foto con\ntexto en\ncualquier\nposición"),
    (3.0, "CRAFT", ARCA_RED,
     "CNN tipo\nVGG/UNet\n\nDetecta\nregiones\nde texto"),
    (6.0, "Polígonos\ndetectados", PURPLE,
     "Lista de\nbboxes\n+ rotación\ndonde hay\ntexto"),
    (9.0, "CRNN", ORANGE,
     "CNN +\nBiLSTM +\nCTC decoder\n\nLee cada\nregión"),
    (12.0, "Texto +\nconfianza", GREEN,
     "['PCJ-3421',\n 0.94]\n\nlista de\nresultados"),
]

for x, title, color, body in stages:
    ax.add_patch(FancyBboxPatch((x, 0.5), 1.7, 4.0,
                                boxstyle="round,pad=0.05",
                                fc=color, ec="white", lw=1.5, alpha=0.9))
    ax.text(x + 0.85, 3.9, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(x + 0.85, 2.2, body, ha="center", va="center",
            fontsize=8.5, color="white")

# Arrows
for x_pos in [2.05, 4.75, 7.75, 10.75]:
    ax.annotate("", xy=(x_pos + 0.95, 2.5), xytext=(x_pos, 2.5),
                arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))

fig.suptitle("EasyOCR por dentro: 2 modelos encadenados (detección + reconocimiento)",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("/root/Arca/clase-27/fig_easyocr_pipeline.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_easyocr_pipeline.png")

print("\nFiguras de clase-27 listas.")
