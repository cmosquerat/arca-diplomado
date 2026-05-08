"""Figuras para Clase 25 — Object Detection con YOLO."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon
from sklearn.datasets import load_sample_images

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})

ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"; LIGHT = "#F5F5F5"

# We don't have a real shelf image; build a synthetic mock-up
def build_mock_shelf(seed=1):
    """Synthetic shelf with 'cans' as colored rectangles."""
    np.random.seed(seed)
    H, W = 320, 640
    img = np.full((H, W, 3), 230, dtype=np.uint8)         # gray background
    # Shelf wood color band
    img[0:H, :] = [240, 235, 225]
    # Two horizontal shelves
    for y_shelf in [110, 220]:
        img[y_shelf:y_shelf+8, :] = [120, 80, 50]
    # Place "cans" — Coca (red), Pepsi (blue), Inca (yellow)
    cans = []
    can_colors = {"Coca":  (200, 30, 50),
                  "Pepsi": (40, 70, 180),
                  "Inca":  (240, 200, 60)}
    placements = [
        # row 1 (top shelf, y=20..110)
        ("Coca",  20,  20, 60, 90),
        ("Coca",  85,  22, 60, 88),
        ("Pepsi", 150, 20, 60, 90),
        ("Inca",  215, 23, 58, 87),
        ("Coca",  280, 20, 60, 90),
        ("Pepsi", 345, 22, 60, 88),
        ("Pepsi", 410, 21, 60, 89),
        ("Inca",  475, 22, 58, 88),
        ("Coca",  540, 20, 60, 90),
        # row 2 (y=130..220)
        ("Pepsi", 30, 132, 60, 88),
        ("Inca",  95, 130, 58, 90),
        ("Inca", 160, 132, 58, 88),
        ("Coca", 225, 130, 60, 90),
        ("Pepsi", 290, 132, 60, 88),
        ("Coca", 355, 130, 60, 90),
        ("Inca", 420, 132, 58, 88),
        ("Pepsi", 485, 130, 60, 90),
        ("Coca", 550, 132, 60, 88),
    ]
    for name, x, y, w, h in placements:
        c = can_colors[name]
        # Body
        img[y:y+h, x:x+w] = c
        # Top white band
        img[y+5:y+15, x:x+w] = [240, 240, 240]
        # Bottom shadow
        img[y+h-5:y+h, x:x+w] = [int(c[0]*0.6), int(c[1]*0.6), int(c[2]*0.6)]
        cans.append((name, x, y, w, h))
    return img, cans

shelf_img, shelf_cans = build_mock_shelf()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: HERO — góndola con bboxes + conteos
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.imshow(shelf_img)
class_colors = {"Coca": ARCA_RED, "Pepsi": BLUE, "Inca": ORANGE}
counts = {"Coca": 0, "Pepsi": 0, "Inca": 0}
for name, x, y, w, h in shelf_cans:
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=class_colors[name], lw=2.5))
    ax.text(x+2, y-2, name, fontsize=8, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc=class_colors[name], ec="none"))
    counts[name] += 1
ax.axis("off")
# Counter panel
panel_text = "  ".join(f"{c}: {n}" for c, n in counts.items())
ax.text(320, 305, f"DETECTADO  →  {panel_text}",
        ha="center", fontsize=14, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.5", fc=ARCA_DARK, ec="none"))
fig.suptitle("Lo que vamos a construir hoy: detector + contador de productos en gondola",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=0.99)
plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_hero.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_hero.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: 4 TAREAS DE VISION
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
# We use a small crop of the shelf for all 4
small = shelf_img[10:215, 10:265]
HH, WW = small.shape[:2]

# 1. Classification
ax = axes[0]
ax.imshow(small); ax.axis("off")
ax.set_title("CLASIFICACION\n('Hay productos en la imagen')",
             fontweight="bold", color=BLUE, fontsize=11)
ax.text(WW/2, -15, '"productos en gondola"\n(1 etiqueta para toda la imagen)',
        ha="center", fontsize=10, color=ARCA_DARK)

# 2. Detection
ax = axes[1]
ax.imshow(small); ax.axis("off")
for name, x, y, w, h in shelf_cans:
    if 10 <= x and x+w <= 275 and y+h <= 215:
        xx, yy = x-10, y-10
        ax.add_patch(Rectangle((xx, yy), w, h, fill=False,
                               ec=class_colors[name], lw=2))
ax.set_title("DETECCION\n('Que hay y donde')",
             fontweight="bold", color=ARCA_RED, fontsize=11)
ax.text(WW/2, -15, '"Coca, Pepsi, Inca + bbox"\n(N etiquetas + cajas)',
        ha="center", fontsize=10, color=ARCA_DARK)

# 3. Segmentation
ax = axes[2]
ax.imshow(small); ax.axis("off")
# Overlay color masks instead of boxes
for name, x, y, w, h in shelf_cans:
    if 10 <= x and x+w <= 275 and y+h <= 215:
        xx, yy = x-10, y-10
        # Pseudo-mask: ellipse for the can shape
        from matplotlib.patches import Ellipse
        e = Ellipse(((xx+w/2), yy+h/2), w*0.85, h*0.95,
                    facecolor=class_colors[name], alpha=0.55, edgecolor="white", lw=1)
        ax.add_patch(e)
ax.set_title("SEGMENTACION\n('Que pixeles son cada cosa')",
             fontweight="bold", color=GREEN, fontsize=11)
ax.text(WW/2, -15, '"mascara por pixel"\n(precision al pixel)',
        ha="center", fontsize=10, color=ARCA_DARK)

# 4. Pose / keypoints (mock with red dots on cans)
ax = axes[3]
ax.imshow(small); ax.axis("off")
for name, x, y, w, h in shelf_cans:
    if 10 <= x and x+w <= 275 and y+h <= 215:
        xx, yy = x-10, y-10
        # 3 keypoints: top, middle, bottom of can
        cx = xx + w/2
        for ky in [yy+10, yy+h/2, yy+h-10]:
            ax.plot(cx, ky, 'o', color=PURPLE, markersize=5,
                    markeredgecolor="white", markeredgewidth=1)
        ax.plot([cx, cx], [yy+10, yy+h-10], color=PURPLE, lw=1.5)
ax.set_title("POSE / KEYPOINTS\n('Puntos de interes')",
             fontweight="bold", color=PURPLE, fontsize=11)
ax.text(WW/2, -15, '"esqueleto / articulaciones"\n(humanos, manos, animales)',
        ha="center", fontsize=10, color=ARCA_DARK)

fig.suptitle("Tareas de vision por computadora: cada una resuelve un problema distinto",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=1.05)
plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_4tareas.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_4tareas.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: BBOX ANATOMY
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")

# Show one can with annotations
ax.add_patch(Rectangle((0.5, 0.5), 5, 4, fc=LIGHT, ec=GRAY, lw=1))
ax.text(3, 4.7, "IMAGEN", ha="center", fontsize=11,
        fontweight="bold", color=ARCA_DARK)
# axes labels for image
ax.annotate("", xy=(5.7, 4.5), xytext=(0.5, 4.5),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.2))
ax.text(0.3, 4.5, "x=0", fontsize=9, color=ARCA_DARK, ha="right")
ax.annotate("", xy=(0.5, 0.3), xytext=(0.5, 4.5),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.2))
ax.text(0.3, 0.5, "y=0", fontsize=9, color=ARCA_DARK, ha="right")

# A "can" inside (just a colored rectangle)
can_x, can_y, can_w, can_h = 1.7, 1.4, 1.4, 2.4
# Drawing rect: matplotlib y goes up; but image coords y goes down
# For pedagogy, let's use image convention (top-left origin) by inverting y
# Actually just draw and label with note
ax.add_patch(Rectangle((can_x, can_y), can_w, can_h, fc=ARCA_RED, ec="white", alpha=0.75))
# bbox highlighted in green
ax.add_patch(Rectangle((can_x, can_y), can_w, can_h,
                        fill=False, ec=GREEN, lw=3, linestyle="--"))
# Mark x, y at top-left of bbox
ax.plot(can_x, can_y+can_h, 'o', color=GREEN, markersize=8, zorder=5)
ax.text(can_x+0.05, can_y+can_h+0.15, "(x, y)",
        fontsize=10, fontweight="bold", color=GREEN)
# Width arrow
ax.annotate("", xy=(can_x+can_w, can_y-0.15), xytext=(can_x, can_y-0.15),
            arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5))
ax.text(can_x+can_w/2, can_y-0.4, "w", ha="center",
        fontsize=10, fontweight="bold", color=GREEN)
# Height arrow
ax.annotate("", xy=(can_x+can_w+0.15, can_y+can_h), xytext=(can_x+can_w+0.15, can_y),
            arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5))
ax.text(can_x+can_w+0.4, can_y+can_h/2, "h", va="center",
        fontsize=10, fontweight="bold", color=GREEN)
# Class label
ax.text(can_x+can_w/2, can_y+can_h+0.7, "clase=Coca",
        ha="center", fontsize=10, color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc=ARCA_RED, ec="none"))

# Right panel: the data
ax.add_patch(FancyBboxPatch((6.5, 1.2), 4.0, 3.0,
                            boxstyle="round,pad=0.1",
                            fc=LIGHT, ec=ARCA_DARK, lw=1.5))
ax.text(8.5, 4.0, "Una bounding box\nes solo 5 numeros",
        ha="center", va="center", fontsize=12,
        fontweight="bold", color=ARCA_DARK)
labels = [("clase", "Coca", ARCA_RED),
          ("x", "200", GREEN),
          ("y", "120", GREEN),
          ("w", "80", GREEN),
          ("h", "150", GREEN)]
for k, (name, val, color) in enumerate(labels):
    y_pos = 3.2 - k*0.35
    ax.text(7.0, y_pos, f"{name}:", fontsize=11, color=ARCA_DARK,
            fontweight="bold")
    ax.text(8.5, y_pos, val, fontsize=11, color=color, fontweight="bold")

fig.suptitle("Anatomia de una bounding box: 4 numeros + 1 clase",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=0.98)
plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_bbox.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_bbox.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: IoU
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

scenarios = [
    ("IoU = 0.2 (mal)", (1.5, 1.0, 3.5, 3.5), (3.0, 2.0, 3.5, 3.5), 0.2, ARCA_RED),
    ("IoU = 0.5 (umbral t\\'ipico)", (1.5, 1.0, 3.5, 3.5), (2.5, 1.5, 3.5, 3.5), 0.5, ORANGE),
    ("IoU = 0.85 (excelente)", (1.5, 1.0, 3.5, 3.5), (1.7, 1.2, 3.4, 3.4), 0.85, GREEN),
]

for ax, (title, gt, pred, iou, color) in zip(axes, scenarios):
    ax.set_xlim(0, 7); ax.set_ylim(0, 5); ax.axis("off")
    # Ground truth (verde claro)
    ax.add_patch(Rectangle((gt[0], gt[1]), gt[2], gt[3],
                           fc=GREEN, ec=GREEN, lw=2.5, alpha=0.25))
    ax.text(gt[0]+0.1, gt[1]+gt[3]+0.1, "Real (GT)",
            fontsize=9, color=GREEN, fontweight="bold")
    # Prediction
    ax.add_patch(Rectangle((pred[0], pred[1]), pred[2], pred[3],
                           fc=BLUE, ec=BLUE, lw=2.5, alpha=0.25))
    ax.text(pred[0]+0.1, pred[1]-0.25, "Predicho",
            fontsize=9, color=BLUE, fontweight="bold")
    # IoU value
    ax.text(3.5, 0.3, f"IoU = {iou:.2f}",
            ha="center", fontsize=14, fontweight="bold", color=color)
    ax.set_title(title, fontweight="bold", color=color, fontsize=11)

fig.suptitle("IoU (Intersection over Union) = area de intersecci\\'on / area de uni\\'on",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.0)
plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_iou.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_iou.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: NMS - many boxes -> one box per object
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Use one section of the shelf
section = shelf_img[10:215, 10:265]

# Left: many overlapping detections
ax = axes[0]
ax.imshow(section); ax.axis("off")
# Generate many slightly off bboxes for the 3 visible cans
np.random.seed(42)
target_cans = [c for c in shelf_cans if 10 <= c[1] <= 220 and c[2]+c[4] <= 215][:3]
for name, x, y, w, h in target_cans:
    xx, yy = x-10, y-10
    # 5 noisy detections
    for _ in range(5):
        dx = np.random.randint(-15, 15); dy = np.random.randint(-15, 15)
        dw = np.random.randint(-10, 10); dh = np.random.randint(-10, 10)
        ax.add_patch(Rectangle((xx+dx, yy+dy), w+dw, h+dh,
                               fill=False, ec=class_colors[name], lw=1.2, alpha=0.55))
ax.set_title("ANTES de NMS\n(YOLO produce muchas detecciones por objeto)",
             fontweight="bold", color=ARCA_RED, fontsize=11)

# Right: after NMS — clean boxes
ax = axes[1]
ax.imshow(section); ax.axis("off")
for name, x, y, w, h in target_cans:
    xx, yy = x-10, y-10
    ax.add_patch(Rectangle((xx, yy), w, h, fill=False,
                           ec=class_colors[name], lw=2.5))
    ax.text(xx+2, yy-3, name, fontsize=8, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc=class_colors[name], ec="none"))
ax.set_title("DESPUES de NMS\n(1 caja final por objeto)",
             fontweight="bold", color=GREEN, fontsize=11)

fig.suptitle("NMS (Non-Maximum Suppression): se queda con la caja de mayor score, descarta las que se solapan mucho",
             fontweight="bold", color=ARCA_DARK, fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_nms.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_nms.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: YOLO grid concept
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
section = shelf_img[10:215, 10:330]
ax.imshow(section)
H_s, W_s = section.shape[:2]

# Draw grid (7x4 typical for YOLO illustration)
n_cols, n_rows = 8, 5
for i in range(n_rows + 1):
    ax.axhline(y=i * H_s / n_rows, color="white", lw=1, alpha=0.7)
for j in range(n_cols + 1):
    ax.axvline(x=j * W_s / n_cols, color="white", lw=1, alpha=0.7)

# Highlight a few cells that contain object centers
for name, x, y, w, h in shelf_cans[:5]:
    if x+w/2 <= 320 and y+h/2 <= 215:
        cx, cy = x-10 + w/2, y-10 + h/2
        col = int(cx / (W_s / n_cols))
        row = int(cy / (H_s / n_rows))
        ax.add_patch(Rectangle((col * W_s/n_cols, row * H_s/n_rows),
                               W_s/n_cols, H_s/n_rows,
                               fill=False, ec=class_colors[name], lw=3))
        # Mark center
        ax.plot(cx, cy, 'o', color=class_colors[name], markersize=10,
                markeredgecolor="white", markeredgewidth=2)

ax.axis("off")
ax.set_title("YOLO divide la imagen en una rejilla. Cada celda predice 'que hay aqui y donde'",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
fig.savefig("/root/Arca/clase-25/fig_yolo_grid.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_yolo_grid.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: YOLO timeline
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")

# Timeline arrow
ax.annotate("", xy=(12.5, 1.0), xytext=(0.5, 1.0),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2.5))

versions = [
    (0.7, "v1\n2015", "Joseph Redmon\n(U. Washington)\nidea revolucionaria:\n1 sola pasada", BLUE),
    (2.5, "v2-v3\n2016-18", "Redmon mejora\nresultados\n(Darknet)", BLUE),
    (4.5, "v4\n2020", "Bochkovskiy\ntoma la posta", PURPLE),
    (6.7, "v5\n2020", "Ultralytics\n(Glenn Jocher)\nport a PyTorch", ORANGE),
    (8.7, "v8\n2023", "Ultralytics\nAPI moderna\n+ segmentation\n+ pose", ARCA_RED),
    (11.0, "v11\n2024", "estado del arte\nactual", GREEN),
]

for x, ver, desc, color in versions:
    # Marker
    ax.plot(x, 1.0, 'o', color=color, markersize=14, zorder=3,
            markeredgecolor="white", markeredgewidth=2)
    # Version label below
    ax.text(x, 0.5, ver, ha="center", fontsize=10, fontweight="bold", color=color)
    # Description above
    ax.text(x, 2.3, desc, ha="center", fontsize=8.5, color=ARCA_DARK)

# Title
ax.text(6.5, 4.5, "Linea de tiempo de YOLO (You Only Look Once)",
        ha="center", fontsize=14, fontweight="bold", color=ARCA_DARK)
ax.text(6.5, 4.0, "una decada de evolucion en deteccion de objetos en tiempo real",
        ha="center", fontsize=10, color=GRAY, style="italic")

fig.savefig("/root/Arca/clase-25/fig_yolo_timeline.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_yolo_timeline.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: YOLO family — speed vs accuracy
# ─────────────────────────────────────────────────────────────────────────────
# Approximate values for YOLOv8 family
models = ["nano (n)", "small (s)", "medium (m)", "large (l)", "xlarge (x)"]
params  = [3.2, 11.2, 25.9, 43.7, 68.2]    # millions
mAP     = [37.3, 44.9, 50.2, 52.9, 53.9]   # mAP @ 50:95
speed_cpu = [80, 50, 25, 12, 7]            # FPS approx

fig, ax = plt.subplots(figsize=(12, 5))
colors_m = [GREEN, BLUE, ORANGE, ARCA_RED, ARCA_DARK]
for x, y, name, c, p in zip(speed_cpu, mAP, models, colors_m, params):
    ax.scatter(x, y, s=p*40, color=c, alpha=0.8, edgecolor="white", lw=2, zorder=3)
    ax.text(x, y+1.0, name, ha="center", fontsize=10, fontweight="bold", color=c)
    ax.text(x, y-1.6, f"{p}M params", ha="center", fontsize=8, color=ARCA_DARK)

ax.set_xlabel("Velocidad (FPS en CPU laptop)  -->", fontsize=11, fontweight="bold")
ax.set_ylabel("mAP @ 50:95 (precision)  -->", fontsize=11, fontweight="bold")
ax.set_xlim(0, 95)
ax.set_ylim(33, 58)
ax.set_title("YOLOv8 — trade-off velocidad vs precisi\\'on. Tama\\~no del circulo = parametros",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
ax.grid(True, alpha=0.3)

# Add annotations for use cases
ax.annotate("EDGE / movil", xy=(80, 37), xytext=(75, 34.5),
            fontsize=10, color=GREEN, fontweight="bold")
ax.annotate("SERVIDOR / GPU", xy=(7, 53.9), xytext=(2, 56),
            fontsize=10, color=ARCA_DARK, fontweight="bold")

plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_yolo_family.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_yolo_family.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: Fine-tuning concept (specific to YOLO)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))

# Left: pretrained YOLO, doesn't know our classes
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
# Backbone
ax.add_patch(FancyBboxPatch((0.5, 1.5), 4.5, 2,
                            boxstyle="round,pad=0.1",
                            fc=BLUE, ec="white", lw=1.5))
ax.text(2.75, 2.5, "Backbone YOLOv8\n(entrenado en COCO)\n80 clases genericas",
        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
# Head — generic COCO
ax.add_patch(FancyBboxPatch((6, 1.5), 3.5, 2,
                            boxstyle="round,pad=0.1",
                            fc=ORANGE, ec="white", lw=1.5))
ax.text(7.75, 2.5, "Cabezal COCO\n80 clases (perro,\ncoche, botella...)",
        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
# Arrow
ax.annotate("", xy=(5.95, 2.5), xytext=(5.05, 2.5),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))
ax.text(5, 4.2, "Pretrained: detecta 'botella'\npero no 'Coca' vs 'Pepsi'",
        ha="center", fontsize=11, color=BLUE, fontweight="bold")

# Right: fine-tuned
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
# Same backbone, slight adjustment
ax.add_patch(FancyBboxPatch((0.5, 1.5), 4.5, 2,
                            boxstyle="round,pad=0.1",
                            fc=BLUE, ec="white", lw=1.5))
ax.text(2.75, 2.5, "Backbone YOLOv8\n(reusamos los pesos)",
        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
ax.text(2.75, 0.9, "puede ser congelado\no fine-tuneado lento",
        ha="center", fontsize=8, color=ARCA_DARK, style="italic")
# New head
ax.add_patch(FancyBboxPatch((6, 1.5), 3.5, 2,
                            boxstyle="round,pad=0.1",
                            fc=GREEN, ec="white", lw=1.5))
ax.text(7.75, 2.5, "Cabezal NUEVO\n3 clases:\nCoca, Pepsi, Inca",
        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
ax.text(7.75, 0.9, "se entrena con\ntus datos etiquetados",
        ha="center", fontsize=8, color=ARCA_DARK, style="italic")
ax.annotate("", xy=(5.95, 2.5), xytext=(5.05, 2.5),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))
ax.text(5, 4.2, "Fine-tuned: detecta Coca, Pepsi, Inca\nen tu gondola con MUY pocas fotos",
        ha="center", fontsize=11, color=GREEN, fontweight="bold")

fig.suptitle("Fine-tuning de YOLO: misma arquitectura, cabezal nuevo + tus datos",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.05)
plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_finetune.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_finetune.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 10: Annotation formats comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

# YOLO format
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.text(5, 5.5, "Formato YOLO (.txt)", ha="center",
        fontweight="bold", fontsize=12, color=ARCA_RED)
ax.text(5, 4.7, "1 archivo .txt por imagen", ha="center",
        fontsize=9, color=ARCA_DARK, style="italic")
ax.add_patch(FancyBboxPatch((0.3, 1.0), 9.4, 3.0,
                            boxstyle="round,pad=0.05",
                            fc="#1F2937", ec="none"))
content = ("0  0.412  0.387  0.094  0.281\n"
           "0  0.594  0.391  0.094  0.275\n"
           "1  0.781  0.387  0.094  0.281\n"
           "2  0.926  0.395  0.091  0.269\n\n"
           "clase   cx     cy     w      h    (normalizados 0-1)")
ax.text(5, 2.5, content, ha="center", va="center",
        fontfamily="monospace", fontsize=9.5, color="white")

# COCO JSON
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.text(5, 5.5, "Formato COCO (.json)", ha="center",
        fontweight="bold", fontsize=12, color=BLUE)
ax.text(5, 4.7, "1 archivo .json para todo el dataset", ha="center",
        fontsize=9, color=ARCA_DARK, style="italic")
ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 3.7,
                            boxstyle="round,pad=0.05",
                            fc="#1F2937", ec="none"))
content_coco = ('{\n'
                '  "images": [{"id":1, "file_name":"img.jpg",\n'
                '              "width":640, "height":320}],\n'
                '  "annotations": [\n'
                '    {"image_id":1, "category_id":0,\n'
                '     "bbox":[200,120,80,150]}],\n'
                '  "categories":[\n'
                '    {"id":0,"name":"Coca"},...]\n'
                '}')
ax.text(5, 2.15, content_coco, ha="center", va="center",
        fontfamily="monospace", fontsize=8, color="white")

# Pascal VOC XML
ax = axes[2]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.text(5, 5.5, "Formato Pascal VOC (.xml)", ha="center",
        fontweight="bold", fontsize=12, color=GREEN)
ax.text(5, 4.7, "1 archivo .xml por imagen", ha="center",
        fontsize=9, color=ARCA_DARK, style="italic")
ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 3.7,
                            boxstyle="round,pad=0.05",
                            fc="#1F2937", ec="none"))
content_voc = ('<annotation>\n'
               '  <filename>img.jpg</filename>\n'
               '  <object>\n'
               '    <name>Coca</name>\n'
               '    <bndbox>\n'
               '      <xmin>200</xmin>\n'
               '      <ymin>120</ymin>\n'
               '      <xmax>280</xmax>\n'
               '      <ymax>270</ymax>\n'
               '    </bndbox>\n'
               '  </object>\n'
               '</annotation>')
ax.text(5, 2.15, content_voc, ha="center", va="center",
        fontfamily="monospace", fontsize=7.5, color="white")

fig.suptitle("Tres formatos de etiquetado. CVAT y Roboflow exportan a los tres.",
             fontweight="bold", color=ARCA_DARK, fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_formats.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_formats.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 11: project pipeline
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4.0))
ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis("off")

steps = [
    (0.3, "1. CAPTURAR\nfotos de gondola", BLUE,
     "Cada estudiante:\n30-50 fotos\ncon celular"),
    (3.0, "2. ETIQUETAR\nen CVAT", ORANGE,
     "Dibujar bbox\n+ asignar clase\n(Coca/Pepsi/Inca)"),
    (5.7, "3. SUBIR\na Drive comun", PURPLE,
     "Carlos consolida\ntodas las imagenes\nen 1 dataset"),
    (8.4, "4. FINE-TUNE\nYOLOv8n", ARCA_RED,
     "1 linea:\nmodel.train(data=...)\nen Colab GPU"),
    (11.1, "5. APP GRADIO\n+ deploy", GREEN,
     "Foto -> bboxes\n+ conteo en vivo\nlink publico"),
]

for x, title, color, body in steps:
    ax.add_patch(FancyBboxPatch((x, 1.0), 2.5, 1.8,
                                boxstyle="round,pad=0.05",
                                fc=color, ec="white", lw=1.5))
    ax.text(x + 1.25, 2.4, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(x + 1.25, 1.4, body, ha="center", va="center",
            fontsize=8.5, color="white")

# Arrows
for i in range(4):
    x_start = 0.3 + 2.5 + i * 2.7
    x_end = 0.3 + i * 2.7 + 2.7
    ax.annotate("", xy=(x_end + 0.05, 1.9), xytext=(x_start + 0.05, 1.9),
                arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))

fig.suptitle("Pipeline del proyecto de la clase 25 + 26",
             fontweight="bold", color=ARCA_DARK, fontsize=14, y=0.95)
plt.tight_layout()
fig.savefig("/root/Arca/clase-25/fig_pipeline.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_pipeline.png")

print("\nTodas las figuras de clase-25 generadas.")
