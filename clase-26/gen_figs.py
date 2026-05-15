"""Figuras nuevas para Clase 26 — Multi-output, sliding window, COCO, team roles."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from PIL import Image

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"; LIGHT = "#F5F5F5"

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Multi-output network — de clasificación a detección
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")

# Input (image)
ax.add_patch(FancyBboxPatch((0.3, 1.5), 1.8, 2.0,
                            boxstyle="round,pad=0.05", fc=BLUE,
                            ec="white", lw=1.5))
ax.text(1.2, 4.0, "Imagen\nentrada", ha="center", fontweight="bold",
        fontsize=10, color=ARCA_DARK)
ax.text(1.2, 2.5, "(96 x 96 x 3)", ha="center", va="center",
        fontsize=9, color="white", fontweight="bold")

# CNN / backbone
ax.add_patch(FancyBboxPatch((3.0, 1.5), 2.5, 2.0,
                            boxstyle="round,pad=0.05", fc=ARCA_RED,
                            ec="white", lw=1.5))
ax.text(4.25, 4.0, "CNN\n(backbone)", ha="center", fontweight="bold",
        fontsize=10, color=ARCA_DARK)
ax.text(4.25, 2.5, "Conv2D\nMaxPool\nFlatten", ha="center", va="center",
        fontsize=9, color="white", fontweight="bold")

# Multi-output heads (5 outputs)
heads = [
    (10.5, 4.3, "x", GREEN, "regresion\n(coordenada)"),
    (10.5, 3.5, "y", GREEN, ""),
    (10.5, 2.7, "w", GREEN, ""),
    (10.5, 1.9, "h", GREEN, ""),
    (10.5, 1.0, "clase", PURPLE, "softmax\n(que es)"),
]
for x_pos, y_pos, label, color, sublabel in heads:
    ax.add_patch(Circle((x_pos, y_pos), 0.35, fc=color, ec="white", lw=1.5))
    ax.text(x_pos, y_pos, label, ha="center", va="center",
            fontsize=11, color="white", fontweight="bold")
    if sublabel:
        ax.text(x_pos + 1.2, y_pos, sublabel, ha="left", va="center",
                fontsize=8.5, color=ARCA_DARK, style="italic")

# Connections from CNN to heads
for _, y_pos, _, _, _ in heads:
    ax.plot([5.55, 10.15], [2.5, y_pos], color=GRAY, lw=0.8, alpha=0.7)

# Arrows between input -> CNN
ax.annotate("", xy=(2.95, 2.5), xytext=(2.15, 2.5),
            arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=1.8))

# Top label
ax.text(7, 4.7, "5 CABEZAS PARALELAS = bounding box (x, y, w, h) + clase",
        ha="center", fontsize=12, fontweight="bold", color=ARCA_DARK)

ax.text(7, 0.3, "Una sola red, multi-output: clasificacion + regresion al mismo tiempo",
        ha="center", fontsize=10, color=ARCA_DARK, style="italic")

fig.suptitle("De clasificacion a deteccion: agregar mas cabezas de salida",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.0)
plt.tight_layout()
fig.savefig("/root/Arca/clase-26/fig_multioutput.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_multioutput.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Sliding window — visualizacion con conteo
# ─────────────────────────────────────────────────────────────────────────────
# Crear una imagen sintética simple con un "objeto" (placa)
np.random.seed(7)
img_demo = np.full((96, 96, 3), 220, dtype=np.uint8)  # fondo claro
# Dibujar un rectángulo (placa) en algún lugar
plate_y, plate_x = 50, 30
plate_h, plate_w = 12, 26
img_demo[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w] = [40, 80, 200]
# Dibujar otras formas para distraer
img_demo[10:20, 70:80] = [200, 80, 40]
img_demo[70:90, 10:20] = [80, 200, 100]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel 1: imagen con grid sliding
ax = axes[0]
ax.imshow(img_demo)
ax.axis("off")
window_size = 16
stride = 8
count = 0
for y in range(0, 96 - window_size + 1, stride):
    for x in range(0, 96 - window_size + 1, stride):
        ax.add_patch(Rectangle((x, y), window_size, window_size,
                               fill=False, ec=ARCA_RED, lw=0.5, alpha=0.4))
        count += 1
ax.set_title(f"Ventana 16x16, stride 8\n{count} pasadas en imagen 96x96",
             fontweight="bold", color=ARCA_DARK, fontsize=11)

# Panel 2: estadística
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
ax.text(5, 7.0, "Tabla de costos", ha="center", fontweight="bold",
        fontsize=13, color=ARCA_DARK)

scenarios = [
    ("Imagen 96x96\nventana 16x16\nstride 8", "121 pasadas", GREEN),
    ("Imagen 640x640\nventana 32x32\nstride 4", "~25,000 pasadas", ORANGE),
    ("Igual + 3 escalas\nde ventana", "~75,000 pasadas", ARCA_RED),
    ("A 1ms/pasada\n(GPU rapida)", "~75 segundos por imagen", ARCA_DARK),
]
y_pos = 5.5
for desc, val, color in scenarios:
    ax.text(1.5, y_pos, desc, ha="left", va="center", fontsize=9.5,
            color=ARCA_DARK)
    ax.text(7.0, y_pos, val, ha="left", va="center", fontsize=10,
            color=color, fontweight="bold")
    y_pos -= 1.3
ax.text(5, 0.3, "Inviable a tiempo real $\\to$ YOLO usa 1 sola pasada",
        ha="center", fontsize=10, fontweight="bold", color=ARCA_RED,
        bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=ARCA_RED))

# Panel 3: YOLO comparison
ax = axes[2]
ax.imshow(img_demo)
ax.axis("off")
# Single bbox where the plate is
ax.add_patch(Rectangle((plate_x-2, plate_y-2), plate_w+4, plate_h+4,
                       fill=False, ec=GREEN, lw=3))
ax.text(plate_x, plate_y - 5, "placa  0.94", fontsize=10, color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc=GREEN, ec="none"))
ax.set_title("YOLO: 1 sola pasada\nresultado directo",
             fontweight="bold", color=GREEN, fontsize=11)

fig.suptitle("Sliding window (izq): solucion ingenua. YOLO (der): eficiente.",
             fontweight="bold", color=ARCA_DARK, fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig("/root/Arca/clase-26/fig_sliding_window.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_sliding_window.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: COCO classes — visualización con ausencias destacadas
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")

# Las 80 clases reales de COCO
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove",
    "skateboard", "surfboard", "tennis_racket", "bottle", "wine_glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch",
    "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
    "hair_drier", "toothbrush",
]
# Grid 10 cols x 8 rows
n_cols = 10
for i, name in enumerate(coco_classes):
    col = i % n_cols
    row = i // n_cols
    x = 0.5 + col * 1.3
    y = 6.5 - row * 0.7
    ax.text(x, y, name, ha="center", va="center",
            fontsize=8, color=GRAY,
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec=GRAY, lw=0.5))

# Highlight what's MISSING for our project
ax.text(7, 0.6,
        "...y NO esta 'placa', NI 'lata de Coca', NI 'producto defectuoso'",
        ha="center", fontsize=12, fontweight="bold", color=ARCA_RED,
        bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec=ARCA_RED, lw=1.5))
ax.text(7, 0.05, "$\\to$ Para todo eso necesitamos FINE-TUNE",
        ha="center", fontsize=11, fontweight="bold", color=ARCA_RED)

ax.text(7, 7.5, "Las 80 clases de COCO (donde se entreno YOLO)",
        ha="center", fontsize=13, fontweight="bold", color=ARCA_DARK)
plt.tight_layout()
fig.savefig("/root/Arca/clase-26/fig_coco_classes.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_coco_classes.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Team roles diagram — equipo de etiquetado
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")

# Manager
ax.add_patch(FancyBboxPatch((5.0, 3.7), 3, 1.0,
                            boxstyle="round,pad=0.08", fc=ARCA_DARK,
                            ec="white", lw=1.5))
ax.text(6.5, 4.45, "Manager / Owner", ha="center", va="center",
        fontsize=11, fontweight="bold", color="white")
ax.text(6.5, 3.95, "(Carlos)", ha="center", va="center",
        fontsize=9, color="white", style="italic")

# Annotators (multiple)
roles = [
    (0.5, 1.5, "Annotator 1", BLUE, "dibuja bboxes\n40 imagenes"),
    (3.0, 1.5, "Annotator 2", BLUE, "dibuja bboxes\n40 imagenes"),
    (5.5, 1.5, "Annotator 3", BLUE, "dibuja bboxes\n40 imagenes"),
    (8.0, 1.5, "Annotator 4", BLUE, "dibuja bboxes\n40 imagenes"),
    (10.5, 1.5, "Reviewer", GREEN, "verifica calidad\nde labels"),
]

for x, y, label, color, desc in roles:
    ax.add_patch(FancyBboxPatch((x, y), 2.0, 1.0,
                                boxstyle="round,pad=0.05", fc=color,
                                ec="white", lw=1.5))
    ax.text(x + 1, y + 0.7, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")
    ax.text(x + 1, y + 0.3, desc, ha="center", va="center",
            fontsize=8, color="white")

# Arrows from manager to all annotators
for x, _, _, _, _ in roles[:4]:
    ax.annotate("", xy=(x + 1, 2.55), xytext=(6.5, 3.65),
                arrowprops=dict(arrowstyle="->", color=ARCA_DARK,
                                lw=1.2, alpha=0.5))

# Arrows from annotators to reviewer
for x, _, _, _, _ in roles[:4]:
    ax.annotate("", xy=(11.45, 2.0), xytext=(x + 2, 2.0),
                arrowprops=dict(arrowstyle="->", color=GREEN,
                                lw=0.8, alpha=0.4, linestyle=":"))

ax.text(6.5, 0.4,
        "Manager asigna lotes -> Annotators etiquetan -> Reviewer aprueba o rechaza",
        ha="center", fontsize=10, color=ARCA_DARK, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec=ARCA_DARK, lw=0.5))

fig.suptitle("Roles tipicos de un equipo de annotation profesional",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.0)
plt.tight_layout()
fig.savefig("/root/Arca/clase-26/fig_team_roles.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_team_roles.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Label Studio en Colab — diagrama del setup
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")

# Teacher's machine (Colab)
ax.add_patch(FancyBboxPatch((0.3, 1.5), 3.5, 2.5,
                            boxstyle="round,pad=0.08", fc=ARCA_DARK,
                            ec="white", lw=1.5))
ax.text(2.05, 3.7, "Colab del Profesor", ha="center", fontweight="bold",
        fontsize=11, color="white")
ax.text(2.05, 3.1, "Label Studio\ncorriendo en\npuerto 8080",
        ha="center", va="center", fontsize=9, color="white",
        fontweight="bold")
ax.text(2.05, 1.85, "(GPU T4 gratis)", ha="center", fontsize=8,
        color="white", style="italic")

# tmole tunnel
ax.add_patch(FancyBboxPatch((4.5, 1.8), 2.5, 1.4,
                            boxstyle="round,pad=0.05", fc=ORANGE,
                            ec="white", lw=1.5))
ax.text(5.75, 2.7, "tmole tunnel", ha="center", fontweight="bold",
        fontsize=10, color="white")
ax.text(5.75, 2.2, "URL publica:\nhttps://abc.tunnelmole.net",
        ha="center", va="center", fontsize=8, color="white",
        fontfamily="monospace")

# Students
students = [
    (8.5, 3.5, "Estudiante 1"),
    (8.5, 2.5, "Estudiante 2"),
    (8.5, 1.5, "Estudiante 3"),
    (11.0, 3.5, "Estudiante 4"),
    (11.0, 2.5, "..."),
    (11.0, 1.5, "Estudiante N"),
]
for x, y, label in students:
    ax.add_patch(FancyBboxPatch((x, y), 2.0, 0.7,
                                boxstyle="round,pad=0.05", fc=GREEN,
                                ec="white", lw=1.2))
    ax.text(x + 1, y + 0.35, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")

# Arrows
ax.annotate("", xy=(4.45, 2.5), xytext=(3.85, 2.5),
            arrowprops=dict(arrowstyle="<->", color=ARCA_DARK, lw=2))
ax.text(4.15, 2.8, "expone", fontsize=8, color=ARCA_DARK)

for _, y, _ in students:
    ax.annotate("", xy=(8.45, y + 0.35), xytext=(7.05, 2.5),
                arrowprops=dict(arrowstyle="<->", color=ARCA_DARK,
                                lw=0.8, alpha=0.5))

ax.text(7, 4.5,
        "Una sola instancia de Label Studio, varios estudiantes conectados",
        ha="center", fontsize=11, fontweight="bold", color=ARCA_DARK)
ax.text(7, 0.4,
        "Todos ven el mismo proyecto, roles distintos, trabajan en paralelo",
        ha="center", fontsize=9.5, color=ARCA_DARK, style="italic")

fig.savefig("/root/Arca/clase-26/fig_labelstudio_setup.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_labelstudio_setup.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Métricas — visualización de mAP, precision, recall
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Panel A: Precision-Recall curve (típica)
ax = axes[0]
recall = np.linspace(0, 1, 100)
# Curve that drops near recall=1
precision = 1 - 0.1*recall - 0.5*recall**3 - 0.3*recall**8
precision = np.clip(precision, 0.1, 1.0)
ax.plot(recall, precision, color=ARCA_RED, lw=3)
ax.fill_between(recall, 0, precision, color=ARCA_RED, alpha=0.15)
# Annotate AP (area under curve)
import numpy as np
ap = np.trapz(precision, recall)
ax.text(0.5, 0.5, f"AP = {ap:.2f}\n(area bajo la curva)",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color=ARCA_DARK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ARCA_DARK))
ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
ax.set_title("Curva Precision-Recall (mAP@0.5 = area)",
             fontweight="bold", color=ARCA_DARK, fontsize=11)
ax.grid(True, alpha=0.3)

# Panel B: Bar chart de las 4 metricas principales
ax = axes[1]
metrics = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
values_before = [0.0, 0.0, 0.0, 0.0]                  # pretrained YOLO en placas
values_after  = [0.92, 0.87, 0.88, 0.61]              # despues de fine-tune
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, values_before, w, label="Pretrained", color=GRAY)
ax.bar(x + w/2, values_after,  w, label="Fine-tuned", color=ARCA_RED)
for i, (b, a) in enumerate(zip(values_before, values_after)):
    ax.text(i - w/2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9,
            color=GRAY, fontweight="bold")
    ax.text(i + w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9,
            color=ARCA_RED, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel("Valor", fontweight="bold")
ax.set_title("Antes vs despues del fine-tuning",
             fontweight="bold", color=ARCA_DARK, fontsize=11)
ax.legend()

fig.suptitle("Metricas para evaluar deteccion: 4 numeros + 1 curva",
             fontweight="bold", color=ARCA_DARK, fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("/root/Arca/clase-26/fig_metrics.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_metrics.png")

print("\nFiguras nuevas de clase-26 listas.")
