"""Figuras para Clase 28 — Modelos en producción + Segmentación + Conteo de píldoras."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
})
ARCA_RED = "#C82B40"; ARCA_DARK = "#6B1525"
GREEN = "#16A34A"; BLUE = "#2563EB"; GRAY = "#9CA3AF"
ORANGE = "#EA580C"; PURPLE = "#7C3AED"; LIGHT = "#F5F5F5"
CYAN = "#0EA5E9"; PINK = "#EC4899"

OUT = "/root/Arca/clase-28"


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Hero — pipeline de la clase (brain + pills)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 4.5); ax.axis("off")

# Top track: brain tumor
ax.text(0.3, 4.0, "BLOQUE A — Detección Médica", fontsize=11,
        fontweight="bold", color=ARCA_DARK)
brain_stages = [
    (0.5, "Dataset\n(LS)", BLUE),
    (3.0, "Train\nyolo26n", ARCA_RED),
    (5.5, "Métricas\nmedical", PURPLE),
    (8.0, "Guardar\nbest.pt", ORANGE),
    (10.5, "Cargar\nen otro lado", GREEN),
]
for i, (x, label, color) in enumerate(brain_stages):
    ax.add_patch(FancyBboxPatch((x, 2.6), 2.0, 1.0, boxstyle="round,pad=0.05",
                                fc=color, ec="white", lw=2))
    ax.text(x + 1.0, 3.1, label, ha="center", va="center",
            fontsize=9.5, color="white", fontweight="bold")
    if i < len(brain_stages) - 1:
        ax.annotate("", xy=(brain_stages[i+1][0], 3.1), xytext=(x + 2.0, 3.1),
                    arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))

# Bottom track: pills
ax.text(0.3, 2.0, "BLOQUE B — Conteo Industrial", fontsize=11,
        fontweight="bold", color=ARCA_DARK)
pill_stages = [
    (0.5, "Roboflow\n(100 imgs)", BLUE),
    (3.0, "LS sin\netiquetas", PURPLE),
    (5.5, "Polígonos\n(equipo)", PINK),
    (8.0, "Train\nyolo-seg", ARCA_RED),
    (10.5, "App\nGradio", GREEN),
]
for i, (x, label, color) in enumerate(pill_stages):
    ax.add_patch(FancyBboxPatch((x, 0.6), 2.0, 1.0, boxstyle="round,pad=0.05",
                                fc=color, ec="white", lw=2))
    ax.text(x + 1.0, 1.1, label, ha="center", va="center",
            fontsize=9.5, color="white", fontweight="bold")
    if i < len(pill_stages) - 1:
        ax.annotate("", xy=(pill_stages[i+1][0], 1.1), xytext=(x + 2.0, 1.1),
                    arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))

plt.tight_layout()
fig.savefig(f"{OUT}/fig_hero.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_hero.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Precision-Recall trade-off — el slider de conf
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: Curva PR teórica
ax = axes[0]
recall = np.linspace(0.05, 1.0, 100)
precision = np.exp(-2.5 * (recall - 0.3)**2) * 0.97
precision[recall > 0.85] *= np.linspace(1.0, 0.55, (recall > 0.85).sum())
ax.plot(recall, precision, color=ARCA_RED, lw=2.5)
ax.fill_between(recall, 0, precision, alpha=0.15, color=ARCA_RED)

# Marker: alto conf (placas)
ax.scatter([0.55], [0.92], s=200, color=BLUE, zorder=5, edgecolor="white", lw=2)
ax.annotate("conf=0.5 (placas)\nP alta, R baja", xy=(0.55, 0.92),
            xytext=(0.20, 0.45), fontsize=9.5, color=BLUE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))

# Marker: bajo conf (tumor)
ax.scatter([0.92], [0.55], s=200, color=ARCA_RED, zorder=5, edgecolor="white", lw=2)
ax.annotate("conf=0.15 (tumor)\nR alta, P baja", xy=(0.92, 0.55),
            xytext=(0.40, 0.20), fontsize=9.5, color=ARCA_RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ARCA_RED, lw=1.5))

ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Curva Precision–Recall:\nel mismo modelo, distinto umbral",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
ax.grid(alpha=0.3)

# Right: dos dominios con preferencia distinta
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("Qué duele más en cada dominio",
             fontweight="bold", color=ARCA_DARK, fontsize=12)

# Placas box
ax.add_patch(FancyBboxPatch((0.3, 3.3), 4.4, 2.4, boxstyle="round,pad=0.1",
                            fc=BLUE, alpha=0.12, ec=BLUE, lw=2))
ax.text(2.5, 5.3, "LPR (Placas)", ha="center", fontweight="bold",
        fontsize=12, color=BLUE)
ax.text(2.5, 4.7, "FP: leer placa fantasma →", ha="center", fontsize=9.5)
ax.text(2.5, 4.3, "registro falso, ruido en sistema", ha="center",
        fontsize=9.5, color=GRAY)
ax.text(2.5, 3.8, "Métrica clave: PRECISION", ha="center",
        fontweight="bold", fontsize=11, color=ARCA_RED)
ax.text(2.5, 3.5, "(prefiero perder uno a inventar)",
        ha="center", fontsize=8.5, color=GRAY, style="italic")

# Brain box
ax.add_patch(FancyBboxPatch((5.3, 3.3), 4.4, 2.4, boxstyle="round,pad=0.1",
                            fc=ARCA_RED, alpha=0.12, ec=ARCA_RED, lw=2))
ax.text(7.5, 5.3, "Brain Tumor", ha="center", fontweight="bold",
        fontsize=12, color=ARCA_RED)
ax.text(7.5, 4.7, "FN: perder tumor real →", ha="center", fontsize=9.5)
ax.text(7.5, 4.3, "vida humana en riesgo", ha="center",
        fontsize=9.5, color=GRAY)
ax.text(7.5, 3.8, "Métrica clave: RECALL", ha="center",
        fontweight="bold", fontsize=11, color=ARCA_RED)
ax.text(7.5, 3.5, "(prefiero alarma falsa a perderlo)",
        ha="center", fontsize=8.5, color=GRAY, style="italic")

# Pills box
ax.add_patch(FancyBboxPatch((2.8, 0.4), 4.4, 2.4, boxstyle="round,pad=0.1",
                            fc=GREEN, alpha=0.12, ec=GREEN, lw=2))
ax.text(5.0, 2.4, "Píldoras", ha="center", fontweight="bold",
        fontsize=12, color=GREEN)
ax.text(5.0, 1.8, "FP y FN ambos cuentan mal →", ha="center", fontsize=9.5)
ax.text(5.0, 1.4, "dosificación incorrecta", ha="center",
        fontsize=9.5, color=GRAY)
ax.text(5.0, 0.9, "Métrica clave: F1 / mAP@0.5", ha="center",
        fontweight="bold", fontsize=11, color=GREEN)

plt.tight_layout()
fig.savefig(f"{OUT}/fig_precision_recall.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_precision_recall.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: mAP@0.5 vs mAP@0.5:0.95 — la diferencia
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

# Left: bbox match a distinto IoU
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("Una predicción, evaluada a distintos IoU",
             fontweight="bold", color=ARCA_DARK, fontsize=12)

# GT
gt = Rectangle((2, 2), 5, 3, fill=False, edgecolor=GREEN, lw=3, linestyle="--")
ax.add_patch(gt)
ax.text(4.5, 5.2, "GT (real)", ha="center", fontsize=10,
        color=GREEN, fontweight="bold")

# Pred
pred = Rectangle((2.7, 2.4), 4.6, 2.7, fill=False, edgecolor=ARCA_RED, lw=2.5)
ax.add_patch(pred)
ax.text(5.0, 5.7, "Pred (modelo)", ha="center", fontsize=10,
        color=ARCA_RED, fontweight="bold")

# IoU label
ax.text(5, 0.7, "IoU = 0.72", ha="center", fontsize=14, fontweight="bold",
        color=ARCA_DARK)
ax.text(5, 0.3, "→ cuenta como TP a IoU≥0.5 (sí)  |  a IoU≥0.75 (no)",
        ha="center", fontsize=10, color=GRAY)

# Right: mAP@0.5 vs mAP@0.5:0.95
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("mAP@0.5 vs mAP@0.5:0.95",
             fontweight="bold", color=ARCA_DARK, fontsize=12)

# mAP@0.5 -- only one threshold
ax.add_patch(FancyBboxPatch((0.3, 3.5), 9.4, 2.0, boxstyle="round,pad=0.1",
                            fc=BLUE, alpha=0.12, ec=BLUE, lw=2))
ax.text(0.7, 5.1, "mAP@0.5", fontweight="bold", fontsize=11, color=BLUE)
ax.text(0.7, 4.6, "AP medido a UN solo umbral: IoU ≥ 0.5",
        fontsize=10, color=ARCA_DARK)
ax.text(0.7, 4.1, "Permisivo. Detección 'suficiente' cuenta.",
        fontsize=9, color=GRAY, style="italic")
ax.text(0.7, 3.7, "↑ típicamente más alto",
        fontsize=9, color=BLUE, fontweight="bold")

# mAP@0.5:0.95 -- average over 10 thresholds
ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 2.5, boxstyle="round,pad=0.1",
                            fc=ARCA_RED, alpha=0.12, ec=ARCA_RED, lw=2))
ax.text(0.7, 2.4, "mAP@0.5:0.95", fontweight="bold", fontsize=11, color=ARCA_RED)
ax.text(0.7, 1.9, "AP promedio en 10 umbrales: 0.50, 0.55, …, 0.95",
        fontsize=10, color=ARCA_DARK)
ax.text(0.7, 1.4, "Estricto. Exige cajas BIEN alineadas.",
        fontsize=9, color=GRAY, style="italic")
ax.text(0.7, 1.0, "↓ típicamente más bajo. Métrica oficial COCO.",
        fontsize=9, color=ARCA_RED, fontweight="bold")
ax.text(0.7, 0.55, "Es la que usa Ultralytics para escoger 'best.pt'",
        fontsize=8.5, color=GRAY, style="italic")

plt.tight_layout()
fig.savefig(f"{OUT}/fig_map_thresholds.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_map_thresholds.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Guardar y cargar — best.pt journey
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
ax.set_title("El recorrido de un modelo: entrenar → guardar → cargar → predecir",
             fontweight="bold", color=ARCA_DARK, fontsize=13)

# Stage 1: Train
ax.add_patch(FancyBboxPatch((0.3, 2.7), 2.5, 1.5, boxstyle="round,pad=0.05",
                            fc=BLUE, ec="white", lw=2))
ax.text(1.55, 3.7, "Train", ha="center", color="white",
        fontweight="bold", fontsize=12)
ax.text(1.55, 3.2, "model.train(...)", ha="center", color="white",
        fontsize=8.5, fontfamily="monospace")

# Stage 2: Disk artifacts
ax.add_patch(FancyBboxPatch((3.5, 2.4), 3.0, 2.1, boxstyle="round,pad=0.05",
                            fc=ARCA_DARK, ec="white", lw=2))
ax.text(5.0, 4.0, "runs/brain/", ha="center", color="white",
        fontweight="bold", fontsize=10)
artifacts = ["weights/best.pt", "weights/last.pt", "results.png",
             "confusion_matrix.png", "args.yaml"]
for i, a in enumerate(artifacts):
    ax.text(5.0, 3.5 - i*0.25, a, ha="center", color="white",
            fontsize=8, fontfamily="monospace")

# Stage 3: best.pt isolated
ax.add_patch(FancyBboxPatch((7.0, 2.7), 2.5, 1.5, boxstyle="round,pad=0.05",
                            fc=ORANGE, ec="white", lw=2))
ax.text(8.25, 3.7, "best.pt", ha="center", color="white",
        fontweight="bold", fontsize=12)
ax.text(8.25, 3.2, "6 MB · portable", ha="center", color="white",
        fontsize=8.5)

# Stage 4: Load + predict elsewhere
ax.add_patch(FancyBboxPatch((10.0, 2.7), 2.7, 1.5, boxstyle="round,pad=0.05",
                            fc=GREEN, ec="white", lw=2))
ax.text(11.35, 3.7, "Predict", ha="center", color="white",
        fontweight="bold", fontsize=12)
ax.text(11.35, 3.2, "YOLO('best.pt')", ha="center", color="white",
        fontsize=8.5, fontfamily="monospace")

# Arrows top
for x1, x2 in [(2.8, 3.5), (6.5, 7.0), (9.5, 10.0)]:
    ax.annotate("", xy=(x2, 3.45), xytext=(x1, 3.45),
                arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))

# Where to deploy (bottom row)
ax.text(6.5, 1.9, "Una vez tienes best.pt, lo cargas DONDE QUIERAS:",
        ha="center", fontsize=10, color=ARCA_DARK, fontweight="bold")

deploys = [
    (1.0, "Notebook\nnuevo", BLUE),
    (3.7, "Script\nbatch", PURPLE),
    (6.5, "App Gradio /\nStreamlit", GREEN),
    (9.0, "API\nFastAPI", ORANGE),
    (11.5, "Edge /\nMobile", ARCA_RED),
]
for x, label, color in deploys:
    ax.add_patch(FancyBboxPatch((x - 0.9, 0.4), 1.8, 1.1,
                                boxstyle="round,pad=0.05",
                                fc=color, alpha=0.85, ec="white", lw=1.5))
    ax.text(x, 0.95, label, ha="center", va="center",
            color="white", fontweight="bold", fontsize=9)

plt.tight_layout()
fig.savefig(f"{OUT}/fig_save_load.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_save_load.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Detection vs Segmentation API — same shape, different outputs
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

# Left: Detection API
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
ax.set_title("yolo26n.pt  →  results[0].boxes",
             fontweight="bold", color=BLUE, fontsize=12)

ax.add_patch(FancyBboxPatch((0.3, 0.5), 9.4, 6.0, boxstyle="round,pad=0.1",
                            fc=BLUE, alpha=0.08, ec=BLUE, lw=2))
ax.text(5.0, 6.0, "boxes attributes", ha="center", fontweight="bold",
        fontsize=11, color=BLUE)
ax.text(0.7, 5.3, "boxes.xyxy", fontweight="bold", color=ARCA_DARK,
        fontsize=10, fontfamily="monospace")
ax.text(0.7, 4.9, "tensor (N, 4) — [x1, y1, x2, y2] en píxeles",
        color=ARCA_DARK, fontsize=9)
ax.text(0.7, 4.3, "boxes.xywhn", fontweight="bold", color=ARCA_DARK,
        fontsize=10, fontfamily="monospace")
ax.text(0.7, 3.9, "tensor (N, 4) — normalizado [0,1]",
        color=ARCA_DARK, fontsize=9)
ax.text(0.7, 3.3, "boxes.conf", fontweight="bold", color=ARCA_DARK,
        fontsize=10, fontfamily="monospace")
ax.text(0.7, 2.9, "tensor (N,) — confianza por detección",
        color=ARCA_DARK, fontsize=9)
ax.text(0.7, 2.3, "boxes.cls", fontweight="bold", color=ARCA_DARK,
        fontsize=10, fontfamily="monospace")
ax.text(0.7, 1.9, "tensor (N,) — class_id por detección",
        color=ARCA_DARK, fontsize=9)
ax.text(0.7, 1.3, "boxes.id", fontweight="bold", color=ARCA_DARK,
        fontsize=10, fontfamily="monospace")
ax.text(0.7, 0.9, "tensor (N,) — solo si usaste model.track()",
        color=ARCA_DARK, fontsize=9)

# Right: Segmentation API
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
ax.set_title("yolo26n-seg.pt  →  results[0].boxes  +  .masks",
             fontweight="bold", color=GREEN, fontsize=12)

ax.add_patch(FancyBboxPatch((0.3, 0.5), 9.4, 6.0, boxstyle="round,pad=0.1",
                            fc=GREEN, alpha=0.08, ec=GREEN, lw=2))
ax.text(5.0, 6.0, ".boxes + masks attributes", ha="center",
        fontweight="bold", fontsize=11, color=GREEN)
ax.text(0.7, 5.3, "masks.data", fontweight="bold", color=ARCA_DARK,
        fontsize=10, fontfamily="monospace")
ax.text(0.7, 4.9, "tensor (N, H, W) — máscara binaria por objeto",
        color=ARCA_DARK, fontsize=9)
ax.text(0.7, 4.3, "masks.xy", fontweight="bold", color=ARCA_DARK,
        fontsize=10, fontfamily="monospace")
ax.text(0.7, 3.9, "lista de arrays (P, 2) — polígonos en píxeles",
        color=ARCA_DARK, fontsize=9)
ax.text(0.7, 3.3, "masks.xyn", fontweight="bold", color=ARCA_DARK,
        fontsize=10, fontfamily="monospace")
ax.text(0.7, 2.9, "lista de arrays (P, 2) — polígonos normalizados",
        color=ARCA_DARK, fontsize=9)
ax.text(0.7, 2.3, "+ boxes.* (mismos atributos que detection)",
        color=GRAY, fontsize=9, style="italic")
ax.text(0.7, 1.3, "Área del objeto =", fontweight="bold",
        color=ARCA_RED, fontsize=10)
ax.text(0.7, 0.9, "int(masks.data[i].sum())   # pixeles 'encendidos'",
        color=ARCA_DARK, fontsize=9, fontfamily="monospace")

plt.tight_layout()
fig.savefig(f"{OUT}/fig_api_seg.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_api_seg.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Las 80 clases de COCO — agrupadas
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 7.5); ax.axis("off")
ax.set_title("Las 80 clases de COCO (yolo26n-seg ya las reconoce sin entrenar)",
             fontweight="bold", color=ARCA_DARK, fontsize=14)

categories = [
    ("Personas y partes", ["person"], BLUE, 0.3, 6.5),
    ("Vehículos",
     ["bicycle", "car", "motorcycle", "airplane", "bus", "train",
      "truck", "boat"], ORANGE, 0.3, 5.7),
    ("Mobiliario urbano",
     ["traffic light", "fire hydrant", "stop sign", "parking meter",
      "bench"], GRAY, 0.3, 4.9),
    ("Animales",
     ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
      "bear", "zebra", "giraffe"], PURPLE, 0.3, 4.1),
    ("Accesorios",
     ["backpack", "umbrella", "handbag", "tie", "suitcase"],
     PINK, 0.3, 3.3),
    ("Deportes",
     ["frisbee", "skis", "snowboard", "sports ball", "kite",
      "baseball bat", "baseball glove", "skateboard", "surfboard",
      "tennis racket"], CYAN, 0.3, 2.5),
    ("Cocina",
     ["bottle", "wine glass", "cup", "fork", "knife",
      "spoon", "bowl"], ARCA_RED, 0.3, 1.7),
    ("Comida (relevante para conteo)",
     ["banana", "apple", "sandwich", "orange", "broccoli",
      "carrot", "hot dog", "pizza", "donut", "cake"],
     GREEN, 0.3, 0.9),
]

for title, items, color, x, y in categories:
    ax.text(x, y, title, fontweight="bold", color=color, fontsize=11)
    items_text = " · ".join(items)
    ax.text(x, y - 0.3, items_text, fontsize=9, color=ARCA_DARK,
            fontfamily="monospace")

# Note box
ax.add_patch(FancyBboxPatch((9.0, 0.2), 4.8, 1.4, boxstyle="round,pad=0.08",
                            fc=GREEN, alpha=0.12, ec=GREEN, lw=2))
ax.text(11.4, 1.3, "Categorías \"contables\"", ha="center",
        fontweight="bold", color=GREEN, fontsize=11)
ax.text(11.4, 0.9, "person · bottle · cup · banana ·\norange · pizza · sports ball",
        ha="center", color=ARCA_DARK, fontsize=9, fontfamily="monospace")
ax.text(11.4, 0.35, "Ideal para demo de conteo",
        ha="center", color=GRAY, fontsize=8.5, style="italic")

# Missing classes note
ax.add_patch(FancyBboxPatch((9.0, 4.5), 4.8, 2.5, boxstyle="round,pad=0.08",
                            fc=ARCA_RED, alpha=0.12, ec=ARCA_RED, lw=2))
ax.text(11.4, 6.7, "Lo que NO tiene COCO", ha="center",
        fontweight="bold", color=ARCA_RED, fontsize=11)
ax.text(11.4, 6.2, "× pill / píldora", ha="center", color=ARCA_DARK, fontsize=10)
ax.text(11.4, 5.9, "× placa vehicular", ha="center", color=ARCA_DARK, fontsize=10)
ax.text(11.4, 5.6, "× tumor cerebral", ha="center", color=ARCA_DARK, fontsize=10)
ax.text(11.4, 5.3, "× tu producto específico", ha="center", color=ARCA_DARK, fontsize=10)
ax.text(11.4, 4.8, "→ fine-tune obligatorio",
        ha="center", color=ARCA_RED, fontweight="bold", fontsize=10)

plt.tight_layout()
fig.savefig(f"{OUT}/fig_coco_classes.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_coco_classes.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Motivación píldoras — pegadas, conteo difícil
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

# Left: many pills, overlapping — bbox fails
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
ax.set_title("Píldoras pegadas: bboxes solapan",
             fontweight="bold", color=ARCA_RED, fontsize=12)

# Pills as ellipses (random positions, overlapping)
from matplotlib.patches import Ellipse
np.random.seed(42)
pill_coords = []
for _ in range(18):
    x = np.random.uniform(1, 9)
    y = np.random.uniform(1.5, 5.5)
    pill_coords.append((x, y))

for x, y in pill_coords:
    ax.add_patch(Ellipse((x, y), 1.2, 0.6, angle=np.random.uniform(-30, 30),
                         fc="#F5F5F5", ec=ARCA_DARK, lw=1.2))
    # bbox approx
    ax.add_patch(Rectangle((x - 0.7, y - 0.4), 1.4, 0.8,
                            fill=False, edgecolor=ARCA_RED, lw=1, alpha=0.5))

ax.text(5, 0.7, "18 píldoras reales → NMS borra varias\n"
                "→ conteo dice 11", ha="center", fontsize=10,
        color=ARCA_DARK)
ax.text(5, 6.5, "Bounding boxes (rojo)", ha="center",
        fontsize=10, color=ARCA_RED, fontweight="bold")

# Right: masks — one color per instance, no overlap problem
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
ax.set_title("Segmentación: máscaras por píldora",
             fontweight="bold", color=GREEN, fontsize=12)

# Same pills with distinct colors (instance segmentation)
palette = [ARCA_RED, BLUE, GREEN, ORANGE, PURPLE, CYAN, PINK,
           "#84CC16", "#F59E0B", "#06B6D4", "#A855F7", "#10B981",
           "#EF4444", "#3B82F6", "#FB923C", "#6366F1", "#14B8A6",
           "#F472B6"]
for (x, y), color in zip(pill_coords, palette):
    ax.add_patch(Ellipse((x, y), 1.2, 0.6, angle=np.random.uniform(-30, 30),
                         fc=color, ec="white", lw=1.5, alpha=0.92))

ax.text(5, 0.7, "18 máscaras únicas → conteo exacto\n"
                "→ len(masks.data) = 18 ✓", ha="center", fontsize=10,
        color=ARCA_DARK)
ax.text(5, 6.5, "Instance masks (un color por píldora)", ha="center",
        fontsize=10, color=GREEN, fontweight="bold")

plt.tight_layout()
fig.savefig(f"{OUT}/fig_pills_motivacion.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_pills_motivacion.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Pipeline de píldoras — Roboflow → LS → train → app
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 4.5); ax.axis("off")
ax.set_title("Pipeline de la app de conteo de píldoras",
             fontweight="bold", color=ARCA_DARK, fontsize=13)

stages = [
    (0.2, "Roboflow\nUniverse", BLUE,
     "Bajar dataset\n(100 imgs)"),
    (2.5, "Label Studio\n(sin etiquetas)", PURPLE,
     "Subir imágenes\n+ PolygonLabels"),
    (4.8, "Etiquetado\n(equipo)", PINK,
     "Dibujar polígono\npor cada píldora"),
    (7.1, "Bajar\nlabels", BLUE,
     "label-studio-sdk\nformato YOLO seg"),
    (9.4, "Entrenar\nyolo26n-seg", ARCA_RED,
     "10 epochs\n~5 min en T4"),
    (11.7, "App\nGradio", GREEN,
     "img → conteo\n+ visualización"),
]

for i, (x, label, color, body) in enumerate(stages):
    ax.add_patch(FancyBboxPatch((x, 1.8), 2.1, 1.6, boxstyle="round,pad=0.06",
                                fc=color, ec="white", lw=2))
    ax.text(x + 1.05, 2.95, label, ha="center", color="white",
            fontweight="bold", fontsize=10)
    ax.text(x + 1.05, 2.25, body, ha="center", color="white",
            fontsize=8.5)
    if i < len(stages) - 1:
        ax.annotate("", xy=(stages[i+1][0], 2.6), xytext=(x + 2.1, 2.6),
                    arrowprops=dict(arrowstyle="->", color=ARCA_DARK, lw=2))

# Tools row
tools_y = 0.4
for i, (x, _, color, _) in enumerate(stages):
    tools = ["roboflow.Roboflow", "LS API", "Manual\n(humanos)",
             "label_studio_sdk", "ultralytics.YOLO", "gradio.Interface"]
    ax.text(x + 1.05, tools_y, tools[i], ha="center", color=color,
            fontsize=8, fontfamily="monospace", fontweight="bold")
ax.text(0.0, tools_y, "Herramienta:", color=GRAY, fontsize=9, fontweight="bold")

plt.tight_layout()
fig.savefig(f"{OUT}/fig_pill_pipeline.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_pill_pipeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: Confusion Matrix (didáctica)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6.5))
ax.set_xlim(-0.3, 4.5); ax.set_ylim(-0.3, 4.5); ax.axis("off")
ax.set_title("Confusion Matrix — leer el sesgo del modelo",
             fontweight="bold", color=ARCA_DARK, fontsize=13)

# 2x2 grid
data = [[180, 30], [20, 170]]
cell_colors = [[GREEN, ARCA_RED], [ORANGE, GREEN]]
cell_labels = [["TN", "FP"], ["FN", "TP"]]
cell_desc = [["Bien!\n(dijo neg,\nera neg)",
              "Falsa alarma\n(dijo pos,\nera neg)"],
             ["¡PELIGRO!\n(perdió un\ntumor real)",
              "Bien!\n(dijo pos,\nera pos)"]]

for i in range(2):
    for j in range(2):
        x, y = j * 2.0, (1 - i) * 2.0
        ax.add_patch(Rectangle((x, y), 2.0, 2.0, fc=cell_colors[i][j],
                                alpha=0.25, ec=cell_colors[i][j], lw=2))
        ax.text(x + 1.0, y + 1.5, str(data[i][j]), ha="center", va="center",
                fontsize=24, fontweight="bold", color=cell_colors[i][j])
        ax.text(x + 1.0, y + 1.0, cell_labels[i][j], ha="center", va="center",
                fontsize=11, color=ARCA_DARK, fontweight="bold")
        ax.text(x + 1.0, y + 0.4, cell_desc[i][j], ha="center", va="center",
                fontsize=8.5, color=ARCA_DARK, style="italic")

# Labels
ax.text(2.0, 4.3, "PREDICCIÓN DEL MODELO", ha="center", fontweight="bold",
        fontsize=11, color=ARCA_DARK)
ax.text(1.0, 4.05, "negative", ha="center", fontsize=10, color=ARCA_DARK)
ax.text(3.0, 4.05, "positive", ha="center", fontsize=10, color=ARCA_DARK)

ax.text(-0.5, 2.0, "REAL", ha="center", va="center", fontweight="bold",
        fontsize=11, color=ARCA_DARK, rotation=90)
ax.text(-0.25, 3.0, "negative", ha="center", va="center", fontsize=10,
        color=ARCA_DARK, rotation=90)
ax.text(-0.25, 1.0, "positive", ha="center", va="center", fontsize=10,
        color=ARCA_DARK, rotation=90)

# Calculations
ax.text(4.7, 3.0, "Recall (S) = TP/(TP+FN)\n= 170 / 190 = 0.89",
        fontsize=9, color=ARCA_RED, fontweight="bold")
ax.text(4.7, 1.6, "Precision = TP/(TP+FP)\n= 170 / 200 = 0.85",
        fontsize=9, color=BLUE, fontweight="bold")
ax.text(4.7, 0.4, "F1 = 2·P·R/(P+R) = 0.87",
        fontsize=9, color=GREEN, fontweight="bold")

plt.tight_layout()
fig.savefig(f"{OUT}/fig_confusion.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_confusion.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10: Gradio app mockup
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("App Gradio: contar píldoras desde una imagen",
             fontweight="bold", color=ARCA_DARK, fontsize=13)

# Browser frame
ax.add_patch(FancyBboxPatch((0.3, 0.3), 11.4, 5.3, boxstyle="round,pad=0.05",
                            fc="white", ec=GRAY, lw=2))
# Browser top bar
ax.add_patch(Rectangle((0.3, 5.0), 11.4, 0.6, fc="#F3F4F6", ec="none"))
ax.add_patch(Circle((0.7, 5.3), 0.10, fc="#EF4444", ec="none"))
ax.add_patch(Circle((1.0, 5.3), 0.10, fc="#F59E0B", ec="none"))
ax.add_patch(Circle((1.3, 5.3), 0.10, fc="#10B981", ec="none"))
ax.text(6.0, 5.3, "https://xxxxx.gradio.live", ha="center", va="center",
        fontsize=9, color=GRAY, fontfamily="monospace")

# Title bar
ax.add_patch(Rectangle((0.3, 4.4), 11.4, 0.6, fc=ARCA_RED, ec="none"))
ax.text(6.0, 4.7, "Contador de Píldoras — yolo26n-seg fine-tuned",
        ha="center", va="center", fontsize=12, color="white",
        fontweight="bold")

# Left: input upload
ax.add_patch(FancyBboxPatch((0.7, 0.8), 5.0, 3.4, boxstyle="round,pad=0.05",
                            fc=LIGHT, ec=GRAY, lw=1.5, linestyle="--"))
ax.text(3.2, 3.8, "INPUT", ha="center", fontweight="bold",
        color=GRAY, fontsize=10)
ax.add_patch(Rectangle((2.3, 2.0), 1.8, 1.2, fc="white", ec=GRAY, lw=2))
ax.add_patch(Circle((3.2, 2.6), 0.35, fc="none", ec=GRAY, lw=2))
ax.add_patch(Circle((3.2, 2.6), 0.18, fc=GRAY, ec="none"))
ax.text(3.2, 1.4, "Subir imagen del blister", ha="center",
        fontsize=10, color=ARCA_DARK)
ax.text(3.2, 1.0, "(drag & drop o click)", ha="center",
        fontsize=8, color=GRAY, style="italic")

# Right: output
ax.add_patch(FancyBboxPatch((6.3, 0.8), 5.0, 3.4, boxstyle="round,pad=0.05",
                            fc="#FEF9C3", ec=ORANGE, lw=1.5))
ax.text(8.8, 3.8, "OUTPUT", ha="center", fontweight="bold",
        color=ORANGE, fontsize=10)
# Annotated image area
ax.add_patch(Rectangle((6.6, 1.6), 4.4, 2.0, fc="#FEF3C7", ec=GRAY, lw=1))
# Sample pills colored
for i, color in enumerate(palette[:12]):
    x = 6.8 + (i % 6) * 0.7
    y = 1.8 + (i // 6) * 0.7
    ax.add_patch(Ellipse((x, y), 0.5, 0.3, fc=color, ec="white", lw=1))

ax.text(8.8, 1.1, "Píldoras detectadas: 12", ha="center",
        fontweight="bold", fontsize=13, color=ARCA_RED)

plt.tight_layout()
fig.savefig(f"{OUT}/fig_gradio_mockup.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_gradio_mockup.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 11: Conf slider effect (recall vs precision por umbral)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))

conf = np.linspace(0.05, 0.95, 100)
recall_curve = 1 / (1 + np.exp(8 * (conf - 0.55)))
precision_curve = 1 / (1 + np.exp(-10 * (conf - 0.30)))
f1_curve = 2 * recall_curve * precision_curve / (recall_curve + precision_curve + 1e-9)

ax.plot(conf, recall_curve, color=ARCA_RED, lw=2.5, label="Recall")
ax.plot(conf, precision_curve, color=BLUE, lw=2.5, label="Precision")
ax.plot(conf, f1_curve, color=GREEN, lw=2.5, linestyle="--", label="F1")

# Vertical lines marking common operating points
ax.axvline(0.15, color=ARCA_RED, linestyle=":", alpha=0.7)
ax.text(0.15, 1.05, "conf=0.15\n(tumor)", ha="center",
        color=ARCA_RED, fontsize=9, fontweight="bold")

ax.axvline(0.5, color=BLUE, linestyle=":", alpha=0.7)
ax.text(0.5, 1.05, "conf=0.5\n(placa)", ha="center",
        color=BLUE, fontsize=9, fontweight="bold")

best_idx = np.argmax(f1_curve)
ax.axvline(conf[best_idx], color=GREEN, linestyle=":", alpha=0.7)
ax.text(conf[best_idx], 1.05, f"conf={conf[best_idx]:.2f}\n(F1 óptimo)",
        ha="center", color=GREEN, fontsize=9, fontweight="bold")

ax.set_xlabel("Umbral de confianza (conf)", fontsize=11)
ax.set_ylabel("Métrica", fontsize=11)
ax.set_title("Cómo el conf threshold mueve precision, recall y F1",
             fontweight="bold", color=ARCA_DARK, fontsize=12)
ax.legend(loc="lower center", fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_xlim(0, 1.0)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT}/fig_conf_slider.png", dpi=180, bbox_inches="tight")
plt.close()
print("OK fig_conf_slider.png")


print("\nTodas las figuras OK.")
