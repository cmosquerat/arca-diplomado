"""Genera todas las figuras para la presentación de Clase 29 (Autoencoders).

Las figuras se guardan en el mismo directorio que este script. Usar matplotlib
con colores corporativos Arca/UDLA. Los datos son sintéticos o simulados para
no requerir entrenar un modelo aquí — las cifras se muestran como ilustración
del concepto.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                 Circle, FancyArrow, ConnectionPatch, Polygon)
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path(__file__).parent

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

ARCA_RED = "#C82B40"
ARCA_DARK = "#6B1525"
GREEN = "#16A34A"
BLUE = "#2563EB"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
GRAY = "#9CA3AF"
LIGHT_GRAY = "#EBEBEB"
textDark = "#2D2D2D"


def save(name):
    plt.savefig(OUT / name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {name}")


# ═════════════════════════════════════════════════════════════════════════════
# fig_hero.png — concepto principal del autoencoder
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 4.5); ax.axis('off')

# Título arriba
ax.text(6.5, 4.1,
        'Codificación automática en un espacio latente',
        ha='center', fontsize=15, fontweight='bold', color=ARCA_DARK)

# INPUT
ax.add_patch(Rectangle((0.4, 1.2), 1.7, 1.7, fc=LIGHT_GRAY,
                        ec='black', lw=1.5))
ax.text(1.25, 2.05, 'INPUT', ha='center', fontsize=11, fontweight='bold')
ax.text(1.25, 1.65, '28×28', ha='center', fontsize=9, color=GRAY)
ax.text(1.25, 0.7, 'dato original', ha='center', fontsize=10,
        style='italic', color=GRAY)

# ENCODER (trapezoid converging right)
encoder_poly = Polygon([[2.5, 0.7], [2.5, 3.4], [4.4, 2.6], [4.4, 1.5]],
                        closed=True, fc=ARCA_RED, ec=ARCA_DARK, lw=1.5, alpha=0.85)
ax.add_patch(encoder_poly)
ax.text(3.4, 2.05, 'ENCODER', ha='center', color='white',
        fontsize=11, fontweight='bold')

# LATENTE
ax.add_patch(FancyBboxPatch((4.8, 1.5), 1.9, 1.1,
                              boxstyle="round,pad=0.05",
                              fc=PURPLE, ec=ARCA_DARK, lw=1.5, alpha=0.9))
ax.text(5.75, 2.25, 'LATENTE', ha='center', color='white',
        fontsize=12, fontweight='bold')
ax.text(5.75, 1.8, 'z ∈ ℝ³²', ha='center', color='white',
        fontsize=10, family='monospace')
ax.text(5.75, 0.7, 'nuevo sistema\nde coordenadas',
        ha='center', fontsize=10, style='italic', color=PURPLE)

# DECODER (trapezoid diverging right)
decoder_poly = Polygon([[7.1, 1.5], [7.1, 2.6], [9.0, 3.4], [9.0, 0.7]],
                        closed=True, fc=ARCA_RED, ec=ARCA_DARK, lw=1.5, alpha=0.85)
ax.add_patch(decoder_poly)
ax.text(8.05, 2.05, 'DECODER', ha='center', color='white',
        fontsize=11, fontweight='bold')

# OUTPUT
ax.add_patch(Rectangle((9.4, 1.2), 1.7, 1.7, fc=LIGHT_GRAY,
                        ec='black', lw=1.5))
ax.text(10.25, 2.05, '≈ INPUT', ha='center', fontsize=11, fontweight='bold')
ax.text(10.25, 1.65, '28×28', ha='center', fontsize=9, color=GRAY)
ax.text(10.25, 0.7, 'examen, no producto',
        ha='center', fontsize=10, style='italic', color=GRAY)

# Flechas
for (x0, x1) in [(2.1, 2.45), (4.45, 4.75), (6.7, 7.05), (9.05, 9.35)]:
    ax.annotate('', xy=(x1, 2.05), xytext=(x0, 2.05),
                 arrowprops=dict(arrowstyle='->', lw=1.8, color='black'))

# El producto real
ax.annotate('', xy=(5.75, 3.6), xytext=(5.75, 2.8),
             arrowprops=dict(arrowstyle='->', lw=1.5, color=PURPLE))
ax.text(7.6, 3.7, 'el activo que nos llevamos',
        fontsize=10, color=PURPLE, fontweight='bold')

plt.tight_layout()
save('fig_hero.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_motivacion_labels.png — costo de etiquetado por tarea
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4.5))

tareas = ['Clasificación\n(imagen → clase)',
          'Detección\n(bounding boxes)',
          'Segmentación\n(polígonos)',
          'OCR / texto\n(transcripción)']
img_por_hora = [200, 60, 12, 35]
costo_relativo = [1, 3.3, 16.7, 5.7]

colores = [GREEN, BLUE, ARCA_RED, ORANGE]
bars = ax.barh(tareas, img_por_hora, color=colores, alpha=0.85,
                edgecolor='black', linewidth=1)

for bar, n, c in zip(bars, img_por_hora, costo_relativo):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
             f'{n} img/h   (costo relativo: {c:.1f}x)',
             va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Imágenes etiquetadas por hora-humano (referencia industria)',
               fontsize=11)
ax.set_xlim(0, 280)
ax.set_title('Etiquetar es caro — y se vuelve más caro mientras más fino el label',
              fontsize=12, fontweight='bold', color=ARCA_DARK, pad=15)
ax.grid(axis='x', alpha=0.3)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.tight_layout()
save('fig_motivacion_labels.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_familia.png — PCA, Word2Vec, t-SNE, AE como familia
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis('off')

ax.text(5.5, 4.6,
         'Todos aprenden un sistema de coordenadas para los datos',
         ha='center', fontsize=14, fontweight='bold', color=ARCA_DARK)
ax.text(5.5, 4.15,
         'Lo que cambia es la familia funcional y el régimen de entrenamiento',
         ha='center', fontsize=11, style='italic', color=GRAY)

items = [
    ('PCA', 'Tabular\n(clase 21)', 'Lineal\n(SVD)', GREEN, 0.5),
    ('Word2Vec', 'Palabras', 'Skip-gram\n(predicción)', BLUE, 3.0),
    ('t-SNE / UMAP', 'Visualización 2D\n(clase 21)', 'No paramétrico\n(vecindades)', ORANGE, 5.5),
    ('Autoencoder', 'Cualquiera\n(clase 29)', 'Red + reconstrucción\nno lineal', ARCA_RED, 8.0),
]

for nombre, dato, como, color, x in items:
    ax.add_patch(FancyBboxPatch((x, 1.0), 2.3, 2.4,
                                  boxstyle="round,pad=0.05",
                                  fc=color, ec='black', lw=1.5, alpha=0.85))
    ax.text(x + 1.15, 3.0, nombre, ha='center', color='white',
             fontsize=13, fontweight='bold')
    ax.text(x + 1.15, 2.35, dato, ha='center', color='white', fontsize=10)
    ax.text(x + 1.15, 1.45, como, ha='center', color='white', fontsize=9, style='italic')

ax.text(5.5, 0.4, '→ el autoencoder generaliza la idea a cualquier tipo de dato y a relaciones no lineales',
         ha='center', fontsize=10, color=ARCA_DARK, fontweight='bold')

plt.tight_layout()
save('fig_familia.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_arquitectura.png — encoder, latente, decoder (arquitectura técnica)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')

# Input layer
ax.add_patch(Rectangle((0.3, 0.5), 0.5, 4, fc=LIGHT_GRAY, ec='black'))
ax.text(0.55, 4.7, '784', ha='center', fontsize=10, fontweight='bold')
ax.text(0.55, 0.2, 'input', ha='center', fontsize=9, color=GRAY)

# Encoder: 128
ax.add_patch(Rectangle((1.8, 1.2), 0.5, 2.6, fc=ARCA_RED, ec='black', alpha=0.85))
ax.text(2.05, 4.0, '128', ha='center', fontsize=10, fontweight='bold')

# Latente: 32
ax.add_patch(Rectangle((3.4, 1.9), 0.5, 1.2, fc=PURPLE, ec='black', alpha=0.95))
ax.text(3.65, 3.3, '32', ha='center', fontsize=11, fontweight='bold', color=PURPLE)
ax.text(3.65, 0.2, 'latente z', ha='center', fontsize=9, color=PURPLE,
         fontweight='bold')

# Decoder: 128
ax.add_patch(Rectangle((5.0, 1.2), 0.5, 2.6, fc=ARCA_RED, ec='black', alpha=0.85))
ax.text(5.25, 4.0, '128', ha='center', fontsize=10, fontweight='bold')

# Output
ax.add_patch(Rectangle((6.6, 0.5), 0.5, 4, fc=LIGHT_GRAY, ec='black'))
ax.text(6.85, 4.7, '784', ha='center', fontsize=10, fontweight='bold')
ax.text(6.85, 0.2, 'output ≈ input', ha='center', fontsize=9, color=GRAY)

# Curly braces (etiquetas encoder/decoder)
ax.annotate('', xy=(3.7, 4.7), xytext=(0.8, 4.7),
             arrowprops=dict(arrowstyle='-[', lw=1.5, color=ARCA_DARK))
ax.text(2.25, 4.85, 'ENCODER (comprime)', ha='center',
         fontsize=10, color=ARCA_DARK, fontweight='bold')

ax.annotate('', xy=(7.1, 4.7), xytext=(3.9, 4.7),
             arrowprops=dict(arrowstyle='-[', lw=1.5, color=ARCA_DARK))
ax.text(5.5, 4.85, 'DECODER (reconstruye)', ha='center',
         fontsize=10, color=ARCA_DARK, fontweight='bold')

# Caja explicación a la derecha
exp_x = 8.0
ax.add_patch(FancyBboxPatch((exp_x, 0.4), 5.6, 4.2,
                              boxstyle="round,pad=0.1",
                              fc=LIGHT_GRAY, ec=ARCA_DARK, lw=1.5))
ax.text(exp_x + 2.8, 4.2, 'Las tres piezas',
         ha='center', fontsize=12, fontweight='bold', color=ARCA_DARK)

textos = [
    ('Encoder', 'traduce el dato a coordenadas latentes', ARCA_RED),
    ('Latente z', 'el dato escrito en el sistema nuevo', PURPLE),
    ('Decoder', 'recupera el dato desde el latente', ARCA_RED),
    ('', '', None),
    ('Loss', 'MSE(input, decoder(encoder(input)))', BLUE),
    ('Régimen', 'auto-supervisado — sin etiquetas', GREEN),
]
y = 3.6
for nombre, desc, color in textos:
    if nombre:
        ax.text(exp_x + 0.2, y, nombre + ':',
                 fontsize=10, fontweight='bold', color=color)
        ax.text(exp_x + 1.3, y, desc, fontsize=10, color='black')
    y -= 0.5

plt.tight_layout()
save('fig_arquitectura.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_examen.png — reconstrucción es el examen, no el producto
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 13); ax.set_ylim(0, 4.5); ax.axis('off')

# Diagrama similar al hero pero resaltando "el producto"
ax.add_patch(Rectangle((0.5, 1.3), 1.4, 1.4, fc=LIGHT_GRAY, ec='black', lw=1.5))
ax.text(1.2, 2.0, 'INPUT', ha='center', fontweight='bold')

ax.add_patch(Polygon([[2.3, 0.9], [2.3, 3.1], [4.0, 2.4], [4.0, 1.6]],
                       closed=True, fc=ARCA_RED, ec=ARCA_DARK, alpha=0.6))
ax.text(3.15, 2.0, 'ENCODER', ha='center', color='white', fontweight='bold')

# Latente DESTACADO con halo
ax.add_patch(FancyBboxPatch((4.4, 1.5), 1.7, 1.1,
                              boxstyle="round,pad=0.1",
                              fc=PURPLE, ec=ARCA_DARK, lw=3))
ax.text(5.25, 2.05, 'LATENTE', ha='center', color='white',
         fontsize=12, fontweight='bold')

ax.add_patch(Polygon([[6.5, 1.6], [6.5, 2.4], [8.2, 3.1], [8.2, 0.9]],
                       closed=True, fc=ARCA_RED, ec=ARCA_DARK, alpha=0.6))
ax.text(7.35, 2.0, 'DECODER', ha='center', color='white', fontweight='bold')

ax.add_patch(Rectangle((8.6, 1.3), 1.4, 1.4, fc=LIGHT_GRAY, ec='black', lw=1.5))
ax.text(9.3, 2.0, '≈ INPUT', ha='center', fontweight='bold')

# Anotaciones
ax.annotate('PRODUCTO\n(lo que usamos)',
             xy=(5.25, 2.7), xytext=(5.25, 3.9),
             ha='center', fontsize=11, fontweight='bold', color=PURPLE,
             arrowprops=dict(arrowstyle='->', color=PURPLE, lw=2))

ax.annotate('EXAMEN\n(verifica que el latente describe)',
             xy=(9.3, 1.3), xytext=(11.0, 0.4),
             ha='center', fontsize=10, fontweight='bold', color=GRAY,
             arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

# Flechas
for (x0, x1) in [(1.9, 2.25), (4.05, 4.35), (6.1, 6.45), (8.25, 8.55)]:
    ax.annotate('', xy=(x1, 2.0), xytext=(x0, 2.0),
                 arrowprops=dict(arrowstyle='->', lw=1.8))

plt.tight_layout()
save('fig_examen.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_latente_2d.png — espacio latente coloreado por dígito
# ═════════════════════════════════════════════════════════════════════════════
# Simulamos un espacio latente plausible: 10 clusters Gaussianos en 2D
np.random.seed(42)
centers = np.array([
    [ 0.0,  3.0], [-2.5,  1.5], [ 2.5,  1.8],
    [-3.0, -1.0], [ 0.5, -2.5], [ 3.0, -1.0],
    [-1.0,  0.5], [ 1.5, -0.5], [ 0.0,  1.0], [ 2.0,  3.0],
])
points = []
labels = []
for k, c in enumerate(centers):
    n = 800
    pts = np.random.normal(loc=c, scale=0.6, size=(n, 2))
    points.append(pts)
    labels.append(np.full(n, k))
points = np.vstack(points)
labels = np.concatenate(labels)

fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='tab10', s=6, alpha=0.7)
ax.set_xlabel('z₁  (coordenada latente 1)', fontsize=11)
ax.set_ylabel('z₂  (coordenada latente 2)', fontsize=11)
ax.set_title('Espacio latente 2D de un AE entrenado en MNIST\nsin haber visto las etiquetas',
              fontsize=12, fontweight='bold', color=ARCA_DARK, pad=12)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.grid(alpha=0.2)

# Anotaciones en algunos clusters
ax.annotate('cluster del "0"', xy=(0.0, 3.0), xytext=(-3.5, 4.5),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('clusters parecidos\nse pegan (4 ↔ 9)', xy=(-1.0, 0.5),
             xytext=(-5.5, -3.5), fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='black'))

plt.colorbar(sc, ticks=range(10), label='Dígito real (no usado en entrenamiento)')
plt.tight_layout()
save('fig_latente_2d.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_pca_vs_ae.png — comparación
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# PCA simulado: clusters más mezclados, distribución elongada
np.random.seed(7)
pca_pts = []
pca_lbl = []
for k in range(10):
    cx = np.cos(k * 0.628) * 3 + np.random.normal(0, 0.5)
    cy = np.sin(k * 0.628) * 1.5 + np.random.normal(0, 0.3)
    pts = np.random.normal(loc=[cx, cy], scale=[1.3, 0.9], size=(800, 2))
    pca_pts.append(pts)
    pca_lbl.append(np.full(800, k))
pca_pts = np.vstack(pca_pts)
pca_lbl = np.concatenate(pca_lbl)
sc1 = axes[0].scatter(pca_pts[:, 0], pca_pts[:, 1],
                       c=pca_lbl, cmap='tab10', s=5, alpha=0.6)
axes[0].set_title('PCA (lineal)', fontsize=13, fontweight='bold', color=ARCA_DARK)
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
axes[0].grid(alpha=0.2)

# AE: usamos los clusters de antes (más separados)
sc2 = axes[1].scatter(points[:, 0], points[:, 1],
                       c=labels, cmap='tab10', s=5, alpha=0.6)
axes[1].set_title('Autoencoder (no lineal)', fontsize=13, fontweight='bold', color=ARCA_DARK)
axes[1].set_xlabel('z₁'); axes[1].set_ylabel('z₂')
axes[1].grid(alpha=0.2)

for ax in axes:
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

fig.suptitle('Misma data (MNIST) → dos sistemas de coordenadas 2D distintos',
              fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
save('fig_pca_vs_ae.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_conv_ae.png — diagrama limpio del Conv Autoencoder
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11.5, 4.5))
ax.set_xlim(-0.3, 10.5); ax.set_ylim(-2.3, 3.2); ax.axis('off')

# Cinco cajas: alturas decrecientes hacia el bottleneck y crecientes después
boxes_config = [
    (0.7, 2.6, LIGHT_GRAY, '28×28×1', 'input',      'black'),
    (2.7, 1.9, ARCA_RED,   '14×14×16', None,        None),
    (4.7, 1.0, PURPLE,     '7×7×8',   'bottleneck', PURPLE),
    (6.7, 1.9, ARCA_RED,   '14×14×8',  None,        None),
    (8.7, 2.6, LIGHT_GRAY, '28×28×1', 'output',     'black'),
]

box_width = 0.85
for x, h, color, dim, role, role_color in boxes_config:
    edge = ARCA_DARK if color != PURPLE else PURPLE
    lw = 2.5 if color == PURPLE else 1.5
    ax.add_patch(Rectangle((x - box_width / 2, -h / 2), box_width, h,
                            fc=color, ec=edge, lw=lw, alpha=0.92))
    # Dimensión debajo
    ax.text(x, -h / 2 - 0.25, dim, ha='center', va='top',
             fontsize=10, color=ARCA_DARK, fontweight='bold')
    # Rol (input/output/bottleneck) arriba o abajo
    if role == 'bottleneck':
        ax.text(x, -h / 2 - 0.75, role, ha='center', va='top',
                 fontsize=11, color=role_color, fontweight='bold',
                 style='italic')
    elif role:
        ax.text(x, h / 2 + 0.2, role, ha='center', va='bottom',
                 fontsize=11, color=role_color, fontweight='bold')

# Flechas con etiquetas de operación
op_labels = ['Conv2D\n+ MaxPool', 'Conv2D\n+ MaxPool',
             'Conv2D\n+ UpSampling', 'Conv2D\n+ UpSampling']
for i, op in enumerate(op_labels):
    x_from = boxes_config[i][0] + box_width / 2 + 0.05
    x_to = boxes_config[i + 1][0] - box_width / 2 - 0.05
    ax.annotate('', xy=(x_to, 0), xytext=(x_from, 0),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color=ARCA_DARK))
    ax.text((x_from + x_to) / 2, 0.55, op, ha='center', va='bottom',
             fontsize=8, color=textDark if 'Pool' in op else textDark)

# Brackets ENCODER (cubre input + e1 + bottleneck partial — encoder produce el bottleneck)
ax.plot([0.15, 4.7], [1.8, 1.8], color=ARCA_DARK, lw=1.5)
ax.plot([0.15, 0.15], [1.7, 1.8], color=ARCA_DARK, lw=1.5)
ax.plot([4.7, 4.7], [1.7, 1.8], color=ARCA_DARK, lw=1.5)
ax.text(2.42, 2.0, 'ENCODER', ha='center', va='bottom',
         fontsize=12, fontweight='bold', color=ARCA_DARK)

# Brackets DECODER (cubre bottleneck partial + d1 + output)
ax.plot([4.7, 9.25], [-1.8, -1.8], color=ARCA_DARK, lw=1.5)
ax.plot([4.7, 4.7], [-1.7, -1.8], color=ARCA_DARK, lw=1.5)
ax.plot([9.25, 9.25], [-1.7, -1.8], color=ARCA_DARK, lw=1.5)
ax.text(6.97, -2.05, 'DECODER', ha='center', va='top',
         fontsize=12, fontweight='bold', color=ARCA_DARK)

textDark = "#2D2D2D"  # asegurar disponible
plt.tight_layout()
save('fig_conv_ae.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_denoising.png — concepto del denoising
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.axis('off')

# X + noise → AE → X
ax.add_patch(Rectangle((0.5, 1.3), 1.6, 1.6, fc=LIGHT_GRAY, ec='black'))
ax.text(1.3, 2.1, 'INPUT', ha='center', fontweight='bold')
ax.text(1.3, 1.7, '+ ruido', ha='center', fontsize=10, color=ARCA_RED,
         fontweight='bold')

ax.annotate('', xy=(2.9, 2.1), xytext=(2.2, 2.1),
             arrowprops=dict(arrowstyle='->', lw=2))

ax.add_patch(FancyBboxPatch((3.0, 1.0), 3.5, 2.2,
                              boxstyle="round,pad=0.1",
                              fc=PURPLE, ec=ARCA_DARK, lw=1.5, alpha=0.85))
ax.text(4.75, 2.5, 'AUTOENCODER', ha='center', color='white',
         fontsize=11, fontweight='bold')
ax.text(4.75, 1.9, 'latente describe', ha='center', color='white',
         fontsize=10, style='italic')
ax.text(4.75, 1.5, 'solo señal estructurada', ha='center', color='white',
         fontsize=10, style='italic')

ax.annotate('', xy=(7.2, 2.1), xytext=(6.6, 2.1),
             arrowprops=dict(arrowstyle='->', lw=2))

ax.add_patch(Rectangle((7.4, 1.3), 1.6, 1.6, fc=LIGHT_GRAY, ec='black'))
ax.text(8.2, 2.1, 'OUTPUT', ha='center', fontweight='bold')
ax.text(8.2, 1.7, 'limpio', ha='center', fontsize=10, color=GREEN,
         fontweight='bold')

# Etiquetas debajo
ax.text(4.75, 0.4,
         'Entrenamiento:  input = X + ruido,   target = X',
         ha='center', fontsize=11, fontweight='bold', color=ARCA_DARK)

# A la derecha: por qué funciona
ax.add_patch(FancyBboxPatch((9.6, 0.5), 3.3, 3.0,
                              boxstyle="round,pad=0.1",
                              fc=LIGHT_GRAY, ec=ARCA_DARK, lw=1.5))
ax.text(11.25, 3.1, 'Por qué funciona',
         ha='center', fontsize=11, fontweight='bold', color=ARCA_DARK)
ax.text(9.75, 2.5, '• Ruido = incompresible',
         fontsize=10)
ax.text(9.75, 2.0, '  (no hay redundancia)',
         fontsize=9, color=GRAY, style='italic')
ax.text(9.75, 1.4, '• Señal = comprimible',
         fontsize=10)
ax.text(9.75, 0.9, '  (bordes, simetrías)',
         fontsize=9, color=GRAY, style='italic')

plt.tight_layout()
save('fig_denoising.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_anomaly_concept.png — concepto detección de anomalías
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis('off')

ax.text(6.5, 4.7,
         '1. Entrenar SOLO con datos del régimen normal',
         ha='center', fontsize=12, fontweight='bold', color=GREEN)

# Train phase
ax.add_patch(Rectangle((0.5, 2.5), 1.5, 1.3, fc=GREEN, ec='black', alpha=0.7))
ax.text(1.25, 3.15, 'DATOS\nNORMALES', ha='center', color='white',
         fontsize=10, fontweight='bold')

ax.annotate('', xy=(2.6, 3.15), xytext=(2.1, 3.15),
             arrowprops=dict(arrowstyle='->', lw=2))

ax.add_patch(FancyBboxPatch((2.7, 2.3), 2.2, 1.7,
                              boxstyle="round,pad=0.1",
                              fc=PURPLE, ec=ARCA_DARK, lw=1.5, alpha=0.85))
ax.text(3.8, 3.25, 'AE entrena', ha='center', color='white',
         fontsize=10, fontweight='bold')
ax.text(3.8, 2.75, 'espacio latente', ha='center', color='white',
         fontsize=9, style='italic')
ax.text(3.8, 2.5, 'para la población normal', ha='center', color='white',
         fontsize=9, style='italic')

# Inference phase
ax.text(6.5, 1.9,
         '2. En producción: medir reconstruction error',
         ha='center', fontsize=12, fontweight='bold', color=ARCA_DARK)

# Normal case
ax.add_patch(Rectangle((0.5, 0.4), 1.0, 1.0, fc=GREEN, ec='black', alpha=0.4))
ax.text(1.0, 0.9, 'normal\nnuevo', ha='center', fontsize=9)
ax.annotate('', xy=(2.4, 0.9), xytext=(1.6, 0.9),
             arrowprops=dict(arrowstyle='->', lw=1.5))
ax.add_patch(Rectangle((2.5, 0.5), 1.5, 0.8, fc=PURPLE, ec='black', alpha=0.5))
ax.text(3.25, 0.9, 'AE entrenado', ha='center', fontsize=9, color='white')
ax.annotate('', xy=(4.9, 0.9), xytext=(4.1, 0.9),
             arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(5.5, 0.9, 'error BAJO ✓', fontsize=10, fontweight='bold', color=GREEN)

# Anomaly case
ax.add_patch(Rectangle((7.5, 0.4), 1.0, 1.0, fc=ARCA_RED, ec='black', alpha=0.5))
ax.text(8.0, 0.9, 'anomalía', ha='center', fontsize=9, color='white')
ax.annotate('', xy=(9.4, 0.9), xytext=(8.6, 0.9),
             arrowprops=dict(arrowstyle='->', lw=1.5))
ax.add_patch(Rectangle((9.5, 0.5), 1.5, 0.8, fc=PURPLE, ec='black', alpha=0.5))
ax.text(10.25, 0.9, 'AE entrenado', ha='center', fontsize=9, color='white')
ax.annotate('', xy=(11.9, 0.9), xytext=(11.1, 0.9),
             arrowprops=dict(arrowstyle='->', lw=1.5))
ax.text(12.4, 0.9, 'error ALTO ⚠', fontsize=10, fontweight='bold', color=ARCA_RED)

# Mensaje clave
ax.text(6.5, 4.15,
         'El latente solo sabe describir lo que vio →  lo que no encaja produce error de reconstrucción alto',
         ha='center', fontsize=10, style='italic', color=GRAY)

plt.tight_layout()
save('fig_anomaly_concept.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_anomaly_hist.png — histograma errores normal vs anomalía
# ═════════════════════════════════════════════════════════════════════════════
np.random.seed(0)
err_normal = np.random.gamma(2, 0.01, 5000)
err_anom = np.random.gamma(8, 0.018, 5000) + 0.02

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(err_normal, bins=60, alpha=0.7, color=GREEN,
        label='Normal (no anómalo)', density=True)
ax.hist(err_anom, bins=60, alpha=0.7, color=ARCA_RED,
        label='Anómalo', density=True)

# Threshold line
threshold = np.percentile(err_normal, 95)
ax.axvline(threshold, color='black', ls='--', lw=2,
            label=f'Threshold (p95 de normales)')
ax.text(threshold + 0.005, ax.get_ylim()[1] * 0.85,
         '→ todo a la derecha\nse marca como anomalía',
         fontsize=10, fontweight='bold')

ax.set_xlabel('Reconstruction error (MSE)', fontsize=11)
ax.set_ylabel('Densidad', fontsize=11)
ax.set_title('Distribución de errores — el threshold se elige sobre la curva normal',
              fontsize=12, fontweight='bold', color=ARCA_DARK)
ax.legend(loc='upper right', fontsize=11)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.grid(alpha=0.2)

plt.tight_layout()
save('fig_anomaly_hist.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_anomaly_arca.png — casos de negocio Arca
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis('off')

ax.text(5.5, 4.85,
         'Casos donde la anomalía es rara y el "normal" abunda',
         ha='center', fontsize=13, fontweight='bold', color=ARCA_DARK)

casos = [
    ('Línea de\nembotellado', 'botella OK',
     'tapa torcida\netiqueta movida', GREEN),
    ('Inspección\nde latas', 'lata limpia',
     'abolladura\nrasguño', BLUE),
    ('Mantenimiento\npredictivo', 'vibración normal',
     'patrón pre-falla', ORANGE),
    ('Transacciones\nde TPV', 'compra típica',
     'fraude / error', PURPLE),
]

for i, (titulo, normal, anom, color) in enumerate(casos):
    x = 0.3 + i * 2.7
    ax.add_patch(FancyBboxPatch((x, 0.4), 2.4, 4.0,
                                  boxstyle="round,pad=0.05",
                                  fc=color, ec='black', lw=1.5, alpha=0.85))
    ax.text(x + 1.2, 3.7, titulo, ha='center', color='white',
             fontsize=11, fontweight='bold')

    ax.text(x + 1.2, 2.95, 'Normal:', ha='center', color='white',
             fontsize=9, fontweight='bold')
    ax.text(x + 1.2, 2.55, normal, ha='center', color='white', fontsize=9)

    ax.text(x + 1.2, 1.75, 'Anomalía:', ha='center', color='white',
             fontsize=9, fontweight='bold')
    ax.text(x + 1.2, 1.25, anom, ha='center', color='white', fontsize=9)

plt.tight_layout()
save('fig_anomaly_arca.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_tabular.png — AE tabular (telemetría industrial)
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Izquierda: 2 features mostrando régimen normal con 2 modos + anomalías
np.random.seed(42)
n1 = 800
modo1 = np.random.normal([50, 120], [4, 8], size=(n1, 2))
n2 = 600
modo2 = np.random.normal([65, 140], [4, 7], size=(n2, 2))
anoms = np.column_stack([np.random.normal(55, 5, 100),
                          np.random.normal(200, 15, 100)])

axes[0].scatter(modo1[:, 0], modo1[:, 1], s=12, alpha=0.6, color=GREEN, label='Modo 1 normal')
axes[0].scatter(modo2[:, 0], modo2[:, 1], s=12, alpha=0.6, color=BLUE, label='Modo 2 normal')
axes[0].scatter(anoms[:, 0], anoms[:, 1], s=20, alpha=0.8, color=ARCA_RED,
                 marker='x', linewidth=1.5, label='Anomalía')
axes[0].set_xlabel('Temperatura', fontsize=11)
axes[0].set_ylabel('Presión', fontsize=11)
axes[0].set_title('Telemetría: dos modos normales + anomalías',
                   fontsize=12, fontweight='bold', color=ARCA_DARK)
axes[0].legend(loc='upper left', fontsize=10)
axes[0].spines['left'].set_visible(True)
axes[0].spines['bottom'].set_visible(True)
axes[0].grid(alpha=0.2)

# Derecha: histograma de errores reconstrucción
err_n = np.random.gamma(2, 0.05, 1400)
err_a = np.random.gamma(6, 0.12, 100) + 0.5

axes[1].hist(err_n, bins=40, alpha=0.7, color=GREEN,
              label='Normal', density=True)
axes[1].hist(err_a, bins=20, alpha=0.7, color=ARCA_RED,
              label='Anomalía', density=True)
axes[1].set_xlabel('Reconstruction error', fontsize=11)
axes[1].set_ylabel('Densidad', fontsize=11)
axes[1].set_title('AE captura correlaciones del régimen normal',
                   fontsize=12, fontweight='bold', color=ARCA_DARK)
axes[1].legend(fontsize=10)
axes[1].spines['left'].set_visible(True)
axes[1].spines['bottom'].set_visible(True)
axes[1].grid(alpha=0.2)

plt.tight_layout()
save('fig_tabular.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_unet.png — arquitectura U-Net con skip connections
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')

# Niveles del encoder (descendiendo)
niveles_enc = [
    (1.0,  4.5, 0.8, 1.2, '256×256\n×64', ARCA_RED),
    (2.5,  3.7, 0.7, 0.9, '128×128\n×128', ARCA_RED),
    (4.0,  3.0, 0.6, 0.6, '64×64\n×256', ARCA_RED),
    (5.5,  2.4, 0.5, 0.4, '32×32\n×512', ARCA_RED),
]
# Bottleneck
ax.add_patch(FancyBboxPatch((6.6, 2.0), 0.8, 1.0,
                              boxstyle="round,pad=0.05",
                              fc=PURPLE, ec=ARCA_DARK, lw=2))
ax.text(7.0, 2.5, '16×16\n×1024', ha='center', color='white',
         fontsize=9, fontweight='bold')
ax.text(7.0, 1.5, 'bottleneck', ha='center', color=PURPLE,
         fontsize=9, fontweight='bold', style='italic')

# Niveles del decoder (ascendiendo)
niveles_dec = [
    (8.1,  2.4, 0.5, 0.4, '32×32\n×512', ARCA_RED),
    (9.4,  3.0, 0.6, 0.6, '64×64\n×256', ARCA_RED),
    (10.8, 3.7, 0.7, 0.9, '128×128\n×128', ARCA_RED),
    (12.3, 4.5, 0.8, 1.2, '256×256\n×64', ARCA_RED),
]

for x, y, w, h, lbl, color in niveles_enc + niveles_dec:
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec='black', alpha=0.85))
    ax.text(x + w/2, y - 0.25, lbl, ha='center', fontsize=8)

# Flechas encoder
for i in range(len(niveles_enc) - 1):
    x0, y0, w0, h0, _, _ = niveles_enc[i]
    x1, y1, w1, h1, _, _ = niveles_enc[i + 1]
    ax.annotate('', xy=(x1, y1 + h1/2), xytext=(x0 + w0, y0 + h0/2),
                 arrowprops=dict(arrowstyle='->', lw=1.5))
# Encoder → bottleneck
ax.annotate('', xy=(6.6, 2.5), xytext=(6.0, 2.6),
             arrowprops=dict(arrowstyle='->', lw=1.5))
# Bottleneck → decoder
ax.annotate('', xy=(8.1, 2.6), xytext=(7.4, 2.5),
             arrowprops=dict(arrowstyle='->', lw=1.5))
# Flechas decoder
for i in range(len(niveles_dec) - 1):
    x0, y0, w0, h0, _, _ = niveles_dec[i]
    x1, y1, w1, h1, _, _ = niveles_dec[i + 1]
    ax.annotate('', xy=(x1, y1 + h1/2), xytext=(x0 + w0, y0 + h0/2),
                 arrowprops=dict(arrowstyle='->', lw=1.5))

# SKIP CONNECTIONS (líneas curvas horizontales)
for (xe, ye, we, he, _, _), (xd, yd, wd, hd, _, _) in zip(niveles_enc, reversed(niveles_dec)):
    cx0, cy0 = xe + we/2, ye + he
    cx1, cy1 = xd + wd/2, yd + hd
    midy = max(cy0, cy1) + 0.7
    ax.annotate('', xy=(cx1, cy1), xytext=(cx0, cy0),
                 arrowprops=dict(arrowstyle='->', lw=1.2, color=ORANGE,
                                  ls='--', connectionstyle="arc3,rad=-0.3"))

# Labels grandes
ax.text(3.0, 5.7, 'ENCODER', fontsize=12, fontweight='bold', color=ARCA_DARK)
ax.text(10.5, 5.7, 'DECODER', fontsize=12, fontweight='bold', color=ARCA_DARK)
ax.text(7.0, 5.7, '↑ skip connections (concat)', fontsize=10,
         color=ORANGE, fontweight='bold', ha='center')

# Output
ax.add_patch(Rectangle((13.4, 4.5), 0.4, 1.2, fc=LIGHT_GRAY, ec='black'))
ax.text(13.6, 4.2, 'máscara\nseg', ha='center', fontsize=8, fontweight='bold')

# Caption
ax.text(7.0, 0.6,
         'U-Net = encoder-decoder convolucional + skip connections que conservan resolución espacial',
         ha='center', fontsize=11, fontweight='bold', color=ARCA_DARK)
ax.text(7.0, 0.15,
         'el skip "saltea" el bottleneck → bordes finos preservados → máscara nítida',
         ha='center', fontsize=10, style='italic', color=GRAY)

plt.tight_layout()
save('fig_unet.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_familia_ae.png — la familia AE
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis('off')

# AE base en el centro arriba
ax.add_patch(FancyBboxPatch((5.0, 4.5), 3.0, 1.0,
                              boxstyle="round,pad=0.1",
                              fc=ARCA_RED, ec=ARCA_DARK, lw=2))
ax.text(6.5, 5.0, 'Autoencoder', ha='center', color='white',
         fontsize=13, fontweight='bold')
ax.text(6.5, 4.7, '(encoder-decoder por reconstrucción)',
         ha='center', color='white', fontsize=9, style='italic')

# Variantes (debajo, 5 cajas)
variantes = [
    ('Denoising AE', 'input degradado\n→ target limpio', GREEN,    0.2),
    ('U-Net',        'AE + skip conn.\n→ segmentación',  BLUE,     2.8),
    ('VAE',          'latente Gaussiano\n→ generación',  ORANGE,   5.4),
    ('VQ-VAE',       'latente discreto\n→ DALL-E base',  PURPLE,   8.0),
    ('Diffusion',    'denoising iterativo\n→ SOTA generación', ARCA_DARK, 10.6),
]

for nombre, desc, color, x in variantes:
    ax.add_patch(FancyBboxPatch((x, 1.5), 2.3, 2.3,
                                  boxstyle="round,pad=0.05",
                                  fc=color, ec='black', lw=1.5, alpha=0.85))
    ax.text(x + 1.15, 3.1, nombre, ha='center', color='white',
             fontsize=11, fontweight='bold')
    ax.text(x + 1.15, 2.2, desc, ha='center', color='white',
             fontsize=9, style='italic')

    # Flecha del AE base a la variante
    ax.annotate('', xy=(x + 1.15, 3.8), xytext=(6.5, 4.5),
                 arrowprops=dict(arrowstyle='->', lw=1, color=GRAY,
                                  connectionstyle="arc3,rad=0.1"))

ax.text(6.5, 0.6,
         'Cambiar el target / agregar regularización al latente / iterar →  toda la generación moderna',
         ha='center', fontsize=11, fontweight='bold', color=ARCA_DARK)

plt.tight_layout()
save('fig_familia_ae.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_cuando_usar.png — decisión rápida
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis('off')

ax.text(5.5, 5.7,
         '¿Necesito un autoencoder?',
         ha='center', fontsize=14, fontweight='bold', color=ARCA_DARK)

# Pregunta 1
ax.add_patch(FancyBboxPatch((4.0, 4.3), 3.0, 0.8,
                              boxstyle="round,pad=0.05",
                              fc=LIGHT_GRAY, ec=ARCA_DARK, lw=1.5))
ax.text(5.5, 4.7, '¿Tienes etiquetas?', ha='center', fontsize=11, fontweight='bold')

# Sí → modelo supervisado
ax.annotate('SÍ', xy=(2.5, 3.5), xytext=(4.0, 4.3),
             fontsize=10, fontweight='bold', color=ARCA_RED,
             arrowprops=dict(arrowstyle='->', lw=1.5))
ax.add_patch(FancyBboxPatch((0.8, 3.0), 3.4, 0.7,
                              boxstyle="round,pad=0.05",
                              fc=ARCA_RED, ec='black', alpha=0.85))
ax.text(2.5, 3.35, 'Modelo supervisado',
         ha='center', color='white', fontsize=11, fontweight='bold')
ax.text(2.5, 2.6, 'CNN, RF, XGBoost, MLP...',
         ha='center', fontsize=9, color=GRAY, style='italic')

# No → siguiente pregunta
ax.annotate('NO', xy=(7.5, 3.5), xytext=(7.0, 4.3),
             fontsize=10, fontweight='bold', color=GREEN,
             arrowprops=dict(arrowstyle='->', lw=1.5))
ax.add_patch(FancyBboxPatch((6.0, 3.0), 3.4, 0.7,
                              boxstyle="round,pad=0.05",
                              fc=LIGHT_GRAY, ec=ARCA_DARK, lw=1.5))
ax.text(7.7, 3.35, '¿Qué necesitás?',
         ha='center', fontsize=10, fontweight='bold')

# Variantes finales
opciones = [
    ('Detectar\nanomalías',  'AE\nanomaly det.', GREEN, 0.7),
    ('Limpiar\nruido',        'Denoising AE',   BLUE,  3.3),
    ('Comprimir\nfeatures',   'AE estándar\n(o PCA si lineal)', ORANGE, 5.9),
    ('Generar\nimágenes',     'VAE / Diffusion\n(no AE clásico)', PURPLE, 8.5),
]

for label, modelo, color, x in opciones:
    ax.add_patch(FancyBboxPatch((x, 0.9), 2.2, 1.4,
                                  boxstyle="round,pad=0.05",
                                  fc=color, ec='black', lw=1.2, alpha=0.85))
    ax.text(x + 1.1, 1.8, label, ha='center', color='white',
             fontsize=10, fontweight='bold')
    ax.text(x + 1.1, 1.2, modelo, ha='center', color='white',
             fontsize=9, style='italic')

    # Flecha desde "qué necesitás"
    ax.annotate('', xy=(x + 1.1, 2.3), xytext=(7.7, 3.0),
                 arrowprops=dict(arrowstyle='->', lw=0.8, color=GRAY,
                                  connectionstyle="arc3,rad=0.15"))

plt.tight_layout()
save('fig_cuando_usar.png')

# ═════════════════════════════════════════════════════════════════════════════
# Reconstrucciones REALES de MNIST: entrenamos AE con sklearn MLPRegressor
# ═════════════════════════════════════════════════════════════════════════════
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

print("\nDescargando MNIST y entrenando AEs reales para figuras...")
try:
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X_all = mnist.data.astype('float32') / 255.0
    y_all = mnist.target.astype(int)
except Exception as e:
    print(f"No se pudo descargar MNIST: {e}")
    raise

# Split simple
X_tr_flat = X_all[:60000]
y_tr_m = y_all[:60000]
X_te_flat = X_all[60000:]
y_te_m = y_all[60000:]
print(f"  Train: {X_tr_flat.shape}, Test: {X_te_flat.shape}")

# Para velocidad, entrenamos con un subset
SUB = 15000
rng_sub = np.random.RandomState(0)
sub_idx = rng_sub.choice(len(X_tr_flat), SUB, replace=False)
X_sub = X_tr_flat[sub_idx]

# AE Dense con bottleneck=32: 784 → 128 → 32 → 128 → 784
ae_real = MLPRegressor(hidden_layer_sizes=(128, 32, 128),
                        activation='relu',
                        solver='adam',
                        batch_size=256,
                        max_iter=30,
                        learning_rate_init=1e-3,
                        random_state=42,
                        early_stopping=False,
                        verbose=False)
ae_real.fit(X_sub, X_sub)
print(f"  AE 32D entrenado (loss final: {ae_real.loss_:.4f})")

# AE Dense con bottleneck=2 (para visualizar el latente)
ae_2d = MLPRegressor(hidden_layer_sizes=(128, 32, 2, 32, 128),
                      activation='relu',
                      solver='adam',
                      batch_size=256,
                      max_iter=40,
                      learning_rate_init=1e-3,
                      random_state=42,
                      early_stopping=False,
                      verbose=False)
ae_2d.fit(X_sub, X_sub)
print(f"  AE 2D entrenado (loss final: {ae_2d.loss_:.4f})")


def forward_to_layer(X, model, target_layer):
    """Forward pass manual a través de las capas del MLP hasta target_layer (1-indexed).

    target_layer=1 es la primera hidden, etc. Para AE (128, 32, 128) el
    bottleneck 32 es target_layer=2.
    """
    h = X
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        h = h @ W + b
        if i + 1 == target_layer:
            return h
        # Activación ReLU para hidden, identity para output
        if i < len(model.coefs_) - 1:
            h = np.maximum(0, h)
    return h


# ── fig_compresion_por_digito.png: reconstrucción REAL del AE ───────────────
fig, axes = plt.subplots(10, 10, figsize=(11, 11))
for d in range(10):
    idxs = np.where(y_te_m == d)[0][:5]
    for k, idx in enumerate(idxs):
        x = X_te_flat[idx:idx + 1]
        x_rec = ae_real.predict(x)
        axes[d, 2 * k].imshow(x.reshape(28, 28), cmap='gray_r')
        axes[d, 2 * k].axis('off')
        axes[d, 2 * k + 1].imshow(x_rec.reshape(28, 28), cmap='gray_r')
        axes[d, 2 * k + 1].axis('off')
    axes[d, 0].text(-10, 14, f"{d}", fontsize=16, fontweight='bold',
                     color=ARCA_DARK, va='center')

for k in range(5):
    axes[0, 2 * k].set_title('orig', fontsize=8, color=GRAY, pad=2)
    axes[0, 2 * k + 1].set_title('recon', fontsize=8, color=ARCA_RED, pad=2)

fig.suptitle('Reconstrucción REAL del AE Dense (bottleneck=32) — 5 ejemplos por dígito',
              fontsize=13, fontweight='bold', color=ARCA_DARK, y=0.995)
plt.tight_layout()
save('fig_compresion_por_digito.png')


# ── fig_latente_2d.png: scatter REAL del encoder 2D ─────────────────────────
# El bottleneck=2 está en la capa 3 del modelo (128, 32, 2, 32, 128) — index 3
# (porque después del primer hidden 128 → segundo hidden 32 → bottleneck 2)
latent_test = forward_to_layer(X_te_flat, ae_2d, target_layer=3)
print(f"  Latente 2D shape: {latent_test.shape}")


# ── fig_latente_por_digito.png: 784 → 32 → 784 por dígito ───────────────────
# Tomamos UN ejemplo de cada dígito (0-9) y mostramos el flujo completo:
#   original (784 px)  →  latente (32 dim)  →  reconstrucción (784 px)
idxs_per_digit = [int(np.where(y_te_m == d)[0][0]) for d in range(10)]
X_examples = X_te_flat[idxs_per_digit]                    # 10 × 784
latents_32 = forward_to_layer(X_examples, ae_real,
                                target_layer=2)             # 10 × 32
recons = ae_real.predict(X_examples)                       # 10 × 784

vmin, vmax = latents_32.min(), latents_32.max()

fig = plt.figure(figsize=(13, 10))
# 10 filas (dígitos) × 3 columnas (original, latente, reconstrucción)
# Anchos relativos: imagen chica, heatmap latente más ancho, imagen chica
gs = fig.add_gridspec(10, 3, hspace=0.25, wspace=0.15,
                       width_ratios=[1.2, 5.0, 1.2])

# Encabezados de columna
ax_h0 = fig.add_subplot(gs[0, 0]); ax_h0.set_axis_off()
ax_h0.set_title('Original\n28×28 = 784 px',
                 fontsize=11, fontweight='bold', color=ARCA_DARK, pad=12)
ax_h1 = fig.add_subplot(gs[0, 1]); ax_h1.set_axis_off()
ax_h1.set_title('Espacio latente — 32 dimensiones\n(compresión 24× respecto al input)',
                 fontsize=11, fontweight='bold', color=PURPLE, pad=12)
ax_h2 = fig.add_subplot(gs[0, 2]); ax_h2.set_axis_off()
ax_h2.set_title('Reconstrucción\n28×28 = 784 px',
                 fontsize=11, fontweight='bold', color=ARCA_DARK, pad=12)

# Rehacer todas las subplots (el header borra la fila 0; las regeneramos)
for d in range(10):
    # Imagen original
    ax_o = fig.add_subplot(gs[d, 0])
    ax_o.imshow(X_examples[d].reshape(28, 28), cmap='gray_r')
    ax_o.set_xticks([]); ax_o.set_yticks([])
    ax_o.set_ylabel(f'{d}', rotation=0, ha='right', va='center',
                     fontsize=15, fontweight='bold', color=ARCA_DARK,
                     labelpad=10)

    # Latente (heatmap horizontal de 32 celdas)
    ax_l = fig.add_subplot(gs[d, 1])
    ax_l.imshow(latents_32[d:d+1], cmap='viridis', aspect='auto',
                 vmin=vmin, vmax=vmax)
    ax_l.set_yticks([])
    if d < 9:
        ax_l.set_xticks([])
    else:
        ax_l.set_xticks(np.arange(0, 32, 4))
        ax_l.set_xticklabels(np.arange(0, 32, 4), fontsize=8)
        ax_l.set_xlabel('Dimensión latente (índice 0..31)',
                         fontsize=10, color=ARCA_DARK)

    # Reconstrucción
    ax_r = fig.add_subplot(gs[d, 2])
    ax_r.imshow(recons[d].reshape(28, 28), cmap='gray_r')
    ax_r.set_xticks([]); ax_r.set_yticks([])

fig.suptitle('Cómo se ve el espacio latente para cada dígito',
              fontsize=15, fontweight='bold', color=ARCA_DARK, y=0.985)
fig.text(0.5, 0.02,
          'Cada fila: el mismo dígito comprimido a 32 números (centro) y reconstruido (derecha). '
          'Patrón latente distinto por clase = el encoder aprendió a codificar cada dígito de forma única.',
          ha='center', fontsize=10, style='italic', color=GRAY, wrap=True)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
save('fig_latente_por_digito.png')

fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(latent_test[:, 0], latent_test[:, 1],
                 c=y_te_m, cmap='tab10', s=4, alpha=0.6)
ax.set_xlabel('z₁  (coordenada latente 1)', fontsize=11)
ax.set_ylabel('z₂  (coordenada latente 2)', fontsize=11)
ax.set_title('Espacio latente 2D real — AE entrenado sin etiquetas',
              fontsize=12, fontweight='bold', color=ARCA_DARK, pad=12)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.grid(alpha=0.2)
plt.colorbar(sc, ticks=range(10), label='Dígito real (no usado en entrenamiento)')
plt.tight_layout()
save('fig_latente_2d.png')


# ── fig_pca_vs_ae.png: PCA REAL vs AE latente REAL ──────────────────────────
pca = PCA(n_components=2).fit(X_tr_flat)
pca_test = pca.transform(X_te_flat)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(pca_test[:, 0], pca_test[:, 1],
                 c=y_te_m, cmap='tab10', s=5, alpha=0.6)
axes[0].set_title('PCA (lineal)', fontsize=13, fontweight='bold', color=ARCA_DARK)
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
axes[0].grid(alpha=0.2)

axes[1].scatter(latent_test[:, 0], latent_test[:, 1],
                 c=y_te_m, cmap='tab10', s=5, alpha=0.6)
axes[1].set_title('Autoencoder (no lineal)', fontsize=13, fontweight='bold', color=ARCA_DARK)
axes[1].set_xlabel('z₁'); axes[1].set_ylabel('z₂')
axes[1].grid(alpha=0.2)

for ax in axes:
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

fig.suptitle('MNIST en 2D: PCA vs Autoencoder (ambos entrenados sobre los datos reales)',
              fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save('fig_pca_vs_ae.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_ruido_ejemplo.png — cómo se ve el ruido gaussiano sobre MNIST real
# ═════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
X_te_imgs = X_te_flat.reshape(-1, 28, 28)
idxs_demo = np.arange(8)
limpios = X_te_imgs[idxs_demo]

# Tres niveles de ruido para mostrar el efecto
niveles = [0.2, 0.4, 0.6]
fig, axes = plt.subplots(4, 8, figsize=(13, 5.5))

for j, idx in enumerate(idxs_demo):
    axes[0, j].imshow(limpios[j], cmap='gray')
    axes[0, j].axis('off')

for i, nivel in enumerate(niveles):
    for j, idx in enumerate(idxs_demo):
        noisy = limpios[j] + nivel * np.random.normal(size=limpios[j].shape)
        noisy = np.clip(noisy, 0, 1)
        axes[i + 1, j].imshow(noisy, cmap='gray')
        axes[i + 1, j].axis('off')

# Etiquetas a la izquierda
labels_y = ['Original\n(limpio)',
            f'Ruido σ={niveles[0]}',
            f'Ruido σ={niveles[1]}  ←  el que usamos',
            f'Ruido σ={niveles[2]}']
colors_y = [GREEN, GRAY, ARCA_RED, GRAY]
for i, (lab, col) in enumerate(zip(labels_y, colors_y)):
    axes[i, 0].text(-14, 14, lab, fontsize=10, fontweight='bold',
                     color=col, va='center', ha='right')

fig.suptitle('Cómo se ve el ruido gaussiano sobre MNIST a distintos niveles σ',
              fontsize=13, fontweight='bold', color=ARCA_DARK, y=0.99)
plt.tight_layout()
save('fig_ruido_ejemplo.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_pretraining.png — Pretraining sin labels → finetune supervisado
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')

ax.text(7, 5.7,
         'Pretraining: el encoder es un extractor de features universal',
         ha='center', fontsize=14, fontweight='bold', color=ARCA_DARK)

# FASE 1: Entrenar AE con muchos datos sin label
ax.text(3.0, 5.0, 'FASE 1 — sin labels',
         ha='center', fontsize=11, fontweight='bold', color=GREEN)
ax.add_patch(FancyBboxPatch((0.5, 2.5), 1.6, 1.6,
                              boxstyle="round,pad=0.05",
                              fc=LIGHT_GRAY, ec='black', lw=1.5))
ax.text(1.3, 3.5, '100.000\nimágenes', ha='center', fontsize=10, fontweight='bold')
ax.text(1.3, 2.85, 'SIN labels', ha='center', fontsize=9,
         color=GREEN, fontweight='bold', style='italic')

ax.annotate('', xy=(2.5, 3.3), xytext=(2.15, 3.3),
             arrowprops=dict(arrowstyle='->', lw=1.8))

ax.add_patch(FancyBboxPatch((2.6, 2.3), 2.6, 2.0,
                              boxstyle="round,pad=0.05",
                              fc=PURPLE, ec=ARCA_DARK, lw=1.5, alpha=0.85))
ax.text(3.9, 3.65, 'Entrenar AE', ha='center', color='white',
         fontsize=11, fontweight='bold')
ax.text(3.9, 3.10, 'fit(X, X)', ha='center', color='white',
         fontsize=10, family='monospace')
ax.text(3.9, 2.55, 'auto-supervisado', ha='center', color='white',
         fontsize=9, style='italic')

# Separador
ax.plot([5.7, 5.7], [0.5, 5.5], color=GRAY, lw=1, ls='--')

# FASE 2: Usar encoder + downstream
ax.text(10.5, 5.0, 'FASE 2 — con pocos labels',
         ha='center', fontsize=11, fontweight='bold', color=ARCA_RED)

ax.add_patch(FancyBboxPatch((6.0, 2.5), 1.6, 1.6,
                              boxstyle="round,pad=0.05",
                              fc=LIGHT_GRAY, ec='black', lw=1.5))
ax.text(6.8, 3.5, '500\nimágenes', ha='center', fontsize=10, fontweight='bold')
ax.text(6.8, 2.85, 'con label', ha='center', fontsize=9,
         color=ARCA_RED, fontweight='bold', style='italic')

ax.annotate('', xy=(8.0, 3.3), xytext=(7.65, 3.3),
             arrowprops=dict(arrowstyle='->', lw=1.8))

# Encoder reutilizado (highlighted)
ax.add_patch(FancyBboxPatch((8.1, 2.7), 1.8, 1.2,
                              boxstyle="round,pad=0.05",
                              fc=PURPLE, ec=ARCA_DARK, lw=2.5))
ax.text(9.0, 3.45, 'encoder', ha='center', color='white',
         fontsize=11, fontweight='bold')
ax.text(9.0, 3.0, '(reutilizado)', ha='center', color='white',
         fontsize=9, style='italic')

ax.annotate('', xy=(10.3, 3.3), xytext=(9.95, 3.3),
             arrowprops=dict(arrowstyle='->', lw=1.8))

ax.add_patch(FancyBboxPatch((10.4, 2.5), 2.6, 1.6,
                              boxstyle="round,pad=0.05",
                              fc=ARCA_RED, ec=ARCA_DARK, lw=1.5, alpha=0.85))
ax.text(11.7, 3.65, 'Clasificador', ha='center', color='white',
         fontsize=11, fontweight='bold')
ax.text(11.7, 3.10, 'Dense(N_clases)', ha='center', color='white',
         fontsize=9, family='monospace')
ax.text(11.7, 2.7, 'sklearn / Dense', ha='center', color='white',
         fontsize=9, style='italic')

# Resultado abajo
ax.add_patch(FancyBboxPatch((1.0, 0.4), 11.5, 1.4,
                              boxstyle="round,pad=0.1",
                              fc=LIGHT_GRAY, ec=ARCA_DARK, lw=1.5))
ax.text(6.75, 1.4, 'Resultado: clasificador con 500 labels supera al mismo Dense entrenado desde cero',
         ha='center', fontsize=11, fontweight='bold', color=ARCA_DARK)
ax.text(6.75, 0.85, 'el encoder ya aprendió "qué importa visualmente" durante la fase 1',
         ha='center', fontsize=10, style='italic', color=GRAY)

plt.tight_layout()
save('fig_pretraining.png')


# ═════════════════════════════════════════════════════════════════════════════
# fig_denoising_series.png — denoising sobre series temporales
# ═════════════════════════════════════════════════════════════════════════════
t = np.linspace(0, 8 * np.pi, 800)
clean = (np.sin(t) + 0.5 * np.sin(2.5 * t) + 0.3 * np.cos(0.7 * t))
np.random.seed(0)
noisy = clean + np.random.normal(0, 0.6, len(t))
# Simulación de reconstrucción: suavizado moderado
from scipy.ndimage import gaussian_filter1d
recon = gaussian_filter1d(noisy, sigma=4)

fig, axes = plt.subplots(3, 1, figsize=(13, 5.5), sharex=True)
axes[0].plot(t, noisy, color=GRAY, lw=0.8)
axes[0].set_title('Input: serie con ruido (sensor de planta industrial)',
                   fontweight='bold', color=ARCA_DARK, fontsize=11, loc='left')
axes[0].set_ylabel('input', fontweight='bold')

axes[1].plot(t, recon, color=ARCA_RED, lw=1.5)
axes[1].set_title('Output del AE: reconstrucción (solo estructura comprimible sobrevive)',
                   fontweight='bold', color=ARCA_DARK, fontsize=11, loc='left')
axes[1].set_ylabel('output AE', fontweight='bold')

axes[2].plot(t, clean, color=GREEN, lw=1.5)
axes[2].set_title('Señal real (no observada en producción)',
                   fontweight='bold', color=ARCA_DARK, fontsize=11, loc='left')
axes[2].set_ylabel('verdad', fontweight='bold')
axes[2].set_xlabel('tiempo (muestras)', fontweight='bold')

for ax in axes:
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.grid(alpha=0.3)
    ax.set_ylim(-3.5, 3.5)

fig.suptitle('Denoising AE sobre series temporales (Conv1D encoder-decoder)',
              fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save('fig_denoising_series.png')


print("\nTodas las figuras generadas correctamente.")
