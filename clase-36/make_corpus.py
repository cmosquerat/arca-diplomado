"""Genera el corpus pedagogico para clase 36.

Sale en clase-36/corpus/:
  manual_mtto_arca.pdf            — manual de planta embotelladora, 6 paginas
  manual_mtto_arca_ESCANEADO.pdf  — mismo manual rasterizado (PyPDF falla)
  codigos_error.csv               — la tabla embedded como CSV
  fotos/compresor.png             — diagrama sintetico del compresor
  fotos/panel_control.png         — panel con codigo de error
  fotos/etiquetadora.png          — etiquetadora con etiqueta torcida
"""
import os, fitz, csv
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.lib import colors
from reportlab.lib.units import cm

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(ROOT, "corpus")
FOTOS = os.path.join(CORPUS, "fotos")
os.makedirs(FOTOS, exist_ok=True)

ARCA_RED = "#C82B40"
ARCA_DARK = "#6B1525"
ARCA_GRAY = "#F5F5F5"


# =============================================================================
#  1. Tabla de codigos de error → CSV
# =============================================================================
CODIGOS = [
    ("ERR-001", "Compresor", "Vibración fuera de rango (>75 Hz)",
     "Verificar rodamientos. Cambiar si supera 50.000 h de operación."),
    ("ERR-002", "Compresor", "Presión de descarga por debajo del mínimo",
     "Revisar filtro de entrada. Limpieza correctiva."),
    ("ERR-003", "Compresor", "Temperatura del aceite supera 85 °C",
     "Verificar enfriador. Comprobar nivel y calidad del aceite."),
    ("ERR-004", "Llenadora", "Cabezal sin presión (bajo umbral)",
     "Inspeccionar electroválvula del cabezal. Reemplazar empaque si hay fuga."),
    ("ERR-005", "Llenadora", "Volumen llenado fuera de tolerancia ±2 %",
     "Calibrar sensor de nivel. Si persiste, reemplazar pistón."),
    ("ERR-006", "Llenadora", "Botella detectada sin tapa al salir",
     "Validar sensor óptico de tapa. Limpiar lente."),
    ("ERR-007", "Llenadora", "Pérdida sostenida de presión en cabezal 2",
     "Parar línea, inspeccionar sello del cabezal 2, reemplazar O-ring."),
    ("ERR-008", "Etiquetadora", "Etiqueta aplicada con desviación >3°",
     "Calibrar rodillo aplicador. Verificar tensión del rollo de etiqueta."),
    ("ERR-009", "Etiquetadora", "Sin etiqueta en botella",
     "Verificar sensor de presencia y rollo de etiquetas."),
    ("ERR-010", "Etiquetadora", "Pegamento por debajo del nivel mínimo",
     "Reponer adhesivo en el tanque, purgar línea de aplicador."),
    ("ERR-011", "Transportador", "Banda con sobrecarga",
     "Verificar acumulación aguas abajo. Pausar entrada."),
    ("ERR-012", "Transportador", "Velocidad inconsistente",
     "Revisar variador de frecuencia y acoples del motor."),
    ("ERR-013", "PLC", "Pérdida de comunicación con HMI",
     "Reiniciar red Ethernet industrial. Verificar cable apantallado."),
    ("ERR-014", "PLC", "Watchdog timeout",
     "Reiniciar PLC. Si persiste, contactar a soporte Siemens/Allen-Bradley."),
    ("ERR-015", "General", "Botón de emergencia accionado",
     "Inspeccionar la línea, liberar emergencia con personal autorizado."),
]

with open(os.path.join(CORPUS, "codigos_error.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["codigo", "equipo", "causa", "accion"])
    for row in CODIGOS:
        w.writerow(row)
print(f"OK codigos_error.csv ({len(CODIGOS)} filas)")


# =============================================================================
#  2. Fotos sinteticas de equipos
# =============================================================================
plt.rcParams["figure.dpi"] = 110


def fig_compresor():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    # Cilindro principal
    ax.add_patch(mp.FancyBboxPatch((1.5, 1.5), 6, 4.5,
        boxstyle="round,pad=0.05,rounding_size=0.4", facecolor="#888",
        edgecolor=ARCA_DARK, lw=2.5))
    # tapa lateral
    ax.add_patch(mp.Circle((1.7, 3.75), 1, facecolor="#666", edgecolor=ARCA_DARK, lw=2))
    ax.add_patch(mp.Circle((1.7, 3.75), 0.35, facecolor="#444", edgecolor=ARCA_DARK))
    # ventilador
    for ang in range(0, 360, 60):
        import numpy as np
        ax.plot([1.7, 1.7 + 0.9*np.cos(np.radians(ang))],
                [3.75, 3.75 + 0.9*np.sin(np.radians(ang))], color="#222", lw=2.5)
    # Etiqueta
    ax.text(4.5, 4, "COMPRESOR\nGA-160 VSD", fontsize=14, color="white",
            ha="center", va="center", fontweight="bold")
    # Tubería de salida
    ax.add_patch(mp.Rectangle((7.5, 3.4), 3.5, 0.7, facecolor="#aaa",
                              edgecolor=ARCA_DARK, lw=2))
    ax.add_patch(mp.Rectangle((10.5, 2.5), 1, 2.5, facecolor="#888",
                              edgecolor=ARCA_DARK, lw=2))
    # Manómetro
    ax.add_patch(mp.Circle((9, 5.5), 0.6, facecolor="white",
                           edgecolor=ARCA_DARK, lw=2))
    ax.text(9, 5.5, "8.5 bar", fontsize=8, ha="center", va="center",
            color=ARCA_DARK, fontweight="bold")
    # placa
    ax.add_patch(mp.FancyBboxPatch((3.5, 0.2), 5, 0.9,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=ARCA_RED, edgecolor=ARCA_DARK))
    ax.text(6, 0.65, "Línea 2 · Sala compresores", color="white",
            fontsize=11, ha="center", fontweight="bold")
    plt.savefig(os.path.join(FOTOS, "compresor.png"), bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("OK fotos/compresor.png")


def fig_panel_control():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.add_patch(mp.FancyBboxPatch((1, 1), 8, 5,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor="#222", edgecolor="#444", lw=2.5))
    ax.add_patch(mp.FancyBboxPatch((1.5, 3.5), 7, 2,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor="#0a4", edgecolor="#0e7", lw=1.5))
    ax.text(5, 4.7, "ERR-007", color="#fff", fontsize=28,
            fontweight="bold", ha="center", va="center", family="monospace")
    ax.text(5, 3.8, "PRES BAJA CAB-2", color="#fff", fontsize=12,
            ha="center", family="monospace")
    for i, c in enumerate(["#c33", "#3c3", "#39c", "#fc3"]):
        ax.add_patch(mp.Circle((2 + i*2, 2), 0.35, facecolor=c,
                               edgecolor="#666", lw=2.5))
    ax.text(5, 0.55, "PANEL CONTROL · LÍNEA 2", color="#222", fontsize=11,
            ha="center", fontweight="bold")
    plt.savefig(os.path.join(FOTOS, "panel_control.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("OK fotos/panel_control.png")


def fig_etiquetadora():
    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
    # Banda transportadora
    ax.add_patch(mp.Rectangle((0, 1), 14, 1, facecolor="#444",
                              edgecolor=ARCA_DARK, lw=1.5))
    ax.add_patch(mp.Rectangle((0, 0.8), 14, 0.2, facecolor="#222"))
    # 3 botellas
    for i, x in enumerate([2.5, 6, 10]):
        # cuerpo botella
        ax.add_patch(mp.Rectangle((x-0.6, 2), 1.2, 2.5, facecolor="#d0e8f5",
                                   edgecolor=ARCA_DARK, lw=1.5))
        ax.add_patch(mp.Rectangle((x-0.4, 4.5), 0.8, 0.7,
                                   facecolor="#d0e8f5", edgecolor=ARCA_DARK))
        ax.add_patch(mp.Rectangle((x-0.3, 5.2), 0.6, 0.5,
                                   facecolor=ARCA_RED, edgecolor=ARCA_DARK))
        # etiqueta (torcida en la #3)
        if i == 2:
            # rotada — esta es la fallada
            from matplotlib.transforms import Affine2D
            r = mp.Rectangle((x-0.45, 2.7), 0.9, 1.6, facecolor=ARCA_RED,
                             edgecolor="#fff", lw=1)
            t = Affine2D().rotate_deg_around(x, 3.5, -22) + ax.transData
            r.set_transform(t)
            ax.add_patch(r)
            ax.annotate("etiqueta\ntorcida", (x+1, 5.5), fontsize=10,
                        color=ARCA_RED, fontweight="bold",
                        ha="center", va="bottom")
        else:
            ax.add_patch(mp.Rectangle((x-0.45, 2.7), 0.9, 1.6,
                                       facecolor=ARCA_RED, edgecolor="#fff"))
    # rodillo etiquetador
    ax.add_patch(mp.Circle((11, 3.5), 0.8, facecolor="#888",
                           edgecolor=ARCA_DARK, lw=2))
    ax.text(11, 3.5, "R", color="white", ha="center", va="center",
            fontsize=14, fontweight="bold")
    # Flecha de movimiento
    ax.annotate("", xy=(13, 1.5), xytext=(0.5, 1.5),
                arrowprops=dict(arrowstyle="->", color="white", lw=2.5))
    ax.text(7, 0.4, "ETIQUETADORA · LÍNEA 2", color=ARCA_DARK, fontsize=11,
            ha="center", fontweight="bold")
    plt.savefig(os.path.join(FOTOS, "etiquetadora.png"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("OK fotos/etiquetadora.png")


fig_compresor()
fig_panel_control()
fig_etiquetadora()


# =============================================================================
#  3. PDF del manual completo (texto + tabla + figuras embebidas)
# =============================================================================
PDF = os.path.join(CORPUS, "manual_mtto_arca.pdf")
doc = SimpleDocTemplate(PDF, pagesize=LETTER, title="Manual MTTO Arca",
                        leftMargin=2*cm, rightMargin=2*cm,
                        topMargin=2*cm, bottomMargin=2*cm)
ss = getSampleStyleSheet()
h1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=colors.HexColor(ARCA_RED),
                    fontSize=18, spaceAfter=12)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor(ARCA_DARK),
                    fontSize=13, spaceAfter=8)
body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=14)
small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=8, leading=10,
                       textColor=colors.HexColor("#666"))

story = []
story.append(Paragraph("Manual de Mantenimiento — Planta Embotelladora", h1))
story.append(Paragraph("Arca Continental Ecuador · Versión 2026.06", small))
story.append(Spacer(1, 14))

# Sección 1 — Compresor
story.append(Paragraph("1. Compresor Atlas Copco GA-160 VSD", h2))
story.append(Paragraph(
    "El compresor GA-160 VSD es la fuente principal de aire comprimido de la "
    "Línea 2. Opera a presión nominal de 8.5 bar con frecuencia variable "
    "ajustada al consumo de planta. Las verificaciones de rutina son: (1) "
    "nivel de aceite cada 8 horas, (2) temperatura de descarga, (3) "
    "vibración con sensor inductivo en el cabezal.", body))
story.append(Spacer(1, 6))
story.append(Image(os.path.join(FOTOS, "compresor.png"), width=11*cm, height=7*cm))
story.append(Paragraph("Figura 1 — Vista general del compresor GA-160 VSD.", small))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Síntomas habituales: vibración fuera de rango (>75 Hz) indica rodamientos "
    "fatigados; presión por debajo del mínimo suele señalar filtro de entrada "
    "saturado; temperatura del aceite >85 °C apunta a un enfriador sucio.", body))

story.append(PageBreak())

# Sección 2 — Llenadora
story.append(Paragraph("2. Llenadora rotativa KHS Innofill", h2))
story.append(Paragraph(
    "La llenadora rotativa tiene 24 cabezales que llenan en simultáneo. Cada "
    "cabezal cuenta con sensor de nivel inductivo y electroválvula de "
    "actuación rápida. La tolerancia volumétrica es de ±2 %. Si un cabezal "
    "se desvía, la línea pasa a alarma pero sigue operando con los 23 "
    "restantes; si dos o más fallan, parada inmediata.", body))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Caso típico: cabezal 2 pierde presión sostenidamente (ERR-007). Causa "
    "raíz frecuente: O-ring del sello desgastado o electroválvula con suciedad. "
    "Procedimiento: parar línea, inspeccionar sello del cabezal 2, reemplazar "
    "O-ring con la pieza KHS-OR-024 disponible en stock.", body))

story.append(PageBreak())

# Sección 3 — Etiquetadora
story.append(Paragraph("3. Etiquetadora rotativa KRONES Topmodul", h2))
story.append(Paragraph(
    "La etiquetadora aplica la etiqueta principal con un rodillo a 60 °C. La "
    "alineación se calibra al inicio de cada turno y se vigila por sensor "
    "óptico de salida. Una desviación mayor a 3 ° activa ERR-008.", body))
story.append(Spacer(1, 6))
story.append(Image(os.path.join(FOTOS, "etiquetadora.png"), width=12*cm, height=7*cm))
story.append(Paragraph("Figura 2 — Tres botellas, la tercera con etiqueta torcida.", small))
story.append(PageBreak())

# Sección 4 — Tabla de códigos
story.append(Paragraph("4. Tabla maestra de códigos de error", h2))
story.append(Paragraph(
    "Cuando un equipo entra en alarma, el HMI despliega el código junto con "
    "una descripción corta. La acción correctiva la determina el tabulado "
    "siguiente, mantenido por Ingeniería de Confiabilidad.", body))
story.append(Spacer(1, 8))

table_data = [["Código", "Equipo", "Causa", "Acción correctiva"]] + [
    list(r) for r in CODIGOS
]
t = Table(table_data, colWidths=[2*cm, 2.6*cm, 5.5*cm, 6.5*cm], repeatRows=1)
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor(ARCA_RED)),
    ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
    ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",   (0,0), (-1,-1), 8),
    ("VALIGN",     (0,0), (-1,-1), "TOP"),
    ("GRID",       (0,0), (-1,-1), 0.4, colors.HexColor("#999")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1),
     [colors.white, colors.HexColor(ARCA_GRAY)]),
]))
story.append(t)
story.append(Spacer(1, 8))
story.append(Paragraph("Tabla 1 — Códigos de error vigentes 2026 (15 entradas).", small))

doc.build(story)
print(f"OK manual_mtto_arca.pdf ({os.path.getsize(PDF)} bytes)")


# =============================================================================
#  4. Version "escaneada" del manual (cada pagina como imagen)
# =============================================================================
ESC = os.path.join(CORPUS, "manual_mtto_arca_ESCANEADO.pdf")
src = fitz.open(PDF)
out = fitz.open()
for page in src:
    pix = page.get_pixmap(dpi=72)
    sub = fitz.open()
    p = sub.new_page(width=pix.width, height=pix.height)
    p.insert_image(p.rect, stream=pix.tobytes("png"))
    out.insert_pdf(sub)
out.save(ESC)
src.close(); out.close()
print(f"OK manual_mtto_arca_ESCANEADO.pdf ({os.path.getsize(ESC)} bytes)")

print("\nCorpus listo en", CORPUS)
