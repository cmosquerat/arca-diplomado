"""
Figuras del deck clase 36:
  fig_arco.png                 — 6 paradas
  fig_pdf_pipeline.png         — 2 ramas: PyPDF | Scout OCR
  fig_imagen_pipeline.png      — foto → Scout → caption → embedding → Chroma
  fig_chroma_anatomy.png       — add() / query() / persist
  fig_rag_multimodal.png       — arquitectura final B5
"""
import os, warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mp
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCA_RED, ARCA_DARK, ARCA_GRAY = "#C82B40", "#6B1525", "#F5F5F5"
ARCA_GREEN, ARCA_BLUE, ARCA_ORANGE, ARCA_PURPLE = "#16A34A", "#2563EB", "#EA580C", "#7C3AED"
TEXT_MUTED = "#6B7280"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.dpi": 130,
})


def save(fig, name):
    fig.savefig(os.path.join(ROOT, name), bbox_inches="tight",
                facecolor="white", dpi=130)
    plt.close(fig)
    print(f"  guardada {name}")


def _box(ax, x, y, w, h, text, fill, fontcolor="white", fontsize=11, weight="bold"):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=fill, edgecolor=ARCA_DARK, lw=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=fontcolor, fontsize=fontsize, fontweight=weight)


def _arrow(ax, x1, y1, x2, y2, color=ARCA_DARK, lw=2.4, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# =============================================================================
# FIG ARCO — 6 paradas
# =============================================================================
def fig_arco():
    fig, ax = plt.subplots(figsize=(13.5, 3.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 3); ax.axis("off")
    stops = [
        ("0. El dolor", "RAG c35 falla\nen PDF sucio", ARCA_DARK),
        ("1. PDFs", "PyPDF o OCR\ncon Scout", ARCA_BLUE),
        ("2. Imágenes", "caption con\nVLM", ARCA_ORANGE),
        ("3. Tablas", "markdown\n(breve)", ARCA_PURPLE),
        ("4. Chroma", "vector DB\nque persiste", ARCA_GREEN),
        ("5. Asistente\nmtto Arca", "RAG\nmultimodal", ARCA_RED),
    ]
    w, h, gap = 2.0, 1.6, 0.25
    x0 = (14 - 6*w - 5*gap) / 2
    for i, (t1, t2, c) in enumerate(stops):
        x = x0 + i * (w + gap)
        _box(ax, x, 0.7, w, h, f"{t1}\n\n{t2}", fill=c, fontsize=9.5)
        if i < len(stops) - 1:
            _arrow(ax, x + w, 0.7 + h/2, x + w + gap, 0.7 + h/2,
                   color=ARCA_DARK, lw=1.8)
    save(fig, "fig_arco.png")


# =============================================================================
# FIG PDF PIPELINE — 2 ramas
# =============================================================================
def fig_pdf_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.5); ax.axis("off")

    # PDF entrante
    _box(ax, 0.3, 2.2, 1.8, 1.1, "PDF\n(input)", fill=ARCA_DARK, fontsize=11)
    # decisión
    _box(ax, 2.6, 2.2, 2.3, 1.1, "¿tiene texto\nextraíble?", fill=ARCA_GRAY,
         fontcolor=ARCA_DARK, fontsize=10)
    _arrow(ax, 2.1, 2.75, 2.6, 2.75)

    # Rama A: SÍ → PyPDF
    _arrow(ax, 4.9, 3.0, 6.5, 4.2, color=ARCA_GREEN)
    ax.text(5.6, 3.95, "sí", fontsize=11, fontweight="bold", color=ARCA_GREEN)
    _box(ax, 6.5, 3.8, 2.6, 1.1, "PyPDF\nPdfReader", fill=ARCA_GREEN, fontsize=10)

    # Rama B: NO → Scout OCR
    _arrow(ax, 4.9, 2.5, 6.5, 1.3, color=ARCA_RED)
    ax.text(5.6, 1.4, "no", fontsize=11, fontweight="bold", color=ARCA_RED)
    _box(ax, 6.5, 0.7, 2.6, 1.1, "Scout OCR\n(VLM)", fill=ARCA_RED, fontsize=10)

    # Convergen en TEXTO
    _arrow(ax, 9.1, 4.35, 10.5, 3.1, color=ARCA_GREEN)
    _arrow(ax, 9.1, 1.25, 10.5, 2.4, color=ARCA_RED)
    _box(ax, 10.5, 2.2, 1.8, 1.1, "TEXTO", fill=ARCA_DARK, fontsize=11)

    # Anotaciones laterales
    ax.text(11, 4.95, "PDF nativo\n(texto incrustado)",
            ha="center", fontsize=9, color=ARCA_GREEN, style="italic", fontweight="bold")
    ax.text(11, 0.6, "PDF escaneado\n(página = imagen)",
            ha="center", fontsize=9, color=ARCA_RED, style="italic", fontweight="bold")

    # Título arriba a la izquierda para no superponer
    ax.text(0.3, 5.2, "pdf_a_texto()  :  una función, dos caminos",
            ha="left", fontsize=12.5, fontweight="bold", color=ARCA_DARK)
    save(fig, "fig_pdf_pipeline.png")


# =============================================================================
# FIG IMAGEN PIPELINE — foto → Scout → caption → embedding → Chroma
# =============================================================================
def fig_imagen_pipeline():
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 3.6); ax.axis("off")

    _box(ax, 0.2, 1.2, 2.0, 1.4, "🖼\nfoto del\ncompresor",
         fill=ARCA_DARK, fontsize=10)
    _arrow(ax, 2.2, 1.9, 3.0, 1.9)
    _box(ax, 3.0, 1.2, 2.4, 1.4, "Llama 4\nScout VLM", fill=ARCA_ORANGE, fontsize=11)
    _arrow(ax, 5.4, 1.9, 6.2, 1.9)
    _box(ax, 6.2, 1.2, 2.6, 1.4, "Caption\n(texto)", fill=ARCA_GRAY,
         fontcolor=ARCA_DARK, fontsize=11)
    _arrow(ax, 8.8, 1.9, 9.6, 1.9)
    _box(ax, 9.6, 1.2, 2.0, 1.4, "Embedder\nmpnet", fill=ARCA_GREEN, fontsize=11)
    _arrow(ax, 11.6, 1.9, 12.4, 1.9)
    _box(ax, 12.4, 1.2, 1.4, 1.4, "Chroma\n📦", fill=ARCA_BLUE, fontsize=11)

    # Caption ejemplo abajo
    ax.text(7.5, 0.45,
            "ej.: \"Compresor GA-160 VSD operando a 8.5 bar en la sala "
            "de compresores de la Línea 2.\"",
            ha="center", fontsize=9.5, style="italic", color=TEXT_MUTED,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=ARCA_RED, lw=1))

    ax.text(7, 3.3, "Imagen → texto → vector. Mismo espacio que el resto del corpus.",
            ha="center", fontsize=11.5, fontweight="bold", color=ARCA_DARK)
    save(fig, "fig_imagen_pipeline.png")


# =============================================================================
# FIG CHROMA ANATOMY — add / query / persist
# =============================================================================
def fig_chroma_anatomy():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")

    # Collection (corazón)
    _box(ax, 5.5, 2.3, 3.0, 1.6, "Collection\nChroma", fill=ARCA_BLUE,
         fontsize=12)

    # add (input)
    _box(ax, 0.3, 4.4, 2.0, 1.0, "documents +\nmetadata + ids",
         fill=ARCA_GRAY, fontcolor=ARCA_DARK, fontsize=9.5)
    _box(ax, 0.3, 3.0, 2.0, 1.0, "embeddings\n(opcional)", fill=ARCA_GREEN,
         fontsize=10)
    _box(ax, 2.7, 3.7, 2.0, 1.0, ".add()", fill=ARCA_DARK, fontsize=11.5)
    _arrow(ax, 2.3, 4.9, 2.7, 4.4); _arrow(ax, 2.3, 3.5, 2.7, 4.0)
    _arrow(ax, 4.7, 4.2, 5.5, 3.5)

    # query
    _box(ax, 0.3, 0.6, 2.0, 1.0, "query texts\no embeddings",
         fill=ARCA_GRAY, fontcolor=ARCA_DARK, fontsize=9.5)
    _box(ax, 2.7, 0.6, 2.0, 1.0, ".query()", fill=ARCA_DARK, fontsize=11.5)
    _arrow(ax, 2.3, 1.1, 2.7, 1.1)
    _arrow(ax, 4.7, 1.1, 5.5, 2.6)

    # results
    _arrow(ax, 8.5, 2.6, 9.4, 1.1, color=ARCA_RED, lw=2.5)
    _box(ax, 9.4, 0.6, 3.0, 1.0, "top-K docs + metadata\n+ distances",
         fill=ARCA_RED, fontsize=10)

    # persist (lateral)
    _arrow(ax, 7, 2.3, 7, 1.0, color=TEXT_MUTED, lw=2, style="-|>")
    _box(ax, 9.4, 4.4, 3.0, 1.0, "disco\n(.chroma/)", fill=ARCA_PURPLE,
         fontsize=10)
    _arrow(ax, 8.5, 3.5, 9.4, 4.6, color=ARCA_PURPLE, lw=2.2, style="<->")
    ax.text(9.0, 4.0, "persistente", fontsize=9, color=ARCA_PURPLE,
            fontweight="bold", style="italic")

    ax.text(7, 5.5, "Chroma: .add() y .query() son todo lo que necesitás",
            ha="center", fontsize=13, fontweight="bold", color=ARCA_DARK)
    save(fig, "fig_chroma_anatomy.png")


# =============================================================================
# FIG RAG MULTIMODAL — arquitectura B5
# =============================================================================
def fig_rag_multimodal():
    fig, ax = plt.subplots(figsize=(14, 6.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")

    # 3 fuentes (columna izquierda)
    sources = [(5.4, "📄 PDF\nmanual", ARCA_DARK),
               (3.6, "🖼 fotos\nde equipos", ARCA_DARK),
               (1.8, "📊 tabla\ncódigos CSV", ARCA_DARK)]
    for y, t, c in sources:
        _box(ax, 0.2, y, 1.8, 1.3, t, fill=c, fontsize=9.5)

    # Pipelines (procesadores)
    procs = [(5.4, "pdf_a_texto()", ARCA_BLUE),
             (3.6, "caption(img)\ncon Scout", ARCA_ORANGE),
             (1.8, "tabla_a_md()", ARCA_PURPLE)]
    for y, t, c in procs:
        _box(ax, 2.5, y, 2.4, 1.3, t, fill=c, fontsize=9.5)

    # Embedder común
    _box(ax, 5.4, 3.6, 2.0, 1.3, "Embedder\nmpnet", fill=ARCA_GREEN, fontsize=10.5)

    # Chroma (almacén central)
    _box(ax, 8.0, 3.0, 2.4, 2.4, "Chroma\ncollection\n(con metadata\nmodalidad)",
         fill=ARCA_BLUE, fontsize=10.5)

    # Flechas: 3 fuentes → 3 procs
    for y in [5.4, 3.6, 1.8]:
        _arrow(ax, 2.0, y+0.65, 2.5, y+0.65)
        _arrow(ax, 4.9, y+0.65, 5.4, 4.25, color=ARCA_DARK, lw=1.5)
    _arrow(ax, 7.4, 4.25, 8.0, 4.2)

    # Query side (parte derecha inferior)
    _box(ax, 0.2, 0.2, 1.8, 1.2, "👤 query\nusuario", fill=ARCA_DARK, fontsize=10)
    _box(ax, 2.5, 0.2, 2.4, 1.2, "embed query", fill=ARCA_GREEN, fontsize=10)
    _arrow(ax, 2.0, 0.8, 2.5, 0.8)
    _arrow(ax, 4.9, 0.8, 8.0, 3.0, color=ARCA_RED, lw=2)
    _arrow(ax, 9.1, 3.0, 11.0, 1.1, color=ARCA_RED, lw=2)
    _box(ax, 10.8, 0.2, 2.2, 1.2, "top-K\ncontexto", fill=ARCA_RED, fontsize=10)

    # LLM final
    _arrow(ax, 13.0, 0.8, 13.0, 3.0, color=ARCA_DARK, lw=2)
    _box(ax, 11.5, 3.0, 2.4, 1.3, "LLM\nLlama 3.1-8B", fill=ARCA_DARK, fontsize=10)
    _arrow(ax, 12.7, 4.3, 12.7, 5.2)
    _box(ax, 11.3, 5.2, 2.6, 1.2, "Respuesta\n+ cita", fill=ARCA_GREEN, fontsize=10)

    ax.text(7, 6.6, "Asistente de mantenimiento Arca · 3 ingestas + 1 query",
            ha="center", fontsize=13, fontweight="bold", color=ARCA_DARK)
    save(fig, "fig_rag_multimodal.png")


# =============================================================================
if __name__ == "__main__":
    print("Generando figuras clase 36…")
    fig_arco()
    fig_pdf_pipeline()
    fig_imagen_pipeline()
    fig_chroma_anatomy()
    fig_rag_multimodal()
    print("✓ listo")
