"""
Figuras para clase 35 — Llevando tu LLM a producción.

Genera:
  fig_rag_pipeline.png            (F41) — 3 cajas Indexar/Recuperar/Generar
  fig_chatbot_arca_blindado.png   (F50) — arquitectura final con 3 defensas

Estilo: paleta Arca (arcaRed, arcaDark, arcaGray + codeBlue/Orange/Green/Purple).
No usa TensorFlow (TF+matplotlib deadlockea en macOS — lección de clase-33).
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


def _round_box(ax, x, y, w, h, text, fill, fontcolor="white",
               fontsize=12, weight="bold"):
    box = mp.FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.15",
                            facecolor=fill, edgecolor=ARCA_DARK, lw=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=fontcolor, fontsize=fontsize, fontweight=weight)


def _arrow(ax, x1, y1, x2, y2, color=ARCA_DARK, lw=2.4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))


# =============================================================================
#  F41 — RAG pipeline (3 cajas: Indexar / Recuperar / Generar)
# =============================================================================
def fig_rag_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")

    # 3 cajas grandes
    _round_box(ax, 0.4, 2.3, 3.6, 1.6, "1. INDEXAR",
               fill=ARCA_BLUE, fontsize=16)
    _round_box(ax, 4.7, 2.3, 3.6, 1.6, "2. RECUPERAR",
               fill=ARCA_ORANGE, fontsize=16)
    _round_box(ax, 9.0, 2.3, 3.6, 1.6, "3. GENERAR",
               fill=ARCA_GREEN, fontsize=16)

    # Subtítulos debajo
    for x, sub in [(0.4 + 1.8, "docs → embeddings\nguardar en store"),
                   (4.7 + 1.8, "query → embedding\ntop-K más cercanos"),
                   (9.0 + 1.8, "context + query → LLM\nrespuesta con citas")]:
        ax.text(x, 1.7, sub, ha="center", va="top",
                fontsize=10, color=TEXT_MUTED)

    # Inputs arriba de cada caja
    for x, lbl in [(0.4 + 1.8, "manuales · tickets · políticas"),
                   (4.7 + 1.8, "pregunta del usuario"),
                   (9.0 + 1.8, "respuesta blindada")]:
        ax.text(x, 4.6, lbl, ha="center", va="bottom",
                fontsize=10, color=ARCA_DARK, style="italic")

    # Flechas dentro: indexar -> recuperar (offline) y recuperar -> generar (online)
    _arrow(ax, 4.05, 3.1, 4.65, 3.1, ARCA_DARK)
    _arrow(ax, 8.35, 3.1, 8.95, 3.1, ARCA_DARK)

    # Banda inferior con la moraleja
    ax.text(6.5, 0.6,
            "Indexar es offline.   Recuperar + Generar es por cada query.",
            ha="center", va="center", fontsize=11, color=ARCA_DARK,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=ARCA_GRAY,
                     edgecolor=ARCA_RED, lw=1.5))

    save(fig, "fig_rag_pipeline.png")


# =============================================================================
#  F50 — Arquitectura chatbot Arca blindado
# =============================================================================
def fig_chatbot_arca_blindado():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.5); ax.axis("off")

    # Capa de entrada (usuario)
    _round_box(ax, 0.3, 5.6, 2.3, 1.0,
               "Usuario\n(pregunta libre)",
               fill=ARCA_DARK, fontsize=10)

    # Defensa 1: sanitizer
    _round_box(ax, 3.2, 5.6, 2.6, 1.0,
               "Sanitizer\n(anonimiza PII)",
               fill=ARCA_RED, fontsize=10)
    ax.text(4.5, 4.9, "Defensa C\n(anti-filtrado)",
            ha="center", va="top", fontsize=9, color=ARCA_RED,
            fontweight="bold")

    # Defensa 2: wrapper con delimitadores
    _round_box(ax, 6.4, 5.6, 2.8, 1.0,
               "Wrapper\n<USUARIO>…</USUARIO>",
               fill=ARCA_RED, fontsize=10)
    ax.text(7.8, 4.9, "Defensa B\n(anti-injection)",
            ha="center", va="top", fontsize=9, color=ARCA_RED,
            fontweight="bold")

    # Retrieval: embedding + top-K
    _round_box(ax, 9.8, 5.6, 2.3, 1.0,
               "Embedding\n(query → vec)",
               fill=ARCA_ORANGE, fontsize=10)
    _round_box(ax, 9.8, 3.9, 2.3, 1.0,
               "Top-K docs\n(coseno)",
               fill=ARCA_ORANGE, fontsize=10)
    _arrow(ax, 10.95, 5.6, 10.95, 4.9, ARCA_ORANGE, lw=2.0)

    # Store
    _round_box(ax, 12.4, 3.9, 1.4, 1.0,
               "Vector\nstore",
               fill=ARCA_BLUE, fontsize=10)
    _arrow(ax, 12.4, 4.4, 12.1, 4.4, ARCA_BLUE, lw=2.0)

    # System prompt defensivo
    _round_box(ax, 5.5, 3.0, 4.0, 1.0,
               "System prompt blindado\n(rol + reglas anti-alucinación)",
               fill=ARCA_RED, fontsize=10)
    ax.text(7.5, 2.3, "Defensa A\n(anti-alucinación)",
            ha="center", va="top", fontsize=9, color=ARCA_RED,
            fontweight="bold")

    # LLM (Groq)
    _round_box(ax, 5.5, 0.7, 4.0, 1.2,
               "LLM\n(Groq Llama 3.1-8B)",
               fill=ARCA_DARK, fontsize=11)

    # Validador de salida
    _round_box(ax, 10.5, 0.7, 3.0, 1.2,
               "Validador salida\n(regex + filtros)",
               fill=ARCA_RED, fontsize=10)
    ax.text(12.0, 0.4, "Defensa C\n(de nuevo)",
            ha="center", va="top", fontsize=8, color=ARCA_RED,
            fontweight="bold")

    # Flechas principales
    _arrow(ax, 2.6, 6.1, 3.2, 6.1)       # usuario → sanitizer
    _arrow(ax, 5.8, 6.1, 6.4, 6.1)       # sanitizer → wrapper
    _arrow(ax, 9.2, 6.1, 9.8, 6.1)       # wrapper → embedding
    _arrow(ax, 10.95, 3.9, 10.95, 3.4, ARCA_ORANGE)  # top-K → system
    _arrow(ax, 10.5, 3.5, 9.5, 3.5, ARCA_ORANGE)
    _arrow(ax, 7.5, 3.0, 7.5, 1.9)       # system → LLM
    _arrow(ax, 9.5, 1.3, 10.5, 1.3)      # LLM → validador
    _arrow(ax, 12.0, 1.9, 12.0, 5.6, ARCA_GREEN, lw=1.8)  # respuesta vuelve

    # Salida (etiqueta lateral)
    ax.text(13.7, 4.0, "Respuesta\nal usuario",
            ha="center", va="center", fontsize=10, color=ARCA_GREEN,
            fontweight="bold", rotation=90)

    # Título superior
    ax.text(7, 7.2, "Chatbot Arca blindado · arquitectura mínima viable",
            ha="center", va="center", fontsize=13, color=ARCA_DARK,
            fontweight="bold")

    save(fig, "fig_chatbot_arca_blindado.png")


# =============================================================================
#  DATASET UNIVERSAL para enseñar embeddings (B3)
#  Tres temas obvios — cualquier audiencia los reconoce sin contexto Arca.
# =============================================================================
BEBIDAS = [   # nombre conservado por compatibilidad; ahora son oraciones tematicas
    ("Correr al amanecer",
     "Me encanta salir a correr por la manana, el aire fresco me despierta."),
    ("Trotar diariamente",
     "Salir a trotar todos los dias es la mejor rutina para mantenerse en forma."),
    ("Cardio para la salud",
     "Hacer ejercicio cardiovascular regularmente mejora la salud del corazon."),
    ("Pizza italiana",
     "La pizza italiana tradicional lleva masa fina, tomate y queso mozzarella."),
    ("Pasta casera",
     "Una buena pasta italiana se prepara con tomate fresco y albahaca verde."),
    ("Sushi japones",
     "El sushi japones autentico requiere arroz vinagrado y pescado muy fresco."),
    ("Python para datos",
     "Python es el lenguaje ideal para hacer analisis de datos y ciencia de datos."),
    ("Aprender machine learning",
     "Aprender machine learning requiere matematicas, programacion y mucha practica."),
]
CLUSTER = [0, 0, 0, 1, 1, 1, 2, 2]   # 0=deportes, 1=cocina, 2=tecnologia
CLUSTER_COLOR = {0: ARCA_BLUE, 1: ARCA_ORANGE, 2: ARCA_GREEN}
CLUSTER_NAME  = {0: "Deportes", 1: "Cocina", 2: "Tecnologia"}

_EMBED_CACHE = os.path.join(ROOT, "_bebidas_emb.npy")


_MODEL_NAME = None


def _load_model():
    """Open-source multilingual sentence embeddings. Orden de preferencia:
       1. Alibaba-NLP/gte-multilingual-base   (305MB, 768d, top MTEB, requires trust_remote_code)
       2. paraphrase-multilingual-mpnet-base  (970MB, 768d, sentence-transformers clasico)
       3. paraphrase-multilingual-MiniLM-L12  (118MB, 384d, ultimo recurso)
    """
    global _MODEL_NAME
    from sentence_transformers import SentenceTransformer
    candidates = [
        ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "mpnet", {}),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "minilm", {}),
    ]
    for repo, name, kwargs in candidates:
        try:
            m = SentenceTransformer(repo, **kwargs)
            _MODEL_NAME = name
            print(f"  modelo: {repo}")
            return m
        except Exception as e:
            print(f"  {repo} no disponible ({type(e).__name__}: {str(e)[:80]})")
    raise RuntimeError("Ningun modelo de embeddings disponible")


def _encode_docs(model, textos):
    """Prefija para e5 si aplica."""
    if _MODEL_NAME == "e5":
        textos = [f"passage: {t}" for t in textos]
    return model.encode(textos, normalize_embeddings=True)


def _encode_query(model, q):
    if _MODEL_NAME == "e5":
        q = f"query: {q}"
    return model.encode([q], normalize_embeddings=True)[0]


def get_embeddings():
    """Embed BEBIDAS once, cache to disk for fast re-runs."""
    import numpy as np
    if os.path.exists(_EMBED_CACHE):
        return np.load(_EMBED_CACHE)
    model = _load_model()
    textos = [f"{n}. {d}" for n, d in BEBIDAS]
    emb = _encode_docs(model, textos)
    np.save(_EMBED_CACHE, emb)
    print(f"  embeddings shape={emb.shape}, cache guardado")
    return emb


# =============================================================================
#  F31 — Concepto: sentence embedding = codigo de barras semantico
# =============================================================================
def fig_sentence_emb_concept():
    import numpy as np
    emb = get_embeddings()
    # 2 frases muy distintas (deportes vs cocina) y 2 muy parecidas (parafrasis de correr/trotar)
    i1, i2 = 0, 3    # "Correr al amanecer" vs "Pizza italiana"
    v1, v2 = emb[i1, :64], emb[i2, :64]
    cos_far = float(np.dot(emb[i1], emb[i2]))
    j1, j2 = 0, 1    # "Correr al amanecer" vs "Trotar diariamente" — parafrasis
    w1, w2 = emb[j1, :64], emb[j2, :64]
    cos_near = float(np.dot(emb[j1], emb[j2]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 5.6),
                             gridspec_kw={"height_ratios": [1, 1],
                                          "width_ratios":  [4, 1]})
    fig.subplots_adjust(hspace=0.55, wspace=0.15)

    def _bar(ax, vec, name, color):
        ax.imshow(vec.reshape(1, -1), aspect="auto", cmap="RdBu_r",
                  vmin=-0.25, vmax=0.25)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_title(name, fontsize=11, color=color, loc="left", pad=4)
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(1.4)

    _bar(axes[0, 0], v1, BEBIDAS[i1][0], ARCA_BLUE)
    _bar(axes[1, 0], v2, BEBIDAS[i2][0], ARCA_GREEN)

    axes[0, 1].axis("off")
    axes[1, 1].axis("off")
    axes[0, 1].text(0.5, 0.5, f"cos = {cos_far:.2f}\n(temas distintos)",
                    ha="center", va="center", fontsize=13, color=ARCA_DARK,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=ARCA_GRAY,
                              edgecolor=ARCA_RED, lw=1.5))
    axes[1, 1].text(0.5, 0.5, f"vs \"Trotar diariamente\"\ncos = {cos_near:.2f}\n(parafrasis)",
                    ha="center", va="center", fontsize=12, color=ARCA_DARK,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=ARCA_GRAY,
                              edgecolor=ARCA_GREEN, lw=1.5))

    dims = emb.shape[1]
    fig.suptitle(f"Cada bebida es un \"codigo de barras\" semantico de {dims} dims",
                 fontsize=13, color=ARCA_DARK, fontweight="bold", y=1.02)
    save(fig, "fig_sentence_emb_concept.png")


# =============================================================================
#  F32 — Pipeline: oracion -> encoder -> mean pooling -> vector
# =============================================================================
def fig_emb_pipeline_transformer():
    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.5); ax.axis("off")

    _round_box(ax, 0.2, 1.6, 2.4, 1.3,
               "Oracion\n\"Salir a trotar\ntodos los dias...\"",
               fill=ARCA_DARK, fontsize=9.5)
    _round_box(ax, 3.0, 1.6, 2.0, 1.3,
               "Tokenizer\n(BPE)", fill=ARCA_BLUE, fontsize=10)
    _round_box(ax, 5.5, 1.4, 2.6, 1.7,
               "Transformer\nencoder\n(pre-entrenado)",
               fill=ARCA_ORANGE, fontsize=10)
    # Tokens (vectores por token saliendo del encoder)
    for k, tok in enumerate(["v_1", "v_2", "v_3", "v_n"]):
        _round_box(ax, 8.4 + k*0.55, 1.7, 0.5, 1.1, tok,
                   fill=ARCA_GRAY, fontcolor=ARCA_DARK, fontsize=9)
    _round_box(ax, 10.9, 1.6, 1.4, 1.3,
               "mean\npool", fill=ARCA_PURPLE, fontsize=10)
    _round_box(ax, 12.5, 1.6, 1.3, 1.3,
               "vector\n768 d",
               fill=ARCA_GREEN, fontsize=10.5)

    # Flechas
    for x1, x2 in [(2.6, 3.0), (5.0, 5.5), (8.1, 8.4), (10.6, 10.9), (12.3, 12.5)]:
        _arrow(ax, x1, 2.25, x2, 2.25)

    # Banda inferior
    ax.text(7, 0.5,
            "No entrenas: usas un modelo pre-entrenado en miles de millones de oraciones.",
            ha="center", va="center", fontsize=10.5, color=ARCA_DARK,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=ARCA_GRAY,
                     edgecolor=ARCA_RED, lw=1.5))
    # Titulo
    ax.text(7, 4.1,
            "De oracion a vector unico en 4 pasos",
            ha="center", va="center", fontsize=13, color=ARCA_DARK,
            fontweight="bold")
    save(fig, "fig_emb_pipeline_transformer.png")


# =============================================================================
#  F34 — Proyeccion 2D de las 8 bebidas Arca (PCA)
# =============================================================================
def fig_emb_bebidas_2d():
    import numpy as np
    from sklearn.decomposition import PCA
    emb = get_embeddings()
    coords = PCA(n_components=2, random_state=0).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(11, 6.4))
    for ci in sorted(set(CLUSTER)):
        pts = [coords[i] for i, c in enumerate(CLUSTER) if c == ci]
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=320, color=CLUSTER_COLOR[ci],
                   edgecolor=ARCA_DARK, linewidth=1.4, alpha=0.85,
                   label=CLUSTER_NAME[ci], zorder=3)
    for i, (n, _) in enumerate(BEBIDAS):
        ax.annotate(n, coords[i], xytext=(8, 6), textcoords="offset points",
                    fontsize=10.5, color=ARCA_DARK, fontweight="bold",
                    zorder=4)
    ax.set_xlabel("PCA componente 1", fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel("PCA componente 2", fontsize=10, color=TEXT_MUTED)
    ax.set_title("8 oraciones proyectadas en 2D — tres clusters semanticos obvios",
                 color=ARCA_DARK)
    ax.legend(loc="best", frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    save(fig, "fig_emb_oraciones_2d.png")


# =============================================================================
#  F35 — Vecinos de la consulta "bebida sin azucar" + umbral
# =============================================================================
def fig_vecinos_bebidas():
    import numpy as np
    from sklearn.decomposition import PCA
    model = _load_model()
    emb = get_embeddings()
    q = _encode_query(model, "hacer deporte por la manana")
    # Apilamos para PCA conjunto
    full = np.vstack([emb, q.reshape(1, -1)])
    coords = PCA(n_components=2, random_state=0).fit_transform(full)
    bebidas_xy = coords[:-1]
    q_xy = coords[-1]
    sims = emb @ q
    order = np.argsort(-sims)
    top3 = set(order[:3].tolist())

    fig, ax = plt.subplots(figsize=(11, 6.4))
    for i, (n, _) in enumerate(BEBIDAS):
        is_top = i in top3
        ax.scatter(*bebidas_xy[i], s=360 if is_top else 220,
                   color=CLUSTER_COLOR[CLUSTER[i]],
                   edgecolor=ARCA_RED if is_top else ARCA_DARK,
                   linewidth=2.6 if is_top else 1.2,
                   alpha=0.9, zorder=3)
        ax.annotate(f"{n}\n(cos={sims[i]:.2f})", bebidas_xy[i],
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=9.5,
                    color=ARCA_RED if is_top else ARCA_DARK,
                    fontweight="bold" if is_top else "normal", zorder=4)
    # Query como estrella roja
    ax.scatter(*q_xy, s=520, color=ARCA_RED, marker="*",
               edgecolor=ARCA_DARK, linewidth=1.6, zorder=5,
               label='Query: "hacer deporte por la manana"')
    # Circulo umbral aproximado
    radius = np.linalg.norm(bebidas_xy[order[2]] - q_xy) * 1.08
    circ = mp.Circle(q_xy, radius, fill=False, edgecolor=ARCA_RED,
                     linewidth=2.0, linestyle="--", zorder=2)
    ax.add_patch(circ)
    ax.set_title('La consulta cae cerca de las 3 oraciones de "deportes"',
                 color=ARCA_DARK)
    ax.set_xlabel("PCA 1", fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel("PCA 2", fontsize=10, color=TEXT_MUTED)
    ax.legend(loc="best", frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    save(fig, "fig_vecinos_oraciones.png")


# =============================================================================
#  F38 — Matriz coseno 8x8 entre bebidas Arca
# =============================================================================
def fig_coseno_bebidas_8x8():
    import numpy as np
    emb = get_embeddings()
    sim = emb @ emb.T   # ya normalizado
    n = len(BEBIDAS)
    nombres = [b[0] for b in BEBIDAS]

    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(sim, cmap="Reds", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(nombres, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(nombres, fontsize=10)
    for i in range(n):
        for j in range(n):
            v = sim[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.7 else ARCA_DARK, fontsize=9.5,
                    fontweight="bold")
    # Marcos de cluster
    cluster_bounds = [(0, 3), (3, 6), (6, 8)]
    for (a, b) in cluster_bounds:
        rect = mp.Rectangle((a-0.5, a-0.5), b-a, b-a,
                            fill=False, edgecolor=ARCA_DARK, linewidth=2.4)
        ax.add_patch(rect)
    ax.set_title("Similitud coseno entre las 8 oraciones\n"
                 "(3 bloques diagonales = 3 clusters semanticos)",
                 color=ARCA_DARK, pad=14)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    save(fig, "fig_coseno_oraciones_8x8.png")


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    print("Generando figuras clase 35…")
    fig_rag_pipeline()
    fig_chatbot_arca_blindado()
    print("--- figuras de embeddings (dataset bebidas Arca) ---")
    fig_sentence_emb_concept()
    fig_emb_pipeline_transformer()
    fig_emb_bebidas_2d()
    fig_vecinos_bebidas()
    fig_coseno_bebidas_8x8()
    print("✓ listo")
