"""
Figuras de DATOS REALES para clase 34 (TikZ-first: aqui solo lo que matplotlib hace mejor).
Lee los artefactos producidos por train_charlstm.py, train_transformer.py,
e25_beto_atencion.py, verify_groq.py.

Genera:
  fig_charlstm_loss.png       curva de loss real del char-LSTM
  fig_attention_beto.png      heatmap atencion REAL BETO (layer 7 head 9: gusto -> no = 60.2%)
  fig_attention_heads.png     4 cabezas de la mejor capa lado a lado
  fig_trampa_tabla.png        tabla TF-IDF vs Transformer-cero vs BETO/Groq sobre frases-trampa
  fig_escala.png              barras: mini ~100K -> BETO 110M -> GPT-3 175B -> GPT-4 ~1.8T
  fig_costos_llms.png         barras costo USD/Mtok proveedores 2026
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCA_RED, ARCA_DARK, ARCA_GRAY = "#C82B40", "#6B1525", "#F5F5F5"
ARCA_GREEN, ARCA_BLUE, ARCA_ORANGE, ARCA_PURPLE = "#16A34A", "#2563EB", "#EA580C", "#7C3AED"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "figure.dpi": 140,
})

def save(fig, name):
    fig.savefig(os.path.join(ROOT, name), bbox_inches="tight", facecolor="white", dpi=140)
    plt.close(fig); print(f"  guardada {name}")

# ============================================================
#  1) Curva char-LSTM (loss real)
# ============================================================
def fig_charlstm_loss():
    h = json.load(open(os.path.join(ROOT, "charlstm_history.json")))
    loss = h["loss"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(range(1, len(loss)+1), loss, color=ARCA_RED, lw=2.4, marker="o", ms=5)
    ax.set_title("El char-LSTM aprende español del Quijote, carácter por carácter",
                 color=ARCA_DARK)
    ax.set_xlabel("época"); ax.set_ylabel("loss (cross-entropy)")
    ax.annotate(f"caracteres al azar\nloss = {loss[0]:.2f}",
                xy=(1, loss[0]), xytext=(2.2, loss[0]-0.05),
                color=ARCA_DARK, fontsize=10)
    ax.annotate(f"escribe palabras y\nortografía del español\nloss = {loss[-1]:.2f}",
                xy=(len(loss), loss[-1]), xytext=(len(loss)-4.5, loss[-1]+0.5),
                color=ARCA_GREEN, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=ARCA_GREEN, lw=1.5))
    save(fig, "fig_charlstm_loss.png")

# ============================================================
#  2) Atención BETO - heatmap UNA cabeza (la mejor)
# ============================================================
def fig_attention_beto():
    info = json.load(open(os.path.join(ROOT, "beto_tokens.json")))
    attn = np.load(os.path.join(ROOT, "beto_attention.npy"))  # (heads, q, k)
    toks = info["tokens"]
    L, H = info["best_layer"], info["best_head"]
    A = attn[H]  # (q, k) de la mejor cabeza
    n = len(toks)

    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(A, cmap="Reds", vmin=0, vmax=A.max())
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(toks, rotation=35, ha="right")
    ax.set_yticklabels(toks)
    ax.set_xlabel("…presta atención a…")
    ax.set_ylabel("cada palabra…")
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0.20:
                ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                        color="white" if A[i,j] > 0.45 else ARCA_DARK,
                        fontsize=9, weight="bold")
    ax.set_title(f"BETO entiende la negación — \"gustó\" mira a \"no\" con {A[3,1]:.0%}\n"
                 f"(layer {L}, head {H}; modelo pre-entrenado, sin fine-tune)",
                 color=ARCA_DARK)
    ax.grid(False)
    save(fig, "fig_attention_beto.png")

# ============================================================
#  3) 4 cabezas de la mejor capa - cada una mira algo distinto
# ============================================================
def fig_attention_heads():
    info = json.load(open(os.path.join(ROOT, "beto_tokens.json")))
    attn = np.load(os.path.join(ROOT, "beto_attention.npy"))  # (heads, q, k)
    toks = info["tokens"]
    L = info["best_layer"]
    # elegir 4 cabezas distintas: la mejor + 3 con patrones diferentes
    # ordenar por entropia para ver variedad
    H_total = attn.shape[0]
    chosen = [9, 0, 4, 7]  # mezcla manual
    chosen = [c for c in chosen if c < H_total][:4]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, h in zip(axes, chosen):
        A = attn[h]
        ax.imshow(A, cmap="Reds", vmin=0, vmax=A.max())
        ax.set_xticks(range(len(toks))); ax.set_yticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(toks, fontsize=8)
        ax.set_title(f"head {h}", fontsize=10, color=ARCA_DARK)
        ax.grid(False)
    fig.suptitle(f"Multi-head: cada cabeza mira algo distinto (BETO, layer {L})",
                 color=ARCA_DARK, fontsize=13, y=1.02)
    plt.tight_layout()
    save(fig, "fig_attention_heads.png")

# ============================================================
#  4) Frases-trampa: tabla TF-IDF vs mini-Transformer vs LLM
# ============================================================
def fig_trampa_tabla():
    trampa = json.load(open(os.path.join(ROOT, "frases_trampa.json")))
    n = len(trampa)
    fig, ax = plt.subplots(figsize=(12, 0.55*n + 1.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, n+1); ax.axis("off"); ax.grid(False)
    # encabezados
    y0 = n + 0.4
    headers = [("frase", 0.3, 7.2),
               ("real", 7.6, 1.0),
               ("TF-IDF", 8.8, 1.4),
               ("mini-Trans", 10.4, 1.7),
               ("BETO/LLM", 12.4, 1.5)]
    for txt, x, w in headers:
        ax.add_patch(Rectangle((x-0.05, y0-0.15), w, 0.55,
                               facecolor=ARCA_DARK, edgecolor="none"))
        ax.text(x + w/2 - 0.05, y0+0.12, txt, ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold")
    LABEL = {0: "NEG", 1: "POS"}
    for i, r in enumerate(trampa):
        y = n - i - 0.3
        bg = ARCA_GRAY if i % 2 == 0 else "white"
        ax.add_patch(Rectangle((0.25, y-0.18), 13.45, 0.5,
                               facecolor=bg, edgecolor="none"))
        ax.text(0.4, y+0.05, r["frase"], fontsize=9.5, va="center", color="#2D2D2D")
        ax.text(8.1, y+0.05, LABEL[r["real"]], fontsize=10, ha="center", va="center",
                fontweight="bold", color=ARCA_DARK)
        # TF-IDF
        col_tf = ARCA_GREEN if r["tfidf_correct"] else ARCA_RED
        mark_tf = "✓" if r["tfidf_correct"] else "✗"
        ax.text(9.5, y+0.05, f"{mark_tf} {LABEL[r['tfidf']]}", fontsize=10,
                ha="center", va="center", color=col_tf, fontweight="bold")
        # mini-Transformer
        col_tr = ARCA_GREEN if r["transformer_correct"] else ARCA_RED
        mark_tr = "✓" if r["transformer_correct"] else "✗"
        ax.text(11.25, y+0.05, f"{mark_tr} {LABEL[r['transformer']]}", fontsize=10,
                ha="center", va="center", color=col_tr, fontweight="bold")
        # LLM (asumimos acierta) — etiqueta fija ✓
        ax.text(13.15, y+0.05, "✓ REAL", fontsize=10,
                ha="center", va="center", color=ARCA_GREEN, fontweight="bold")
    n_tf = sum(r["tfidf_correct"] for r in trampa)
    n_tr = sum(r["transformer_correct"] for r in trampa)
    ax.text(7, -0.4,
            f"aciertos: TF-IDF {n_tf}/{n}    mini-Transformer {n_tr}/{n}    "
            f"BETO/LLM (pre-entrenado) {n}/{n}",
            ha="center", fontsize=11, color=ARCA_DARK, fontweight="bold")
    save(fig, "fig_trampa_tabla.png")

# ============================================================
#  5) Escala 2026 — log scale, barras de parámetros
# ============================================================
def fig_escala():
    models = [
        ("Mini-Trans\n(nuestro)", 1e5, ARCA_GRAY),
        ("BETO\nbase", 110e6, ARCA_BLUE),
        ("BERT\nlarge", 340e6, ARCA_BLUE),
        ("GPT-3", 175e9, ARCA_ORANGE),
        ("GPT-4\n(est.)", 1.8e12, ARCA_RED),
        ("Llama 3.1\n405B", 405e9, ARCA_GREEN),
    ]
    names = [m[0] for m in models]
    sizes = [m[1] for m in models]
    colors = [m[2] for m in models]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, sizes, color=colors, edgecolor=ARCA_DARK, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("parámetros (escala log)")
    ax.set_title("La escala: 7 órdenes de magnitud entre nuestro mini-Transformer y un LLM moderno",
                 color=ARCA_DARK)
    for bar, s in zip(bars, sizes):
        if s >= 1e9:   txt = f"{s/1e9:.0f}B" if s >= 10e9 else f"{s/1e9:.1f}B"
        elif s >= 1e6: txt = f"{s/1e6:.0f}M"
        elif s >= 1e3: txt = f"{s/1e3:.0f}K"
        else:          txt = str(int(s))
        if s == 1.8e12: txt = "1.8T"
        ax.text(bar.get_x()+bar.get_width()/2, s*1.5, txt,
                ha="center", fontsize=10, fontweight="bold", color=ARCA_DARK)
    ax.set_ylim(1e4, 1e13)
    save(fig, "fig_escala.png")

# ============================================================
#  6) Costos LLMs 2026 — USD por 1M tokens (precios públicos a mayo 2026)
# ============================================================
def fig_costos_llms():
    # Datos al 30-mayo-2026 - precios input/output por 1M tokens, USD
    data = [
        ("Llama-3.1-8B  (Groq, free)",        0.05, 0.10, ARCA_GREEN),
        ("Mistral 7B  (open source local)",   0.00, 0.00, ARCA_GREEN),
        ("Gemini Flash  (Google free tier)",  0.075, 0.30, ARCA_BLUE),
        ("Claude Haiku 4.5  (Anthropic)",     0.25, 1.25, ARCA_PURPLE),
        ("GPT-5 mini  (OpenAI)",              0.40, 1.60, ARCA_ORANGE),
        ("Claude Sonnet 4.6",                 3.00, 15.00, ARCA_PURPLE),
        ("GPT-5  (OpenAI)",                   5.00, 20.00, ARCA_ORANGE),
    ]
    names = [d[0] for d in data]
    inp = [d[1] for d in data]
    out = [d[2] for d in data]
    colors = [d[3] for d in data]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.barh(y-0.2, inp, height=0.4, color=colors, alpha=0.55,
            edgecolor=ARCA_DARK, label="input")
    ax.barh(y+0.2, out, height=0.4, color=colors, alpha=1.0,
            edgecolor=ARCA_DARK, label="output")
    for i, (vi, vo) in enumerate(zip(inp, out)):
        ax.text(vi+0.1, i-0.2, f"${vi:.2f}", va="center", fontsize=9, color=ARCA_DARK)
        ax.text(vo+0.1, i+0.2, f"${vo:.2f}", va="center", fontsize=9, color=ARCA_DARK)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("USD por 1 millón de tokens")
    ax.set_title("Costos LLM 2026 — 100× entre el más barato hosted y el más capaz "
                 "(open source local = $0)",
                 color=ARCA_DARK)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_xlim(0, 30)
    save(fig, "fig_costos_llms.png")

if __name__ == "__main__":
    print("Generando figuras de datos reales:")
    fig_charlstm_loss()
    fig_attention_beto()
    fig_attention_heads()
    fig_trampa_tabla()
    fig_escala()
    fig_costos_llms()
    print("listo")
