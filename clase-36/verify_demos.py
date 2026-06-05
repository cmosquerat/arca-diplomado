"""
verify_demos.py — reproduce los demos clave de clase 36 en vivo.

Uso:
    export GROQ_API_KEY=gsk_...
    python verify_demos.py

Reproduce 4 demos:
  1. PyPDF crudo sobre el reporte real → texto sucio
  2. pymupdf4llm sobre el mismo reporte → markdown limpio + figuras
  3. Llama 4 Scout captioning sobre 3 figuras del reporte
  4. RAG end-to-end: indexar texto + captions en Chroma, hacer 4 queries
"""
import os, sys, base64, re, glob, shutil, time
if "GROQ_API_KEY" not in os.environ:
    sys.exit("Necesitas exportar GROQ_API_KEY (gratis en console.groq.com).")

import numpy as np
from openai import OpenAI
from pypdf import PdfReader
import pymupdf4llm
from sentence_transformers import SentenceTransformer
import chromadb

# El verify_demos.py vive en clase-36/. Usamos paths relativos.
HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "corpus/reporte_arca_2025_capitulo.pdf")
DB  = os.path.join(HERE, "_arca_db_verify")

client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1")
TXT_MODEL    = "llama-3.1-8b-instant"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def chat(s, u, t=0.0, max_t=400):
    r = client.chat.completions.create(model=TXT_MODEL, max_tokens=max_t, temperature=t,
        messages=[{"role": "system", "content": s},
                  {"role": "user",   "content": u}])
    return r.choices[0].message.content.strip()


def chat_vision(prompt, image_bytes, t=0.0, max_t=300):
    b64 = base64.b64encode(image_bytes).decode()
    r = client.chat.completions.create(model=VISION_MODEL, max_tokens=max_t, temperature=t,
        messages=[{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}])
    return r.choices[0].message.content.strip()


def caption(path, prompt=None):
    if prompt is None:
        prompt = "Describí en 1-2 frases. Citá codigos, modelos o marcas si los ves."
    with open(path, "rb") as f:
        return chat_vision(prompt, f.read(), t=0.0, max_t=180)


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# =============================================================================
banner("DEMO 1 — PyPDF crudo sobre el reporte (texto sucio)")
pages = [p.extract_text() or "" for p in PdfReader(PDF).pages]
print(f"PyPDF: {len(pages)} páginas, {sum(len(p) for p in pages):,} chars")
print(f"\nPreview de la página 44 (la 4 del capítulo):\n{pages[3][:500]!r}")
print("\n→ Fijate cómo 'Sostenible' queda partido en 'S\\nostenible'.")

# =============================================================================
banner("DEMO 2 — pymupdf4llm sobre el mismo reporte")
FIGS = "/tmp/_verify_figs"
shutil.rmtree(FIGS, ignore_errors=True)
os.makedirs(FIGS, exist_ok=True)
t0 = time.time()
md_full = pymupdf4llm.to_markdown(PDF, write_images=True,
                                   image_path=FIGS, image_format="png", dpi=110)
print(f"pymupdf4llm: {len(md_full):,} chars markdown · "
      f"{len(os.listdir(FIGS))} figs extraídas · {time.time()-t0:.1f}s")

# =============================================================================
banner("DEMO 3 — Llama 4 Scout captioning sobre 3 figuras")
figs_big = sorted(glob.glob(f"{FIGS}/*.png"), key=os.path.getsize, reverse=True)[:3]
caps = []
for f in figs_big:
    c = caption(f)
    caps.append(c)
    print(f"\n{os.path.basename(f)} ({os.path.getsize(f)/1024:.0f} KB)")
    print(f"  -> {c}")

# =============================================================================
banner("DEMO 4 — RAG end-to-end (Chroma + texto + captions)")
embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
if os.path.exists(DB):
    shutil.rmtree(DB)
chroma = chromadb.PersistentClient(path=DB)
col = chroma.get_or_create_collection("rep", metadata={"hnsw:space": "cosine"})

def sliding(t, tam=400, overlap=80):
    t = re.sub(r"\s+", " ", t).strip()
    out, i = [], 0
    while i < len(t):
        c = t[i:i+tam].strip()
        if len(c) > 80: out.append(c)
        i += tam - overlap
    return out

chunks = sliding(md_full)
col.add(documents=chunks,
        embeddings=embedder.encode(chunks, normalize_embeddings=True).tolist(),
        metadatas=[{"modalidad": "texto"} for _ in chunks],
        ids=[f"t{i}" for i in range(len(chunks))])
col.add(documents=caps,
        embeddings=embedder.encode(caps, normalize_embeddings=True).tolist(),
        metadatas=[{"modalidad": "figura", "path": f} for f in figs_big],
        ids=[f"i{i}" for i in range(len(caps))])
print(f"docs en Chroma: {col.count()}  ({len(chunks)} texto + {len(caps)} figura)")

SYS = ("Eres asistente sobre el Reporte Anual Arca 2025. "
       "Responde UNICAMENTE con CONTEXTO. Si no aparece, di 'No aparece en mi base'. "
       "Cita modalidad (texto/figura). Sé conciso.")

def rag(q, k=4):
    qe = embedder.encode([q], normalize_embeddings=True).tolist()
    r = col.query(query_embeddings=qe, n_results=k)
    ctx = "\n---\n".join(f"[{m['modalidad']}] {d}"
                        for d, m in zip(r["documents"][0], r["metadatas"][0]))
    return chat(SYS, f"CONTEXTO:\n{ctx}\n\nPREGUNTA: {q}", t=0.0, max_t=400)


for q in [
    "¿En qué índices ESG está Arca Continental?",
    "¿Qué iniciativas de sostenibilidad tienen?",
    "¿Arca tiene presencia en mercados de Latinoamérica?",
    "¿Cuánto cobramos al proveedor X el mes pasado?",
]:
    print(f"\n[Q] {q}")
    print(f"[A] {rag(q)[:400]}")

print("\n" + "=" * 72)
print("✓ TODOS LOS DEMOS PASARON")
print("=" * 72)
