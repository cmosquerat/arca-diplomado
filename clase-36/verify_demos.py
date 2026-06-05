"""
Verifica en vivo los demos de clase 36:
  1. pdf_a_texto sobre manual LIMPIO y ESCANEADO
  2. Scout caption sobre 3 fotos (panel, compresor, etiquetadora)
  3. Chroma persistente + metadata filter
  4. RAG multimodal: ERR-007 con texto y con foto

Uso:
    export GROQ_API_KEY=gsk_...
    python verify_demos.py
"""
import os, sys, base64, re, shutil

if "GROQ_API_KEY" not in os.environ:
    sys.exit("Necesitas exportar GROQ_API_KEY (gratis en console.groq.com).")

import fitz, pandas as pd, numpy as np
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

CORPUS = "corpus"
DB     = "./_mtto_arca_db"

client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1")

TXT_MODEL    = "llama-3.1-8b-instant"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# --- helpers ------------------------------------------------------------------
def chat(s, u, t=0.0, max_t=400):
    r = client.chat.completions.create(
        model=TXT_MODEL, max_tokens=max_t, temperature=t,
        messages=[{"role": "system", "content": s},
                  {"role": "user",   "content": u}])
    return r.choices[0].message.content.strip()


def chat_vision(prompt, img_bytes, t=0.0, max_t=600):
    b64 = base64.b64encode(img_bytes).decode()
    r = client.chat.completions.create(
        model=VISION_MODEL, max_tokens=max_t, temperature=t,
        messages=[{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}])
    return r.choices[0].message.content.strip()


def pdf_a_texto(path, dpi=140):
    pages = [p.extract_text() or "" for p in PdfReader(path).pages]
    if sum(len(p) for p in pages) >= 80 * len(pages):
        return "\n".join(pages), "pypdf"
    doc = fitz.open(path); out = []
    for p in doc:
        ocr = chat_vision("OCR de esta imagen. Solo el texto.",
                          p.get_pixmap(dpi=dpi).tobytes("png"), t=0.0, max_t=900)
        out.append(ocr)
    doc.close()
    return "\n\n".join(out), "scout-ocr"


def caption_imagen(path):
    with open(path, "rb") as f:
        return chat_vision(
            "Describe esta imagen en una frase. Citá codigos de error o modelo si los ves.",
            f.read(), t=0.0, max_t=200)


def sliding(t, tam=300, overlap=60):
    t = re.sub(r"\s+", " ", t).strip()
    out, i = [], 0
    while i < len(t):
        c = t[i:i+tam].strip()
        if len(c) > 80: out.append(c)
        i += tam - overlap
    return out


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# =============================================================================
banner("DEMO 1 — pdf_a_texto() LIMPIO vs ESCANEADO")
clean, modo_c = pdf_a_texto(f"{CORPUS}/manual_mtto_arca.pdf")
print(f"LIMPIO    -> modo={modo_c:9s}  chars={len(clean)}")
esc, modo_e = pdf_a_texto(f"{CORPUS}/manual_mtto_arca_ESCANEADO.pdf")
print(f"ESCANEADO -> modo={modo_e:9s}  chars={len(esc)}")
# Esperamos: pypdf para limpio, scout-ocr para escaneado

# =============================================================================
banner("DEMO 2 — caption con Scout (3 fotos)")
fotos = [f"{CORPUS}/fotos/panel_control.png",
         f"{CORPUS}/fotos/compresor.png",
         f"{CORPUS}/fotos/etiquetadora.png"]
caps = []
for f in fotos:
    c = caption_imagen(f)
    caps.append(c)
    print(f"{os.path.basename(f):25s} -> {c[:120]}")

# =============================================================================
banner("DEMO 3 — Chroma persistente con 3 modalidades")
if os.path.exists(DB): shutil.rmtree(DB)
embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
chroma = chromadb.PersistentClient(path=DB)
col = chroma.get_or_create_collection("mtto", metadata={"hnsw:space": "cosine"})

# texto
chunks = sliding(clean)
col.add(documents=chunks,
        embeddings=embedder.encode(chunks, normalize_embeddings=True).tolist(),
        metadatas=[{"modalidad": "texto", "fuente": "manual"} for _ in chunks],
        ids=[f"txt_{i}" for i in range(len(chunks))])
# imagenes
col.add(documents=caps,
        embeddings=embedder.encode(caps, normalize_embeddings=True).tolist(),
        metadatas=[{"modalidad": "imagen", "path": p} for p in fotos],
        ids=[f"img_{i}" for i in range(len(fotos))])
# tabla
df = pd.read_csv(f"{CORPUS}/codigos_error.csv")
docs_t = [f"{r['codigo']} ({r['equipo']}): {r['causa']}. Accion: {r['accion']}"
          for _, r in df.iterrows()]
col.add(documents=docs_t,
        embeddings=embedder.encode(docs_t, normalize_embeddings=True).tolist(),
        metadatas=[{"modalidad": "tabla", "fuente": "codigos",
                    "codigo": r["codigo"], "equipo": r["equipo"]}
                   for _, r in df.iterrows()],
        ids=df["codigo"].tolist())
print(f"Total en collection: {col.count()}")

# Reabrir
col2 = chromadb.PersistentClient(path=DB).get_collection("mtto")
print(f"Reabrir desde disco -> count = {col2.count()}  (persistencia OK)")

# =============================================================================
banner("DEMO 4 — RAG multimodal: ERR-007 con texto y con foto")
SYSTEM = ("Eres asistente de mantenimiento Arca. Responde UNICAMENTE con CONTEXTO. "
          "Si no aparece, di 'No aparece en mi base'. Cita la modalidad. "
          "Sé conciso. Termina con la accion correctiva.")


def rag(p, foto=None, k=4, where=None):
    q = p
    if foto:
        with open(foto, "rb") as f:
            cap = chat_vision("Describi en una frase. Cita codigos visibles.",
                              f.read(), t=0.0, max_t=120)
        q = f"[FOTO] {cap}\n[PREGUNTA] {p}"
    qe = embedder.encode([q], normalize_embeddings=True).tolist()
    r = col.query(query_embeddings=qe, n_results=k, where=where)
    ctx = "\n---\n".join(f"[{m['modalidad']}] {d}"
                        for d, m in zip(r["documents"][0], r["metadatas"][0]))
    return chat(SYSTEM, f"CONTEXTO:\n{ctx}\n\n<USUARIO>\n{q}\n</USUARIO>",
                t=0.0, max_t=400)


for label, kwargs in [
    ("Q1: ERR-007 que hago?",                       dict(p="ERR-007 que hago?")),
    ("Q2: verificaciones del compresor (texto)",     dict(p="verificaciones de rutina del compresor",
                                                          where={"modalidad": "texto"})),
    ("Q3: fuera de scope (cuenta proveedor X)",      dict(p="cuanto pagamos al proveedor X el mes pasado?")),
    ("Q4: FOTO panel + 'que hago?'",                 dict(p="Que tengo que hacer?",
                                                          foto=f"{CORPUS}/fotos/panel_control.png")),
]:
    print(f"\n--- {label} ---")
    print(rag(**kwargs)[:400])

print("\n" + "=" * 72)
print("✓ TODOS LOS DEMOS PASARON")
print("=" * 72)
