"""
Construye Clase_36_RAG_Multimodal.ipynb (v2 sin slop).

Orden andropedagógico:
  Setup
  B0 — RAG: qué, cómo, por qué (recordatorio breve)
  B1 — RAG simple de clase 35 (recap)
  B2 — el dolor: PyPDF sobre el Reporte Arca 2025
  B3 — modelos multimodales (Llama 4 Scout via Groq)
  B4 — captioning de figuras del Reporte
  B5 — pymupdf4llm: una librería que hace el 90 %
  B6 — Asistente Arca v2 (texto + captions juntos en Chroma)
  PBL final
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md   = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =============================================================================
# Portada + setup
# =============================================================================
md("""# Clase 36 — RAG multimodal sobre el Reporte Anual Integrado Arca 2025

*Diplomado en Data Science Aplicada con Python · Arca Continental Ecuador · UDLA*

---

**Hilo del día:**
> En clase 35 armamos un chatbot Arca que cita el Código de Ética. Hoy queremos uno que lea el **Reporte Anual Integrado 2025** — 189 páginas, 415 figuras. ¿Por qué el RAG simple se rompe, qué le ponemos para que funcione, y cuánto código real escribimos vs cuánto le pedimos a una librería profesional?

**Plan:**
- **B2** — ver con nuestros ojos cómo PyPDF rompe el reporte
- **B3** — Llama 4 Scout via Groq: el modelo que "también puede ver"
- **B4** — captioning de las figuras del Reporte Arca
- **B5** — `pymupdf4llm`: una librería que hace el 90 %
- **B6** — el asistente Arca v2 funcionando con texto + figuras""")

md("""## Setup""")

code("""# Colab ----------------------------------------------------------------------
!pip install -q openai sentence-transformers pypdf pymupdf pymupdf4llm chromadb gradio""")

code("""import os, getpass, base64, re, glob, io
import numpy as np, requests
from openai import OpenAI

os.environ["GROQ_API_KEY"] = getpass.getpass("GROQ_API_KEY: ")
client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1")

MODEL_TEXT   = "llama-3.1-8b-instant"
MODEL_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"

def chat(system, user, t=0.0, max_t=400):
    \"\"\"LLM solo texto.\"\"\"
    r = client.chat.completions.create(
        model=MODEL_TEXT,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        temperature=t, max_tokens=max_t)
    return r.choices[0].message.content.strip()

def chat_vision(prompt, image_bytes, t=0.0, max_t=300):
    \"\"\"Llama 4 Scout: texto + imagen (base64).\"\"\"
    b64 = base64.b64encode(image_bytes).decode()
    r = client.chat.completions.create(
        model=MODEL_VISION,
        messages=[{"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}},
        ]}],
        temperature=t, max_tokens=max_t)
    return r.choices[0].message.content.strip()

print(chat("Eres conciso.", "Hola en 4 palabras."))""")

md("""### Descargar el corpus: el reporte real y el capítulo de demo""")
code("""BASE = ("https://raw.githubusercontent.com/cmosquerat/arca-diplomado/"
        "main/clase-36/corpus")
os.makedirs("corpus/figuras_extraidas", exist_ok=True)

# El capítulo del reporte (pp 41–71, 6.6 MB) — para los demos en clase
CAP = "corpus/reporte_arca_2025_capitulo.pdf"
if not os.path.exists(CAP):
    with open(CAP, "wb") as f:
        f.write(requests.get(f"{BASE}/reporte_arca_2025_capitulo.pdf",
                             timeout=60).content)
print(f"capítulo: {os.path.getsize(CAP):,} bytes")""")

# =============================================================================
# B0 — RAG: qué, cómo
# =============================================================================
md("""---

# Bloque 0 — RAG: qué, para qué, cómo combate la alucinación

Esto ya lo viste en clase 35 — un párrafo de refresco.

**RAG** = *Retrieval-Augmented Generation*. Antes de responder, el modelo **recupera los documentos relevantes y los pega en su propio contexto**.

```
pregunta → recupero docs → LLM con contexto → respuesta + cita
```

¿Por qué pega contra la alucinación? Porque el modelo ya no tiene que **recordar** la respuesta — solo **leerla**. Si el system prompt dice *"si no aparece en el contexto, decí 'no aparece'"*, el modelo deja de inventar.""")

# =============================================================================
# B1 — RAG simple de clase 35
# =============================================================================
md("""---

# Bloque 1 — RAG simple de clase 35 (recap rápido)

El chatbot Arca de Ética que funcionó. Lo armamos en 4 piezas: PyPDF + chunking por `\\n\\n` + embeddings con `mpnet` + búsqueda con NumPy. Eso es **lo que vamos a estresar hoy con un PDF más realista**.""")

code("""from pypdf import PdfReader

# El reporte real (capítulo)
pages_pypdf = [p.extract_text() or "" for p in PdfReader("corpus/reporte_arca_2025_capitulo.pdf").pages]
texto_pypdf = "\\n".join(pages_pypdf)
print(f"PyPDF sobre cap. del reporte: {len(texto_pypdf):,} chars en {len(pages_pypdf)} páginas")""")

md("""👆 Sale texto. ¿Es texto **utilizable**? Mirémoslo de cerca.""")

# =============================================================================
# B2 — El dolor
# =============================================================================
md("""---

# Bloque 2 — El dolor: PyPDF sobre el reporte real

El capítulo va de la página 41 a la 71. Foquémonos en la **página 44** del PDF original (la 4 del capítulo): tiene 22 logos, una foto y tres columnas de texto.""")

code("""# Página 44 del PDF -> es la pagina 4 (índice 3) del capítulo
pag = pages_pypdf[3]
print(pag[:1500])""")

md("""👆 **Fijate**:
- "Comité" / "de" / "Sostenibilidad" se separan en líneas con tabs raros.
- "Sostenible" se parte en **"S"** y **"ostenible"** (esa S quedó suelta de la columna).
- "Best-in-Class" se vuelve "Best \\n . \\n in \\n . \\n Class".
- **De las 22 figuras (los logos de FTSE4Good, S&P, MSCI, etc.) no aparece ninguna.** Para el RAG son invisibles.

Si tu pregunta es "¿en qué índices ESG está Arca?" y la respuesta solo está en los **logos visuales**, este PyPDF no la rescata.""")

# =============================================================================
# B3 — Multimodales
# =============================================================================
md("""---

# Bloque 3 — Modelos multimodales: el modelo "también ve"

Hasta ahora nuestros LLMs recibían solo texto. **Llama 4 Scout** acepta texto + imagen en la misma llamada. Es el reemplazo de Llama 3.2 Vision en Groq.

El cliente es el mismo. Lo único que cambia es:
- `model="meta-llama/llama-4-scout-17b-16e-instruct"`
- `content` ahora es una **lista** con texto + image_url (base64)""")

code("""# Bajamos una figura concreta del reporte (mujer con cajas de Coca-Cola, pag. 52)
fig_url = f"{BASE}/figs/fig_mujer_cocacola.png"
fig_local = "corpus/fig_mujer_cocacola.png"
if not os.path.exists(fig_local):
    with open(fig_local, "wb") as f:
        f.write(requests.get(fig_url, timeout=30).content)

from IPython.display import Image, display
display(Image(filename=fig_local, width=300))""")

code("""with open(fig_local, "rb") as f:
    img_bytes = f.read()

respuesta = chat_vision(
    "Describí esta foto en 2 frases. ¿Quién aparece? ¿Qué está haciendo? "
    "¿Qué marcas se ven?",
    img_bytes, t=0.0, max_t=200)
print(respuesta)""")

md("""👆 Eso es captioning. La foto entró como bytes, salió como texto.""")

# =============================================================================
# B4 — Image captioning
# =============================================================================
md("""---

# Bloque 4 — Captioning de figuras del reporte

Lo encapsulamos en una función y lo probamos sobre 3 figuras distintas del reporte.""")

code("""def caption(path, prompt=None):
    \"\"\"Una imagen -> una frase de texto describiéndola.\"\"\"
    if prompt is None:
        prompt = ("Describí esta imagen en 1 a 2 frases. "
                  "Si ves un código, modelo de equipo o marca, citalo.")
    with open(path, "rb") as f:
        return chat_vision(prompt, f.read(), t=0.0, max_t=180)""")

code("""# Bajamos 2 figuras más y captioneamos 3
for nombre in ["fig_mercado.png", "fig_camiones.png"]:
    src = f"{BASE}/figs/{nombre}"
    dst = f"corpus/{nombre}"
    if not os.path.exists(dst):
        with open(dst, "wb") as f:
            f.write(requests.get(src, timeout=30).content)

for f in [fig_local, "corpus/fig_mercado.png", "corpus/fig_camiones.png"]:
    print(f"\\n=== {os.path.basename(f)} ===")
    print(caption(f))""")

md("""👆 Tres figuras, tres descripciones. Cada una **es texto** y entra al RAG igual que un párrafo del manual.""")

# =============================================================================
# B5 — pymupdf4llm
# =============================================================================
md("""---

# Bloque 5 — `pymupdf4llm`: una sola línea hace el 90 %

Es un parser Python pensado para RAG. Te devuelve **markdown limpio** y, si lo pedís, te **extrae todas las figuras como PNGs**.""")

code("""import pymupdf4llm, time

t0 = time.time()
md_texto = pymupdf4llm.to_markdown(
    "corpus/reporte_arca_2025_capitulo.pdf",
    write_images=True,
    image_path="corpus/figuras_extraidas/auto",
    image_format="png",
    dpi=110,
)
print(f"31 páginas: {time.time()-t0:.1f}s, {len(md_texto):,} chars de markdown")

os.makedirs("corpus/figuras_extraidas/auto", exist_ok=True)
figs_auto = sorted(glob.glob("corpus/figuras_extraidas/auto/*.png"))
print(f"figuras extraídas: {len(figs_auto)}")""")

md("""### Comparemos PyPDF vs pymupdf4llm en la página 44""")
code("""# La página 44 del PDF original es la 4 del capítulo (índice 3)
# pymupdf4llm pone "----- " entre paginas — buscamos la palabra
pos = md_texto.find("44**")
print(md_texto[pos-50:pos+1100])""")

md("""👆 Comparalo con el output de PyPDF de antes. Mismas palabras, **enteras**, con `**bold**` para encabezados, y las figuras marcadas como `<==> picture ... <==>` apuntando a su PNG en `figs/`.""")

# =============================================================================
# B6 — Asistente Arca v2
# =============================================================================
md("""---

# Bloque 6 — El asistente Arca v2

Reusamos el RAG simple de clase 35 — solo cambia **la ingesta**: ahora `pymupdf4llm` + captioning, con metadata `modalidad`.""")

md("""## 6.1 — Chroma persistente""")
code("""import chromadb
from sentence_transformers import SentenceTransformer
import shutil

embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

DB = "./reporte_arca_db"
if os.path.exists(DB): shutil.rmtree(DB)
chroma = chromadb.PersistentClient(path=DB)
col = chroma.get_or_create_collection("reporte_arca",
                                       metadata={"hnsw:space": "cosine"})
print(f"collection lista; documents: {col.count()}")""")

md("""## 6.2 — Chunking del markdown + indexar""")
code("""def sliding(t, tam=400, overlap=80):
    t = re.sub(r"\\s+", " ", t).strip()
    out, i = [], 0
    while i < len(t):
        c = t[i:i+tam].strip()
        if len(c) > 80: out.append(c)
        i += tam - overlap
    return out

chunks_txt = sliding(md_texto)
print(f"chunks de texto: {len(chunks_txt)}")
embs_txt = embedder.encode(chunks_txt, normalize_embeddings=True).tolist()
col.add(documents=chunks_txt, embeddings=embs_txt,
        metadatas=[{"modalidad":"texto"} for _ in chunks_txt],
        ids=[f"txt_{i}" for i in range(len(chunks_txt))])
print(f"total en collection: {col.count()}")""")

md("""## 6.3 — Captioning de las figuras y indexar

⚠️ El captioning de 100+ figuras quema el free tier de Groq. Para clase, **tomamos las primeras 12 figuras** (basta para los demos). En producción real, lo dejarías corriendo de fondo.""")
code("""# Tomamos solo las 12 figuras más grandes -> contienen el contenido visual real
import os
figs_grandes = sorted(figs_auto,
                      key=lambda f: os.path.getsize(f), reverse=True)[:12]

caps, metas, ids = [], [], []
for i, f in enumerate(figs_grandes):
    cap = caption(f)
    print(f"  [{i+1}/{len(figs_grandes)}] {os.path.basename(f)}: {cap[:100]}")
    caps.append(cap)
    metas.append({"modalidad":"figura","path":f})
    ids.append(f"img_{i}")

embs_img = embedder.encode(caps, normalize_embeddings=True).tolist()
col.add(documents=caps, embeddings=embs_img, metadatas=metas, ids=ids)
print(f"\\ntotal en collection: {col.count()}")""")

md("""## 6.4 — `rag()` — el de clase 35, sin cambios""")
code("""SYSTEM_ARCA = (
    "Eres asistente sobre el Reporte Anual Integrado Arca Continental 2025. "
    "Responde UNICAMENTE con info del CONTEXTO. Si no aparece, di "
    "'No aparece en mi base'. Cita la modalidad de cada fuente (texto/figura). "
    "Sé conciso. Si la respuesta mezcla texto y figura, mencionalos."
)

def rag(pregunta, k=4, where=None):
    qe = embedder.encode([pregunta], normalize_embeddings=True).tolist()
    r = col.query(query_embeddings=qe, n_results=k, where=where)
    ctx = "\\n---\\n".join(
        f"[{m['modalidad']}] {d}"
        for d, m in zip(r["documents"][0], r["metadatas"][0])
    )
    return chat(SYSTEM_ARCA, f"CONTEXTO:\\n{ctx}\\n\\nPREGUNTA: {pregunta}",
                t=0.0, max_t=400)""")

md("""## 6.5 — Probemos""")
code("""for q in [
    "¿En qué índices ESG está Arca Continental?",
    "¿Qué iniciativas de sostenibilidad mencionan?",
    "¿Arca tiene presencia en mercados de Latinoamérica?",  # respuesta probable: caption
    "¿Cuánto cobramos al proveedor X el mes pasado?",       # fuera de scope
]:
    print(f"\\n[Q] {q}")
    print(f"[A] {rag(q)[:500]}")""")

md("""## 6.6 — Gradio (Bonus)""")
code("""import gradio as gr

def chat_arca(pregunta):
    return rag(pregunta, k=5)

demo = gr.Interface(
    fn=chat_arca,
    inputs=gr.Textbox(label="Preguntá al Reporte Arca 2025"),
    outputs=gr.Textbox(label="Respuesta + cita"),
    examples=["¿En qué índices ESG está Arca?",
              "¿Qué iniciativas tienen para reducir emisiones?",
              "¿Cuántos países opera Arca?"],
)
# demo.launch(share=True)   # descomentar en clase""")

# =============================================================================
# BONUS — RAG sobre tablas grandes con tool use (agent loop)
# =============================================================================
md("""---

# Bonus — RAG sobre tablas grandes con *tool use*

¿Y si el dato no es texto sino una tabla **grande** (miles de filas)? El RAG simple no sirve: la tabla no cabe en el contexto. Solución: **el LLM no mira la tabla; pide al runtime que la consulte por él** — eso es *function calling / tool use*.

**Ejemplo real:** dataset OWID CO2 — 50.411 filas × 79 columnas, cubre 254 países desde 1750.""")

code("""# Bajar dataset real OWID CO2 (~5 MB)
import pandas as pd
URL_CO2 = "https://github.com/owid/co2-data/raw/master/owid-co2-data.csv"
df = pd.read_csv(URL_CO2)
print(f"Dataset OWID CO2: {len(df):,} filas × {len(df.columns)} columnas")
print(f"  Países: {df['country'].nunique()}, Años: {df['year'].min()}–{df['year'].max()}")
print(f"\\nColumnas más usadas: country, year, iso_code, co2, co2_per_capita, population")""")

md("""## Definir la tool (`ejecutar_pandas`)""")
code("""import json

TOOLS = [{
    "type": "function",
    "function": {
        "name": "ejecutar_pandas",
        "description": "Ejecuta una expresión pandas sobre el DataFrame `df` ya cargado. Devuelve el resultado.",
        "parameters": {
            "type": "object",
            "properties": {
                "codigo": {"type": "string",
                           "description": "Una expresión pandas. Ej: df[df.year==2020].nlargest(5,'co2')"}
            },
            "required": ["codigo"]
        }
    }
}]

def ejecutar_pandas(codigo: str) -> str:
    try:
        return str(eval(codigo, {"df": df, "pd": pd}))[:1500]
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

SYSTEM_AGENT = (
    "Eres analista de datos sobre OWID CO2 dataset. DataFrame `df` ya está cargado. "
    "Columnas: country, year, iso_code, co2 (mt), co2_per_capita, population, gdp, "
    "coal_co2, oil_co2, gas_co2. El dataset incluye agregados (World, Asia, OECD); "
    "para listar SOLO países usá df[df.iso_code.notna()]. Nombres en inglés: "
    "Mexico, Brazil, United States, China. Llamá la tool ejecutar_pandas con UNA "
    "expresión pandas. Cuando tengas el resultado, redactá un insight breve en "
    "español con números concretos."
)""")

md("""## El loop del agente — el LLM decide cuándo llamar la tool y cuándo parar""")
code("""def agent(pregunta, model="llama-3.3-70b-versatile", max_turns=4, verbose=True):
    msgs = [{"role":"system","content": SYSTEM_AGENT},
            {"role":"user",  "content": pregunta}]
    for turn in range(max_turns):
        r = client.chat.completions.create(model=model, messages=msgs,
            tools=TOOLS, tool_choice="auto", max_tokens=700, temperature=0)
        m = r.choices[0].message
        if not m.tool_calls:
            return m.content                              # respuesta final
        msgs.append({"role":"assistant","content":m.content,
                     "tool_calls":[{"id":x.id,"type":"function",
                       "function":{"name":x.function.name,
                                   "arguments":x.function.arguments}}
                       for x in m.tool_calls]})
        for tc in m.tool_calls:
            args = json.loads(tc.function.arguments)
            if verbose: print(f"\\n[t{turn+1}] CODE: {args['codigo']}")
            out = ejecutar_pandas(args["codigo"])
            if verbose: print(f"     RESULT: {out[:300]}")
            msgs.append({"role":"tool","tool_call_id":tc.id,"content":out})
    return "(max_turns)"  """)

md("""## Probemos con 3 preguntas reales""")
code("""for q in [
    "¿Cuáles son los 5 países (no regiones) que más CO2 emitieron en 2022?",
    "¿Cuál es el CO2 per cápita de Mexico vs United States en 2022?",
    "¿Qué porcentaje del CO2 total mundial de 2022 representó China?",
]:
    print("\\n" + "="*72)
    print(f"PREGUNTA: {q}")
    print("="*72)
    print("\\n" + agent(q))""")

md("""👆 El LLM **escribió las queries pandas** y el runtime las ejecutó. El alumno solo tipeó la pregunta en español.

**Patrón general** — el mismo loop sirve para:
- Una tabla SQL (cambiar `ejecutar_pandas` por `ejecutar_sql`).
- Una API REST (`hacer_request`).
- Una base de Chroma (`buscar_documentos`) — y ahí cerramos el círculo: el LLM decide cuándo hacer RAG y cuándo consultar datos numéricos.""")

# =============================================================================
# PBL
# =============================================================================
md("""---

# PBL final — 3 ejercicios""")

md("""## E1 — Re-indexá el reporte completo (189 pp)

Bajá el reporte entero, pasalo por pymupdf4llm, indexá. Compará 3 respuestas con vs. sin el capítulo de financieros.""")
code("""# URL del reporte completo (27 MB) directamente desde Arca Continental:
URL_FULL = ("https://fdecorpwsprdeus2-bfb4cgcde4ftcff6.a01.azurefd.net/"
            "files/2026-06/Reporte%20Anual%20Integrado%202025.pdf")

# TODO 1: descargar
# TODO 2: pymupdf4llm.to_markdown(...) sobre el completo
# TODO 3: re-indexar en una collection nueva
# TODO 4: 3 queries de financieros (e.g. EBITDA, utilidad, ventas por región)
""")

md("""## E2 — Tu propio PDF

Subí un PDF de tu trabajo (reporte, manual, política). Pasalo por la misma función `pdf_a_chunks(...)`. Indexalo.

**Criterio:** 3 queries reales que muestren el valor.""")
code("""# TODO 1: cargá tu PDF a Colab
# TODO 2: pymupdf4llm.to_markdown(MI_PDF, write_images=True, ...)
# TODO 3: chunkear + indexar
# TODO 4: 3 queries (2 que sí están en el PDF, 1 que no)
""")

md("""## E3 — Comparar pymupdf4llm vs Docling

Sobre el **mismo** PDF (puede ser un manual pequeño), ejecutá pymupdf4llm y Docling. Pegá los dos outputs lado a lado. ¿Cuál sirve para tu caso?

⚠️ Docling baja modelos PyTorch y tarda ~1 min en cargar, y por página el procesamiento es lento. Hacelo sobre 2–3 páginas.""")
code("""# !pip install -q docling

# from docling.document_converter import DocumentConverter
# converter = DocumentConverter()
# result = converter.convert("mini.pdf")
# md_docling = result.document.export_to_markdown()
# TODO: compará len(md_docling) vs len(md_pymupdf), y mirá tablas / figuras
""")

md("""---

# Cierre

Aprendiste a:
1. Ver con tus ojos cómo PyPDF **destroza un PDF corporativo** real (palabras partidas, figuras invisibles, layout perdido).
2. Usar **Llama 4 Scout via Groq** como modelo multimodal — el mismo cliente OpenAI, distinto `model`.
3. Convertir cualquier imagen en texto con **image captioning** y meterla al mismo RAG.
4. Reemplazar el `PdfReader` por **pymupdf4llm**: una línea, markdown limpio + figuras extraídas.
5. Armar un asistente Arca v2 que cita texto **y** figuras del Reporte Anual 2025.

**Producción real = librería pro + multimodal + RAG.**""")

# =============================================================================
nb.cells = cells
nbf.write(nb, "Clase_36_RAG_Multimodal.ipynb")
print(f"Notebook generado: {len(cells)} celdas")
