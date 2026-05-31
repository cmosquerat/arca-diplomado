"""
Construye Clase_34_LSTM_a_LLM.ipynb --- versión PBL (Project-Based Learning).

Perfil de los estudiantes: profesionales seniors de Arca Continental Ecuador
(embotelladora). La teoría de LSTM/Atención/QKV YA está en el deck.
Este notebook es 100% APLICADO:

  - 10 mini-aplicaciones demostradas (con código que corre y produce resultados reales).
  - 3 de las apps son interfaces Gradio (se lanzan dentro del notebook).
  - Apps cubren Groq (cloud free) + Ollama (local) intercambiables (mismo código,
    distinto base_url) --- enseñamos el ESTÁNDAR OpenAI-compatible.
  - 5 ejercicios PBL con esqueleto + criterio de aceptación claro.

HILO: "el oficio nuevo es elegir, orquestar y construir UI sobre modelos pre-entrenados".
"""
import json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Clase_34_LSTM_a_LLM.ipynb")
REPO = "https://raw.githubusercontent.com/cmosquerat/arca-diplomado/main/clase-34"

cells = []
def md(t):
    L = t.strip("\n").split("\n")
    cells.append({"cell_type":"markdown","metadata":{},"source":[(s+"\n") for s in L[:-1]]+[L[-1]]})
def code(t):
    s = t.strip("\n").split("\n")
    cells.append({"cell_type":"code","metadata":{},"outputs":[],"execution_count":None,
                  "source":[(x+"\n") for x in s[:-1]]+[s[-1]]})

# =============================================================================
#  PORTADA
# =============================================================================
md("""
# Clase 34 --- Del LSTM al LLM (Notebook PBL)
### 10 aplicaciones que construyes hoy + 5 que armas tú

**Diplomado en Data Science Aplicada con Python para la Toma de Decisiones**
Arca Continental Ecuador | UDLA

---

> **¿Cómo le enseñamos a una máquina a entender el lenguaje de Arca ---
> manuales, tickets, quejas --- sin contratar 50 lingüistas?**

La teoría (Transformer, atención, BETO, escala de LLMs) está en el **deck**.
Este notebook es **100% práctico**: vas a salir hoy con apps que el lunes
puedes mostrarle a tu jefe.

### Lo que vas a construir

| # | App | Para qué sirve en Arca |
|---|---|---|
| 1 | Hello LLM + anatomía del endpoint | Entender qué pasa "por debajo" |
| 2 | Clasificador zero-shot de tickets | Triage automático de la cola |
| 3 | Extractor JSON estructurado | De texto libre a campos para tu SAP |
| 4 | Resumidor de reportes de turno | 1 página → 3 viñetas en 2 segundos |
| 5 | Generador de respuestas a quejas | Borrador para tu CSR, no autopublicar |
| 6 | Chat multi-turno con memoria | Asistente de mantenimiento "que recuerda" |
| 7 | Búsqueda semántica de tickets | "Tráeme tickets parecidos a éste" |
| 8 | **Gradio**: clasificador con UI | App web en 10 líneas |
| 9 | **Gradio**: chat con tus documentos | RAG mínimo con UI |
| 10 | **Gradio**: comparador Groq vs Ollama | El mismo prompt, dos proveedores |

### Lo que vas a entregar (5 ejercicios PBL)

E1 - Tu clasificador con TUS categorías.
E2 - Extractor de tu dominio (órdenes de trabajo).
E3 - App Gradio de chat sobre un documento tuyo.
E4 - Comparativa de 2 proveedores en la misma tarea.
E5 - Mini-RAG sobre tus tickets reales.

Cada ejercicio tiene **esqueleto de código + criterio de aceptación**. No es teoría;
es construir y mostrar que corre.

---
""")

# =============================================================================
#  0. SETUP
# =============================================================================
md("""
## 0. Setup --- corre esta celda primero

Instalamos lo necesario. Funciona en Colab (con GPU T4) y en local.
""")

code("""
import sys, os
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    os.system("pip install -q openai sentence-transformers transformers gradio")
print("✓ Setup listo")
""")

# =============================================================================
#  1. ANATOMÍA DEL ENDPOINT (OpenAI-compatible = PROTOCOLO, NO MARCA)
# =============================================================================
md("""
## 1. Anatomía del endpoint --- "OpenAI-compatible" es un PROTOCOLO, no la marca

**Concepto fundamental que evita confusión**:

> `from openai import OpenAI` **NO** significa que estás usando ChatGPT.

Es como `import requests` --- no significa que estés llamando a un sitio específico.
Es solo la librería cliente que habla un protocolo HTTP particular.

### El estándar OpenAI-compatible

OpenAI publicó la **forma del JSON** que su API espera y devuelve.
Esa forma se volvió un **estándar de facto**. Hoy CUALQUIER servidor de LLM
que respete esa forma puede usar el mismo cliente:

| Provider | base_url | Quién lo corre |
|---|---|---|
| OpenAI (GPT-5, etc.) | `https://api.openai.com/v1` | OpenAI (pago) |
| **Groq** | `https://api.groq.com/openai/v1` | Groq (free con límites) |
| **Ollama** (local) | `http://localhost:11434/v1` | Tu propia máquina (gratis ilimitado) |
| Together / Fireworks | `https://api.together.xyz/v1` | etc. |
| vLLM en tu servidor | `http://tu-servidor:8000/v1` | Tú, on-prem |
| Anthropic Claude | (otro protocolo, librería `anthropic`) | --- |

**Misma librería + cambiás 2 líneas (base_url + api_key) = hablas con otro.**
Eso es lo poderoso del estándar.

### La forma del request/response

Vamos a verlo. Hacemos UNA llamada a Groq y miramos el **JSON crudo** que se manda y se recibe.
""")

code("""
import getpass
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = getpass.getpass("Pega tu GROQ_API_KEY (console.groq.com, gratis sin tarjeta): ")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
""")

code("""
from openai import OpenAI

# Cliente apuntado a Groq. Cambiando base_url, apunta a quien quieras.
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",  # <-- el provider vive aquí
)
MODEL = "llama-3.1-8b-instant"

# Una llamada
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Eres asistente de planta de Arca. Responde corto."},
        {"role": "user",   "content": "¿Qué hago primero si una llenadora vibra raro?"},
    ],
    temperature=0.3,
    max_tokens=120,
)
print(resp.choices[0].message.content)
""")

md("""
### El JSON crudo --- esto es TODO lo que viaja

La respuesta `resp` es un objeto Python. Por debajo es un JSON. Veámoslo.
""")

code("""
import json as jsn

# El response como dict (es lo que viene del servidor)
print("=== RESPONSE JSON ===")
print(jsn.dumps(resp.model_dump(), ensure_ascii=False, indent=2)[:1200])

# Lo que tú mandaste, como JSON
print("\\n\\n=== REQUEST JSON (lo que la librería envió al server) ===")
print(jsn.dumps({
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "Eres asistente de planta de Arca. Responde corto."},
        {"role": "user",   "content": "¿Qué hago primero si una llenadora vibra raro?"},
    ],
    "temperature": 0.3,
    "max_tokens": 120,
}, ensure_ascii=False, indent=2))
""")

md("""
**Lo que ves**:
- El request es un JSON con `model`, `messages` (lista de roles `system`/`user`/`assistant`),
  hiperparámetros (`temperature`, `max_tokens`).
- El response trae `choices[0].message.content` (texto generado), `usage` (tokens consumidos),
  `finish_reason` (`stop` = terminó bien, `length` = se quedó sin tokens).

**Eso es todo. Ni más, ni menos.** Cualquier provider que respete esta forma se puede usar
con `openai`. Si mañana sale "Mistral en la nube", pondrás `base_url="https://api.mistral..."`
y todo tu código sigue funcionando.

### Helper unificado --- Groq o Ollama (el mismo código)
""")

code("""
def make_client(provider="groq"):
    \"\"\"Devuelve (client, model_name) para Groq cloud u Ollama local. Mismo código abajo.\"\"\"
    if provider == "groq":
        return OpenAI(api_key=GROQ_API_KEY,
                      base_url="https://api.groq.com/openai/v1"), "llama-3.1-8b-instant"
    elif provider == "ollama":
        # Requiere `ollama serve` corriendo localmente con `ollama pull llama3.2:3b`
        return OpenAI(api_key="ollama-no-key-needed",
                      base_url="http://localhost:11434/v1"), "llama3.2:3b"
    raise ValueError(provider)

def llm(prompt, system=None, provider="groq", model=None,
        temperature=0.3, max_tokens=400):
    cli, default_model = make_client(provider)
    msgs = []
    if system: msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    r = cli.chat.completions.create(
        model=model or default_model, messages=msgs,
        temperature=temperature, max_tokens=max_tokens,
    )
    return r.choices[0].message.content, r.usage
""")

md("""
### Ollama en tu máquina (3 pasos)

En Colab no podemos correr Ollama (es un servidor que necesita instalarse en tu OS).
Pero en tu máquina personal:

```bash
# Mac/Linux: 1 línea
curl https://ollama.com/install.sh | sh

# Windows: descarga el instalador en ollama.com/download

# Después:
ollama pull llama3.2:3b     # ~2GB, modelo chiquito y rápido
ollama serve                # arranca el server en localhost:11434
```

Una vez con Ollama corriendo, en tus scripts cambia `provider="groq"` → `provider="ollama"`
y todo funciona idéntico. **Sin cambiar nada más del código**.

**¿Por qué importa Ollama?**
- **Gratis ilimitado** (corre en tu CPU/GPU).
- **Privacidad**: tus datos NUNCA salen de tu red. Para info sensible de Arca, esto es ORO.
- **Sin internet**: funciona offline.
- **Sin rate limits**.

**Trade-off**: en una máquina sin GPU, los modelos chicos (3B) son rápidos pero menos capaces
que GPT-4. Para tareas rutinarias (clasificar, extraer) son perfectos.

---
""")

# =============================================================================
#  2. EL PROBLEMA QUE RESOLVEMOS (recap corto de clase 33)
# =============================================================================
md("""
## 2. El problema que resolvemos (recap corto)

En clase 33 vimos que TF-IDF + LogReg llega a 84% en sentimiento, pero **falla en negación**.
Repetimos el experimento UNA vez para tenerlo en mente, y seguimos a las apps.
""")

code("""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CSV_URL = \"""" + REPO + """/muchocine_sentimiento.csv\"
df = pd.read_csv(CSV_URL)
df["review"] = df["review"].astype(str)
Xtr, Xte, ytr, yte = train_test_split(df["review"], df["label"], test_size=0.2,
                                       random_state=42, stratify=df["label"])
vec = TfidfVectorizer(strip_accents="unicode", min_df=3, ngram_range=(1,2), max_features=30000)
clf = LogisticRegression(max_iter=1000, C=3.0).fit(vec.fit_transform(Xtr), ytr)
acc = accuracy_score(yte, clf.predict(vec.transform(Xte)))
print(f"Baseline TF-IDF+LogReg en muchocine: {acc:.1%}")

# 4 frases-trampa de negación: TF-IDF se cae
TRAMPAS = [
    ("no me gustó nada la trama", 0),
    ("no es para nada mala, me sorprendió", 1),
    ("no podría estar más contento", 1),
    ("buena fotografía pero no, no funciona", 0),
]
print(f"\\n{'Real':<10} {'Pred TF-IDF':<14} Frase")
for f, real in TRAMPAS:
    p = clf.predict(vec.transform([f]))[0]
    ok = "✓" if p == real else "✗"
    label_real = "POS" if real else "NEG"
    label_pred = "POS" if p else "NEG"
    print(f"{ok} {label_real:<7}  {label_pred:<11}  {f}")
""")

md("""
**El problema queda planteado**: TF-IDF cuenta palabras, no entiende el orden ni la negación.
**Ahora resolvemos esto (y muchas otras tareas) llamando a un LLM.**

---
""")

# =============================================================================
#  3. LAS 10 APPS
# =============================================================================
md("""
## 3. Las 10 apps --- construidas paso a paso

Cada app es **funcional, corta y reusable**. Las primeras 7 son funciones Python
(las llamás desde donde quieras). Las últimas 3 son apps Gradio (con UI web).

### App 1 --- Hello LLM (ya la viste arriba, ahora la entendemos)

La función `llm(prompt)` que definimos arriba es tu "navaja suiza". Ya está lista.
Te la dejamos por si quieres revisarla:
""")

code("""
out, usage = llm("Di hola en una palabra.")
print(f"Respuesta: {out!r}")
print(f"Tokens usados: prompt={usage.prompt_tokens}, output={usage.completion_tokens}")
""")

md("""
### App 2 --- Clasificador zero-shot de tickets

Sin entrenar nada, le pides al LLM que clasifique en categorías que TÚ defines.
""")

code('''
def clasificar(texto, categorias, provider="groq"):
    cats = ", ".join(categorias)
    prompt = (f"Clasifica el siguiente texto en UNA SOLA categoría de: {cats}.\\n"
              f"Texto: \\"{texto}\\"\\n"
              f"Responde SÓLO con la categoría exacta, sin explicación.")
    r, _ = llm(prompt, max_tokens=20, temperature=0.0, provider=provider)
    return r.strip().lower()

# Demo
CATEGORIAS = ["mantenimiento", "queja_cliente", "logistica", "calidad", "ventas"]
EJEMPLOS = [
    "El compresor número 3 vibra raro desde anoche",
    "Cliente reclamó botella derramada al recibir el pedido",
    "El camión a Guayaquil llegó con 4 horas de atraso",
    "Lote 4521 con sabor distinto, pedimos retención",
    "Cerramos contrato con la cadena Tia para 2027",
]
for t in EJEMPLOS:
    print(f"  [{clasificar(t, CATEGORIAS):<15}] {t}")
''')

md("""
### App 3 --- Extractor JSON estructurado (texto libre → SAP)

Lo más útil para integrar a sistemas: convertir texto en datos estructurados.
""")

code('''
import json as jsn

def extraer_ticket(texto):
    """Convierte texto libre en JSON {equipo, problema, urgencia}."""
    instr = ('Extrae los datos del ticket como JSON con llaves EXACTAS:\\n'
             '{"equipo": str, "problema": str, "urgencia": "alta"|"media"|"baja"}.\\n'
             'Sólo el JSON, sin explicación ni markdown.\\n\\n'
             f'Ticket: "{texto}"')
    raw, _ = llm(instr, max_tokens=200, temperature=0.0)
    # Limpiar posibles backticks de markdown
    raw = raw.strip().strip("`").replace("json\\n", "").strip()
    try:
        return jsn.loads(raw)
    except Exception:
        return {"error": "no parseó JSON", "raw": raw}

# Demo
TICKETS = [
    "El compresor número 3 está echando vapor desde anoche, urge revisión.",
    "La llenadora de la línea 2 paró por sensor de tapa.",
    "Cambio de filtro programado para el chiller 9 la próxima semana.",
]
for t in TICKETS:
    print(f"\\n>>> {t}")
    print(jsn.dumps(extraer_ticket(t), ensure_ascii=False, indent=2))
''')

md("""
### App 4 --- Resumidor de reportes de turno

1 página de texto → 3 viñetas accionables en 2 segundos.
""")

code('''
def resumir(texto, n_vinetas=3):
    p = (f"Resume el siguiente reporte en EXACTAMENTE {n_vinetas} viñetas, "
         f"máximo 15 palabras cada una. Sólo el resumen:\\n\\n{texto}")
    r, _ = llm(p, max_tokens=250, temperature=0.2)
    return r

REPORTE = """Reporte de turno línea 2, jueves 30 mayo 2026, turno noche.
Producción objetivo: 18.000 botellas. Producción real: 16.450 (91%).
Causas del desfase: (1) parada 35 min a las 23:40 por fallo sensor tapa
posición 7 (cambio rápido con repuesto en bodega). (2) parada 22 min a
las 02:15 por atasco en carrusel etiquetado, desatascado a mano.
Indicadores calidad: rechazo 0.3% (OK, <0.5%). Notas: filtro aire
compresor 5 sucio, cambiar próximo mantenimiento jueves 6 junio."""
print(resumir(REPORTE))
''')

md("""
### App 5 --- Generador de respuestas a quejas (borrador para CSR)

Importante: el LLM redacta el BORRADOR, el agente humano revisa y publica.
""")

code('''
SISTEMA_CSR = ("Eres asistente del equipo de servicio al cliente de Arca Continental Ecuador. "
               "Redactas BORRADORES de respuesta para que un agente humano revise y envíe. "
               "Sé empático, asume responsabilidad, ofrece una solución concreta. "
               "Máximo 3 oraciones, tono profesional pero cercano.")

QUEJAS = [
    "Compré dos cajas y una llegó con la mitad de las botellas con tapa rota.",
    "Me prometieron entrega el martes y es viernes. Necesito respuesta YA.",
    "La bebida sabe vencida pero la fecha dice 2027. ¿Qué pasa?",
]
for q in QUEJAS:
    resp, _ = llm(q, system=SISTEMA_CSR, max_tokens=180)
    print(f"\\n--- QUEJA ---\\n{q}")
    print(f"\\n--- BORRADOR DE RESPUESTA (revisar antes de enviar) ---\\n{resp}")
''')

md("""
### App 6 --- Chat multi-turno con memoria (asistente de mantenimiento)

El LLM no recuerda nada entre llamadas. Para "conversación", **tú mandas el historial completo**
en cada llamada.
""")

code('''
def crear_chat(sistema):
    historial = [{"role": "system", "content": sistema}]
    def turno(mensaje_usuario, max_tokens=250, temperature=0.4):
        historial.append({"role": "user", "content": mensaje_usuario})
        cli, model = make_client("groq")
        r = cli.chat.completions.create(
            model=model, messages=historial,
            max_tokens=max_tokens, temperature=temperature)
        reply = r.choices[0].message.content
        historial.append({"role": "assistant", "content": reply})
        return reply
    return turno, historial

# Demo: conversación que requiere memoria
turno, hist = crear_chat(
    "Eres técnico senior de mantenimiento industrial en planta embotelladora. "
    "Eres concreto, das pasos numerados.")

for msg in [
    "Mi llenadora vibra más de lo normal desde anoche. ¿Por dónde empiezo?",
    "Ya revisé los pernos, están firmes. ¿Qué sigue?",
    "El motor también está caliente. ¿Eso cambia el diagnóstico?",
]:
    print(f"\\n>>> USUARIO: {msg}")
    print(f"--- BOT ---\\n{turno(msg)}")

print(f"\\n[historial guardado: {len(hist)} mensajes]")
''')

md("""
### App 7 --- Búsqueda semántica de tickets

Convertís cada ticket en un vector ("embedding"). Una nueva queja → también vector →
los vectores más cercanos son tickets parecidos. **Sin que las palabras coincidan literalmente.**

Usamos `sentence-transformers` con un modelo multilingüe chico.
""")

code('''
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Modelo multilingüe rápido (~80MB)
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Base de tickets
TICKETS = [
    "el compresor 3 está echando vapor desde anoche",
    "presión del sistema bajó por debajo de lo normal",
    "la llenadora se atascó con una botella",
    "no funciona el aire acondicionado del depósito",
    "falla en el sensor de temperatura del horno",
    "la etiquetadora pega mal las etiquetas",
    "queja cliente: la bebida sabe extraño",
    "el cliente recibió el producto vencido",
    "chiller 9 hace ruido metálico irregular",
    "carrusel de tapado se queda trabado intermitente",
]
EMBS = embedder.encode(TICKETS, show_progress_bar=False)
print(f"Embeddings: {EMBS.shape}")

def buscar(query, k=3):
    q = embedder.encode([query], show_progress_bar=False)
    sims = cosine_similarity(q, EMBS)[0]
    idx = np.argsort(-sims)[:k]
    return [(sims[i], TICKETS[i]) for i in idx]

for query in ["problema con temperatura", "cliente molesto",
              "máquina hace ruido raro"]:
    print(f"\\n🔍 \\"{query}\\"")
    for sim, t in buscar(query, k=3):
        print(f"   {sim:.2f}  {t}")
''')

md("""
**Observa**: las queries usan palabras DISTINTAS a los tickets (`temperatura` no aparece
literal en `sensor del horno`, `cliente molesto` no es literal `queja`). El modelo entiende
**el significado**, no las palabras.

Esto es la base de RAG (Retrieval-Augmented Generation) — la próxima clase del Módulo 6.

---

## Apps con Gradio --- aplicaciones web en el notebook

**Gradio** es una librería que envuelve tu función Python en una UI web.
3-5 líneas y tienes una app que el navegador renderiza, lista para mostrar al jefe.

> **Nota**: en Colab, `demo.launch(share=True)` te da una URL pública temporal (72h).
> En local, `demo.launch()` te abre el navegador con `localhost:7860`.
""")

md("""
### App 8 --- Gradio: Clasificador de tickets con UI

Una caja para pegar texto, un dropdown para elegir provider, un botón.
La función ya la tenemos (`clasificar`); solo le ponemos UI.
""")

code('''
import gradio as gr

CATS_DEFAULT = "mantenimiento, queja_cliente, logistica, calidad, ventas"

def app_clasificar(texto, categorias_csv, provider):
    cats = [c.strip() for c in categorias_csv.split(",") if c.strip()]
    if not texto.strip(): return "(pega un texto)"
    try:
        cat = clasificar(texto, cats, provider=provider)
        return f"**{cat}**\\n\\n_({len(cats)} categorías evaluadas con {provider})_"
    except Exception as e:
        return f"⚠️ Error con {provider}: {type(e).__name__}: {e}"

demo_clasif = gr.Interface(
    fn=app_clasificar,
    inputs=[
        gr.Textbox(label="Ticket / texto a clasificar", lines=3,
                   placeholder="Ej: El compresor número 3 vibra raro desde anoche"),
        gr.Textbox(label="Categorías (separadas por comas)", value=CATS_DEFAULT),
        gr.Radio(["groq", "ollama"], value="groq", label="Provider"),
    ],
    outputs=gr.Markdown(label="Categoría predicha"),
    title="Clasificador de tickets · Arca",
    description="Pega cualquier ticket o queja. Define tus categorías. El LLM elige una.",
    flagging_mode="never",
)
# Descomentá la siguiente línea para lanzar la app en este notebook:
# demo_clasif.launch(share=True, inline=True)
print("✓ App definida. Para lanzar: demo_clasif.launch(share=True, inline=True)")
''')

md("""
### App 9 --- Gradio: Chat con tu documento (mini-RAG)

Le pegas un documento (manual, reporte) y le haces preguntas. El LLM responde **usando solo
el documento** (no su conocimiento general). Es RAG en su forma más simple.
""")

code('''
def chat_con_doc(documento, pregunta):
    if not documento.strip() or not pregunta.strip():
        return "(pega un documento y haz una pregunta)"
    prompt = (
        "Responde la pregunta usando SÓLO la información del siguiente documento. "
        "Si la respuesta no aparece en el documento, di literalmente: 'No aparece en el documento.'\\n\\n"
        f"=== DOCUMENTO ===\\n{documento}\\n\\n"
        f"=== PREGUNTA ===\\n{pregunta}\\n\\n"
        f"=== RESPUESTA ==="
    )
    r, _ = llm(prompt, max_tokens=300, temperature=0.0)
    return r

DOC_DEMO = """Manual rápido: llenadora modelo XF-200, planta Arca Quito.
Capacidad: 12.000 botellas/hora. Presión operativa: 2.5 a 3.2 bar.
Mantenimiento preventivo cada 200 horas. Filtro de aire: cambio semanal.
Lubricación del carrusel: aceite ISO 220, cada 80 horas. Sensor de tapa:
si falla, código de error E07; revisar conexión antes de reemplazar."""

demo_doc = gr.Interface(
    fn=chat_con_doc,
    inputs=[
        gr.Textbox(label="Documento", lines=10, value=DOC_DEMO),
        gr.Textbox(label="Pregunta", placeholder="¿Cada cuánto se cambia el filtro de aire?"),
    ],
    outputs=gr.Textbox(label="Respuesta del LLM", lines=4),
    title="Chat con tu documento · Arca",
    description="Le pegas un manual/reporte/política y le preguntas. El LLM responde sólo con lo que está ahí.",
    flagging_mode="never",
)
# demo_doc.launch(share=True, inline=True)
print("✓ App definida. Para lanzar: demo_doc.launch(share=True, inline=True)")
''')

md("""
### App 10 --- Gradio: Comparador Groq vs Ollama

El mismo prompt, dos proveedores, resultados lado a lado. Para que veas que **el código es el mismo,
sólo cambia el `base_url`**.
""")

code('''
def comparar(prompt_user, temp):
    out = {}
    for p in ["groq", "ollama"]:
        try:
            r, u = llm(prompt_user, provider=p, temperature=temp, max_tokens=250)
            out[p] = f"### {p.upper()}\\n{r}\\n\\n_tokens: {u.total_tokens}_"
        except Exception as e:
            out[p] = f"### {p.upper()} ⚠️\\n_no disponible_: {type(e).__name__}\\n{str(e)[:200]}"
    return out.get("groq", ""), out.get("ollama", "")

demo_comp = gr.Interface(
    fn=comparar,
    inputs=[
        gr.Textbox(label="Prompt", lines=3,
                   value="Da 3 causas posibles de vibración en una llenadora industrial."),
        gr.Slider(0.0, 1.5, value=0.3, label="Temperature"),
    ],
    outputs=[
        gr.Markdown(label="Groq (cloud, gratis hasta 30 RPM)"),
        gr.Markdown(label="Ollama (local, gratis ilimitado)"),
    ],
    title="Comparador Groq vs Ollama · Arca",
    description="Mismo prompt, mismos parámetros, distinto provider. El código de abajo es IDÉNTICO, solo cambia el base_url.",
    flagging_mode="never",
)
# demo_comp.launch(share=True, inline=True)
print("✓ App definida. Para lanzar: demo_comp.launch(share=True, inline=True)")
''')

md("""
### Lanzar las 3 apps juntas (opcional)

Si quieres ver las tres apps en una sola interfaz con tabs:
""")

code('''
tabbed = gr.TabbedInterface(
    [demo_clasif, demo_doc, demo_comp],
    ["Clasificador", "Chat con doc", "Groq vs Ollama"],
    title="Asistente NLP del supervisor de planta · Arca Continental",
)
# tabbed.launch(share=True, inline=True)
print("✓ TabbedInterface lista. Para lanzar: tabbed.launch(share=True, inline=True)")
''')

# =============================================================================
#  4. LOS 5 EJERCICIOS PBL
# =============================================================================
md("""
---

## 4. Los 5 ejercicios PBL --- tu turno

Cada ejercicio tiene **esqueleto + criterio de aceptación**. La idea es que adaptes lo de arriba
a TU dominio real en Arca y traigas el resultado a la próxima clase (o lo subas al repo).

### E1 --- Tu clasificador con TUS categorías

**Objetivo**: tomar la función `clasificar()` y adaptarla a categorías reales de tu área
(mantenimiento, calidad, ventas, recursos humanos, lo que sea).

**Criterio de aceptación**:
- Defines al menos **5 categorías** propias.
- Pruebas con al menos **10 textos reales** (puedes inventarlos plausibles o usar tickets anonimizados).
- Imprimes una tabla `texto → categoría predicha`.
- Anotas 1-2 errores que el modelo cometa, y por qué.
""")

code('''
# ESQUELETO E1 --- adapta y completa
MIS_CATEGORIAS = [
    "tu_categoria_1",
    "tu_categoria_2",
    # ...
]
MIS_TEXTOS = [
    "...",
    # 10 mínimo
]
# for t in MIS_TEXTOS:
#     print(f"[{clasificar(t, MIS_CATEGORIAS)}]  {t}")
''')

md("""
### E2 --- Extractor de TU dominio

**Objetivo**: armar un extractor JSON para un tipo de documento real de tu área.
Ejemplo: órdenes de trabajo con campos `(equipo, técnico_asignado, repuestos_necesarios,
horas_estimadas, prioridad)`.

**Criterio de aceptación**:
- Defines el **schema JSON** que esperas como salida (al menos 4 campos).
- Pruebas con al menos **5 textos** distintos.
- El LLM devuelve JSON parseable (`json.loads` no falla) en al menos 4 de los 5.
""")

code('''
# ESQUELETO E2
def mi_extractor(texto):
    schema = ('{"equipo": str, "tecnico": str, "repuestos": [str], '
              '"horas": int, "prioridad": "alta"|"media"|"baja"}')
    instr = (f'Extrae los datos del siguiente texto como JSON con schema:\\n{schema}\\n'
             f'Sólo JSON, sin explicación.\\n\\nTexto: "{texto}"')
    raw, _ = llm(instr, max_tokens=300, temperature=0.0)
    # ... limpiar y parsear como en App 3
    return raw

# Prueba con tus textos
# MIS_OT = ["...", "...", "...", "...", "..."]
# for ot in MIS_OT:
#     print(mi_extractor(ot))
''')

md("""
### E3 --- Tu propia app Gradio de chat sobre un documento

**Objetivo**: lanzar una app Gradio donde el usuario pega un documento de TU dominio
(manual, política, reporte) y hace preguntas.

**Criterio de aceptación**:
- Usas `demo_doc` o construyes uno nuevo.
- Pones un **documento por defecto** que sea relevante a tu trabajo (no el de la llenadora del ejemplo).
- Lanzas la app y haces **al menos 3 preguntas** que se respondan correctamente.
- Anotas 1 pregunta que el LLM no sepa responder (porque no está en el doc) y verificas que
  responde "No aparece en el documento" en vez de inventar.
""")

code('''
# ESQUELETO E3 --- adapta DOC_DEMO al tuyo y lanza
MI_DOC = """
PEGA AQUÍ tu manual / política / reporte (texto plano, sin imágenes).
"""
# Reutilizamos la función chat_con_doc
# demo_mio = gr.Interface(
#     fn=chat_con_doc,
#     inputs=[gr.Textbox(label="Documento", lines=10, value=MI_DOC),
#             gr.Textbox(label="Pregunta")],
#     outputs=gr.Textbox(label="Respuesta", lines=4),
#     title="Tu app personalizada",
# )
# demo_mio.launch(share=True, inline=True)
''')

md("""
### E4 --- Comparativa de 2 proveedores en una tarea de tu área

**Objetivo**: tomar UNA tarea concreta (clasificar, extraer, resumir, responder), correrla
con **dos modelos distintos** y comparar.

**Criterio de aceptación**:
- Comparas al menos **2 proveedores** (Groq + Ollama, o Groq + otro modelo Groq como
  `llama-3.3-70b-versatile`).
- Pruebas con al menos **10 inputs** distintos.
- Reportas en una tabla: **acc/calidad subjetiva** (1-5 estrellas tuyas), **latencia promedio**,
  **costo estimado** (Ollama=$0, Groq=$0 hasta el rate limit, Claude Sonnet=$3/M tok input).
- Concluyes con 1 párrafo: **¿cuál usarías en producción y por qué?**
""")

code('''
# ESQUELETO E4
import time
def medir(provider, model, prompts, **kwargs):
    rs = []
    for p in prompts:
        t0 = time.time()
        try:
            r, u = llm(p, provider=provider, model=model, **kwargs)
            rs.append({"prompt": p, "respuesta": r, "latencia": time.time()-t0,
                       "tokens": u.total_tokens, "ok": True})
        except Exception as e:
            rs.append({"prompt": p, "error": str(e)[:200], "ok": False})
    return rs

# MIS_PROMPTS = [...]
# A = medir("groq", "llama-3.1-8b-instant", MIS_PROMPTS, max_tokens=200)
# B = medir("groq", "llama-3.3-70b-versatile", MIS_PROMPTS, max_tokens=200)
# Compara: latencia, tokens, calidad subjetiva. Reporta en una tabla pandas.
''')

md("""
### E5 --- Mini-RAG sobre tus tickets reales (proyecto integrador)

**Objetivo**: indexás al menos **50 tickets de tu área** (anonimizados), montás búsqueda semántica
y conectás el resultado a un LLM para que responda "qué tickets parecidos hay y qué hicieron".

**Criterio de aceptación**:
- Tienes **50+ textos** indexados con `embedder.encode` (App 7).
- Función `buscar_y_responder(query) → respuesta del LLM citando tickets parecidos`.
- Demuestras con al menos **3 queries**.
- Como bonus, lo metés en una app Gradio.
""")

code('''
# ESQUELETO E5
MIS_TICKETS = [
    # 50+ textos. Pueden venir de un CSV, BD, copy-paste, lo que sea.
    "...",
]
MIS_EMBS = embedder.encode(MIS_TICKETS, show_progress_bar=False)

def buscar_y_responder(query, k=3):
    # 1) embed la query
    q = embedder.encode([query], show_progress_bar=False)
    # 2) k vecinos
    sims = cosine_similarity(q, MIS_EMBS)[0]
    top = np.argsort(-sims)[:k]
    contexto = "\\n".join(f"- {MIS_TICKETS[i]}" for i in top)
    # 3) preguntarle al LLM
    p = (f"Usa estos {k} tickets parecidos como contexto:\\n{contexto}\\n\\n"
         f"Pregunta: {query}\\n\\nResponde citando los tickets relevantes.")
    r, _ = llm(p, max_tokens=300, temperature=0.2)
    return r

# print(buscar_y_responder("compresor con problemas"))
''')

# =============================================================================
#  5. CIERRE
# =============================================================================
md("""
---

## 5. Lo que te llevas

| Capa | Cuándo se usa |
|---|---|
| Función Python (`llm`, `clasificar`, `extraer_ticket`, `resumir`, `chat_con_doc`, `buscar`) | scripts, batch jobs, APIs internas |
| App Gradio | demos a stakeholders, prototipos, herramientas internas |
| Toggle Groq/Ollama | gratis hasta el rate limit (Groq) o privado on-prem (Ollama) |
| Estándar OpenAI-compatible | un código, muchos providers --- nunca te casas con uno |

### Regla del oficio

> El oficio nuevo del data scientist en Arca es **elegir el provider/modelo correcto,
> orquestar las llamadas, y construir UI sobre eso**. No entrenar modelos desde cero.

### El siguiente paso --- Módulo 6

- **Prompting avanzado**: few-shot, chain-of-thought, role prompting con plantillas.
- **RAG en serio**: embeddings + FAISS/ChromaDB + reranking + chunking.
- **Agentes**: LLMs que ejecutan acciones (consultar BD, mandar emails, llamar APIs).
- **Web scraping**: extraer datos de la web para alimentar todo lo anterior.
- **APIs en producción**: rate limits, fallback entre providers, observabilidad de costos.

---

## 6. Entrega de los ejercicios

Sube tu notebook con los 5 ejercicios completados al repo (o entrégalo al docente).
Recuerda:
- No subas tu `GROQ_API_KEY`. Usa `getpass` o variables de entorno.
- Si tus textos son reales de Arca, **anonimizá** nombres, marcas, identificadores.

*Código + datos: github.com/cmosquerat/arca-diplomado/tree/main/clase-34*
""")

# =============================================================================
#  Persistir
# =============================================================================
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"✓ Notebook generado: {OUT} ({len(cells)} celdas)")
