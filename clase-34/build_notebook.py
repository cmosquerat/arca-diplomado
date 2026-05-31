"""
Construye Clase_34_LSTM_a_LLM.ipynb --- versión PBL v5 (final).

Perfil: profesionales seniors de Arca Continental (embotelladora), primer contacto
real con LLMs. La teoría completa está en el deck (53 pp.).

Estructura (~75 celdas):
  0. Setup
  1. Anatomía del endpoint OpenAI-compatible (PROTOCOLO, no marca)
  2. Recap clase 33 (negación rompe TF-IDF)
  3. Pre-Transformer en acción: char-LSTM sobre Don Quijote (demo histórica)
  4. ¿Y si lo entrenamos nosotros? Mini-Transformer Keras (spoiler: no funciona ~51%)
  5. 10 apps demostradas con Groq (clasif, extrae, resume, responde, chat, búsqueda, +3 Gradio)
  6. 2 ejemplos resueltos paso a paso (prioridad email + tema artículo)
  7. 5 ejercicios de clasificación variada (urgencia, tipo queja, área, producto, intención)
  8. Cierre

Narrativa: LSTM histórico genera Quijote → entrenar Transformer desde cero no funciona →
por eso usamos LLM pre-entrenado vía Groq → ahora resuelves 5 problemas de clasificación.
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
# Clase 34 — Del LSTM al LLM (Notebook PBL)

**Diplomado en Data Science Aplicada con Python para la Toma de Decisiones**
Arca Continental Ecuador | UDLA

---

> **Pregunta del día**: ¿cómo le enseñamos a una máquina a entender el lenguaje de Arca
> —manuales, tickets, quejas— sin contratar 50 lingüistas?

La teoría completa (Transformer, atención, BETO, escala) está en el **deck**.
Este notebook recorre la misma historia en código, terminando con **5 ejercicios prácticos**.

### Ruta del notebook

| Parte | Qué | Cuánto |
|---|---|---|
| 0–2 | Setup, anatomía del endpoint OpenAI, recap clase 33 | corto |
| **3** | **char-LSTM sobre Don Quijote** (demo histórica: así se hacía antes) | corre ~2 min en GPU |
| **4** | **Mini-Transformer Keras** (spoiler: entrenar desde cero NO funciona, ~51%) | corre ~30 s |
| 5 | **10 apps demostradas con Groq** (clasif, extrae, resume, chat, búsqueda + 3 Gradio) | el grueso |
| 6 | 2 ejemplos resueltos paso a paso | scaffold |
| 7 | **5 ejercicios de clasificación variada** | tu entregable |
| 8 | Cierre | corto |

### Los 5 ejercicios (datos provistos, todos de clasificación, dominios distintos)

| # | Qué clasifica | Categorías |
|---|---|---|
| E1 | Urgencia de ticket de mantenimiento | ALTA / MEDIA / BAJA |
| E2 | Tipo de queja de cliente | PRODUCTO / ENTREGA / FACTURACIÓN / ATENCIÓN |
| E3 | Área responsable (routing interno) | MANTENIMIENTO / CALIDAD / LOGÍSTICA / COMERCIAL / RRHH |
| E4 | Categoría comercial de producto | GASEOSA / AGUA / JUGO / ISOTÓNICA / ENERGÉTICA |
| E5 | Intención del mensaje del cliente | CONSULTA / RECLAMO / SUGERENCIA / AGRADECIMIENTO |

---
""")

# =============================================================================
#  0. SETUP
# =============================================================================
md("""
## 0. Setup

Instalamos lo necesario. Funciona en Colab (con GPU T4) y en local.
""")

code("""
import sys, os
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    os.system("pip install -q openai sentence-transformers gradio tensorflow")
print("✓ Setup listo")
""")

# =============================================================================
#  1. ANATOMÍA DEL ENDPOINT
# =============================================================================
md("""
## 1. Anatomía del endpoint — "OpenAI-compatible" es un PROTOCOLO, no la marca

**Concepto que evita confusión**:

> `from openai import OpenAI` **NO** significa que estás usando ChatGPT.

Es la librería cliente que habla un protocolo HTTP particular. OpenAI publicó la forma del
JSON que su API espera y devuelve; esa forma se volvió un estándar de facto. Hoy varios
providers respetan ese contrato — el mismo cliente Python sirve para todos:

| Provider | base_url | Quién lo corre |
|---|---|---|
| OpenAI (GPT-5, etc.) | `https://api.openai.com/v1` | OpenAI (pago) |
| **Groq** *(usamos hoy)* | `https://api.groq.com/openai/v1` | Groq (gratis hasta 30 RPM) |
| Together / Fireworks | `https://api.together.xyz/v1` | etc. |
| Anthropic Claude | (otro protocolo, librería `anthropic`) | — |

**Cambiar de provider = cambiar 2 líneas** (`base_url` + `api_key`).

### Pega tu API key de Groq

Crea cuenta gratis en https://console.groq.com (sin tarjeta), genera una API key,
pégala cuando te la pida la siguiente celda:
""")

code("""
import getpass
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = getpass.getpass("Pega tu GROQ_API_KEY: ")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
""")

md("""
### El cliente y el helper `llm()`

Una sola instancia, una función de 4 líneas. Esto es todo el "infra" del notebook.
""")

code("""
from openai import OpenAI

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
MODEL  = "llama-3.1-8b-instant"

def llm(prompt, system=None, model=MODEL, temperature=0.3, max_tokens=400):
    msgs = ([{"role":"system","content":system}] if system else []) \\
           + [{"role":"user","content":prompt}]
    r = client.chat.completions.create(model=model, messages=msgs,
                                       temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content
""")

md("""
### El JSON crudo — qué viaja por debajo

La librería envuelve un HTTP POST con JSON. Veámoslo crudo con `model_dump()`:
""")

code("""
import json as jsn

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role":"system","content":"Eres asistente de planta de Arca. Responde corto."},
        {"role":"user","content":"¿Qué hago primero si una llenadora vibra raro?"},
    ],
    temperature=0.3, max_tokens=120,
)

print("=== RESPONSE (lo que devuelve el server) ===")
print(jsn.dumps(resp.model_dump(), ensure_ascii=False, indent=2)[:1100])
print()
print("=== El texto generado está en .choices[0].message.content ===")
print(resp.choices[0].message.content)
""")

md("""
**Lo que ves**:
- `choices[0].message.content` → texto generado.
- `usage` → tokens consumidos.
- `finish_reason` → `stop` (terminó bien) o `length` (se quedó sin tokens).

Eso es todo. Cualquier provider que respete esta forma se usa con `openai`.

---
""")

# =============================================================================
#  2. RECAP CLASE 33
# =============================================================================
md("""
## 2. El problema que resolvemos (recap clase 33)

En la clase pasada, TF-IDF + LogReg sobre `muchocine_sentimiento` llegó a **84%** de
accuracy. Pero **no entiende la negación**. Veámoslo:
""")

code("""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CSV_URL = \"""" + REPO + """/muchocine_sentimiento.csv\"
df = pd.read_csv(CSV_URL); df["review"] = df["review"].astype(str)
Xtr, Xte, ytr, yte = train_test_split(df["review"], df["label"],
                                       test_size=0.2, random_state=42, stratify=df["label"])
vec = TfidfVectorizer(strip_accents="unicode", min_df=3, ngram_range=(1,2), max_features=30000)
clf_tfidf = LogisticRegression(max_iter=1000, C=3.0).fit(vec.fit_transform(Xtr), ytr)
acc_tfidf = accuracy_score(yte, clf_tfidf.predict(vec.transform(Xte)))
print(f"Baseline TF-IDF: {acc_tfidf:.1%}")

TRAMPAS = [
    ("no me gustó nada la trama",            0),
    ("no es para nada mala, me sorprendió",  1),
    ("no podría estar más contento",         1),
    ("buena fotografía pero no, no funciona",0),
]
print(f"\\n{'Real':<6} {'Pred':<6} Frase")
for f, real in TRAMPAS:
    p = int(clf_tfidf.predict(vec.transform([f]))[0])
    ok = "✓" if p == real else "✗"
    print(f"{ok} {'POS' if real else 'NEG':<4} {'POS' if p else 'NEG':<4}   {f}")
""")

md("""
TF-IDF cuenta palabras y no entiende el orden. **¿Cuál era la respuesta histórica?**
Redes con memoria: LSTM. Vamos a verla en acción antes de pasar a los LLMs.

---
""")

# =============================================================================
#  3. CHAR-LSTM SOBRE DON QUIJOTE
# =============================================================================
md("""
## 3. Antes del Transformer: char-LSTM sobre Don Quijote

**Pregunta histórica**: ¿qué se hacía para que una máquina "generara" texto en español
antes de los Transformers? Respuesta: LSTM caracter-a-caracter.

Vamos a entrenar uno sobre **Don Quijote** (~2 MB en español del siglo XVII) y ver qué
escribe. La idea es **predecir el siguiente carácter** y realimentar. Es **exactamente la
misma idea** que hace GPT hoy, pero con caracteres (no tokens) y sin atención.

> Entrenamiento: ~2 min en GPU T4 de Colab. En CPU es más lento (~10 min) pero también corre.
""")

code('''
import urllib.request, re, os

CORPUS_URL = "https://www.gutenberg.org/files/2000/2000-0.txt"
CACHE = "/content/quijote.txt" if IN_COLAB else "/tmp/quijote.txt"

if os.path.isfile(CACHE):
    text = open(CACHE).read()
    print(f"Quijote cacheado: {len(text):,} chars")
else:
    print("Descargando Don Quijote...")
    req = urllib.request.Request(CORPUS_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    # Recortar header/footer Gutenberg
    s = re.search(r"\\*\\*\\* START OF.*?\\*\\*\\*", raw, flags=re.S | re.I)
    e = re.search(r"\\*\\*\\* END OF.*?\\*\\*\\*",   raw, flags=re.S | re.I)
    if s: raw = raw[s.end():]
    if e: raw = raw[:e.start()]
    raw = raw.lower().replace("\\r", "")
    raw = re.sub(r"\\n{3,}", "\\n\\n", raw)
    raw = re.sub(r"[^a-z0-9áéíóúñü¿¡!?,.;:\\'\\"\\-\\n() ]", "", raw)
    raw = re.sub(r"[ \\t]{2,}", " ", raw)
    text = raw.strip()
    open(CACHE, "w").write(text)
    print(f"Descargado y limpio: {len(text):,} chars")

print(f"\\nMuestra del corpus:\\n{text[1500:1800]}")
''')

code("""
import numpy as np

chars = sorted(set(text))
V = len(chars)
ci = {c: i for i, c in enumerate(chars)}
ic = {i: c for c, i in ci.items()}
data = np.array([ci[c] for c in text], dtype=np.int32)
print(f"Vocabulario: {V} caracteres distintos")
print(f"Primeros 20: {chars[:20]}")
""")

md("""
### Construir el dataset de ventanas deslizantes

Cada ejemplo: 60 caracteres de contexto → el siguiente carácter.
""")

code("""
SEQ, STEP = 60, 10
X = np.array([data[i:i+SEQ] for i in range(0, len(data)-SEQ-1, STEP)], dtype=np.int32)
y = np.array([data[i+SEQ]   for i in range(0, len(data)-SEQ-1, STEP)], dtype=np.int32)
print(f"Ejemplos de entrenamiento: {len(X):,}  (SEQ={SEQ}, STEP={STEP})")
print(f"Ejemplo X[0]: {repr(''.join(ic[i] for i in X[0]))}")
print(f"Ejemplo y[0]: {repr(ic[y[0]])}")
""")

md("""
### El modelo --- 8 líneas de Keras
""")

code("""
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input

tf.keras.utils.set_random_seed(42)
print("GPU:", tf.config.list_physical_devices('GPU'))

device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
with tf.device(device):
    model = Sequential([
        Input((SEQ,)),
        layers.Embedding(V, 64),
        layers.LSTM(256),
        layers.Dense(V),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
model.summary()
""")

code("""
import time
t0 = time.time()
with tf.device(device):
    hist = model.fit(X, y, epochs=10, batch_size=128, verbose=2)
print(f"\\n✓ Entrenamiento: {time.time()-t0:.1f}s")
print(f"Loss inicial: {hist.history['loss'][0]:.2f} → final: {hist.history['loss'][-1]:.2f}")
""")

md("""
### Generar texto --- el termostato de la `temperature`

Para generar, pedimos al modelo la distribución del siguiente carácter, muestreamos uno
(con cierta `temperature`), lo agregamos y repetimos.

- **temp 0.2**: siempre el más probable → conservador, repetitivo.
- **temp 0.8**: muestreo más arriesgado → más variado.
- **temp 1.2**: casi uniforme → creativo, a veces gibberish.
""")

code("""
def sample(seed, n=300, temperature=0.5):
    s = [ci.get(c, 0) for c in seed.lower()][-SEQ:]
    s = [0] * (SEQ - len(s)) + s
    out = seed
    with tf.device(device):
        for _ in range(n):
            logits = model.predict(np.array([s]), verbose=0)[0] / max(temperature, 1e-6)
            p = np.exp(logits - logits.max()); p /= p.sum()
            nx = int(np.random.choice(V, p=p))
            out += ic[nx]
            s = s[1:] + [nx]
    return out

SEED = "en un lugar de la mancha de cuyo nombre no quiero acordarme "
for T in [0.2, 0.8, 1.2]:
    print(f"\\n{'='*70}\\nTEMP {T}\\n{'='*70}")
    print(sample(SEED, n=280, temperature=T))
""")

md("""
### Los 3 muros del LSTM (por qué no escaló a GPT)

| Muro | Síntoma |
|---|---|
| **Secuencial** | Lee paso a paso. No paraleliza. La GPU está ociosa el 90% del tiempo. |
| **Olvido** | El "vector resumen" del encoder pierde info de los primeros tokens (cuello de botella). |
| **No escala** | Entrenar modelos más grandes no rinde proporcionalmente. |

La pregunta de 2017: *¿y si nos olvidamos de la recurrencia? ¿y si dejamos que cada palabra
mire a todas las demás a la vez?* → eso es la atención → eso es el **Transformer**. Lo vemos
ya en código.

---
""")

# =============================================================================
#  4. MINI-TRANSFORMER KERAS (no aprende, ~51%)
# =============================================================================
md("""
## 4. ¿Y si entrenamos un Transformer desde cero?

**Spoiler**: con 2.620 reseñas, **no funciona**. Llega a ~51% (azar). Vamos a verlo en código
y entender por qué — esto justifica todo lo que viene después (LLMs pre-entrenados).
""")

code("""
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 8000
MAXLEN = 120
DIM = 32
HEADS = 2

# Tokenizar (reusamos df de la sección 2)
Xtr_t, Xte_t, ytr2, yte2 = train_test_split(df["review"], df["label"],
                                             test_size=0.2, random_state=42, stratify=df["label"])
tok = Tokenizer(num_words=VOCAB_SIZE, lower=True, oov_token="<oov>")
tok.fit_on_texts(Xtr_t)
Xtr_seq = pad_sequences(tok.texts_to_sequences(Xtr_t), maxlen=MAXLEN, padding="post", truncating="post")
Xte_seq = pad_sequences(tok.texts_to_sequences(Xte_t), maxlen=MAXLEN, padding="post", truncating="post")
print(f"Vocab efectivo: {min(len(tok.word_index)+1, VOCAB_SIZE)}, MAXLEN={MAXLEN}")
""")

code("""
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angles = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe.astype("float32")

def build_mini_transformer():
    inp = layers.Input((MAXLEN,))
    x = layers.Embedding(VOCAB_SIZE, DIM)(inp)
    x = x + positional_encoding(MAXLEN, DIM)
    a = layers.MultiHeadAttention(num_heads=HEADS, key_dim=16)(x, x)
    x = layers.LayerNormalization()(x + a)
    f = layers.Dense(64, activation="relu")(x)
    f = layers.Dense(DIM)(f)
    x = layers.LayerNormalization()(x + f)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)

with tf.device(device):
    mini = build_mini_transformer()
    mini.compile("adam", "binary_crossentropy", metrics=["accuracy"])
print(f"Mini-Transformer: {mini.count_params():,} parámetros")
""")

code("""
import time
t0 = time.time()
with tf.device(device):
    hist_m = mini.fit(Xtr_seq, ytr2, validation_split=0.15,
                       epochs=8, batch_size=32, verbose=2)
print(f"\\n✓ {time.time()-t0:.1f}s")

pred_m = (mini.predict(Xte_seq, verbose=0).ravel() > 0.5).astype(int)
acc_mini = accuracy_score(yte2, pred_m)
print(f"\\nBaseline TF-IDF:     {acc_tfidf:.3f}")
print(f"Mini-Transformer:    {acc_mini:.3f}")
print(f"\\n{'✓ Supera al TF-IDF' if acc_mini > acc_tfidf else '✗ NO supera (era esperado)'}")
""")

md("""
### Por qué falló (y por qué eso es la lección)

- ~250.000 parámetros vs ~2.000 ejemplos = **ratio brutal de sobreparametrización**.
- La atención se inicia al azar; necesita millones de ejemplos para encontrar patrones reales.
- 2.620 reseñas alcanzan a TF-IDF (palabras + estadística), pero **no a un Transformer
  entrenándose desde cero**.

**El trade-off honesto**: entrenar un Transformer desde cero **no es viable** para 99% de
los problemas reales del negocio. *El paradigma correcto es PARTIR de uno pre-entrenado.*

Y eso son BERT, BETO, Llama, GPT, Claude — los **LLMs**. Los usamos vía API en lo que sigue.

---
""")

# =============================================================================
#  5. LAS 10 APPS CON GROQ
# =============================================================================
md("""
## 5. Las 10 apps demostradas (con Groq)

Cada app es una función Python corta y reusable. Las primeras 7 son funciones puras;
las últimas 3 son apps Gradio (UI web).

### App 1 — Hello LLM
""")

code("""
print(llm("Da 3 causas posibles de vibración en una llenadora industrial. Sé conciso."))
""")

md("""
### App 2 — Clasificador zero-shot
""")

code('''
def clasificar(texto, categorias):
    cats = ", ".join(categorias)
    prompt = (f'Clasifica el siguiente texto en UNA SOLA categoría de: {cats}.\\n'
              f'Texto: "{texto}"\\n'
              'Responde SÓLO con la categoría exacta, sin explicación.')
    return llm(prompt, max_tokens=20, temperature=0.0).strip().lower()

CATS = ["mantenimiento", "queja_cliente", "logistica", "calidad", "ventas"]
EJEMPLOS = [
    "El compresor 3 vibra raro desde anoche",
    "Cliente reclamó botella derramada al recibir el pedido",
    "Camión a Guayaquil llegó con 4 horas de atraso",
    "Lote 4521 con sabor distinto, pedimos retención",
    "Cerramos el contrato con la cadena Tia",
]
for t in EJEMPLOS:
    print(f"  [{clasificar(t, CATS):<15}] {t}")
''')

md("""
### App 3 — Extracción de campos en formato libre
""")

code('''
def extraer_campos(texto):
    prompt = (f'Lee la siguiente orden de trabajo y devuelve, una por línea:\\n'
              f'- equipo: <equipo>\\n'
              f'- técnico: <técnico responsable>\\n'
              f'- prioridad: <alta/media/baja>\\n'
              f'No agregues nada más.\\n\\n'
              f'OT: "{texto}"')
    return llm(prompt, max_tokens=150, temperature=0.0)

OT = "OT-1041: el compresor 3 vibra fuerte. Asignar a Juan Pérez. Prioridad alta."
print(extraer_campos(OT))
''')

md("""
### App 4 — Resumidor de reportes de turno
""")

code('''
def resumir(texto, n_vinetas=3):
    p = (f"Resume el siguiente reporte en EXACTAMENTE {n_vinetas} viñetas, "
         f"máximo 15 palabras cada una. Sólo el resumen:\\n\\n{texto}")
    return llm(p, max_tokens=250, temperature=0.2)

REPORTE = ("Reporte turno línea 2, 30 mayo 2026 noche. Objetivo 18.000 botellas. "
           "Real 16.450 (91%). Causas: parada 35 min a las 23:40 por sensor de tapa "
           "posición 7 (cambio rápido); parada 22 min a las 02:15 por atasco en "
           "carrusel etiquetado. Rechazo calidad 0.3% (OK <0.5%). Filtro compresor 5 "
           "sucio, cambiar próximo mantenimiento jueves 6 junio.")
print(resumir(REPORTE))
''')

md("""
### App 5 — Generador de respuestas a quejas (borrador para CSR)
""")

code('''
SISTEMA_CSR = ("Eres asistente del equipo de servicio al cliente de Arca Continental. "
               "Redactás BORRADORES para que un agente humano revise. Empático, asumes "
               "responsabilidad, ofreces una solución concreta. Máximo 3 oraciones.")

queja = "Compré dos cajas y una llegó con la mitad de las botellas con la tapa rota."
print(llm(queja, system=SISTEMA_CSR, max_tokens=200))
''')

md("""
### App 6 — Chat multi-turno con memoria

El LLM no recuerda nada entre llamadas. **Vos mantenés la lista de mensajes.**
""")

code('''
def hablar(historial, mensaje, model=MODEL, max_tokens=250):
    historial.append({"role": "user", "content": mensaje})
    r = client.chat.completions.create(
        model=model, messages=historial,
        max_tokens=max_tokens, temperature=0.4)
    reply = r.choices[0].message.content
    historial.append({"role": "assistant", "content": reply})
    return reply

historial = [{"role":"system","content":"Eres técnico senior de mantenimiento. Concreto, pasos numerados."}]
for msg in [
    "Mi llenadora vibra más de lo normal. ¿Por dónde empiezo?",
    "Ya revisé los pernos, están firmes. ¿Qué sigue?",
    "El motor también está caliente. ¿Cambia el diagnóstico?",
]:
    print(f"\\n>>> {msg}")
    print(hablar(historial, msg))
''')

md("""
### App 7 — Búsqueda semántica con `sentence-transformers`

Convertís cada texto en un vector. Una nueva query → vector → cercanía coseno → tickets parecidos.
""")

code('''
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

INCIDENTES = [
    ("compresor 1 dejó de arrancar tras corte de luz",        "reset del variador y purga de aire"),
    ("llenadora línea 1 con desviación de volumen",            "calibrar sensor de nivel y limpiar boquillas"),
    ("etiquetadora arruga etiquetas en lote nuevo",            "ajustar tensión del rodillo, limpiar pegamento"),
    ("código E07 en llenadora 2",                              "revisar conexión del sensor de tapa antes de cambiarlo"),
    ("chiller 9 perdió presión durante la noche",              "buscar fuga, recargar refrigerante"),
    ("PLC línea 1 con pantalla en negro",                      "ciclo de energía y restaurar backup"),
    ("contador de botellas no avanza",                         "limpieza del encoder óptico"),
    ("filtro de aire saturado en compresor 5",                 "reemplazo del cartucho y registro en bitácora"),
    ("temperatura del aceite del compresor 5 fuera de rango",  "agregar refrigerante y revisar ventilador"),
    ("vibración elevada en motor del transportador",           "balanceo y verificación de pernos"),
]
TEXTOS = [t for t, _ in INCIDENTES]
EMBS = embedder.encode(TEXTOS, show_progress_bar=False)

def buscar(query, k=3):
    q = embedder.encode([query], show_progress_bar=False)
    sims = cosine_similarity(q, EMBS)[0]
    top = np.argsort(-sims)[:k]
    return [(float(sims[i]), INCIDENTES[i]) for i in top]

for q in ["problema con el sensor de tapa", "máquina hace ruido raro"]:
    print(f"\\n🔍 \\"{q}\\"")
    for sim, (texto, accion) in buscar(q, k=3):
        print(f"   {sim:.2f}  {texto}")
        print(f"          → acción tomada: {accion}")
''')

md("""
### App 8 — Gradio: clasificador con UI
""")

code('''
import gradio as gr

def app_clasificar(texto, categorias_csv):
    cats = [c.strip() for c in categorias_csv.split(",") if c.strip()]
    if not texto.strip(): return "(pega un texto)"
    return f"**{clasificar(texto, cats)}**"

demo_clasif = gr.Interface(
    fn=app_clasificar,
    inputs=[
        gr.Textbox(label="Ticket o queja", lines=3,
                   placeholder="Ej: El compresor 3 vibra desde anoche"),
        gr.Textbox(label="Categorías (separadas por comas)",
                   value="mantenimiento, queja_cliente, logistica, calidad, ventas"),
    ],
    outputs=gr.Markdown(label="Categoría"),
    title="Clasificador de tickets · Arca",
    flagging_mode="never",
)
# demo_clasif.launch(share=True, inline=True)
print("✓ App definida. Lanza con: demo_clasif.launch(share=True, inline=True)")
''')

md("""
### App 9 — Gradio: chat con tu documento
""")

code('''
MANUAL_DEMO = """Manual operativo rápido — Llenadora modelo XF-200, planta Arca Quito.

ESPECIFICACIONES
- Capacidad nominal: 12.000 botellas/hora.
- Presión operativa: entre 2.5 y 3.2 bar.
- Voltaje: 380 V trifásico.

MANTENIMIENTO PREVENTIVO
- Cambio de filtro de aire: cada semana.
- Lubricación del carrusel: aceite ISO 220, cada 80 horas.
- Mantenimiento mayor: cada 200 horas.

CÓDIGOS DE ERROR
- E03: baja presión, revisar bomba.
- E07: falla del sensor de tapa, revisar conexión antes de reemplazar el sensor.
- E12: temperatura del aceite fuera de rango, parar el equipo.

CONTACTOS
- Soporte técnico: ext. 2410 (Mario Núñez)."""

def chat_con_doc(documento, pregunta):
    if not documento.strip() or not pregunta.strip():
        return "(pega un documento y haz una pregunta)"
    prompt = (
        "Responde la pregunta usando SÓLO el siguiente documento. "
        "Si la respuesta no aparece, di: 'No aparece en el documento.'\\n\\n"
        f"=== DOCUMENTO ===\\n{documento}\\n\\n"
        f"=== PREGUNTA ===\\n{pregunta}\\n\\n"
        "=== RESPUESTA ==="
    )
    return llm(prompt, max_tokens=250, temperature=0.0)

demo_doc = gr.Interface(
    fn=chat_con_doc,
    inputs=[
        gr.Textbox(label="Documento", lines=10, value=MANUAL_DEMO),
        gr.Textbox(label="Pregunta",
                   placeholder="¿Cada cuánto se cambia el filtro de aire?"),
    ],
    outputs=gr.Textbox(label="Respuesta", lines=4),
    title="Chat con tu documento · Arca",
    flagging_mode="never",
)
# demo_doc.launch(share=True, inline=True)
print("✓ App definida.")
''')

md("""
### App 10 — Gradio: comparador Llama 8B vs Llama 70B
""")

code('''
def comparar_modelos(prompt_user, temperature):
    out = {}
    for name, m in [("8B", "llama-3.1-8b-instant"), ("70B", "llama-3.3-70b-versatile")]:
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=m, messages=[{"role":"user","content":prompt_user}],
                temperature=temperature, max_tokens=300)
            txt = r.choices[0].message.content
            out[name] = (f"### {name}  "
                         f"_({(time.time()-t0):.1f}s, {r.usage.total_tokens} tokens)_\\n\\n{txt}")
        except Exception as e:
            out[name] = f"### {name} ⚠️ no disponible\\n\\n{type(e).__name__}"
    return out["8B"], out["70B"]

demo_comp = gr.Interface(
    fn=comparar_modelos,
    inputs=[
        gr.Textbox(label="Prompt", lines=3,
                   value="Da 3 causas posibles de vibración en una llenadora industrial."),
        gr.Slider(0.0, 1.5, value=0.3, label="Temperatura"),
    ],
    outputs=[
        gr.Markdown(label="Llama-3.1-8B"),
        gr.Markdown(label="Llama-3.3-70B"),
    ],
    title="Comparador 8B vs 70B · Arca",
    flagging_mode="never",
)
# demo_comp.launch(share=True, inline=True)
print("✓ App definida.")
''')

md("""
### Tabbed: las 3 apps Gradio en una sola UI
""")

code('''
tabbed = gr.TabbedInterface(
    [demo_clasif, demo_doc, demo_comp],
    ["Clasificador", "Chat con doc", "8B vs 70B"],
    title="Asistente NLP · Arca Continental",
)
# tabbed.launch(share=True, inline=True)
print("✓ TabbedInterface lista.")
''')

# =============================================================================
#  6. DOS EJEMPLOS RESUELTOS PASO A PASO
# =============================================================================
md("""
---

## 6. Dos ejemplos resueltos paso a paso

Antes de tus 5 ejercicios, te dejamos **2 ejemplos completos** del mismo tipo, ya resueltos.
Mirá cómo se estructura el prompt y cómo se imprime la salida.

### Ejemplo A — Clasificar la prioridad de 3 emails

Tarea: para cada email, devolver `URGENTE`, `NORMAL` o `SPAM`.
""")

code('''
EMAILS_EJ = [
    "Tu paquete está retenido en aduana, abona la tasa AQUÍ para liberarlo HOY.",
    "Recordatorio: reunión mañana 10am en sala de juntas para revisar KPIs del mes.",
    "ALERTA: el compresor 3 acaba de parar, perdiendo producción. Atención inmediata.",
]

print("Email → Prioridad")
print("-" * 70)
for e in EMAILS_EJ:
    prompt = (f'Clasifica este email en UNA palabra: URGENTE, NORMAL o SPAM. '
              f'Sin explicación.\\n'
              f'Email: "{e}"')
    etiqueta = llm(prompt, max_tokens=10, temperature=0.0).strip().upper()
    print(f"[{etiqueta:<8}] {e}")
''')

md("""
**Lo que pasó**:
- Un solo prompt por email.
- `temperature=0.0` y `max_tokens=10` porque la respuesta es UNA palabra.
- `.strip().upper()` para normalizar el formato.

### Ejemplo B — Clasificar el tema de 3 artículos

Tarea: clasificar cada título de artículo en `TECNOLOGÍA`, `DEPORTES`, `SALUD` o `FINANZAS`.
""")

code('''
ARTICULOS_EJ = [
    "Nueva CPU de Apple promete el doble de rendimiento con la mitad del consumo",
    "Selección de Ecuador clasificó al Mundial 2026 con goleada en el último partido",
    "Estudio confirma beneficios del ejercicio diario para la salud cardiovascular",
]

print("Artículo → Tema")
print("-" * 80)
for a in ARTICULOS_EJ:
    prompt = (f'Clasifica el tema de este artículo en UNA palabra: '
              f'TECNOLOGÍA, DEPORTES, SALUD o FINANZAS. Sin explicación.\\n'
              f'Título: "{a}"')
    tema = llm(prompt, max_tokens=10, temperature=0.0).strip().upper()
    print(f"[{tema:<11}] {a}")
''')

md("""
**Patrón general** (los 5 ejercicios siguen este mismo molde):
1. Lista de N textos.
2. Para cada texto, un prompt corto pidiendo UNA categoría.
3. `temperature=0.0` (queremos la mejor predicción, no creatividad).
4. `max_tokens=10` (la respuesta es una palabra).
5. Imprimir tabla `texto → etiqueta`.

---
""")

# =============================================================================
#  7. LOS 5 EJERCICIOS
# =============================================================================
md("""
## 7. Tus 5 ejercicios — clasificación variada

Los 5 son clasificación pero **cada uno clasifica algo distinto**. Datos provistos.
Resolvé cada uno escribiendo el prompt apropiado e imprimiendo una tabla.
**No hay validador automático** — vos mirás la tabla y juzgás si tiene sentido.

---

### E1 — Clasificar URGENCIA de 10 tickets de mantenimiento

Categorías: `ALTA` / `MEDIA` / `BAJA`.

**Tip**: en el prompt, dale al LLM una pista de qué significa cada nivel (ej.: ALTA = atención
inmediata; MEDIA = esta semana; BAJA = se puede agendar).
""")

code('''
TICKETS_URGENCIA = [
    "El compresor 3 echa vapor desde anoche y la presión bajó al mínimo.",
    "Ruido leve en el carrusel de tapado, agendar inspección para el próximo turno.",
    "La llenadora línea 2 paró por completo, producción frenada.",
    "Calibración mensual del medidor de Brix programada para el viernes.",
    "Pintado de zona de carga pendiente para mantenimiento general.",
    "Sensor de tapa fallando intermitente, lote 9012 con scrap aumentando.",
    "Cambio de filtros HEPA del depósito 2, no urgente.",
    "El chiller 9 perdió presión, refrigeración del producto comprometida YA.",
    "Lubricación rutinaria del rodillo de la etiquetadora, próxima semana.",
    "Falla del PLC de la línea 1, pantalla en negro, no se puede producir.",
]

# COMPLETA: para cada ticket, llama a `llm` y obtén la urgencia.
# Imprime tabla ticket (truncado) → urgencia.

# for t in TICKETS_URGENCIA:
#     ...
''')

md("""
### E2 — Clasificar TIPO DE QUEJA de 10 quejas de clientes

Categorías: `PRODUCTO` / `ENTREGA` / `FACTURACIÓN` / `ATENCIÓN`.
""")

code('''
QUEJAS = [
    "La gaseosa que recibí sabe rara, no es la habitual.",
    "Me cobraron dos veces el mismo pedido en la app.",
    "Hace una semana llamé por mi reposición y nadie me responde.",
    "Mi pedido llegó tres días después de lo prometido.",
    "Recibí mi caja con 4 botellas con la tapa rota.",
    "El operador del call center fue muy descortés conmigo.",
    "Pagué por 24 unidades pero llegaron 22 en la caja.",
    "La factura llegó con el RUC equivocado, no puedo declarar.",
    "Sigo esperando que me respondan el reclamo desde hace 10 días.",
    "El producto llegó vencido aunque la etiqueta dice marzo 2027.",
]

# COMPLETA: para cada queja, llama a `llm` y obtén la categoría.
# Imprime tabla queja → categoría.

# for q in QUEJAS:
#     ...
''')

md("""
### E3 — Clasificar ÁREA RESPONSABLE de 10 incidencias

Categorías: `MANTENIMIENTO` / `CALIDAD` / `LOGÍSTICA` / `COMERCIAL` / `RRHH`.
""")

code('''
INCIDENCIAS = [
    "El medidor de Brix necesita recalibración trimestral.",
    "Falta gente en el turno de la línea 3 desde ayer.",
    "Bloqueo de la ruta Manta-Portoviejo por inundación.",
    "Cliente Tia pide renegociar precios para el próximo trimestre.",
    "Detectamos partícula extraña en una botella del lote 8830.",
    "El compresor 5 tiene fuga de aceite, agendar reparación.",
    "Diferencia de 12 cajas entre lo despachado y lo facturado.",
    "Renuncia inesperada del supervisor del turno noche.",
    "La promoción 2x1 disparó las ventas pero falta inventario.",
    "Reclamo del retail: etiquetas ilegibles en el lote 9012.",
]

# COMPLETA: para cada incidencia, llama a `llm` y obtén el área responsable.

# for i in INCIDENCIAS:
#     ...
''')

md("""
### E4 — Clasificar CATEGORÍA COMERCIAL de 10 productos

Categorías: `GASEOSA` / `AGUA` / `JUGO` / `ISOTÓNICA` / `ENERGÉTICA`.

Input: nombre + descripción de 1 línea.
""")

code('''
PRODUCTOS = [
    ("Cola Fresh",     "refresco carbonatado sabor cola, 500 ml, sin azúcar"),
    ("Vital Pura",     "agua natural sin gas, 600 ml, mineralización media"),
    ("Tropic Mango",   "jugo 100% natural de mango, sin conservantes, 450 ml"),
    ("PowerSport",     "bebida con electrolitos para deportistas, 750 ml"),
    ("EnergyMax",      "bebida con cafeína y taurina, 250 ml, ideal para estudiar"),
    ("Sparkle Limón",  "agua con gas sabor limón, sin azúcar, 600 ml"),
    ("Cítrico Mix",    "mezcla de jugos de naranja, mandarina y maracuyá, 1 litro"),
    ("Aqua Mineral",   "agua mineral natural, fuente subterránea, botella 1.5 L"),
    ("Boost Pro",      "bebida funcional con BCAA y cafeína, 500 ml"),
    ("Cola Cero",      "cola sin azúcar y sin calorías, 355 ml"),
]

# COMPLETA: para cada producto, llama a `llm` con el nombre + descripción
# y obtén la categoría comercial. Imprime tabla nombre → categoría.

# for nombre, desc in PRODUCTOS:
#     ...
''')

md("""
### E5 — Clasificar INTENCIÓN del mensaje del cliente

Categorías: `CONSULTA` (pregunta info) / `RECLAMO` (queja activa) / `SUGERENCIA` (propone
mejora) / `AGRADECIMIENTO` (positivo, no queja).

**Atención**: esto NO es sentimiento. Distingue **qué quiere** el cliente, no si está
contento o no. Una pregunta cortés es `CONSULTA`, no `POSITIVO`.
""")

code('''
MENSAJES = [
    "¿En qué tiendas venden la nueva presentación de 750 ml?",
    "Sería buena idea sacar la edición sin azúcar en 500 ml también.",
    "Gracias por la atención rápida, resolvieron mi caso en un día.",
    "La gaseosa me llegó caliente otra vez, qué pésimo servicio.",
    "¿Tienen el sabor de manzana disponible en el supermercado de Cumbayá?",
    "Sugerencia: agreguen un botón para reportar entregas tardías en la app.",
    "Excelente promoción la del 2x1, sigan así por favor.",
    "Llevo 3 días esperando mi pedido y nadie me responde.",
    "¿Cómo hago para devolver una caja que vino con productos vencidos?",
    "Recomendarían quitar el envoltorio extra, es desperdicio de plástico.",
]

# COMPLETA: para cada mensaje, llama a `llm` y obtén la intención.

# for m in MENSAJES:
#     ...
''')

# =============================================================================
#  8. CIERRE
# =============================================================================
md("""
---

## 8. Lo que te llevas

Hoy recorriste **la historia completa** en código:

1. **Pre-Transformer**: viste un char-LSTM entrenando sobre el Quijote y generando español.
   Es exactamente lo que hacía Google Translate en 2016.
2. **Transformer desde cero**: probaste entrenarlo y vio que **no funciona** con poquitos datos.
   Te justifica el siguiente paso.
3. **LLM pre-entrenado vía API**: en 5 líneas hablás con Llama 3.1 (gratis vía Groq) y le
   resolvés 5 tareas distintas de clasificación.

### Regla del oficio

> En tu trabajo, antes de pedir un proyecto largo, **prototipá 1 día con un Notebook + Groq**.
> Si el LLM te resuelve el 80% de la tarea, ya tienes el business case.

### El siguiente paso — Módulo 6 (IA Generativa)

- **Prompting avanzado** (few-shot, plantillas reusables).
- **RAG**: conectar el LLM a tus propios documentos.
- **Agentes**: LLMs que ejecutan acciones (consultar BD, mandar emails).
- **Web scraping** para alimentar todo lo anterior.

---

## 9. Entrega

Subí tu notebook con los 5 ejercicios resueltos. Antes de subir:
- **No incluyas tu `GROQ_API_KEY`** (ya usamos `getpass`, así que está bien).
- Para cada ejercicio, **revisá la tabla impresa** y anotá en una celda markdown
  si los resultados te parecen razonables (1-2 oraciones).

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
