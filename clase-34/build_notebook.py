"""
Construye Clase_34_LSTM_a_LLM.ipynb --- Del LSTM al LLM (puente al Módulo 6).
Compatible con Colab T4 (instala deps si hace falta).

HILO: "¿Cómo le enseñamos a una máquina a entender el lenguaje de Arca
       --- manuales, tickets, quejas --- sin contratar 50 lingüistas?"

Estructura:
  0. Setup
  1. Recap clase 33 (TF-IDF baseline + el problema de la negación)
  2. Bloque 1 -- char-LSTM sobre Don Quijote (historia + 3 muros)
  3. Bloque 2 -- Atención DESDE CERO (numpy + interpretabilidad)
  4. Bloque 3a -- Mini-Transformer Keras (honest fail)
  5. Bloque 3b -- BETO EN PROFUNDIDAD: 5 apps + interpretabilidad por capa
  6. Bloque 4 -- LLMs con Groq: 8 apps + comparativa modelos/temperatura
  7. Cierre + ejercicios
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
# Clase 34 --- Del LSTM al LLM
### Cómo nacieron los Transformers --- y cómo los usamos hoy en Arca

**Diplomado en Data Science Aplicada con Python para la Toma de Decisiones**
Arca Continental Ecuador | UDLA

---

## La pregunta del día

> **¿Cómo le enseñamos a una máquina a entender el lenguaje de Arca ---
> manuales de mantenimiento, tickets de planta, quejas de consumidor ---
> sin contratar 50 lingüistas?**

Esta pregunta nos acompaña en los 4 bloques. Cada bloque es una respuesta cada vez mejor:

1. **Antes del Transformer**: cómo se hacía con RNN/LSTM (y por qué chocaba con 3 muros).
2. **Atención**: el paper que cambió todo (2017). Lo desarmamos desde cero en NumPy.
3. **BETO pre-entrenado**: 5 aplicaciones prácticas + interpretabilidad por capas.
4. **LLMs con Groq**: 8 aplicaciones empresariales reales con API gratis.

**Cada sección incluye interpretabilidad**: no es magia, vamos a abrirle la caja.

---
""")

# =============================================================================
#  0. SETUP
# =============================================================================
md("## 0. Setup --- corre esta celda primero")
code("""
import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import os
    os.system("pip install -q transformers sentence-transformers openai gensim")
    print("✓ Dependencias instaladas")
else:
    print("Modo local --- asumiendo deps ya instaladas")

# GPU?
try:
    import torch
    print(f"PyTorch GPU: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
except: pass
try:
    import tensorflow as tf
    print(f"TF GPU: {bool(tf.config.list_physical_devices('GPU'))}")
except: pass
""")

md("""
**API key de Groq** (gratis, sin tarjeta, en https://console.groq.com).
Cuando llegues al Bloque 4 vas a necesitarla. Te la pedimos con `getpass` para
que no quede guardada en el notebook.
""")

# =============================================================================
#  1. RECAP CLASE 33 -- el problema de la negación
# =============================================================================
md("""
---
## 1. Recap clase 33 --- el problema de la negación

En la clase pasada construimos la **escalera de representación de texto**:
BoW → TF-IDF → Word2Vec → atención (concepto).

Nuestro mejor clasificador clásico (TF-IDF + Regresión Logística) llegó a **84% de accuracy**
sobre muchocine_sentimiento. Es mucho. Pero tiene un problema fundamental:
**no entiende la negación**.

Veámoslo en acción.
""")

code(f"""
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
np.random.seed(42)

df = pd.read_csv("{REPO}/muchocine_sentimiento.csv")
df["review"] = df["review"].astype(str)
print(f"Corpus: {{len(df)}} reseñas, balance: {{df['label'].value_counts().to_dict()}}")

Xtr, Xte, ytr, yte = train_test_split(df["review"], df["label"], test_size=0.2,
                                       random_state=42, stratify=df["label"])
vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", min_df=3,
                      ngram_range=(1, 2), max_features=30000)
clf = LogisticRegression(max_iter=1000, C=3.0).fit(vec.fit_transform(Xtr), ytr)
acc = accuracy_score(yte, clf.predict(vec.transform(Xte)))
print(f"TF-IDF + LogReg acc = {{acc:.4f}}")
""")

md("""
### 1.1. Interpretabilidad --- ¿qué palabras pesan más?

Una regresión logística es interpretable: cada palabra tiene un coeficiente.
Coeficiente positivo → empuja a POSITIVO. Negativo → empuja a NEGATIVO.
""")

code("""
import numpy as np
feature_names = np.array(vec.get_feature_names_out())
coefs = clf.coef_[0]
top_pos = np.argsort(coefs)[-15:][::-1]
top_neg = np.argsort(coefs)[:15]
print("Top 15 palabras POSITIVAS:")
for i in top_pos: print(f"  {coefs[i]:+.3f}  {feature_names[i]}")
print("\\nTop 15 palabras NEGATIVAS:")
for i in top_neg: print(f"  {coefs[i]:+.3f}  {feature_names[i]}")
""")

md("""
### 1.2. Las frases-trampa --- donde TF-IDF se cae

Ahora la prueba: 8 frases con negación, curadas a propósito.
""")

code("""
TRAMPAS = [
    ("no me gustó nada la trama, qué decepción", 0),
    ("no es para nada mala, me sorprendió", 1),
    ("esperaba mucho más, no la recomiendo", 0),
    ("no podría estar más contento con esta película", 1),
    ("ni una sola escena aburrida, brillante", 1),
    ("no entiendo cómo a alguien le puede gustar esto", 0),
    ("no me arrepiento de haberla visto", 1),
    ("buena fotografía pero no, no funciona", 0),
]
LABEL = {0: "NEGATIVO", 1: "POSITIVO"}

frases = [t[0] for t in TRAMPAS]
reales = [t[1] for t in TRAMPAS]
preds = clf.predict(vec.transform(frases))
probs = clf.predict_proba(vec.transform(frases))[:, 1]

print(f"{'real':<10} {'pred':<10} {'prob+':<8} frase")
print("-"*90)
for f, r, p, pr in zip(frases, reales, preds, probs):
    ok = "✓" if r == p else "✗"
    print(f"{ok} {LABEL[r]:<8} {LABEL[p]:<8} {pr:.2f}     {f}")
aciertos = sum(1 for r, p in zip(reales, preds) if r == p)
print(f"\\nAciertos: {aciertos}/8 ({aciertos*100//8}%)")
""")

md("""
**Lo que ves**: TF-IDF acierta en ~4-5 de las 8 trampas. Las palabras "no" y "gustó"
son features independientes, no sabe que están relacionadas. **El conteo no captura el orden ni
el contexto.**

Esto es exactamente lo que motivó la búsqueda de algo mejor: las **redes con memoria** (LSTM)
y, después, los **Transformers**.

---
""")

# =============================================================================
#  2. BLOQUE 1 -- char-LSTM SOBRE DON QUIJOTE
# =============================================================================
md("""
## 2. Bloque 1 --- Antes del Transformer: una LSTM que escribe español

### 2.1. Contexto histórico

| Año | Hito |
|---|---|
| 1986 | Backpropagation through time (RNN) |
| 1997 | **LSTM** (Hochreiter & Schmidhuber) --- compuertas, memoria a largo plazo |
| 2014 | **Seq2seq** (Sutskever et al.) --- encoder-decoder LSTM para traducción |
| 2016 | Google Translate adopta seq2seq con atención |
| 2017 | **Attention is All You Need** --- adiós a la recurrencia |

Entre 2014 y 2017, **toda la NLP del mundo corría sobre LSTM**. Google Translate,
Siri, autocompletado de teclado, todos. Hasta que se chocaron con 3 muros.

### 2.2. La idea más loca: una LSTM que GENERA texto

En vez de pedirle a la LSTM que clasifique, le pedimos que **prediga el siguiente carácter**.
Luego ese carácter lo realimentamos como entrada. Y así, carácter por carácter, escribe sola.

> Esta es **exactamente la idea** detrás de GPT --- pero con tokens en vez de caracteres,
> y atención en vez de recurrencia, y a escala bestial.

Vamos a entrenar uno con **Don Quijote** (2 MB de español del siglo XVII) y ver qué genera.
""")

code(f"""
import urllib.request, re

CORPUS_URL = "https://www.gutenberg.org/files/2000/2000-0.txt"
CACHE = "/tmp/quijote.txt" if not IN_COLAB else "/content/quijote.txt"

try:
    text = open(CACHE).read()
    print(f"✓ Quijote en cache ({{len(text)}} chars)")
except FileNotFoundError:
    print(f"Descargando {{CORPUS_URL}} ...")
    raw = urllib.request.urlopen(CORPUS_URL, timeout=60).read().decode("utf-8", "ignore")
    # Recortar header/footer Gutenberg
    s = re.search(r"\\*\\*\\* START OF.*?\\*\\*\\*", raw, flags=re.S | re.I)
    e = re.search(r"\\*\\*\\* END OF.*?\\*\\*\\*", raw, flags=re.S | re.I)
    if s: raw = raw[s.end():]
    if e: raw = raw[:e.start()]
    raw = raw.lower().replace("\\r", "")
    raw = re.sub(r"\\n{{3,}}", "\\n\\n", raw)
    raw = re.sub(r"[^a-z0-9áéíóúñü¿¡!?,.;:'\\"\\-\\n() ]", "", raw)
    raw = re.sub(r"[ \\t]{{2,}}", " ", raw)
    text = raw.strip()
    open(CACHE, "w").write(text)
    print(f"✓ Guardado ({{len(text)}} chars)")
""")

code("""
chars = sorted(set(text))
V = len(chars)
ci = {c: i for i, c in enumerate(chars)}
ic = {i: c for c, i in ci.items()}
print(f"Vocabulario: {V} caracteres únicos")
print(f"Primeros 20: {chars[:20]}")
print(f"\\nMuestra del corpus:\\n{text[2000:2400]}")
""")

md("""
### 2.3. Construir el dataset --- ventanas deslizantes

Cada **ejemplo de entrenamiento** es: una ventana de 60 caracteres → el siguiente carácter.
""")

code("""
import numpy as np

SEQ = 60
STEP = 10  # cada cuántos caracteres tomamos una nueva ventana
data = np.array([ci[c] for c in text], dtype=np.int32)
X = np.array([data[i:i+SEQ] for i in range(0, len(data)-SEQ-1, STEP)], dtype=np.int32)
y = np.array([data[i+SEQ]   for i in range(0, len(data)-SEQ-1, STEP)], dtype=np.int32)
print(f"X shape: {X.shape}   y shape: {y.shape}")
print(f"\\nEjemplo X[0] (60 caracteres):")
print(repr("".join(ic[i] for i in X[0])))
print(f"\\ny[0] (siguiente carácter): {repr(ic[y[0]])}")
""")

md("""
### 2.4. El modelo --- 8 líneas de Keras

`Embedding(vocab, 64) → LSTM(256) → Dense(vocab, softmax)`. ~350K parámetros.
""")

code("""
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input
tf.keras.utils.set_random_seed(42)

with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
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

md("""
### 2.5. Entrenar (~1-3 min en GPU T4)
""")

code("""
import time
t0 = time.time()
with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    hist = model.fit(X, y, epochs=10, batch_size=128, verbose=2)
print(f"\\n✓ Entrenamiento: {time.time()-t0:.1f}s")
print(f"Loss inicial: {hist.history['loss'][0]:.3f}   final: {hist.history['loss'][-1]:.3f}")
""")

md("""
### 2.6. Interpretabilidad --- la curva de aprendizaje

El loss empieza en ~3 (puro azar sobre ~70 caracteres) y baja a ~1.2.
""")

code("""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(hist.history['loss'])+1), hist.history['loss'],
        color="#C82B40", lw=2.4, marker="o", ms=6)
ax.set_xlabel("época"); ax.set_ylabel("loss")
ax.set_title("El char-LSTM aprende español carácter por carácter")
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
""")

md("""
### 2.7. Generación --- y el termostato de la temperatura

Para generar, predecimos el siguiente carácter, lo agregamos, repetimos.
Pero al elegir, podemos modular con **temperatura**:

- `T = 0.2` → siempre el más probable (conservador, repetitivo).
- `T = 1.0` → muestreamos según las probabilidades del modelo.
- `T = 1.5` → más uniforme (creativo, caótico).

Matemáticamente: `softmax(logits / T)`. Bajar T = enfocar; subir T = explorar.
""")

code("""
def sample(seed, n=320, temperature=0.5):
    s = [ci.get(c, 0) for c in seed.lower()][-SEQ:]
    s = [0] * (SEQ - len(s)) + s
    out = seed
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        for _ in range(n):
            logits = model.predict(np.array([s]), verbose=0)[0] / max(temperature, 1e-6)
            p = np.exp(logits - logits.max()); p /= p.sum()
            nx = int(np.random.choice(V, p=p))
            out += ic[nx]; s = s[1:] + [nx]
    return out

SEED = "en un lugar de la mancha de cuyo nombre no quiero acordarme "
print("="*70, "\\nTEMP 0.2 (conservador, repetitivo)\\n", "="*70)
print(sample(SEED, 280, 0.2))
print("\\n", "="*70, "\\nTEMP 0.8 (balance)\\n", "="*70)
print(sample(SEED, 280, 0.8))
print("\\n", "="*70, "\\nTEMP 1.2 (creativo, casi gibberish)\\n", "="*70)
print(sample(SEED, 280, 1.2))
""")

md("""
**¿Qué ves?**
- Sin que nadie le explicara qué es una palabra, ortografía o gramática, la red **descubrió** que:
  - Las letras se agrupan en palabras con espacios.
  - Hay puntuación que rompe oraciones.
  - Algunas palabras son frecuentes (`la`, `el`, `que`, `de`).
  - El estilo es Cervantes: `sancho`, `caballero`, `mancha`.
- Pero **se pierde el hilo** en frases medianas: las oraciones empiezan bien y divagan.

**Esto era el TECHO pre-Transformer en 2016.** GPT hace exactamente lo mismo
--- predecir el siguiente token --- pero con atención y a escala bestial.

### 2.8. Los 3 muros de la LSTM --- por qué no escaló

| # | Muro | Síntoma |
|---|---|---|
| 1 | **Secuencial** | Lee paso a paso. No paraleliza. La GPU está parada el 90% del tiempo. |
| 2 | **Olvido** | El "vector resumen" del encoder pierde info de los primeros tokens. |
| 3 | **No escala** | Entrenar modelos más grandes no rinde proporcionalmente. |

La pregunta de 2017: *¿y si nos olvidamos de la recurrencia? ¿y si dejamos que cada palabra
mire a todas las demás a la vez?*

---
""")

# =============================================================================
#  3. BLOQUE 2 -- ATENCIÓN DESDE CERO
# =============================================================================
md("""
## 3. Bloque 2 --- Atención desde cero (con NumPy)

### 3.1. Contexto: el paper

**Vaswani et al. (Google Brain, NeurIPS 2017)** --- *Attention is All You Need*.

> Más de 150.000 citas a 2026. Uno de los papers más citados de la historia de la
> computación. Propone el **Transformer**, la base de GPT, BERT, Claude, Gemini, Llama.

La idea radical: **cada palabra mira a todas las demás al mismo tiempo y aprende
cuáles le importan**. Nada de leer en orden. Todo en paralelo.

### 3.2. La fórmula

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V$$

Vamos a construir esto **en NumPy puro**, paso por paso, con una frase chiquita.
""")

code("""
import numpy as np
np.random.seed(0)

# Una frase de 5 palabras. Cada palabra es un vector de 4 dimensiones.
frase = ["no", "me", "gustó", "nada", "esto"]
N = len(frase)
D = 4  # dimension del embedding

# Embeddings (simulamos los que aprendería el modelo)
E = np.array([
    [ 1.0, -0.5,  0.2,  0.8],   # no   (negación)
    [ 0.1,  0.3, -0.1,  0.2],   # me   (pronombre)
    [-0.5,  0.9,  0.4, -0.2],   # gustó (verbo afectivo)
    [ 0.6, -0.3,  0.1,  0.7],   # nada (intensificador negativo)
    [ 0.2,  0.1, -0.4,  0.1],   # esto (pronombre)
])
print(f"Embeddings E: shape {E.shape}\\n{E}")
""")

md("""
### 3.3. Las 3 proyecciones --- Query, Key, Value

Cada palabra se proyecta a 3 espacios distintos. Las matrices `Wq, Wk, Wv` las **aprende**
el modelo. Aquí las generamos al azar para ilustrar la mecánica.
""")

code("""
DK = 4  # dimension de Q y K
Wq = np.random.randn(D, DK) * 0.3
Wk = np.random.randn(D, DK) * 0.3
Wv = np.random.randn(D, DK) * 0.3

Q = E @ Wq   # (N, DK)
K = E @ Wk   # (N, DK)
V = E @ Wv   # (N, DK)

print(f"Q (lo que cada palabra busca):\\n{Q}\\n")
print(f"K (lo que cada palabra ofrece):\\n{K}\\n")
print(f"V (lo que cada palabra aporta):\\n{V}")
""")

md("""
### 3.4. Scores --- ¿cuánto se parecen Query y Key?

`scores[i, j]` = cuánto se parece el query de la palabra i con el key de la palabra j.
Producto punto escalado.
""")

code("""
scores = (Q @ K.T) / np.sqrt(DK)
print(f"Scores brutos (Q·K^T / √dk):\\n{np.round(scores, 2)}")
""")

md("""
### 3.5. Softmax --- normalizar a pesos de atención

Cada fila suma 1: la palabra i reparte su atención entre las N palabras.
""")

code("""
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

attn = softmax(scores, axis=-1)
print(f"Atención (cada fila suma 1):")
import pandas as pd
df_attn = pd.DataFrame(np.round(attn, 2), index=frase, columns=frase)
df_attn.columns.name = "...presta atención a..."; df_attn.index.name = "cada palabra..."
display(df_attn)
""")

md("""
### 3.6. Interpretabilidad --- visualizar la atención
""")

code("""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(attn, cmap="Reds", vmin=0, vmax=attn.max())
ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xticklabels(frase, rotation=30, ha="right")
ax.set_yticklabels(frase)
for i in range(N):
    for j in range(N):
        if attn[i,j] > 0.15:
            ax.text(j, i, f"{attn[i,j]:.2f}", ha="center", va="center",
                    color="white" if attn[i,j]>0.4 else "black", fontsize=9)
ax.set_title("Atención (matrices Q, K, V aleatorias)")
plt.tight_layout(); plt.show()
""")

md("""
> **Importante**: como `Wq, Wk, Wv` son aleatorias, esta atención no significa nada.
> En un modelo real (BETO, GPT) **estas matrices se aprenden** sobre miles de millones de palabras
> hasta que la atención se vuelve significativa --- como veremos ya con BETO en la siguiente sección.

### 3.7. La salida --- mezclar los V según la atención
""")

code("""
output = attn @ V   # (N, DK)
print(f"Salida de la atención (cada palabra = mezcla ponderada de los V):\\n{output}")
print(f"\\n✓ Esto es **exactamente** lo que hace una capa de atención.")
print(f"En la realidad: D=768 (BETO), 12 capas, 12 cabezas por capa, millones de palabras de entrenamiento.")
""")

md("""
### 3.8. Positional encoding --- meterle el orden de vuelta

Como todo se procesa en paralelo, el modelo **no sabe el orden**. Solución del paper:
sumar un patrón senoidal único a cada posición.
""")

code("""
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angles = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe

PE = positional_encoding(50, 64)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
im = axes[0].imshow(PE.T, cmap="RdBu_r", aspect="auto")
axes[0].set_title("Positional encoding (50 posiciones, 64 dimensiones)")
axes[0].set_xlabel("posición en la secuencia"); axes[0].set_ylabel("dimensión")
plt.colorbar(im, ax=axes[0])

# Mostrar 4 dimensiones específicas
for d, c in zip([0, 4, 16, 50], ["red", "blue", "green", "purple"]):
    axes[1].plot(PE[:, d], label=f"dim {d}", color=c)
axes[1].set_title("4 dimensiones específicas (sinusoides a distintas frecuencias)")
axes[1].set_xlabel("posición"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

md("""
Cada dimensión es una sinusoide a distinta frecuencia. Combinadas, dan una **firma única**
para cada posición que el modelo puede aprender a decodificar.

---
""")

# =============================================================================
#  4. BLOQUE 3a -- MINI-TRANSFORMER EN KERAS
# =============================================================================
md("""
## 4. Bloque 3a --- Mini-Transformer en Keras (spoiler: no aprende)

Ya que vimos atención desde cero, construyamos uno **completo en Keras** y veamos qué pasa
si lo entrenamos sobre las 2.620 reseñas de muchocine.

> **Spoiler honesto**: nuestro mini-Transformer no va a superar al TF-IDF (84%).
> Va a quedar en ~50-55% (azar). **Esa es la lección**: un Transformer desde cero necesita
> datos masivos. Y nos motiva el siguiente paso: usar uno PRE-ENTRENADO (BETO).
""")

code("""
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 8000
MAXLEN = 120
DIM = 32
HEADS = 2

# Tokenizar
Xtr_t, Xte_t, ytr2, yte2 = train_test_split(df["review"], df["label"], test_size=0.2,
                                             random_state=42, stratify=df["label"])
tok = Tokenizer(num_words=VOCAB_SIZE, lower=True, oov_token="<oov>")
tok.fit_on_texts(Xtr_t)
Xtr_seq = pad_sequences(tok.texts_to_sequences(Xtr_t), maxlen=MAXLEN, padding="post", truncating="post")
Xte_seq = pad_sequences(tok.texts_to_sequences(Xte_t), maxlen=MAXLEN, padding="post", truncating="post")
print(f"Vocab efectivo: {min(len(tok.word_index)+1, VOCAB_SIZE)}, MAXLEN={MAXLEN}")
""")

code("""
def build_mini_transformer():
    inp = layers.Input((MAXLEN,))
    x = layers.Embedding(VOCAB_SIZE, DIM)(inp)
    # Positional (sumar)
    pe = positional_encoding(MAXLEN, DIM).astype("float32")
    x = x + pe
    # 1 bloque de atención
    a = layers.MultiHeadAttention(num_heads=HEADS, key_dim=16)(x, x)
    x = layers.LayerNormalization()(x + a)
    f = layers.Dense(64, activation="relu")(x)
    f = layers.Dense(DIM)(f)
    x = layers.LayerNormalization()(x + f)
    # Clasificar
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)

with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    mini = build_mini_transformer()
    mini.compile("adam", "binary_crossentropy", metrics=["accuracy"])
mini.summary()
""")

code("""
import time
t0 = time.time()
with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    hist_m = mini.fit(Xtr_seq, ytr2, validation_split=0.15, epochs=8, batch_size=32, verbose=2)
print(f"\\n✓ {time.time()-t0:.1f}s")

acc_mini = (mini.predict(Xte_seq, verbose=0).ravel() > 0.5).astype(int)
acc_mini_v = accuracy_score(yte2, acc_mini)
print(f"\\nMini-Transformer acc = {acc_mini_v:.3f}")
print(f"TF-IDF baseline acc = {acc:.3f}")
print(f"\\n{'✓ Lo supera' if acc_mini_v > acc else '✗ No lo supera (era esperado)'}")
""")

md("""
**¿Por qué no aprendió?**
- ~250.000 parámetros vs ~2.000 ejemplos = ratio brutal sobre-parametrizado.
- La atención inicia aleatoria, necesita muchísimos datos para encontrar patrones reales.
- **El paradigma correcto no es entrenar desde cero. Es PARTIR de uno pre-entrenado.**

Y eso son **BETO, BERT, GPT y todos los LLMs**.

---
""")

# =============================================================================
#  5. BLOQUE 3b -- BETO EN PROFUNDIDAD
# =============================================================================
md("""
## 5. Bloque 3b --- BETO en profundidad

**BETO** = **B**ERT trained on **E**spanish text by **T**he Universidad de Chile **O**rganization.

| Característica | Valor |
|---|---|
| Arquitectura | BERT-base (encoder Transformer) |
| Capas | 12 |
| Cabezas por capa | 12 → **144 atenciones distintas** mirando cada frase |
| Dimensión | 768 |
| Parámetros | ~110 millones |
| Pre-entrenado en | Wikipedia ES + libros + noticias (~3 mil millones de palabras) |
| Tareas en pre-entrenamiento | Masked Language Modeling + Next Sentence Prediction |

Vamos a usarlo. **Sin entrenar nada**, ya entiende español.
""")

code("""
from transformers import AutoTokenizer, AutoModel
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
print(f"Cargando {MODEL_NAME} (~440MB primera vez, después cacheado)...")
tok_b = AutoTokenizer.from_pretrained(MODEL_NAME)
mdl_b = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True, attn_implementation="eager")
mdl_b.eval().to(device)
print(f"✓ BETO cargado: {sum(p.numel() for p in mdl_b.parameters()):,} parámetros")
""")

md("""
### 5.1. Interpretabilidad #1 --- la atención sobre la negación

Le pasamos *"no me gustó nada la trama"* y vemos a dónde mira cada palabra
en la **mejor cabeza** (la que más fuerte conecta `gustó → no`).
""")

code("""
FRASE = "no me gustó nada la trama"
inp = tok_b(FRASE, return_tensors="pt").to(device)
tokens = tok_b.convert_ids_to_tokens(inp["input_ids"][0])
with torch.no_grad():
    out_b = mdl_b(**inp)
# attentions: tupla de 12 capas, cada una (batch, heads, q, k)
attentions = torch.stack(out_b.attentions, dim=0).squeeze(1).cpu().numpy()  # (12, 12, N, N)
print(f"Atenciones shape: {attentions.shape}")
print(f"Tokens BETO: {tokens}")
""")

code("""
# Buscar la mejor cabeza para "gustó → no"
def idx_of(toks, target):
    for i, t in enumerate(toks):
        if t.lower().replace("##", "") == target.lower(): return i
    return None

i_gusto = idx_of(tokens, "gusto") or idx_of(tokens, "gustó")
i_no = idx_of(tokens, "no")
scores = np.array([[attentions[l, h, i_gusto, i_no] for h in range(12)] for l in range(12)])
best_l, best_h = np.unravel_index(scores.argmax(), scores.shape)
print(f"Mejor (capa, cabeza): ({best_l}, {best_h}), atención 'gustó'→'no' = {scores[best_l, best_h]:.2%}")

fig, ax = plt.subplots(figsize=(6, 5))
A = attentions[best_l, best_h]
ax.imshow(A, cmap="Reds", vmin=0, vmax=A.max())
ax.set_xticks(range(len(tokens))); ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=30, ha="right"); ax.set_yticklabels(tokens)
for i in range(len(tokens)):
    for j in range(len(tokens)):
        if A[i,j] > 0.2:
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                    color="white" if A[i,j]>0.45 else "black", fontsize=8)
ax.set_title(f"BETO capa {best_l} cabeza {best_h}: \\\"gustó\\\" presta atención a \\\"no\\\"")
plt.tight_layout(); plt.show()
""")

md("""
### 5.2. Interpretabilidad #2 --- ¿qué hace cada CAPA?

Diferentes capas aprenden patrones distintos. Vamos a comparar
capa **temprana** (sintaxis), **media** (semántica), **tardía** (tarea).
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
LAYERS = [1, 4, 7, 11]
for ax, l in zip(axes, LAYERS):
    A = attentions[l, best_h]
    ax.imshow(A, cmap="Reds", vmin=0, vmax=A.max())
    ax.set_xticks(range(len(tokens))); ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tokens, fontsize=7)
    ax.set_title(f"capa {l}")
fig.suptitle(f"BETO --- la atención cambia entre capas (cabeza {best_h})", y=1.02)
plt.tight_layout(); plt.show()
""")

md("""
**Patrón típico** (descubierto por la comunidad):
- **Capas 0-3**: atención local --- cada palabra mira a sus vecinas inmediatas (sintaxis).
- **Capas 4-8**: relaciones semánticas --- sujetos con verbos, negaciones con adjetivos.
- **Capas 9-11**: enfocadas en la tarea (en BERT puro: tokens `[CLS]` y `[SEP]`).

### 5.3. App 1 de BETO --- Masked Language Modeling (rellenar huecos)

BETO se pre-entrenó adivinando palabras **enmascaradas**. Eso lo hace muy bueno
para autocompletar. Lo usamos en frases industriales de Arca.
""")

code("""
from transformers import pipeline
fill = pipeline("fill-mask", model=MODEL_NAME, device=0 if device=="cuda" else -1)

FRASES_MASK = [
    "el [MASK] de la línea 2 está dañado.",
    "la presión del compresor está [MASK] de lo normal.",
    "el operario reportó un [MASK] en la llenadora.",
    "necesitamos repuestos para el [MASK] del motor.",
    "la queja del cliente dice que el producto [MASK] mal.",
]
for f in FRASES_MASK:
    print(f"\\n  > {f}")
    for r in fill(f, top_k=5):
        print(f"      {r['score']:.2%}  {r['token_str']}")
""")

md("""
**Observa**: BETO no fue entrenado en datos de Arca. Aprende posibilidades del español general.
Para un dominio específico (mantenimiento industrial), un **fine-tuning** sobre 5-10k frases
de tus tickets le enseñaría vocabulario propio: ``llenadora``, ``etiquetadora``, ``carrusel``.

### 5.4. App 2 de BETO --- embeddings de oraciones (búsqueda semántica)

BETO produce un **vector por oración**. Oraciones de significado parecido → vectores cercanos.
Esto permite buscar por significado, no por palabras exactas.
""")

code("""
from sklearn.metrics.pairwise import cosine_similarity

def embed_sentence(texto):
    inp = tok_b(texto, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        out = mdl_b(**inp)
    # Mean pooling sobre los tokens
    mask = inp["attention_mask"].unsqueeze(-1).float()
    summed = (out.last_hidden_state * mask).sum(1)
    counts = mask.sum(1)
    return (summed / counts).cpu().numpy()[0]

# Catálogo de tickets ficticios
TICKETS = [
    "el compresor número 3 está echando vapor desde anoche",
    "la presión del sistema bajó por debajo de lo normal",
    "la llenadora se atascó con una botella",
    "no funciona el aire acondicionado del depósito",
    "falla en el sensor de temperatura del horno",
    "la etiquetadora pega mal las etiquetas",
    "queja de cliente: la bebida sabe extraño",
    "el cliente recibió el producto vencido",
]
EMBS = np.array([embed_sentence(t) for t in TICKETS])
print(f"Embeddings: {EMBS.shape} (dim={EMBS.shape[1]} por oración)")

QUERY = "problema con la temperatura"
q_emb = embed_sentence(QUERY)
sims = cosine_similarity([q_emb], EMBS)[0]
order = np.argsort(-sims)
print(f"\\nBúsqueda semántica para: \\\"{QUERY}\\\"")
print(f"  (¡fíjate que la query NO contiene la palabra 'sensor' ni 'horno'!)\\n")
for i in order[:5]:
    print(f"  {sims[i]:.3f}  {TICKETS[i]}")
""")

md("""
**Esto es la base de RAG**: con tu catálogo de documentos embebido, una pregunta del usuario
se convierte también en vector y traes los documentos más cercanos. **Módulo 6** lo desarrolla
con detalle (FAISS, ChromaDB, sentence-transformers).

### 5.5. App 3 de BETO --- clustering de tickets

Con los mismos embeddings, podemos **agrupar tickets parecidos** sin etiquetas.
""")

code("""
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(EMBS)
proj = PCA(n_components=2).fit_transform(EMBS)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#C82B40", "#2563EB", "#16A34A"]
for c in range(3):
    mask = km.labels_ == c
    ax.scatter(proj[mask, 0], proj[mask, 1], s=200, c=colors[c],
               label=f"cluster {c}", alpha=0.7, edgecolors="black")
for i, t in enumerate(TICKETS):
    ax.annotate(t[:35]+"...", (proj[i, 0], proj[i, 1]),
                fontsize=8, xytext=(5, 5), textcoords="offset points")
ax.set_title("BETO agrupa tickets por significado (KMeans sobre embeddings)")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

print("\\nTickets por cluster:")
for c in range(3):
    print(f"\\n--- Cluster {c} ---")
    for i, t in enumerate(TICKETS):
        if km.labels_[i] == c: print(f"  • {t}")
""")

md("""
### 5.6. App 4 de BETO --- comparación de quejas (detección de duplicados)
""")

code("""
QUEJAS = [
    "la botella vino abierta y derramada",
    "recibí mi pedido con la tapa rota y todo mojado",   # casi duplicado de la anterior
    "el sabor del refresco es muy raro, no es normal",
    "la gaseosa tiene un sabor extraño, parece vencida",  # casi duplicado
    "demoró 5 días en llegar, muy lento",
]
qE = np.array([embed_sentence(q) for q in QUEJAS])
sim_matrix = cosine_similarity(qE)

fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(sim_matrix, cmap="Greens", vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(QUEJAS))); ax.set_yticks(range(len(QUEJAS)))
ax.set_xticklabels([f"Q{i+1}" for i in range(len(QUEJAS))])
ax.set_yticklabels([f"Q{i+1}: {q[:35]}..." for i, q in enumerate(QUEJAS)])
for i in range(len(QUEJAS)):
    for j in range(len(QUEJAS)):
        ax.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center",
                color="white" if sim_matrix[i,j]>0.85 else "black", fontsize=9)
ax.set_title("Similitud entre quejas (Q1↔Q2 y Q3↔Q4 deberían ser altas)")
plt.tight_layout(); plt.show()
""")

md("""
### 5.7. App 5 de BETO --- "verdadero" sentiment con sentence-transformers fine-tuned

Aunque BETO base no clasifica, hay variantes **fine-tuned para sentimiento**.
Por ejemplo, `pysentimiento/robertuito-sentiment-analysis`.
""")

code("""
from transformers import pipeline
sent_pipe = pipeline("text-classification",
                     model="pysentimiento/robertuito-sentiment-analysis",
                     device=0 if device=="cuda" else -1)

print("Evaluando las 8 frases-trampa con un Transformer FINE-TUNED:\\n")
preds_beto = []
for (f, real), tf_pred in zip(TRAMPAS, preds):
    out = sent_pipe(f)[0]
    # robertuito devuelve {label: POS/NEU/NEG, score}
    label_map = {"POS": 1, "NEG": 0, "NEU": 1}
    bp = label_map[out["label"]]
    preds_beto.append(bp)
    ok = "✓" if bp == real else "✗"
    ok_tf = "✓" if tf_pred == real else "✗"
    print(f"{ok}  Transformer={LABEL[bp]:<8} | {ok_tf} TF-IDF={LABEL[tf_pred]:<8} | {f}")
n_b = sum(1 for r, p in zip(reales, preds_beto) if r == p)
n_t = sum(1 for r, p in zip(reales, preds) if r == p)
print(f"\\nAciertos: Transformer pre-entrenado {n_b}/8 vs TF-IDF {n_t}/8")
""")

md("""
**Conclusión Bloque 3b**: con un modelo PRE-ENTRENADO en miles de millones de palabras,
**no necesitas entrenar nada** para tareas comunes. Cargas el modelo, le hablas,
te responde con calidad alta.

---
""")

# =============================================================================
#  6. BLOQUE 4 -- LLMs CON GROQ
# =============================================================================
md("""
## 6. Bloque 4 --- LLMs con Groq

Llegamos al **último escalón**: un Transformer escalado al máximo, entrenado por miles de
millones de USD, accesible vía API en milisegundos.

### 6.1. Setup --- API key gratis

1. Crea cuenta en https://console.groq.com (sin tarjeta de crédito).
2. Genera una API key (botón "API Keys").
3. La pegas cuando te la pida la siguiente celda (con `getpass`).

**Importante**: Groq es **gratis** hasta 30 requests/minuto y 1.000 requests/día
con Llama 3.1 8B. Más que suficiente para esta clase.
""")

code("""
import getpass, os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = getpass.getpass("Pega tu GROQ_API_KEY (no se imprime): ")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from openai import OpenAI
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
MODEL = "llama-3.1-8b-instant"

# Prueba de conexión
test = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Di hola en español, en una palabra."}],
    max_tokens=20,
)
print(f"✓ Conexión OK. Respuesta: {test.choices[0].message.content}")
""")

md("""
### 6.2. Helper para llamar al LLM
""")

code("""
def llm(prompt, system=None, model=MODEL, max_tokens=400, temperature=0.3):
    msgs = []
    if system: msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    r = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature)
    return r.choices[0].message.content, r.usage
""")

md("""
### 6.3. App 1 --- Generación de respuestas a quejas de cliente
""")

code("""
QUEJAS_REALES = [
    "Compré dos cajas de refresco y una llegó con la mitad de las botellas con la tapa rota.",
    "Me prometieron entrega el martes y todavía no llega. Es viernes. Necesito respuesta.",
    "La bebida sabe rara, parece vencida. Pero la fecha dice marzo 2027. ¿Qué pasa?",
]
sistema = ("Eres un agente de servicio al cliente de Arca Continental Ecuador, "
           "una embotelladora. Respondes con empatía, asumes responsabilidad, "
           "propones una solución concreta. Máximo 3 oraciones.")
for q in QUEJAS_REALES:
    resp, _ = llm(q, system=sistema, max_tokens=200)
    print(f"\\n--- QUEJA ---\\n{q}")
    print(f"\\n--- RESPUESTA SUGERIDA ---\\n{resp}")
""")

md("""
### 6.4. App 2 --- Descripciones de producto (marketing)
""")

code("""
prod = "Refresco de cola sin azúcar, lata de 330ml, edición limitada con sabor a cereza."
desc, _ = llm(f"Escribe una descripción de marketing de 50 palabras para: {prod}. "
              "Tono: juvenil, energético, con un call-to-action.",
              max_tokens=120, temperature=0.7)
print(desc)
""")

md("""
### 6.5. App 3 --- Instrucciones de mantenimiento paso a paso
""")

code("""
prompt = ("Genera un instructivo de 5 pasos para revisar y limpiar el filtro de un "
          "compresor industrial de planta embotelladora. Formato: lista numerada con "
          "verbo en imperativo, máximo 15 palabras por paso.")
ins, _ = llm(prompt, max_tokens=300, temperature=0.2)
print(ins)
""")

md("""
### 6.6. App 4 --- Chat multi-turno con memoria

Un LLM no recuerda nada entre llamadas. Para "conversación", **vos** mandás el historial completo.
""")

code("""
def chat(history, user_msg, model=MODEL, max_tokens=300):
    history.append({"role": "user", "content": user_msg})
    r = client.chat.completions.create(
        model=model, messages=history, max_tokens=max_tokens, temperature=0.4)
    reply = r.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply

history = [{"role": "system", "content": "Eres un asistente de mantenimiento de Arca. Eres conciso."}]
for msg in [
    "Mi llenadora está vibrando más de lo normal. ¿Qué hago primero?",
    "Ya verifiqué eso y los pernos están firmes. ¿Qué sigue?",
    "El motor está caliente, sí. ¿Qué hago?",
]:
    print(f"\\nUSUARIO: {msg}")
    reply = chat(history, msg)
    print(f"BOT: {reply}")
print(f"\\n(historial: {len(history)} mensajes guardados)")
""")

md("""
### 6.7. App 5 --- Clasificación zero-shot (sin etiquetas)
""")

code('''
def clasificar_zeroshot(texto, categorias):
    cats = ", ".join(categorias)
    prompt = (f"Clasifica el siguiente texto en UNA SOLA de estas categorías: {cats}.\\n"
              f"Texto: \\"{texto}\\"\\n"
              f"Responde SÓLO con la categoría exacta, sin explicación.")
    r, _ = llm(prompt, max_tokens=20, temperature=0.0)
    return r.strip()

CATS = ["mantenimiento", "queja_cliente", "logistica", "calidad", "ventas"]
TEXTS = [
    "El compresor número 3 vibra raro desde anoche",
    "El cliente reclamó que la botella vino derramada",
    "El camión no llegó al cliente de Guayaquil",
    "Lote 4521 tiene un sabor distinto, pedimos retención",
    "Cerramos el contrato con la cadena Tia",
]
print(f"{'Texto':<55} {'Categoría'}")
print("-"*80)
for t in TEXTS:
    c = clasificar_zeroshot(t, CATS)
    print(f"{t[:53]:<55} → {c}")
''')

md("""
**Nota**: con TF-IDF o BETO necesitarías miles de ejemplos etiquetados para entrenar
un clasificador. Con un LLM, lo haces con un prompt de 2 líneas. **Esa es la magia.**

### 6.8. App 6 --- Extracción de información a JSON
""")

code('''
import json as jsn

TICKETS_RAW = [
    "El compresor número 3 está echando vapor desde anoche, urge revisión antes de la mañana.",
    "La llenadora de la línea 2 paró por sensor de tapa, repuesto pedido a Cali.",
    "Equipo: enfriadora 9. Síntoma: ruido metálico. Urgencia: alta. Reportado por: Juan Pérez.",
]
INSTR = ('Extrae los datos del ticket en JSON con las llaves EXACTAS:\\n'
         '{"equipo": str, "problema": str, "urgencia": "alta"|"media"|"baja", '
         '"reportado_por": str|null}.\\n'
         'Sólo el JSON, sin explicación.\\n\\nTicket: "{t}"')

for t in TICKETS_RAW:
    out, _ = llm(INSTR.format(t=t), max_tokens=200, temperature=0.0)
    print(f"\\n>>> {t}")
    try:
        data = jsn.loads(out.strip().strip("`").replace("json\\n", ""))
        print(jsn.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"(no parseó JSON: {out[:200]})")
''')

md("""
### 6.9. App 7 --- Resumen de un documento largo
""")

code("""
REPORTE = '''Reporte de turno --- línea 2, jueves 30 de mayo 2026, turno noche (22:00-06:00).
Inició turno con calibración de la llenadora 2A. Producción objetivo: 18.000 botellas. Producción real: 16.450
(91% del objetivo). Razones del desfase: (1) parada de 35 minutos a las 23:40 por fallo del sensor de tapa
en la posición 7, repuesto disponible en bodega, cambio rápido. (2) parada de 22 minutos a las 02:15 por
atasco en el carrusel de etiquetado; se desatascó manualmente, no requirió mantenimiento. Indicadores de
calidad: rechazo de botellas en 0,3% (esperado <0,5%, OK). Temperatura del aceite del compresor 5: 78°C
(normal). Notas: el operario de relevo Juan Sánchez reportó que el filtro de aire del compresor 5 está sucio,
sugiere cambiar en próximo mantenimiento programado (jueves 6 de junio). Próximas acciones: cambio de filtro
compresor 5, revisar histórico de fallos sensor tapa posición 7 (segundo evento este mes).'''
prompt = f"Resume este reporte de turno en exactamente 3 viñetas, máximo 15 palabras cada una:\\n\\n{REPORTE}"
res, _ = llm(prompt, max_tokens=250, temperature=0.2)
print(res)
""")

md("""
### 6.10. App 8 --- Q&A sobre un texto (RAG manual, sin embeddings)

Le das contexto y le haces preguntas. Es el corazón de RAG (recuperar contexto relevante,
darlo al LLM, preguntar).
""")

code("""
preguntas = [
    "¿Cuántas botellas se produjeron y cuál era el objetivo?",
    "¿Qué pasó a las 23:40?",
    "¿Cuándo es el próximo mantenimiento programado?",
    "¿Quién reportó el filtro sucio?",
]
for p in preguntas:
    prompt = f"Responde usando SÓLO el siguiente contexto. Si no está, di 'no aparece'.\\n\\nContexto:\\n{REPORTE}\\n\\nPregunta: {p}"
    r, _ = llm(prompt, max_tokens=120, temperature=0.0)
    print(f"\\nP: {p}\\nR: {r}")
""")

md("""
### 6.11. Interpretabilidad #1 --- temperatura del LLM

Misma tarea, distinta temperatura. Vemos cómo varía la creatividad.
""")

code("""
prompt = "Inventa un nombre creativo para un nuevo refresco de mandarina con jengibre. SÓLO el nombre."
print("Misma pregunta, 5 corridas, distintas temperaturas:\\n")
for T in [0.0, 0.3, 0.7, 1.2]:
    print(f"\\nT={T}:")
    for _ in range(3):
        r, _ = llm(prompt, max_tokens=20, temperature=T)
        print(f"  → {r.strip()}")
""")

md("""
**Patrón**: T=0 → siempre lo mismo (determinista). T alto → más variedad y riesgo de gibberish.

### 6.12. Interpretabilidad #2 --- modelos chicos vs grandes

Misma tarea, distintos modelos. Veamos la diferencia.
""")

code("""
MODELOS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
TAREA = ("Explica en 2 oraciones la diferencia entre un encoder Transformer (como BERT) "
         "y un decoder Transformer (como GPT).")

for m in MODELOS:
    try:
        r, usage = llm(TAREA, model=m, max_tokens=200, temperature=0.2)
        print(f"\\n=== {m} ({usage.total_tokens} tokens) ===")
        print(r)
    except Exception as e:
        print(f"\\n{m}: no disponible ({type(e).__name__})")
""")

md("""
### 6.13. Interpretabilidad #3 --- streaming token por token

Cuando hablas con ChatGPT y ves la respuesta aparecer letra por letra, eso es **streaming**:
el modelo te manda cada token en cuanto lo genera (sin esperar a terminar toda la respuesta).
""")

code("""
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Cuéntame un chiste corto sobre programadores."}],
    max_tokens=80, stream=True, temperature=0.7,
)
print("Generando token por token:\\n")
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print("\\n\\n✓ Eso es lo que ves cuando hablas con ChatGPT.")
""")

md("""
### 6.14. Comparativa final --- las 4 técnicas en una tabla
""")

code("""
import pandas as pd
tabla = pd.DataFrame([
    ["TF-IDF + LogReg",       "rapidísimo",  "minutos",  "84%",      "$0",          "alta"],
    ["Mini-Trans (cero)",     "lento",       "30s GPU",  "~50%",     "$0 (local)",  "media"],
    ["BETO (pre-entrenado)",  "moderado",    "0 (uso)",  "fine-tune",  "$0 (local)", "alta"],
    ["LLM (Groq Llama 8B)",   "milisegundos","0 (uso)",  "zero-shot",  "$0.05/Mtok", "baja"],
], columns=["Técnica", "Inferencia", "Entrenamiento", "Acc/calidad", "Costo", "Interpretabilidad"])
display(tabla)
""")

md("""
---

## 7. Cierre --- la respuesta a la pregunta del día

> **¿Cómo le enseñamos a una máquina a entender el lenguaje de Arca?**

**No le enseñamos.** *Elegimos el modelo correcto y le hablamos en nuestro dominio.*
El oficio nuevo es **elegir y orquestar**, no entrenar.

### Cómo llegamos:

1. Antes de 2017 las LSTM tenían **3 muros**: secuencial, olvido, no escalan.
2. *"Attention is All You Need"* (2017) propuso el **Transformer**: cada palabra mira a todas.
3. Entrenar uno desde cero requiere datos masivos (lo vimos: nuestro mini no aprendió).
4. Un **pre-entrenado** (BETO) ya entiende español: **gustó → no** al 60% sin instrucción.
5. Un **LLM** es eso a otra escala: GPT predice el siguiente token, igual que nuestro char-LSTM.
6. En **5 líneas de Python** hablamos a un Llama 3.1 vía Groq, **gratis**.

### Lo que te llevas para Arca el lunes

- Para **autocompletar tickets repetitivos** → LSTM chico (CPU).
- Para **clasificar quejas en español on-prem** → BETO + fine-tune con 2.000 ejemplos.
- Para **clasificar/extraer/responder sin entrenar** → LLM vía API (Groq, free para empezar).
- **Regla de costo**: tareas masivas → Llama-8B en Groq ($0.05/M tok); tareas críticas → Claude Sonnet ($15/M tok).
- **Nunca** entrenes un Transformer desde cero con < 100k ejemplos.

---

## 8. Ejercicios sugeridos (para casa)

1. **Char-LSTM**: cambia el corpus a tus propios manuales de mantenimiento. ¿Qué genera?
2. **Atención NumPy**: cambia el `Wq/Wk/Wv` de aleatorios a entrenados (gradient descent simple).
3. **BETO embeddings**: arma tu propio motor de búsqueda semántica sobre 100 tickets reales de Arca.
4. **BETO fine-tuning**: con la libreta de `transformers Trainer`, fine-tunea BETO sobre 1.000 quejas etiquetadas. ¿Cuánto sube el accuracy?
5. **LLM prompting**: diseña un prompt que extraiga `equipo, fecha, severidad, accionable` de tickets libres. Mide acc.
6. **LLM con RAG**: indexa los reportes de turno del último mes con BETO + FAISS, y haz Q&A con Groq.

---

## 9. Módulo 6 --- lo que viene

| Tema | Por qué importa |
|---|---|
| Prompting avanzado | Few-shot, chain-of-thought, role prompting |
| RAG | Conectar LLMs a tus documentos (sin re-entrenar) |
| Agentes | LLMs que ejecutan acciones (consultar APIs, escribir DB) |
| Web scraping | Extraer datos de la web para alimentar todo lo anterior |
| APIs en producción | Rate limits, fallback, costos, observabilidad |

Hoy entendiste el motor. En Módulo 6 lo pones a trabajar para Arca Continental.

---
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
