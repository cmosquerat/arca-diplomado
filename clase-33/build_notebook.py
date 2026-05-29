"""
Construye Clase_33_NLP.ipynb --- Introduccion a NLP: de representar texto a entender el contexto.
Compatible con Colab (lee CSVs desde GitHub, instala deps si hace falta).

HILO: "¿Como le ensenamos a una maquina a entender el SIGNIFICADO del texto?"
4 formas de representar texto, cada una captura algo que la anterior no podia:
  0. El problema
  1. De texto a numeros
  2. Contar palabras (BoW) + App1 sentimiento (+ limite: el orden)
  3. Pesar palabras (Zipf, stopwords, TF-IDF) + App2 spam (+ limite: fraseo nuevo)
  4. Embeddings (Word2Vec) + busqueda coseno
  5. Atencion / Transformers (significado contextual) + limite
  6. Cierre
"""
import json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Clase_33_NLP.ipynb")
REPO = "https://raw.githubusercontent.com/cmosquerat/arca-diplomado/main/clase-33"

cells = []
def md(t):
    L = t.strip("\n").split("\n")
    cells.append({"cell_type":"markdown","metadata":{},"source":[(s+"\n") for s in L[:-1]]+[L[-1]]})
def code(t):
    s = t.strip("\n").split("\n")
    cells.append({"cell_type":"code","metadata":{},"outputs":[],"execution_count":None,
                  "source":[(x+"\n") for x in s[:-1]]+[s[-1]]})

# ===== PORTADA =====
md("""
# Clase 33 --- Introduccion a NLP
### De representar texto a entender el contexto

**Diplomado en Data Science Aplicada con Python para la Toma de Decisiones**
Arca Continental Ecuador | UDLA

---

**Pregunta que ancla toda la clase:** *¿como convertimos texto en numeros que capturen su
SIGNIFICADO, para poder clasificarlo y buscarlo?*

Recorreremos **4 formas de representar texto**, cada vez mas ricas. Cada una captura algo que la
anterior no podia:

1. **Contar palabras** (Bag-of-Words) --- funciona, pero "no me gusto" ≈ "me gusto".
2. **Pesar por rareza** (TF-IDF) --- mejor, pero las palabras siguen sin relacion entre si.
3. **Embeddings** --- cada palabra es un vector; el significado es geometria (+ busqueda coseno).
4. **Atencion / Transformers** --- el significado depende del contexto. La base de los LLMs.
""")

# ===== 0. SETUP =====
md("## 0. Setup --- corre esta celda primero")
code(f"""
import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    import os
    os.system("pip install -q wordcloud gensim spacy sentence-transformers")
    os.system("python -m spacy download es_core_news_md")   # vectores pre-entrenados (caso final)
    print("Dependencias listas.")
    print("IMPORTANTE: activa la GPU (la atencion y el embedding contextual la aprovechan):")
    print("  Entorno de ejecucion > Cambiar tipo de entorno > T4 GPU")

CSV_SENT = "{REPO}/muchocine_sentimiento.csv"   # resenas de cine (positivo/negativo)
CSV_SPAM = "{REPO}/spam_es.csv"                  # SMS spam/ham en espanol
CSV_EMO  = "{REPO}/emociones_es.csv"             # mensajes etiquetados por emocion (6 clases)
print("\\nDatos:\\n ", CSV_SENT, "\\n ", CSV_SPAM, "\\n ", CSV_EMO)
""")
code("""
import warnings; warnings.filterwarnings("ignore")
import re, unicodedata
from collections import Counter
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize":(11,4.5),"axes.grid":True,"grid.alpha":0.3,
                     "grid.linestyle":"--","axes.spines.top":False,"axes.spines.right":False,
                     "axes.titleweight":"bold"})
ARCA_RED,ARCA_DARK,ARCA_BLUE,ARCA_GREEN,ARCA_ORANGE,ARCA_PURPLE = (
    "#C82B40","#6B1525","#2563EB","#16A34A","#EA580C","#7C3AED")
np.random.seed(42)
print("OK")
""")

# ===== 0bis. EL PROBLEMA =====
md("""
## El problema --- ponte en el lugar de la maquina

Antes de nada, intenta clasificar estos mensajes (tu cerebro lo hace en un instante):

| Mensaje | Pregunta |
|---|---|
| "La fotografia es preciosa, pero la trama me aburrio" | ¿positivo o negativo? |
| "No me gusto nada, esperaba mucho mas" | ¿positivo o negativo? |
| "Su cuenta sera suspendida. Verifique sus datos aqui" | ¿spam o legitimo? |
| "Oye, ¿llegas a la cena? confirma porfa" | ¿spam o legitimo? |

Para ti es trivial. Pero la maquina **solo ve una secuencia de caracteres**. No sabe que
"aburrio" es malo, ni que "no" invierte una frase, ni que un mensaje urgente sobre tu cuenta
huele a phishing.

**Toda la clase es construir representaciones que le den ese significado.** Empecemos.
""")

# ===== 1. TEXTO A NUMEROS =====
md("""
## 1. De texto a numeros

Recordemos de donde venimos: una **imagen** (clase 28-29) es una matriz de pixeles; una
**serie** (clase 32) es una secuencia de valores. Un **texto** tambien se vuelve numeros.

### El atomo: para la maquina, un caracter es un numero (Unicode)
""")
code("""
for c in "cafe": print(f"  '{c}' -> ord = {ord(c)}")
print(f"\\n  'ñ' -> {ord('ñ')}    espacio ' ' -> {ord(' ')}    '!' -> {ord('!')}")
print("\\nEl mismo 'cafe' puede medir distinto segun como se codifique el acento:")
print(" NFC:", len(unicodedata.normalize('NFC','café')), "caracteres")
print(" NFD:", len(unicodedata.normalize('NFD','café')), "caracteres  -> por eso se NORMALIZA")
""")
md("""
**Interpretacion:** la maquina nunca ve "letras": ve numeros. Y "café" puede medir 4 o 5 segun como
venga el acento --- por eso, antes de cualquier modelo, hay que **normalizar** el texto a una forma
comun. Si no, "café" y "café" (que se ven iguales) contarian como palabras distintas.
""")
md("### Tokenizar y normalizar")
code("""
def tokenizar(texto):
    texto = texto.lower()                                   # minusculas
    texto = "".join(c for c in unicodedata.normalize("NFD", texto)
                    if unicodedata.category(c) != "Mn")     # quita acentos
    return re.findall(r"[a-zñ]+", texto)                    # solo palabras

print(tokenizar("¡No me GUSTÓ nada la película!"))   # -> ['no','me','gusto','nada','la','pelicula']
""")
md("""
**Interpretacion:** en una sola funcion pasamos de una frase con mayusculas, acentos y signos a una
lista limpia de palabras comparables. "GUSTÓ" y "gusto" ahora son el **mismo** token --- ya podemos
empezar a contarlas.
""")

# ===== 2. BoW + SENTIMIENTO =====
md("""
## 2. Contar palabras (Bag-of-Words)

- **Corpus:** la coleccion de documentos. **Vocabulario:** las palabras distintas.
- **Bag-of-Words:** conservo *que* palabras y *cuantas*, **tiro el orden** (de ahi "bolsa").
  Cada documento -> un vector del tamano del vocabulario. El corpus -> una matriz documento x palabra.
""")
code("""
from sklearn.feature_extraction.text import CountVectorizer
corpus = ["gratis gana premio gratis", "hola la reunion es hoy"]
cv = CountVectorizer()
bow = cv.fit_transform(corpus)
print("Vocabulario:", list(cv.get_feature_names_out()))
print(pd.DataFrame(bow.toarray(), columns=cv.get_feature_names_out(), index=["D1","D2"]))
""")
md("""
**Para que sirve:** es el **puente de texto a numeros** --- convierte un documento en un vector
de largo fijo que cualquier clasificador (¡la Regresion Logistica del Modulo 4!) puede comer.
Propiedades: dispersa (casi todo ceros), alta dimension, y **ciega al orden**.

### App 1 --- sentimiento de resenas de cine
""")
code("""
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(CSV_SENT); df["review"] = df["review"].astype(str)
X = np.array(df["review"].astype(str).tolist())
y = np.array(df["label"].astype(int).tolist())
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", min_df=3,
                      ngram_range=(1,2), max_features=30000)
clf = LogisticRegression(max_iter=1000, C=3.0).fit(vec.fit_transform(Xtr), ytr)
pred = clf.predict(vec.transform(Xte))
acc_sent = accuracy_score(yte, pred)
print(f"Accuracy sentimiento: {acc_sent:.1%}\\n")
print(classification_report(yte, pred, target_names=["negativo","positivo"]))
""")
md("""
**Interpretacion:** ~84% de acierto en resenas que el modelo **no vio al entrenar**, con un modelo
que tarda segundos. El `classification_report` muestra precision y recall parecidos en ambas clases
(el dataset esta balanceado). Mensaje clave: una representacion clasica (TF-IDF) + un clasificador
lineal ya resuelve la mayor parte del problema. **Siempre empieza por aqui.**
""")
code("""
fn = np.array(vec.get_feature_names_out()); co = clf.coef_[0]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].barh(fn[np.argsort(co)[-12:]], co[np.argsort(co)[-12:]], color=ARCA_GREEN, alpha=0.85); axes[0].set_title("Empujan a POSITIVO")
axes[1].barh(fn[np.argsort(co)[:12]], co[np.argsort(co)[:12]], color=ARCA_RED, alpha=0.85); axes[1].set_title("Empujan a NEGATIVO")
for ax in axes: ax.axvline(0, color="black", lw=0.6)
plt.tight_layout(); plt.show()
""")
md("""
**Interpretacion:** la Regresion Logistica es **interpretable**: cada palabra tiene un peso. A la
izquierda, las que mas empujan a NEGATIVO ("pesima", "aburrida", "topica"...); a la derecha, las que
empujan a POSITIVO ("imprescindible", "obra maestra", "emocionante"...). Esto te deja **explicarle al
negocio** por que el modelo decide lo que decide --- algo que un modelo caja-negra no da.
""")
code("""
def sentimiento(texto):
    p = clf.predict_proba(vec.transform([texto]))[0, 1]
    return ("POSITIVO" if p >= 0.5 else "NEGATIVO"), p
for f in ["una obra maestra, me encanto", "un bodrio, perdi dos horas"]:
    et, p = sentimiento(f); print(f"  [{et}  p={p:.2f}]  {f}")
""")
md("""
**Interpretacion:** el modelo ya clasifica frases nuevas con una probabilidad. Pruebalo tu: cambia
los textos y mira como reacciona. (Intenta una negacion como "no es para nada mala" --- guarda lo
que pasa, lo retomamos enseguida.)

### Una grieta que conviene ver ya: el conteo no ve el orden

84% esta muy bien. Pero observa este caso (lo retomaremos al final de la clase):
""")
code("""
a, b = "me gusto la pelicula", "no me gusto la pelicula"
cv = CountVectorizer()
M = cv.fit_transform([a, b]).toarray()
print("vocabulario:", list(cv.get_feature_names_out()))
print(f'  "{a}"  (positivo) -> {M[0]}')
print(f'  "{b}"  (negativo) -> {M[1]}')
print("\\n  Los dos vectores difieren SOLO en la palabra 'no'. Significan lo OPUESTO,")
print("  pero para el conteo son casi el mismo punto: el orden y la negacion se pierden.")
""")

# ===== 3. ZIPF/TFIDF + SPAM =====
md("""
## 3. No todas las palabras valen igual (Zipf, stopwords, TF-IDF)

En el conteo crudo, las palabras que mas puntuan son "de", "la", "que"... porque son las que
mas se repiten. Pero esas no distinguen una resena buena de una mala: estan en todas. Necesitamos
**pesar** las palabras: mucho a las informativas, poco a las comunes. Para entender por que,
veamos un patron universal del lenguaje.

### La ley de Zipf
Si ordenas las palabras de la mas a la menos frecuente, la frecuencia es ~ inversa a su rango:
la 1ra aparece ~2x la 2da, ~3x la 3ra, ~10x la 10ma. Es un patron universal del lenguaje.
""")
code("""
todos = [w for t in df["review"] for w in tokenizar(t)]
frec = Counter(todos)
print(f"Vocabulario: {len(frec):,} palabras unicas | {len(todos):,} tokens")
print("Top 10:", [w for w,_ in frec.most_common(10)])
counts = np.array(sorted(frec.values(), reverse=True)); ranks = np.arange(1, len(counts)+1)
fig, ax = plt.subplots(figsize=(9,4))
ax.loglog(ranks, counts, color=ARCA_BLUE, lw=1.8, label="corpus real")
ax.loglog(ranks, counts[0]/ranks, color=ARCA_RED, ls="--", lw=1.4, label="Zipf ideal (1/rango)")
ax.set_xlabel("rango (log)"); ax.set_ylabel("frecuencia (log)"); ax.legend(); plt.tight_layout(); plt.show()
""")
md("""
**Para que sirve Zipf:** (1) justifica las **stopwords** (las del tope estan en todos los docs,
no discriminan); (2) es la razon de **TF-IDF** (pesar por rareza corrige el sesgo); (3) explica
la matriz dispersa; (4) predice las palabras nuevas (OOV).

**Stopwords --- pero quitarlas no es dogma:** en "no me gusto", "no" es stopword pero invierte
el sentido. Dependen del idioma y del dominio.

**TF-IDF** pesa una palabra alto si es frecuente en el documento pero rara en el corpus.

### App 2 --- detectar spam con el MISMO toolkit
""")
code("""
spam = pd.read_csv(CSV_SPAM); spam["texto"] = spam["texto"].astype(str)
print(spam["label_txt"].value_counts().to_dict())
Xs = np.array(spam["texto"].astype(str).tolist()); ys = np.array(spam["label"].astype(int).tolist())
Xstr, Xste, ystr, yste = train_test_split(Xs, ys, test_size=0.25, random_state=42, stratify=ys)
vec_sp = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1,2), min_df=2)
clf_sp = LogisticRegression(max_iter=1000, C=5).fit(vec_sp.fit_transform(Xstr), ystr)
acc_spam = accuracy_score(yste, clf_sp.predict(vec_sp.transform(Xste)))
print(f"Accuracy spam (test limpio): {acc_spam:.1%}  -> el clasico GANA en spam limpio.")
""")
md("""
**Interpretacion:** el MISMO pipeline (train/test + TF-IDF + Regresion Logistica), cambiando solo el
corpus, acierta ~93% en SMS que no vio al entrenar. La leccion: las herramientas que aprendimos son
**generales** y ya resuelven problemas reales. El spam es, de hecho, el caso de libro de la
clasificacion de texto.

### El limite real: fraseo nuevo (lo veremos a fondo en un momento)""")
code("""
def es_spam(texto):
    p = clf_sp.predict_proba(vec_sp.transform([texto]))[0, 1]
    return ("SPAM" if p >= 0.5 else "HAM"), p
evasivos = [("Su paquete esta retenido en aduana, abone la tasa para liberarlo","spam"),
            ("Hemos bloqueado su tarjeta por seguridad, reactivela aqui","spam"),
            ("Oye llegas a la cena de manana? confirma porfa","ham")]
for m, real in evasivos:
    et, p = es_spam(m); flag = "OK" if et.lower()==real else "<-- SE EQUIVOCA"
    print(f"  [{et} p={p:.2f}] (real:{real}) {flag}\\n      {m}\\n")
""")
md("""
**Interpretacion:** con fraseo nuevo deja pasar phishing y marca como spam algo normal. ¿Por que?
Porque para BoW/TF-IDF **las palabras son simbolos sin relacion**: "bueno" y "genial" son columnas
independientes, no sabe que significan casi lo mismo. Si no vio la palabra, esta perdido.

**-> El verdadero salto: que las palabras tengan SIGNIFICADO. Eso son los embeddings.**
""")

# ===== 4. EMBEDDINGS =====
md("""
## 4. Embeddings: el significado como geometria

**Idea:** en vez de un indice arbitrario, a cada palabra le damos un **vector** de numeros.
Lo entrenamos para que palabras que aparecen en **contextos parecidos** queden **cerca** en el
espacio. Asi el significado se vuelve **geometria** --- la misma idea del espacio latente del
autoencoder (clase 29) y de PCA.

Entrenamos **Word2Vec** sobre nuestras propias resenas (sin descargar nada):
""")
code("""
from gensim.models import Word2Vec
frases = [tokenizar(t) for t in df["review"]]
w2v = Word2Vec(frases, vector_size=100, window=5, min_count=5, epochs=20, seed=42, sg=1, workers=2)
print(f"Vocabulario Word2Vec: {len(w2v.wv)} palabras, cada una un vector de 100 numeros.")
print("\\nVector de 'buena' (primeros 8 numeros):")
print(np.round(w2v.wv["buena"][:8], 3))
""")
md("""
**Interpretacion:** ya no hay un indice arbitrario por palabra: ahora cada una es un **vector** de
100 numeros (arriba se ven 8 de ellos para "buena"). Esos numeros no significan nada por separado;
lo que importa es la **posicion relativa** entre palabras. Veamoslo con los vecinos.
""")
code("""
# Vecinos por significado (similitud coseno entre vectores)
for w in ["buena", "aburrida", "excelente", "actores", "obra"]:
    if w in w2v.wv:
        print(f"  {w:10}-> " + ", ".join(f"{x}" for x,_ in w2v.wv.most_similar(w, topn=6)))
""")
md("""
Sin decirle nunca que significan, el modelo aprendio que "aburrida" se parece a
"tediosa/monotona/pretenciosa". **Eso es significado capturado como geometria.**
""")
code("""
# Mapa 2D de los embeddings (PCA): los sinonimos se agrupan
from sklearn.decomposition import PCA
pos = ["buena","excelente","gran","mejor","maravillosa","brillante","obra","estupenda","genial"]
neg = ["mala","aburrida","tediosa","lenta","pesima","horrible","peor","floja","monotona"]
base = [w for w,_ in frec.most_common(300) if w in w2v.wv and len(w) > 3][:100]
words = list(dict.fromkeys(base + [w for w in pos+neg if w in w2v.wv]))
P = PCA(n_components=2, random_state=42).fit_transform(np.array([w2v.wv[w] for w in words]))
fig, ax = plt.subplots(figsize=(11,6))
for (x,yy), w in zip(P, words):
    if w in pos: ax.scatter(x,yy,c=ARCA_GREEN,s=40); ax.text(x,yy+0.02,w,fontsize=9,color=ARCA_GREEN,ha="center")
    elif w in neg: ax.scatter(x,yy,c=ARCA_RED,s=40); ax.text(x,yy+0.02,w,fontsize=9,color=ARCA_RED,ha="center")
    else: ax.scatter(x,yy,c="#CBD5E1",s=10)
ax.set_title("Embeddings en 2D (PCA): los sinonimos se agrupan"); plt.tight_layout(); plt.show()
""")
md("""
**Interpretacion:** comprimimos los vectores de 100 a 2 dimensiones (PCA, como en clase de reduccion
de dimensionalidad) solo para poder verlos. Las palabras positivas (verde) tienden a un lado y las
negativas (rojo) a otro: el modelo coloco el **significado** como **geometria**, sin que nadie le
dijera que "buena" es positiva. (El 2D es una sombra del espacio real de 100D, asi que la separacion
no es perfecta.)
""")
md("""
### Busqueda por coseno --- buscar por SIGNIFICADO, no por palabras

La **similitud coseno** mide el angulo entre dos vectores (1 = misma direccion, 0 = sin relacion).
Si representamos cada documento como el **promedio** de los vectores de sus palabras, podemos
buscar comentarios parecidos *aunque usen otras palabras*. Es la base del buscador semantico y del
RAG (Modulo 6).
""")
code("""
def docvec(texto):
    vs = [w2v.wv[w] for w in tokenizar(texto) if w in w2v.wv]
    return np.mean(vs, axis=0) if vs else np.zeros(w2v.vector_size)

D = np.array([docvec(t) for t in df["review"]])
consulta = "una de las mejores peliculas que he visto, una obra maestra"
sims = cosine_similarity([docvec(consulta)], D)[0]
print(f'Consulta: "{consulta}"\\n')
for rank, i in enumerate(np.argsort(-sims)[:3], 1):
    print(f"  #{rank}  coseno={sims[i]:.2f}")
    print(f'     "{re.sub(chr(32)+"+"," ", df["review"].iloc[i])[:110]}..."\\n')
""")
md("""
**Interpretacion:** representamos cada resena como el **promedio** de los vectores de sus palabras y
buscamos las mas parecidas por coseno. Las que salen comparten el **sentido** de la consulta (resenas
muy positivas), aunque no usen exactamente las mismas palabras. Eso es **busqueda semantica** --- la
base de los buscadores modernos y del RAG (Modulo 6).

> **Ejercicio.** Cambia `consulta` por algo negativo ("una peli aburridisima y lenta") y mira
> que recupera. Esta buscando por significado, no por coincidencia de palabras.

### Lo que los embeddings todavia no resuelven: el contexto

Word2Vec le da a cada palabra **un solo** vector. Pero "banco" (de sentarse) y "banco" (de dinero)
comparten vector. Y si promediamos los vectores de "no me gusto", seguimos haciendo una bolsa:
**no capturamos que "no" modifica a "gusto".**

**-> Necesitamos que el significado dependa del CONTEXTO. Eso es la atencion.**
""")

# ===== 5. ATENCION =====
md("""
## 5. Atencion y Transformers

**Atencion:** para cada palabra, el modelo mira a **todas** las demas y aprende **cuanto pesa**
cada una para interpretarla. En "no me gusto nada", "gusto" aprende a mirar fuerte a "no".

### Por dentro: Query, Key, Value
- Cada palabra genera una **Query** ("¿quien me importa?"), y cada palabra ofrece una **Key**.
- El **score** = Query · Key (producto punto): si "gusto" pregunta y "no" responde con una key
  relevante, el score es alto -> mucho peso.
- Se combinan los **Value** ponderados: "gusto" incorpora informacion de "no".
- Query, Key y Value son proyecciones **aprendidas** del embedding. Todas las palabras lo hacen a
  la vez, **en paralelo** (por eso un Transformer escala mejor que una LSTM).

### *Attention Is All You Need* (2017)
El paper de Vaswani et al. introdujo el **Transformer** (embeddings + atencion apilada). Es uno de
los papers mas influyentes de la historia reciente: la base de ChatGPT, los traductores y casi todo
el NLP moderno.

### Un mini-modelo de atencion en Keras --- y visualizar a donde mira
> Nota: esto entrena una red. En **GPU** son segundos; en CPU ~1-2 min.
""")
code("""
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
tf.keras.utils.set_random_seed(42)

VOCAB, MAXLEN, DIM = 12000, 60, 32
tok = Tokenizer(num_words=VOCAB, oov_token="<OOV>"); tok.fit_on_texts(Xtr)
Xtr_s = pad_sequences(tok.texts_to_sequences(Xtr), maxlen=MAXLEN, padding="post", truncating="post")
Xte_s = pad_sequences(tok.texts_to_sequences(Xte), maxlen=MAXLEN, padding="post", truncating="post")

inp = layers.Input(shape=(MAXLEN,))
emb = layers.Embedding(VOCAB, DIM)(inp)
ctx, scores = layers.MultiHeadAttention(num_heads=1, key_dim=DIM)(emb, emb, return_attention_scores=True)
x = layers.GlobalAveragePooling1D()(ctx)
out = layers.Dense(1, activation="sigmoid")(layers.Dense(32, activation="relu")(x))
modelo = models.Model(inp, out)
visor  = models.Model(inp, scores)          # mismo modelo, devuelve la atencion
modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
modelo.summary()
""")
code("""
hist = modelo.fit(Xtr_s, ytr, validation_split=0.15, epochs=6, batch_size=32, verbose=1)
acc_att = modelo.evaluate(Xte_s, yte, verbose=0)[1]
print(f"\\nAtencion: {acc_att:.1%}  |  Regresion Logistica (baseline): {acc_sent:.1%}")
""")
md("""
**Interpretacion:** entrenamos una red con UNA capa de atencion en pocas lineas. Fijate que **no le
gana** al baseline TF-IDF: con solo 2.600 resenas, un Transformer desde cero no tiene datos
suficientes para lucirse. Su poder real aparece **pre-entrenado a escala**. Lo valioso aqui no es la
accuracy, sino poder **mirar a donde atiende** cada palabra.
""")
code("""
frase = "no me gusto nada la trama"
seq = pad_sequences(tok.texts_to_sequences([frase]), maxlen=MAXLEN, padding="post")
palabras = tokenizar(frase); n = len(palabras)
A = visor.predict(seq, verbose=0)[0, 0, :n, :n]
fig, ax = plt.subplots(figsize=(6.5, 5.5)); im = ax.imshow(A, cmap="Reds"); ax.grid(False)
ax.set_xticks(range(n)); ax.set_xticklabels(palabras, rotation=30, ha="right")
ax.set_yticks(range(n)); ax.set_yticklabels(palabras)
ax.set_xlabel("...mira a esta palabra"); ax.set_ylabel("cada palabra...")
ax.set_title("Pesos de atencion aprendidos"); plt.colorbar(im, fraction=0.046); plt.tight_layout(); plt.show()
print("Lee una fila: a donde mira esa palabra. Observa si 'gusto'/'nada' miran a 'no'.")
""")
md("""
**Interpretacion:** a diferencia del Bag-of-Words, el modelo **puede** relacionar "gusto" con "no".
Con 2.600 resenas no siempre sale limpio (ni le gana al baseline): el poder real del Transformer
aparece con **pre-entrenamiento a escala masiva**. Y eso nos lleva al limite.

### El limite --- y que viene
Los transformers de verdad tienen **miles de millones de parametros** (GPT-3: 175.000 millones).
**No caben en una GPU normal.** Por eso casi nadie los entrena desde cero: se usan **pre-entrenados**,
servidos en la nube o via API --- el corazon del **Modulo 6 (IA Generativa)**.
""")

# ===== 6. CASO APLICADO: EMOCIONES (CAPSTONE) =====
md("""
## 6. Caso aplicado --- detectar la emocion del cliente (multi-clase)

Cerramos con un caso de principio a fin sobre un problema NUEVO y mas dificil: en vez de 2 clases
(positivo/negativo), clasificamos **6 emociones** --- util para la **voz del cliente / call center**
(¿el mensaje transmite alegria, enojo, miedo...?). Haremos el recorrido completo:
**EDA --> TF-IDF + Regresion Logistica --> Word2Vec --> un embedding pre-entrenado mas avanzado.**

> El corpus son mensajes cortos en espanol etiquetados por emocion (alegria, tristeza, amor, enojo,
> miedo, sorpresa). La tecnica es identica para transcripciones de llamadas o chats de soporte.
""")
code("""
emo = pd.read_csv(CSV_EMO); emo["texto"] = emo["texto"].astype(str)
print(f"Mensajes: {len(emo)}")
print(emo["emocion"].value_counts())
emo[["texto","emocion"]].head(5)
""")
md("""
### EDA --- lo primero que salta: las clases estan DESBALANCEADAS

'alegria' y 'tristeza' son mayoria; 'sorpresa' es rara. Eso importa: un modelo puede sacar buena
*accuracy* prediciendo siempre la clase mayoritaria. Por eso miraremos tambien el **F1 macro**
(promedia el acierto de cada clase por igual) y la **matriz de confusion**.
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
emo["emocion"].value_counts().plot.barh(ax=ax[0], color=ARCA_BLUE, alpha=0.85)
ax[0].set_title("Mensajes por emocion (desbalance)"); ax[0].invert_yaxis()
emo["n_palabras"] = emo["texto"].apply(lambda t: len(tokenizar(t)))
ax[1].hist(emo["n_palabras"], bins=30, color=ARCA_GREEN, alpha=0.7)
ax[1].set_title("Longitud de los mensajes (palabras)"); ax[1].set_xlabel("palabras")
plt.tight_layout(); plt.show()
""")
md("""
**Interpretacion:** dos cosas saltan. (1) El **desbalance** es fuerte: 'alegria' y 'tristeza' son la
mayoria, 'sorpresa' es rara --- tendremos que mirar el F1 macro, no solo la accuracy. (2) Los mensajes
son **cortos** (pocas palabras): eso favorece a los metodos de palabras clave como TF-IDF.
""")
code("""
# Palabras de contenido mas frecuentes por emocion (descubrimiento)
STOP = set('''de la los las un una que se su lo le es son como mas pero ya no si al me mi te tu
esta este muy y o a en por para con sin sobre tan tambien ha han hay fue era porque cuando muy
siento sentir me he ha que del'''.split())
for em in ["alegria","tristeza","enojo","miedo"]:
    c = Counter(w for t in emo[emo.emocion==em]["texto"] for w in tokenizar(t)
                if w not in STOP and len(w) > 3)
    print(f"{em:10}:", [w for w,_ in c.most_common(8)])
""")
md("""
**Interpretacion:** ya antes de entrenar, las palabras de contenido delatan la emocion ('feliz',
'maravilloso' en alegria; 'deprimido', 'solo' en tristeza...). Como en el caso de sentimiento, el EDA
nos dice que el problema es **separable** con palabras clave.

### Paso 1 --- TF-IDF + Regresion Logistica (lo que ya sabemos)
""")
code("""
from sklearn.metrics import f1_score, ConfusionMatrixDisplay

Xe = np.array(emo["texto"].tolist()); ye = np.array(emo["label"].tolist())
Xetr, Xete, yetr, yete = train_test_split(Xe, ye, test_size=0.2, random_state=42, stratify=ye)
EMO = ["tristeza","alegria","amor","enojo","miedo","sorpresa"]   # label 0..5

vec_e = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1,2),
                        min_df=3, max_features=20000)
clf_e = LogisticRegression(max_iter=2000, C=5, class_weight="balanced")
clf_e.fit(vec_e.fit_transform(Xetr), yetr)
pred_e = clf_e.predict(vec_e.transform(Xete))

base = pd.Series(yetr).value_counts(normalize=True).iloc[0]
print(f"Baseline (predecir siempre la mayoria): {base:.1%}")
print(f"TF-IDF + LogReg:  accuracy = {accuracy_score(yete,pred_e):.1%}   F1 macro = {f1_score(yete,pred_e,average='macro'):.3f}")
""")
md("""
**Interpretacion:** con 6 clases el problema es mas dificil que pos/neg, pero TF-IDF + LogReg ya
supera por mucho al baseline de mayoria (~34%). Usamos `class_weight="balanced"` para que las clases
raras (sorpresa, amor) cuenten; por eso miramos el **F1 macro** ademas de la accuracy.
""")
code("""
fig, ax = plt.subplots(figsize=(6,5))
ConfusionMatrixDisplay.from_predictions(yete, pred_e, display_labels=EMO,
    xticks_rotation=45, cmap="Blues", ax=ax, colorbar=False)
ax.set_title("Emociones --- TF-IDF + LogReg"); plt.tight_layout(); plt.show()
""")
md("""
**Interpretacion:** la diagonal (aciertos) es fuerte, pero los errores son **reveladores**: el modelo
confunde emociones que de verdad se parecen --- 'amor' con 'alegria', 'miedo' con 'tristeza'. No son
errores tontos: son emociones vecinas. Las clases raras (sorpresa) son las mas dificiles por tener
pocos ejemplos.

### Paso 2 --- Word2Vec (el embedding basico, entrenado por nosotros)

Convertimos cada mensaje en el **promedio** de los vectores de sus palabras y clasificamos sobre eso.
""")
code("""
from gensim.models import Word2Vec
w2v_e = Word2Vec([tokenizar(t) for t in Xetr], vector_size=100, window=5,
                 min_count=3, epochs=20, sg=1, seed=42, workers=2)

def doc_vector(texto, modelo):
    vs = [modelo.wv[w] for w in tokenizar(texto) if w in modelo.wv]
    return np.mean(vs, axis=0) if vs else np.zeros(modelo.vector_size)

Wtr = np.array([doc_vector(t, w2v_e) for t in Xetr])
Wte = np.array([doc_vector(t, w2v_e) for t in Xete])
clf_w = LogisticRegression(max_iter=2000, C=5, class_weight="balanced").fit(Wtr, yetr)
pw = clf_w.predict(Wte)
print(f"Word2Vec (basico) + LogReg:  accuracy = {accuracy_score(yete,pw):.1%}   F1 macro = {f1_score(yete,pw,average='macro'):.3f}")
print("\\nOjo: promediar vectores de un Word2Vec entrenado solo en 6.000 mensajes cortos da una")
print("representacion POBRE del mensaje. Puede incluso quedar por debajo del baseline de mayoria.")
""")
md("""
### Paso 3 --- Un embedding mas avanzado: vectores PRE-ENTRENADOS (spaCy)

`es_core_news_md` trae vectores de palabra **pre-entrenados en un corpus enorme** (millones de
palabras), no solo nuestros 6.000 mensajes. Usamos el vector de cada mensaje (`doc.vector`,
el promedio de sus palabras) y clasificamos igual.
""")
code("""
import spacy
nlp = spacy.load("es_core_news_md", disable=["tagger","parser","ner","lemmatizer","attribute_ruler"])

Str = np.array([d.vector for d in nlp.pipe(Xetr.tolist(), batch_size=256)])
Ste = np.array([d.vector for d in nlp.pipe(Xete.tolist(), batch_size=256)])
clf_s = LogisticRegression(max_iter=2000, C=5, class_weight="balanced").fit(Str, yetr)
ps = clf_s.predict(Ste)
print(f"spaCy pre-entrenado + LogReg:  accuracy = {accuracy_score(yete,ps):.1%}   F1 macro = {f1_score(yete,ps,average='macro'):.3f}")
""")
code("""
# Lo que un embedding pre-entrenado SI hace muy bien: similitud por significado
for w in ["feliz", "triste", "enojado"]:
    tok_w = nlp(w)
    sims = []
    for palabra in ["alegre","contento","deprimido","furioso","molesto","asustado","amor","sorpresa"]:
        sims.append((palabra, round(tok_w.similarity(nlp(palabra)), 2)))
    sims.sort(key=lambda x: -x[1])
    print(f"{w:10} se parece a:", sims[:4])
""")
md("""
**Interpretacion:** spaCy (pre-entrenado a escala) clasifica mejor que nuestro Word2Vec, pero donde
de verdad luce es en la **similitud por significado**: "feliz" sale cerca de "alegre/contento". Aun
asi, para CLASIFICAR estos textos cortos, promediar vectores sigue por debajo de TF-IDF.

### Paso 4 --- El embedding CONTEXTUAL (un Transformer pre-entrenado)

Hasta aqui, todo embedding daba **un vector fijo por palabra** y luego promediabamos. Un
**Transformer** hace algo distinto: le da a cada palabra un vector que **depende de la frase
completa**, y devuelve directamente un vector del mensaje que captura el contexto.

Usamos `sentence-transformers` con un modelo multilingue. Se **descarga solo, sin ningun token ni
API key** (100% reproducible en Colab); aprovecha la **GPU** si esta activada.
""")
code("""
# pip install sentence-transformers  (ya instalado en el setup; en Colab usa GPU)
from sentence_transformers import SentenceTransformer
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "(activa GPU en Colab para que vuele)")

# Modelo multilingue (incluye espanol). Se descarga sin token. Lo reusamos al final.
# Alternativa mas ligera/rapida: "paraphrase-multilingual-MiniLM-L12-v2".
modelo_ctx = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=device)
Ctr = modelo_ctx.encode(Xetr.tolist(), batch_size=64, show_progress_bar=True)
Cte = modelo_ctx.encode(Xete.tolist(), batch_size=64, show_progress_bar=True)

clf_c = LogisticRegression(max_iter=3000, C=5, class_weight="balanced").fit(Ctr, yetr)
pc = clf_c.predict(Cte)
print(f"\\nEmbedding CONTEXTUAL + LogReg:  accuracy = {accuracy_score(yete,pc):.1%}   F1 macro = {f1_score(yete,pc,average='macro'):.3f}")
""")
md("""
### Interpretacion --- la escalera de los embeddings

Ordenando por accuracy en este problema (los numeros exactos salen arriba):

| Enfoque | Idea | Resultado |
|---|---|---|
| **Word2Vec (propio)** | promedio de vectores entrenados en 6k mensajes | el peor (~0.30) |
| **spaCy (pre-entrenado)** | promedio de vectores entrenados a escala | mejor (~0.42) |
| **Contextual (Transformer)** | vector del mensaje segun su contexto | mucho mejor (~0.62) |
| **TF-IDF + LogReg** | palabras clave pesadas por rareza | el mejor aqui (~0.68) |

**Lo que se ve (el avance):** cada embedding mejora sobre el anterior --- el **contexto SI
aporta**: contextual ($\\sim$0.62) $>$ pre-entrenado estatico ($\\sim$0.42) $>$ propio ($\\sim$0.30). Y el
contextual casi alcanza a TF-IDF (~0.68) **sin haberse entrenado para esta tarea** (lo usamos tal cual).

**La leccion honesta:** en textos cortos y muy ``de palabras clave'', TF-IDF sigue siendo durisimo
de batir --- ``mas avanzado'' no es automaticamente ``mejor'' para todo problema. **El embedding
contextual brilla** cuando el significado depende del contexto, en textos largos, y sobre todo
cuando se **ajusta (fine-tuning)** a la tarea: ahi supera a todo lo demas.

**Eso es exactamente la clase 34:** usar Transformers pre-entrenados y ajustarlos. Hoy ya viste que
se descargan sin token y corren en Colab.
""")

# ===== 7. BONUS: BUSQUEDA SEMANTICA DE DOCUMENTOS =====
md("""
## 7. Bonus --- buscar documentos por SIGNIFICADO (la antesala del Modulo 6)

La otra gran aplicacion de los embeddings (ademas de clasificar) es la **busqueda semantica**:
encontrar el documento que responde a una pregunta **aunque no comparta las palabras**. Es el
corazon de los buscadores modernos y del **RAG** (darle a un LLM los documentos relevantes).

Montamos una mini "base de conocimiento" tecnica de planta y la consultamos en lenguaje natural,
reusando el mismo Transformer (`modelo_ctx`) del paso anterior.
""")
code("""
# Mini base de conocimiento (fichas tecnicas / instructivos de planta)
docs = [
    "La llenadora debe limpiarse con solucion CIP a 80 grados durante 20 minutos tras cada turno.",
    "El compresor de aire opera a 7 bar; revisar el filtro de admision cada 500 horas.",
    "La caldera genera vapor a 10 bar para la pasteurizacion; presion maxima 12 bar.",
    "El CO2 se inyecta a 4 volumenes para la carbonatacion de la bebida.",
    "El control de calidad mide los grados Brix con un refractometro para verificar el azucar.",
    "La cinta transportadora se lubrica con jabon diluido para reducir la friccion.",
    "El agua de proceso se trata por osmosis inversa para eliminar sales y minerales.",
    "El paletizado robotico apila 60 cajas por tarima al final de la linea.",
]
# 1) Convertir cada documento en un vector (una sola vez)
emb_docs = modelo_ctx.encode(docs, normalize_embeddings=True)
print("base de conocimiento:", emb_docs.shape, "(documentos x dimensiones)")
""")
code("""
def buscar(consulta, k=2):
    q = modelo_ctx.encode(consulta, normalize_embeddings=True)
    sims = emb_docs @ q                      # coseno (vectores normalizados)
    orden = np.argsort(-sims)[:k]
    print(f'CONSULTA: "{consulta}"')
    for i in orden:
        print(f"   {sims[i]:.2f}  {docs[i]}")
    print()

buscar("como desinfecto la maquina que llena las botellas")
buscar("cada cuanto cambio el filtro del aire comprimido")
buscar("medir el azucar de la bebida")
""")
md("""
**Interpretacion:** ninguna consulta usa las palabras del documento ("desinfectar" no esta en el
texto, que dice "CIP/limpiar"; "azucar" lo resuelve con "Brix"). Un buscador por palabras (Ctrl+F,
TF-IDF) **fallaria**; el embedding encuentra el documento correcto porque compara **significados**.

Esto es exactamente lo que hace un sistema **RAG**: ante una pregunta, recupera los documentos mas
parecidos por embedding y se los pasa a un LLM para que responda. **Buscar manuales, fichas tecnicas,
politicas o tickets por significado** es de las aplicaciones mas utiles --- y es el puente directo al
**Modulo 6**.

> **Ejercicio.** Anade tus propios documentos a `docs` (o cambialos por temas de tu area) y hazle
> preguntas. Mira como recupera por sentido, no por coincidencia de palabras.
""")

# ===== 8. CIERRE =====
md("""
## 8. Cierre

| Tecnica | Representacion | Captura | Falla en |
|---|---|---|---|
| 1. Contar | Bag-of-Words | que palabras hay | el orden / la negacion |
| 1.5 Pesar | TF-IDF | cuales importan | el significado |
| 2. Significado | Embeddings (+coseno) | sinonimos cercanos | el contexto (vector fijo) |
| 3. Contexto | Atencion / Transformers | significado contextual | (necesita escala -> pre-entrenados) |

### Arbol de decision practico
```
¿Clasificar rapido e interpretar?        -> TF-IDF + Regresion Logistica
¿Buscar / comparar por significado?       -> Embeddings + similitud coseno
¿Entender contexto complejo / generar?    -> Transformers (pre-entrenados)
```

### Lo que te llevas hoy
1. NLP es, ante todo, **convertir texto en numeros** que capturan cada vez mas significado.
2. **BoW / TF-IDF**: simples, rapidos, interpretables. Resuelven sentimiento (~84%) y spam (~93%).
3. **Embeddings**: el significado como geometria; **busqueda coseno** = buscar por sentido.
4. **Atencion**: el significado depende del contexto; base de los **Transformers** y los LLMs.
5. **Empieza simple.** Sube de complejidad solo cuando lo simple ya no alcanza.

**Clase 34:** transformers y modelos pre-entrenados en accion (rampa al Modulo 6).

---
*Codigo + datos: github.com/cmosquerat/arca-diplomado/tree/main/clase-33*
""")

nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
with open(OUT,"w") as f: json.dump(nb, f, indent=1)
print(f"Notebook generado: {OUT} ({len(cells)} celdas)")
