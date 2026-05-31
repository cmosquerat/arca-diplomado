"""
E1 - Char-LSTM sobre Don Quijote. Genera samples en 3 temperaturas.
Outputs: corpus_quijote.txt, samples_charlstm.txt, charlstm_history.json
"""
import os, re, json, time, warnings, urllib.request
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(ROOT, "corpus_quijote.txt")
SAMPLES_PATH = os.path.join(ROOT, "samples_charlstm.txt")
HISTORY_PATH = os.path.join(ROOT, "charlstm_history.json")
np.random.seed(42)

# ============================================================
#  1) Corpus
# ============================================================
def download_quijote():
    url = "https://www.gutenberg.org/files/2000/2000-0.txt"
    print(f"Descargando {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read().decode("utf-8", errors="ignore")

def clean_gutenberg(text):
    # Recortar header/footer Gutenberg
    start = re.search(r"\*\*\* START OF.*?\*\*\*", text, flags=re.S | re.I)
    end   = re.search(r"\*\*\* END OF.*?\*\*\*", text, flags=re.S | re.I)
    if start: text = text[start.end():]
    if end:   text = text[:end.start()]
    # lowercase + colapsar espacios + quitar saltos múltiples
    text = text.lower()
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Mantener letras, dígitos, acentos, ñ, signos básicos
    text = re.sub(r"[^a-z0-9áéíóúñü¿¡!?,.;:'\"\-\n() ]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

if os.path.isfile(CORPUS_PATH):
    print(f"Corpus cacheado en {CORPUS_PATH}")
    text = open(CORPUS_PATH).read()
else:
    raw = download_quijote()
    text = clean_gutenberg(raw)
    with open(CORPUS_PATH, "w") as f: f.write(text)
    print(f"Corpus guardado en {CORPUS_PATH} ({len(text)} chars)")

print(f"Corpus: {len(text)} chars")
chars = sorted(set(text)); V = len(chars)
print(f"Vocabulario: {V} caracteres únicos")
ci = {c: i for i, c in enumerate(chars)}
ic = {i: c for c, i in ci.items()}
data = np.array([ci[c] for c in text], dtype=np.int32)

# ============================================================
#  2) Modelo
# ============================================================
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input
tf.keras.utils.set_random_seed(42)
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))
assert tf.config.list_physical_devices('GPU'), "ABORT: sin GPU disponible"

SEQ = 60
STEP = 10  # reducir dataset para no abusar del cooling
print(f"Construyendo ventanas (SEQ={SEQ}, STEP={STEP})...")
X = np.array([data[i:i+SEQ] for i in range(0, len(data)-SEQ-1, STEP)], dtype=np.int32)
Y = np.array([data[i+SEQ]   for i in range(0, len(data)-SEQ-1, STEP)], dtype=np.int32)
print(f"Ejemplos: {len(X):,}")

with tf.device('/GPU:0'):
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
print(model.summary())

# ============================================================
#  3) Entrenamiento (forzado en GPU)
# ============================================================
EPOCHS = 10
t0 = time.time()
with tf.device('/GPU:0'):
    hist = model.fit(X, Y, epochs=EPOCHS, batch_size=128, verbose=2)
print(f"Entrenamiento: {time.time()-t0:.1f}s")

# ============================================================
#  4) Generación con temperatura
# ============================================================
def sample(seed: str, n: int = 300, temperature: float = 0.5) -> str:
    s = [ci.get(c, 0) for c in seed.lower()][-SEQ:]
    s = [0] * (SEQ - len(s)) + s
    out = seed
    with tf.device('/GPU:0'):
        for _ in range(n):
            logits = model.predict(np.array([s]), verbose=0)[0] / max(temperature, 1e-6)
            p = np.exp(logits - logits.max()); p /= p.sum()
            nx = int(np.random.choice(V, p=p))
            out += ic[nx]
            s = s[1:] + [nx]
    return out

SEED = "en un lugar de la mancha de cuyo nombre no quiero acordarme "
TEMPS = [0.4, 0.8, 1.2]
print("\n=== MUESTRAS ===")
samples_out = []
for T in TEMPS:
    s = sample(SEED, n=320, temperature=T)
    print(f"\n--- temp {T} ---\n{s}")
    samples_out.append((T, s))

# ============================================================
#  5) Persistir artefactos
# ============================================================
with open(SAMPLES_PATH, "w") as f:
    for T, s in samples_out:
        f.write(f"=== temperatura {T} ===\n{s}\n\n")
print(f"\nSamples guardados en {SAMPLES_PATH}")

with open(HISTORY_PATH, "w") as f:
    json.dump({
        "corpus_chars": len(text),
        "vocab": V,
        "seq": SEQ,
        "step": STEP,
        "epochs": EPOCHS,
        "loss": [float(x) for x in hist.history["loss"]],
        "seed": SEED,
        "temps": TEMPS,
    }, f, indent=2)
print(f"Historia guardada en {HISTORY_PATH}")
