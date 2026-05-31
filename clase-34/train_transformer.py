"""
E2 - Mini-Transformer Keras sobre muchocine_sentimiento.csv.
Reporta accuracy global, evalúa 8 frases-trampa de negación, extrae atención real.
Outputs: transformer_results.json, attention_matrix.npy, frases_trampa.json,
         transformer_history.json
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
tf.keras.utils.set_random_seed(42)
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))
assert tf.config.list_physical_devices('GPU'), "ABORT: sin GPU disponible"

# ============================================================
#  1) Datos
# ============================================================
df = pd.read_csv(os.path.join(ROOT, "muchocine_sentimiento.csv"))
df["review"] = df["review"].astype(str)
textos = df["review"].tolist()
labels = df["label"].astype(int).values
print(f"Corpus: {len(textos)} reseñas | labels: {dict(zip(*np.unique(labels, return_counts=True)))}")

Xtr_t, Xte_t, ytr, yte = train_test_split(textos, labels, test_size=0.2, random_state=42, stratify=labels)

# ============================================================
#  2) Baseline TF-IDF (replica de clase-33 para anclar)
# ============================================================
print("\n--- Baseline TF-IDF ---")
vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", min_df=3,
                     ngram_range=(1, 2), max_features=30000)
Xtr_tfidf = vec.fit_transform(Xtr_t)
Xte_tfidf = vec.transform(Xte_t)
clf_tfidf = LogisticRegression(max_iter=1000, C=3.0).fit(Xtr_tfidf, ytr)
acc_tfidf = accuracy_score(yte, clf_tfidf.predict(Xte_tfidf))
print(f"TF-IDF + LogReg acc = {acc_tfidf:.4f}")

# ============================================================
#  3) Tokenizar para Keras
# ============================================================
VOCAB_SIZE = 8000
MAXLEN = 200  # las reseñas son largas; capta el contexto donde está la negación
tok = Tokenizer(num_words=VOCAB_SIZE, lower=True, oov_token="<oov>")
tok.fit_on_texts(Xtr_t)
Xtr_seq = pad_sequences(tok.texts_to_sequences(Xtr_t), maxlen=MAXLEN, padding="post", truncating="post")
Xte_seq = pad_sequences(tok.texts_to_sequences(Xte_t), maxlen=MAXLEN, padding="post", truncating="post")
print(f"Tokenizer: vocab efectivo={min(len(tok.word_index)+1, VOCAB_SIZE)}, MAXLEN={MAXLEN}")

# ============================================================
#  4) Mini-Transformer (Encoder) - mas chico para 2.6k samples
# ============================================================
DIM = 32
HEADS = 2
KDIM = 16
FF = 64
DROPOUT = 0.2
N_BLOCKS = 1  # un solo bloque para no sobreparametrizar

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angles = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.constant(pe[None, :, :], dtype=tf.float32)

class TransformerBlock(layers.Layer):
    def __init__(self, dim, heads, kdim, ff_dim, dropout=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=heads, key_dim=kdim, dropout=dropout)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(dim),
        ])
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
    def call(self, x, training=False, return_attention=False):
        n = self.ln1(x)
        if return_attention:
            a, scores = self.mha(n, n, training=training, return_attention_scores=True)
        else:
            a = self.mha(n, n, training=training)
            scores = None
        x = x + self.drop1(a, training=training)
        n2 = self.ln2(x)
        f = self.ffn(n2)
        x = x + self.drop2(f, training=training)
        return (x, scores) if return_attention else x

PE = positional_encoding(MAXLEN, DIM)
emb_layer = layers.Embedding(VOCAB_SIZE, DIM)
block1 = TransformerBlock(DIM, HEADS, KDIM, FF, DROPOUT)
block2 = TransformerBlock(DIM, HEADS, KDIM, FF, DROPOUT)

def build_train_model():
    inp = layers.Input((MAXLEN,))
    x = emb_layer(inp) + PE
    x = block1(x)
    x = block2(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)

with tf.device('/GPU:0'):
    model = build_train_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
print(f"Mini-Transformer params: {model.count_params():,}")

def extract_attention(seq_batch):
    """Pasa por las capas entrenadas y extrae los attention scores del block1."""
    with tf.device('/GPU:0'):
        x = emb_layer(seq_batch) + PE
        n = block1.ln1(x)
        _, scores = block1.mha(n, n, return_attention_scores=True)
    return scores.numpy()

# ============================================================
#  5) Entrenamiento
# ============================================================
EPOCHS = 10
t0 = time.time()
with tf.device('/GPU:0'):
    hist = model.fit(Xtr_seq, ytr, validation_split=0.15, epochs=EPOCHS, batch_size=32, verbose=2)
print(f"Entrenamiento: {time.time()-t0:.1f}s")

pred = (model.predict(Xte_seq, verbose=0).ravel() > 0.5).astype(int)
acc_trans = accuracy_score(yte, pred)
print(f"\n=== ACCURACY GLOBAL ===")
print(f"TF-IDF + LogReg:        {acc_tfidf:.4f}")
print(f"Mini-Transformer:       {acc_trans:.4f}")

# ============================================================
#  6) Frases-trampa de negación
# ============================================================
trampa = [
    ("no me gustó nada la trama, qué decepción", 0),
    ("no es para nada mala, me sorprendió", 1),
    ("esperaba mucho más, no la recomiendo", 0),
    ("no podría estar más contento con esta película", 1),
    ("ni una sola escena aburrida, brillante", 1),
    ("no entiendo cómo a alguien le puede gustar esto", 0),
    ("no me arrepiento de haberla visto", 1),
    ("buena fotografía pero no, no funciona", 0),
]
frases = [t[0] for t in trampa]
yt = np.array([t[1] for t in trampa])

# TF-IDF
tfidf_pred = clf_tfidf.predict(vec.transform(frases))
# Transformer
seqs = pad_sequences(tok.texts_to_sequences(frases), maxlen=MAXLEN, padding="post", truncating="post")
trans_pred = (model.predict(seqs, verbose=0).ravel() > 0.5).astype(int)

print("\n=== FRASES-TRAMPA ===")
print(f"{'real':<6} {'tfidf':<6} {'trans':<6} frase")
trampa_results = []
for (frase, y_real), p_tf, p_tr in zip(trampa, tfidf_pred, trans_pred):
    print(f"{y_real:<6} {p_tf:<6} {p_tr:<6} {frase}")
    trampa_results.append({
        "frase": frase, "real": int(y_real),
        "tfidf": int(p_tf), "transformer": int(p_tr),
        "tfidf_correct": bool(p_tf == y_real),
        "transformer_correct": bool(p_tr == y_real),
    })
tfidf_correct = sum(1 for r in trampa_results if r["tfidf_correct"])
trans_correct = sum(1 for r in trampa_results if r["transformer_correct"])
print(f"\nAciertos trampa: TF-IDF {tfidf_correct}/8  Transformer {trans_correct}/8")

# ============================================================
#  7) Atención REAL sobre "no me gustó nada la trama"
# ============================================================
FRASE_VIZ = "no me gustó nada la trama"
viz_seq = pad_sequences(tok.texts_to_sequences([FRASE_VIZ]), maxlen=MAXLEN,
                       padding="post", truncating="post")
attn = extract_attention(viz_seq)
print(f"\nAttention shape (batch, heads, q, k): {attn.shape}")
# Cortar a la longitud real de la frase tokenizada
toks_real = [tok.index_word.get(i, "?") for i in viz_seq[0] if i != 0]
N = len(toks_real)
attn_real = attn[0, :, :N, :N]
print(f"Tokens reales ({N}): {toks_real}")

# ============================================================
#  8) Guardar artefactos
# ============================================================
np.save(os.path.join(ROOT, "attention_matrix.npy"), attn_real)
with open(os.path.join(ROOT, "attention_tokens.json"), "w") as f:
    json.dump({"tokens": toks_real, "frase": FRASE_VIZ}, f, ensure_ascii=False, indent=2)
with open(os.path.join(ROOT, "frases_trampa.json"), "w") as f:
    json.dump(trampa_results, f, ensure_ascii=False, indent=2)
with open(os.path.join(ROOT, "transformer_results.json"), "w") as f:
    json.dump({
        "acc_tfidf": float(acc_tfidf),
        "acc_transformer": float(acc_trans),
        "params_transformer": int(model.count_params()),
        "vocab_size": VOCAB_SIZE,
        "maxlen": MAXLEN,
        "dim": DIM, "heads": HEADS, "ff": FF, "dropout": DROPOUT,
        "epochs": EPOCHS,
        "trampa_tfidf_correct": tfidf_correct,
        "trampa_transformer_correct": trans_correct,
    }, f, indent=2)
with open(os.path.join(ROOT, "transformer_history.json"), "w") as f:
    json.dump({k: [float(x) for x in v] for k, v in hist.history.items()}, f, indent=2)
print("\nArtefactos guardados: attention_matrix.npy, attention_tokens.json,")
print("                       frases_trampa.json, transformer_results.json,")
print("                       transformer_history.json")
