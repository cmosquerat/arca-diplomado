"""
Construye Clase_25_Detection.ipynb desde cero.
Estructura:
  0. Header + imports
  1. EDA del dataset cats vs dogs
  2. Re-entrenar la CNN clasica (copia de clase 24 cells 32-36)
  3. Diagnostico: matriz de confusion + top-N errores
  4. VGG16 al rescate (TL)
  5. Tu propio dataset (Coca vs Pepsi, copia de clase 24 seccion 9)
  6. App 1: Sliding Window
  7. App 2: YOLO preentrenado
  8. Cierre
"""
import json
from pathlib import Path

cells = []

def _to_lines(text):
    """Convierte un bloque a lista de lineas con \\n al final salvo la ultima (formato ipynb)."""
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _to_lines(text)})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _to_lines(text)})

# ============================================================================
# 0. HEADER + IMPORTS
# ============================================================================
md("""# Clase 25: De clasificador a detector

**Objetivos de hoy:**
1. Diagnosticar **por qué** la CNN clasica de clase 24 se quedo en ~80% sobre cats vs dogs.
2. Resolver eso con **transfer learning** (VGG16) y ver el salto.
3. Aplicar el mismo flujo a un dataset propio (Coca vs Pepsi via Gradio).
4. Intentar **detectar** con esa CNN usando *sliding window* y ver el techo.
5. Conocer **YOLO**, probarlo preentrenado, y descubrir que no ve placas.

> Notebook pensado para Colab. Activa **Runtime → Change runtime type → T4 GPU** antes de empezar.""")

md("""## 0. Imports y entorno""")

code("""!pip install -q ultralytics gradio""")

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, warnings, urllib.request, io, os
from pathlib import Path
from PIL import Image
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

print(f"TensorFlow: {tf.__version__}")
print(f"GPU disponible: {len(tf.config.list_physical_devices('GPU'))} dispositivo(s)")

IMG_SIZE = 128       # tamano de trabajo: alto suficiente para que VGG16 luzca""")

# ============================================================================
# 1. EDA DEL DATASET
# ============================================================================
md("""---
## 1. EDA: ¿con qué estamos trabajando?

Antes de entrenar nada, conozcamos el dataset. Vamos a usar el clásico **`cats_vs_dogs`** de Microsoft Research / Kaggle (~25 000 imágenes a alta resolución, varios cientos de píxeles por lado).

Tomamos un subset balanceado de **4000 imágenes** y las redimensionamos a **128×128** para que entrenar siga siendo rápido pero VGG16 tenga detalle real con qué trabajar.

> **¿Por qué importa este paso?** En clase 24 entrenamos a ciegas y nos dio ~80%. Si no entendemos las características del dataset, no podemos explicar por qué falla ni por qué transfer learning ayuda.""")

code("""# Cargar cats_vs_dogs desde TensorFlow Datasets (la primera vez descarga ~700 MB)
ds_full, info = tfds.load("cats_vs_dogs", split="train",
                          as_supervised=True, with_info=True, shuffle_files=True)

class_names_cd = ["gato", "perro"]   # 0 = cat, 1 = dog en el dataset
print(f"Total disponible: {info.splits['train'].num_examples} imagenes")
print(f"Clases: {info.features['label'].names}")
print(f"Vamos a tomar 4000 balanceadas, redimensionar a {IMG_SIZE}x{IMG_SIZE}.")""")

code("""# Tomar 2000 por clase y redimensionar
def preparar(img, lbl):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.uint8)
    return img, lbl

ds_cats = ds_full.filter(lambda x, y: y == 0).take(2000).map(preparar)
ds_dogs = ds_full.filter(lambda x, y: y == 1).take(2000).map(preparar)

X_cd, y_cd = [], []
for img, lbl in ds_cats.concatenate(ds_dogs):
    X_cd.append(img.numpy()); y_cd.append(int(lbl.numpy()))
X_cd = np.array(X_cd); y_cd = np.array(y_cd)

# Mezclar para que train/test split sea aleatorio
perm = np.random.RandomState(42).permutation(len(X_cd))
X_cd, y_cd = X_cd[perm], y_cd[perm]

print(f"X_cd: {X_cd.shape}   (N, alto, ancho, canales)")
print(f"y_cd: {y_cd.shape}   rango: {y_cd.min()}..{y_cd.max()}")
print(f"Balance: {dict(zip(class_names_cd, np.bincount(y_cd)))}")""")

md("""**Observaciones rápidas:**
- 4000 imágenes balanceadas (50/50). Sin problema de clases desbalanceadas.
- Resolución **128×128** después del resize — alta. Suficiente para distinguir orejas, hocico, ojos, pelaje.
- Pixeles `uint8` en [0, 255]. Hay que normalizar antes de entrenar.""")

code("""# Visualizar muestras al azar de cada clase
np.random.seed(42)
fig, axes = plt.subplots(2, 8, figsize=(15, 4))
for k, clase in enumerate(class_names_cd):
    idxs = np.random.choice(np.where(y_cd == k)[0], 8, replace=False)
    for j, idx in enumerate(idxs):
        ax = axes[k, j]
        ax.imshow(X_cd[idx])
        ax.axis("off")
        if j == 0:
            ax.set_ylabel(clase, fontsize=12, fontweight="bold", rotation=0,
                          labelpad=30, va="center")
plt.suptitle(f"Muestras del dataset ({IMG_SIZE}x{IMG_SIZE} RGB)", fontweight="bold")
plt.tight_layout(); plt.show()""")

md("""**Lo que vemos:**
- Variación enorme de poses, razas, fondos, colores, iluminación.
- Animales en distintos planos (primer plano, cuerpo entero, varios animales por foto).
- Algunos casos son ambiguos hasta para humanos.

A esta resolución VGG16 sí tiene detalle real con qué trabajar — hocicos, ojos, texturas de pelo. Eso es lo que faltaba con CIFAR a 32×32.""")

code("""# Distribucion de brillo medio por clase (un proxy simple de variabilidad)
brillo = X_cd.mean(axis=(1, 2, 3))   # promedio sobre H, W, C

fig, ax = plt.subplots(figsize=(9, 3.5))
for k, clase in enumerate(class_names_cd):
    ax.hist(brillo[y_cd == k], bins=30, alpha=0.6, label=clase,
            color=["#C82B40", "#2563EB"][k])
ax.set_xlabel("Brillo medio (0=oscuro, 255=claro)")
ax.set_ylabel("Cantidad de imagenes")
ax.set_title("Distribucion de brillo por clase", fontweight="bold")
ax.legend()
plt.tight_layout(); plt.show()
print(f"Brillo gato:  media={brillo[y_cd==0].mean():.1f}, std={brillo[y_cd==0].std():.1f}")
print(f"Brillo perro: media={brillo[y_cd==1].mean():.1f}, std={brillo[y_cd==1].std():.1f}")""")

md("""**Lectura:** los histogramas se solapan casi totalmente. **El brillo no separa gatos de perros** — el modelo no puede usar un atajo simple. Va a tener que aprender estructura visual (forma de las orejas, hocico, cuerpo). Y eso es difícil a 32×32.""")

# ============================================================================
# 2. RE-ENTRENAR LA CNN CLASICA
# ============================================================================
md("""---
## 2. Re-entrenamos la CNN clásica de la clase pasada

Mismo código de clase 24: una CNN pequeña entrenada desde cero. La meta es **reproducir el ~80%** y luego analizar dónde falla.""")

code("""# Normalizar y splitear
X_cd_n = X_cd.astype("float32") / 255.0
X_tr, X_te, y_tr, y_te = train_test_split(
    X_cd_n, y_cd, test_size=0.2, random_state=42, stratify=y_cd)
print(f"Train: {X_tr.shape}, Test: {X_te.shape}")""")

code("""# Misma idea que en clase 24, ajustada al input mas grande (128x128).
# Agregamos un bloque conv extra porque la imagen tiene 4x mas pixeles.
cnn_cd = Sequential([
    layers.Conv2D(32, 3, activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax"),
], name="CNN_CatsDogs")

cnn_cd.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
cnn_cd.summary()""")

code("""# Early stopping: para cuando val_loss deja de mejorar y restaura los pesos
# de la mejor epoca vista.
es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

t0 = time.time()
history_cd = cnn_cd.fit(
    X_tr, y_tr,
    epochs=30,                           # ponemos un techo alto, EarlyStopping decide
    batch_size=32,
    validation_split=0.1,
    callbacks=[es],
    verbose=2,
)
print(f"\\nTiempo: {time.time()-t0:.1f}s  ({len(history_cd.history['loss'])} epochs)")
acc_cd = cnn_cd.evaluate(X_te, y_te, verbose=0)[1]
print(f"Accuracy en TEST: {acc_cd:.3f}  (en clase 22 con MLP llegamos a ~60%)")""")

# ============================================================================
# 3. DIAGNOSTICO DE ERRORES
# ============================================================================
md("""---
## 3. Diagnóstico: ¿dónde falla?

Llegamos a ~80%. Eso significa que **20 de cada 100 imágenes están mal clasificadas**. ¿Cuáles? ¿Por qué?

> Esto es lo que en clase 24 no hicimos. Y es justo lo que da la pista para mejorar.""")

code("""# Predicciones sobre TEST con sus probabilidades
probs_cd = cnn_cd.predict(X_te, verbose=0)
y_pred_cd = probs_cd.argmax(axis=1)
conf_cd = probs_cd.max(axis=1)

# Matriz de confusion
cm = confusion_matrix(y_te, y_pred_cd)
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Reds")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > cm.max()/2 else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(class_names_cd); ax.set_yticklabels(class_names_cd)
ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
ax.set_title(f"CNN clasica: confusion matrix (acc={acc_cd:.1%})", fontweight="bold")
plt.tight_layout(); plt.show()
print(classification_report(y_te, y_pred_cd, target_names=class_names_cd))""")

code("""# Top 16 errores: las imagenes que el modelo predijo mal con MAS confianza
# (osea, las que el modelo estaba SEGURO de que eran X y eran Y)
errores_idx = np.where(y_pred_cd != y_te)[0]
print(f"Total de errores: {len(errores_idx)} / {len(y_te)}")

# Ordenamos por confianza descendente
errores_ord = errores_idx[np.argsort(-conf_cd[errores_idx])]
peores = errores_ord[:16]

fig, axes = plt.subplots(2, 8, figsize=(15, 4.5))
for j, idx in enumerate(peores):
    ax = axes[j // 8, j % 8]
    ax.imshow(X_te[idx])
    real = class_names_cd[y_te[idx]]
    pred = class_names_cd[y_pred_cd[idx]]
    c = conf_cd[idx]
    ax.set_title(f"real:{real}\\npred:{pred} ({c:.0%})",
                 fontsize=8, color="#C82B40")
    ax.axis("off")
plt.suptitle("Top 16 errores: el modelo se equivoco con alta confianza",
             fontweight="bold", color="#C82B40")
plt.tight_layout(); plt.show()""")

md("""**Patrones comunes que vamos a ver en los errores (típicos de CIFAR cats vs dogs):**

| Patrón | Por qué falla |
|---|---|
| Animal **muy pequeño** en la foto | A 32×32 ya casi no se ve, el fondo domina |
| **Pose rara** (acostado, de espaldas, recortado) | La CNN aprendió el "perfil prototipo" |
| **Fondo confuso** (otro animal, persona, mucha textura) | Los filtros de la CNN se distraen |
| Cachorro / gatito | Proporciones distintas al animal adulto |
| **Color atípico** (gato negro, perro blanco peludo) | La CNN se apoyaba en color general |

**Diagnóstico estructural:** la CNN clásica tiene solo **3 bloques convolucionales y ~120K parámetros**. Aprende filtros muy básicos a partir de 4000 imágenes a 32×32. **No hay manera de que capture la riqueza visual de "gato" o "perro" con tan poca capacidad y tan poca data.**

¿Solución? **Usar una red que ya vio millones de imágenes.**""")

# Guardamos algunos ejemplos para comparar despues
code("""# Guardamos los indices de los peores errores para volver a verlos despues con VGG16
peores_idx_para_comparar = peores.copy()
print(f"Guardamos {len(peores_idx_para_comparar)} ejemplos para comparar despues con VGG16.")""")

# ============================================================================
# 4. VGG16 AL RESCATE
# ============================================================================
md("""---
## 4. VGG16 al rescate: transfer learning

VGG16 fue entrenado en **ImageNet (1.4M de imágenes, 1000 categorías)**. Sus filtros ya saben detectar bordes, texturas, formas, partes de animales. Le ponemos una cabeza nueva y la entrenamos solo con nuestras 3200 imágenes.

> **Detalle técnico:** VGG16 fue entrenado a 224×224 pero acepta cualquier resolución ≥ 32×32. Trabajar a **128×128** nos da detalle real (ojos, hocico, textura de pelo) y entrena rápido en GPU de Colab.""")

code("""from tensorflow.keras.applications import VGG16

base = VGG16(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False     # congelar
print(f"VGG16 cargado. Parametros totales: {base.count_params():,} (todos congelados)")""")

code("""modelo_vgg = Sequential([
    base,
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax"),
], name="VGG16_TransferLearning")

modelo_vgg.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

trainable = sum(np.prod(v.shape) for v in modelo_vgg.trainable_variables)
print(f"Parametros entrenables: {trainable:,} (vs {modelo_vgg.count_params():,} totales)")
print("Solo entrenamos la cabeza. El backbone VGG16 esta congelado.")""")

code("""es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

t0 = time.time()
history_vgg = modelo_vgg.fit(X_tr, y_tr,
                             epochs=15,                  # techo alto, EarlyStopping decide
                             batch_size=32,
                             validation_split=0.1,
                             callbacks=[es],
                             verbose=2)
print(f"\\nTiempo: {time.time()-t0:.1f}s  ({len(history_vgg.history['loss'])} epochs)")
acc_vgg = modelo_vgg.evaluate(X_te, y_te, verbose=0)[1]
print(f"Accuracy VGG16: {acc_vgg:.3f}  (vs CNN clasica: {acc_cd:.3f})")""")

code("""# Comparacion visual del salto
fig, ax = plt.subplots(figsize=(8, 3.5))
modelos = ["MLP\\n(clase 22)", "CNN clasica\\n(clase 24)", "VGG16 + TL\\n(hoy)"]
accs    = [0.60, acc_cd, acc_vgg]
colors  = ["#9CA3AF", "#EA580C", "#16A34A"]
bars = ax.bar(modelos, accs, color=colors)
for b, v in zip(bars, accs):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.0%}",
            ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.axhline(0.5, ls="--", color="gray", lw=1, label="azar")
ax.set_ylim(0.4, 1.0); ax.set_ylabel("Accuracy")
ax.set_title("Cats vs Dogs: el salto del transfer learning", fontweight="bold")
ax.legend()
plt.tight_layout(); plt.show()""")

md("""## 4.1 ¿Y los errores de antes? Revisemos las MISMAS imágenes con VGG16""")

code("""# Tomamos los peores errores de la CNN clasica y vemos como los clasifica VGG16
probs_vgg = modelo_vgg.predict(X_te[peores_idx_para_comparar], verbose=0)
pred_vgg = probs_vgg.argmax(axis=1)
conf_vgg = probs_vgg.max(axis=1)

fig, axes = plt.subplots(2, 8, figsize=(15, 5))
aciertos_recuperados = 0
for j, idx in enumerate(peores_idx_para_comparar):
    ax = axes[j // 8, j % 8]
    ax.imshow(X_te[idx])
    real = class_names_cd[y_te[idx]]
    pred = class_names_cd[pred_vgg[j]]
    correcto = pred_vgg[j] == y_te[idx]
    if correcto: aciertos_recuperados += 1
    color = "#16A34A" if correcto else "#C82B40"
    marca = "OK" if correcto else "X"
    ax.set_title(f"[{marca}] real:{real}\\nVGG:{pred} ({conf_vgg[j]:.0%})",
                 fontsize=8, color=color)
    ax.axis("off")
plt.suptitle(f"Las 16 imagenes que la CNN clasica fallo: VGG16 acierta {aciertos_recuperados}/16",
             fontweight="bold")
plt.tight_layout(); plt.show()
print(f"\\nDe los {len(peores_idx_para_comparar)} errores mas confiados de la CNN clasica,")
print(f"VGG16 recupera {aciertos_recuperados} ({aciertos_recuperados/len(peores_idx_para_comparar):.0%}).")""")

md("""**Lectura final del bloque:**

- VGG16 trae **138 millones de parámetros preentrenados**. Aunque congelados, sus filtros saben ver mucho mejor que los nuestros desde cero.
- La cabeza nueva (4-5 capas chicas) es lo único que entrenamos: rápido, poca data necesaria.
- El salto se explica por una sola idea: **representaciones visuales aprendidas en ImageNet son universales**. Funcionan para gatos, perros, latas, placas, casi cualquier cosa visual.

Ahora apliquemos esto a un dataset propio.""")

# ============================================================================
# 5. TU PROPIO DATASET (COCA VS PEPSI) — copia de clase 24 seccion 9
# ============================================================================
md("""---
## 5. Tu propio dataset: Coca-Cola vs Pepsi

Vamos a construir el flujo **completo** de un proyecto de visión real:
1. App de recolección — Gradio recibe imágenes que **ustedes descargan de Google** y las guarda etiquetadas.
2. Cargar y preprocesar.
3. Entrenar — transfer learning con VGG16 (igual que arriba).
4. App de predicción.

> **Cómo recolectar:** abran Google Images, busquen "coca cola lata" / "pepsi lata", descarguen ~30 imágenes por marca con **variedad** (distintos ángulos, fondos, iluminación). Después las suben todas juntas al recolector.""")

code("""import gradio as gr

DATA_DIR = "dataset_latas"
class_names = ["coca_cola", "pepsi"]

for clase in class_names:
    os.makedirs(f"{DATA_DIR}/{clase}", exist_ok=True)
print("Carpetas listas:")
for clase in class_names:
    print(f"  {DATA_DIR}/{clase}/")""")

md("""### 5.1 App de recolección con upload

Selecciona la clase y sube las imágenes que descargaste de Google. Puedes subir **muchas a la vez**. Mínimo ~30 por clase, variadas.""")

code("""def guardar_fotos(archivos, clase):
    if not archivos:
        return "Sube imagenes primero."
    folder = f"{DATA_DIR}/{clase}"
    saved = 0
    for f in archivos:
        try:
            img = Image.open(f.name).convert("RGB")
        except Exception as e:
            continue
        n = len(os.listdir(folder))
        fname = f"{folder}/img_{n:03d}.jpg"
        img.save(fname, "JPEG", quality=92)
        saved += 1
    counts = {c: len(os.listdir(f"{DATA_DIR}/{c}")) for c in class_names}
    return (f"Guardadas {saved} fotos en {folder}.\\n"
            f"Total acumulado: " + " | ".join(f"{c}: {counts[c]}" for c in class_names))

with gr.Blocks(title="Recolector de latas") as recolector:
    gr.Markdown("# Recolector: Coca-Cola vs Pepsi")
    gr.Markdown("**Flujo:** descarga imagenes de Google Images -> selecciona la clase -> subelas todas -> Guardar. Repite con la otra clase.")
    with gr.Row():
        archivos = gr.File(file_count="multiple", file_types=["image"],
                           label="Imagenes (puedes subir muchas a la vez)")
        with gr.Column():
            clase = gr.Radio(class_names, label="Clase", value="coca_cola")
            btn = gr.Button("Guardar imagenes", variant="primary", size="lg")
            output = gr.Textbox(label="Estado", lines=3)
    btn.click(guardar_fotos, inputs=[archivos, clase], outputs=output)

recolector.launch(share=True, debug=False)""")

md("""**Al buscar en Google**, recordar:
- Distintos **ángulos** (frontal, lateral, arriba, en mano)
- **Fondos distintos** (no solo el clásico anuncio publicitario)
- Variar **iluminación** y entornos (tienda, mesa, refrigerador)
- Evitar duplicados y memes

**Sin diversidad, el modelo memoriza el fondo en vez de aprender la lata.**""")

md("""### 5.2 Cargar las fotos recolectadas""")

code("""X_imgs, y_labels = [], []
for label, clase in enumerate(class_names):
    folder = Path(DATA_DIR) / clase
    files = sorted(folder.glob("*.jpg"))
    print(f"  {clase}: {len(files)} fotos")
    for fp in files:
        img = np.array(Image.open(fp).convert("RGB"))
        img_resized = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy()
        X_imgs.append(img_resized)
        y_labels.append(label)

X = np.array(X_imgs, dtype="float32") / 255.0
y = np.array(y_labels)
print(f"\\nTotal: {len(X)} fotos, shape: {X.shape}")
print(f"Balance: {dict(zip(class_names, np.bincount(y)))}")""")

code("""X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Visualizar muestras
fig, axes = plt.subplots(2, 6, figsize=(13, 4.5))
for k, clase in enumerate(class_names):
    idxs = np.where(y == k)[0][:6]
    for j, idx in enumerate(idxs):
        ax = axes[k, j]
        ax.imshow(X[idx])
        ax.axis("off")
        if j == 0:
            ax.set_ylabel(clase, fontsize=11, fontweight="bold")
plt.suptitle("Muestras del dataset recolectado", fontweight="bold")
plt.tight_layout(); plt.show()""")

md("""### 5.3 Entrenar con VGG16 (mismo patrón que arriba)

Backbone congelado + cabeza nueva + data augmentation (porque tenemos muy poca data).""")

code("""base_l = VGG16(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_l.trainable = False

modelo_latas = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),
    base_l,
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(32, activation="relu"),
    layers.Dense(len(class_names), activation="softmax"),
], name="VGG16_Latas")

modelo_latas.compile(optimizer="adam",
                     loss="sparse_categorical_crossentropy",
                     metrics=["accuracy"])
modelo_latas.summary()""")

code("""es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history_l = modelo_latas.fit(X_tr_l, y_tr_l,
                              epochs=25,                  # techo, EarlyStopping decide
                              batch_size=16,
                              validation_split=0.15,
                              callbacks=[es],
                              verbose=2)
acc_l = modelo_latas.evaluate(X_te_l, y_te_l, verbose=0)[1]
print(f"\\nAccuracy en test: {acc_l:.3f}  ({len(history_l.history['loss'])} epochs)")""")

code("""# Curvas y confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(history_l.history["loss"], label="train", color="#C82B40")
axes[0].plot(history_l.history["val_loss"], label="val", color="#2563EB")
axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(history_l.history["accuracy"], label="train", color="#C82B40")
axes[1].plot(history_l.history["val_accuracy"], label="val", color="#2563EB")
axes[1].set_title("Accuracy"); axes[1].legend()

y_pred_l = modelo_latas.predict(X_te_l, verbose=0).argmax(axis=1)
cm_l = confusion_matrix(y_te_l, y_pred_l)
ax = axes[2]
ax.imshow(cm_l, cmap="Reds")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_l[i,j]), ha="center", va="center",
                fontsize=14, fontweight="bold")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
ax.set_title(f"Confusion (acc={acc_l:.0%})")
plt.tight_layout(); plt.show()""")

md("""### 5.4 App de predicción""")

code("""def predecir(imagen):
    if imagen is None:
        return None
    img_resized = tf.image.resize(imagen, (IMG_SIZE, IMG_SIZE)).numpy() / 255.0
    img_batch = img_resized[None, ...]
    probs = modelo_latas.predict(img_batch, verbose=0)[0]
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

with gr.Blocks(title="Clasificador de latas") as predictor:
    gr.Markdown("# Clasificador: Coca-Cola vs Pepsi")
    gr.Markdown("Toma una foto de una lata o sube una imagen.")
    with gr.Row():
        cam = gr.Image(sources=["webcam", "upload"], type="numpy", label="Foto")
        out = gr.Label(num_top_classes=2, label="Prediccion")
    cam.change(predecir, inputs=cam, outputs=out)

predictor.launch(share=True, debug=False)""")

# ============================================================================
# 6. APP 1: SLIDING WINDOW
# ============================================================================
md("""---
## 6. App 1: ¿y si uso esta CNN para *detectar*?

Tenemos una CNN que clasifica `coca_cola` vs `pepsi` a nivel imagen. ¿Y si quiero **detectar** las latas en una foto con varias?

**Idea naive:** barrer la imagen con una ventana deslizante. Por cada parche, preguntar a la CNN qué es. Si la confianza pasa un umbral, dibujar caja.

> Esto es exactamente como se hacía detección antes de YOLO. Vamos a ver por qué se reemplazó.""")

code("""def sliding_window_detect(img, modelo, ventana=IMG_SIZE, paso=64,
                          umbral=0.85, max_lado=512):
    \"\"\"
    Barre la imagen con ventanas de tamano fijo y clasifica cada parche.

    Trucos para que sea usable en clase:
      1) Si la imagen es enorme, la reducimos a max_lado px (mantiene aspect ratio).
      2) Recolectamos TODOS los parches primero, luego una unica llamada
         batched a model.predict (en vez de N llamadas, una por parche).
    \"\"\"
    # 1) Reducir tamano si hace falta
    H0, W0 = img.shape[:2]
    scale = min(1.0, max_lado / max(H0, W0))
    if scale < 1.0:
        new_size = (int(W0 * scale), int(H0 * scale))
        img = np.array(Image.fromarray(img.astype("uint8")).resize(new_size))
    H, W = img.shape[:2]

    # 2) Recolectar parches y coordenadas
    parches, coords = [], []
    for y in range(0, H - ventana + 1, paso):
        for x in range(0, W - ventana + 1, paso):
            parches.append(img[y:y+ventana, x:x+ventana])
            coords.append((x, y))

    if not parches:
        return [], img

    # 3) UNA sola prediccion batched
    batch = np.asarray(parches, dtype="float32") / 255.0
    probs = modelo.predict(batch, verbose=0, batch_size=64)

    # 4) Filtrar por umbral
    cajas = []
    for (x, y), p in zip(coords, probs):
        clase = int(p.argmax())
        conf  = float(p.max())
        if conf > umbral:
            cajas.append((x, y, ventana, ventana, clase, conf))
    return cajas, img

print("Funcion sliding_window_detect lista (batched).")""")

code("""def dibujar_cajas(img, cajas, class_names):
    \"\"\"Dibuja las cajas sobre la imagen y devuelve la copia anotada.\"\"\"
    from PIL import ImageDraw, ImageFont
    img_pil = Image.fromarray(img.astype("uint8")).convert("RGB")
    draw = ImageDraw.Draw(img_pil)
    colores = ["#C82B40", "#2563EB", "#16A34A", "#EA580C"]
    for (x, y, w, h, k, c) in cajas:
        col = colores[k % len(colores)]
        draw.rectangle([x, y, x+w, y+h], outline=col, width=3)
        draw.text((x+3, y+3), f"{class_names[k]} {c:.0%}", fill=col)
    return np.array(img_pil)""")

md("""### 6.1 App Gradio del detector casero""")

code("""def detectar_sliding(imagen, umbral):
    if imagen is None:
        return None, "Sube una foto"
    t0 = time.time()
    cajas, img_proc = sliding_window_detect(imagen, modelo_latas,
                                            ventana=IMG_SIZE, paso=64,
                                            umbral=umbral)
    dt = time.time() - t0
    img_anotada = dibujar_cajas(img_proc, cajas, class_names)
    msg = (f"{len(cajas)} cajas en {dt:.2f}s "
           f"(umbral={umbral:.2f}, ventana={IMG_SIZE}, paso=64)")
    return img_anotada, msg

with gr.Blocks(title="Sliding Window") as app1:
    gr.Markdown("# Detector casero: sliding window con la CNN de latas")
    gr.Markdown("Sube una foto con varias latas. La CNN va a barrer toda la imagen con ventanas y dibujar caja en los parches que reconoce como Coca o Pepsi.")
    with gr.Row():
        inp = gr.Image(sources=["upload", "webcam"], type="numpy", label="Foto")
        out_img = gr.Image(label="Detecciones")
    umbral = gr.Slider(0.5, 0.99, value=0.9, step=0.01, label="Umbral de confianza")
    out_msg = gr.Textbox(label="Estado")
    btn = gr.Button("Detectar", variant="primary")
    btn.click(detectar_sliding, inputs=[inp, umbral], outputs=[out_img, out_msg])

app1.launch(share=True, debug=False)""")

md("""### 6.2 ¿Qué van a observar?

| Síntoma | Por qué |
|---|---|
| **N predicciones por imagen** | Estamos clasificando ~30-40 parches por foto. Lo hacemos rápido porque batcheamos en una sola llamada al modelo, pero conceptualmente el detector tiene que evaluar **muchas regiones independientes**. |
| **Cajas duplicadas** | Cada lata aparece detectada por varias ventanas vecinas. Hay rectángulos solapados sobre el mismo objeto — sliding window no sabe que son la misma lata. |
| **Tamaño fijo** | La ventana es 128×128. Una lata mucho más grande o pequeña se detecta mal. Habría que repetir el barrido a varias escalas (más lento aún). |
| **Falsos positivos** | Una etiqueta roja en una bolsa no es una Coca, pero la CNN dice que sí con confianza alta. La CNN nunca aprendió a decir "no es ninguna de las dos". |

**Conclusión pedagógica:** la clasificación tiene un **techo natural** para tareas de detección. Hace falta una arquitectura *diseñada* para producir múltiples bboxes en una pasada, sin barrer la imagen, y con un mecanismo para descartar duplicados.

Esa es **YOLO**.""")

# ============================================================================
# 7. APP 2: YOLO PREENTRENADO
# ============================================================================
md("""---
## 7. App 2: YOLO preentrenado

YOLO produce **todas las cajas en una sola pasada** de la red. Y viene preentrenado en **COCO** (80 categorías comunes: persona, carro, perro, botella, ...). Sin entrenar nada, ya detecta esas 80 cosas.""")

code("""from ultralytics import YOLO

# Modelo nano (~6 MB), corre en CPU
yolo_model = YOLO("yolo11n.pt")
print("YOLO cargado.")
print(f"Numero de clases: {len(yolo_model.names)}")
print(f"Algunas clases: {list(yolo_model.names.values())[:15]} ...")""")

md("""> **Nota:** uso `yolo11n.pt` porque es la versión más estable disponible al momento de la clase. La API es idéntica a la de YOLO26 — basta cambiar el nombre del peso cuando esté disponible.""")

code("""def detectar_yolo(imagen, conf):
    if imagen is None:
        return None, "Sube una foto"
    t0 = time.time()
    res = yolo_model.predict(imagen, conf=conf, verbose=False)
    dt = time.time() - t0
    img_anotada = res[0].plot()                    # BGR (OpenCV)
    img_rgb = img_anotada[:, :, ::-1]              # a RGB
    cajas = res[0].boxes
    n = 0 if cajas is None else len(cajas)
    if n > 0:
        clases_det = [yolo_model.names[int(c)] for c in cajas.cls.cpu().numpy()]
        msg = f"{n} objetos en {dt*1000:.0f}ms. Clases: {set(clases_det)}"
    else:
        msg = f"Ningun objeto detectado en {dt*1000:.0f}ms."
    return img_rgb, msg

with gr.Blocks(title="YOLO") as app2:
    gr.Markdown("# YOLO preentrenado en COCO")
    gr.Markdown("Sube una foto. YOLO detecta lo que reconoce de sus 80 clases.")
    with gr.Row():
        inp = gr.Image(sources=["upload", "webcam"], type="numpy", label="Foto")
        out_img = gr.Image(label="Detecciones")
    conf = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="Umbral de confianza")
    out_msg = gr.Textbox(label="Estado")
    btn = gr.Button("Detectar", variant="primary")
    btn.click(detectar_yolo, inputs=[inp, conf], outputs=[out_img, out_msg])

app2.launch(share=True, debug=False)""")

md("""### 7.1 Experimento 1: suban una foto cualquiera

Foto de la oficina, de la calle, de la planta, de su gato, de una nevera con producto. Cualquier cosa.

**Lo que van a ver:**
- Personas, carros, perros, sillas, botellas, **detectados al instante con confianza alta**.
- Una sola pasada: **milisegundos** por foto, no segundos como sliding window.
- **Sin cajas duplicadas** (NMS ya se encarga).
- **Múltiples objetos**: 5 personas en la foto = 5 cajas con etiqueta correcta.

Ese es el salto vs sliding window.""")

md("""### 7.2 Experimento 2: ahora suban una foto con una placa

Una foto de un carro o camión, de la calle, de una placa de Ecuador, lo que sea con una placa visible.

**Lo que van a ver:**
- YOLO detecta `car` con alta confianza.
- A veces detecta `truck`, `person` (conductor), `wheel`.
- **Pero NO detecta la placa.** Ni siquiera la dibuja.

**¿Por qué?** Porque `license plate` **no está entre las 80 clases de COCO**. YOLO nunca vio una placa etiquetada como tal — solo carros enteros. No es un bug, es que **el modelo solo ve lo que le enseñaron a ver**.""")

# ============================================================================
# 8. CIERRE
# ============================================================================
md("""---
## 8. Cierre y siguiente clase

| Concepto | Lo que aprendimos hoy |
|---|---|
| **CNN clásica** | Llega a ~80% en cats vs dogs. Falla en imágenes pequeñas, poses raras, fondos confusos. **Techo de capacidad.** |
| **Diagnóstico de errores** | Matriz de confusión + top-N errores. Mostrar al modelo cuándo se equivoca con confianza es el primer paso para mejorarlo. |
| **Transfer learning (VGG16)** | Backbone preentrenado + cabeza nueva. Salto de ~80% a ~95% sin recolectar más data. |
| **Tu propio dataset** | Gradio webcam + VGG16. El flujo completo en menos de 1 hora. |
| **Sliding window** | Funciona, pero es lento, genera duplicados y no escala. **Techo de la clasificación.** |
| **YOLO preentrenado** | Una pasada. 80 clases COCO. Brutal pero **no ve placas**. |

### Próxima clase
- **Métricas de detección**: mAP@0.5, precision, recall en bboxes.
- Etiquetar nuestro propio dataset de placas.
- **Fine-tunear YOLO26** sobre el dataset propio (mismo movimiento que clase 24, ahora con detección).
- Integrar OCR (lectura del texto de la placa).
- Desplegar la app LPR completa.

> **Para la próxima:** descargar `plates_unlabeled.zip` del repo y crear cuenta en `app.cvat.ai` (gratis). No hay que etiquetar todavía — eso lo hacemos juntos en clase.""")

# ============================================================================
# Escribir el notebook
# ============================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "Clase_25_Detection.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"OK: {out} ({len(cells)} celdas)")
