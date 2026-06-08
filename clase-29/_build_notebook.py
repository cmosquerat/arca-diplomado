"""Construye Clase_29_Autoencoders.ipynb desde celdas declaradas en orden."""
import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append(("markdown", text))


def code(text):
    CELLS.append(("code", text))


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
md("""# Clase 29: Autoencoders — Aprender Sin Etiquetas

**Diplomado en Data Science Aplicada con Python** · Arca Continental Ecuador x UDLA

---

**Objetivos de hoy:**
1. **Cerrar la clase 28**: entrenar el segmentador de píldoras desde Roboflow y armar la app de conteo.
2. Entender por qué **necesitamos aprender sin etiquetas** y conectar con PCA / KMeans (clases 19-21).
3. Definir el **autoencoder** desde su nombre: **codificación automática en un espacio latente** — la red aprende su propio sistema de coordenadas para describir los datos.
4. Entrenar nuestro primer AE en **MNIST** (Dense y Convolucional) y visualizar el espacio latente.
5. Aplicarlo a **denoising**, **detección de anomalías** (caso de negocio para Arca), **datos tabulares** y entender cómo **U-Net** (segmentación) usa la misma arquitectura encoder-decoder.
6. Saber **cuándo NO usar** un AE y qué viene después (VAE, autoencoders secuenciales).

> Correr en Colab con GPU: *Runtime → Change runtime type → T4 GPU*.""")


# ─────────────────────────────────────────────────────────────────────────────
# 0. Setup
# ─────────────────────────────────────────────────────────────────────────────
md("""## 0. Setup

Instalamos las librerías que necesitamos hoy. **Ultralytics** solo en la primera mitad (cierre de clase 28); **TensorFlow/Keras** para autoencoders.""")

code("""!pip install -q -U pillow
!pip install -q ultralytics roboflow gradio""")

code("""# Fix de PIL si Colab tiene la version vieja en memoria
import sys
for mod in list(sys.modules):
    if mod.startswith(("PIL", "matplotlib")):
        del sys.modules[mod]""")

code("""import os, io, time, glob, shutil, zipfile, warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model

import torch
device_torch = "cuda" if torch.cuda.is_available() else "cpu"
print(f"TensorFlow: {tf.__version__}  |  Torch: {torch.__version__}")
print(f"GPU TF:    {len(tf.config.list_physical_devices('GPU'))} dispositivo(s)")
print(f"GPU Torch: {device_torch}")""")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cierre Clase 28 — entrenar yolo26n-seg de píldoras
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 1. Cierre de Clase 28: Entrenar el Segmentador de Píldoras

La clase pasada llegamos hasta la **galería de demos** del modelo pretrained, pero no alcanzamos a entrenar nuestro propio segmentador. Lo hacemos rápido ahora porque la receta es **la misma de clase 27** — la única diferencia es el sufijo `-seg` en el modelo.

**Plan:**
1. Bajar el dataset de píldoras directamente desde **Roboflow Universe** (ya etiquetado).
2. Generar el `data.yaml`.
3. Entrenar `yolo26n-seg` por 20 epochs.
4. Construir la app Gradio de conteo.

> ⚠️ Nos saltamos el flujo de Label Studio porque no alcanzamos a etiquetar suficientes imágenes. **Roboflow Universe** funciona como un "GitHub de datasets" — alguien más ya hizo el trabajo de etiquetado.

### 1.1 Bajar el dataset de Roboflow""")

code("""from roboflow import Roboflow

ROBOFLOW_API_KEY = "DJqoR0JeH6JaOrpH712W"

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project_rf = rf.workspace("abstract").project("pillsegmentation-oyygy")
version_rf = project_rf.versions()[0]
dataset_rf = version_rf.download("yolov8", location="/content/pills_raw")

print(f"\\nDataset bajado en: {dataset_rf.location}")""")

code("""# Verificar contenido
pills_raw = Path(dataset_rf.location)
for sub in ["train/images", "valid/images", "test/images"]:
    n = len(list((pills_raw / sub).glob("*"))) if (pills_raw / sub).exists() else 0
    print(f"  {sub}: {n} imágenes")""")

md("""### 1.2 Ver el dataset antes de entrenar

Antes de lanzar el entrenamiento, miramos qué tipo de imágenes y qué tipo de etiquetas trae el dataset. Eso es buena práctica con cualquier dataset nuevo — ahorra sorpresas durante el entrenamiento.""")

code("""from matplotlib.patches import Polygon as MplPoly

def leer_poligonos_yolo(path_txt):
    \"\"\"Lee un .txt de YOLO seg: cada línea es 'cls x1 y1 x2 y2 ...' normalizado.\"\"\"
    if not path_txt.exists():
        return []
    out = []
    for linea in path_txt.read_text().strip().splitlines():
        partes = linea.split()
        if len(partes) < 7:
            continue
        cls = int(partes[0])
        coords = list(map(float, partes[1:]))
        pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        out.append((cls, pts))
    return out

# Mostrar 6 imagenes del set de entrenamiento con sus poligonos.
# IMPORTANTE: el dataset viene de video; los frames consecutivos son casi
# identicos. Tomamos 6 imagenes DISTRIBUIDAS a lo largo del dataset (no las
# primeras 6) para ver diversidad real: distintos colores de pildora,
# distintos fondos, distintos angulos.
todas_imgs = sorted((pills_raw / "train" / "images").glob("*.jpg"))
n_total = len(todas_imgs)
indices_diversos = [int(i * n_total / 6) for i in range(6)]
img_paths = [todas_imgs[i] for i in indices_diversos]
print(f"Total de imagenes train: {n_total}")
print(f"Indices tomados: {indices_diversos}")

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, ip in zip(axes.flat, img_paths):
    lp = pills_raw / "train" / "labels" / (ip.stem + ".txt")
    img = np.array(Image.open(ip))
    H, W = img.shape[:2]
    ax.imshow(img)
    ax.axis("off")
    polys = leer_poligonos_yolo(lp)
    for _, pts in polys:
        pts_abs = [(p[0] * W, p[1] * H) for p in pts]
        ax.add_patch(MplPoly(pts_abs, alpha=0.35,
                              fc="cyan", ec="blue", lw=1.5))
    ax.set_title(f"{ip.name}  ·  {len(polys)} pildoras",
                  fontsize=9, fontweight="bold")

plt.suptitle("Muestras del dataset de pildoras (6 frames distribuidos)",
              fontweight="bold")
plt.tight_layout()
plt.show()""")

md("""**Lo que vemos:**
- Las píldoras vienen en distintos colores y tamaños (azules, blancas, rojas, etc.) — el modelo deberá generalizar.
- Algunas imágenes tienen píldoras pegadas entre sí — este es el caso donde detección con bbox falla y por eso queremos segmentación.
- Las etiquetas son polígonos cerrados (no bboxes). Cada polígono = una píldora.
- El fondo cambia: a veces tabla de madera, a veces fondo claro. El modelo debe aprender a separar pastilla de fondo independientemente del color del fondo.

### 1.3 `data.yaml` para segmentación

YOLO infiere automáticamente que el dataset es de seg porque los archivos `.txt` contienen **polígonos** en lugar de bboxes de 5 columnas. No hay que decirle nada especial.""")

code("""yaml_pills = f'''path: {Path(dataset_rf.location).absolute()}
train: train/images
val:   valid/images

names:
  0: pill
'''
yaml_path = Path(dataset_rf.location) / "data_local.yaml"
yaml_path.write_text(yaml_pills)
print(yaml_pills)""")

md("""### 1.4 Entrenar `yolo26n-seg`

Mismo `train()` que detection — el sufijo `-seg` activa el head de segmentación. **~6-8 min en T4** con 20 epochs.""")

code("""from ultralytics import YOLO

seg_pills = YOLO("yolo26n-seg.pt")
results = seg_pills.train(
    data=str(yaml_path),
    epochs=20,
    imgsz=640,
    batch=8,
    device=0 if device_torch == "cuda" else "cpu",
    project="/content/runs",
    name="pills",
    exist_ok=True,
    verbose=False,
)
print("Entrenamiento terminado.")

PILL_BEST = sorted(glob.glob("/content/runs/**/pills/weights/best.pt",
                              recursive=True))[0]
print(f"best.pt: {PILL_BEST}")""")

md("""### 1.5 Validar — la métrica que importa es `seg.map50`

Cuando entrenas con `-seg`, Ultralytics reporta DOS juegos de métricas: `box.*` (bbox derivada de la máscara) y `seg.*` (la máscara en sí). **La segunda es la que importa** para nuestro problema.""")

code("""best_pills = YOLO(PILL_BEST)
m = best_pills.val(data=str(yaml_path), verbose=False)
print(f"BBox  mAP@0.5      = {m.box.map50:.3f}")
print(f"Mask  mAP@0.5      = {m.seg.map50:.3f}    ← la que importa")
print(f"Mask  mAP@0.5:0.95 = {m.seg.map:.3f}")""")

md("""### 1.6 Ejemplo de conteo sobre una imagen de test

Antes de envolver el modelo en una app, lo probamos a mano sobre una imagen del set de test. Esto valida que la predicción funciona como esperamos y nos deja ver el output crudo del modelo.""")

code("""# Tomar una imagen cualquiera del set de test
test_imgs = sorted((pills_raw / "test" / "images").glob("*.jpg"))
img_test = str(test_imgs[0])
print(f"Imagen a procesar: {img_test}")

# Predecir con el modelo entrenado
resultado = best_pills(img_test, conf=0.25, verbose=False)[0]

# El número de píldoras es la cantidad de máscaras detectadas
n_pildoras = len(resultado.masks.data) if resultado.masks is not None else 0
print(f"Píldoras detectadas: {n_pildoras}")""")

md("""La salida del modelo trae: bboxes, máscaras como tensores, áreas, scores de confianza. Lo más simple para contar es **contar máscaras**:""")

code("""# Visualizar el resultado con conteo prominente
img_anotada = resultado.plot()[..., ::-1]   # YOLO devuelve BGR, pasamos a RGB

fig, ax = plt.subplots(figsize=(11, 8))
ax.imshow(img_anotada)
ax.axis("off")
ax.set_title(f"Píldoras detectadas: {n_pildoras}",
              fontsize=20, fontweight="bold", color="#C82B40", pad=12)
plt.tight_layout()
plt.show()""")

md("""**Cómo leer la imagen anotada:**
- Cada píldora detectada está sombreada con un color distinto.
- La etiqueta encima de cada una muestra la clase (`pill`) y la confianza del modelo.
- Las píldoras pegadas reciben máscaras **separadas** — esto es lo que NO podríamos lograr con bboxes que se superpondrían.

### Calcular áreas para reglas de negocio

Si algún cliente quiere contar **solo píldoras enteras** (descartar fragmentos), filtramos por área de la máscara:""")

code("""# Áreas de cada píldora en píxeles
areas = resultado.masks.data.sum(dim=(1, 2)).cpu().numpy()
print(f"Píldoras detectadas:  {len(areas)}")
print(f"Área promedio:        {areas.mean():.0f} px")
print(f"Área mínima:          {areas.min():.0f} px")
print(f"Área máxima:          {areas.max():.0f} px")

# Aplicar regla: solo cuentan las que tengan ≥ 500 px de área
AREA_MIN = 500
n_enteras = int((areas >= AREA_MIN).sum())
n_descartadas = len(areas) - n_enteras
print(f"\\nCon filtro de área ≥ {AREA_MIN}:")
print(f"  Píldoras enteras:        {n_enteras}")
print(f"  Fragmentos descartados:  {n_descartadas}")""")

md("""### 1.7 App Gradio: la misma lógica en una interfaz web

Ahora que sabemos que la lógica funciona, la envolvemos en una app Gradio. La función `contar` hace exactamente lo mismo de arriba: predicción + filtro por área.""")

code("""import gradio as gr

def contar(imagen, conf, area_min):
    res = best_pills(imagen, conf=conf, verbose=False)[0]
    if res.masks is None:
        return imagen, "Sin detecciones"
    areas = res.masks.data.sum(dim=(1, 2)).cpu().numpy()
    n_total = len(areas)
    n_validas = int((areas >= area_min).sum())
    anotada = res.plot()[..., ::-1]
    msg = (f"Píldoras detectadas: {n_total}\\n"
            f"Enteras (≥ {area_min} px): {n_validas}")
    return anotada, msg

demo = gr.Interface(
    fn=contar,
    inputs=[
        gr.Image(type="numpy", label="Imagen del blister"),
        gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confianza mínima"),
        gr.Slider(100, 2000, value=500, step=50, label="Área mínima (px)"),
    ],
    outputs=[
        gr.Image(label="Detección"),
        gr.Textbox(label="Resultado", lines=2),
    ],
    title="Contador de Píldoras",
)
demo.launch(share=True, debug=False)""")

md("""**Listo.** Con eso queda cerrado el módulo de imágenes:
- Clasificación → CNN (clase 22-24)
- Detección → YOLO bbox (clase 25-26)
- Segmentación → YOLO masks (clase 27-29)

Ahora damos un giro grande: **¿y si NO tenemos etiquetas?**""")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cambio de tema
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 2. El Problema: Aprender Sin Etiquetas

### Una observación incómoda

Mira todo lo que hicimos hasta aquí:

| Tarea | Necesita | Cuántas etiquetas |
|-------|----------|--------------------|
| Clasificación (cats/dogs, MNIST) | Imagen + clase | Miles |
| Detección de placas | Imagen + bbox por cada placa | Miles de bboxes |
| Segmentación de tumor/píldoras | Imagen + polígono por cada objeto | Cientos de polígonos |

**El cuello de botella del mundo real:** etiquetar es **caro y lento**. Un humano etiqueta ~200 imágenes/hora para clasificación, ~30 para detección, ~10 para segmentación. Una planta de producción genera **decenas de miles de imágenes al día**.

> "Los datos sin etiquetar son baratos. Los datos etiquetados son caros. La pregunta interesante es: **¿qué podemos aprender de los datos baratos?**"

### Esto NO es nuevo en el curso

Ya tocamos aprendizaje **no supervisado** dos veces:

| Clase | Técnica | Qué hace |
|-------|---------|----------|
| 19-20 | **K-Means** | Agrupa puntos por cercanía |
| 21 | **PCA / t-SNE** | Comprime dimensiones, encuentra ejes principales |

Ambos son métodos **lineales o basados en distancia**. Hoy vemos su versión **neuronal y no lineal**: el autoencoder.""")


# ─────────────────────────────────────────────────────────────────────────────
# 3. La idea del autoencoder
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 3. ¿Qué Es un Autoencoder?

### El nombre lo dice todo

**Auto** + **encoder** = **codificación automática**. Es una red que aprende, por sí misma, **cómo codificar** los datos en un sistema de coordenadas distinto al original. No le decimos qué ejes usar; le damos los datos y ella construye su propio esquema de codificación.

A ese sistema de coordenadas internas le llamamos **espacio latente**. El autoencoder es, esencialmente, **una máquina para aprender espacios latentes**.

### Esto no es nuevo — ya lo vieron varias veces

| Técnica | Qué aprende | Cómo |
|---------|-------------|------|
| PCA (clase 21) | Coordenadas lineales para datos tabulares | SVD, ejes ortogonales fijos |
| Word2Vec | Coordenadas para palabras (embeddings) | Predecir contexto |
| t-SNE / UMAP (clase 21) | Coordenadas 2D para visualizar | Preservar vecindades |
| **Autoencoder** | Coordenadas para cualquier dato | Reconstrucción aprendida con la red |

La diferencia clave: el AE puede aprender un sistema de coordenadas **no lineal**. Donde PCA dibuja ejes rectos en el espacio original, el AE dibuja ejes que se curvan para acomodar la estructura real de los datos.

### Las tres piezas

```
   INPUT          ENCODER         LATENTE          DECODER         OUTPUT
  (28x28)    ───────────────►    (32 dim)    ───────────────►    (28x28)
                                                                    ≈ INPUT
```

| Pieza | Función |
|-------|---------|
| **Encoder** | Traduce un dato a sus coordenadas latentes |
| **Latente** | El dato re-escrito en el nuevo sistema de coordenadas |
| **Decoder** | Recupera el dato original a partir del latente |

### Por qué la reconstrucción es solo el examen

La reconstrucción no es el producto, es **la prueba**. Si el decoder logra recuperar el input a partir del latente, entonces el latente contiene la información necesaria. Es un mecanismo de validación, no el objetivo.

$$\\text{Loss} = \\text{MSE}(X, \\text{decoder}(\\text{encoder}(X)))$$

Como el "label" es el mismo input, no hace falta etiquetar nada. A este régimen se le llama **aprendizaje auto-supervisado**: la supervisión la generan los datos.

### Lo que nos llevamos al final

El producto del entrenamiento NO es la reconstrucción borrosa del output. Es el **encoder** — el traductor entrenado al espacio latente. Las aplicaciones del AE son distintos usos de ese traductor:

| Uso | Qué hacemos con el latente |
|-----|----------------------------|
| Visualización | Plotear el latente 2D/3D coloreado por alguna variable de interés |
| Detección de anomalías | Medir qué tan mal reconstruye → señal de que el latente no describe el dato |
| Compresión / pretraining | Pasar `encoder(X)` como feature a un modelo downstream |
| Denoising | El decoder solo reconstruye lo que el latente puede describir → el ruido se queda fuera |
| Generación | Muestrear puntos del latente y pasarlos por el decoder (AE clásico funciona mediocre, VAE bien) |

### Por qué tiene que haber compresión

Si el latente tuviera 784 dimensiones, el encoder podría aprender la identidad (copia perfecta, latente = input) sin haber descrito nada nuevo. La compresión a un espacio más chico (32, en este ejemplo) **es la restricción** que obliga al encoder a elegir qué información retener. Lo que sobrevive al cuello es, por construcción, lo que distingue las observaciones entre sí en tus datos.

Eso responde una pregunta práctica directa:

> ¿Qué es informativo en mis datos?

El espacio latente entrenado es la respuesta operativa del modelo a esa pregunta.""")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Conexión con PCA
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 4. Conexión con PCA — Qué Asume y Cuándo Lo Supera

PCA (clase 21) hace **exactamente** lo mismo conceptualmente:

```
   X (784 dim)    ─────►    componentes (k dim)    ─────►    X_reconstruido
                 W                                W^T
```

PCA es matemáticamente equivalente a un **autoencoder lineal** (sin función de activación):

- Si tu AE tiene **una sola capa Dense en encoder y otra en decoder**, **sin activación**, y minimiza MSE → estás haciendo PCA con gradient descent en vez de SVD.

### Diferencias clave

| | **PCA** | **Autoencoder** |
|---|---|---|
| Linealidad | Solo lineal | Lineal o no-lineal (con activaciones) |
| Cómo se entrena | SVD (cerrado, una vez) | Gradient descent (iterativo) |
| Componentes | Ortogonales, ordenados por varianza | Sin orden, sin ortogonalidad |
| Interpretabilidad | Alta (los ejes tienen significado) | Baja (el latente es "caja negra") |
| Captura patrones | Lineales | **No lineales** (rotaciones, curvas, manifolds) |
| Coste | Barato | Caro (entrenar una red) |

### Cuándo cada uno

- **PCA**: si los datos son aproximadamente lineales, si necesitas interpretabilidad, o como **primer baseline siempre**.
- **Autoencoder**: si los datos tienen **estructura no lineal** (imágenes, audio, secuencias) y tienes suficiente data + cómputo.

> **Regla**: prueba PCA primero. Si la reconstrucción es mala, sube a un AE no lineal.""")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Estructura típica de un autoencoder + 6 capas Keras
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 5. Estructura Típica de un Autoencoder

Antes de codificar, una vista panorámica. Hay **cinco layouts** que cubren el 95% de los casos prácticos. Saber cuál corresponde a tu problema te ahorra prototipar a ciegas.

### Los cinco layouts

| Layout | Encoder | Decoder | Cuándo usarlo |
|--------|---------|---------|---------------|
| **Dense AE** | `Dense → Dense` | `Dense → Dense` | Datos tabulares, vectores, embeddings |
| **Conv AE** | `Conv2D + Pool` | `Conv2D + UpSampling` | Imágenes (lo veremos hoy con MNIST) |
| **Conv AE híbrido** | `Conv2D + Pool → Flatten → Dense` | `Dense → Reshape → Conv2D + UpSampling` | Imágenes cuando el latente debe ser un vector pequeño (visualización, búsqueda) |
| **U-Net** | `Conv2D + Pool` | `Conv2D + UpSampling` **+ skip connections** | Segmentación pixel-wise (clase 28) |
| **Sequence-to-sequence AE** | `LSTM / Transformer` | `LSTM / Transformer` | Series temporales, texto, audio secuencial (Módulo 5) |

### El patrón común

Todos comparten la misma estructura conceptual:

1. **Input shape** depende del dato (vector, imagen, secuencia).
2. **Encoder** que reduce dimensionalidad progresivamente.
3. **Bottleneck** — la dimensión más chica de la red.
4. **Decoder** que es **aproximadamente simétrico** al encoder (no obligatorio).
5. **Output shape igual al input shape** (para reconstrucción).
6. **Loss** = distancia entre input y output (MSE, BCE, SSIM, etc).

### Asimetrías permitidas

El decoder NO tiene que ser un espejo exacto del encoder. Decisiones comunes:
- Decoder más simple (menos parámetros) — entrena más rápido.
- Decoder con `Conv2DTranspose` en lugar de `UpSampling2D + Conv2D` — más parámetros pero aprende cómo agrandar.
- Capa final con activación distinta al resto del decoder (típicamente `sigmoid` o `linear`).""")

md("""---
## 6. Las 6 Capas Keras Que Aparecen en Todo Autoencoder de Imágenes

Repasamos las piezas del lego antes de armarlas. Si reconoces estas 6, puedes leer cualquier arquitectura de AE.

| Capa | Qué hace | Parámetros clave |
|------|----------|------------------|
| `keras.Input(shape=...)` | Declara el shape del dato. **Siempre** la primera línea | `shape` sin la dim de batch |
| `layers.Dense(n, activation=...)` | Fully-connected con `n` unidades | activation: relu / sigmoid / linear |
| `layers.Conv2D(n, k, padding=..., activation=...)` | `n` filtros de `k×k` | `padding="same"` mantiene tamaño |
| `layers.MaxPooling2D(2)` | Reduce a la mitad cada dim espacial | Sin parámetros aprendibles |
| `layers.UpSampling2D(2)` | Duplica cada dim espacial | Sin parámetros aprendibles |
| `layers.Conv2DTranspose(n, k, strides=2)` | "Convolución inversa" — aprende cómo agrandar | Alternativa a UpSampling + Conv |

### La API Functional de Keras — cómo se escribe

Vamos a usar **siempre** Functional API. Es más explícita que Sequential y deja ver claramente la separación encoder/decoder, que es el corazón conceptual del autoencoder.

**La regla mental:** una capa Keras se comporta como **una función**. Recibe un tensor, devuelve otro tensor. Las "encadenas" llamándolas con paréntesis:

```python
# Receta general en 4 pasos
# 1. Declarar el input
entrada = keras.Input(shape=(784,))

# 2. Pasar por capas como funciones (cada capa devuelve un tensor)
h = layers.Dense(128, activation="relu")(entrada)
h = layers.Dense(64,  activation="relu")(h)
salida = layers.Dense(10, activation="softmax")(h)

# 3. Envolver entrada + salida en un Model
modelo = keras.Model(entrada, salida)

# 4. Compilar y entrenar como cualquier modelo
modelo.compile(optimizer="adam", loss="categorical_crossentropy")
```

Lo único que cambia con un AE: vamos a definir **dos Models** que comparten capas — uno encoder, otro decoder — y luego un tercero que los compone (el AE entero).""")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Primer autoencoder: Dense en MNIST (Functional API)
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 7. Primer Autoencoder: Dense en MNIST con Functional API

Empezamos con capas Dense puras para aislar el mecanismo del AE de las convoluciones. **Toda la construcción es Functional**: declarar input, pasar por capas como funciones, envolver en `Model`.""")

code("""# Cargar MNIST y normalizar a [0, 1]
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

# Aplanar: cada imagen 28x28 -> vector de 784
X_train_flat = X_train.reshape(-1, 784)
X_test_flat  = X_test.reshape(-1, 784)
print(f"X_train_flat: {X_train_flat.shape}  (60.000 vectores de 784)")""")

md("""### 7.1 Construir el AE paso a paso con Functional

El truco es definir las capas del encoder y del decoder por separado, conectarlas, y armar **tres modelos** que comparten las mismas capas:

1. `encoder` — solo de input al latente.
2. `decoder` — solo de latente a output.
3. `autoencoder` — el encoder y decoder pegados, lo que entrenamos.

Más adelante usaremos los tres por separado: el autoencoder para entrenar, el encoder para extraer el latente, el decoder para generar.""")

code("""LATENT_DIM = 32

# === ENCODER: 784 -> 128 -> 32 ===
entrada     = keras.Input(shape=(784,), name="entrada")
h_enc       = layers.Dense(128, activation="relu", name="enc_h1")(entrada)
latente     = layers.Dense(LATENT_DIM, activation="relu", name="latente")(h_enc)

# Wrap como modelo separado (solo input -> latente)
encoder = keras.Model(entrada, latente, name="encoder")

# === DECODER: 32 -> 128 -> 784 ===
entrada_z   = keras.Input(shape=(LATENT_DIM,), name="entrada_z")
h_dec       = layers.Dense(128, activation="relu", name="dec_h1")(entrada_z)
reconstr    = layers.Dense(784,  activation="sigmoid", name="reconstr")(h_dec)

decoder = keras.Model(entrada_z, reconstr, name="decoder")

# === AUTOENCODER: encoder seguido del decoder ===
# entrada -> encoder(entrada) -> decoder(eso)
salida_ae   = decoder(encoder(entrada))
autoencoder = keras.Model(entrada, salida_ae, name="autoencoder")

# Compilar el AE para entrenamiento.
# binary_crossentropy es la receta del tutorial oficial de Keras para
# pixels en [0, 1] -- converge más rápido y más estable que MSE.
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

print("Encoder shape:", encoder.output_shape)
print("Decoder shape:", decoder.output_shape)
autoencoder.summary()""")

md("""**Lo que acabamos de construir:**

- `encoder` y `decoder` son modelos independientes. Podemos llamar `encoder.predict(x)` y obtener solo el latente.
- `autoencoder` reutiliza las **mismas capas** que encoder y decoder — no son copias. Cuando entrenas el autoencoder, los pesos del encoder y decoder se actualizan automáticamente.
- Para el output usamos `sigmoid` porque los pixels están en `[0, 1]`. Con `binary_crossentropy` como loss, el entrenamiento es estable.""")

code("""# Entrenar: target = mismo input (NO hay y_train, es auto-supervisado)
history = autoencoder.fit(X_train_flat, X_train_flat,
                           epochs=20, batch_size=256,
                           validation_split=0.1, verbose=2)""")

md("""**Observación importante:** miramos `fit(X, X)`, no `fit(X, y)`. La firma visible de un AE.""")

code("""# Curva de loss
plt.figure(figsize=(7, 3.5))
plt.plot(history.history["loss"], label="train", color="#C82B40")
plt.plot(history.history["val_loss"], label="val", color="#2563EB")
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.legend()
plt.title("Reconstruction loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()""")

md("""### 7.3 Visualizar reconstrucciones

Tomamos 10 dígitos de test, los pasamos por el AE, y los comparamos lado a lado con el original.""")

code("""# Tomar 10 ejemplos al azar de test
idxs = np.random.RandomState(42).choice(len(X_test_flat), 10, replace=False)
originales = X_test_flat[idxs]
recons = autoencoder.predict(originales, verbose=0)

fig, axes = plt.subplots(2, 10, figsize=(14, 3))
for i in range(10):
    axes[0, i].imshow(originales[i].reshape(28, 28), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(recons[i].reshape(28, 28), cmap="gray")
    axes[1, i].axis("off")
axes[0, 0].set_title("Original", loc="left", fontweight="bold", fontsize=11)
axes[1, 0].set_title("Reconstruido", loc="left", fontweight="bold", fontsize=11)
plt.suptitle(f"AE Dense (latente={LATENT_DIM}): comprimió 784 → {LATENT_DIM} → 784",
              fontweight="bold")
plt.tight_layout()
plt.show()""")

md("""**Interpretación:**
- La reconstrucción es **borrosa pero identificable**. La borrosidad viene del MSE como criterio: el mínimo del error promedio entre todos los 3 plausibles es un 3 suavizado, no uno con trazos puntuales nítidos.
- Compresión de 24× (784 → 32 dimensiones) sin pérdida total de identidad. Esos 32 números bastan para regenerar la forma — lo que significa que ahí está la información que el modelo considera distintiva.
- La memoria del encoder no es por píxel sino por **estructura**: bordes, simetrías, regiones cerradas. Eso explica por qué la reconstrucción conserva la silueta pero pierde el grano fino.

### 7.4 Ver la compresión de UN dígito (paso a paso)

El latente del Dense AE es un vector de 32 números. No es una "imagen" en sí, pero podemos **dibujarlo como un heatmap** — cada uno de los 32 valores como una celda de color. Así se ve literalmente cuánto comprime el AE.""")

code("""# Tomamos UN dígito específico y mostramos el flujo paso a paso
idx_demo = 0   # primer dígito del test
x_orig = X_test_flat[idx_demo:idx_demo+1]                  # (1, 784)
z      = encoder.predict(x_orig, verbose=0)                # (1, 32)
x_rec  = autoencoder.predict(x_orig, verbose=0)            # (1, 784)

print(f"Forma del input:        {x_orig.shape}    →  {x_orig.size} valores")
print(f"Forma del latente:      {z.shape}     →  {z.size} valores")
print(f"Forma de reconstrucción:{x_rec.shape}   →  {x_rec.size} valores")
print(f"\\nCompresión: 784 → 32  (factor {784//32}x)")""")

code("""# Visualizar las 3 etapas con tamaños proporcionales a su dimensión
fig = plt.figure(figsize=(13, 4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 1])

# (1) Imagen original
ax0 = fig.add_subplot(gs[0])
ax0.imshow(x_orig.reshape(28, 28), cmap="gray")
ax0.set_title("INPUT\\n28×28 = 784 valores",
               fontsize=12, fontweight="bold", color="#6B1525")
ax0.set_xticks([])
ax0.set_yticks([])

# (2) Latente: 32 valores como heatmap horizontal (4x8 para que sea cómodo de ver)
ax1 = fig.add_subplot(gs[1])
z_grid = z.reshape(4, 8)   # acomodar los 32 valores en una cuadrícula 4x8
im = ax1.imshow(z_grid, cmap="viridis", aspect="auto")
ax1.set_title("LATENTE\\n32 valores  (compresión 24×)",
               fontsize=12, fontweight="bold", color="#7C3AED")
ax1.set_xticks(range(8))
ax1.set_yticks(range(4))
# Anotar cada celda con su valor
for i in range(4):
    for j in range(8):
        ax1.text(j, i, f"{z_grid[i, j]:.2f}",
                  ha="center", va="center", fontsize=8,
                  color="white" if z_grid[i, j] < z_grid.mean() else "black")
plt.colorbar(im, ax=ax1, fraction=0.04)

# (3) Reconstrucción
ax2 = fig.add_subplot(gs[2])
ax2.imshow(x_rec.reshape(28, 28), cmap="gray")
ax2.set_title("OUTPUT\\n28×28 = 784 valores",
               fontsize=12, fontweight="bold", color="#6B1525")
ax2.set_xticks([])
ax2.set_yticks([])

# Flechas entre etapas
fig.text(0.34, 0.5, "encoder\\n→", fontsize=14, fontweight="bold",
          color="#C82B40", ha="center", va="center")
fig.text(0.68, 0.5, "decoder\\n→", fontsize=14, fontweight="bold",
          color="#C82B40", ha="center", va="center")

plt.suptitle("Compresión real del Dense AE sobre UN dígito",
              fontweight="bold", fontsize=14, color="#6B1525")
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()""")

md("""**Lo que tienes que ver en esta figura:**

- A la **izquierda**, el input: 784 pixels, claramente reconocible.
- En el **centro**, los 32 valores que el encoder devolvió — ESA es la versión comprimida del dígito. Es lo que se almacenaría, transmitiría o indexaría en producción.
- A la **derecha**, lo que el decoder reconstruye desde esos 32 valores. Se parece al input pero pierde detalle fino.

> **Comparación con Conv AE (sección 9):** allí el latente es `(4, 4, 8) = 128 valores` que se ven como una mini-imagen. Aquí es un vector plano de 32 valores que vemos como un heatmap 4×8. Misma idea, distinta forma del latente.

### 7.5 Compresión por dígito — comparar varias clases

Ahora aplicamos lo mismo a 10 clases distintas: vemos cómo varía la reconstrucción por dígito y observamos que algunos comprimen mejor que otros.""")

code("""# 5 ejemplos por dígito y su reconstrucción
fig, axes = plt.subplots(10, 10, figsize=(11, 11))
for d in range(10):
    idxs = np.where(y_test == d)[0][:5]
    for k, idx in enumerate(idxs):
        x = X_test_flat[idx:idx+1]
        x_rec = autoencoder.predict(x, verbose=0)
        axes[d, 2*k].imshow(x.reshape(28, 28), cmap="gray_r");     axes[d, 2*k].axis("off")
        axes[d, 2*k+1].imshow(x_rec.reshape(28, 28), cmap="gray_r"); axes[d, 2*k+1].axis("off")
    axes[d, 0].text(-10, 14, f"{d}", fontsize=14, fontweight="bold",
                     color="#6B1525", va="center")

# Etiquetas de columna
for k in range(5):
    axes[0, 2*k].set_title("orig",  fontsize=8, color="gray", pad=2)
    axes[0, 2*k+1].set_title("recon", fontsize=8, color="#C82B40", pad=2)

plt.suptitle("Compresión por dígito — 5 ejemplos × 10 clases",
              fontweight="bold", y=0.995)
plt.tight_layout()
plt.show()""")

md("""**Lo que aparece sistemáticamente:**

- **Dígitos "fáciles"** (1, 0, 7): reconstrucción casi exacta. Sus trazos son rígidos y la geometría es predecible — encajan en pocas coordenadas latentes.
- **Dígitos "difíciles"** (5, 8): reconstrucción visiblemente borrosa. Tienen más variabilidad estructural y curvas cruzadas que no caben tan bien en 32 dimensiones.
- **Errores cruzados** (4 ↔ 9, 3 ↔ 8): la reconstrucción a veces se "desvía" hacia el dígito vecino más cercano en el latente.

> Este patrón no es ruido — es **información estructural** sobre la geometría del manifold de MNIST. Dos dígitos que comprimen distinto significan que el AE necesita más coordenadas para uno que para el otro.""")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualizar el espacio latente
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 8. ¿Qué Aprende el Latente? — Visualizar con bottleneck=2

Si forzamos el cuello de botella a **2 dimensiones**, podemos hacer un scatter plot del latente y ver cómo el AE organizó los dígitos en el plano.""")

code("""# AE con latente de SOLO 2 dimensiones (para visualizar)
enc2_in = keras.Input(shape=(784,))
x = layers.Dense(128, activation="relu")(enc2_in)
x = layers.Dense(32,  activation="relu")(x)
lat2 = layers.Dense(2, activation="linear", name="latent_2d")(x)   # ¡solo 2!
encoder_2d = Model(enc2_in, lat2)

dec2_in = keras.Input(shape=(2,))
x = layers.Dense(32,  activation="relu")(dec2_in)
x = layers.Dense(128, activation="relu")(x)
dec2_out = layers.Dense(784, activation="sigmoid")(x)
decoder_2d = Model(dec2_in, dec2_out)

ae_in  = keras.Input(shape=(784,))
ae_2d  = Model(ae_in, decoder_2d(encoder_2d(ae_in)))
ae_2d.compile(optimizer="adam", loss="mse")

ae_2d.fit(X_train_flat, X_train_flat, epochs=15, batch_size=256,
          validation_split=0.1, verbose=2)""")

code("""# Pasar todo el test set por el encoder y plotear el latente coloreado por clase
latent_test = encoder_2d.predict(X_test_flat, verbose=0)
print(f"latent_test.shape = {latent_test.shape}  (10.000 puntos en 2D)")

fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(latent_test[:, 0], latent_test[:, 1],
                      c=y_test, cmap="tab10", s=4, alpha=0.6)
ax.set_xlabel("Latente z1"); ax.set_ylabel("Latente z2")
ax.set_title("Espacio latente 2D del AE — coloreado por dígito real",
              fontweight="bold")
plt.colorbar(scatter, ticks=range(10), label="Dígito")
plt.tight_layout()
plt.show()""")

md("""**Lo importante de esta visualización:**
- Los dígitos se separan en clusters dentro del espacio latente. El entrenamiento no usó las etiquetas — la estructura de clases existe en los datos mismos y el encoder la refleja al construir sus coordenadas.
- El solape entre 4-9 y 3-8 mide **similitud visual**: el espacio latente coloca cerca a las observaciones que son difíciles de distinguir por su forma.
- Esto cumple el mismo rol pedagógico que t-SNE en clase 21, pero con una ventaja operativa: a diferencia de t-SNE, este encoder es **paramétrico** — podemos pasarle datos nuevos sin reentrenar.""")

code("""# Comparar con PCA (también 2D) — ¿qué tan distinto?
from sklearn.decomposition import PCA

pca_2d = PCA(n_components=2).fit(X_train_flat)
pca_test = pca_2d.transform(X_test_flat)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, data, title in zip(axes,
    [pca_test, latent_test],
    ["PCA (lineal)", "Autoencoder Dense (no lineal)"]):
    sc = ax.scatter(data[:, 0], data[:, 1], c=y_test, cmap="tab10", s=4, alpha=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
plt.suptitle("Misma data, dos compresiones a 2D distintas",
              fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()""")

md("""**Interpretación:**
- PCA encuentra los **2 ejes de mayor varianza**. Los dígitos quedan en una nube continua donde se mezclan más.
- El AE no lineal puede *curvar* el plano latente y **separa mejor los clusters**. Lo paga con menos interpretabilidad — los ejes z1/z2 no tienen significado claro.

### Bonus: pasear por el latente y generar dígitos

Si elegimos un punto en el plano latente y lo pasamos solo por el **decoder**, obtenemos una imagen. Esto es **proto-generación**.""")

code("""# Generar una grilla 5x5 de puntos en el latente y decodificar cada uno
grid = np.linspace(-3, 3, 5)
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for i, y in enumerate(grid):
    for j, x in enumerate(grid):
        z = np.array([[x, y]])
        img = decoder_2d.predict(z, verbose=0).reshape(28, 28)
        axes[i, j].imshow(img, cmap="gray")
        axes[i, j].axis("off")
plt.suptitle("Paseo por el espacio latente — cada celda es un punto (z1, z2) decodificado",
              fontweight="bold")
plt.tight_layout()
plt.show()""")

md("""**Limitación de los AE clásicos para generar:** los puntos en el latente que no corresponden a datos reales producen imágenes **borrosas o sin sentido**. Para generar de verdad necesitamos un **VAE** (Variational Autoencoder), que regulariza el latente — eso se ve en Módulo 5 más adelante.""")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Conv Autoencoder
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 9. Conv Autoencoder — Código Directo del Blog Oficial de Keras

Usamos la **arquitectura exacta del blog oficial de Keras** (`blog.keras.io/building-autoencoders-in-keras.html`). Está probada, converge, y tiene una ventaja para nuestra clase: el bottleneck es espacial `(4, 4, 8)` — podemos **ver la imagen comprimida** como una imagen pequeña.

**El truco pedagógico clave:** con un Dense AE el latente es 1D (un vector de 32 números) — difícil de ver. Con este Conv AE el latente es una mini-imagen de `4×4×8` — la "compresión" se VE.""")

code("""# Reorganizar MNIST a forma 4D para Conv2D: (N, 28, 28, 1)
X_train_img = X_train[..., None]
X_test_img  = X_test[..., None]
print(f"X_train_img: {X_train_img.shape}")""")

md("""### 9.1 Construir el Conv AE con Functional API

Sigue el mismo patrón que el Dense AE: declarar Input, pasar por capas como funciones, envolver en `Model`. La diferencia es solo que ahora las capas son `Conv2D` + `MaxPooling2D` / `UpSampling2D`.""")

code("""# === Arquitectura del blog oficial de Keras ===
# Encoder: 28x28x1 → 14x14x16 → 7x7x8 → 4x4x8 (bottleneck espacial)
input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
# encoded shape: (4, 4, 8) = 128 valores
# Compresión: 784 → 128 (6.1x más chico)

# Decoder: 4x4x8 → 8x8x8 → 16x16x8 → 28x28x16 → 28x28x1
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)  # SIN padding -> 24x24
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Tres modelos compartiendo las mismas capas
conv_ae      = keras.Model(input_img, decoded,  name="conv_autoencoder")
conv_encoder = keras.Model(input_img, encoded,  name="conv_encoder")

conv_ae.compile(optimizer='adam', loss='binary_crossentropy')
conv_ae.summary()""")

md("""**Cosas importantes:**

- El bottleneck `encoded` es **espacial**: 4×4 espacial × 8 canales = 128 valores. Mucho más chico que los 784 del input.
- `conv_encoder` es un modelo separado que solo va del input al bottleneck. Lo vamos a usar para visualizar la imagen comprimida.
- Loss `binary_crossentropy` con sigmoid final — receta verificada del blog de Keras.""")

code("""# Entrenar (15-20 epochs, ~1-2 min en Colab T4)
conv_ae.fit(X_train_img, X_train_img,
             epochs=15, batch_size=128,
             validation_data=(X_test_img, X_test_img), verbose=2)""")

md("""### 9.2 LA visualización clave — ver la imagen comprimida

Esto es lo que el blog oficial de Keras llama "**how good our autoencoder is at compressing**". Tomamos varios dígitos, los pasamos por el encoder, y mostramos:

1. La imagen original (28×28 = 784 pixels).
2. **La imagen comprimida** — el latente `(4, 4, 8)` desplegado como una imagen de `4×32` para que sea visible.
3. La reconstrucción (28×28).

Es la única visualización donde se **ve** literalmente qué tan chica es la versión comprimida.""")

code("""# Tomar 10 ejemplos fijos del test set
N_VIS = 10
X_vis = X_test_img[:N_VIS]
y_vis = y_test[:N_VIS]

# Pasar por encoder (para tener el comprimido) y por el AE entero (para reconstruir)
comprimidos    = conv_encoder.predict(X_vis, verbose=0)    # (10, 4, 4, 8)
reconstruidos  = conv_ae.predict(X_vis, verbose=0)         # (10, 28, 28, 1)

print(f"Comprimido shape: {comprimidos.shape}")
print(f"Cada imagen pasó de 784 valores a 4x4x8 = {4*4*8} valores")""")

code("""# La visualización del blog de Keras: mostramos el comprimido reshape a (4, 32)
# Se ve como una imagen pequeñita -- la compresión es VISUAL
fig, axes = plt.subplots(3, N_VIS, figsize=(18, 5))
for i in range(N_VIS):
    # Original
    axes[0, i].imshow(X_vis[i].squeeze(), cmap='gray')
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    axes[0, i].set_title(f"{y_vis[i]}", fontsize=9, color='gray')

    # Comprimido: (4,4,8) -> reshape (4, 32) para verlo como imagen
    # Apilamos los 8 canales horizontalmente
    comp_img = comprimidos[i].reshape((4, 4 * 8))
    axes[1, i].imshow(comp_img, cmap='viridis', aspect='auto')
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

    # Reconstrucción
    axes[2, i].imshow(reconstruidos[i].squeeze(), cmap='gray')
    axes[2, i].set_xticks([])
    axes[2, i].set_yticks([])

# Etiquetas a la izquierda
axes[0, 0].set_ylabel('Original\\n28×28 = 784 px',
                       fontweight='bold', color='#6B1525', fontsize=10)
axes[1, 0].set_ylabel('Comprimido\\n4×4×8 = 128 valores',
                       fontweight='bold', color='#7C3AED', fontsize=10)
axes[2, 0].set_ylabel('Reconstrucción\\n28×28 = 784 px',
                       fontweight='bold', color='#6B1525', fontsize=10)

plt.suptitle('Aquí se VE la compresión: imagen original → versión comprimida → reconstrucción',
              fontweight='bold', fontsize=13, color='#6B1525')
plt.tight_layout()
plt.show()""")

md("""**Lo que tenemos que destacar:**

- La fila del medio **ES** la imagen comprimida. Cada dígito tiene su patrón único en el latente espacial.
- Pasamos de 784 valores por imagen a **128 valores** (compresión 6.1×) y la reconstrucción sigue siendo claramente identificable.
- Esa fila del medio es lo que se almacena/transmite en una aplicación real (búsqueda visual, recomendadores, compresión de imágenes). El decoder solo se usa cuando se necesita volver al input original.""")

md("""### 9.3 Comparación Dense AE vs Conv AE

Mismas 10 imágenes pasadas por los dos AEs entrenados. Comparamos la calidad de reconstrucción.""")

code("""# Indices locales (no depender de un idxs global de otra celda)
idxs_comp = np.arange(10)
originales_comp = X_test[idxs_comp]                            # (10, 28, 28)
recons_dense    = autoencoder.predict(X_test_flat[idxs_comp],
                                       verbose=0).reshape(-1, 28, 28)
recons_conv     = conv_ae.predict(X_test_img[idxs_comp],
                                   verbose=0).squeeze()

fig, axes = plt.subplots(3, 10, figsize=(15, 4.5))
for i in range(10):
    axes[0, i].imshow(originales_comp[i], cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(recons_dense[i], cmap="gray")
    axes[1, i].axis("off")
    axes[2, i].imshow(recons_conv[i], cmap="gray")
    axes[2, i].axis("off")

axes[0, 0].set_title("Original", loc="left", fontweight="bold", fontsize=11)
axes[1, 0].set_title("Dense AE\\n(latente 32)", loc="left",
                      fontweight="bold", fontsize=10, color="#C82B40")
axes[2, 0].set_title("Conv AE\\n(latente 4×4×8=128)", loc="left",
                      fontweight="bold", fontsize=10, color="#7C3AED")
plt.tight_layout()
plt.show()""")

md("""**Observación:** las reconstrucciones del **Conv AE son más nítidas**. La razón: el encoder convolucional aprovecha la estructura 2D (filtros que se comparten en todas las posiciones), igual que las CNN supervisadas de clase 24.""")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Aplicación 1: Compresión y Pretraining
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 10. Aplicación 1 — Compresión y Pretraining

La aplicación más directa del autoencoder: usar el **encoder entrenado como extractor de features** para otra tarea. El espacio latente que aprendió por reconstrucción funciona como representación comprimida que otro modelo puede consumir.

### 10.1 ¿Por qué sirve esto?

El encoder vio millones de imágenes y aprendió **qué es informativo en ellas**. Si después tenemos un problema supervisado con **pocos labels** (caro etiquetar), el encoder ya hace la mitad del trabajo: convierte la imagen en 32 números útiles. Solo nos queda entrenar un clasificador chiquito sobre esos 32 números.

| | Sin pretraining | Con pretraining |
|---|---|---|
| Input al clasificador | 784 píxeles crudos | 32 features del encoder |
| Parámetros del clasificador | Muchos (entrada grande) | Pocos (entrada compacta) |
| Datos labeled necesarios | Miles | Cientos pueden bastar |
| Datos sin label aprovechados | 0 | Todos los que tengas |

### 10.2 Demo — Búsqueda por similitud usando el latente

El caso de uso más directo del encoder en producción: **búsqueda por similitud**. Dada una imagen-query, encontrar las más parecidas en una base de datos. La idea: comparar 128 valores (latente Conv) en lugar de 784 píxeles es ~6× más rápido por comparación, y el resultado suele ser **mejor** porque el latente captura forma, no diferencias pixel-a-pixel.""")

code("""from sklearn.metrics.pairwise import euclidean_distances

# Tomamos un subset del test set como "base de datos"
N_DB = 3000
X_db_img  = X_test_img[:N_DB]
X_db_flat = X_test_flat[:N_DB]
y_db      = y_test[:N_DB]

# Comprimimos toda la base con el encoder convolucional
db_latente = conv_encoder.predict(X_db_img, verbose=0)        # (3000, 4, 4, 8)
db_latente = db_latente.reshape(N_DB, -1)                      # (3000, 128)
print(f"Base de datos comprimida: {X_db_flat.shape} → {db_latente.shape}")
print(f"Storage por imagen: 784 floats → 128 floats  (compresión 6.1x)")""")

code("""# Elegimos un query del test (uno que NO esté en la 'base de datos')
query_idx = 3001
query_img    = X_test[query_idx]
query_pixel  = X_test_flat[query_idx:query_idx+1]
query_latent = conv_encoder.predict(X_test_img[query_idx:query_idx+1],
                                      verbose=0).reshape(1, -1)

# Vecinos en píxel (sin AE) y en latente
dist_pixel = euclidean_distances(query_pixel,  X_db_flat).flatten()
dist_lat   = euclidean_distances(query_latent, db_latente).flatten()
nn_pixel = np.argsort(dist_pixel)[:5]
nn_lat   = np.argsort(dist_lat)[:5]

print(f"Query: dígito {y_test[query_idx]}")
print(f"Vecinos en pixel: dígitos {y_db[nn_pixel].tolist()}")
print(f"Vecinos en latente: dígitos {y_db[nn_lat].tolist()}")""")

code("""# Visualizar query + 5 vecinos en cada espacio
fig, axes = plt.subplots(2, 6, figsize=(13, 4.5))

# Query (mismo en ambas filas)
for fila in (0, 1):
    axes[fila, 0].imshow(query_img, cmap='gray')
    axes[fila, 0].set_title(f"QUERY\\n(dígito {y_test[query_idx]})",
                              fontweight='bold', fontsize=10, color='#C82B40')
    axes[fila, 0].axis('off')

# Vecinos en pixel space
for i, idx in enumerate(nn_pixel):
    axes[0, i+1].imshow(X_test[idx], cmap='gray')
    axes[0, i+1].set_title(f"#{i+1}  dígito {y_db[idx]}", fontsize=10)
    axes[0, i+1].axis('off')

# Vecinos en latent space
for i, idx in enumerate(nn_lat):
    axes[1, i+1].imshow(X_test[idx], cmap='gray')
    axes[1, i+1].set_title(f"#{i+1}  dígito {y_db[idx]}", fontsize=10)
    axes[1, i+1].axis('off')

# Etiquetas de fila
fig.text(0.05, 0.72, "Píxeles\\n(784 dim)", fontweight='bold',
          fontsize=11, color='gray', ha='center')
fig.text(0.05, 0.28, "Latente\\n(128 dim)", fontweight='bold',
          fontsize=11, color='#7C3AED', ha='center')

plt.suptitle("Búsqueda por similitud — 6× menos almacenamiento, mismo recall semántico",
              fontweight='bold')
plt.tight_layout(rect=[0.08, 0, 1, 0.95])
plt.show()""")

md("""**Lo que vemos:** ambas búsquedas suelen encontrar dígitos parecidos al query. La diferencia operativa: **comparar vectores de 128 dim es 6× más rápido y usa 6× menos memoria** que comparar imágenes crudas. En producción, con millones de imágenes, esa diferencia es la diferencia entre que una búsqueda demore 100 ms o 600 ms.

### 10.3 Otros usos del encoder en producción

| Caso | Cómo se usa el encoder |
|------|-------------------------|
| Buscar productos similares a una foto | Una tienda con millones de productos. Comprimes cada foto a un vector chico y comparas vectores en lugar de imágenes. Bibliotecas: FAISS, Qdrant. |
| Sistemas de recomendación | Entrenas un AE sobre la matriz "qué usuario vio qué película" (Netflix). El encoder produce las dimensiones de gusto de cada usuario. |
| Compresión de imágenes con redes | JPEG-AI: encoder reduce la imagen a pocos números que se transmiten, decoder reconstruye al otro lado. |
| Reducir vectores grandes para almacenar | Modelos de lenguaje producen vectores de 768 valores por texto. Con cientos de millones de textos, almacenar tanto es caro. Un AE comprime esos vectores a 64 manteniendo qué tan parecidos son entre sí. |""")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Aplicación 2: Denoising
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 11. Aplicación 2 — Denoising

**Variación del entrenamiento:** alimentamos el encoder con datos degradados pero usamos los datos limpios como target de reconstrucción. El espacio latente queda parametrizado para describir solo la señal estructurada — la degradación no encuentra cómo proyectarse en él.

```
   X + ruido    →    AE    →    X (limpio)
```""")

code("""# Generar versiones ruidosas
NOISE = 0.4
rng = np.random.RandomState(42)

X_train_noisy = X_train_img + NOISE * rng.normal(size=X_train_img.shape)
X_test_noisy  = X_test_img  + NOISE * rng.normal(size=X_test_img.shape)
X_train_noisy = np.clip(X_train_noisy, 0, 1)
X_test_noisy  = np.clip(X_test_noisy,  0, 1)

# Mostrar antes/después de meterle ruido
fig, axes = plt.subplots(2, 8, figsize=(13, 3.5))
for i in range(8):
    axes[0, i].imshow(X_train_img[i].squeeze(), cmap="gray"); axes[0, i].axis("off")
    axes[1, i].imshow(X_train_noisy[i].squeeze(), cmap="gray"); axes[1, i].axis("off")
axes[0, 0].set_title("Limpio (target)", loc="left", fontweight="bold")
axes[1, 0].set_title(f"Con ruido (input, σ={NOISE})", loc="left", fontweight="bold")
plt.tight_layout()
plt.show()""")

code("""# Denoising AE: arquitectura del blog oficial de Keras (verificada que converge)
# 32 filtros + binary_crossentropy + Functional API
input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

denoising_ae = keras.Model(input_img, decoded, name="denoising_autoencoder")
denoising_ae.compile(optimizer='adam', loss='binary_crossentropy')

# Aquí está el truco: input=ruidoso, target=limpio
denoising_ae.fit(X_train_noisy, X_train_img,
                  epochs=10, batch_size=128,
                  validation_data=(X_test_noisy, X_test_img), verbose=2)""")

code("""# Resultado: input ruidoso → reconstrucción limpia (10 ejemplos fijos del test)
idxs_den = np.arange(10)
denoised = denoising_ae.predict(X_test_noisy[idxs_den], verbose=0).squeeze()

fig, axes = plt.subplots(3, 10, figsize=(15, 4.5))
for i in range(10):
    axes[0, i].imshow(X_test_img[idxs_den[i]].squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(X_test_noisy[idxs_den[i]].squeeze(), cmap="gray")
    axes[1, i].axis("off")
    axes[2, i].imshow(denoised[i], cmap="gray")
    axes[2, i].axis("off")

axes[0, 0].set_title("Limpio (verdad)", loc="left", fontweight="bold", fontsize=10)
axes[1, 0].set_title("Con ruido (input)", loc="left", fontweight="bold", fontsize=10)
axes[2, 0].set_title("Denoised (output)", loc="left", fontweight="bold", fontsize=10)
plt.tight_layout()
plt.show()""")

md("""### Por qué funciona

El ruido gaussiano es **estructuralmente incompresible**: cada píxel se sortea independiente, no hay redundancia entre vecinos. El dígito subyacente SÍ es comprimible — está hecho de trazos coherentes, regiones cerradas y simetrías locales.

El espacio latente del autoencoder tiene 32 dimensiones. Durante el entrenamiento esas dimensiones se ajustan para describir lo que en los datos resulta comprimible. El ruido no llega al latente porque no hay coordenadas que lo representen. El decoder reconstruye desde el latente y devuelve solo la componente estructurada.

> **Generalización del principio:** un denoising AE no "limpia ruido" — reconstruye desde un latente que solo aprendió señales estructuradas. Lo que no encaja en el latente queda fuera. Aplica a ruido gaussiano, oclusiones parciales, manchas, artefactos JPEG — siempre que el AE haya visto el patrón de degradación durante el entrenamiento.

### Consideración operativa

El AE aprende a remover el **patrón de degradación específico** que vio en entrenamiento. Si entrenas con ruido gaussiano, no remueve ruido tipo sal-y-pimienta. En producción la práctica estándar es **generar degradaciones sintéticas que imiten las reales** y entrenar con esos pares (sucio, limpio).""")


md("""### 11.5 Denoising en datos no-imagen

El mismo principio aplica a series temporales (sensores industriales, ECG, audio), a datos tabulares con ruido de captura, etc. La receta no cambia: input degradado → target limpio, arquitectura encoder-decoder adaptada al tipo de dato (`Conv1D` para series, `Dense` para tabular).

### 11.6 Cuándo Sí y Cuándo NO Usar Denoising

| Condición | ¿Aplica denoising AE? |
|-----------|------------------------|
| Ruido y señal son estructuralmente distintos (uno comprime, el otro no) | ✓ Sí |
| Puedes simular el patrón de degradación con razonable fidelidad | ✓ Sí |
| El task downstream se beneficia de input limpio (OCR, clasificación, forecasting) | ✓ Sí |
| El ruido es **variabilidad legítima** del proceso que importa para el negocio | ✗ No — lo estarías borrando |
| Métodos clásicos resuelven (mediana, gaussiano, EMA, Savitzky-Golay) | ✗ No — usá lo clásico, es más simple y barato |
| El tipo de degradación cambia con el tiempo y no puedes re-entrenar | ✗ No — modelo va a degradar en producción |
| Solo tienes ejemplos sucios, no pares (sucio, limpio) | ✗ No — necesitas el target limpio para supervisar |

> **Regla pragmática:** intenta primero con un filtro clásico. Si no alcanza, sube a denoising AE. Si lo subes, asegúrate de tener un set de validación con degradaciones realistas — no del set de entrenamiento.""")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Aplicación 3: Detección de Anomalías con AE (a profundidad sobre ECG5000)
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 12. Aplicación 3 — Detección de Anomalías a Profundidad

Es el caso de uso más rentable de los autoencoders en la industria. Lo vemos completo con un dataset real: **ECG5000** (PhysioNet / UCR), 5.000 latidos cardíacos reales medidos en pacientes, etiquetados como **normales** (sinusales) o **anómalos** (arritmias). Es el dataset que usa el tutorial oficial de TensorFlow para enseñar anomaly detection con AE.

### 12.1 La idea, formulada con precisión

Entrenamos el AE **únicamente con datos del régimen normal**. El espacio latente que resulta es un sistema de coordenadas **diseñado para describir esa población — y solo esa.**

Cuando llega una observación nueva:
- Si pertenece al régimen normal, el latente la representa bien y el decoder reconstruye con bajo error.
- Si NO pertenece, las coordenadas latentes no fueron parametrizadas para describirla. La reconstrucción degrada y el error crece.

> **Reconstruction error alto = el espacio latente no tiene cómo describir esta observación = anomalía.**

### 12.2 Qué es exactamente el "error de reconstrucción"

Para una observación $x$ y su reconstrucción $\hat{x} = \text{decoder}(\text{encoder}(x))$, dos opciones estándar:

| Métrica | Fórmula | Cuándo usar |
|---------|---------|-------------|
| **MAE** (mean absolute error) | $\frac{1}{n}\sum_i |x_i - \hat{x}_i|$ | Default robusto. Lo usa el tutorial oficial de TF para ECG5000. Penaliza errores grandes y pequeños proporcionalmente. |
| **MSE** (mean squared error) | $\frac{1}{n}\sum_i (x_i - \hat{x}_i)^2$ | Penaliza fuerte los errores grandes (cuadrático). Más sensible a outliers. |
| **SSIM** | métrica perceptual | Solo para imágenes cuando MSE/MAE no capturan defectos localizados. |

Aquí usamos **MAE** para mantenernos alineados con el tutorial oficial. Ese mismo error sirve como **score de anomalía** — un número por cada observación.

### 12.3 Por qué ECG5000 es el dataset estándar

| Característica | Por qué importa |
|----------------|-----------------|
| Real, de pacientes reales | No es sintético. Las anomalías son arritmias clínicas, no perturbaciones inventadas. |
| Público y descarga directa | Google hostea el CSV (sin Kaggle, sin auth). ~5 MB. |
| Etiquetas binarias incluidas | Permite EVALUAR el detector (precision/recall/F1) — lo que en producción casi nunca tenés. |
| Tabular (cada latido es un vector de 140 valores) | Vale como template para CUALQUIER caso tabular: fraude, mantenimiento predictivo, telemetría. |
| Tutorial oficial documentado | `tensorflow.org/tutorials/generative/autoencoder` usa este mismo dataset. Receta verificada. |""")

code("""# Cargar ECG5000 directamente desde el mirror de Google
import pandas as pd

URL_ECG = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
df = pd.read_csv(URL_ECG, header=None)

# Última columna = etiqueta. 1 = latido normal, 0 = anómalo (arrítmico)
labels  = df.values[:, -1].astype(int)
signals = df.values[:, :-1].astype("float32")    # 140 valores por latido

# Normalizar [0, 1] por dataset entero
vmin, vmax = signals.min(), signals.max()
signals = (signals - vmin) / (vmax - vmin)

print(f"Dataset: {signals.shape}")
print(f"Latidos normales:  {(labels == 1).sum()}")
print(f"Latidos anómalos:  {(labels == 0).sum()}")""")

code("""# Visualizar 3 latidos normales y 3 anómalos para ver con qué trabajamos
fig, axes = plt.subplots(2, 3, figsize=(13, 5), sharey=True)
idx_n = np.where(labels == 1)[0][:3]
idx_a = np.where(labels == 0)[0][:3]

for i, idx in enumerate(idx_n):
    axes[0, i].plot(signals[idx], color="#16A34A", lw=1.5)
    axes[0, i].set_title(f"Normal #{idx}", fontweight="bold", color="#16A34A")
    axes[0, i].grid(alpha=0.3)
for i, idx in enumerate(idx_a):
    axes[1, i].plot(signals[idx], color="#C82B40", lw=1.5)
    axes[1, i].set_title(f"Anómalo #{idx}", fontweight="bold", color="#C82B40")
    axes[1, i].grid(alpha=0.3)

plt.suptitle("ECG5000: forma real de latidos (normales arriba, anómalos abajo)",
              fontweight="bold")
plt.tight_layout()
plt.show()""")

md("""**Observen** las diferencias morfológicas: los normales tienen un patrón QRS bien definido y de amplitud regular; los anómalos rompen la forma — pueden tener picos donde no debería haber, ausencia del complejo, o ritmo claramente alterado. **Esa es la señal que el AE va a aprovechar.**

### 12.4 Split de datos — la decisión crítica

El AE solo se entrena con normales. Los anómalos quedan reservados para evaluar.""")

code("""# Split: el AE solo ve normales
X_normal = signals[labels == 1]    # ~2900 latidos normales
X_anom   = signals[labels == 0]    # ~2100 latidos anómalos

# 85% de normales para entrenar; 15% normales + TODOS los anómalos para test
n_tr = int(0.85 * len(X_normal))
X_tr_normal = X_normal[:n_tr]              # train
X_te_normal = X_normal[n_tr:]              # test normales
X_te_anom   = X_anom                       # test anómalos

print(f"Train (solo normales):   {X_tr_normal.shape}")
print(f"Test normales:           {X_te_normal.shape}")
print(f"Test anómalos:           {X_te_anom.shape}")""")

md("""### 12.5 Arquitectura — del tutorial oficial de TensorFlow

Dense AE con bottleneck=8 y loss MAE. Es la arquitectura del tutorial oficial.""")

code("""# Functional API + arquitectura del tutorial oficial de TF
input_ecg = keras.Input(shape=(140,))
x = layers.Dense(32, activation="relu")(input_ecg)
x = layers.Dense(16, activation="relu")(x)
bottleneck = layers.Dense(8, activation="relu")(x)        # bottleneck

y = layers.Dense(16, activation="relu")(bottleneck)
y = layers.Dense(32, activation="relu")(y)
out_ecg = layers.Dense(140, activation="sigmoid")(y)

anom_ae = keras.Model(input_ecg, out_ecg, name="ecg_anomaly_ae")
anom_ae.compile(optimizer="adam", loss="mae")

history = anom_ae.fit(X_tr_normal, X_tr_normal,
                       epochs=30, batch_size=64,
                       validation_data=(X_te_normal, X_te_normal),
                       verbose=2)""")

md("""### 12.6 Calcular el error de reconstrucción

Aplicamos el AE a las dos poblaciones y calculamos el MAE por muestra — ESE es el score de anomalía.""")

code("""# Pasar normales y anómalos por el AE
recons_n = anom_ae.predict(X_te_normal, verbose=0)
recons_a = anom_ae.predict(X_te_anom,   verbose=0)

# Error por muestra (MAE): un número por cada latido
err_normal = np.mean(np.abs(X_te_normal - recons_n), axis=1)
err_anom   = np.mean(np.abs(X_te_anom   - recons_a), axis=1)

print(f"Error promedio NORMALES:  {err_normal.mean():.4f}  (std {err_normal.std():.4f})")
print(f"Error promedio ANÓMALOS:  {err_anom.mean():.4f}  (std {err_anom.std():.4f})")
print(f"Ratio anomalía/normal:    {err_anom.mean() / err_normal.mean():.2f}×")""")

md("""### 12.7 Lo más importante: el histograma de errores

Toda la información operativa del detector está en esta figura. Cómo se lee:

- **Si las dos distribuciones están bien separadas**, el detector va a funcionar bien con casi cualquier threshold razonable.
- **Si se solapan mucho**, vas a tener trade-off precision/recall serio — hay que decidir qué error de los dos cuesta más.
- **La forma de las colas importa**: una anomalía sutil cae en el solape. Una anomalía severa está bien a la derecha.""")

code("""fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(err_normal, bins=50, alpha=0.7, color="#16A34A",
         label=f"Normales (n={len(err_normal)})", density=True)
ax.hist(err_anom, bins=50, alpha=0.7, color="#C82B40",
         label=f"Anómalos (n={len(err_anom)})", density=True)
ax.set_xlabel("Reconstruction error (MAE)", fontweight="bold")
ax.set_ylabel("Densidad")
ax.set_title("Distribución del error de reconstrucción — el separador es esto",
              fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()""")

md("""### 12.8 Elegir el threshold — tres estrategias estándar

El threshold convierte el score continuo en una decisión binaria (normal / anómalo). Las tres formas más usadas:

| Estrategia | Cálculo | Cuándo usarla |
|------------|---------|----------------|
| **Percentil de normales** | `np.percentile(err_normal, 95)` | Default. Acepta un 5% de falsos positivos en normales conocidos. |
| **Media + k·desviación** | `err_normal.mean() + k * err_normal.std()` | Receta del tutorial oficial de TF (k=1). Asume distribución aproximadamente normal de errores. |
| **Optimizar F1 sobre validación** | búsqueda sobre umbrales que maximiza F1 con un set de validación etiquetado | Cuando tenés algunas anomalías etiquetadas y querés balancear precision/recall óptimamente. |

Las tres dan resultados ligeramente distintos. Lo importante: **el threshold es una palanca de negocio**, no un hiperparámetro escondido. Subes → menos falsos positivos, más anomalías que se escapan. Bajas → más falsos positivos, menos anomalías perdidas.""")

code("""# Calcular las tres estrategias
th_p95     = np.percentile(err_normal, 95)
th_meanstd = err_normal.mean() + err_normal.std()   # k=1 (tutorial TF)
th_max     = err_normal.max()                       # extremo: ningún FP

print(f"Threshold p95:        {th_p95:.4f}")
print(f"Threshold mean + std: {th_meanstd:.4f}")
print(f"Threshold max:        {th_max:.4f}")

# Vamos a usar la estrategia del tutorial oficial: mean + std
threshold = th_meanstd
print(f"\nUsamos threshold = {threshold:.4f}  (mean + std, tutorial TF)")""")

md("""### 12.9 Evaluar como clasificador binario — todas las métricas

Como tenemos las etiquetas verdaderas (anómalo / normal), podemos calcular todas las métricas de clasificación estándar.""")

code("""from sklearn.metrics import (confusion_matrix, classification_report,
                                   precision_recall_curve, roc_curve, auc)

# Concatenar normales y anómalos del test con etiqueta binaria (1 = anómalo)
errores_test = np.concatenate([err_normal, err_anom])
y_verdad     = np.concatenate([np.zeros(len(err_normal), dtype=int),
                                 np.ones(len(err_anom),   dtype=int)])

# Predicciones con el threshold elegido
y_pred = (errores_test > threshold).astype(int)

# Matriz de confusión
cm = confusion_matrix(y_verdad, y_pred)
print("Matriz de confusión:")
print(f"                  Pred. Normal | Pred. Anómalo")
print(f"  Real Normal         {cm[0,0]:5d}    |     {cm[0,1]:5d}")
print(f"  Real Anómalo        {cm[1,0]:5d}    |     {cm[1,1]:5d}")
print()
print(classification_report(y_verdad, y_pred,
                              target_names=["Normal", "Anómalo"], digits=3))""")

md("""**Cómo interpretar cada métrica en producción:**

- **Precision (anómalo)**: de cada 100 alarmas que dispara el sistema, cuántas son anomalías reales. Bajo = muchas falsas alarmas → cansancio del operador.
- **Recall (anómalo)**: de cada 100 anomalías reales, cuántas detecta el sistema. Bajo = anomalías que se escapan → riesgo de falla en producción.
- **F1**: balance entre las dos. Útil para reportar un solo número.
- **Soporte**: tamaño de cada clase en el test.

> En ECG: un Recall alto suele ser más importante (no querés que un paciente con arritmia pase desapercibido). En fraude bancario: la Precision puede pesar más (no querés bloquear clientes legítimos).""")

md("""### 12.10 La curva ROC — métrica independiente del threshold

La curva ROC muestra el trade-off **para todos los thresholds posibles**. El AUC (área bajo la curva) resume el poder discriminativo del detector en un solo número, sin tener que elegir threshold.""")

code("""# Curva ROC y AUC
fpr, tpr, ths_roc = roc_curve(y_verdad, errores_test)
roc_auc = auc(fpr, tpr)

# Curva Precision-Recall (más útil cuando hay desbalance de clases)
prec, rec, ths_pr = precision_recall_curve(y_verdad, errores_test)
pr_auc = auc(rec, prec)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(fpr, tpr, color="#C82B40", lw=2, label=f"AUC = {roc_auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1, label="azar")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].set_title("Curva ROC", fontweight="bold")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

axes[1].plot(rec, prec, color="#7C3AED", lw=2, label=f"AP = {pr_auc:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Curva Precision-Recall", fontweight="bold")
axes[1].legend(loc="lower left")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()""")

md("""**Cómo leer estas curvas:**

- **ROC AUC**: 0.5 = azar; 1.0 = perfecto. Para anomaly detection con AE sobre ECG5000 suele rondar **0.95+**. Por encima de 0.85 es un detector útil en producción.
- **PR-AUC** (precision-recall): más informativa cuando las anomalías son raras. Si en producción esperás 1% de anomalías, la curva PR te dice qué pasará con tu precision al detectarlas.
- Si necesitás **más recall** (no perder anomalías), bajas el threshold → te mueves a la derecha en la curva PR (más recall, menos precision).
- Si necesitás **más precision** (menos falsas alarmas), subes el threshold.

### 12.11 Casos concretos — los más anómalos vs los más normales""")

code("""# Visualizar 3 casos: anómalo detectado, normal correcto, falso positivo
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

idx_a = int(np.argmax(err_anom))
axes[0].plot(X_te_anom[idx_a],   color="gray",   lw=1.5, label="Real")
axes[0].plot(recons_a[idx_a],    color="#C82B40", lw=1.5, label="Reconstrucción")
axes[0].set_title(f"Anómalo bien detectado\nerror={err_anom[idx_a]:.3f}",
                   color="#C82B40", fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

idx_n = int(np.argmin(err_normal))
axes[1].plot(X_te_normal[idx_n], color="gray",   lw=1.5, label="Real")
axes[1].plot(recons_n[idx_n],    color="#16A34A", lw=1.5, label="Reconstrucción")
axes[1].set_title(f"Normal bien reconstruido\nerror={err_normal[idx_n]:.3f}",
                   color="#16A34A", fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

idx_fp = int(np.argmax(err_normal))
axes[2].plot(X_te_normal[idx_fp], color="gray",   lw=1.5, label="Real")
axes[2].plot(recons_n[idx_fp],    color="#EA580C", lw=1.5, label="Reconstrucción")
axes[2].set_title(f"Normal con error alto\n(falso positivo: error={err_normal[idx_fp]:.3f})",
                   color="#EA580C", fontweight="bold", fontsize=10)
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)

plt.suptitle("Tres casos típicos sobre ECG5000", fontweight="bold")
plt.tight_layout()
plt.show()""")

md("""### 12.12 Pipeline completo en producción""")

code('''def es_anomalia(x, modelo, threshold):
    """Pipeline de inferencia: dada una observacion nueva, devolver
    (es_anomalia, score_error_reconstruccion)."""
    if x.ndim == 1:
        x = x[None, :]
    x_rec = modelo.predict(x, verbose=0)
    error = float(np.mean(np.abs(x - x_rec), axis=1).item())
    return (error > threshold), error

# Demo en un latido al azar
demo_idx = 5
es_anom, score = es_anomalia(X_te_anom[demo_idx], anom_ae, threshold)
print(f"Score del latido: {score:.4f}")
print(f"Threshold:         {threshold:.4f}")
print(f"Veredicto:         {'ANOMALIA' if es_anom else 'normal'}")''')

md("""### 12.13 Consideraciones críticas en producción

| Punto | Recomendación |
|-------|---------------|
| **El set "normal" debe ser realmente normal** | Si se cuelan anomalías sin etiquetar, el AE las aprende como variantes normales y deja de detectarlas. Limpiar primero con reglas de negocio o un primer filtro con IsolationForest. |
| **Múltiples regímenes normales legítimos** | Si la operación tiene varios "modos normales" (ej. máquina arrancando vs en régimen), el AE necesita ver muestras de TODOS. Un modo no observado en train se ve como anomalía en producción. |
| **Escalas distintas entre features** | Normalizar (`StandardScaler` o `MinMaxScaler`) antes de entrenar. Sin escalar, las features de mayor magnitud dominan la loss y los errores. |
| **Variables categóricas** | One-hot encoding primero. AE asume features numéricas. |
| **Bottleneck demasiado grande** | El AE aprende la identidad y deja de detectar. Mantener entre 1/5 y 1/20 del input. |
| **Drift del régimen normal** | El comportamiento "normal" cambia con el tiempo (estaciones, productos nuevos). El AE necesita re-entrenamiento periódico — definir el ciclo desde el inicio. |
| **Calibrar el threshold con datos de producción** | El threshold p95 sobre el set de train suele ser conservador. Recalibrar con datos reales de producción mejora el detector. |

### 12.14 Cuándo NO usar AE para anomaly detection

- Cuando tenés **muchos ejemplos etiquetados de anomalías** → clasificación supervisada gana siempre.
- Cuando las anomalías son **del mismo régimen que lo normal** (diferencias muy sutiles) → probar primero `IsolationForest` o `OneClassSVM`. Suelen ser mejores en tabular y son más interpretables.
- Cuando necesitás **explicar qué hace anómala a una muestra** → AE da un score pero no el por qué. Métodos basados en árboles (Isolation Forest) dan importancia de features.""")


# ─────────────────────────────────────────────────────────────────────────────
# 11. U-Net y conexión con segmentación
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 13. Aplicación 4 — U-Net: Segmentación Como Encoder-Decoder

### Conexión con clase 28

Esta mañana entrenamos `yolo26n-seg` para contar píldoras. El head de segmentación de YOLO es una variante de **U-Net** — y U-Net es, estructuralmente, un autoencoder convolucional con **conexiones laterales** (skip connections). La arquitectura encoder-decoder que llevamos toda la clase montando es la misma idea base.

### Qué es U-Net

```
INPUT 256x256x3
   │
   ├─ Conv + Pool ─→  128x128x64  ──────────────┐
   │                                              skip
   ├─ Conv + Pool ─→   64x64x128 ────────────┐   │
   │                                          skip │
   ├─ Conv + Pool ─→   32x32x256 ────────┐   │   │
   │                                      skip│   │
   ├─ Conv + Pool ─→   16x16x512  (bottleneck)│   │
   │                                       │   │   │
   ├─ UpSampling ─→    32x32x256 ←─concat──┘   │   │
   │                                            │   │
   ├─ UpSampling ─→    64x64x128 ←─concat──────┘   │
   │                                                │
   ├─ UpSampling ─→   128x128x64 ←─concat──────────┘
   │
   └─ Conv 1x1 ─→ 256x256xC (máscara segmentación, C clases)
```

**Las dos diferencias con un AE clásico:**

1. **Output no es la imagen reconstruida**, es la **máscara de segmentación** (pixel-level classification).
2. **Skip connections**: el decoder no solo recibe el latente, recibe también las **feature maps del encoder** del mismo nivel. Esto resuelve un problema: al comprimir tanto, el AE puro pierde los detalles finos (bordes exactos del objeto). Las skip connections los **traen de vuelta**.

### Por qué el AE puro no sirve para segmentación

Si entrenas un AE estándar con `target = máscara`, funciona pero da **máscaras borrosas**. La compresión a través del bottleneck pierde la **resolución espacial**. Las skip connections preservan esa resolución porque "saltean" el bottleneck.

### Tabla resumen: la familia AE → segmentación

| Arquitectura | Output | Skip connections | Caso de uso |
|--------------|--------|-------------------|-------------|
| **AE clásico** | Reconstrucción (input ≈ output) | No | Compresión, anomalía, denoising |
| **U-Net** | Máscara pixel-wise | Sí | Segmentación semántica (médica, satelital) |
| **FPN / PAN** | Múltiples escalas | Sí (multi-nivel) | YOLO, detección moderna |
| **VAE** | Reconstrucción + latente regular | No | Generación |
| **Diffusion** | Denoising iterativo | No (en cada paso) | Generación SOTA (Stable Diffusion) |

**Conclusión:** la arquitectura encoder-decoder es **una idea fundamental** que se reusa en clasificación de pixeles, generación y compresión. Cambiar el target y agregar/quitar piezas te lleva a U-Net, VAE, GAN, diffusion.""")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Consideraciones, cuándo usar, cuándo NO
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 14. ¿Cuándo Usar un Autoencoder (y Cuándo NO)?

### Decisión rápida

| Situación | ¿AE? | Mejor alternativa |
|-----------|------|-------------------|
| Tienes labels y quieres clasificar | NO | CNN / MLP / RF supervisado |
| Quieres detectar anomalías raras | ✅ | OK con AE; alternativas: IsolationForest, OneClassSVM |
| Quieres comprimir features no lineales | ✅ | Si los datos son aproximadamente lineales: PCA |
| Quieres generar imágenes nuevas | NO (mediocre) | VAE, GAN, Diffusion |
| Quieres limpiar ruido / restaurar | ✅ | Denoising AE |
| Quieres segmentar pixeles | NO (AE puro) | U-Net (AE + skips) |
| Quieres pre-entrenar sin labels | ✅ | AE → reutilizar encoder + cabezal supervisado |

### Las 4 trampas más comunes

**1. Bottleneck demasiado grande → el AE aprende la identidad**

Si el cuello de botella es del mismo tamaño que el input (o casi), la red puede simplemente "copiar" pixel a pixel sin aprender nada útil. **Regla práctica**: bottleneck entre 1/4 y 1/50 del input.

**2. Entrenar con anomalías mezcladas → el AE las aprende a reconstruir**

Si tu dataset de "normal" tiene 1-5% de anomalías sin etiquetar, el AE las puede aprender como variantes normales. Limpia el set normal con reglas de negocio o un primer filtro de IsolationForest.

**3. MSE no captura "rareza visual"**

MSE penaliza diferencias pixel a pixel. Un defecto pequeño pero crítico (un pixel rojo en una etiqueta blanca) puede dar MSE bajo. Para visión industrial real, considera **SSIM** o métricas perceptuales.

**4. Sobreajuste fácil con poco data**

Como cualquier red, un AE puede memorizar datos chicos. Dropout, regularización L2 en el latente, o aumentación ayudan.

### Lo que asume todo autoencoder

| Asunción | Qué pasa si falla |
|----------|-------------------|
| Los datos tienen regularidades comprimibles | Si son ruido puro, no aprende nada |
| Lo que quieres detectar (anomalía/ruido) NO es la mayoría | Si la "anomalía" es el 50%, el AE las aprende también |
| Tu loss refleja lo que importa | Si MSE no captura tu defecto, cambia de loss |""")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Próximos pasos
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 15. ¿Qué Viene Después? — Familia AE

Los autoencoders son la **puerta de entrada** a una familia entera de modelos:

| Modelo | Idea extra | Para qué sirve |
|--------|-----------|-----------------|
| **VAE** (Variational AE) | El latente sigue una distribución (Gaussiana). Permite muestrear puntos válidos. | Generación, latente continuo |
| **Denoising AE** | Input ruidoso → output limpio. Lo vimos hoy. | Restauración, pretraining |
| **Sparse AE** | Penaliza neuronas latentes activas. | Features interpretables |
| **Contractive AE** | Penaliza sensibilidad del latente al input. | Latente robusto a perturbaciones |
| **U-Net** | AE con skip connections. Lo vimos hoy. | Segmentación |
| **VQ-VAE** | Latente discreto (vector quantization). | Base de DALL-E, Stable Diffusion |
| **Diffusion** | Denoising iterativo en lugar de un solo paso. | Estado del arte en generación |
| **Sequence-to-sequence AE** | Encoder y decoder son RNN/Transformer. | Traducción, resumen de texto |

**Lo que vamos a profundizar en Módulo 5 (Modelos Avanzados):**
- RNN / LSTM para secuencias.
- Series temporales con encoder-decoder (forecasting de ventas, demanda).
- Auto-supervisión como técnica general (los modelos modernos tipo BERT, GPT son auto-supervisados a su manera).""")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Resumen + Tarea
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 16. Resumen

| Concepto | Idea clave |
|----------|------------|
| **Autoencoder** | Red que copia su input por un cuello de botella → aprende a comprimir |
| **Loss** | MSE (o BCE) entre input y output. **Sin etiquetas.** |
| **Encoder / Decoder** | Dos sub-modelos que pueden usarse por separado |
| **Latente** | Representación comprimida. Útil por sí sola para visualizar, clustering, etc. |
| **PCA vs AE** | PCA = AE lineal. AE permite no linealidad pero pierde interpretabilidad |
| **Conv AE** | Misma idea con Conv2D + UpSampling para imágenes |
| **Denoising** | Entrenar con input ruidoso, target limpio → red aprende a quitar ruido |
| **Anomaly Detection** | Entrenar solo con NORMAL. Error de reconstrucción alto = anomalía |
| **U-Net** | AE con skip connections → segmentación |
| **Cuándo NO** | Si tienes labels, si necesitas generación de calidad, si el bottleneck es demasiado grande |

**La idea central:** un autoencoder aprende un **espacio latente propio para tus datos** — un sistema de coordenadas no lineal, entrenado por reconstrucción y sin etiquetas. El encoder entrenado es el activo que te llevas; cada aplicación (anomalía, denoising, segmentación, generación) es un uso distinto de ese mismo activo.

### Tarea para casa

1. Tomar el **Conv AE** de denoising y entrenarlo con `Fashion-MNIST` en lugar de MNIST. Probar con `NOISE = 0.5` y `NOISE = 0.7`. ¿A qué nivel de ruido se rompe?

2. Tomar el **AE de anomaly detection** y cambiar la clase "normal" de `0` a `1`. ¿El umbral cambia? ¿Cuáles dígitos son los más "anómalos" relativos a 1?

3. Para los curiosos: armar un AE tabular sobre un dataset real (sugerencia: `credit_card_fraud` de Kaggle o cualquier dataset de telemetría que tengan a mano). Reportar precision/recall en detección.

**Próxima clase:** cierre del Módulo 4 (clase 30, miércoles 20 de mayo) — repaso integrador del bloque de Machine Learning + transición al Módulo 5 (Modelos Avanzados).""")


# ─────────────────────────────────────────────────────────────────────────────
# Build .ipynb
# ─────────────────────────────────────────────────────────────────────────────
def build():
    cells = []
    for kind, source in CELLS:
        if kind == "markdown":
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": source,
            })
        else:
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": source,
                "outputs": [],
                "execution_count": None,
            })

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = Path(__file__).parent / "Clase_29_Autoencoders.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Notebook escrito: {out}")
    print(f"# celdas: {len(cells)}")


if __name__ == "__main__":
    build()
