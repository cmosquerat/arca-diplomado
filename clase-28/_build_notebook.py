"""Construye Clase_28_Segmentacion.ipynb desde celdas declaradas en orden."""
import json
import uuid
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append(("markdown", text))


def code(text):
    CELLS.append(("code", text))


# ─────────────────────────────────────────────────────────────────────────────
# 0. Header
# ─────────────────────────────────────────────────────────────────────────────
md("""# Clase 28: Modelos en Producción + Segmentación para Conteo

**Diplomado en Data Science Aplicada con Python** · Arca Continental Ecuador x UDLA

---

Hoy retomamos donde quedó la clase 27: el dataset de **Brain Tumor** ya está etiquetado en Label Studio, pero falta entrenarlo. Lo entrenamos, discutimos qué métrica escoger según el dominio, y vemos cómo **guardar el modelo** para usarlo en otro lugar.

Después introducimos formalmente **segmentación** y construimos una **app de conteo de píldoras** end-to-end.

**Estructura:**

1. **Brain Tumor**: bajar dataset desde LS y entrenar.
2. **Métricas a profundidad**: cuál escoger (precision vs recall vs F1 vs mAP).
3. **Guardar y cargar el modelo** (`best.pt` portable, ONNX para producción).
4. **Segmentación**: por qué y cuándo. Detection vs Segmentation.
5. **API YOLO Segmentation**: `masks.data`, `.xy`, `.xyn`, área desde máscara.
6. **Clases COCO de seg**: qué reconoce sin entrenar (frutas, personas, pizza…).
7. **Bajar dataset de píldoras** desde Roboflow Universe.
8. **Subir a LS sin etiquetas** y bajar después de etiquetar.
9. **Entrenar `yolo26n-seg`** sobre píldoras.
10. **App Gradio**: contar píldoras desde una foto.

> Correr en Colab con GPU: *Runtime → Change runtime type → T4 GPU*.""")

# ─────────────────────────────────────────────────────────────────────────────
# 0. Setup
# ─────────────────────────────────────────────────────────────────────────────
md("""## 0. Setup

Instalamos las librerías. **Si después de esto al hacer `from ultralytics import YOLO` te sale el error `PIL._typing._Ink`**, no reinicies el runtime — corre la celda siguiente para limpiar el cache de módulos.""")

code("""!pip install -q -U pillow
!pip install -q ultralytics label-studio-sdk roboflow gradio""")

md("""**Fix sin reiniciar runtime.** Colab tiene un `PIL` viejo cargado en memoria; el `pip install` solo escribe el nuevo a disco. Forzamos a Python a olvidar PIL y matplotlib:""")

code("""# Solo correr esta celda si la siguiente falla con 'PIL._typing._Ink'
import sys
for mod in list(sys.modules):
    if mod.startswith(("PIL", "matplotlib")):
        del sys.modules[mod]
print("Cache de PIL/matplotlib limpiado. Ahora podés re-importar.")""")

code("""import os, io, re, shutil, time, zipfile, random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPoly
from PIL import Image
import torch
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Torch:  {torch.__version__}")
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU:    {torch.cuda.get_device_name(0)}")""")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Brain Tumor — cargar modelo pre-entrenado
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 1. Cierre Clase 27: Cargar el Detector de Tumores Pre-Entrenado

Entrenar `yolo26m` con 40 epochs sobre las 893 MRI toma **~45 minutos** en una T4. Para no quemar tiempo de clase, el profesor lo entrenó offline y trae el modelo en un zip. Hoy lo cargamos, lo validamos y lo usamos.

### 1.1 Bajar el dataset desde Label Studio

Aún necesitamos el dataset (para `val()` y para visualizar). Mismo flujo que clase 27.""")

code("""LS_URL   = "https://label-studio-production-281f.up.railway.app"
LS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA4NTgzNzMwOCwiaWF0IjoxNzc4NjM3MzA4LCJqdGkiOiI1ODc4YzcwNjhiYjU0M2IzYTZmNmI0NDZiYTFlZGIwNyIsInVzZXJfaWQiOiIxIn0.AclTHiVDnV8BIw5jHK3IPEb7sgMhHi3R2xUb2djLT6I"

from label_studio_sdk import LabelStudio
ls = LabelStudio(base_url=LS_URL, api_key=LS_TOKEN)

for p in ls.projects.list():
    print(f"  id={p.id:>3}  {p.title:50s}  tasks={p.task_number}")""")

code("""# Buscamos los dos proyectos que vamos a usar hoy
projects = list(ls.projects.list())
ID_BRAIN = next(p.id for p in projects if "Brain Tumor" in p.title)
ID_PILLS = next((p.id for p in projects if "Pill Segmentation" in p.title), None)
print(f"ID_BRAIN = {ID_BRAIN}")
print(f"ID_PILLS = {ID_PILLS}  ← lo usaremos en sección 6.6 y en 8")""")

md("""### 1.2 Exportar etiquetas y bajar imágenes

Reutilizamos las funciones que escribimos en clase 27. Las dejamos aquí para que el notebook sea autocontenido:""")

code("""def export_labels(project_id, out_zip, out_dir, export_type="YOLO"):
    \"\"\"Crea export, baja zip, descomprime.\"\"\"
    export = ls.projects.exports.create(id=project_id)
    with open(out_zip, "wb") as f:
        for chunk in ls.projects.exports.download(
            id=project_id, export_pk=export.id, export_type=export_type):
            f.write(chunk)
    if Path(out_dir).exists():
        shutil.rmtree(out_dir)
    with zipfile.ZipFile(out_zip) as z:
        z.extractall(out_dir)
    return Path(out_dir)

def _get_access_token():
    if len(LS_TOKEN) > 100:
        return requests.post(f"{LS_URL}/api/token/refresh",
                              json={"refresh": LS_TOKEN}).json()["access"]
    return LS_TOKEN

def fetch_ls_images(project_id, out_dir):
    \"\"\"Baja todas las imágenes del proyecto.\"\"\"
    access = _get_access_token()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in ls.tasks.list(project=project_id, page_size=200):
        stored = t.data["image"].split("/")[-1]
        url = LS_URL + t.data["image"]
        r = requests.get(url, headers={"Authorization": f"Bearer {access}"},
                          allow_redirects=True)
        if r.status_code == 401:
            access = _get_access_token()
            r = requests.get(url, headers={"Authorization": f"Bearer {access}"},
                              allow_redirects=True)
        r.raise_for_status()
        (out_dir / stored).write_bytes(r.content)
        n += 1
        if n % 100 == 0: print(f"  [{n}]")
    print(f"  {n} imágenes bajadas a {out_dir}")
    return n

def organize_yolo(base_dir):
    \"\"\"Pone imágenes/labels en images/train y labels/train con basenames pareados.\"\"\"
    src_img = base_dir / "images_raw"
    dst_img = base_dir / "images" / "train"
    dst_img.mkdir(parents=True, exist_ok=True)
    for f in src_img.iterdir():
        shutil.copy(f, dst_img / f.name)

    final_lbl = base_dir / "labels" / "train"
    final_lbl.mkdir(parents=True, exist_ok=True)
    for f in list((base_dir / "labels").glob("*.txt")):
        clean = re.sub(r'^[0-9a-f]+__', '', f.name)
        shutil.move(str(f), str(final_lbl / clean))

    img_stems = {p.stem for p in dst_img.iterdir()}
    lbl_stems = {p.stem for p in final_lbl.glob("*.txt")}
    matched = img_stems & lbl_stems
    print(f"  Imágenes: {len(img_stems)} | Labels: {len(lbl_stems)} | Match: {len(matched)}")""")

code("""brain_dir = export_labels(ID_BRAIN, "/content/brain.zip", "/content/brain")
fetch_ls_images(ID_BRAIN, brain_dir / "images_raw")
organize_yolo(brain_dir)""")

md("""### 1.3 `data.yaml` — el manifiesto del dataset

YOLO necesita un YAML que diga *dónde están las imágenes* y *qué clases hay*. Es el mismo formato que usamos para entrenar, ahora lo usamos para validar.""")

code("""yaml_brain = f'''path: {brain_dir.absolute()}
train: images/train
val:   images/train

names:
  0: negative
  1: positive
'''
yaml_path_brain = brain_dir / "data.yaml"
yaml_path_brain.write_text(yaml_brain)
print(yaml_brain)""")

md("""### 1.4 Bajar el modelo pre-entrenado desde GitHub

El profesor entrenó `yolo26m` por 40 epochs offline y subió **solo `best.pt`** (44 MB) al repo del curso. Para validar y predecir no necesitamos nada más — `model.val()` regenera la confusion matrix, las curvas PR y los `val_batch*_pred.jpg` cuando los necesitemos.""")

code("""# Bajamos el modelo directo desde el repo público del curso
BEST_URL  = "https://raw.githubusercontent.com/cmosquerat/arca-diplomado/main/clase-28/best.pt"
BEST_BRAIN = "/content/best.pt"

!wget -q -O {BEST_BRAIN} {BEST_URL}
assert Path(BEST_BRAIN).exists() and Path(BEST_BRAIN).stat().st_size > 1e6, \\
    "Algo falló al bajar el modelo. Revisar la URL."

print(f"best.pt:  {BEST_BRAIN}")
print(f"Tamaño:   {Path(BEST_BRAIN).stat().st_size/1e6:.1f} MB")""")

md("""### 1.5 Cargar el modelo y verificar que está vivo

Una vez cargado, el modelo es exactamente el mismo que si lo hubiéramos entrenado en este Colab. **`best.pt` es portable: 44 MB que tienen TODO el conocimiento del modelo.**""")

code("""from ultralytics import YOLO

modelo_brain = YOLO(BEST_BRAIN)

print(f"Tarea:    {modelo_brain.task}")
print(f"Clases:   {modelo_brain.names}")
n_params = sum(p.numel() for p in modelo_brain.model.parameters())
print(f"Parámetros: {n_params:,}  (~{n_params/1e6:.1f}M  → es yolo26m)")

# Sanity check: predict sobre una imagen del dataset
img_test = next((brain_dir / "images" / "train").glob("*.jpg"))
r = modelo_brain(str(img_test), conf=0.25, verbose=False)[0]
print(f"\\nSanity check: {len(r.boxes)} detecciones en {img_test.name}")""")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Métricas a profundidad
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 2. Métricas a Profundidad: ¿Cuál Escoger?

Cuando entrenas YOLO, te reporta varias métricas. **Saber cuál mirar depende del dominio.** Acá las desglosamos.

### 2.1 Las 4 métricas que YOLO reporta

| Métrica | Fórmula | Qué mide |
|---------|---------|----------|
| **Precision** | TP / (TP + FP) | De lo que detecté, ¿qué % era real? |
| **Recall** | TP / (TP + FN) | De lo real, ¿qué % detecté? |
| **F1** | 2·P·R / (P+R) | Media armónica. Equilibra ambos. |
| **mAP@0.5** | AP a IoU≥0.5 | "Cajas suficientemente bien" |
| **mAP@0.5:0.95** | AP promedio en 10 umbrales | Cajas BIEN alineadas (más estricto) |

> **mAP** = mean Average Precision = promedio (sobre clases) del área bajo la curva precision-recall.""")

code("""# Validamos el modelo contra el dataset. val() regenera las visualizaciones
# (confusion matrix, PR curves, val_batch*_pred.jpg) en una carpeta nueva.
m_b = modelo_brain.val(data=str(yaml_path_brain), verbose=False)

# La carpeta de salida queda en m_b.save_dir
VAL_DIR = Path(m_b.save_dir)
print(f"Outputs de val regenerados en: {VAL_DIR}\\n")

print(f"mAP@0.5       = {m_b.box.map50:.3f}")
print(f"mAP@0.5:0.95  = {m_b.box.map:.3f}")
print(f"Precision     = {m_b.box.mp:.3f}")
print(f"Recall (Sens) = {m_b.box.mr:.3f}")
print()
print("Por clase:")
for i, name in enumerate(["negative", "positive"]):
    if i < len(m_b.box.maps):
        print(f"  {name:10s}  mAP@0.5:0.95 = {m_b.box.maps[i]:.3f}")""")

md("""### 2.2 Decidir según el dominio

| Dominio | Qué duele más | Métrica clave | Acción |
|---------|---------------|---------------|--------|
| **Placas (LPR)** | Leer placa fantasma → falso registro | **Precision** | conf alto (~0.5) |
| **Brain Tumor** | Perder un tumor real | **Recall (Sensitivity)** | conf bajo (~0.15) |
| **Píldoras** | Contar mal (ambos lados pesan) | **F1 / mAP@0.5** | conf medio (~0.3) |

**La métrica no la elige el modelo, la elige el costo del error en tu dominio.** Decide primero, optimiza después.""")

md("""### 2.3 El conf threshold mueve la métrica — visualmente

YOLO entrenó con un solo modelo, pero en `predict()` ajustas `conf` para decidir el balance. **Lo importante es VER cómo cambia la salida**, no solo el número.

Para que se note el efecto, buscamos una imagen donde la ground truth marque tumor:""")

code("""# Buscar una imagen que tenga label de tumor positivo (class 1)
def imagen_con_tumor():
    for lbl in sorted((brain_dir / "labels" / "train").glob("*.txt")):
        contenido = lbl.read_text().strip()
        if any(line.startswith("1 ") for line in contenido.splitlines()):
            img = (brain_dir / "images" / "train") / (lbl.stem + ".jpg")
            if img.exists(): return img
    return next((brain_dir / "images" / "train").glob("*.jpg"))

img_test = imagen_con_tumor()
print(f"Imagen elegida: {img_test.name}")""")

code("""# Comparamos 4 umbrales SOBRE LA MISMA IMAGEN, mostrando bboxes
fig, axes = plt.subplots(1, 4, figsize=(20, 6))

for ax, conf in zip(axes, [0.5, 0.25, 0.15, 0.05]):
    r = modelo_brain(str(img_test), conf=conf, verbose=False)[0]
    ax.imshow(r.plot()[..., ::-1])
    ax.axis('off')
    n = len(r.boxes)
    # Promedio de confidence solo si hay detecciones
    avg_conf = float(r.boxes.conf.mean()) if n > 0 else 0.0
    ax.set_title(f"conf={conf}  |  {n} detecciones\\n"
                  f"conf promedio: {avg_conf:.2f}",
                  fontsize=11, fontweight='bold',
                  color='#16A34A' if conf == 0.15 else '#2563EB')

plt.suptitle(f"Mismo modelo, distinto umbral — {img_test.name}",
             fontsize=13, fontweight='bold', color='#6B1525')
plt.tight_layout(); plt.show()

print("\\nObservaciones esperadas:")
print("  conf=0.5  → solo detecciones MUY seguras (precision↑, recall↓)")
print("  conf=0.05 → modelo paranoico (precision↓, recall↑)")
print("  Para tumor cerebral, conf bajo es preferible: mejor falsa alarma")
print("  que perder un tumor real (el radiólogo descarta los extra).")""")

md("""### 2.4 Confusion matrix (regenerada por `val()`)

Cuando llamamos `model.val()`, Ultralytics generó la matriz de confusión fresca en `VAL_DIR/confusion_matrix.png`. Lee así:
- **Diagonal** = aciertos (TP por clase).
- **Fuera de diagonal** = confusiones del modelo.
- En tumor: lo crítico es que `positive → background` (perder un tumor) sea LO MÁS BAJO posible.""")

code("""cm_path = VAL_DIR / "confusion_matrix.png"
if cm_path.exists():
    plt.figure(figsize=(9, 8))
    plt.imshow(Image.open(cm_path)); plt.axis('off')
    plt.title("Confusion matrix — brain tumor (regenerada por val())")
    plt.show()
else:
    print(f"No encontré {cm_path}.")""")

md("""### 2.4b Predicciones sobre el set de validación

`val()` también guarda 3 mosaicos con predicciones vs ground truth. Útil para ver dónde se confunde el modelo en imágenes reales:""")

code("""val_pred = VAL_DIR / "val_batch0_pred.jpg"
val_lbl  = VAL_DIR / "val_batch0_labels.jpg"

if val_pred.exists() and val_lbl.exists():
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    axes[0].imshow(Image.open(val_lbl));  axes[0].axis('off')
    axes[0].set_title("Ground truth (real)", fontsize=12)
    axes[1].imshow(Image.open(val_pred)); axes[1].axis('off')
    axes[1].set_title("Predicciones del modelo", fontsize=12)
    plt.tight_layout(); plt.show()""")

md("""### 2.5 La regla de oro

> El número de la métrica es solo eso, un número. Lo que decide si el modelo sirve es **leer la confusion matrix con un experto del dominio**. Un mAP de 0.85 puede esconder que el modelo es ciego ante la clase que más importa.""")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Guardar y cargar el modelo
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 3. Guardar el Modelo de Forma Portable

Ya vimos que el zip que el profesor trajo contenía mucho ruido (`last.pt`, optimizer states, logs). **Lo único que se necesita para usar el modelo es `best.pt`** — los ~44 MB con los pesos.

### 3.1 Copiar `best.pt` a un lugar limpio y versionado

Convención: `tarea_arquitectura_fecha.pt`. Nunca se sobreescribe.""")

code("""import datetime
models_dir = Path("/content/models")
models_dir.mkdir(exist_ok=True)

today = datetime.date.today().isoformat()
versioned_name = f"brain_yolo26m_{today}.pt"
versioned_path = models_dir / versioned_name

shutil.copy(BEST_BRAIN, versioned_path)
print(f"Modelo guardado: {versioned_path}")
print(f"Tamaño:          {versioned_path.stat().st_size/1e6:.1f} MB")""")

md("""### 3.2 Cargarlo "desde cero" — simulando otro Colab

Para confirmar que `best.pt` es realmente portable, borramos las variables del entorno y volvemos a cargar el modelo SOLO desde el archivo:""")

code("""del modelo_brain        # simulamos un kernel nuevo

# Cargar solo desde el archivo .pt:
modelo_cargado = YOLO(str(versioned_path))

print(f"Tarea:    {modelo_cargado.task}")
print(f"Clases:   {modelo_cargado.names}")

# Y ya podemos predecir:
r = modelo_cargado(str(img_test), conf=0.15, verbose=False)[0]
print(f"\\nDetecciones en imagen de test: {len(r.boxes)}")

# Recuperamos la variable de trabajo
modelo_brain = modelo_cargado""")

md("""### 3.3 Export ONNX — para producción sin Python

Para correr el modelo en C++, Java, JavaScript, o en un microservicio sin instalar PyTorch, exportamos a **ONNX** (Open Neural Network Exchange). Es el formato universal de deep learning.""")

code("""onnx_path = modelo_brain.export(format="onnx")
print(f"\\nExportado a: {onnx_path}")
print(f"Tamaño:      {os.path.getsize(onnx_path)/1e6:.1f} MB")
print()
print("Otros formatos disponibles:")
print("  - tflite   (móvil Android)")
print("  - coreml   (iOS / macOS)")
print("  - engine   (TensorRT en NVIDIA Jetson, 60+ FPS)")
print("  - openvino (Intel)")
print("  - paddle   (PaddlePaddle)")""")

md("""### 3.4 Versionado simple (sin MLflow ni nada caro)

Para empezar en una empresa pequeña, basta con esta convención:

```
models/
├── brain_yolo26m_2026-05-15.pt    ← entrenado hoy
├── brain_yolo26m_2026-05-22.pt    ← re-entrenado con más datos
├── placas_yolo26n_2026-05-10.pt
└── pills_yolo26n-seg_2026-05-20.pt
```

**Reglas:**
- El nombre dice **tarea + arquitectura + fecha**.
- Nunca se sobreescribe (cada re-entrenamiento es un archivo nuevo).
- En producción se apunta al archivo específico, no a un "latest".

> Cuando crezcas: **MLflow** o **Weights & Biases** gestionan esto + métricas + comparación. Pero para empezar, esta convención funciona.""")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Segmentación — introducción
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 4. Tarea Nueva: Segmentación

### 4.1 Detection vs Segmentation — la diferencia visual

- **Detection:** un rectángulo (bbox) por objeto. 4 números: `[x, y, w, h]`.
- **Segmentation:** un **polígono** o **máscara binaria** por objeto. Decenas o cientos de puntos.

| | Detection | Segmentation |
|--|--|--|
| Output por objeto | 4 números | Polígono (∼20 puntos) |
| Etiquetar | ~4 clicks | ~6-15 clicks |
| Velocidad inferencia | rápido | 10-20% más lento |
| Tamaño modelo | 6 MB | 7 MB |
| Cuándo usar | objetos separados | **objetos pegados, área, conteo** |

### 4.2 ¿Cuándo necesitamos segmentación?

1. **Objetos pegados o solapados:** las bboxes se solapan, NMS borra detecciones válidas. Ejemplo: una línea de píldoras pegadas.
2. **Hay que medir área:** si necesitas saber qué tan grande es algo (caja chica vs grande, mancha en una hoja de planta), la bbox sobre-estima.
3. **Objetos no rectangulares:** una manzana vista de lado en una bbox tiene 30% de fondo "rosa". Una máscara solo captura la manzana.

> **Regla práctica:** si el conteo o el área importan, usa segmentación. Si solo importa "está o no está", detection.""")

# ─────────────────────────────────────────────────────────────────────────────
# 5. API YOLO Segmentation
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 5. API de YOLO Segmentation a Profundidad

**Bonita simetría:** la API es **idéntica** a detection. Solo cambia el modelo (sufijo `-seg`) y los outputs incluyen máscaras.""")

code("""seg_model = YOLO("yolo26n-seg.pt")
print(f"Tarea:    {seg_model.task}")
print(f"# clases: {len(seg_model.names)}  (las mismas 80 de COCO)")
print(f"Ejemplos: {list(seg_model.names.values())[:8]}")""")

md("""### 5.1 Predict — mismo método, output enriquecido

`results[0]` ahora tiene **dos cosas**:
- `boxes` — igual que detection (xyxy, conf, cls, …).
- `masks` — la novedad.""")

code("""# Bajamos una foto con varios objetos COCO (clásica de Ultralytics)
import urllib.request
def download(url, name):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        Path(name).write_bytes(r.read())
    return name

# bus.jpg trae: 1 bus + 4 personas → buen mix para ver multi-clase
demo_img = download(
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg",
    "/content/demo_bus.jpg",
)

res = seg_model(demo_img, conf=0.4, verbose=False)[0]
print(f"Objetos detectados: {len(res.boxes)}")
print(f"Clases:             {[seg_model.names[int(c)] for c in res.boxes.cls]}")
print(f"¿Hay máscaras?:     {res.masks is not None}")
print(f"Shape masks.data:   {tuple(res.masks.data.shape)}  (N, H, W)")

# Y visualizamos — sin esto los números no significan nada
plt.figure(figsize=(10, 8))
plt.imshow(res.plot()[..., ::-1])
plt.axis('off')
plt.title(f"yolo26n-seg sobre bus.jpg — {len(res.boxes)} objetos con máscara")
plt.show()""")

md("""### 5.2 Los 3 formatos de máscara

| Atributo | Tipo | Uso |
|----------|------|-----|
| `masks.data` | tensor `(N, H, W)` binario | Para hacer operaciones por pixel (área, IoU, overlay) |
| `masks.xy` | lista de `(P, 2)` en píxeles | Para dibujar con matplotlib o OpenCV |
| `masks.xyn` | lista de `(P, 2)` normalizado [0,1] | Para guardar en formato YOLO label (`.txt`) |""")

code("""# masks.data: tensor binario (N objetos, H, W)
print(f"masks.data shape: {res.masks.data.shape}")
print(f"masks.data dtype: {res.masks.data.dtype}")
print(f"Valores únicos:   {torch.unique(res.masks.data).tolist()}  (binario)")
print()

# masks.xy: lista de polígonos en píxeles
print(f"# polígonos en masks.xy: {len(res.masks.xy)}")
print(f"Primer polígono shape:   {res.masks.xy[0].shape}  (P puntos, 2 coords)")
print(f"Primeros 3 puntos:        {res.masks.xy[0][:3].tolist()}")
print()

# masks.xyn: lo mismo pero normalizado
print(f"Primer polígono xyn:     {res.masks.xyn[0][:3].tolist()}")""")

md("""### 5.3 Calcular el área de cada máscara

El área en píxeles es **literalmente** la suma de la máscara binaria. Útil para filtrar objetos por tamaño (descartar fragmentos, o quedarse solo con los grandes):""")

code("""# Tabla de áreas
print(f"{'idx':>4}  {'clase':<15}  {'conf':>5}  {'área (px)':>10}")
areas = []
for i, mask in enumerate(res.masks.data):
    area_px = int(mask.sum().item())
    clase = res.names[int(res.boxes.cls[i])]
    conf  = float(res.boxes.conf[i])
    print(f"{i:>4}  {clase:<15}  {conf:>5.2f}  {area_px:>10}")
    areas.append(area_px)

# Visualizar: original + overlay coloreado por tamaño (más grande = más opaco)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
img_arr = np.array(Image.open(demo_img))

axes[0].imshow(img_arr); axes[0].axis('off')
axes[0].set_title("Imagen original")

axes[1].imshow(img_arr); axes[1].axis('off')
axes[1].set_title("Máscaras coloreadas por tamaño (objeto más grande = más visible)")

max_area = max(areas) if areas else 1
colors = plt.cm.viridis(np.linspace(0.2, 0.95, len(res.masks.xy)))
for i, (poly, color) in enumerate(zip(res.masks.xy, colors)):
    alpha = 0.3 + 0.5 * (areas[i] / max_area)   # más alpha = más grande
    axes[1].add_patch(MplPoly(poly, alpha=alpha, fc=color, ec='white', lw=1.5))
    # etiqueta con clase y área
    cx, cy = poly.mean(axis=0)
    axes[1].text(cx, cy, f"{areas[i]}px",
                  ha='center', fontsize=9, color='white',
                  fontweight='bold',
                  bbox=dict(boxstyle='round', fc='black', alpha=0.6))
plt.tight_layout(); plt.show()""")

md("""### 5.4 Visualizar las máscaras

`res.plot()` ya dibuja todo (bbox + máscara coloreada). Si quieres control fino, usa `masks.xy` con matplotlib:""")

code("""fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Izq: plot() automático
axes[0].imshow(res.plot()[..., ::-1])
axes[0].axis('off')
axes[0].set_title("res.plot() — automático")

# Der: control manual con masks.xy
img_arr = np.array(Image.open(demo_img))
axes[1].imshow(img_arr); axes[1].axis('off')
axes[1].set_title("Polígonos manuales (masks.xy)")

colors = plt.cm.tab10(np.linspace(0, 1, max(len(res.masks.xy), 1)))
for poly, color in zip(res.masks.xy, colors):
    axes[1].add_patch(MplPoly(poly, alpha=0.4, fc=color, ec=color, lw=2))

plt.tight_layout(); plt.show()""")

md("""### 5.5 IoU de máscaras (Mask IoU)

Para evaluar segmentación se usa **IoU de píxeles**, no de rectángulos. Implementación de 3 líneas:""")

code("""def mask_iou(m1, m2):
    \"\"\"IoU entre dos máscaras binarias del mismo shape.\"\"\"
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0

# Comparamos las dos primeras máscaras (ejemplo)
if len(res.masks.data) >= 2:
    m1 = res.masks.data[0].bool().cpu().numpy()
    m2 = res.masks.data[1].bool().cpu().numpy()
    print(f"IoU entre máscara 0 y 1: {mask_iou(m1, m2):.3f}")
    print(f"(esperamos cerca de 0: son objetos distintos)")""")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Aplicaciones del pretrained — galería de demos antes de entrenar
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 6. Aplicaciones del Pretrained: Galería de Demos *Antes* de Entrenar

`yolo26n-seg.pt` viene pre-entrenado en **COCO** (80 clases). Antes de tocar el dataset propio vale la pena explorar **qué viene gratis** — para muchas demos de cliente, validación de PoC o automatizaciones simples, no necesitas entrenar nada.

### 6.1 Las 80 Clases Agrupadas

Conocer las clases es media batalla — te dice qué problemas puedes resolver hoy mismo:""")

code("""COCO_GROUPS = {
    "Personas":          ["person"],
    "Vehículos":         ["bicycle", "car", "motorcycle", "airplane", "bus",
                          "train", "truck", "boat"],
    "Mobiliario urbano": ["traffic light", "fire hydrant", "stop sign",
                          "parking meter", "bench"],
    "Animales":          ["bird", "cat", "dog", "horse", "sheep", "cow",
                          "elephant", "bear", "zebra", "giraffe"],
    "Accesorios":        ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "Deportes":          ["frisbee", "skis", "snowboard", "sports ball", "kite",
                          "baseball bat", "baseball glove", "skateboard",
                          "surfboard", "tennis racket"],
    "Cocina":            ["bottle", "wine glass", "cup", "fork", "knife",
                          "spoon", "bowl"],
    "Comida":            ["banana", "apple", "sandwich", "orange", "broccoli",
                          "carrot", "hot dog", "pizza", "donut", "cake"],
    "Muebles":           ["chair", "couch", "potted plant", "bed",
                          "dining table", "toilet"],
    "Electrónica":       ["tv", "laptop", "mouse", "remote", "keyboard",
                          "cell phone"],
    "Electrodomésticos": ["microwave", "oven", "toaster", "sink",
                          "refrigerator"],
    "Otros":             ["book", "clock", "vase", "scissors", "teddy bear",
                          "hair drier", "toothbrush"],
}
for grupo, items in COCO_GROUPS.items():
    print(f"{grupo:<22} | {' · '.join(items)}")""")

md("""### 6.2 Galería de Demos — 6 Aplicaciones Reales

Bajamos 6 imágenes reales (Unsplash) y le pasamos cada una al modelo. Las URLs ya están **verificadas** — todas dan detecciones útiles para discutir en clase.""")

code("""# URLs verificadas: cada una rinde detecciones útiles
GALERIA = {
    "frutas_mix":   ("https://images.unsplash.com/photo-1610348725531-843dff563e2c?w=900",
                      "Multi-clase: 5+ tipos de comida en una sola foto"),
    "naranjas":     ("https://images.unsplash.com/photo-1610832958506-aa56368176cf?w=900",
                      "Conteo masivo: ¿cuántas naranjas hay?"),
    "calle":        ("https://images.unsplash.com/photo-1517649763962-0c623066013b?w=900",
                      "Escena urbana: personas + bicicletas mezcladas"),
    "ovejas":       ("https://images.unsplash.com/photo-1484557985045-edf25e08da73?w=900",
                      "Inventario ganadero: contar ovejas en un campo"),
    "pizza":        ("https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=900",
                      "Comida lista: cuántas pizzas, qué tan grandes"),
    "cocina":       ("https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=900",
                      "Hogar: personas + utensilios juntos"),
}

# Bajar todas
for nombre, (url, _) in GALERIA.items():
    download(url, f"/content/galeria_{nombre}.jpg")

print(f"{len(GALERIA)} imágenes bajadas.")""")

md("""#### Inferencia sobre las 6 imágenes

Una sola llamada por imagen. Vamos a ver el conteo de cada clase detectada:""")

code("""from collections import Counter

resultados = {}
for nombre, (_, desc) in GALERIA.items():
    res = seg_model(f"/content/galeria_{nombre}.jpg", conf=0.25, verbose=False)[0]
    clases = Counter(seg_model.names[int(c)] for c in res.boxes.cls)
    resultados[nombre] = (res, clases, desc)
    print(f"{nombre:<14} n={len(res.boxes):>2}  {dict(clases)}")
    print(f"               → {desc}")""")

md("""#### Galería visual: las 6 al mismo tiempo""")

code("""fig, axes = plt.subplots(2, 3, figsize=(18, 11))
for ax, (nombre, (res, clases, desc)) in zip(axes.flat, resultados.items()):
    ax.imshow(res.plot()[..., ::-1])
    ax.axis('off')
    titulo = f"{nombre}: {len(res.boxes)} detecciones\\n"
    titulo += " · ".join(f"{v} {k}" for k, v in clases.items())
    ax.set_title(titulo, fontsize=10)
plt.tight_layout(); plt.show()""")

md("""### 6.3 Demo dirigido: conteo de naranjas

**Caso de uso real:** una distribuidora frutícola quiere automatizar el conteo en cajones. COCO ya incluye `orange` (id 49) → cero entrenamiento.""")

code("""res_o = seg_model("/content/galeria_naranjas.jpg", conf=0.25, verbose=False)[0]

# Conteo
n_orange = sum(1 for c in res_o.boxes.cls if int(c) == 49)
print(f"Naranjas en la foto: {n_orange}")

# Áreas (proxy de tamaño / madurez)
if res_o.masks is not None:
    areas = [(i, int(m.sum().item())) for i, (m, c) in
              enumerate(zip(res_o.masks.data, res_o.boxes.cls))
              if int(c) == 49]
    areas.sort(key=lambda x: -x[1])
    print(f"\\nLa más grande:  idx={areas[0][0]}, área={areas[0][1]} px")
    print(f"La más chica:   idx={areas[-1][0]}, área={areas[-1][1]} px")
    print(f"Promedio:       {int(np.mean([a for _, a in areas]))} px")

plt.figure(figsize=(11, 7))
plt.imshow(res_o.plot()[..., ::-1])
plt.axis('off')
plt.title(f"Conteo de naranjas: {n_orange} (todo COCO, cero entrenamiento)")
plt.show()""")

md("""### 6.4 Filtrar por clase en el `predict()`

Si solo te interesa una clase, podés filtrarla **dentro** del modelo con `classes=[id]` — el output ya viene limpio:""")

code("""res_solo_naranjas = seg_model(
    "/content/galeria_frutas_mix.jpg",
    classes=[49],          # solo orange
    conf=0.25,
    verbose=False,
)[0]
print(f"Detecciones (solo orange): {len(res_solo_naranjas.boxes)}")

plt.figure(figsize=(11, 7))
plt.imshow(res_solo_naranjas.plot()[..., ::-1])
plt.axis('off')
plt.title(f"Filtrando classes=[49] → solo {len(res_solo_naranjas.boxes)} naranjas")
plt.show()""")

md("""### 6.5 Demo de inventario: contar ovejas en un campo

Caso: una hacienda quiere contar ovejas desde una foto aérea de su finca.""")

code("""res_o = resultados["ovejas"][0]
n_ovejas = sum(1 for c in res_o.boxes.cls if seg_model.names[int(c)] == "sheep")
print(f"Ovejas detectadas: {n_ovejas}")

if res_o.masks is not None and n_ovejas > 0:
    areas = [(i, int(m.sum().item())) for i, (m, c) in
              enumerate(zip(res_o.masks.data, res_o.boxes.cls))
              if seg_model.names[int(c)] == "sheep"]
    areas.sort(key=lambda x: -x[1])
    print(f"\\nOveja más grande (más cercana): idx={areas[0][0]}, área={areas[0][1]} px")
    print(f"Oveja más chica (más lejos):     idx={areas[-1][0]}, área={areas[-1][1]} px")

# Visualizar
plt.figure(figsize=(12, 8))
plt.imshow(res_o.plot()[..., ::-1])
plt.axis('off')
plt.title(f"Inventario ganadero: {n_ovejas} ovejas detectadas — todo COCO pretrained")
plt.show()""")

md("""### 6.6 Demo crítico: cuando el pretrained FALLA

Hagamos algo que **no está en COCO**: una imagen de píldoras (la que vamos a etiquetar después). Veamos qué pasa:""")

code("""# Bajamos 3 imágenes del proyecto LS (las píldoras que el equipo va a etiquetar)
assert ID_PILLS is not None, "Necesitamos el proyecto LS de píldoras (sección 1.1)"

access = _get_access_token()
pill_samples = []
for i, t in enumerate(list(ls.tasks.list(project=ID_PILLS, page_size=3))[:3]):
    url = LS_URL + t.data["image"]
    r = requests.get(url, headers={"Authorization": f"Bearer {access}"})
    fname = f"/content/pill_demo_{i}.jpg"
    Path(fname).write_bytes(r.content)
    pill_samples.append(fname)

# Pasamos cada una al pretrained y vemos qué cree que son
print(f"{'imagen':>14}  {'N':>3}  {'clases detectadas (falsas)':<50}")
for f in pill_samples:
    res_pill = seg_model(f, conf=0.15, verbose=False)[0]
    clases = [seg_model.names[int(c)] for c in res_pill.boxes.cls]
    print(f"{f:>14}  {len(res_pill.boxes):>3}  {clases}")

print()
print("Conclusión: el pretrained confunde píldoras con 'sports ball' (formas")
print("redondas) o no detecta nada. COCO no tiene clase 'pill'. → fine-tune.")""")

code("""# Visualizamos el fallo más colorido (la imagen con más 'detecciones falsas')
mejor_fallo = None; max_falsas = -1
for f in pill_samples:
    res = seg_model(f, conf=0.15, verbose=False)[0]
    if len(res.boxes) > max_falsas:
        max_falsas = len(res.boxes)
        mejor_fallo = (f, res)

if mejor_fallo and max_falsas > 0:
    f, res_pill = mejor_fallo
    plt.figure(figsize=(11, 7))
    plt.imshow(res_pill.plot()[..., ::-1])
    plt.axis('off')
    plt.title(f"{f}: pretrained dice '{seg_model.names[int(res_pill.boxes.cls[0])]}'  → MAL")
    plt.show()
else:
    print("Todas las muestras dieron 0 detecciones — pretrained ciego completo.")""")

code("""# Visualizamos el fallo
plt.figure(figsize=(11, 7))
plt.imshow(res_pill.plot()[..., ::-1])
plt.axis('off')
plt.title(f"Pretrained sobre píldoras: {len(res_pill.boxes)} 'detecciones' (todas falsas)")
plt.show()""")

md("""### 6.7 Lección de la galería

| Para… | Pretrained alcanza | Hay que fine-tunear |
|-------|-------------------|---------------------|
| Contar **personas** | ✓ | — |
| Contar **vehículos** | ✓ | — |
| Contar **frutas/comida** estándar | ✓ | — |
| Contar **animales de granja** | ✓ | — |
| **Productos específicos** (Coca, Pepsi…) | ✗ | ✓ |
| **Píldoras** / **placas** / **tumores** | ✗ | ✓ |
| Cualquier objeto **fuera de COCO** | ✗ | ✓ |

**La pregunta antes de entrenar siempre es:** *¿mi clase está en COCO?* Si sí, usa pretrained. Si no, fine-tune. Esto te ahorra semanas de etiquetado innecesario.""")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Pills dataset
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 7. Caso de Negocio: Contar Píldoras

**Contexto.** Una farmacéutica quiere automatizar el conteo de píldoras en sus blísters durante el control de calidad. Una persona cuenta ~600 píldoras por hora; una cámara con visión puede hacer 10.000.

**Por qué segmentación (y no detection):**
- Las píldoras están **pegadas** en filas — bboxes solapan, NMS borra detecciones.
- Algunas píldoras tienen rotación distinta.
- Queremos **conteo exacto**, no "más o menos".

**Dataset:** [`pillsegmentation` de Roboflow Universe](https://universe.roboflow.com/abstract/pillsegmentation-oyygy). Lo bajamos con la API de Roboflow.""")

md("""### 7.1 Bajar de Roboflow con el SDK oficial

El SDK de Roboflow simplifica todo: API key, workspace, project, version. Una sola llamada baja el dataset listo para YOLO.""")

code("""from roboflow import Roboflow

ROBOFLOW_API_KEY = "DJqoR0JeH6JaOrpH712W"  # key pública del cliente

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project_rf = rf.workspace("abstract").project("pillsegmentation-oyygy")

# Listar versiones disponibles
versions = project_rf.versions()
print(f"Versiones disponibles: {[v.version for v in versions]}")

# Última versión
version_rf = versions[0]
print(f"Bajando versión {version_rf.version}...")

# El SDK lo baja en formato yolov8 (compatible con yolo26)
dataset_rf = version_rf.download("yolov8", location="/content/pills_raw")
print(f"\\nDataset bajado en: {dataset_rf.location}")""")

code("""# Inspeccionar lo que bajó
pills_raw = Path(dataset_rf.location)
for sub in ["train/images", "valid/images", "test/images",
            "train/labels", "valid/labels", "test/labels"]:
    p = pills_raw / sub
    n = len(list(p.glob("*"))) if p.exists() else 0
    print(f"  {sub}: {n}")""")

md("""### 7.2 Visualizar 4 imágenes con sus polígonos de Roboflow""")

code("""def read_yolo_poly(path):
    \"\"\"Lee un .txt de YOLO seg: 'cls x1 y1 x2 y2 ...' normalizado.\"\"\"
    if not path.exists(): return []
    out = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 7: continue
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        out.append((cls, pts))
    return out

pills_imgs = sorted((pills_raw / "train" / "images").glob("*.jpg"))[:8]
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, ip in zip(axes.flat, pills_imgs[:4]):
    lp = pills_raw / "train" / "labels" / (ip.stem + ".txt")
    img = np.array(Image.open(ip)); H, W = img.shape[:2]
    ax.imshow(img); ax.axis('off')
    polys = read_yolo_poly(lp)
    for cls, pts in polys:
        abs_pts = [(p[0]*W, p[1]*H) for p in pts]
        ax.add_patch(MplPoly(abs_pts, alpha=0.35, fc='cyan',
                              ec='blue', lw=1.5))
    ax.set_title(f"{ip.name}  |  {len(polys)} píldoras", fontsize=9)
plt.tight_layout(); plt.show()""")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Sube a LS sin etiquetas
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 8. Subir 100 Imágenes a Label Studio Sin Etiquetas

Para que el equipo practique etiquetando polígonos, subimos 100 imágenes **sin labels** al LS. El profesor lo hizo con `setup_labelstudio.py` (en el repo).

Verificamos que el proyecto ya esté arriba:""")

code("""# ID_PILLS ya fue definido en la sección 1.1 (al listar proyectos LS).
# Solo confirmamos los detalles:
if ID_PILLS is not None:
    p = next(p for p in ls.projects.list() if p.id == ID_PILLS)
    print(f"Proyecto LS: id={ID_PILLS}, título='{p.title}'")
    print(f"Tasks:       {p.task_number}")
    print(f"URL:         {LS_URL}/projects/{ID_PILLS}/data")
else:
    print("Proyecto NO encontrado. Correr setup_labelstudio.py primero.")""")

md("""### 8.1 Labeling config en LS

El proyecto se creó con este labeling config — clave: usa `<PolygonLabels>`, no `<RectangleLabels>`:

```xml
<View>
  <Header value="Segmentación de píldoras"/>
  <Image name="image" value="$image"/>
  <PolygonLabels name="label" toName="image">
    <Label value="pill" background="#C82B40"/>
  </PolygonLabels>
</View>
```

**Tarea para el equipo (15 minutos en clase):** entrar al proyecto, dibujar polígonos sobre **al menos 5 imágenes**. Cada píldora = un polígono. Tip: usar tecla **Z** para cerrar polígonos rápido.""")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Bajar etiquetas y entrenar
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 9. Bajar Etiquetas + Entrenar Segmentación

### 9.1 Exportar de LS

LS exporta segmentación al formato YOLO seg: `cls x1 y1 x2 y2 …` normalizado. Mismo flow que detection — la única diferencia es que el modelo a usar es `-seg`.""")

code("""# Si el equipo no terminó de etiquetar, usamos el dataset original de Roboflow
# para que la clase no se atasque. Pero el flow es idéntico:
if ID_PILLS is not None:
    pills_ls_dir = export_labels(ID_PILLS, "/content/pills.zip", "/content/pills_ls")
    n_imgs = fetch_ls_images(ID_PILLS, pills_ls_dir / "images_raw")
    organize_yolo(pills_ls_dir)
    pill_data_dir = pills_ls_dir
else:
    print("LS no disponible, usaremos el dataset original de Roboflow.")
    pill_data_dir = None""")

md("""### 9.2 `data.yaml` para segmentación

**Igual que detection.** No hay nada especial — YOLO infiere que es seg porque los `.txt` tienen polígonos en lugar de 5 columnas.""")

code("""# Para asegurar la clase, entrenamos sobre los datos de Roboflow
# (más imágenes que las 100 que el equipo etiquetó).
yaml_pills = f'''path: {Path(dataset_rf.location).absolute()}
train: train/images
val:   valid/images

names:
  0: pill
'''
yaml_path_pills = Path(dataset_rf.location) / "data_local.yaml"
yaml_path_pills.write_text(yaml_pills)
print(yaml_pills)""")

md("""### 9.3 Entrenar `yolo26n-seg`

Mismo comando que detection. El sufijo `-seg` le dice a Ultralytics que va a aprender polígonos.""")

code("""seg_pills = YOLO("yolo26n-seg.pt")
results_pills = seg_pills.train(
    data=str(yaml_path_pills),
    epochs=20,
    imgsz=640,
    batch=8,
    device=device,
    project="/content/runs",
    name="pills",
    exist_ok=True,
    verbose=False,
)
print("Pills: entrenamiento terminado.")

# Localizar best.pt (ruta exacta varía según versión)
PILL_BEST = sorted(glob.glob("/content/runs/**/pills/weights/best.pt", recursive=True))[0]
PILL_RUN_DIR = str(Path(PILL_BEST).parent.parent)
print(f"\\nbest.pt:  {PILL_BEST}")
print(f"run dir:  {PILL_RUN_DIR}")""")

md("""### 9.4 Métricas de segmentación

Cuando entrenas con un `-seg`, Ultralytics reporta **dos juegos de métricas**:
- `box.*` — métricas de la bbox derivada de la máscara.
- `seg.*` — métricas de la **máscara** (Mask IoU).

La que importa para segmentación es `seg.map50` y `seg.map`.""")

code("""best_pills = YOLO(PILL_BEST)
m_p = best_pills.val(data=str(yaml_path_pills), verbose=False)

print(f"BBox  mAP@0.5      = {m_p.box.map50:.3f}")
print(f"BBox  mAP@0.5:0.95 = {m_p.box.map:.3f}")
print()
print(f"Mask  mAP@0.5      = {m_p.seg.map50:.3f}    ← la que importa")
print(f"Mask  mAP@0.5:0.95 = {m_p.seg.map:.3f}")

# Mostrar las curvas de entrenamiento (results.png)
results_png = Path(PILL_RUN_DIR) / "results.png"
if results_png.exists():
    plt.figure(figsize=(14, 6))
    plt.imshow(Image.open(results_png)); plt.axis('off')
    plt.title("Curvas de entrenamiento — pills")
    plt.show()""")

md("""### 9.5 Guardamos el modelo de píldoras igual que el de tumor""")

code("""pill_model_path = models_dir / f"pills_yolo26n-seg_{today}.pt"
shutil.copy(PILL_BEST, pill_model_path)
print(f"Modelo de píldoras: {pill_model_path}")
print(f"Tamaño:             {pill_model_path.stat().st_size/1e6:.1f} MB")""")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Conteo + App Gradio
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 10. App de Conteo: el Producto Final

### 10.1 Conteo en una sola línea

Con segmentación, contar es trivial:""")

code("""# Tomamos una imagen de test
test_img = next((Path(dataset_rf.location) / "test" / "images").glob("*.jpg"))
res_pill = best_pills(str(test_img), conf=0.25, verbose=False)[0]

n_pills = len(res_pill.masks.data) if res_pill.masks is not None else 0
print(f"Píldoras detectadas: {n_pills}")

# Áreas individuales (puede ayudar a filtrar pastillas rotas)
if res_pill.masks is not None:
    areas = [int(m.sum().item()) for m in res_pill.masks.data]
    print(f"Área promedio: {np.mean(areas):.0f} px")
    print(f"Área mínima:   {np.min(areas)} px  (¿partida?)")
    print(f"Área máxima:   {np.max(areas)} px")

plt.figure(figsize=(12, 8))
plt.imshow(res_pill.plot()[..., ::-1])
plt.axis('off')
plt.title(f"Conteo: {n_pills} píldoras")
plt.show()""")

md("""### 10.2 Filtrar por tamaño (regla de negocio)

A veces el modelo detecta **fragmentos** (píldoras partidas). Si la regla del cliente es "solo contar píldoras enteras", filtramos por área:""")

code("""def contar_pildoras(img_path, modelo, conf=0.25, area_min=500):
    \"\"\"Conteo con filtro por área mínima.\"\"\"
    res = modelo(str(img_path), conf=conf, verbose=False)[0]
    if res.masks is None:
        return 0, 0, res

    areas = res.masks.data.sum(dim=(1, 2)).cpu().numpy()
    n_total   = len(areas)
    n_validas = int((areas >= area_min).sum())
    return n_total, n_validas, res

n_total, n_validas, res_full = contar_pildoras(test_img, best_pills, area_min=500)
print(f"Píldoras totales: {n_total}")
print(f"Píldoras enteras (área ≥ 500 px): {n_validas}")
print(f"Fragmentos descartados:           {n_total - n_validas}")

# Visualizar: pintar verde lo válido, rojo lo descartado
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(np.array(Image.open(test_img)))
ax.axis('off')

if res_full.masks is not None:
    areas_np = res_full.masks.data.sum(dim=(1, 2)).cpu().numpy()
    for poly, area in zip(res_full.masks.xy, areas_np):
        color = '#16A34A' if area >= 500 else '#C82B40'
        ax.add_patch(MplPoly(poly, alpha=0.45, fc=color, ec='white', lw=1.5))

ax.set_title(f"Conteo con filtro: {n_validas} enteras (verde) / "
              f"{n_total - n_validas} descartadas (rojo)",
              fontsize=12, fontweight='bold')
plt.show()""")

md("""### 10.3 App Gradio — interfaz para no-programadores

Gradio expone tu modelo como una **mini-app web** con dos líneas. Útil para:
- Validar el modelo con stakeholders (suben fotos y ven el conteo).
- Demos en una reunión.
- "MVP" antes de invertir en una API o app móvil.""")

code("""import gradio as gr

# Recargamos el modelo (si el kernel se reinició funciona igual)
modelo_pills = YOLO(str(pill_model_path))

def contar(imagen, conf, area_min):
    \"\"\"Función que Gradio expone: imagen → (imagen anotada, conteo).\"\"\"
    res = modelo_pills(imagen, conf=conf, verbose=False)[0]
    if res.masks is None:
        return imagen, "Sin detecciones"

    areas = res.masks.data.sum(dim=(1, 2)).cpu().numpy()
    n_total = len(areas)
    n_validas = int((areas >= area_min).sum())
    anotada = res.plot()[..., ::-1]  # BGR → RGB
    msg = f"Píldoras detectadas: {n_total}\\nEnteras (≥ {area_min} px): {n_validas}"
    return anotada, msg

demo = gr.Interface(
    fn=contar,
    inputs=[
        gr.Image(type="numpy", label="Imagen del blister"),
        gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confianza mínima"),
        gr.Slider(100, 2000, value=500, step=50, label="Área mínima (píxeles)"),
    ],
    outputs=[
        gr.Image(label="Detección"),
        gr.Textbox(label="Resultado", lines=2),
    ],
    title="Contador de Píldoras",
    description="Sube una foto del blister. El modelo segmenta cada píldora y cuenta.",
)

demo.launch(share=True, debug=False)""")

md("""> En Colab `share=True` te da un link `https://xxxxx.gradio.live` que puedes compartir 72 horas. Para producción real: deployar en Hugging Face Spaces (gratis) o detrás de un FastAPI propio.""")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Resumen
# ─────────────────────────────────────────────────────────────────────────────
md("""---
## 11. Resumen y Tarea

**Lo que hicimos hoy:**

1. Terminamos el flujo de **Brain Tumor**: bajar de LS → entrenar → métricas.
2. Aprendimos **qué métrica escoger** según el dominio (precision/recall/F1/mAP).
3. **Guardamos el modelo** (`best.pt`) y lo cargamos como si fuéramos un compañero distinto.
4. Exportamos a **ONNX** para deploy fuera de Python.
5. Introdujimos **segmentación**: cuándo, por qué, qué devuelve YOLO.
6. Usamos la API de seg: `masks.data`, `.xy`, `.xyn`, área desde máscara.
7. Vimos las **80 clases COCO** que `yolo26n-seg` reconoce gratis.
8. Bajamos un dataset de píldoras de **Roboflow Universe**.
9. Subimos 100 imágenes a **LS sin etiquetas**, entrenamos `yolo26n-seg`.
10. Construimos una **app Gradio** que cuenta píldoras desde una foto.

**Tarea para casa:**

1. Terminar de etiquetar las 100 imágenes de píldoras en LS (mínimo 10 por estudiante).
2. Re-entrenar `yolo26n-seg` solo con sus etiquetas (no las de Roboflow). Comparar `seg.map50`.
3. Probar la app Gradio con una foto **propia** (cualquier objeto contable: monedas, granos, frutas).
4. Para los curiosos: cambiar el labeling config a multi-clase (ej. `pill_red`, `pill_blue`) y re-entrenar.

**Próxima clase:** despliegue real — meter el modelo detrás de una API FastAPI, dockerizar y conectar con una app web.""")


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

    out = Path("/root/Arca/clase-28/Clase_28_Segmentacion.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Notebook escrito: {out}")
    print(f"# celdas: {len(cells)}")


if __name__ == "__main__":
    build()
