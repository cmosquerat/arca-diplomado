# Plan Clase 25 — Detección de Placas + OCR

**Total: 120 min (2 horas)**

---

## Bloque A — Cierre Clase 24 (~30 min) — REUSO DE MATERIAL

> Aquí terminamos lo que quedó pendiente de clase 24. **Reusamos las figuras de transfer learning y el código de Coca vs Pepsi.** Los slides son los mismos / casi los mismos.

| Slide | Contenido | Origen |
|-------|-----------|--------|
| A.1 | Recap: el problema de los 4000 fotos sin alcanzar | Clase 24 |
| A.2 | Concepto de transfer learning (figura `fig_finetune.png` de clase 24) | Clase 24 reuso |
| A.3 | Por qué funciona: capas tempranas/medias/tardías | Clase 24 reuso |
| A.4 | ImageNet como dataset universal | Clase 24 reuso |
| A.5 | VGG16 como backbone simple | Clase 24 reuso |
| A.6 | Tabla VGG vs nuestra CNN | Clase 24 reuso |
| A.7 | Código transfer learning con VGG16 | Clase 24 reuso |
| A.8 | Demo en vivo: entrenar Coca vs Pepsi (5 min) | Notebook clase 24 |
| A.9 | Resultados Coca vs Pepsi | Notebook clase 24 |
| A.10 | **Limitación clave**: el modelo solo dice "es Coca o Pepsi" para UNA imagen entera. No cuenta, no ubica. **¿Y si quiero saber DÓNDE está cada lata?** | Puente al bloque B |

**Cierre del bloque**: la pregunta abierta — "¿y si tengo 10 productos en una foto?" → motiva detección.

---

## Bloque B — Tareas de Visión por Computadora (~25 min)

> Panorama amplio antes de meternos en detección. Para que entiendan que *clasificación es solo una de muchas tareas*.

| Slide | Contenido | Figura |
|-------|-----------|--------|
| B.1 | Las 5 tareas principales en visión: panorama | tabla resumen |
| B.2 | **Clasificación** detallada (lo que ya saben hacer) | `fig_task_classification.png` |
| B.3 | **Detección de objetos** detallada | `fig_task_detection.png` |
| B.4 | **Segmentación** (semantic + instance) detallada | `fig_task_segmentation.png` |
| B.5 | **Pose estimation / keypoints** detallada | `fig_task_pose.png` |
| B.6 | **OCR** (reconocimiento de texto en imágenes) detallada | `fig_task_ocr.png` |
| B.7 | Tabla de decisión: "qué tarea para qué problema" | tabla |
| B.8 | "Hoy nos enfocamos en detección + OCR. Veamos por qué." | texto |

**Cierre del bloque**: introducimos el problema concreto del proyecto.

---

## Bloque C — El Problema y los Bounding Boxes (~15 min)

> Ahora baja la lupa a detección.

| Slide | Contenido | Figura |
|-------|-----------|--------|
| C.1 | **El problema concreto**: leer la placa de un vehículo automáticamente. Caso real LATAM. | `fig_motivacion_problema.png` |
| C.2 | El pipeline en 2 pasos: detectar → leer (cascada) | `fig_ocr_cascade_hero.png` |
| C.3 | Anatomía de un bounding box (x, y, w, h, clase) | `fig_bbox_anatomy.png` (TikZ) |
| C.4 | IoU como métrica de calidad | `fig_iou_diagram.png` (TikZ) |
| C.5 | Por qué CNN clásica falla → necesitamos detector | comparación tabla |

---

## Bloque D — YOLO en Profundidad (~20 min)

| Slide | Contenido | Figura |
|-------|-----------|--------|
| D.1 | La idea YOLO: una sola pasada de CNN | texto + diagrama |
| D.2 | Rejilla de celdas | `fig_grid_concept.png` (TikZ) |
| D.3 | NMS — limpiar duplicados | `fig_nms_concept.png` (TikZ) |
| D.4 | Historia: 2015 (Redmon) → 2024 (Ultralytics) | `fig_yolo_timeline.png` |
| D.5 | Familia YOLOv8/v11: nano, small, medium, large, xlarge | `fig_yolo_family.png` |
| D.6 | Trade-off velocidad vs precisión | tabla |
| D.7 | Hardware: ¿necesito GPU? | tabla |

---

## Bloque E — Métricas (~10 min)

| Slide | Contenido | Figura |
|-------|-----------|--------|
| E.1 | mAP@0.5 y mAP@0.5:0.95 explicados | texto |
| E.2 | Precision y recall por clase | tabla |
| E.3 | Trade-offs prácticos: confidence threshold, imgsz | tabla |

---

## Bloque F — Etiquetado con CVAT (~15 min)

| Slide | Contenido | Figura |
|-------|-----------|--------|
| F.1 | ¿Qué significa "etiquetar"? | texto + diagrama |
| F.2 | Software open source: CVAT vs Label Studio vs labelImg | tabla |
| F.3 | Roboflow como freemium (mención) | tabla |
| F.4 | Demo CVAT en vivo: dibujar bbox sobre placa | `fig_cvat_screenshot.png` |
| F.5 | Formatos de exportación: YOLO, COCO, VOC | `fig_formats_compare.png` (matplotlib) |

---

## Bloque G — Fine-Tuning de YOLO (~10 min)

> **Conexión clave**: misma idea que vieron en TL del bloque A, ahora aplicada a YOLO. *No es un concepto nuevo, es el mismo*.

| Slide | Contenido | Figura |
|-------|-----------|--------|
| G.1 | "Recordemos transfer learning…" → mismo principio | `fig_finetune.png` (REUSO clase 24) |
| G.2 | Aplicado a YOLO: backbone preentrenado en COCO + cabeza nueva para placas | diagrama |
| G.3 | Pretrained YOLO: ¿detecta placas? **No, COCO no tiene esa clase** | demo antes/después |
| G.4 | El código en una línea: `model.train(data="data.yaml", epochs=30)` | code |
| G.5 | El archivo `data.yaml` explicado | code |

---

## Bloque H — El Proyecto (~15 min)

| Slide | Contenido | Figura |
|-------|-----------|--------|
| H.1 | **El reto colectivo**: detector + OCR de placas | `fig_hero_plate_dramatico.png` |
| H.2 | Dataset: ~200 imágenes sin etiquetar disponibles en `clase-25/plates_unlabeled.zip` | screenshot |
| H.3 | Pipeline completo del proyecto: 5 pasos | `fig_pipeline_proyecto.png` (TikZ) |
| H.4 | Tu tarea para la próxima clase: etiquetar tu lote en CVAT y subir a Drive | enumerate |
| H.5 | Mini-tutorial CVAT en 7 pasos | enumerate |

---

## Bloque I — Cierre (~5 min)

| Slide | Contenido |
|-------|-----------|
| I.1 | Resumen de las 5 ideas clave |
| I.2 | Qué viene la próxima clase: consolidar dataset, fine-tunear YOLO, agregar OCR, app Gradio |
| I.3 | QR encuesta |

---

## Notas técnicas

### YOLO version a usar
**YOLO11 (Ultralytics, sep 2024)** — última versión estable. La API es idéntica a v8 (`from ultralytics import YOLO; YOLO("yolo11n.pt")`). Si confirmas usamos esa.

### Dataset que pre-cargo
Sub-muestra de **RodoSol-ALPR** (placas brasileñas en autopistas) — ~200 imgs, sin labels. La hosteo como `clase-25/plates_unlabeled.zip`. Las placas brasileñas son visualmente similares a las ecuatorianas (alfanuméricas, una línea).

### Reuso de figuras de clase 24
- `fig_finetune.png` (concepto TL)
- `fig_keras_stack.png` (Keras + backend)
- `fig_imagenet.png` (ImageNet info)
- `fig_vgg_vs_nuestra.png` (comparación)

### Figuras nuevas a generar
**Por AI** (en `IMAGE_PROMPTS.md`):
- `fig_hero_plate.png`
- `fig_hero_plate_dramatico.png`
- `fig_task_classification/detection/segmentation/pose/ocr.png` (5 figuras)
- `fig_cvat_screenshot.png`
- `fig_etiquetado_humano.png`
- `fig_ocr_cascade_hero.png`
- `fig_motivacion_problema.png`
- `fig_industria_inspeccion.png` (opcional)
- `fig_control_acceso_planta.png` (opcional)

**Por TikZ/matplotlib** (yo las hago):
- `fig_bbox_anatomy.png`
- `fig_iou_diagram.png`
- `fig_nms_concept.png`
- `fig_grid_concept.png`
- `fig_yolo_timeline.png`
- `fig_yolo_family.png`
- `fig_pipeline_proyecto.png`
- `fig_formats_compare.png`
