# Prompts para imágenes — Clase 25

Lista de imágenes a generar con AI (DALL-E 3 / GPT-4o Image / Midjourney / Stable Diffusion). Las que son **diagramas técnicos o gráficos de datos** se quedan en TikZ/matplotlib (no las pongo aquí porque AI no las hace bien).

**Estilo unificado para todas**: estética profesional, fotorrealista cuando dice "foto", iluminación clean, fondos sin distracciones. Que parezcan de un libro de texto serio o caso industrial real.

---

## A. Hero / portada del bloque del proyecto

### `fig_hero_plate.png`
> **Prompt:** Photorealistic close-up of a red sedan car parked on a Latin American street. The license plate is clearly visible and **highlighted with a green bounding box drawn over it**. Above the bounding box, a clean overlay text shows the recognized plate text in white sans-serif font (e.g., "PCJ-3421"). Soft shadows, professional product-photography lighting, clean background. The image should feel like a "computer vision system in action" demo. Aspect ratio 16:9.

### `fig_hero_plate_dramatico.png`
> **Prompt:** Wide-angle photograph of a parking lot with multiple cars from above. Each car has a **green bounding box around its license plate** with the recognized text floating above the box in clean white letters (e.g., "ABC-123", "XYZ-789"). Daylight, slight grading toward dark/cinematic. Show ~5-8 cars detected simultaneously to convey "scaled detection". 16:9.

---

## B. Las 4 tareas de visión (uno por slide)

> Para estos quiero estilo **fotografía editorial limpia** con overlays mínimos. La idea es que la MISMA escena se muestre con 4 tipos diferentes de salida.

### `fig_task_classification.png`
> **Prompt:** Photograph of a single object on a clean neutral background — a soda can, photographed straight-on, professional product photography. Above the can, a single big text label appears: "Clase: lata" with confidence score 0.97. No bounding boxes, no segmentation. The image conveys: "the model says WHAT this is, nothing more". Aspect 16:9, clean background.

### `fig_task_detection.png`
> **Prompt:** Photograph of a busy supermarket aisle with multiple soda cans of different brands on a shelf. **Each can has a colored bounding box** drawn around it with a small label tag (e.g., "Coca", "Pepsi"). Multiple boxes visible — at least 8-10 cans detected. Clean professional retail photography, daylight. The image conveys "WHAT and WHERE". Aspect 16:9.

### `fig_task_segmentation.png`
> **Prompt:** Same supermarket aisle scene as detection, but instead of bounding boxes, **each can has a precise pixel-level mask overlay** in a translucent color (red for Coca, blue for Pepsi). Masks follow the exact contour of the cans. The image conveys "pixel-precise outline of each object". Aspect 16:9.

### `fig_task_pose.png`
> **Prompt:** Photograph of a worker on a beverage factory production line, wearing safety vest. **Skeleton overlay drawn on the worker** with colored dots at joints (head, shoulders, elbows, wrists, hips, knees) connected by thin colored lines. Industrial setting in background. The image conveys "key points of a person/object". Aspect 16:9.

### `fig_task_ocr.png`
> **Prompt:** Close-up photograph of a license plate or product label with text. **A green bounding box around the text region**, and floating above it the recognized text string in white clean sans-serif: e.g., "PCJ-3421" or "BEST BEFORE 12-2026". Conveys "reading text from an image". Aspect 16:9.

---

## C. CVAT y etiquetado

### `fig_cvat_screenshot.png`
> **Prompt:** Realistic screenshot of the CVAT.ai annotation interface on a desktop browser. Show: a car photograph in the main canvas with a green bounding box being drawn around the license plate. Left sidebar shows tools (rectangle, polygon, polyline, point), right sidebar shows label list with "license_plate" highlighted. UI in dark theme. Modern web app aesthetic. Aspect 16:9.

### `fig_etiquetado_humano.png`
> **Prompt:** Photo of a person sitting at a computer desk, focused on a screen showing license plate images with bounding boxes being drawn. Over-the-shoulder or side view. Modern office, soft natural lighting, focused atmosphere. Conveys "the labeling work behind every CV model". Aspect 16:9.

---

## D. Pipeline OCR (cascada detección → reconocimiento)

### `fig_ocr_cascade_hero.png`
> **Prompt:** A horizontal infographic-style image showing the pipeline as 4 stages with arrows between them:
> 1. **Input**: photo of a car (small thumbnail style)
> 2. **Step 1: Detect plate** — same car with a green bounding box around the plate
> 3. **Step 2: Crop** — close-up of just the cropped plate region
> 4. **Output: Read text** — text result floating "PCJ-3421"
>
> Each stage is a clean panel with a label below, connected by arrows. Modern flat design, professional. White background. Aspect 16:9.

---

## E. Industrial / contexto profesional (opcionales si quieres "ambientar")

### `fig_industria_inspeccion.png`
> **Prompt:** Wide photo of a beverage manufacturing line. Cans moving on a conveyor belt, an overhead camera mounted to inspect them, professional industrial setting, blue/silver tones, soft factory lighting. Conveys "computer vision deployed on a production line". Aspect 16:9.

### `fig_control_acceso_planta.png`
> **Prompt:** Photo of an entrance gate to an industrial plant with a vehicle approaching. A pole-mounted camera is visible, and overlay graphics suggest the camera is reading the vehicle's license plate. The gate is preparing to open. Industrial security context, daylight. Aspect 16:9.

---

## F. CTA / motivación

### `fig_motivacion_problema.png`
> **Prompt:** Split-image (2 panels side by side):
> - **Left panel**: a security guard at a parking lot booth manually writing down license plates from arriving cars on a clipboard. Tired, slow, manual work.
> - **Right panel**: a clean modern dashboard on a screen showing license plates being automatically read in real-time, displayed as a list with timestamps.
>
> Conveys the "before/after automation" story. Aspect 16:9.

---

## Cómo usar este archivo

1. Abre tu generador favorito (DALL-E 3 / GPT-4o Image / Midjourney).
2. Copia el prompt completo de cada figura.
3. Genera. Si la primera no convence, agregá variaciones tipo "more clinical / more industrial / less stylized".
4. Guarda con el nombre exacto de archivo indicado (`fig_hero_plate.png`, etc.) en `/clase-25/`.

## Lo que NO está en este archivo (queda en TikZ/matplotlib)

Estas figuras son técnicas y se generan por código (más nítidas, tipográficamente perfectas):

- `fig_yolo_timeline` — línea de tiempo YOLO 2015→2024
- `fig_yolo_family` — gráfico velocidad vs precisión
- `fig_bbox_anatomy` — esquema de bbox con (x,y,w,h)
- `fig_iou_diagram` — IoU geométrico
- `fig_nms_concept` — NMS antes/después
- `fig_grid_concept` — rejilla YOLO
- `fig_finetune_concept` — TL backbone+head (REUSAMOS de clase 24)
- `fig_keras_stack` — capas Keras (REUSAMOS de clase 24)
- `fig_pipeline_proyecto` — pipeline de pasos del proyecto
- `fig_formats_compare` — formatos YOLO/COCO/VOC
