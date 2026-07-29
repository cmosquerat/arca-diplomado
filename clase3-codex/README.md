# Módulo 3 · Clase 3

**Cuando una recta no alcanza: no linealidad, árboles de decisión y Random Forest.**

## Uso en Google Colab

Abrir el notebook publicado desde:

<https://colab.research.google.com/github/cmosquerat/arca-diplomado/blob/agent/modulo3-clase3-codex/clase3-codex/Clase3_No_Linealidad_Arboles_RandomForest.ipynb>

La primera celda descarga automáticamente el dataset desde:

<https://raw.githubusercontent.com/cmosquerat/arca-diplomado/refs/heads/agent/modulo3-clase3-codex/clase3-codex/operacion_pozos_volve.csv>

No requiere montar Google Drive ni subir archivos manualmente.

## Contenido

- `presentacion.pdf`: presentación lista para impartir.
- `Clase3_No_Linealidad_Arboles_RandomForest.ipynb`: taller ejecutable en Colab.
- `operacion_pozos_volve.csv`: dataset limpio de operación de pozos productores Volve.
- `litologia_force2020.csv`: segundo dataset para clasificación arenisca/lutita.
- `build_materials.py`: generación reproducible del dataset, figuras y notebook a partir de la fuente pública.

## Reproducir el material

1. Descargar `Volve production data.xlsx` desde el repositorio público de [Volve-Field](https://github.com/jcreyesh/Volve-Field).
2. Ejecutar `python build_materials.py /ruta/Volve\ production\ data.xlsx`.
3. Compilar `pdflatex presentacion.tex`.

La fuente original es la hoja `Daily Production Data`; se conservan únicamente pozos productores, filas completas y días con producción y horas en línea positivas.
