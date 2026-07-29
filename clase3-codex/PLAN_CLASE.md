# Módulo 3 · Clase 3

## Título propuesto

**Cuando una recta no alcanza: no linealidad, árboles de decisión y Random Forest**

## Decisión de diseño

Usar **un dataset nuevo**, de producción de Volve, como caso principal. Las clases anteriores ya usaron registros de pozo y litología; cambiar a un *medidor virtual de flujo* evita la repetición y permite volver a la regresión de la Clase 1 desde una situación operacional: estimar petróleo producido por día con variables de operación.

El hilo de la sesión es deliberadamente continuo:

```text
Clase 1: una recta predice un número
Clase 2: una sigmoide decide entre dos clases y el costo cambia la decisión
Clase 3: la realidad tiene umbrales e interacciones; un árbol aprende esas reglas
          y un bosque las hace más robustas
```

No dedicar la clase a SMOTE o a desbalance. El umbral y el costo ya se trabajaron bien en Clase 2. `class_weight` aparecerá solamente como extensión de 5 minutos: no es una receta automática y se conserva solo si mejora la métrica/costo definido.

## Resultado de aprendizaje

Al terminar, cada participante podrá reconocer cuándo la relación entre variables no puede representarse bien con una recta, entrenar e interpretar un árbol de decisión y comparar de forma honesta un árbol con Random Forest usando un conjunto de prueba.

## Caso principal: medidor virtual de flujo (Volve)

**Pregunta:** “Con horas en línea, presiones, temperatura y choke, ¿cuántos Sm³/día de petróleo producirá el pozo?”

Variables candidatas:

- `ON_STREAM_HRS`
- `AVG_DOWNHOLE_PRESSURE`
- `AVG_WHP_P`
- `AVG_WHT_P`
- `AVG_CHOKE_SIZE_P`
- `DP_CHOKE_SIZE`
- objetivo: `BORE_OIL_VOL`

Fuente de datos prevista: hoja **Daily Production Data** del archivo de producción Volve. El CSV existente en `clase-03/volve_produccion.csv` es útil para contextualizar Volve, pero no incluye presión, temperatura ni choke; por tanto **no debe ser el dataset del notebook central**.

La exploración preliminar del trabajo anterior mostró un contraste suficientemente grande para una clase:

| Modelo | R² de prueba | MAE aproximado |
|---|---:|---:|
| Regresión lineal | 0.595 | 583 Sm³/día |
| Árbol (`max_depth=5`) | 0.886 | 270 Sm³/día |
| Random Forest | 0.981 | 78 Sm³/día |

Estos números deben recalcularse en el notebook final con el archivo de datos versionado y con el split documentado. Son un criterio de viabilidad pedagógica, no una promesa de producción.

## Estructura sugerida de la sesión

| Bloque | Tiempo (para una sesión de 3 h) | Idea que debe quedar |
|---|---:|---|
| Recapitulación y reto | 10 min | Ya sabemos predecir números y tomar decisiones con probabilidades. |
| La recta falla | 25 min | Una relación puede ser curva, tener zonas o depender de combinaciones de variables. |
| De “dos lunas” al pozo | 20 min | La frontera curva se entiende visualmente antes de verla en datos reales. |
| Árbol de decisión | 35 min | El árbol aprende preguntas/cortes del tipo “si choke < x…”. |
| Taller 1 | 30 min | Entrenar, visualizar y limitar profundidad. |
| Overfitting | 20 min | Un resultado perfecto en entrenamiento no demuestra que el modelo sea útil. |
| Random Forest | 30 min | Muchos árboles distintos, promediados, reducen la fragilidad de un solo árbol. |
| Taller 2 y cierre | 30 min | Comparar modelos, interpretar importancia y elegir según el objetivo. |

Si la sesión dura 2 h, conservar los dos talleres y reducir la explicación de impureza/Gini y el segmento opcional de `class_weight`.

## Secuencia de diapositivas

1. **Portada y pregunta guía.** “¿Qué pasa cuando abrir un poco más el choke no aumenta linealmente la producción?”
2. **Dónde estamos.** Línea Clase 1 → Clase 2 → Clase 3.
3. **Predicción no es siempre una recta.** Ejemplos cotidianos: dosis–respuesta, velocidad–consumo, presión–flujo.
4. **La trampa de la recta.** Mostrar una curva y el residuo sistemático que deja una regresión lineal.
5. **Demo sintética: dos lunas.** Recta/logística fallan; la frontera correcta es curva. Es una visualización, no el caso de negocio.
6. **Regreso al pozo.** Esquema choke–presiones–producción y pregunta del medidor virtual.
7. **Baseline honesto.** Regresión lineal; gráfico predicho vs. real y MAE/R².
8. **La intuición del árbol.** “Si GR < 60 es arena” se convierte aquí en “si presión/choke cruza un valor, toma otra regla”.
9. **Anatomía de un árbol.** Nodo, regla, rama, hoja y predicción promedio de la hoja.
10. **Cómo escoge un corte.** Reducir variación (MSE en regresión); sin derivación matemática pesada.
11. **El árbol ve zonas, no rectas.** Superficie escalonada/particiones; explicar su fortaleza y límite.
12. **Práctica 1.** Entrenar un árbol pequeño, visualizarlo y leer tres reglas en voz alta.
13. **Profundidad: aprender o memorizar.** Curvas train/test para varias profundidades.
14. **El árbol sin límite.** R²=1 en train como señal de alerta, no como trofeo.
15. **Puente.** “¿Podemos conservar las reglas y reducir la fragilidad?”
16. **Random Forest.** Bootstrap + subconjunto aleatorio de variables + promedio; dibujo de un bosque votando/promediando.
17. **Comparación justa.** Mismo split, mismas variables, misma métrica: lineal vs. árbol vs. bosque.
18. **Importancia de variables.** Qué variables usó más el bosque; advertir que importancia no significa causalidad.
19. **Práctica 2.** Cambiar `max_depth`, `n_estimators` y `min_samples_leaf`; registrar train, test y MAE.
20. **Extensión breve: clases desbalanceadas.** `class_weight` solo se acepta si mejora el costo acordado; enlaza con Clase 2.
21. **Cierre.** Checklist: baseline, split, generalización, interpretación y objetivo operacional.
22. **Próxima clase.** Validación cruzada, ajuste de hiperparámetros y split por pozo para evitar leakage geológico/operacional.

## Notebook: recorrido mínimo

1. Cargar y limpiar datos; quitar filas sin mediciones, `oil <= 0` y `horas <= 0`.
2. EDA corta: distribución de `oil` y dos gráficos de dispersión (`choke`/presión versus `oil`).
3. `train_test_split` reproducible y baseline `LinearRegression`.
4. `DecisionTreeRegressor(max_depth=5)`; visualizar árbol y medir R²/MAE en train y test.
5. Bucle de profundidades para mostrar sobreajuste.
6. `RandomForestRegressor`; comparar con una tabla y gráfico predicho vs. real.
7. Gráfico de `feature_importances_` y reflexión de ingeniería.
8. Reto: buscar el mejor modelo bajo un límite explícito de complejidad e informar por qué se eligió.

## Límites que se deben decir explícitamente

- Un split aleatorio por filas puede mezclar días del mismo pozo; hoy sirve para aprender el modelo, pero puede sobreestimar su desempeño operacional.
- La importancia del modelo no prueba causalidad ni reemplaza criterio de yacimientos/producción.
- El bosque gana precisión, pero pierde parte de la explicabilidad inmediata del árbol único.
- No escalar las variables para árboles no es un descuido: los cortes dependen del orden, no de la escala. Contrasta con las clases anteriores.

## Entregables para construir después de aprobar el plan

- `presentacion.tex` con unas 22 diapositivas y la estética de `modulo3-clase2`.
- `Clase3_No_Linealidad_Arboles_RandomForest.ipynb`, con celdas de actividad y una solución separada si se desea.
- `gen_figs.py` y figuras reproducibles: dos lunas, falla lineal, árbol, overfitting, bosque y comparación final.
- Dataset de producción Volve versionado o un script de descarga con fuente y checksum.
