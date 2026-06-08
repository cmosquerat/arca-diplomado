"""
Construye Clase_37_Agentes.ipynb

Enfoque FINAL (aprobado por Carlos, con refinamientos):
  - El alumno SOLO escribe en español. El agente programa por ellos (CodeAgent).
  - EXPLICAR TODO y la ESTRUCTURA (qué es un agente, bucle ReAct, parámetros,
    cómo leer el código que genera).
  - EMPEZAR SIMPLE (cálculos), luego datos. Nada de "escribe tu pregunta" abierto:
    ejercicios CONCRETOS, problemas reales con respuesta.
  - PROMPTING: mal prompt REALISTA ("¿cuánto vendimos?") vs bueno. Con ejercicios.
  - EJERCICIOS INTERCALADOS, concretos, con peso.
  - Explorar HERRAMIENTAS de smolagents (búsqueda web).
  - CAPSTONE: "Asistente Estadístico de Ventas Arca" — app FIJA y determinista
    (dataset fijo, preguntas estadísticas, sin subir archivos, nada al azar).
  - Datos reales: ventas Ecuador (Favorita), World Happiness, Mall Customers.
  - Cerebro Groq Llama 3.3 70B.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md   = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =============================================================================
# PORTADA
# =============================================================================
md("""# Clase 37 — Agentes: el analista que programa por ti

*Diplomado en Data Science Aplicada con Python · Arca Continental Ecuador · UDLA*

---

**Pregunta del día:**
> En la clase 36 construimos un agente *a mano*: el bucle, las herramientas, el parseo
> del JSON. Funcionó, pero fue mucho código y muy frágil. Hoy le damos la vuelta: **le
> hablamos en español y el agente escribe y ejecuta el código Python por nosotros.**

**Al final vas a:**
- Entender **cómo funciona un agente por dentro** (el bucle, no la magia).
- Pedirle cálculos y análisis en español; él programa.
- Escribir **buenos prompts** (la habilidad clave de hoy).
- Darle **herramientas** (búsqueda web).
- Construir una **app real**: el *Asistente Estadístico de Ventas de Arca*.

> Tu trabajo hoy: **escribir buenas preguntas en español**. El agente escribe el código.""")

# =============================================================================
# PARTE 0 — Qué es un agente
# =============================================================================
md("""---

# Parte 0 — ¿Qué es un agente? (entender, no memorizar)

Un **modelo de lenguaje (LLM)** es una caja que recibe texto y devuelve texto. Es
brillante prediciendo palabras, pero por sí sola **no hace nada más**: no ejecuta una
cuenta, no abre tu archivo de ventas, no dibuja un gráfico. Solo *habla*.

Un **agente** es esa caja metida en un **bucle que le deja actuar**:

1. **Piensa** — razona qué hacer.
2. **Actúa** — produce una acción (aquí: *escribe código Python*).
3. **Observa** — un programa ejecuta esa acción y le devuelve el resultado.
4. **Repite** — con ese resultado vuelve a pensar; sigue hasta poder responder.

Ese ciclo *pensar → actuar → observar* se llama **ReAct** (de *Reason + Act*, 2022). Es
la base de casi todos los agentes de hoy.

### Dos familias de agentes

| Familia | Cómo "actúa" | Cuándo conviene |
|---|---|---|
| **Tool-calling agent** | elige entre **funciones que tú preparaste** | tareas fijas |
| **Code agent** | **escribe código nuevo** y lo ejecuta | tareas abiertas |

Hoy usamos un **Code agent**: como escribe cualquier código, no está limitado a
funciones pre-hechas. Tú preguntas en español; él inventa el código necesario.""")

# =============================================================================
# PASO 0 — Setup
# =============================================================================
md("""---

# Paso 0 — Preparar el entorno

Usamos **smolagents** (librería de agentes de Hugging Face), fijada en `1.26.0`.
`[litellm]` = usar cualquier modelo (hoy Groq); `[toolkit]` = herramientas listas
(búsqueda web); `[gradio]` = la app web. `pandas`/`matplotlib` los usa el **agente**.""")

code("""!pip install -q "smolagents[litellm,toolkit,gradio]==1.26.0" pandas matplotlib""")

md("""### La API key de Groq

Una **API key** identifica tu cuenta ante el proveedor del modelo. La de **Groq** es
gratis: [console.groq.com](https://console.groq.com) → *API Keys*. La pedimos con
`getpass` (no se ve al teclear) y la guardamos en `GROQ_API_KEY`.""")

code("""import os, getpass
import pandas as pd

os.environ["GROQ_API_KEY"] = getpass.getpass("Pega tu GROQ_API_KEY y presiona Enter: ")
print("Listo, key guardada.")""")

# =============================================================================
# PARTE 1 — Estructura
# =============================================================================
md("""---

# Parte 1 — La estructura de un agente (esto es lo importante)

Dos piezas y un método.""")

md("""## Pieza 1 — El modelo (el "cerebro")

`LiteLLMModel` habla con muchos proveedores; le decimos cuál con `model_id`
(`groq/` = proveedor, `llama-3.3-70b-versatile` = modelo). `temperature=0.0` = estable,
para que **escriba código correcto**, no creativo.""")

code("""from smolagents import LiteLLMModel

modelo = LiteLLMModel(model_id="groq/llama-3.3-70b-versatile", temperature=0.0)
print("Modelo listo:", modelo.model_id)""")

md("""> **¿Y si quiero otro modelo?** Cambiar de cerebro es **una línea**. Hoy usamos Groq
> (gratis, rápido y fiable). Otra opción gratis y muy buena en código es **Qwen3-Next
> Instruct** (vía OpenRouter, con su propia API key):
> ```python
> modelo = LiteLLMModel("openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
>                       api_key="TU_OPENROUTER_KEY")
> ```
> Evita las versiones "Thinking": son más lentas y a veces confunden al agente.""")

md("""## Pieza 2 — El agente (`CodeAgent`)

`CodeAgent` envuelve el modelo en el bucle ReAct y le da un **intérprete de Python**.
Cada parámetro importa:
- **`tools=[]`** — herramientas pre-hechas; vacía a propósito: que escriba su código.
- **`additional_authorized_imports`** — **lista blanca** de librerías que su código
  puede importar (es **seguridad**: por defecto casi no puede importar nada).
- **`max_steps=6`** — cuántas vueltas del bucle; **freno de costo** y de bucles infinitos.""")

code("""from smolagents import CodeAgent

agente = CodeAgent(
    tools=[],
    model=modelo,
    additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot"],
    max_steps=6,
)
print("Agente listo.")""")

md("""## El uso — `agente.run("...")` y cómo leer su salida

smolagents imprime cada vuelta con tres bloques que conviene leer:

```
─ Step 1 ──────────────
Thought:       (su plan en palabras)
Code:          (el código Python que escribió)   ← lo valioso
Observations:  (lo que salió al ejecutarlo)
─ ... Final answer: (la respuesta en español)
```

> 🔑 **Mira siempre el bloque `Code`:** ahí está el código que el agente escribió por ti.""")

# =============================================================================
# PARTE 2 — Simple
# =============================================================================
md("""---

# Parte 2 — Primeros pasos: tareas simples (sin datos)

Veamos al agente resolver algo sencillo para entender el mecanismo. Le pedimos una
cuenta de varios pasos; él escribe el código y lo ejecuta.""")

code('''agente.run(
    "Una caja de 24 gaseosas cuesta 18 dólares. Compro 50 cajas y me dan 15% de "
    "descuento sobre el total. ¿Cuánto pago al final? Muestra el cálculo y el monto en dólares."
)''')

md("""👆 Mira el bloque `Code`: el agente escribió la operación (total → descuento → final)
y la **ejecutó**. No "recordó" el número, lo **calculó**. Esa es la diferencia con un LLM
normal, que podría inventar una cifra parecida.""")

md("""## 🏋️ Ejercicio 1 — Asesora a un distribuidor

**Caso:** un distribuidor va a comprar gaseosas y te pide consejo. Arca tiene una
promoción: *por cada 10 cajas, 1 gratis*. Él planea comprar **250 cajas a 18 USD** cada
una y quiere saber si le conviene.

Usa el agente para **averiguar cuánto le sale realmente cada caja** (contando las gratis),
y así poder aconsejarlo. **Tú decides cómo preguntárselo** — y revisa el bloque `Code`
para confirmar que el cálculo tiene sentido.""")

code('''agente.run(
    ""   # escribe aquí tu pregunta, en español
)''')

# =============================================================================
# PARTE 3 — Prompting
# =============================================================================
md("""---

# Parte 3 — Prompting: la habilidad clave de hoy

El agente es tan bueno como **tu pregunta**. Primero cargamos los datos reales (los
usamos para ver buenos vs malos prompts).""")

code("""BASE = "https://raw.githubusercontent.com/cmosquerat/arca-diplomado/main/clase-37"

ventas    = pd.read_csv(f"{BASE}/ventas_tiendas_ecuador.csv")  # Favorita Ecuador (real)
felicidad = pd.read_csv(f"{BASE}/world_happiness.csv")
clientes  = pd.read_csv(f"{BASE}/mall_customers.csv")

print("ventas:", ventas.shape)
display(ventas.head(3))""")

md("""Le pasamos un DataFrame al agente con `additional_args` (el agente vive en su propio
intérprete y no ve las variables del notebook):

```python
agente.run("tu pregunta", additional_args={"ventas": ventas})
```

### El mismo objetivo, mal prompt vs. buen prompt

Las técnicas de un buen prompt:
1. **Sé específico** — qué número, qué tienda, qué año.
2. **Pide el formato** — "solo el número", "una tabla", "un gráfico".
3. **Da contexto** — qué representan los datos.
4. **Divide** las tareas complejas.""")

md("""#### ❌ Prompt malo (realista: así pregunta un gerente apurado)""")
code('''agente.run("¿cuánto vendimos?", additional_args={"ventas": ventas})''')

md("""👆 Es un prompt **realista** —cualquiera lo escribiría— pero está mal: no dice qué
producto, qué tienda, qué año, ni si quiere unidades o dinero. El agente tiene que
**adivinar** (toma todo el total, o elige algo arbitrario). La respuesta no sirve para decidir.""")

md("""#### ✅ Prompt bueno (específico + formato + contexto)""")
code('''agente.run(
    "Con el DataFrame 'ventas', ¿cuántas unidades de bebidas se vendieron en total en "
    "la tienda Quito-44 durante 2016? Dame solo el número.",
    additional_args={"ventas": ventas},
)''')

md("""👆 Mismo agente, mismo dato. La diferencia fue **el prompt**. Esto es lo más
importante de hoy: **aprender a pedir bien.**""")

md("""## 🏋️ Ejercicio 2 — Convierte un mal prompt en uno útil

**Caso:** el gerente de Guayaquil-51 va a decidir cuánto inventario de bebidas pedir, y
lo único que escribió fue: `"¿cómo van las ventas?"`.

1. **Córrelo** tal cual y observa por qué no le sirve para decidir.
2. **Reescríbelo** para que el agente le dé algo **accionable** para esa decisión (un
   número, un periodo, una comparación — tú decides qué le sirve). Aplica al menos 2
   técnicas de prompting.
3. **Compara** ambas respuestas.""")

code('''# Parte 1 — el prompt vago, tal cual
agente.run("¿cómo van las ventas?", additional_args={"ventas": ventas})''')

code('''# Parte 2 — tu versión útil para decidir el inventario
agente.run(
    "",   # escribe aquí tu prompt mejorado
    additional_args={"ventas": ventas},
)''')

# =============================================================================
# PARTE 4 — Análisis real + gráficas
# =============================================================================
md("""---

# Parte 4 — Análisis y gráficas reales

Aplicando buen prompting, el agente resuelve preguntas de negocio y dibuja gráficas
(escribe el matplotlib él mismo). Columnas de `ventas`: `tienda`, `categoria`, `anio`,
`mes`, `unidades`.""")

code("""agente.run(
    "Con 'ventas', haz un gráfico de barras del total de unidades de bebidas por año en "
    "Quito-44. Ponle título y muéstralo con plt.show().",
    additional_args={"ventas": ventas},
)""")

md("""## 🏋️ Ejercicio 3 — Planificación de inventario

**Caso:** el gerente de Quito-44 va a planificar el inventario del próximo año y quiere
entender **en qué meses se concentran sus ventas de bebidas** (la estacionalidad), para no
quedarse corto en temporada alta.

Pídele al agente lo que necesites para responderle con datos. Tú decides qué preguntar y
cómo lo quieres ver (un número, una tabla, un gráfico).""")

code("""agente.run(
    "",   # escribe aquí tu prompt
    additional_args={"ventas": ventas},
)""")

md("""## 🏋️ Ejercicio 4 — Una lámina para el comité

**Caso:** en la reunión mensual quieren comparar el **desempeño de las dos tiendas**.
Prepara el **apoyo visual** que presentarías. Pídele al agente el gráfico que mejor cuente
esa historia — tú decides qué comparar y qué tipo de gráfico.""")

code("""agente.run(
    "",   # escribe aquí tu prompt para el gráfico
    additional_args={"ventas": ventas},
)""")

# =============================================================================
# PARTE 5 — Herramientas
# =============================================================================
md("""---

# Parte 5 — Darle herramientas al agente

Además de escribir código, el agente puede usar **herramientas listas**. Una útil es la
**búsqueda web** (`WebSearchTool`). Tú no la programas: la enchufas en `tools=[...]`.""")

code("""from smolagents import WebSearchTool

agente_web = CodeAgent(
    tools=[WebSearchTool()],
    model=modelo,
    additional_authorized_imports=["pandas"],
    max_steps=6,
)
agente_web.run("¿En qué año se fundó Arca Continental? Búscalo en la web.")""")

md("""👆 El agente **decidió** usar la búsqueda, la llamó y respondió. Le diste una
capacidad nueva sin escribir código.

> Otras herramientas de smolagents: leer páginas web, ejecutar Python, transcribir
> audio; y conectarse a sistemas externos vía el estándar **MCP**.""")

md("""## 🏋️ Ejercicio 5 — Contexto de mercado (buscar + calcular)

**Caso:** para una decisión necesitas un dato **actual** del mundo real que NO está en tus
datos (por ejemplo: el precio internacional de un insumo, la inflación de Ecuador, el tipo
de cambio del día, etc.).

Usa el `agente_web` para **traer ese dato de internet y combinarlo con un cálculo** que le
sirva al negocio. Tú decides qué dato buscar y qué cálculo hacer. Observa en los pasos
cuándo **busca** y cuándo **calcula**.""")

code("""agente_web.run(
    ""   # escribe aquí tu prompt: un dato real de internet + un cálculo útil
)""")

# ---- 5.2 Generación de imágenes
md("""## Otra herramienta: generar imágenes 🎨

smolagents puede usar como herramienta **cualquier "Space" de Hugging Face** (una mini-app
publicada). Conectamos un generador de imágenes (**FLUX**) con `Tool.from_space`: el agente
podrá **crear imágenes** a partir de una descripción. Otra vez: tú no programas la
herramienta, solo se la enchufas.""")

code("""from smolagents import Tool

generar_imagen = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",   # un Space público de generación de imágenes
    name="generar_imagen",
    description="Genera una imagen a partir de una descripción de texto.",
)

agente_creativo = CodeAgent(tools=[generar_imagen], model=modelo, max_steps=4)

# El agente llama a la herramienta y devuelve la imagen (puede tardar: el Space hace cola)
agente_creativo.run(
    "Genera una imagen para una campaña: una lata de gaseosa sobre una mesa de madera en "
    "una playa tropical de Ecuador al atardecer, estilo fotografía publicitaria."
)""")

md("""👆 El agente entendió el pedido, llamó al generador y devolvió la imagen. Igual que
con la búsqueda web: una capacidad nueva, sin escribir código.

> ⚠️ El Space de FLUX es gratis pero **puede hacer cola** (tarda unos segundos o más). Si
> falla, vuelve a intentar. Los modelos de imagen entienden mejor las descripciones **en
> inglés**, aunque tú escribas el pedido en español.""")

md("""## 🏋️ Ejercicio (didáctico) — Prompting de imágenes

Las imágenes también dependen del prompt. Brief concreto:

1. Pídele al `agente_creativo` una imagen para una **valla publicitaria** de una bebida de
   Arca (di el producto, el lugar, la hora del día y el estilo).
2. Ahora **cambia UN detalle** (por ejemplo, la hora del día o el lugar) y genera otra.
3. En una celda de texto, escribe **qué cambió en la imagen al cambiar esa palabra**.

Así ves, en lo visual, lo mismo de la Parte 3: **mejor prompt, mejor resultado.**""")

code("""# 1) Tu primera imagen
agente_creativo.run(
    ""   # tu brief: producto + lugar + hora del día + estilo
)""")

code("""# 2) La segunda, cambiando UN solo detalle
agente_creativo.run(
    ""   # el mismo brief, cambia UN solo detalle
)""")

# =============================================================================
# PARTE 6 — CAPSTONE
# =============================================================================
md("""---

# Parte 6 — Proyecto: el *Asistente Estadístico de Ventas de Arca* 🏆

Ahora construyes una **app real, enfocada y sin sorpresas**: una herramienta interna que
responde **preguntas estadísticas sobre el dataset fijo de ventas** de Arca. Nada de
subir archivos, nada abierto — un asistente que el equipo usa para consultar estadísticas.

**Escenario:** el equipo comercial te pide cifras todo el día. En vez de responder lo
mismo cada vez, montas un asistente al que le preguntan directo.

### Tu único trabajo: escribir el **system prompt** (en español)

El system prompt (`instructions`) define **quién es el asistente, qué datos tiene y sus
reglas**. Es el corazón de la app — y es puro lenguaje natural.""")

code('''from smolagents import GradioUI

# 👇 ESTE es tu trabajo: completa/ajusta el system prompt en español.
INSTRUCCIONES = f"""Eres el Asistente Estadístico de Ventas de Arca Continental.

Datos: cárgalos SIEMPRE con pandas desde esta URL (no aceptes otros archivos):
{BASE}/ventas_tiendas_ecuador.csv
Columnas: tienda (valores: Quito-44, Guayaquil-51), categoria (bebidas, lacteos),
anio (2013 a 2017), mes (1 a 12), unidades.

Qué puedes responder: estadísticas sobre ESE dataset — totales, promedios, medianas,
desviación estándar, mínimos/máximos, crecimiento porcentual, comparaciones entre
tiendas o años, correlaciones y tendencias. Si te piden un gráfico, hazlo con matplotlib.

Reglas:
- Responde en español, breve y con el número concreto.
- Si la pregunta no es sobre este dataset de ventas, responde que solo analizas las
  ventas de Arca y no respondas otra cosa.
- Nunca inventes cifras: calcúlalas siempre con el código sobre los datos.
"""

asistente = CodeAgent(
    tools=[],
    model=modelo,
    additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot", "numpy"],
    instructions=INSTRUCCIONES,
    max_steps=6,
)

# Levanta la app (se queda corriendo; detén la celda con ⏹ cuando termines)
GradioUI(asistente).launch(share=True)''')

md("""## 🏋️ Ejercicio 6 — Pon a prueba tu asistente

Abre la app (link `https://...gradio.live`) y hazle **al menos 3 preguntas estadísticas
tuyas** sobre las ventas. Invéntalas tú — por ejemplo, una de cada tipo:
- un **promedio** o una **mediana**,
- una **comparación** entre las dos tiendas,
- un **crecimiento porcentual** entre dos años.

**Reto:** pregúntale algo **fuera** del dataset (ej. *"¿cuál es la capital de Francia?"*) y
comprueba que **se niega** — eso confirma que tu system prompt acotó bien la app.

Comparte tu link en el chat del Zoom.""")

# =============================================================================
# PARTE 7 — Peligros
# =============================================================================
md("""---

# Parte 7 — Peligros y límites

### 1) El costo se dispara → `max_steps`""")

code("""agente_topado = CodeAgent(tools=[], model=modelo,
    additional_authorized_imports=["pandas"], max_steps=1)
agente_topado.run(
    "Con 'ventas', calcula el crecimiento año a año de bebidas en Quito-44 y dime el "
    "mejor y peor año.",
    additional_args={"ventas": ventas})""")

md("""Con `max_steps=1` no termina: se corta. En producción, `max_steps` es tu freno de costo.

### 2) El CodeAgent ejecuta código que el modelo escribió
En clase (tu Colab) es seguro. Pero un agente que recibe preguntas de **usuarios
externos** debe correr su código en un **sandbox** (entorno aislado: E2B, Docker). La
lista blanca `additional_authorized_imports` es la primera defensa.

### 3) Si puedes mapear la tarea, NO uses un agente
Regla de Anthropic: usa lo más simple que funcione. El agente brilla en tareas
**abiertas**. Si la tarea es siempre la misma, escríbela directo: más rápida y fiable.

> Si cada paso acierta el **85 %**, un flujo de **10 pasos** acierta de punta a punta
> solo **≈20 %** (0.85¹⁰). Los agentes largos acumulan error.""")

# =============================================================================
# CIERRE
# =============================================================================
md("""---

# Cierre

1. Un **agente** = un LLM en un bucle **pensar → actuar → observar** (ReAct).
2. Un **CodeAgent** actúa escribiendo código. Estructura: **modelo** + **CodeAgent**
   (`tools`, `additional_authorized_imports`, `max_steps`) + **`run()`**.
3. **El buen prompting es la habilidad clave**: específico + formato + contexto.
4. Puedes darle **herramientas** (búsqueda web) sin programar.
5. Construiste una **app real**: el Asistente Estadístico de Ventas de Arca.
6. Cuídalo: `max_steps`, sandbox en producción, y si la tarea es fija, no uses agente.

**La habilidad que cambia:** ya no es "saber escribir pandas", es **saber pedir bien**.
El agente programa; tú piensas el problema.""")

# =============================================================================
nb.cells = cells
nbf.write(nb, "Clase_37_Agentes.ipynb")
print(f"Notebook generado: {len(cells)} celdas")
