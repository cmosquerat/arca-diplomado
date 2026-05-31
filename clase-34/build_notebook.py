"""
Construye Clase_34_LSTM_a_LLM.ipynb --- versión PBL v3 (sólo Groq, datasets embebidos).

Perfil: profesionales seniors de Arca Continental Ecuador (embotelladora).
La teoría (LSTM/atención/Transformer) está en el deck.
Este notebook es 100% APLICADO:
  - Helper `llm()` minimalista (4 líneas) contra Groq cloud.
  - Anatomía del endpoint OpenAI-compatible (PROTOCOLO, no marca).
  - 10 apps demostradas, código directo, sin abstracciones innecesarias.
  - 3 apps Gradio (clasificador, chat-doc, comparador 8B vs 70B).
  - 4 datasets EMBEBIDOS (tickets, OTs, manual+Q&A, incidentes RAG).
  - 5 ejercicios PBL ACOTADOS: input fijo + validador automático + criterio numérico.

Sin Ollama (instalación local complica Colab). Sin transformers/BETO (vive en el deck).
Sólo `openai`, `sentence-transformers`, `gradio`.
"""
import json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "Clase_34_LSTM_a_LLM.ipynb")
REPO = "https://raw.githubusercontent.com/cmosquerat/arca-diplomado/main/clase-34"

cells = []
def md(t):
    L = t.strip("\n").split("\n")
    cells.append({"cell_type":"markdown","metadata":{},"source":[(s+"\n") for s in L[:-1]]+[L[-1]]})
def code(t):
    s = t.strip("\n").split("\n")
    cells.append({"cell_type":"code","metadata":{},"outputs":[],"execution_count":None,
                  "source":[(x+"\n") for x in s[:-1]]+[s[-1]]})

# =============================================================================
#  PORTADA
# =============================================================================
md("""
# Clase 34 — Del LSTM al LLM (Notebook PBL)
### 10 apps construidas hoy + 5 ejercicios con criterio de aceptación

**Diplomado en Data Science Aplicada con Python para la Toma de Decisiones**
Arca Continental Ecuador | UDLA

---

> **Pregunta del día**: ¿cómo le enseñamos a una máquina a entender el lenguaje de Arca
> —manuales, tickets, quejas— sin contratar 50 lingüistas?

La teoría (Transformer, atención, BETO, escala de LLMs) está en el **deck**.
Este notebook es **práctica pura**: vas a construir aplicaciones que funcionan y a
resolver 5 ejercicios con **datos provistos** y **métrica automática**.

### Lo que vamos a construir (10 apps)

| # | App | Para qué |
|---|---|---|
| 1 | Hello LLM + anatomía del endpoint | Entender el JSON crudo del request/response |
| 2 | Clasificador zero-shot | Triage automático de tickets |
| 3 | Extractor JSON estructurado | Texto libre → campos para tu SAP |
| 4 | Resumidor de reportes de turno | 1 página → 3 viñetas |
| 5 | Generador de respuestas a quejas | Borrador para CSR |
| 6 | Chat multi-turno con memoria | Asistente conversacional |
| 7 | Búsqueda semántica de tickets | Tráeme tickets parecidos a éste |
| 8 | **Gradio**: clasificador con UI | App web en 10 líneas |
| 9 | **Gradio**: chat con tu documento | Mini-RAG con UI |
| 10 | **Gradio**: comparador Llama 8B vs 70B | Elegir modelo por costo/calidad |

### Lo que vas a entregar (5 ejercicios)

Todos los ejercicios usan **datos provistos en este notebook** (no inventes los tuyos).
Cada uno tiene un **validador automático** y un **criterio numérico de aceptación**.

| # | Ejercicio | Criterio |
|---|---|---|
| E1 | Clasificar 25 tickets con métricas | acc ≥ 80% |
| E2 | Extraer JSON de 10 órdenes de trabajo | ≥ 8/10 schema válido + campos correctos |
| E3 | Q&A sobre manual técnico (8 preguntas) | ≥ 6/8 respuestas correctas |
| E4 | Comparar Llama 8B vs 70B en los 25 tickets de E1 | tabla con acc/latencia + conclusión |
| E5 | Mini-RAG sobre 50 incidentes históricos | 5/5 queries devuelven incidente relevante |

---
""")

# =============================================================================
#  0. SETUP
# =============================================================================
md("""
## 0. Setup

Instalamos lo necesario. Funciona en Colab y en local.
""")

code("""
import sys, os
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    os.system("pip install -q openai sentence-transformers gradio")
print("✓ Setup listo")
""")

# =============================================================================
#  1. ANATOMÍA DEL ENDPOINT
# =============================================================================
md("""
## 1. Anatomía del endpoint — "OpenAI-compatible" es un PROTOCOLO, no la marca

**Concepto que evita confusión**:

> `from openai import OpenAI` **NO** significa que estás usando ChatGPT.

Es la librería cliente que habla un protocolo HTTP particular. OpenAI publicó la
**forma del JSON** que su API espera y devuelve; esa forma se volvió un estándar de facto.
Hoy varios providers respetan ese contrato — el mismo cliente Python sirve para todos:

| Provider | base_url | Quién lo corre |
|---|---|---|
| OpenAI (GPT-5, GPT-5 mini, …) | `https://api.openai.com/v1` | OpenAI (pago) |
| **Groq** *(usamos hoy)* | `https://api.groq.com/openai/v1` | Groq (free hasta 30 RPM) |
| Together / Fireworks | `https://api.together.xyz/v1` | etc. |
| Anthropic Claude | (otro protocolo, librería `anthropic`) | — |

**Cambiar de provider = cambiar 2 líneas** (`base_url` + `api_key`). Tu código de negocio
queda igual. Eso es lo poderoso del estándar.

### Pega tu API key de Groq

Crea cuenta gratis en https://console.groq.com (sin tarjeta), genera una API key,
pégala cuando te la pida la celda (con `getpass`, no queda en el notebook):
""")

code("""
import getpass
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = getpass.getpass("Pega tu GROQ_API_KEY: ")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
""")

md("""
### El cliente y el helper `llm()`

Una sola instancia de cliente, una función de 4 líneas. Esto es todo lo que necesitas
para el resto del notebook.
""")

code("""
from openai import OpenAI

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
MODEL  = "llama-3.1-8b-instant"   # modelo por defecto (free tier generoso)

def llm(prompt, system=None, model=MODEL, temperature=0.3, max_tokens=400):
    msgs = ([{"role":"system","content":system}] if system else []) \\
           + [{"role":"user","content":prompt}]
    r = client.chat.completions.create(model=model, messages=msgs,
                                       temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content
""")

md("""
### El JSON crudo — qué viaja por debajo

La librería envuelve un HTTP POST con JSON. Veámoslo crudo con `model_dump()`:
""")

code("""
import json as jsn

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role":"system","content":"Eres asistente de planta de Arca. Responde corto."},
        {"role":"user","content":"¿Qué hago primero si una llenadora vibra raro?"},
    ],
    temperature=0.3, max_tokens=120,
)

print("=== RESPONSE (lo que devuelve el server) ===")
print(jsn.dumps(resp.model_dump(), ensure_ascii=False, indent=2)[:1100])
print()
print("=== El texto generado vive en .choices[0].message.content ===")
print(resp.choices[0].message.content)
""")

md("""
**Lo que ves**:
- El response trae `choices[0].message.content` (texto), `usage` (tokens), `finish_reason`
  (`stop` = OK; `length` = se quedó sin tokens).
- `id`, `model`, `created`: metadatos.

Eso es todo. Cualquier provider que respete esta forma se puede usar con `openai`.
Mañana sale un provider nuevo → pones `base_url=...nuevo...` y todo tu código sigue.

---
""")

# =============================================================================
#  2. RECAP CLASE 33 — el problema de la negación
# =============================================================================
md("""
## 2. El problema que resolvemos (recap clase 33)

TF-IDF + LogReg sobre `muchocine_sentimiento` llega a **84%** de accuracy. Bien.
Pero **no entiende la negación**. Veámoslo en 4 frases-trampa:
""")

code("""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CSV_URL = \"""" + REPO + """/muchocine_sentimiento.csv\"
df = pd.read_csv(CSV_URL); df["review"] = df["review"].astype(str)
Xtr, Xte, ytr, yte = train_test_split(df["review"], df["label"],
                                       test_size=0.2, random_state=42, stratify=df["label"])
vec = TfidfVectorizer(strip_accents="unicode", min_df=3, ngram_range=(1,2), max_features=30000)
clf = LogisticRegression(max_iter=1000, C=3.0).fit(vec.fit_transform(Xtr), ytr)
acc_baseline = accuracy_score(yte, clf.predict(vec.transform(Xte)))
print(f"Baseline TF-IDF: {acc_baseline:.1%}")

TRAMPAS = [
    ("no me gustó nada la trama",            0),
    ("no es para nada mala, me sorprendió",  1),
    ("no podría estar más contento",         1),
    ("buena fotografía pero no, no funciona",0),
]
print(f"\\n{'Real':<6} {'Pred':<6} Frase")
for f, real in TRAMPAS:
    p = int(clf.predict(vec.transform([f]))[0])
    ok = "✓" if p == real else "✗"
    print(f"{ok} {'POS' if real else 'NEG':<4} {'POS' if p else 'NEG':<4}   {f}")
""")

md("""
TF-IDF cuenta palabras y no entiende el orden. **Para esto y mucho más, ahora usamos LLMs.**

---
""")

# =============================================================================
#  3. DATASETS EMBEBIDOS + EVALUADORES
# =============================================================================
md("""
## 3. Datasets de trabajo (embebidos en este notebook)

Para que los ejercicios tengan **criterio de aceptación medible**, te damos 4 datasets
sintéticos pero realistas de planta embotelladora. No tienes que conseguir datos —
están aquí abajo. Cada dataset tiene su **validador automático** correspondiente.
""")

code('''
# ============================================================
#  DATASET 1: TICKETS_GOLD (25 tickets × 5 categorías) → E1, E4
# ============================================================
CATEGORIAS = ["mantenimiento", "queja_cliente", "logistica", "calidad", "ventas"]

TICKETS_GOLD = [
    # mantenimiento (5)
    ("El compresor 3 de la línea 2 está vibrando más de lo normal desde anoche.", "mantenimiento"),
    ("La etiquetadora se traba intermitentemente, hay que revisar el rodillo guía.", "mantenimiento"),
    ("Falla recurrente del sensor de tapa en la posición 7 de la llenadora.", "mantenimiento"),
    ("Filtro de aire del compresor 5 saturado, programar cambio.", "mantenimiento"),
    ("El chiller del depósito 2 está perdiendo presión otra vez.", "mantenimiento"),
    # queja_cliente (5)
    ("Recibí mi pedido con 3 botellas con la tapa rota y todo mojado.", "queja_cliente"),
    ("La gaseosa de mi caja sabe extraña, no es la habitual.", "queja_cliente"),
    ("Hace una semana llamé por mi reposición y nadie me responde.", "queja_cliente"),
    ("El producto llegó vencido pero la etiqueta dice marzo 2027, ¿qué pasa?", "queja_cliente"),
    ("Me cobraron dos veces el mismo pedido en la app, devuélvanme.", "queja_cliente"),
    # logistica (5)
    ("El camión a Guayaquil llegó con 4 horas de atraso por bloqueo en la vía.", "logistica"),
    ("Falta de pallets para la salida de hoy en el depósito Quito sur.", "logistica"),
    ("La ruta Manta-Portoviejo se canceló por inundación.", "logistica"),
    ("Diferencia de 12 cajas entre lo despachado y lo facturado al cliente Tia.", "logistica"),
    ("El proveedor de combustible no llegó, el camión 14 no puede salir.", "logistica"),
    # calidad (5)
    ("Lote 4521 con desviación de pH en el control de turno, retención preventiva.", "calidad"),
    ("Detectamos partícula extraña en una botella del lote 8830.", "calidad"),
    ("El nivel de llenado de la línea 1 está 3 mm por debajo del objetivo.", "calidad"),
    ("La auditoría externa pidió evidencia de calibración del medidor de Brix.", "calidad"),
    ("Reclamo del retail: etiquetas con códigos de barra ilegibles en el lote 9012.", "calidad"),
    # ventas (5)
    ("Cerramos el contrato anual con la cadena Tia para 2027.", "ventas"),
    ("El cliente Mi Comisariato pide un descuento adicional por volumen.", "ventas"),
    ("La promoción 2x1 de Quito subió las ventas un 18% esta semana.", "ventas"),
    ("Visita comercial agendada para el lunes con el comprador de Supermaxi.", "ventas"),
    ("Necesitamos reforzar el equipo comercial de la zona costera.", "ventas"),
]
print(f"TICKETS_GOLD: {len(TICKETS_GOLD)} tickets sobre {len(CATEGORIAS)} categorías")


# ============================================================
#  DATASET 2: OTS_GOLD (10 órdenes de trabajo) → E2
# ============================================================
# Schema: {equipo, tecnico, repuestos: list[str], horas: int, prioridad: alta|media|baja}
OTS_GOLD = [
    ("OT-1041: el compresor 3 vibra fuerte, asignar a Juan Pérez, traer rodamiento y filtro, ~4h, prioridad alta.",
     {"equipo": "compresor 3", "tecnico": "Juan Pérez",
      "repuestos": ["rodamiento", "filtro"], "horas": 4, "prioridad": "alta"}),
    ("Cambio programado de aceite en chiller 9 — María Vega, aceite ISO 220 y junta, 2 horas, prioridad baja.",
     {"equipo": "chiller 9", "tecnico": "María Vega",
      "repuestos": ["aceite ISO 220", "junta"], "horas": 2, "prioridad": "baja"}),
    ("Llenadora línea 2 perdió presión. Carlos Sánchez, manguera y empaque, 3 horas, prioridad alta.",
     {"equipo": "llenadora línea 2", "tecnico": "Carlos Sánchez",
      "repuestos": ["manguera", "empaque"], "horas": 3, "prioridad": "alta"}),
    ("Etiquetadora se traba — técnico Ana Rivera, rodillo guía nuevo, 1 hora, media prioridad.",
     {"equipo": "etiquetadora", "tecnico": "Ana Rivera",
      "repuestos": ["rodillo guía"], "horas": 1, "prioridad": "media"}),
    ("Calibración mensual del medidor de Brix, técnico Luis Pardo, sin repuestos, 1 hora, prioridad baja.",
     {"equipo": "medidor de Brix", "tecnico": "Luis Pardo",
      "repuestos": [], "horas": 1, "prioridad": "baja"}),
    ("Sensor de tapa posición 7 falla intermitente; Juan Pérez; sensor capacitivo + cableado; 2 h; alta.",
     {"equipo": "sensor de tapa posición 7", "tecnico": "Juan Pérez",
      "repuestos": ["sensor capacitivo", "cableado"], "horas": 2, "prioridad": "alta"}),
    ("Mantenimiento preventivo del carrusel línea 1, Sofía Castro, grasa y tornillos, 5 horas, media.",
     {"equipo": "carrusel línea 1", "tecnico": "Sofía Castro",
      "repuestos": ["grasa", "tornillos"], "horas": 5, "prioridad": "media"}),
    ("Cambio de filtros HEPA del depósito 2, equipo de limpieza, 3 filtros HEPA, 2 horas, baja prioridad.",
     {"equipo": "depósito 2", "tecnico": "equipo de limpieza",
      "repuestos": ["filtros HEPA"], "horas": 2, "prioridad": "baja"}),
    ("Reparación urgente del horno de retractilado, técnico Mario Núñez, resistencia eléctrica nueva, 6 h, prioridad alta.",
     {"equipo": "horno de retractilado", "tecnico": "Mario Núñez",
      "repuestos": ["resistencia eléctrica"], "horas": 6, "prioridad": "alta"}),
    ("Inspección visual de la cinta transportadora 4 — Ana Rivera — sin repuestos — 1 hora — prioridad baja.",
     {"equipo": "cinta transportadora 4", "tecnico": "Ana Rivera",
      "repuestos": [], "horas": 1, "prioridad": "baja"}),
]
print(f"OTS_GOLD: {len(OTS_GOLD)} órdenes de trabajo")
''')

code('''
# ============================================================
#  DATASET 3: MANUAL_DEMO + PREGUNTAS_GOLD (8 Q&A) → E3
# ============================================================
MANUAL_DEMO = """Manual operativo rápido — Llenadora modelo XF-200, planta Arca Quito.

ESPECIFICACIONES
- Capacidad nominal: 12.000 botellas/hora.
- Presión operativa: entre 2.5 y 3.2 bar.
- Voltaje de alimentación: 380 V trifásico.
- Temperatura ambiente recomendada: 18 a 26 °C.

MANTENIMIENTO PREVENTIVO
- Cambio de filtro de aire: cada semana, sin excepción.
- Lubricación del carrusel: aceite ISO 220, cada 80 horas de operación.
- Mantenimiento mayor: cada 200 horas, incluye inspección de válvulas y sellos.
- Calibración del sensor de tapa: trimestral.

CÓDIGOS DE ERROR
- E03: baja presión, revisar bomba antes de llamar a mantenimiento.
- E07: falla del sensor de tapa, revisar conexión antes de reemplazar el sensor.
- E12: temperatura del aceite fuera de rango, parar el equipo.
- E18: contador de botellas inconsistente, recalibrar el encoder.

CONTACTOS
- Soporte técnico: ext. 2410 (Mario Núñez, supervisor de mantenimiento).
- Repuestos: bodega central, ext. 2207."""

# Preguntas con respuesta esperada (substring/keyword que debe aparecer)
PREGUNTAS_GOLD = [
    ("¿Cuál es la capacidad nominal de la llenadora XF-200?",   "12.000"),
    ("¿Cada cuánto se cambia el filtro de aire?",               "semana"),
    ("¿Qué aceite se usa para lubricar el carrusel?",           "ISO 220"),
    ("¿Qué hago si aparece el código E07?",                     "conexión"),
    ("¿A qué extensión llamo para soporte técnico?",            "2410"),
    ("¿Cuál es la presión operativa correcta?",                 "2.5"),
    ("¿Cada cuántas horas se hace el mantenimiento mayor?",     "200"),
    ("¿Qué hago si la temperatura del aceite está fuera de rango?", "parar"),
]
print(f"MANUAL_DEMO: {len(MANUAL_DEMO.split())} palabras")
print(f"PREGUNTAS_GOLD: {len(PREGUNTAS_GOLD)} preguntas")
''')

code('''
# ============================================================
#  DATASET 4: INCIDENTES_RAG (50 incidentes históricos) → E5
# ============================================================
# Cada incidente: (descripcion, accion_tomada)
INCIDENTES_RAG = [
    ("compresor 1 dejó de arrancar tras corte de luz",                "reset del variador y purga de aire de la línea"),
    ("llenadora línea 1 con desviación de volumen de llenado",         "calibración del sensor de nivel y limpieza de boquillas"),
    ("etiquetadora arruga etiquetas en lote nuevo",                    "ajustar tensión del rodillo de presión, limpiar pegamento"),
    ("código E07 en llenadora 2 toda la mañana",                       "revisar conexión del sensor de tapa antes de reemplazar"),
    ("carrusel suena raro, ruido metálico",                            "lubricar engranajes con grasa de litio"),
    ("paro de la línea 3 por sobrecalentamiento del compresor",        "limpiar intercambiador y verificar nivel de aceite"),
    ("variación de Brix en lote 4521",                                 "retención preventiva, reanálisis en laboratorio"),
    ("chiller 9 perdió presión durante la noche",                      "buscar fuga, recargar refrigerante R-404a"),
    ("operador reporta vibración en bomba dosificadora",               "revisar acoplamiento y alineación del motor"),
    ("falla del PLC de la línea 1, pantalla en negro",                 "ciclo de energía y restaurar último backup"),
    ("etiqueta queda torcida en línea 2",                              "centrar el guiador y ajustar el sensor de posición"),
    ("contador de botellas no avanza",                                 "limpieza del encoder óptico"),
    ("fuga de agua bajo el chiller 4",                                 "sellar junta del intercambiador"),
    ("sensor de presión marca cero",                                   "purgar línea y verificar cableado al PLC"),
    ("retractiladora deja botellas sueltas",                           "ajustar temperatura del horno y cambiar resistencia"),
    ("transportador 2 se atasca con botellas caídas",                  "agregar guías laterales y revisar velocidad"),
    ("contaminación visible en lote 8830",                             "cuarentena del lote, limpieza CIP de tanque"),
    ("compresor 5 con consumo elevado de aceite",                      "cambio de empaque del cárter"),
    ("la bomba CIP no genera presión",                                 "verificar válvula de retención y cebado"),
    ("filtro de aire saturado en compresor 5",                         "reemplazo del cartucho y registro en bitácora"),
    ("cliente del retail reporta etiquetas ilegibles",                 "ajustar contraste de la impresora térmica"),
    ("falla intermitente del sensor de tapa posición 7",               "reemplazo del sensor capacitivo"),
    ("operador no puede iniciar el ciclo de envasado",                 "verificar enclavamientos de seguridad de las puertas"),
    ("pH fuera de rango en mezcla del jarabe",                         "recalibración del pH-metro y verificación de patrones"),
    ("nivel bajo de CO2 en la línea de carbonatación",                 "verificar bombona principal y conexiones"),
    ("carrusel de tapado pierde tapas intermitentemente",              "ajustar el dispensador y verificar guía"),
    ("paro de emergencia activado por operador en línea 3",            "investigar incidente, retroalimentar protocolo"),
    ("la temperatura del jarabe es 3°C inferior al setpoint",          "verificar serpentín del intercambiador"),
    ("válvula proporcional del llenado responde lenta",                "limpieza de la válvula y purga de aire"),
    ("ruido inusual en la bomba de envío de producto terminado",      "inspeccionar cojinete y considerar cambio"),
    ("etiquetadora moja el adhesivo del piso del carrusel",            "fuga en el deposito de pegamento, cambiar empaque"),
    ("código E12 toda la noche en línea 2",                            "se detuvo equipo, esperar a mantenimiento"),
    ("error de comunicación entre PLC y SCADA",                        "reiniciar puerto serial y verificar cable de red"),
    ("scrap del 4% en lote 9012",                                      "revisar guías de transporte y velocidad del carrusel"),
    ("la presión del sistema baja por debajo de 2 bar",                "buscar fuga y verificar el regulador"),
    ("falla del variador de frecuencia del transportador 4",           "cambio del variador, registrar en histórico"),
    ("contaminación cruzada sospechada entre líneas",                  "limpieza CIP completa y validación microbiológica"),
    ("operador reporta olor inusual en zona de envasado",              "verificar ventilación y origen del olor"),
    ("encoder del carrusel da lecturas erráticas",                     "limpieza del disco óptico y reemplazo si persiste"),
    ("falla recurrente del sensor de fin de carrera",                  "reemplazo del sensor y revisión de alineación"),
    ("alarma de bajo nivel en tanque de jarabe",                       "rellenado y verificación del sensor de nivel"),
    ("sello mecánico de la bomba con fuga",                            "reemplazo del sello mecánico"),
    ("vibración elevada en motor del transportador",                   "balanceo y verificación de pernos de anclaje"),
    ("filtro de la línea de agua tratada saturado",                    "reemplazo y reactivación del sistema"),
    ("contador de unidades no coincide con producción real",           "calibración del contador y verificación del PLC"),
    ("etiquetadora se atasca con etiquetas pegadas",                   "limpieza del cabezal y reemplazo de rollo"),
    ("paro por temperatura alta del aceite del compresor 5",           "agregar refrigerante y revisar ventilador del radiador"),
    ("cliente reporta caja con botellas faltantes",                    "investigación con almacén y reposición"),
    ("falla del módulo de pesaje de la línea 1",                       "calibración con pesa patrón"),
    ("operador no puede acceder al SCADA",                             "reset de credenciales y revisión de permisos"),
]
print(f"INCIDENTES_RAG: {len(INCIDENTES_RAG)} incidentes históricos")
''')

md("""
### Los 4 evaluadores

Funciones que reciben tus predicciones y calculan el criterio de aceptación.
Las llamas al final de cada ejercicio. No las modificás.
""")

code('''
import re, json as jsn
from collections import Counter

# -------- EVALUADOR E1 (clasificación) --------
def evaluar_clasificacion(predicciones):
    """Devuelve dict {acc, n_correctas, total, matriz_confusion (dict)}."""
    reales = [c for _, c in TICKETS_GOLD]
    assert len(predicciones) == len(reales), "Necesitas 1 predicción por ticket"
    preds = [p.strip().lower() for p in predicciones]
    n_ok = sum(1 for p, r in zip(preds, reales) if p == r)
    cm = {c: Counter() for c in CATEGORIAS}
    for p, r in zip(preds, reales):
        cm[r][p if p in CATEGORIAS else "(otro)"] += 1
    out = {"acc": n_ok/len(reales), "n_correctas": n_ok, "total": len(reales)}
    print(f"Accuracy: {out['acc']:.1%}  ({n_ok}/{len(reales)})")
    print(f"Criterio E1: acc ≥ 80%  →  {'✅ PASA' if out['acc'] >= 0.80 else '❌ NO pasa'}")
    print("\\nMatriz de confusión (filas = real, columnas = predicción):")
    cols = CATEGORIAS + ["(otro)"]
    print(f"{'REAL\\\\PRED':<18} " + " ".join(f"{c[:5]:>6}" for c in cols))
    for c in CATEGORIAS:
        row = " ".join(f"{cm[c][cc]:>6}" for cc in cols)
        print(f"{c:<18} {row}")
    out["matriz_confusion"] = {r: dict(v) for r, v in cm.items()}
    return out

# -------- EVALUADOR E2 (extracción JSON) --------
def evaluar_extraccion(salidas):
    """Devuelve dict {n_validas, n_correctas_campos_clave, total, detalle}."""
    assert len(salidas) == len(OTS_GOLD), "Necesitas 1 salida por OT"
    n_validas = 0; n_correctas = 0
    detalle = []
    for i, (sal, (_, esp)) in enumerate(zip(salidas, OTS_GOLD)):
        valida = isinstance(sal, dict) and all(k in sal for k in ["equipo","tecnico","horas","prioridad"])
        n_validas += int(valida)
        ok = False
        if valida:
            # Match en campos clave (string-norm: lower, sin acentos básicos)
            def norm(s):
                return re.sub(r"\\s+", " ", str(s).lower().strip())
            equipo_ok    = norm(sal.get("equipo","")) == norm(esp["equipo"])
            tecnico_ok   = norm(sal.get("tecnico","")) == norm(esp["tecnico"])
            prioridad_ok = norm(sal.get("prioridad","")) == norm(esp["prioridad"])
            horas_ok     = int(sal.get("horas", -1)) == int(esp["horas"])
            ok = equipo_ok and tecnico_ok and prioridad_ok and horas_ok
            n_correctas += int(ok)
        detalle.append({"i": i+1, "schema_valido": valida, "campos_clave_ok": ok})
    print(f"Schema válido:        {n_validas}/{len(OTS_GOLD)}")
    print(f"Campos clave correctos: {n_correctas}/{len(OTS_GOLD)}  "
          f"(equipo + tecnico + horas + prioridad)")
    print(f"Criterio E2: ≥ 8/10  →  {'✅ PASA' if n_correctas >= 8 else '❌ NO pasa'}")
    return {"n_validas": n_validas, "n_correctas": n_correctas,
            "total": len(OTS_GOLD), "detalle": detalle}

# -------- EVALUADOR E3 (Q&A sobre manual) --------
def evaluar_qa(respuestas):
    """Substring match flexible: la palabra clave debe aparecer en la respuesta."""
    assert len(respuestas) == len(PREGUNTAS_GOLD), "Necesitas 1 respuesta por pregunta"
    n_ok = 0; detalle = []
    for i, (resp, (q, clave)) in enumerate(zip(respuestas, PREGUNTAS_GOLD)):
        ok = clave.lower() in (resp or "").lower()
        n_ok += int(ok)
        detalle.append({"i": i+1, "pregunta": q, "clave_esperada": clave,
                        "ok": ok, "respuesta": resp})
    print(f"Respuestas correctas: {n_ok}/{len(PREGUNTAS_GOLD)}")
    print(f"Criterio E3: ≥ 6/8  →  {'✅ PASA' if n_ok >= 6 else '❌ NO pasa'}")
    for d in detalle:
        mark = "✓" if d["ok"] else "✗"
        print(f"  {mark} {d['pregunta'][:60]:<60} (esperaba ver: \\\"{d['clave_esperada']}\\\")")
    return {"n_ok": n_ok, "total": len(PREGUNTAS_GOLD), "detalle": detalle}

# -------- EVALUADOR E5 (mini-RAG) --------
QUERIES_RAG_GOLD = [
    ("falla del sensor de tapa de la llenadora",          "código E07"),
    ("equipo perdió presión durante la noche",            "chiller 9"),
    ("contaminación o partícula extraña en producto",     "8830"),
    ("desviación en parámetros de calidad",               "Brix"),
    ("PLC o sistema de control no responde",              "PLC"),
]

def evaluar_rag(top_ks_devueltos):
    """top_ks_devueltos: lista de listas de strings (top-K incidentes por query)."""
    assert len(top_ks_devueltos) == len(QUERIES_RAG_GOLD)
    n_ok = 0
    for (q, clave), top_k in zip(QUERIES_RAG_GOLD, top_ks_devueltos):
        recuperado = " || ".join(top_k).lower()
        ok = clave.lower() in recuperado
        n_ok += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  {mark} \\\"{q[:60]}\\\"  (esperaba \\\"{clave}\\\" en top-{len(top_k)})")
    print(f"\\nQueries con incidente esperado en top-K: {n_ok}/5")
    print(f"Criterio E5: 5/5  →  {'✅ PASA' if n_ok == 5 else '❌ NO pasa'}")
    return {"n_ok": n_ok, "total": 5}

print("✓ Evaluadores listos: evaluar_clasificacion, evaluar_extraccion, evaluar_qa, evaluar_rag")
''')

# =============================================================================
#  4. LAS 10 APPS
# =============================================================================
md("""
---

## 4. Las 10 apps demostradas

Cada app es una función Python concreta y reusable. Las primeras 7 son funciones puras.
Las últimas 3 son apps Gradio (UI web que lanzás en el notebook).

### App 1 — Hello LLM (la función `llm` que ya definimos)

Ya está. Una llamada de prueba:
""")

code("""
print(llm("Da 3 causas posibles de vibración en una llenadora industrial. Sé conciso."))
""")

md("""
### App 2 — Clasificador zero-shot
""")

code('''
def clasificar(texto, categorias=CATEGORIAS):
    cats = ", ".join(categorias)
    prompt = (f'Clasifica el siguiente texto en UNA SOLA categoría de: {cats}.\\n'
              f'Texto: "{texto}"\\n'
              'Responde SÓLO con la categoría exacta, sin explicación.')
    return llm(prompt, max_tokens=20, temperature=0.0).strip().lower()

# Demo sobre 5 tickets aleatorios del dataset
import random
random.seed(0)
muestra = random.sample(TICKETS_GOLD, 5)
for texto, real in muestra:
    p = clasificar(texto)
    ok = "✓" if p == real else "✗"
    print(f"  {ok} [{p:<15}] (real: {real:<15}) {texto[:55]}...")
''')

md("""
### App 3 — Extractor JSON estructurado
""")

code('''
def extraer_ot(texto):
    schema = ('{"equipo": str, "tecnico": str, "repuestos": [str], '
              '"horas": int, "prioridad": "alta"|"media"|"baja"}')
    instr = (f'Extrae los datos de la siguiente orden de trabajo como JSON con schema:\\n{schema}\\n'
             'Sólo JSON válido, sin explicación ni markdown.\\n\\n'
             f'OT: "{texto}"')
    raw = llm(instr, max_tokens=300, temperature=0.0)
    raw = raw.strip().strip("`").replace("json\\n", "").strip()
    try:
        return jsn.loads(raw)
    except Exception:
        return {"_raw": raw, "_error": "no parseó JSON"}

# Demo con la primera OT
ej = extraer_ot(OTS_GOLD[0][0])
print(jsn.dumps(ej, ensure_ascii=False, indent=2))
''')

md("""
### App 4 — Resumidor de reportes de turno
""")

code('''
def resumir(texto, n_vinetas=3):
    p = (f"Resume el siguiente reporte en EXACTAMENTE {n_vinetas} viñetas, "
         f"máximo 15 palabras cada una. Sólo el resumen:\\n\\n{texto}")
    return llm(p, max_tokens=250, temperature=0.2)

REPORTE = ("Reporte turno línea 2, 30 mayo 2026 noche. Objetivo: 18.000 botellas. "
           "Real: 16.450 (91%). Causas: parada 35 min a las 23:40 por sensor de tapa "
           "posición 7 (cambio rápido); parada 22 min a las 02:15 por atasco en "
           "carrusel etiquetado. Rechazo de calidad 0.3% (OK <0.5%). Nota: filtro "
           "del compresor 5 sucio, cambiar próximo mantenimiento jueves 6 junio.")
print(resumir(REPORTE))
''')

md("""
### App 5 — Generador de respuestas a quejas (borrador para CSR)
""")

code('''
SISTEMA_CSR = ("Eres asistente del equipo de servicio al cliente de Arca Continental Ecuador. "
               "Redactas BORRADORES para que un agente humano revise. Empático, asumes "
               "responsabilidad, ofrece una solución concreta. Máximo 3 oraciones.")

queja = "Compré dos cajas y una llegó con la mitad de las botellas con la tapa rota."
print(llm(queja, system=SISTEMA_CSR, max_tokens=200))
''')

md("""
### App 6 — Chat multi-turno con memoria

El LLM no recuerda nada entre llamadas. **Vos mantenés la lista de mensajes.**
""")

code('''
def hablar(historial, mensaje, model=MODEL, max_tokens=250):
    historial.append({"role": "user", "content": mensaje})
    r = client.chat.completions.create(
        model=model, messages=historial,
        max_tokens=max_tokens, temperature=0.4)
    reply = r.choices[0].message.content
    historial.append({"role": "assistant", "content": reply})
    return reply

# Demo: una conversación
historial = [{"role":"system","content":"Eres técnico senior de mantenimiento de planta. Concreto, pasos numerados."}]
for msg in [
    "Mi llenadora vibra más de lo normal. ¿Por dónde empiezo?",
    "Ya revisé los pernos, están firmes. ¿Qué sigue?",
    "El motor también está caliente. ¿Cambia el diagnóstico?",
]:
    print(f"\\n>>> {msg}")
    print(hablar(historial, msg))
''')

md("""
### App 7 — Búsqueda semántica (sentence-transformers)

Convertís cada texto en un vector. Una query nueva → vector → cercanía coseno → tickets parecidos.
""")

code('''
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Indexamos los 50 incidentes históricos
TEXTOS_INC = [t for t, _ in INCIDENTES_RAG]
EMBS_INC = embedder.encode(TEXTOS_INC, show_progress_bar=False)
print(f"Index: {EMBS_INC.shape[0]} incidentes, {EMBS_INC.shape[1]} dimensiones")

def buscar_incidentes(query, k=3):
    q = embedder.encode([query], show_progress_bar=False)
    sims = cosine_similarity(q, EMBS_INC)[0]
    top = np.argsort(-sims)[:k]
    return [(float(sims[i]), INCIDENTES_RAG[i]) for i in top]

# Demo: 2 queries
for q in ["problema con sensor de tapa", "presión que se cae"]:
    print(f"\\n🔍 \\"{q}\\"")
    for sim, (texto, accion) in buscar_incidentes(q, k=3):
        print(f"   {sim:.2f}  {texto}")
        print(f"          → acción: {accion}")
''')

md("""
### Apps con Gradio (UI web)

**Gradio** envuelve tu función Python en una UI. En Colab `launch(share=True)` te da una URL pública;
en local te abre `localhost:7860`.

### App 8 — Gradio: clasificador con UI
""")

code('''
import gradio as gr

def app_clasificar(texto, categorias_csv):
    cats = [c.strip() for c in categorias_csv.split(",") if c.strip()]
    if not texto.strip(): return "(pega un texto)"
    return f"**{clasificar(texto, cats)}**"

demo_clasif = gr.Interface(
    fn=app_clasificar,
    inputs=[
        gr.Textbox(label="Ticket o queja", lines=3,
                   placeholder="Ej: El compresor 3 vibra desde anoche"),
        gr.Textbox(label="Categorías (separadas por comas)",
                   value="mantenimiento, queja_cliente, logistica, calidad, ventas"),
    ],
    outputs=gr.Markdown(label="Categoría"),
    title="Clasificador de tickets · Arca",
    flagging_mode="never",
)
# Descomenta para lanzar:
# demo_clasif.launch(share=True, inline=True)
print("✓ App definida. Lanza con: demo_clasif.launch(share=True, inline=True)")
''')

md("""
### App 9 — Gradio: chat con tu documento (mini-RAG sin embeddings)

Le pegás un documento y le hacés preguntas. El LLM responde **sólo con lo que está en el documento**.
""")

code('''
def chat_con_doc(documento, pregunta):
    if not documento.strip() or not pregunta.strip():
        return "(pega un documento y haz una pregunta)"
    prompt = (
        "Responde la pregunta usando SÓLO la información del siguiente documento. "
        "Si la respuesta no aparece, di literalmente: 'No aparece en el documento.'\\n\\n"
        f"=== DOCUMENTO ===\\n{documento}\\n\\n"
        f"=== PREGUNTA ===\\n{pregunta}\\n\\n"
        "=== RESPUESTA ==="
    )
    return llm(prompt, max_tokens=300, temperature=0.0)

demo_doc = gr.Interface(
    fn=chat_con_doc,
    inputs=[
        gr.Textbox(label="Documento", lines=12, value=MANUAL_DEMO),
        gr.Textbox(label="Pregunta",
                   placeholder="¿Cada cuánto se cambia el filtro de aire?"),
    ],
    outputs=gr.Textbox(label="Respuesta", lines=4),
    title="Chat con tu documento · Arca",
    flagging_mode="never",
)
# demo_doc.launch(share=True, inline=True)
print("✓ App definida. Lanza con: demo_doc.launch(share=True, inline=True)")
''')

md("""
### App 10 — Gradio: comparador Llama 8B vs Llama 70B

Mismo prompt, dos modelos. El 8B es rápido y barato; el 70B es más capaz.
**Es exactamente el mismo código**, solo cambia el `model`.
""")

code('''
import time

MODEL_PEQ = "llama-3.1-8b-instant"
MODEL_GDE = "llama-3.3-70b-versatile"

def comparar_modelos(prompt_user, temperature):
    out = {}
    for name, m in [("8B", MODEL_PEQ), ("70B", MODEL_GDE)]:
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=m, messages=[{"role":"user","content":prompt_user}],
                temperature=temperature, max_tokens=300)
            txt = r.choices[0].message.content
            out[name] = (f"### {name}  "
                         f"_({(time.time()-t0):.1f}s, {r.usage.total_tokens} tokens)_\\n\\n{txt}")
        except Exception as e:
            out[name] = f"### {name} ⚠️ no disponible\\n\\n{type(e).__name__}: {str(e)[:200]}"
    return out["8B"], out["70B"]

demo_comp = gr.Interface(
    fn=comparar_modelos,
    inputs=[
        gr.Textbox(label="Prompt", lines=3,
                   value="Da 3 causas posibles de vibración en una llenadora industrial."),
        gr.Slider(0.0, 1.5, value=0.3, label="Temperatura"),
    ],
    outputs=[
        gr.Markdown(label="Llama-3.1-8B (rápido, barato)"),
        gr.Markdown(label="Llama-3.3-70B (más capaz, más caro)"),
    ],
    title="Comparador Llama 8B vs 70B · Arca",
    description="Mismo prompt, mismos parámetros, distinto modelo. El código es idéntico, sólo cambia el campo `model`.",
    flagging_mode="never",
)
# demo_comp.launch(share=True, inline=True)
print("✓ App definida. Lanza con: demo_comp.launch(share=True, inline=True)")
''')

md("""
### Tabbed: las 3 apps Gradio en una sola interfaz
""")

code('''
tabbed = gr.TabbedInterface(
    [demo_clasif, demo_doc, demo_comp],
    ["Clasificador", "Chat con doc", "8B vs 70B"],
    title="Asistente NLP · Arca Continental",
)
# tabbed.launch(share=True, inline=True)
print("✓ TabbedInterface lista. Lanza con: tabbed.launch(share=True, inline=True)")
''')

# =============================================================================
#  5. LOS 5 EJERCICIOS PBL ACOTADOS
# =============================================================================
md("""
---

## 5. Los 5 ejercicios PBL — tu turno

Cada ejercicio tiene **input fijo (ya provisto)**, **función a completar** y **validador
automático con criterio numérico**. No hay datos que conseguir, no hay métricas que inventar.

### E1 — Clasificar los 25 tickets con métrica real

**Input**: `TICKETS_GOLD` (25 tickets + categoría real). **Categorías**: `CATEGORIAS`.

**Tu función**: `clasificar(texto)` (ya está hecha arriba).

**Tu tarea**: corre `clasificar` sobre cada ticket, junta predicciones en una lista,
llama al evaluador.

**Criterio de aceptación**: **acc ≥ 80%**.
""")

code('''
# E1 --- completa el run y llama al evaluador
preds_e1 = [clasificar(texto) for texto, _ in TICKETS_GOLD]
res_e1 = evaluar_clasificacion(preds_e1)
''')

md("""
### E2 — Extraer JSON de las 10 órdenes de trabajo

**Input**: `OTS_GOLD` (10 OTs + JSON esperado).

**Tu función**: `extraer_ot(texto)` (ya está hecha arriba).

**Tu tarea**: corrés `extraer_ot` sobre cada OT, juntás las salidas en una lista,
llamás al evaluador.

**Criterio**: **≥ 8/10** schema válido + 4 campos clave (`equipo`, `tecnico`, `horas`, `prioridad`)
exactamente correctos.

**Tip**: si fallas mucho, ajustá el prompt en `extraer_ot` (por ejemplo, agregar
ejemplos few-shot dentro del prompt).
""")

code('''
# E2 --- completa el run y llama al evaluador
sals_e2 = [extraer_ot(texto) for texto, _ in OTS_GOLD]
res_e2 = evaluar_extraccion(sals_e2)
''')

md("""
### E3 — Q&A sobre el manual técnico (8 preguntas)

**Input**: `MANUAL_DEMO` (manual operativo de la llenadora XF-200) + `PREGUNTAS_GOLD`
(8 preguntas con palabra clave esperada).

**Tu función**: `chat_con_doc(documento, pregunta)` (ya está hecha arriba).

**Tu tarea**: corrés `chat_con_doc(MANUAL_DEMO, q)` por cada pregunta, juntás las
respuestas en una lista, llamás al evaluador.

**Criterio**: **≥ 6/8** respuestas contienen la palabra clave esperada
(substring match flexible).
""")

code('''
# E3 --- completa el run y llama al evaluador
resp_e3 = [chat_con_doc(MANUAL_DEMO, q) for q, _ in PREGUNTAS_GOLD]
res_e3 = evaluar_qa(resp_e3)
''')

md("""
### E4 — Comparativa Llama 8B vs Llama 70B sobre los mismos 25 tickets

**Input**: `TICKETS_GOLD` (mismos 25 de E1).

**Tu tarea**: clasificar con AMBOS modelos, medir accuracy + latencia, presentar tabla,
escribir **2-3 oraciones** de conclusión.

**Criterio**: tabla completa con columnas `accuracy`, `latencia_promedio_s`, `tokens_promedio`,
y una recomendación escrita.
""")

code('''
# E4 --- completa el run, mide y compara
import time

def clasificar_con(texto, model):
    cats = ", ".join(CATEGORIAS)
    prompt = (f'Clasifica el texto en UNA categoría de: {cats}.\\n'
              f'Texto: "{texto}"\\nResponde SÓLO la categoría exacta.')
    t0 = time.time()
    r = client.chat.completions.create(
        model=model, messages=[{"role":"user","content":prompt}],
        max_tokens=20, temperature=0.0)
    return r.choices[0].message.content.strip().lower(), \\
           time.time()-t0, r.usage.total_tokens

filas = []
for nombre, m in [("8B", "llama-3.1-8b-instant"), ("70B", "llama-3.3-70b-versatile")]:
    lat, toks, correctos = [], [], 0
    for texto, real in TICKETS_GOLD:
        p, dt, tk = clasificar_con(texto, m)
        lat.append(dt); toks.append(tk)
        if p == real: correctos += 1
        time.sleep(0.4)   # respetar rate-limit free de Groq
    filas.append({
        "modelo": nombre,
        "accuracy": correctos / len(TICKETS_GOLD),
        "latencia_promedio_s": sum(lat)/len(lat),
        "tokens_promedio": sum(toks)/len(toks),
    })

import pandas as pd
tabla = pd.DataFrame(filas)
print(tabla.to_string(index=False))

# Tu conclusión (escríbela aquí, 2-3 oraciones):
print("\\n--- Tu conclusión ---")
print("(Reemplaza este texto con tu recomendación: ¿cuál usar en producción y por qué?)")
''')

md("""
### E5 — Mini-RAG sobre los 50 incidentes históricos

**Input**: `INCIDENTES_RAG` (ya indexado en `EMBS_INC`) + `QUERIES_RAG_GOLD` (5 queries
con incidente clave esperado).

**Tu tarea**: por cada query, recupera top-3 incidentes con `buscar_incidentes`, juntá los
textos en una lista, llamá al evaluador.

**Bonus**: arma una app Gradio donde el usuario escriba una query y vea los top-3 + una
respuesta sintetizada del LLM.

**Criterio**: **5/5** queries devuelven el incidente esperado en su top-3.
""")

code('''
# E5 --- completa el run y llama al evaluador
top_ks = []
for q, _ in QUERIES_RAG_GOLD:
    hits = buscar_incidentes(q, k=3)
    textos = [texto for _, (texto, _accion) in hits]
    top_ks.append(textos)

res_e5 = evaluar_rag(top_ks)
''')

code('''
# E5 BONUS --- app Gradio que sintetiza con el LLM
def app_rag(query):
    if not query.strip(): return "(escribe una query)"
    hits = buscar_incidentes(query, k=3)
    contexto = "\\n".join(f"- {t}  →  acción tomada: {a}" for _, (t, a) in hits)
    prompt = (f"Tickets parecidos del historial:\\n{contexto}\\n\\n"
              f"Para la nueva query: \\"{query}\\"\\n\\n"
              "Sugiere los próximos pasos en 3 viñetas concretas, basándote en las acciones del historial.")
    sintesis = llm(prompt, max_tokens=250, temperature=0.2)
    detalle = "\\n\\n".join(f"**Sim {sim:.2f}** — {t}\\n  → acción: {a}"
                            for sim, (t, a) in hits)
    return f"### Recomendación del LLM\\n\\n{sintesis}\\n\\n---\\n### Tickets parecidos\\n\\n{detalle}"

demo_rag = gr.Interface(
    fn=app_rag,
    inputs=gr.Textbox(label="Describe tu nueva incidencia", lines=2),
    outputs=gr.Markdown(),
    title="Mini-RAG · Asistente de incidentes históricos",
    flagging_mode="never",
)
# demo_rag.launch(share=True, inline=True)
print("✓ App E5-bonus definida. Lanza con: demo_rag.launch(share=True, inline=True)")
''')

# =============================================================================
#  6. CIERRE
# =============================================================================
md("""
---

## 6. Lo que te llevas

| Capa | Cuándo se usa |
|---|---|
| Función Python (`llm`, `clasificar`, `extraer_ot`, `resumir`, `chat_con_doc`, `buscar_incidentes`) | scripts, batch jobs, APIs internas |
| App Gradio | demos a stakeholders, prototipos, herramientas internas |
| Validador automático | medir si una solución pasa o no, sin discutir subjetividades |
| Estándar OpenAI-compatible | un código → cualquier provider; nunca te casás con uno |

### Regla del oficio

> El oficio nuevo del data scientist en Arca es **elegir el modelo correcto,
> orquestar las llamadas, validar con métricas y construir UI sobre eso**.
> No entrenar modelos desde cero.

### El siguiente paso — Módulo 6

- **Prompting avanzado** (few-shot, chain-of-thought).
- **RAG en serio**: FAISS/Chroma, reranking, chunking inteligente.
- **Agentes**: LLMs que ejecutan acciones (consultar BD, mandar emails).
- **Web scraping** para alimentar todo lo anterior.
- **APIs en producción**: rate limits, fallback, observabilidad.

---

## 7. Entrega

Sube tu notebook con los 5 ejercicios completados. Antes de subir:
- **No incluyas tu `GROQ_API_KEY`**. Usa `getpass` o variable de entorno (ya lo hicimos).
- Si querés ajustar los prompts de `clasificar` / `extraer_ot` / `chat_con_doc` para subir
  tu accuracy, **adelante** — eso es parte del aprendizaje (prompt engineering).

*Código + datos: github.com/cmosquerat/arca-diplomado/tree/main/clase-34*
""")

# =============================================================================
#  Persistir
# =============================================================================
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"✓ Notebook generado: {OUT} ({len(cells)} celdas)")
