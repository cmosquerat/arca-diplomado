"""
verify_demos.py — valida los demos clave de la clase 37 ANTES de clase.

Uso:
    pip install "smolagents[litellm]" pandas matplotlib
    export GROQ_API_KEY=gsk_...
    python verify_demos.py

Comprueba que el CodeAgent (cerebro Groq Llama 3.3 70B):
  1. Resuelve un cálculo de varios pasos (escribe y ejecuta código).
  2. Analiza el dataset real de ventas (filtra + resume).
  3. Genera una gráfica (escribe matplotlib y guarda el PNG).

Si los 3 pasan, los demos de la clase deberían funcionar en vivo.
"""
import os
import sys

if "GROQ_API_KEY" not in os.environ:
    sys.exit("Necesitas exportar GROQ_API_KEY (gratis en console.groq.com).")

import pandas as pd
from smolagents import CodeAgent, LiteLLMModel

HERE = os.path.dirname(os.path.abspath(__file__))


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# -----------------------------------------------------------------------------
modelo = LiteLLMModel(model_id="groq/llama-3.3-70b-versatile", temperature=0.0)
agente = CodeAgent(
    tools=[],
    model=modelo,
    additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot"],
    max_steps=6,
)

# -----------------------------------------------------------------------------
banner("DEMO 1 — Cálculo de varios pasos")
r1 = agente.run(
    "Una caja de 24 gaseosas cuesta 18 dólares. Compro 50 cajas y me dan 15% de "
    "descuento sobre el total. ¿Cuánto pago al final? Dame solo el monto en dólares."
)
print("Respuesta:", r1)

# -----------------------------------------------------------------------------
banner("DEMO 2 — Análisis sobre datos reales (ventas)")
ventas = pd.read_csv(os.path.join(HERE, "ventas_tiendas_ecuador.csv"))
r2 = agente.run(
    "Con el DataFrame 'ventas', ¿cuántas unidades de bebidas se vendieron en total en "
    "la tienda Quito-44 durante 2016? Dame solo el número.",
    additional_args={"ventas": ventas},
)
print("Respuesta:", r2)

# -----------------------------------------------------------------------------
banner("DEMO 3 — Generar una gráfica")
r3 = agente.run(
    "Con 'ventas', haz un gráfico de barras del total de unidades de bebidas por año en "
    "Quito-44. Ponle título y guárdalo como 'verify_chart.png' con plt.savefig.",
    additional_args={"ventas": ventas},
)
chart = os.path.join(HERE, "verify_chart.png")
ok_chart = os.path.exists("verify_chart.png") or os.path.exists(chart)
print("Respuesta:", r3)
print("¿Gráfico generado?:", "SÍ" if ok_chart else "no se encontró el PNG (revisar a mano)")

# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("✓ Si los 3 demos corrieron sin error, la clase está lista.")
print("=" * 72)
