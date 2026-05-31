"""
E3 - Verificación de Groq (LLM generativo via API gratis).
Tres llamadas reales con outputs guardados literalmente.
Pre-req: export GROQ_API_KEY="..." (gratis sin tarjeta en https://console.groq.com)
         pip install openai
Outputs: groq_outputs.json, groq_outputs.txt
"""
import os, json, time, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get("GROQ_API_KEY")
if not KEY:
    sys.exit("ERROR: define GROQ_API_KEY antes de correr esto.\n"
             "       https://console.groq.com  (gratis, sin tarjeta)")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("ERROR: pip install openai")

client = OpenAI(api_key=KEY, base_url="https://api.groq.com/openai/v1")
MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
print(f"Usando modelo: {MODEL}")

def call(messages, max_tokens=300, temperature=0.3):
    t0 = time.time()
    r = client.chat.completions.create(
        model=MODEL, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    latency = time.time() - t0
    return {
        "content": r.choices[0].message.content,
        "latency_s": round(latency, 2),
        "prompt_tokens": r.usage.prompt_tokens,
        "completion_tokens": r.usage.completion_tokens,
        "total_tokens": r.usage.total_tokens,
    }

outputs = []

# ============================================================
#  Demo 1: Generación libre con contexto industrial
# ============================================================
print("\n--- Demo 1: Generación ---")
d1 = call([{
    "role": "user",
    "content": "Resume en exactamente 3 viñetas el plan de mantenimiento preventivo "
               "semanal de una llenadora industrial en una planta embotelladora. "
               "Sé concreto y técnico."
}])
print(d1["content"])
print(f"[latencia {d1['latency_s']}s, {d1['total_tokens']} tokens]")
outputs.append({"tipo": "generacion", "prompt_resumen": "mantenimiento preventivo llenadora", **d1})

# ============================================================
#  Demo 2: Clasificación de sentimiento (compare con TF-IDF de clase 33)
# ============================================================
print("\n--- Demo 2: Clasificación de sentimiento ---")
resenas = [
    "no me gustó nada la trama, qué decepción",        # NEG
    "no es para nada mala, me sorprendió",             # POS
    "buena fotografía pero la historia no funciona",   # NEG
]
d2 = call([{
    "role": "user",
    "content": "Clasifica cada reseña como POSITIVO o NEGATIVO. "
               "Devuelve sólo el resultado, una palabra por línea, en orden:\n\n"
               + "\n".join(f"{i+1}. {r}" for i, r in enumerate(resenas))
}], max_tokens=50, temperature=0.0)
print(d2["content"])
print(f"[latencia {d2['latency_s']}s, {d2['total_tokens']} tokens]")
outputs.append({"tipo": "clasificacion", "reseñas": resenas, **d2})

# ============================================================
#  Demo 3: Extracción estructurada (JSON) — ticket de planta
# ============================================================
print("\n--- Demo 3: Extracción JSON ---")
ticket = ("El compresor número 3 de la línea 2 está echando vapor desde anoche "
          "y la presión bajó. Urge revisión antes de la mañana.")
d3 = call([{
    "role": "user",
    "content": f"Extrae los datos del siguiente ticket de planta como JSON con las "
               f"llaves exactas: equipo, problema, urgencia (alta/media/baja). "
               f"Devuelve sólo el JSON, sin explicación.\n\nTicket: {ticket}"
}], max_tokens=120, temperature=0.0)
print(d3["content"])
print(f"[latencia {d3['latency_s']}s, {d3['total_tokens']} tokens]")
outputs.append({"tipo": "extraccion_json", "ticket": ticket, **d3})

# ============================================================
#  Persistir
# ============================================================
with open(os.path.join(ROOT, "groq_outputs.json"), "w") as f:
    json.dump({"model": MODEL, "outputs": outputs}, f, ensure_ascii=False, indent=2)
with open(os.path.join(ROOT, "groq_outputs.txt"), "w") as f:
    f.write(f"# Groq verification — modelo {MODEL}\n\n")
    for o in outputs:
        f.write(f"=== {o['tipo']} ({o['latency_s']}s, {o['total_tokens']} tokens) ===\n")
        f.write(o["content"] + "\n\n")
print("\nArtefactos guardados: groq_outputs.json, groq_outputs.txt")
