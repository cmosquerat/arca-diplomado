"""
E2.5 - Cargar BETO (BERT espanol pre-entrenado) y extraer atencion REAL
sobre frases con negacion. Demuestra que un Transformer pre-entrenado
ya "entiende" la negacion sin entrenamiento adicional.

Outputs: beto_attention.npy, beto_tokens.json, beto_layer_head_info.json
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"  # BETO
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Cargando {MODEL_NAME} (descarga ~440MB primera vez)...")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True, attn_implementation="eager")
model.eval().to(device)
print(f"Parametros BETO: {sum(p.numel() for p in model.parameters()):,}")

# Frases candidatas (varias para elegir la que mejor visualice)
FRASES = [
    "no me gustó nada la trama",
    "no es para nada mala",
    "no podría estar más contento",
    "no me arrepiento de haberla visto",
]

all_outputs = {}
for frase in FRASES:
    print(f"\n--- frase: '{frase}' ---")
    inp = tok(frase, return_tensors="pt").to(device)
    tokens = tok.convert_ids_to_tokens(inp["input_ids"][0])
    print(f"tokens BETO: {tokens}")
    with torch.no_grad():
        out = model(**inp)
    # attentions: tuple de 12 layers, cada uno (batch, heads, seq, seq)
    attn = torch.stack(out.attentions, dim=0).squeeze(1).cpu().numpy()  # (layers, heads, seq, seq)
    print(f"attentions shape (layers, heads, q, k): {attn.shape}")
    all_outputs[frase] = {"tokens": tokens, "attn": attn}

# ============================================================
#  Elegir la mejor capa+cabeza para "no me gusto nada la trama":
#  buscar el (layer, head) donde "gustó" presta mas atencion a "no"
# ============================================================
clave = "no me gustó nada la trama"
d = all_outputs[clave]
toks = d["tokens"]
attn = d["attn"]  # (12, 12, n, n)

# Indices de "no" y "gustó" (BETO uncased + WordPiece)
def find_idx(tokens, target):
    for i, t in enumerate(tokens):
        if t.lower().replace("##", "") == target.lower():
            return i
    return None

i_gusto = find_idx(toks, "gusto") or find_idx(toks, "gustó")
i_no    = find_idx(toks, "no")
print(f"\nindex 'gustó' = {i_gusto}, index 'no' = {i_no}")

if i_gusto is not None and i_no is not None:
    L, H = attn.shape[0], attn.shape[1]
    scores = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            scores[l, h] = attn[l, h, i_gusto, i_no]
    best = np.unravel_index(np.argmax(scores), scores.shape)
    print(f"\nMejor (layer,head): {best}, score={scores[best]:.3f}")
    print(f"Top 5 (layer,head) por atencion gustó→no:")
    flat = sorted([(scores[l, h], l, h) for l in range(L) for h in range(H)], reverse=True)[:5]
    for sc, l, h in flat:
        print(f"  layer {l:2d} head {h:2d}: {sc:.3f}")
    BEST_LAYER, BEST_HEAD = int(best[0]), int(best[1])
else:
    BEST_LAYER, BEST_HEAD = 5, 7
    print(f"Fallback (layer,head)=({BEST_LAYER},{BEST_HEAD})")

# ============================================================
#  Guardar artefactos para gen_figs.py
# ============================================================
# Para la frase clave, guardar todos los heads de la mejor layer
viz_attn = attn[BEST_LAYER]  # (heads, seq, seq)
np.save(os.path.join(ROOT, "beto_attention.npy"), viz_attn)

with open(os.path.join(ROOT, "beto_tokens.json"), "w") as f:
    json.dump({
        "frase": clave,
        "tokens": toks,
        "best_layer": BEST_LAYER,
        "best_head": BEST_HEAD,
        "model": MODEL_NAME,
        "params": int(sum(p.numel() for p in model.parameters())),
    }, f, ensure_ascii=False, indent=2)

# Guardar tambien atencion de otras frases (capa elegida)
all_data = {}
for frase, d in all_outputs.items():
    all_data[frase] = {
        "tokens": d["tokens"],
        "attn_layer_best": d["attn"][BEST_LAYER].tolist(),  # (heads, seq, seq)
    }
with open(os.path.join(ROOT, "beto_all_frases.json"), "w") as f:
    json.dump(all_data, f, ensure_ascii=False)

print(f"\nArtefactos guardados:")
print(f"  beto_attention.npy        - {viz_attn.shape} (heads, q, k) layer {BEST_LAYER}")
print(f"  beto_tokens.json          - tokens + best layer/head")
print(f"  beto_all_frases.json      - atencion de 4 frases")
