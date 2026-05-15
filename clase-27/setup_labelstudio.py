"""
Setup de Label Studio para Clase 27 — proyecto: Brain Tumor Detection.

Sube las 893 imágenes (4 MB) y crea anotaciones reales para ~80% de ellas.
Las ~180 restantes quedan vacías para que los estudiantes etiqueten.

Uso:
    export LS_URL="https://label-studio-production-281f.up.railway.app"
    export LS_TOKEN="..."
    python -u setup_labelstudio.py
    python -u setup_labelstudio.py --cleanup   # borra proyectos viejos cold-drinks/coffee
    python -u setup_labelstudio.py --rebuild   # borra y re-crea desde cero

Diseño:
  - Usa label-studio-sdk para lo que tiene método: `projects.create`,
    `tasks.list`, `annotations.create`, `projects.delete`.
  - Para subir binarios (multipart POST) y bajar imágenes (GET con auth)
    se usa `requests` directo porque el SDK no expone esos endpoints.
  - Idempotente: si el proyecto ya tiene 893 tareas, salta la subida y va
    directo a crear annotations.
"""
import io
import os
import random
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import requests
from PIL import Image
from label_studio_sdk import LabelStudio

random.seed(42)

LS_URL = os.environ["LS_URL"].rstrip("/")
LS_TOKEN = os.environ["LS_TOKEN"]

PROJECT_TITLE = "Clase 27 - Brain Tumor Detection"
BRAIN_CLASSES = ["negative", "positive"]
N_PRELABELED_PCT = 80
MAX_SIDE = 1280

BRAIN_TUMOR_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip"

LABELING_CONFIG = """
<View>
  <Header value="Detección de tumores cerebrales (MRI)"/>
  <Text name="instr_1" value="Cada imagen es una resonancia magnética (MRI) de cerebro."/>
  <Text name="instr_2" value="Dibuje un rectángulo sobre cualquier región que considere sospechosa de tumor y asigne 'positive'. Si está seguro de que no hay tumor, asigne 'negative' sobre una región de tejido sano, o deje la imagen sin rectángulos y presione Submit."/>
  <Text name="instr_3" value="Importante: no se espera que sean radiólogos. El objetivo es comparar después el impacto del etiquetado no experto sobre las métricas del modelo."/>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="positive" background="#E60012"/>
    <Label value="negative" background="#16A34A"/>
  </RectangleLabels>
</View>
""".strip()


# ============================================================================
# Auth: refresh JWT → access (para HTTP crudo solamente)
# ============================================================================
_ACCESS = {"token": None, "exp": 0.0}


def _refresh_access():
    if len(LS_TOKEN) < 100:                     # legacy DRF token
        _ACCESS["token"] = LS_TOKEN
        _ACCESS["exp"] = time.time() + 365 * 24 * 3600
        return
    r = requests.post(f"{LS_URL}/api/token/refresh",
                      json={"refresh": LS_TOKEN}, timeout=30)
    r.raise_for_status()
    _ACCESS["token"] = r.json()["access"]
    _ACCESS["exp"] = time.time() + 180


def auth_headers():
    if not _ACCESS["token"] or time.time() > _ACCESS["exp"]:
        _refresh_access()
    return {"Authorization": f"Bearer {_ACCESS['token']}"}


def http_with_retry(method, url, **kwargs):
    """Reintenta hasta 4 veces ante errores de red o 401."""
    headers = kwargs.pop("headers", {})
    last_err = None
    for attempt in range(4):
        try:
            r = requests.request(method, url,
                                 headers={**auth_headers(), **headers},
                                 **kwargs)
            if r.status_code == 401:
                _refresh_access()
                r = requests.request(method, url,
                                     headers={**auth_headers(), **headers},
                                     **kwargs)
            return r
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err


# ============================================================================
# Helpers (binario via HTTP, lógica via SDK)
# ============================================================================
def upload_image_http(project_id: int, image_path: Path) -> None:
    """Sube binario al proyecto. SDK no tiene método para multipart."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    buf.seek(0)
    r = http_with_retry(
        "POST",
        f"{LS_URL}/api/projects/{project_id}/import",
        files={"file": (image_path.name, buf, "image/jpeg")},
        timeout=60,
    )
    r.raise_for_status()


def ls_sanitize(name: str) -> str:
    """Reproduce la sanitización que hace LS al almacenar el archivo.

    LS reemplaza espacios y paréntesis: '77 (4).jpg' → '77_4.jpg'.
    """
    out = name.replace("(", "").replace(")", "")
    out = out.replace(" ", "_")
    return out


def build_filename_to_task_id(ls: LabelStudio, project_id: int) -> dict[str, int]:
    """Mapea nombre original → task_id. Incluye versión sanitizada por LS."""
    mapping = {}
    for t in ls.tasks.list(project=project_id, page_size=200):
        stored = t.data["image"].split("/")[-1]
        # /data/upload/<pid>/<uuid8>-<sanitized_name>
        sanitized = stored[9:] if len(stored) > 9 and stored[8] == "-" else stored
        mapping[sanitized] = t.id
    return mapping


def read_yolo_bbox(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            out.append((int(parts[0]), *map(float, parts[1:5])))
    return out


def bbox_to_ls(cls, cx, cy, w, h):
    return {
        "from_name": "label", "to_name": "image", "type": "rectanglelabels",
        "value": {
            "x": (cx - w / 2) * 100, "y": (cy - h / 2) * 100,
            "width": w * 100, "height": h * 100, "rotation": 0,
            "rectanglelabels": [BRAIN_CLASSES[cls]],
        },
    }


def download_brain_tumor(dest: Path) -> Path:
    if (dest / "images" / "train").exists():
        print(f"  Ya bajado en {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "brain-tumor.zip"
    print("  Bajando brain-tumor.zip (~4 MB)...")
    urllib.request.urlretrieve(BRAIN_TUMOR_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    print(f"  Extraído en {dest}")
    return dest


def get_or_create_project(ls: LabelStudio, title: str, config: str, rebuild: bool):
    """Idempotente: usa proyecto existente si tiene suficientes tasks; si no, lo crea."""
    existing = [p for p in ls.projects.list() if p.title == title]
    if existing and not rebuild:
        p = existing[0]
        print(f"  Proyecto existente reutilizado: id={p.id} (tasks={p.task_number})")
        return p, False
    for p in existing:
        print(f"  Borrando id={p.id}: {p.title}")
        ls.projects.delete(p.id)
    p = ls.projects.create(title=title, label_config=config)
    print(f"  Proyecto creado: id={p.id}")
    return p, True


# ============================================================================
# Setup principal
# ============================================================================
def setup_brain_tumor(ls: LabelStudio, rebuild: bool):
    print("\n=== Brain Tumor Detection ===")
    src = download_brain_tumor(Path("/tmp/brain-tumor"))
    project, is_new = get_or_create_project(ls, PROJECT_TITLE, LABELING_CONFIG, rebuild)

    images = sorted((src / "images" / "train").glob("*.jpg"))
    random.shuffle(images)
    cutoff = int(len(images) * N_PRELABELED_PCT / 100)
    to_prelabel = set(images[:cutoff])

    # FASE 1: subir imágenes solo si es nuevo o le faltan
    if is_new or project.task_number < len(images):
        print(f"  Fase 1: subiendo {len(images)} imágenes...")
        for i, img_path in enumerate(images, 1):
            upload_image_http(project.id, img_path)
            if i % 50 == 0 or i == len(images):
                print(f"    [{i}/{len(images)}]")
    else:
        print(f"  Fase 1: salta (proyecto ya tiene {project.task_number} tareas)")

    # FASE 2: mapear filename -> task_id
    print("  Fase 2: matcheando tasks por filename (vía SDK)...")
    fname_to_tid = build_filename_to_task_id(ls, project.id)
    print(f"    {len(fname_to_tid)} entradas en mapeo")

    # FASE 3: crear annotations (skip si ya tiene)
    print("  Fase 3: identificando tareas ya con annotation...")
    already_annotated = {t.id for t in ls.tasks.list(project=project.id, page_size=200)
                          if t.total_annotations and t.total_annotations > 0}
    print(f"    ya con annotation: {len(already_annotated)}")

    print(f"  Fase 4: creando annotations para ~{cutoff} tareas...")
    n_ok, n_empty, n_skip_match, n_skip_dup = 0, 0, 0, 0
    for img_path in images:
        if img_path not in to_prelabel:
            continue
        tid = fname_to_tid.get(ls_sanitize(img_path.name))
        if tid is None:
            print(f"    !! no match: {img_path.name}")
            n_skip_match += 1
            continue
        if tid in already_annotated:
            n_skip_dup += 1
            continue
        boxes = read_yolo_bbox(src / "labels" / "train" / (img_path.stem + ".txt"))
        if not boxes:
            n_empty += 1
            continue
        result = [bbox_to_ls(*b) for b in boxes]
        try:
            ls.annotations.create(
                id=tid,
                result=result,
                ground_truth=True,
                was_cancelled=False,
            )
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"    [{n_ok} annotations]")
        except Exception as e:
            print(f"    !! error en task {tid}: {type(e).__name__}: {e}")
            n_skip_match += 1

    print(f"  Listo: {len(images)} tareas | nuevas annotations: {n_ok} | "
          f"sin bbox (negative): {n_empty} | ya tenían: {n_skip_dup} | "
          f"sin match: {n_skip_match}")
    return project


# ============================================================================
if __name__ == "__main__":
    t0 = time.time()
    ls = LabelStudio(base_url=LS_URL, api_key=LS_TOKEN)

    if "--cleanup" in sys.argv:
        for prefix in ["Clase 27 - Cold Drinks", "Clase 27 - Coffee"]:
            for p in ls.projects.list():
                if p.title.startswith(prefix):
                    print(f"  Borrando id={p.id}: {p.title}")
                    ls.projects.delete(p.id)

    rebuild = "--rebuild" in sys.argv
    p = setup_brain_tumor(ls, rebuild=rebuild)
    print(f"\nTodo listo en {time.time() - t0:.0f}s.")
    print(f"  Brain Tumor: id={p.id}")
    print(f"  URL:         {LS_URL}/projects/{p.id}/data")
