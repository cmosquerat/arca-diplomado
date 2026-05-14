"""
Setup de Label Studio para Clase 28 — proyecto: Pill Segmentation.

Baja el dataset 'pillsegmentation' del workspace 'abstract' de Roboflow Universe,
toma 100 imágenes al azar y las sube a Label Studio SIN etiquetas, con un
labeling config de PolygonLabels para que los estudiantes dibujen el polígono
de cada píldora.

Uso:
    export LS_URL="https://label-studio-production-281f.up.railway.app"
    export LS_TOKEN="..."
    export ROBOFLOW_API_KEY="DJqoR0JeH6JaOrpH712W"
    python -u setup_labelstudio.py
    python -u setup_labelstudio.py --rebuild   # borra y re-crea
    python -u setup_labelstudio.py --cleanup   # borra brain-tumor viejo

Diseño:
  - Usa label-studio-sdk para create/list/delete project.
  - Usa el SDK de roboflow para descargar el dataset.
  - Para subir binarios (multipart POST) se usa requests directo porque el
    SDK no expone ese endpoint.
  - Idempotente: si el proyecto ya tiene 100 tareas, no re-sube.
"""
import io
import os
import random
import sys
import time
from pathlib import Path

import requests
from PIL import Image
from label_studio_sdk import LabelStudio

random.seed(42)

LS_URL = os.environ["LS_URL"].rstrip("/")
LS_TOKEN = os.environ["LS_TOKEN"]
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "DJqoR0JeH6JaOrpH712W")

PROJECT_TITLE = "Clase 28 - Pill Segmentation"
N_IMAGES = 100
MAX_SIDE = 1280

LABELING_CONFIG = """
<View>
  <Header value="Segmentación de píldoras"/>
  <Text name="instr_1" value="Dibuje un polígono SOBRE CADA píldora visible en la imagen. Cada píldora debe ser una instancia separada."/>
  <Text name="instr_2" value="Si dos píldoras están pegadas, dibuje DOS polígonos distintos (uno por píldora)."/>
  <Text name="instr_3" value="Objetivo: enseñarle al modelo a contar píldoras una por una. La precisión del contorno importa menos que NO juntar dos en una."/>
  <Image name="image" value="$image"/>
  <PolygonLabels name="label" toName="image">
    <Label value="pill" background="#C82B40"/>
  </PolygonLabels>
</View>
""".strip()


# ============================================================================
# Auth: refresh JWT → access (para HTTP crudo solamente)
# ============================================================================
_ACCESS = {"token": None, "exp": 0.0}


def _refresh_access():
    if len(LS_TOKEN) < 100:
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
# Roboflow download
# ============================================================================
def download_roboflow_pills(dest: Path) -> Path:
    """Baja el dataset 'pillsegmentation' de Roboflow Universe."""
    if dest.exists() and any(dest.rglob("*.jpg")):
        print(f"  Ya bajado en {dest}")
        return dest

    print("  Bajando con roboflow SDK...")
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("abstract").project("pillsegmentation-oyygy")

    # Tomamos la última versión disponible
    versions = project.versions()
    if not versions:
        raise RuntimeError("No hay versiones publicadas en Roboflow.")
    version = versions[0]
    print(f"  Versión: {version.version}")

    dest.mkdir(parents=True, exist_ok=True)
    # download() crea su propio subdirectorio
    cwd_old = os.getcwd()
    os.chdir(dest)
    try:
        version.download("yolov8")
    finally:
        os.chdir(cwd_old)
    return dest


def collect_images(roboflow_dir: Path) -> list[Path]:
    """Devuelve todas las imágenes (train/valid/test) de la descarga."""
    images = []
    for sub in ["train", "valid", "test"]:
        p = roboflow_dir / "PillSegmentation-1" / sub / "images"
        if not p.exists():
            # Roboflow a veces usa nombre distinto
            for cand in roboflow_dir.rglob(f"{sub}/images"):
                if cand.is_dir():
                    p = cand
                    break
        if p.exists():
            images.extend(sorted(p.glob("*.jpg")))
            images.extend(sorted(p.glob("*.png")))
    if not images:
        # Fallback: buscar cualquier .jpg
        images = sorted(roboflow_dir.rglob("*.jpg"))
    return images


# ============================================================================
# LS upload
# ============================================================================
def upload_image_http(project_id: int, image_path: Path) -> None:
    """Sube binario a LS. Resize a MAX_SIDE para ahorrar storage."""
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


def get_or_create_project(ls, title, config, rebuild):
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


def setup_pills(ls, rebuild):
    print("\n=== Pill Segmentation ===")

    src = download_roboflow_pills(Path("/tmp/pill-seg"))
    all_imgs = collect_images(src)
    print(f"  Total imágenes en dataset: {len(all_imgs)}")

    random.shuffle(all_imgs)
    selected = all_imgs[:N_IMAGES]
    print(f"  Seleccionadas: {len(selected)}")

    project, is_new = get_or_create_project(
        ls, PROJECT_TITLE, LABELING_CONFIG, rebuild)

    if is_new or project.task_number < N_IMAGES:
        print(f"  Subiendo {len(selected)} imágenes SIN etiquetas...")
        for i, img_path in enumerate(selected, 1):
            upload_image_http(project.id, img_path)
            if i % 10 == 0 or i == len(selected):
                print(f"    [{i}/{len(selected)}]")
    else:
        print(f"  Salta: proyecto ya tiene {project.task_number} tareas")

    return project


# ============================================================================
if __name__ == "__main__":
    t0 = time.time()
    ls = LabelStudio(base_url=LS_URL, api_key=LS_TOKEN)

    if "--cleanup" in sys.argv:
        for prefix in ["Clase 28 - "]:
            for p in ls.projects.list():
                if p.title.startswith(prefix):
                    print(f"  Borrando id={p.id}: {p.title}")
                    ls.projects.delete(p.id)
        sys.exit(0)

    rebuild = "--rebuild" in sys.argv
    p = setup_pills(ls, rebuild=rebuild)
    print(f"\nTodo listo en {time.time() - t0:.0f}s.")
    print(f"  Pill Segmentation: id={p.id}")
    print(f"  URL:               {LS_URL}/projects/{p.id}/data")
