"""Кадр по материалам (вкладка Render pass): нод InegoMaterialGroups в ComfyUI
(см. comfyui/custom_nodes/inego_nodes/materials.py).

Qwen-Image-Edit-2511 рисует кадр плоской иллюстрацией без света (схема); нод режет её на элементы по заливкам,
переносит их многоугольниками на место кадра (масштаб + сдвиг InegoAlign), мелким элементам (окна, двери,
водостоки) край уточняет SAM 2.1 по фото.

Ответ — карта PNG (id элемента = R + 256·G, 0 — без элемента), схема JPEG и список элементов
{id, label, area, color}: каждый кусок кадра — свой элемент, ничего не объединяется.
"""

import asyncio
import base64
import io
import json
import uuid

import httpx
from PIL import Image

from . import comfy
from .comfy import ComfyError, _upload, to_png
from .config import COMFY_TIMEOUT, COMFYUI_URL, QWEN_EDIT_MODEL, QWEN_EDIT_RENDER_MEGAPIXELS, log

MAX_SIDE = 2048  # карта строится в этом размере (по длинной стороне); фронтенд растягивает её на кадр

# схема: тот же кадр плоской иллюстрацией без света (Qwen-Image-Edit-2511) — раскраска, по заливкам которой режутся
# элементы. 2.1 по альбедо красил доску карниза в цвет стены; 2511 слушает «соседние элементы — разными цветами» и
# проводит между ними тонкие контуры (запрет контуров не слушает — они и не мешают, нод делит их между соседями).
# Просьбу о сегментационной карте (каждому элементу условный цвет) обе модели не выполняют, а 2511 при ней додумывает дом
SCHEMA_PROMPT = (
    "Turn image 1 into a clean flat colour illustration of exactly the same scene with no lighting at all: no "
    "shadows, no sun, no shading, no reflections, no texture, no gradients, and no outlines or contour lines "
    "anywhere: shapes meet edge to edge. Each building element is one solid flat colour; neighbouring elements are "
    "always separated by a clearly different colour, so the roof edge boards, fascias and soffits, window frames, "
    "glass, doors, lamps, downpipes, stone cladding panels, plinth and steps each stand out from the wall around "
    "them. Keep the exact geometry, positions and sizes, same camera and framing, crisp straight edges."
)
SCHEMA_SEED = 7


async def build_materials(image: bytes) -> dict:
    src = Image.open(io.BytesIO(image)).convert("RGB")
    src.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        await comfy._check_model(client, QWEN_EDIT_MODEL)  # веса на месте; прежняя модель другого семейства выгружается
        # схема — на родных для 2511 ~1 Мп, InegoAlign возвращает её на место кадра
        qsize = comfy.fit_size(*src.size, QWEN_EDIT_RENDER_MEGAPIXELS, step=16)
        qname = await _upload(client, to_png(src.resize(qsize, Image.LANCZOS)), f"{uuid.uuid4().hex}.png")
        wf = comfy.build_qwen2511_workflow(qname, SCHEMA_PROMPT, SCHEMA_SEED, None)
        wf.pop("save")
        # нод режет схему как есть и сам переносит готовые многоугольники на место кадра (InegoAlign.transform)
        schema = wf.pop("align")["inputs"]["image"]
        name = await _upload(client, to_png(src), f"{uuid.uuid4().hex}.png")
        wf["src"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf["materials"] = {"class_type": "InegoMaterialGroups", "inputs": {"image": ["src", 0], "schema": schema}}
        r = await client.post("/prompt", json={"prompt": wf, "client_id": uuid.uuid4().hex})
        if r.status_code != 200:
            raise ComfyError(f"ComfyUI отклонил workflow: {r.text[:500]}")
        pid = r.json()["prompt_id"]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + COMFY_TIMEOUT
        while loop.time() < deadline:
            await asyncio.sleep(1)
            h = (await client.get(f"/history/{pid}")).json().get(pid)
            if not h:
                continue
            status = h.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise ComfyError(f"Ошибка ComfyUI: {msgs[-1][1].get('exception_message') if msgs else status}")
            if status.get("completed"):
                out = h["outputs"]["materials"]
                files = []
                for f in out["images"]:
                    v = await client.get("/view", params={"filename": f["filename"], "subfolder": f["subfolder"], "type": f["type"]})
                    v.raise_for_status()
                    files.append(v.content)
                groups = json.loads(out["groups"][0])
                log.info("материалы %dx%d: %d элементов", src.width, src.height, len(groups))
                return {
                    "width": src.width,
                    "height": src.height,
                    "map": "data:image/png;base64," + base64.b64encode(files[0]).decode(),
                    "schema": "data:image/jpeg;base64," + base64.b64encode(files[1]).decode() if len(files) > 1 else None,
                    "groups": groups,
                }
        raise ComfyError("ComfyUI не ответил вовремя")
