"""Карта зон кадра (вкладка Render pass): нод InegoZoneMap в ComfyUI (см. comfyui/custom_nodes/inego_nodes/zones.py).

Ответ — карта зон PNG (значение пикселя — id зоны, 0 — без зоны), альфа растительности PNG (мягкие края веток),
id растительности под альфой PNG и список зон {id, label, kind, group}: kind — scene | veg | part | material.
"""

import asyncio
import base64
import io
import json
import uuid

import httpx
import numpy as np
from PIL import Image

from . import comfy
from .comfy import ComfyError, _upload, _use_family, to_png
from .config import (
    COMFY_TIMEOUT, COMFYUI_URL, QWEN21_MODELS, QWEN21_RENDER_MEGAPIXELS, QWEN_EDIT_MODEL,
    QWEN_EDIT_RENDER_MEGAPIXELS, RENDER_MEGAPIXELS, log,
)
from .prompting import q21_tags, q2511_text
from .zone_check import check_zones

ZONES_MAX_SIDE = 2048  # карта строится в этом размере (по длинной стороне); фронтенд растягивает её на кадр


def _png_url(arr: np.ndarray) -> str:
    b = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), "L").save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


# схема материалов: выбранная модель перерисовывает кадр плоскими цветами, как Material ID проход 3D-рендера —
# без отражений и теней. Из неё берётся стекло: во всех пробах оно чисто голубое и точно по краям, а у рам, стоек и
# глухих панелей отражения сбивали остальные способы. Цвета остальных материалов модель выбирает нестабильно (камень
# и штукатурку то разделяет, то нет), поэтому материалы стен — по альбедо в самой ноде
SCHEMA_PROMPT = (
    "Convert image 1 into a flat-colour material ID map of the same image, like a 3D render's material ID pass: "
    "every distinct material is filled with one solid, uniform, flat colour with no shading, no shadows, no "
    "reflections, no texture and no gradients. All glass and windows are one flat cyan; each different wall "
    "material (each brick, plaster, panel colour) is its own distinct flat saturated colour; metal frames and "
    "railings flat yellow; roof flat purple; sky flat white; trees and grass flat green. Keep every edge, "
    "outline, window, mullion, balcony and building exactly in place, same camera and framing."
)
SCHEMA_SEED = 7


def _schema_workflow(name: str, src: Image.Image, model: str) -> tuple[dict, list]:
    """Workflow схемы на выбранной модели и ссылка на её картинку (без SaveImage — её берёт нода карты)."""
    if model in QWEN21_MODELS:
        wf = comfy.build_qwen21_workflow(name, q21_tags(SCHEMA_PROMPT), SCHEMA_SEED, None)
    elif model == QWEN_EDIT_MODEL:
        wf = comfy.build_qwen2511_workflow(name, q2511_text(SCHEMA_PROMPT), SCHEMA_SEED, None)
    else:
        w, h = comfy.fit_size(*src.size, RENDER_MEGAPIXELS)
        wf = comfy.build_workflow(name, w, h, SCHEMA_PROMPT, 1.0, SCHEMA_SEED, None, model=model)
        # klein кадр не совмещает — как у Qwen, по контурам до исходника
        wf["align"] = {"class_type": "InegoAlign",
                       "inputs": {"image": wf["save"]["inputs"]["images"], "source": ["src_img", 0],
                                  "max_scale": 0.08, "max_shift": 0.08}}
        wf["save"]["inputs"]["images"] = ["align", 0]
    schema = wf.pop("save")["inputs"]["images"]
    return wf, schema


def _schema_size(src: Image.Image, model: str) -> tuple[int, int]:
    if model in QWEN21_MODELS:
        return comfy.fit_size(*src.size, QWEN21_RENDER_MEGAPIXELS, step=32)
    if model == QWEN_EDIT_MODEL:
        return comfy.fit_size(*src.size, QWEN_EDIT_RENDER_MEGAPIXELS)
    return comfy.fit_size(*src.size, RENDER_MEGAPIXELS)


async def build_zones(image: bytes, model: str | None = None) -> dict:
    """model — модель генерации из шапки: ею рисуется схема материалов; None — без схемы."""
    src = Image.open(io.BytesIO(image)).convert("RGB")
    src.thumbnail((ZONES_MAX_SIDE, ZONES_MAX_SIDE), Image.LANCZOS)
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        wf: dict = {}
        schema = None
        if model:
            await comfy._check_model(client, model)  # веса на месте; прежняя модель другого семейства выгружается
            sname = await _upload(client, to_png(src.resize(_schema_size(src, model), Image.LANCZOS)),
                                  f"{uuid.uuid4().hex}.png")
            wf, schema = _schema_workflow(sname, src, model)
        else:
            await _use_family(client, "zones")
        name = await _upload(client, to_png(src), f"{uuid.uuid4().hex}.png")
        wf["zsrc"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf["zones"] = {"class_type": "InegoZoneMap", "inputs": {"image": ["zsrc", 0]}}
        if schema is not None:
            wf["zones"]["inputs"]["schema"] = schema
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
                out = h["outputs"]["zones"]
                files = []
                for f in out["images"]:
                    v = await client.get("/view", params={"filename": f["filename"], "subfolder": f["subfolder"], "type": f["type"]})
                    v.raise_for_status()
                    files.append(v.content)
                zones = json.loads(out["zones"][0])
                log.info("карта зон %dx%d: %d зон", src.width, src.height, len(zones))
                # подписи проверяет зрячая модель: детекторы находят самое похожее на слово, а не то, что есть
                ids = np.asarray(Image.open(io.BytesIO(files[0])).convert("L"))
                veg = np.asarray(Image.open(io.BytesIO(files[2])).convert("L"))
                ids, veg, zones = await check_zones(src, ids, veg, zones)
                return {
                    "width": src.width,
                    "height": src.height,
                    "map": _png_url(ids),
                    "alpha": "data:image/png;base64," + base64.b64encode(files[1]).decode(),
                    "veg": _png_url(veg),
                    "zones": zones,
                }
        raise ComfyError("ComfyUI не ответил вовремя")
