"""Кадр по материалам (вкладка Render pass): нод InegoMaterialGroups в ComfyUI
(см. comfyui/custom_nodes/inego_nodes/materials.py).

Сначала Intrinsic v2 снимает с кадра свет (альбедо), по нему Qwen-Image-2.1 рисует кадр плоской заливкой
(схема материалов, InegoAlign возвращает её на место исходника) — материалы и куски ищутся по ней, карта ложится на исходный кадр.

Ответ — карта PNG (значение пикселя — id группы, 0 — без материала), схема JPEG и список групп
{id, label, area, color}: одна группа — один материал по всему кадру, независимо от света и теней.
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
from .comfy import ComfyError, _upload, to_png
from .config import COMFY_TIMEOUT, COMFYUI_URL, QWEN21_RENDER_MEGAPIXELS, QWEN21_TURBO_MODEL, log
from .material_judge import judge_pairs
from .prompting import q21_tags

MAX_SIDE = 2048  # карта строится в этом размере (по длинной стороне); фронтенд растягивает её на кадр

# схема материалов: тот же кадр как base-colour (albedo) проход 3D-рендера — каждый материал одной заливкой своего
# цвета, без света; мелкие детали (водостоки, фонари, рамы) — отдельными пятнами своего цвета
SCHEMA_PROMPT = (
    "Redraw image 1 as a clean flat vector illustration of exactly the same scene that works like the unlit "
    "base-colour (albedo) pass of a 3D render: there is no lighting at all, so every material is one solid flat fill "
    "of its own real colour everywhere it appears: both roof slopes, walls in sun and in shade, sides facing away all "
    "have exactly the same colour as the rest of that material. No texture, no reflections, no shading, no shadows, "
    "no highlights and no gradients. Different materials keep clearly different colours: stone or brick cladding, "
    "plaster, dark panels, window frames, glass, roofing, gutters and downpipes, chimneys, wall lamps, door, steps, "
    "paving, lawn, trees and sky each their own flat colour; glass is a flat solid tint without reflections. Keep "
    "every small detail: every downpipe, gutter, wall lamp, window, frame mullion, door, step, railing and chimney "
    "stays exactly in place and the same size. Same camera and framing, crisp straight edges."
)
SCHEMA_SEED = 7


async def build_materials(image: bytes) -> dict:
    src = Image.open(io.BytesIO(image)).convert("RGB")
    src.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        await comfy._check_model(client, QWEN21_TURBO_MODEL)  # веса на месте; прежняя модель другого семейства выгружается
        qsize = comfy.fit_size(*src.size, QWEN21_RENDER_MEGAPIXELS, step=32)
        qname = await _upload(client, to_png(src.resize(qsize, Image.LANCZOS)), f"{uuid.uuid4().hex}.png")
        wf = comfy.build_qwen21_workflow(qname, q21_tags(SCHEMA_PROMPT), SCHEMA_SEED, None)
        # Qwen рисует схему не по фото, а по его альбедо (Intrinsic v2): по фото в схему переходили тень под навесом
        # и солнце на фронтоне. Совмещение (InegoAlign) — по-прежнему с фото
        wf["albedo"] = {"class_type": "InegoIntrinsicAlbedo", "inputs": {"image": ["src_img", 0]}}
        wf["text"]["inputs"]["images.image_1"] = ["albedo", 0]
        schema = wf.pop("save")["inputs"]["images"]  # уже совмещена с исходником (InegoAlign)
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
                pairs = json.loads(out["pairs"][0]) if "pairs" in out else []
                # спорные пары (близкий цвет на схеме, группы раздельные) — на суд DeepSeek
                ids = np.asarray(Image.open(io.BytesIO(files[0])))
                ids, groups = await judge_pairs(src.resize((ids.shape[1], ids.shape[0])), ids, groups, pairs)
                buf = io.BytesIO()
                Image.fromarray(ids, "L").save(buf, "PNG")
                log.info("материалы %dx%d: %d групп", src.width, src.height, len(groups))
                return {
                    "width": src.width,
                    "height": src.height,
                    "map": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                    "schema": "data:image/jpeg;base64," + base64.b64encode(files[1]).decode() if len(files) > 1 else None,
                    "groups": groups,
                }
        raise ComfyError("ComfyUI не ответил вовремя")
