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
from PIL import Image

from .comfy import ComfyError, _upload, _use_family, to_png
from .config import COMFY_TIMEOUT, COMFYUI_URL, log

ZONES_MAX_SIDE = 2048  # карта строится в этом размере (по длинной стороне); фронтенд растягивает её на кадр


async def build_zones(image: bytes) -> dict:
    src = Image.open(io.BytesIO(image)).convert("RGB")
    src.thumbnail((ZONES_MAX_SIDE, ZONES_MAX_SIDE), Image.LANCZOS)
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        # своим моделям карты нужна видеопамять: модели генерации выгружаются при следующей смене семейства
        await _use_family(client, "zones")
        name = await _upload(client, to_png(src), f"{uuid.uuid4().hex}.png")
        wf = {
            "src": {"class_type": "LoadImage", "inputs": {"image": name}},
            "zones": {"class_type": "InegoZoneMap", "inputs": {"image": ["src", 0]}},
        }
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
                return {
                    "width": src.width,
                    "height": src.height,
                    "map": "data:image/png;base64," + base64.b64encode(files[0]).decode(),
                    "alpha": "data:image/png;base64," + base64.b64encode(files[1]).decode(),
                    "veg": "data:image/png;base64," + base64.b64encode(files[2]).decode(),
                    "zones": zones,
                }
        raise ComfyError("ComfyUI не ответил вовремя")
