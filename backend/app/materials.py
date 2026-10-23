"""Render pass: кадр → плоская иллюстрация Qwen-Image-Edit-2511 (схема) → маски по ней (schema_masks):
каждая область схемы — своя маска, линии схемы — отдельная маска «Разделитель».

Ответ — {width, height, schema, map, groups}: схема PNG, совмещённая с кадром (InegoAlign); карта PNG
(id маски = R + 256·G: 1 — разделитель, 2… — области по убыванию площади) и список масок {id, label, area, color}.
"""

import asyncio
import base64
import io
import uuid

import httpx
import numpy as np
from PIL import Image

from . import comfy, schema_masks
from .comfy import _upload, to_png
from .config import COMFYUI_URL, QWEN_EDIT_MODEL, QWEN_EDIT_RENDER_MEGAPIXELS, log

MAX_SIDE = 2048  # кадр уменьшается до этого размера по длинной стороне

# схема: тот же кадр плоской иллюстрацией без света (Qwen-Image-Edit-2511) — раскраска, по заливкам которой резались
# элементы. 2.1 по альбедо красил доску карниза в цвет стены; 2511 слушает «соседние элементы — разными цветами» и
# проводит между ними тонкие контуры (запрет контуров не слушает — они и нужны: по ним режутся маски, сами они —
# маска «Разделитель»).
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
        schema = (await comfy._run(client, wf))["save"]
    # нарезка — секунды на CPU, мимо цикла событий
    masks = await asyncio.to_thread(_masks, schema, src.size)
    log.info("render pass %dx%d: схема 2511 → %d масок + разделитель", *src.size, len(masks["groups"]) - 1)
    return {
        "width": src.width,
        "height": src.height,
        "schema": "data:image/png;base64," + base64.b64encode(schema).decode(),
        **masks,
    }


def _masks(schema: bytes, size: tuple[int, int]) -> dict:
    w, h = size
    lbl, divider = schema_masks.segment(np.asarray(Image.open(io.BytesIO(schema)).convert("RGB")), (w, h))
    flat = np.asarray(Image.open(io.BytesIO(schema)).convert("RGB").resize((w, h), Image.BICUBIC))
    ids = np.where(divider, 1, 0).astype(np.int32)
    # области — по убыванию площади, у крупных самые различимые цвета во фронтенде
    area = np.bincount(lbl[~divider], minlength=lbl.max() + 1)
    order = [int(i) for i in np.argsort(-area) if area[i] > 0]
    remap = np.zeros(lbl.max() + 1, np.int32)
    remap[order] = np.arange(2, len(order) + 2)
    ids[~divider] = remap[lbl[~divider]]
    total = float(w * h)
    keep = ~divider
    mean = np.stack([np.bincount(lbl[keep], flat[..., c][keep], lbl.max() + 1) for c in range(3)], 1)
    mean /= np.maximum(area, 1)[:, None]
    groups = [{"id": 1, "label": "Разделитель", "area": float(divider.sum()) / total, "color": "#000000"}]
    for k, i in enumerate(order):
        r, g, b = (int(v) for v in mean[i])
        groups.append({"id": k + 2, "label": f"Область {k + 1}", "area": float(area[i]) / total, "color": f"#{r:02x}{g:02x}{b:02x}"})
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0], rgb[..., 1] = ids & 255, ids >> 8
    return {
        "map": "data:image/png;base64," + base64.b64encode(to_png(Image.fromarray(rgb))).decode(),
        "groups": groups,
    }
