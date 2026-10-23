"""Render pass: кадр → плоская иллюстрация Qwen-Image-Edit-2511 (схема), больше ничего.

Нарезка схемы на элементы (заливки, многоугольники, SAM 2.1 по фото) пока снята — она в истории git
(коммит «Render pass: elements from the 2511 flat only…», нод InegoMaterialGroups остаётся в ComfyUI).

Ответ — {width, height, schema}: схема PNG, совмещённая с кадром (InegoAlign).
"""

import base64
import io
import uuid

import httpx
from PIL import Image

from . import comfy
from .comfy import _upload, to_png
from .config import COMFYUI_URL, QWEN_EDIT_MODEL, QWEN_EDIT_RENDER_MEGAPIXELS, log

MAX_SIDE = 2048  # кадр уменьшается до этого размера по длинной стороне

# схема: тот же кадр плоской иллюстрацией без света (Qwen-Image-Edit-2511) — раскраска, по заливкам которой резались
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
        schema = (await comfy._run(client, wf))["save"]
    log.info("render pass %dx%d: схема 2511 готова", src.width, src.height)
    return {
        "width": src.width,
        "height": src.height,
        "schema": "data:image/png;base64," + base64.b64encode(schema).decode(),
    }
