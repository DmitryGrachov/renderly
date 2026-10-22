"""Судья спорных пар материалов (Render pass): DeepSeek смотрит две группы и отвечает, один ли это материал.

Нод InegoMaterialGroups отдаёт пары групп, которые остались раздельными, хотя на плоской схеме у них близкий цвет:
одинаковые серые панели, разложенные Materialistic по разным группам, стёкла в тени. Разметку целиком зрячей модели
не доверяем (на листе из многих плиток она путала номера и меняла верные подписи так же часто, как неверные) —
только узкий вопрос «да / нет» по одной паре: кадр дважды, в первом подсвечена группа A, во втором — B.
Пары спрашиваются параллельно; уверенное «да» сливает группы (система непересекающихся множеств).
"""

import asyncio
import io
import json
import os
import time

import numpy as np
from PIL import Image
from scipy import ndimage

from .config import log
from .prompting import DEEPSEEK_API_KEY, _chat, _data_url

SYSTEM = (
    "You compare two surfaces in a photo or render of a building or scene. You get the same image twice: in image 1 "
    "the surface A is shown in full colour with a bright magenta outline, in image 2 the surface B is shown the same "
    "way; everything else is dimmed. A and B may consist of several separate patches. Decide whether A and B are the "
    "same material: the same product with the same colour and finish (the same brick, the same grey facade panels, "
    "the same window glass, the same plaster, the same metal). Ignore lighting, shade, sun, reflections in glass and "
    "perspective — a wall in shade and the same wall in sun are the same material; glass reflecting different things "
    "is the same glass. Different products are different materials even if similar in colour (window frames vs facade "
    "panels, grass vs bushes, paving vs a metal downpipe). Answer with compact JSON, no explanations: "
    '{"same": true or false, "confidence": <0-1>, "a": "<material of A, 1-3 Russian words>", '
    '"b": "<material of B, 1-3 Russian words>"}'
)
SAME_CONFIDENCE = 0.7  # слияние — только при такой уверенности модели
SIDE = 1024  # кадр для судьи по длинной стороне
JUDGE_EFFORT = os.getenv("MATERIAL_JUDGE_EFFORT", "low")
PARALLEL = 16  # пары независимы; один ответ DeepSeek — 10–30 с


def _highlight(src: Image.Image, ids: np.ndarray, gid: int) -> bytes:
    """Кадр, где группа в полном цвете с пурпурным контуром, остальное затемнено."""
    img = src.copy()
    img.thumbnail((SIDE, SIDE), Image.LANCZOS)
    m = np.asarray(Image.fromarray(ids).resize(img.size, Image.NEAREST)) == gid
    a = np.asarray(img).astype(np.float32)
    a[~m] *= 0.3
    edge = m ^ ndimage.binary_erosion(m, iterations=2)
    a[edge] = (255, 0, 255)
    out = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


async def _ask(src: Image.Image, ids: np.ndarray, a: int, b: int, sem: asyncio.Semaphore) -> dict | None:
    async with sem:
        try:
            raw = await _chat(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Image 1 — surface A."},
                        {"type": "image_url", "image_url": {"url": _data_url(_highlight(src, ids, a), SIDE)}},
                        {"type": "text", "text": "Image 2 — surface B. Are A and B the same material?"},
                        {"type": "image_url", "image_url": {"url": _data_url(_highlight(src, ids, b), SIDE)}},
                    ]},
                ],
                json_mode=True,
                # вопрос узкий «да / нет»: с рассуждением по умолчанию ответ шёл ~70 с, пары — дольше всего разбора
                effort=JUDGE_EFFORT,
            )
            return json.loads(raw)
        except Exception as e:  # noqa: BLE001 — судья необязателен: без ответа пара остаётся раздельной
            log.warning("судья материалов: пара %d-%d без ответа: %s", a, b, e)
            return None


async def judge_pairs(src: Image.Image, ids: np.ndarray, groups: list[dict], pairs: list[list]) -> tuple[np.ndarray, list[dict]]:
    """Карта и группы после слияния уверенных «да». Без ключа DeepSeek или пар — как было."""
    if not DEEPSEEK_API_KEY or not pairs:
        return ids, groups
    t0 = time.time()
    sem = asyncio.Semaphore(PARALLEL)
    answers = await asyncio.gather(*(_ask(src, ids, a, b, sem) for a, b, _ in pairs))
    n = int(ids.max())
    parent = list(range(n + 1))

    def root(g: int) -> int:
        while parent[g] != g:
            g = parent[g]
        return g

    merged = 0
    for (a, b, d), ans in zip(pairs, answers):
        if not ans:
            continue
        same = bool(ans.get("same")) and float(ans.get("confidence", 0)) >= SAME_CONFIDENCE
        log.info("судья материалов: %d-%d (ΔE %.1f): %s %.2f — %s / %s", a, b, d, "да" if ans.get("same") else "нет",
                 float(ans.get("confidence", 0)), ans.get("a"), ans.get("b"))
        if same and root(a) != root(b):
            parent[root(b)] = root(a)
            merged += 1
    log.info("судья материалов: %d пар за %.1f с, слито %d", len(pairs), time.time() - t0, merged)
    if not merged:
        return ids, groups
    # слияние и новые номера — по убыванию площади, как в ноде
    lut = np.array([root(g) for g in range(n + 1)], np.int32)
    ids = lut[ids]
    area = np.bincount(ids.ravel(), minlength=n + 1)
    color = {g["id"]: g["color"] for g in groups}
    order = [g for g in np.argsort(-area) if g and area[g]]
    renum = np.zeros(len(lut), np.int32)
    out = []
    for k, g in enumerate(order, 1):
        renum[g] = k
        out.append({"id": k, "label": f"Материал {k}", "area": round(float(area[g]) / ids.size, 5), "color": color[g]})
    return renum[ids].astype(np.uint8), out
