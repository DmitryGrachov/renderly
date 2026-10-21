"""Разбор сцены для правок Inpainting: свет, атмосфера, цветокор, камера, стилистика и настроение кадра.

Правка (добавить, изменить, материал, удалить) должна выглядеть снятой в том же кадре. Без описания сцены модели
вписывали объект «вообще»: машину не той эпохи, светлый материал в сумерках, сочный цвет в приглушённом кадре.
DeepSeek один раз описывает текущий кадр (кэш по картинке); описание уходит в план правки (что здесь правдоподобно),
в запрос генерации (свет и цветокор) и в пересчёт кадра (в какую фотографию вписывать).

Ультрареализм — всегда: фото, снятое камерой, а не рендер. Без слов «зерно» и «шум»: Qwen-2.1 и FLUX.2 klein
рисуют их буквально («наждачка» по стенам).
"""

import hashlib
import io
import json
import logging
from collections import OrderedDict

from PIL import Image

from .prompting import DEEPSEEK_API_KEY, _chat, _data_url, llm_model

log = logging.getLogger("inego")

SCENE_FIELDS = {
    "light": (
        "time of day; where the sun or main light is relative to the camera (left / right / behind / against) and how "
        "high; hard or soft; light colour in Kelvin; which faces are lit and which are in shade; shadow direction and "
        "softness; artificial lights that are on"
    ),
    "atmosphere": "weather, sky, season, haze or mist, wet or dry surfaces",
    "colour_grade": "white balance in Kelvin, 4-6 palette colours with hex, contrast, saturation, exposure, black level",
    "camera": "focal-length feel, camera height, depth of field, sharpness, lens character",
    "style": (
        "architectural and landscape style, era and region feel, palette of materials, level of upkeep, what kind of "
        "objects, plants, cars and people naturally belong here"
    ),
    "mood": "the atmosphere of the photo in a few words",
}
SCENE_SYSTEM = (
    "You are a director of photography and an architectural photographer. Describe the photograph so that anything "
    "edited into it can be made indistinguishable from it, as if photographed in the same shot. Be concrete and "
    "measurable, describe what you see, no praise words. Never mention grain or noise. English. Answer with JSON "
    "with exactly these keys, each value 15-45 words:\n" + "\n".join(f'- "{k}": {v}' for k, v in SCENE_FIELDS.items())
)

# ультрареализм для всех правок: фото камерой, а не рендер (без «grain» / «noise»)
PHOTO_REAL = (
    "Ultra-realistic: an unretouched real photograph taken in a single shot with a full-frame camera and a sharp "
    "prime lens — physically correct light and falloff, soft contact shadows, real materials with micro-texture, "
    "subtle wear and natural imperfections, true reflections; nothing looks rendered, painted, CGI or pasted."
)

_cache: OrderedDict[str, dict] = OrderedDict()
CACHE_SIZE = 64


async def describe_scene(image: Image.Image) -> dict:
    """Описание кадра по SCENE_FIELDS (кэш по картинке); без DeepSeek или при ошибке — пустой словарь."""
    if not DEEPSEEK_API_KEY:
        return {}
    small = image.convert("RGB").copy()
    small.thumbnail((1536, 1536))
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=90)
    data = buf.getvalue()
    key = hashlib.sha256(data + llm_model.get().encode()).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    try:
        raw = await _chat(
            [
                {"role": "system", "content": SCENE_SYSTEM},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url(data, 1536)}}]},
            ],
            json_mode=True,
        )
        parsed = json.loads(raw)
    except Exception as e:  # noqa: BLE001 — правка не должна падать из-за разбора сцены
        log.warning("Сцена не разобрана (DeepSeek): %s", e)
        return {}
    scene = {k: " ".join(str(parsed.get(k) or "").split()).rstrip(".") for k in SCENE_FIELDS}
    scene = {k: v for k, v in scene.items() if v and v.lower() not in ("none", "n/a")}
    log.info("сцена: %s", json.dumps(scene, ensure_ascii=False))
    _cache[key] = scene
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return scene


def scene_text(scene: dict, keys: tuple[str, ...] = ("light", "atmosphere", "colour_grade", "camera", "mood")) -> str:
    """Описание сцены одной фразой для запроса модели: «Light: …. Atmosphere: …»."""
    names = {"light": "Light", "atmosphere": "Atmosphere", "colour_grade": "Colour grading", "camera": "Camera",
             "style": "Style", "mood": "Mood"}
    return " ".join(f"{names[k]}: {scene[k]}." for k in keys if scene.get(k))
