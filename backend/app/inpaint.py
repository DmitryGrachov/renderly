"""Inpainting на FLUX.2 [klein]: удаление объекта по маске.

Как работает удаление:
  1. кроп с контекстом вокруг маски (маска чуть расширяется — захватить край объекта и контактную тень),
     кроп приводится к ~1 Мп: мелкий объект на большом кадре получает полное разрешение модели;
  2. DeepSeek видит кроп с подсвеченной областью и называет, что удалить и что должно оказаться на его месте
     (продолжение окружающих поверхностей) — klein обучен на правках вида «remove the car»;
  3. кроп подаётся как ReferenceLatent (image 1), шум — только под маской (SetLatentNoiseMask, полный denoise,
     Differential Diffusion по мягкому краю): вне маски кроп не меняется;
  4. результат вклеивается в исходник по растушёванной маске — остальной кадр остаётся пиксель в пиксель.

Тень: маска протягивается вниз (SHADOW_DOWN) под контактную тень, заметную тень за обводкой DeepSeek отдаёт рамкой.
"""

import hashlib
import io
import json
import random
import uuid
from collections import OrderedDict

import httpx
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .comfy import TOTAL_STEPS, ComfyError, _check_model, _run, _sampling, _upload, fit_size, to_png
from .config import COMFYUI_URL, DEFAULT_MODEL, FLUX_MODELS, FLUX_VAE, INPAINT_MEGAPIXELS, log
from .prompting import DEEPSEEK_API_KEY, _box, _chat, _data_url

CONTEXT = 1.0  # контекст вокруг маски: столько же от её размера с каждой стороны
MIN_CROP = 512  # минимальная сторона кропа в пикселях исходника — модели нужен контекст
MAX_ASPECT = 2.0  # кроп не уже 1:2 — иначе у модели мало контекста по короткой стороне
MAX_SHADOW = 0.35  # рамка тени от DeepSeek — не больше этой доли кропа
SHADOW_DOWN = 0.08  # маска протягивается вниз на эту долю высоты объекта — под контактную тень

REMOVE_SYSTEM = (
    "You prepare an instruction for an image editing model that must erase an object from an architectural "
    "photo or render. The user highlighted the area to clear with a red overlay and outline (image 2); image 1 "
    "is the same crop without the highlight. Identify what is inside the highlighted area and what should be "
    "visible there once it is gone: the surfaces, materials and structures that continue from around it "
    "(e.g. 'the brick facade with its window grid', 'lawn grass and the paved path edge'). Describe only what "
    "is really visible around the area, do not invent patterns or details. If the highlight covers only part "
    "of an object, describe just the highlighted part. Look carefully for the shadow the object casts on the "
    "ground or a wall: if any part of it, even a faint soft shadow next to the feet, wheels or base, extends "
    "beyond the red outline, give the bounding box of the whole shadow in image 1 coordinates from 0 to 1000; "
    "if the object casts no shadow or it is fully inside the outline, null. English, concrete, no style words. Answer with JSON: "
    '{"object": "<3-12 words: what to remove>", '
    '"background": "<10-35 words: what is behind it and must fill the area>", '
    '"shadow": [x1, y1, x2, y2] or null}'
)

REMOVE_PROMPT = (
    "Remove {object} from the image completely, together with any shadow and reflection it casts. "
    "In its place show {background}, seamlessly continuing the surrounding surfaces, textures, patterns, lines "
    "and lighting with correct perspective, as if it had never been there. Do not add any new object, person, "
    "plant, sign or text. Everything else stays exactly as in image 1: same colours, light, sharpness and grain."
)
REMOVE_FALLBACK = {
    "object": "the object in the middle of the image",
    "background": "the surfaces and scenery that surround it",
}

_described: OrderedDict[str, dict] = OrderedDict()
CACHE_SIZE = 128


class EmptyMask(ValueError):
    pass


def _binary(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    m = mask.convert("L")
    if m.size != size:
        m = m.resize(size, Image.BILINEAR)
    return m.point(lambda v: 255 if v > 127 else 0)


def _grow(mask: Image.Image, px: int) -> Image.Image:
    """Расширение бинарной маски на ~px (размытие + низкий порог — быстрее MaxFilter на больших радиусах)."""
    if px < 1:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(px / 2)).point(lambda v: 255 if v > 6 else 0)


def _extend_down(mask: Image.Image, px: int, step: int) -> Image.Image:
    """Маска, протянутая вниз на px: контактная тень у ног, колёс и основания почти всегда ниже обводки."""
    out = mask
    for dy in range(step, px + step, step):
        shifted = Image.new("L", mask.size, 0)
        shifted.paste(mask, (0, min(dy, px)))
        out = ImageChops.lighter(out, shifted)
    return out


def _context_box(bbox: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    """Рамка кропа: маска + контекст, не меньше MIN_CROP, пропорции не круче MAX_ASPECT, внутри кадра."""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cw = min(w, max(MIN_CROP, round(bw * (1 + 2 * CONTEXT))))
    ch = min(h, max(MIN_CROP, round(bh * (1 + 2 * CONTEXT))))
    cw = min(w, max(cw, round(ch / MAX_ASPECT)))
    ch = min(h, max(ch, round(cw / MAX_ASPECT)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = round(min(max(0, cx - cw / 2), w - cw))
    top = round(min(max(0, cy - ch / 2), h - ch))
    return left, top, left + cw, top + ch


def _highlight(crop: Image.Image, mask: Image.Image) -> bytes:
    """Кроп с красной подсветкой и контуром выделенной области — для DeepSeek."""
    tint = Image.new("RGB", crop.size, (255, 40, 60))
    out = Image.composite(Image.blend(crop, tint, 0.45), crop, mask)
    edge = ImageChops.subtract(_grow(mask, max(2, min(crop.size) // 200)), mask)
    ImageDraw.Draw(out).bitmap((0, 0), edge, fill=(255, 0, 40))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


async def describe_removal(crop: bytes, highlighted: bytes) -> dict:
    """Что удалить и что должно быть на его месте (DeepSeek); при ошибке — общая формулировка."""
    if not DEEPSEEK_API_KEY:
        return REMOVE_FALLBACK
    key = hashlib.sha256(highlighted).hexdigest()
    if key in _described:
        _described.move_to_end(key)
        return _described[key]
    try:
        raw = await _chat(
            [
                {"role": "system", "content": REMOVE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image 1 (crop):"},
                        {"type": "image_url", "image_url": {"url": _data_url(crop, 1024)}},
                        {"type": "text", "text": "Image 2 (area to clear highlighted in red):"},
                        {"type": "image_url", "image_url": {"url": _data_url(highlighted, 1024)}},
                    ],
                },
            ],
            json_mode=True,
        )
        data = json.loads(raw)
        result = {
            k: " ".join(str(data.get(k) or "").split()).rstrip(".") or REMOVE_FALLBACK[k] for k in REMOVE_FALLBACK
        }
        # тень вне выделения — рамка в долях кропа; слишком большая рамка — скорее ошибка, чем тень
        shadow = _box(data.get("shadow"))
        if shadow and (shadow[2] - shadow[0]) * (shadow[3] - shadow[1]) < MAX_SHADOW:
            result["shadow"] = shadow
    except Exception as e:  # noqa: BLE001 — удаление не должно падать из-за LLM
        log.warning("DeepSeek недоступен, общая формулировка удаления: %s", e)
        return REMOVE_FALLBACK
    _described[key] = result
    if len(_described) > CACHE_SIZE:
        _described.popitem(last=False)
    return result


def build_remove_workflow(crop: str, mask: str, width: int, height: int, prompt: str, seed: int, model: str) -> dict:
    files = FLUX_MODELS[model]
    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": files["unet"], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": files["text_encoder"], "type": "flux2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "diffdiff": {"class_type": "DifferentialDiffusion", "inputs": {"model": ["unet", 0], "strength": 1.0}},
        "src_img": {"class_type": "LoadImage", "inputs": {"image": crop}},
        "mask": {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}},
        "src_lat": {"class_type": "VAEEncode", "inputs": {"pixels": ["src_img", 0], "vae": ["vae", 0]}},
        "noise_lat": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["src_lat", 0], "mask": ["mask", 0]}},
        "text": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        # image 1 — кроп как есть: модель видит объект и по инструкции убирает его, а не дорисовывает «дыру»
        "cond": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["text", 0], "latent": ["src_lat", 0]}},
    }
    image = _sampling(wf, "", ["cond", 0], ["noise_lat", 0], width, height, TOTAL_STEPS, seed, model=["diffdiff", 0])
    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": "inego/remove"}}
    return wf


async def run_remove(image: bytes, mask: bytes, seed: int | None = None, model: str = DEFAULT_MODEL) -> tuple[bytes, int]:
    """JPEG исходника в полном разрешении, где под маской объект удалён, и использованный сид."""
    if seed is None:
        seed = random.randrange(2**48)
    src = Image.open(io.BytesIO(image)).convert("RGB")
    w, h = src.size
    m = _binary(Image.open(io.BytesIO(mask)), src.size)
    bbox = m.getbbox()
    if not bbox:
        raise EmptyMask("Маска пуста — выделите объект")

    # запас по контуру: пользовательская маска приблизительна, край объекта и контактная тень — снаружи неё
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    grow = max(4, round(0.04 * max(bw, bh)), round(0.004 * max(w, h)))
    grown = _grow(_extend_down(m, round(SHADOW_DOWN * bh), grow), grow)
    box = _context_box(grown.getbbox(), w, h)
    crop, crop_mask = src.crop(box), grown.crop(box)
    cw, ch = crop.size
    gw, gh = fit_size(cw, ch, INPAINT_MEGAPIXELS)

    what = await describe_removal(to_png(crop), _highlight(crop, crop_mask))
    prompt = REMOVE_PROMPT.format(**what)
    if shadow := what.get("shadow"):
        # тень объекта за пределами выделения — эллипс по рамке DeepSeek, иначе останется «тень без хозяина»
        crop_mask = crop_mask.copy()
        ImageDraw.Draw(crop_mask).ellipse(
            (round(shadow[0] * cw), round(shadow[1] * ch), round(shadow[2] * cw), round(shadow[3] * ch)), fill=255
        )
    log.info(
        "remove %s %dx%d кроп %s → %dx%d seed=%d тень=%s\n%s", model, w, h, box, gw, gh, seed, what.get("shadow"), prompt
    )

    # мягкий край маски: Differential Diffusion плавно сводит новое со старым
    soft = crop_mask.resize((gw, gh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(max(2, min(gw, gh) // 160)))
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        await _check_model(client, model)
        crop_name = await _upload(client, to_png(crop.resize((gw, gh), Image.LANCZOS)), f"{uuid.uuid4().hex}.png")
        mask_name = await _upload(client, to_png(soft.convert("RGB")), f"{uuid.uuid4().hex}.png")
        out = await _run(client, build_remove_workflow(crop_name, mask_name, gw, gh, prompt, seed, model))
    if "save" not in out:
        raise ComfyError("ComfyUI не вернул результат")

    # вклейка: только расширенная маска с растушёвкой наружу — шов ложится на уже перерисованную кромку
    patch = Image.open(io.BytesIO(out["save"])).convert("RGB").resize((cw, ch), Image.LANCZOS)
    feather = max(2, grow // 2)
    alpha = _grow(crop_mask, feather).filter(ImageFilter.GaussianBlur(feather / 2))
    alpha = ImageChops.lighter(alpha, crop_mask)
    result = src.copy()
    result.paste(patch, box[:2], alpha)
    buf = io.BytesIO()
    result.save(buf, "JPEG", quality=95)
    return buf.getvalue(), seed
