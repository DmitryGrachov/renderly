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
    "is really visible around the area, do not invent patterns or details. If several objects are highlighted, "
    "name all of them. If the highlight covers only part of an object, describe just the highlighted part. Look carefully for the shadow the object casts on the "
    "ground or a wall: if any part of it, even a faint soft shadow next to the feet, wheels or base, extends "
    "beyond the red outline, give the bounding box of the whole shadow in image 1 coordinates from 0 to 1000; "
    "if the object casts no shadow or it is fully inside the outline, null. English, concrete, no style words. Answer with JSON: "
    '{"object": "<3-12 words: what to remove>", '
    '"background": "<10-35 words: what is behind it and must fill the area>", '
    '"keep": "<objects inside the highlight that must stay>" or null, '
    '"shadow": [x1, y1, x2, y2] or null}'
)
# подсказка пользователя: что именно удалить внутри выделения (может быть по-русски)
HINT_RULE = (
    "The user specified what to remove: «{hint}». Remove only that: every instance of it inside the highlight, "
    "translated into a precise English 'object'. Anything else inside the highlight that is not it (benches, "
    "trees, lamps, paving, facade parts...) stays and goes to 'keep'; 'background' describes only what is "
    "behind the removed thing. If the user names something that is not in the highlight, remove what is "
    "closest to their words."
)

REMOVE_PROMPT = (
    "Remove {object} from the image completely, together with any shadow and reflection it casts. "
    "In its place show {background}, seamlessly continuing the surrounding surfaces, textures, patterns, lines "
    "and lighting with correct perspective, as if it had never been there. Do not add any new object, person, "
    "plant, sign or text. Everything else stays exactly as in image 1: same colours, light, sharpness and grain."
)
KEEP_PROMPT = " Keep {keep} exactly as they are in image 1, only the removed thing disappears."
REMOVE_FALLBACK = {
    "object": "the object in the middle of the image",
    "background": "the surfaces and scenery that surround it",
}

EDIT_SYSTEM = (
    "You prepare an instruction for an image editing model that must change one area of an architectural photo "
    "or render. The user highlighted the area with a red overlay and outline (image 2); image 1 is the same crop "
    "without the highlight. The user's request (it may be in Russian) says what must appear or change there. "
    "Write in English what exactly that area shows after the edit: objects, materials, colours, their size "
    "relative to things around them, orientation following the scene perspective, lit from the same direction "
    "as the scene with matching shadows. If the request changes something already there (material, colour, "
    "type), say what it becomes and that its shape and position stay. Fit the request to the size of the area: "
    "the new thing fills it naturally, not a tiny object in the middle, and all of it, including its top (crown, "
    "foliage, lamp head, roof), stays inside the area: scale it down rather than let it be cut off. Do not add "
    "anything the user did not ask "
    "for. Answer with JSON: "
    '{"instruction": "<20-60 words: what the area shows after the edit>", '
    '"location": "<5-20 words: where the area is, by landmarks of image 1>", '
    '"scene": "<10-25 words: photo or render, light direction, time of day, style>"}'
)
EDIT_PROMPT = (
    "Edit only the area {location}: {instruction}. It must belong to the scene ({scene}): match the perspective, "
    "scale, light direction, shadow softness, colour grading, sharpness and grain of image 1, with natural "
    "contact shadows. Everything else stays exactly as in image 1."
)
# область правки подсвечена на image 1 — модель видит, где рисовать, и не путает место по словам
MARK_RULE = (
    " In image 1 the area to edit is marked with a translucent red overlay and outline; in the result it is "
    "completely replaced and no red tint or outline remains."
)
EDIT_MARK = True
EDIT_MARK_OPACITY = 0.45
EDIT_GROW = 0.05  # запас по контуру: объект у края маски не обрезается и не растворяется в фоне
# край маски правки (доля короткой стороны кропа): жёстче, чем у удаления — иначе по мягкому краю
# Differential Diffusion оставляет исходник, и низ нового объекта растворяется в мощении
EDIT_EDGE = 1 / 480
REMOVE_EDGE = 1 / 160

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


def _highlight(crop: Image.Image, mask: Image.Image, opacity: float = 0.45) -> bytes:
    """Кроп с красной подсветкой и контуром выделенной области — для DeepSeek и как image 1 правки."""
    tint = Image.new("RGB", crop.size, (255, 40, 60))
    out = Image.composite(Image.blend(crop, tint, opacity), crop, mask)
    edge = ImageChops.subtract(_grow(mask, max(2, min(crop.size) // 200)), mask)
    ImageDraw.Draw(out).bitmap((0, 0), edge, fill=(255, 0, 40))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


async def describe_removal(crop: bytes, highlighted: bytes, hint: str = "") -> dict:
    """Что удалить, что должно быть на его месте, что в выделении оставить и рамка тени вне маски (DeepSeek).

    hint — уточнение пользователя («люди», «машина»): удаляется только это, остальное в выделении остаётся.
    Без DeepSeek или при ошибке — общая формулировка (с подсказкой — она и есть объект).
    """
    fallback = REMOVE_FALLBACK | ({"object": hint} if hint else {})
    if not DEEPSEEK_API_KEY:
        return fallback
    key = hashlib.sha256(highlighted + hint.encode()).hexdigest()
    if key in _described:
        _described.move_to_end(key)
        return _described[key]
    try:
        raw = await _chat(
            [
                {"role": "system", "content": REMOVE_SYSTEM + (" " + HINT_RULE.format(hint=hint) if hint else "")},
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
            k: " ".join(str(data.get(k) or "").split()).rstrip(".") or fallback[k] for k in fallback
        }
        keep = " ".join(str(data.get("keep") or "").split()).rstrip(".")
        if hint and keep and keep.lower() not in ("null", "none", "nothing"):
            result["keep"] = keep
        # тень вне выделения — рамка в долях кропа; слишком большая рамка — скорее ошибка, чем тень
        shadow = _box(data.get("shadow"))
        if shadow and (shadow[2] - shadow[0]) * (shadow[3] - shadow[1]) < MAX_SHADOW:
            result["shadow"] = shadow
    except Exception as e:  # noqa: BLE001 — удаление не должно падать из-за LLM
        log.warning("DeepSeek недоступен, общая формулировка удаления: %s", e)
        return fallback
    _described[key] = result
    if len(_described) > CACHE_SIZE:
        _described.popitem(last=False)
    return result


def build_workflow(
    crop: str, mask: str, width: int, height: int, prompt: str, seed: int, model: str, prefix: str, reference: str | None = None
) -> dict:
    """Перерисовка кропа только под маской. image 1 — сам кроп или (reference) его копия с подсвеченной областью."""
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
        # вне маски латент остаётся исходным, шум — только под маской
        "noise_lat": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["src_lat", 0], "mask": ["mask", 0]}},
        "text": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
    }
    ref_lat = ["src_lat", 0]
    if reference:
        wf["ref_img"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        wf["ref_lat"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["ref_img", 0], "vae": ["vae", 0]}}
        ref_lat = ["ref_lat", 0]
    # удаление: image 1 — кроп как есть, модель видит объект и по инструкции убирает его, а не дорисовывает «дыру»
    wf["cond"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["text", 0], "latent": ref_lat}}
    image = _sampling(wf, "", ["cond", 0], ["noise_lat", 0], width, height, TOTAL_STEPS, seed, model=["diffdiff", 0])
    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": f"inego/{prefix}"}}
    return wf


class _Region:
    """Исходник, маска (расширенная) и кроп с контекстом вокруг неё."""

    def __init__(self, image: bytes, mask: bytes, grow_frac: float, shadow_down: float = 0.0):
        self.src = Image.open(io.BytesIO(image)).convert("RGB")
        w, h = self.src.size
        m = _binary(Image.open(io.BytesIO(mask)), self.src.size)
        bbox = m.getbbox()
        if not bbox:
            raise EmptyMask("Маска пуста — выделите область")
        # запас по контуру: пользовательская маска приблизительна
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        self.grow = max(4, round(grow_frac * max(bw, bh)), round(0.004 * max(w, h)))
        if shadow_down:
            m = _extend_down(m, round(shadow_down * bh), self.grow)
        grown = _grow(m, self.grow)
        self.box = _context_box(grown.getbbox(), w, h)
        self.crop, self.mask = self.src.crop(self.box), grown.crop(self.box)
        self.gen_size = fit_size(*self.crop.size, INPAINT_MEGAPIXELS)

    async def generate(
        self, prompt: str, seed: int, model: str, prefix: str, edge: float, reference: Image.Image | None = None
    ) -> bytes:
        """Перерисовка под маской в ComfyUI и вклейка в исходник: JPEG в полном разрешении."""
        gw, gh = self.gen_size
        # мягкий край маски: Differential Diffusion плавно сводит новое со старым
        soft = self.mask.resize((gw, gh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(max(1, round(min(gw, gh) * edge))))
        async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
            await _check_model(client, model)
            crop_name = await _upload(client, to_png(self.crop.resize((gw, gh), Image.LANCZOS)), f"{uuid.uuid4().hex}.png")
            mask_name = await _upload(client, to_png(soft.convert("RGB")), f"{uuid.uuid4().hex}.png")
            ref_name = None
            if reference is not None:
                ref_name = await _upload(client, to_png(reference.resize((gw, gh), Image.LANCZOS)), f"{uuid.uuid4().hex}.png")
            out = await _run(client, build_workflow(crop_name, mask_name, gw, gh, prompt, seed, model, prefix, ref_name))
        if "save" not in out:
            raise ComfyError("ComfyUI не вернул результат")

        # вклейка: только расширенная маска с растушёвкой наружу — шов ложится на уже перерисованную кромку
        patch = Image.open(io.BytesIO(out["save"])).convert("RGB").resize(self.crop.size, Image.LANCZOS)
        feather = max(2, self.grow // 2)
        alpha = ImageChops.lighter(_grow(self.mask, feather).filter(ImageFilter.GaussianBlur(feather / 2)), self.mask)
        result = self.src.copy()
        result.paste(patch, self.box[:2], alpha)
        buf = io.BytesIO()
        result.save(buf, "JPEG", quality=95)
        return buf.getvalue()


async def run_remove(
    image: bytes, mask: bytes, seed: int | None = None, model: str = DEFAULT_MODEL, hint: str | None = None
) -> tuple[bytes, int]:
    """JPEG исходника в полном разрешении, где под маской объект удалён, и использованный сид.

    hint — необязательное уточнение, что именно удалить в выделении; пусто — удаляется всё выделенное.
    """
    if seed is None:
        seed = random.randrange(2**48)
    # маска протягивается вниз — под контактную тень у ног, колёс и основания
    r = _Region(image, mask, 0.04, SHADOW_DOWN)
    cw, ch = r.crop.size

    what = await describe_removal(to_png(r.crop), _highlight(r.crop, r.mask), (hint or "").strip())
    prompt = REMOVE_PROMPT.format(**what) + (KEEP_PROMPT.format(keep=what["keep"]) if what.get("keep") else "")
    if shadow := what.get("shadow"):
        # тень объекта за пределами выделения — эллипс по рамке DeepSeek, иначе останется «тень без хозяина»
        r.mask = r.mask.copy()
        ImageDraw.Draw(r.mask).ellipse(
            (round(shadow[0] * cw), round(shadow[1] * ch), round(shadow[2] * cw), round(shadow[3] * ch)), fill=255
        )
    log.info(
        "remove %s %dx%d кроп %s → %dx%d seed=%d тень=%s\n%s",
        model, *r.src.size, r.box, *r.gen_size, seed, what.get("shadow"), prompt,
    )
    return await r.generate(prompt, seed, model, "remove", REMOVE_EDGE), seed


async def describe_edit(crop: bytes, highlighted: bytes, request: str) -> dict:
    """Инструкция правки на английском (что появится в области, где она, какая сцена); без DeepSeek — запрос как есть."""
    fallback = {"instruction": request, "location": "in the middle of the image", "scene": "the same scene as image 1"}
    if not DEEPSEEK_API_KEY:
        return fallback
    key = hashlib.sha256(highlighted + b"edit" + request.encode()).hexdigest()
    if key in _described:
        _described.move_to_end(key)
        return _described[key]
    try:
        raw = await _chat(
            [
                {"role": "system", "content": EDIT_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"User request: «{request}»"},
                        {"type": "text", "text": "Image 1 (crop):"},
                        {"type": "image_url", "image_url": {"url": _data_url(crop, 1024)}},
                        {"type": "text", "text": "Image 2 (area to edit highlighted in red):"},
                        {"type": "image_url", "image_url": {"url": _data_url(highlighted, 1024)}},
                    ],
                },
            ],
            json_mode=True,
        )
        data = json.loads(raw)
        result = {k: " ".join(str(data.get(k) or "").split()).rstrip(".") or fallback[k] for k in fallback}
    except Exception as e:  # noqa: BLE001 — правка не должна падать из-за LLM
        log.warning("DeepSeek недоступен, запрос правки как есть: %s", e)
        return fallback
    _described[key] = result
    if len(_described) > CACHE_SIZE:
        _described.popitem(last=False)
    return result


async def run_edit(
    image: bytes, mask: bytes, request: str, seed: int | None = None, model: str = DEFAULT_MODEL
) -> tuple[bytes, int]:
    """JPEG исходника в полном разрешении, где под маской нарисовано то, что попросил пользователь, и сид."""
    if seed is None:
        seed = random.randrange(2**48)
    r = _Region(image, mask, EDIT_GROW)
    highlighted = _highlight(r.crop, r.mask)
    what = await describe_edit(to_png(r.crop), highlighted, request.strip())
    prompt = EDIT_PROMPT.format(**what) + (MARK_RULE if EDIT_MARK else "")
    log.info("edit %s %dx%d кроп %s → %dx%d seed=%d «%s»\n%s", model, *r.src.size, r.box, *r.gen_size, seed, request, prompt)
    reference = Image.open(io.BytesIO(_highlight(r.crop, r.mask, EDIT_MARK_OPACITY))).convert("RGB") if EDIT_MARK else None
    return await r.generate(prompt, seed, model, "edit", EDIT_EDGE, reference), seed
