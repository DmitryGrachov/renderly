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
    "You plan an edit of an architectural photo or render so that the new element looks native, as if it was "
    "photographed in the same shot. Image 1 is the whole frame with the user's area highlighted in red; image 2 "
    "is a close-up of that area. The user's request (it may be in Russian) says what must appear or change there. "
    "The red area marks WHERE, not how big: the element gets its true real-world size at that spot, even if that "
    "is much bigger or smaller than the highlight. Work it out from the whole frame: 1) find scale references near "
    "the spot (door about 2.1 m, storey about 3 m, person about 1.7 m, car about 4.5 m long and 1.4 m high, bench, "
    "bin, paving slabs, kerb) and the ground plane and perspective (horizon height, direction of nearby edges such "
    "as the kerb, facade lines); 2) estimate the element's real size and its size in this frame at that spot; "
    "3) give the bounding box of the whole element in image 1, standing on the ground at the highlighted spot "
    "(bottom of the box where it touches the ground) at true scale. If the request changes something already "
    "there (material, colour, species, style of the same thing), its shape and position stay. If it turns into a "
    "different kind of object (a person into a dog, a tree into a lamp post), the old thing disappears completely "
    "with its shadow and the new one gets its own true size standing at the same spot; then the instruction starts "
    "with what disappears. Measure "
    "the look of the whole frame: light direction and colour, shadow hardness and direction, ambient and sky "
    "colour, white balance, contrast, saturation, haze, depth of field, grain. Do not add anything the user did not "
    "ask for. English. Answer with JSON: "
    '{"instruction": "<30-80 words: what appears, its real size compared with named neighbours, orientation along '
    'the scene perspective (e.g. parallel to the kerb, receding to the right), materials and colours as they look '
    'under this light>", '
    '"location": "<5-20 words: where it stands, by landmarks of image 1>", '
    '"look": "<25-60 words: the measured light, shadows, colour grading, atmosphere and optics of the frame>", '
    '"kind": "add" if a new element appears, "change" if the same thing changes material, colour, species or '
    'style, "replace" if it becomes a different kind of object, '
    '"box": [x1, y1, x2, y2] in image 1, x as thousandths of its width and y as thousandths of its height}'
)
EDIT_PROMPT = (
    "Edit only {location}: {instruction}. Integrate it as if it was photographed in the same shot: {look}. "
    "Match the camera perspective, lens and horizon of image 1; light it from the same direction and with the "
    "same colour as everything around it, with contact shadows and cast shadows of the same direction and "
    "softness as the other objects, reflections of the surrounding scene on glossy surfaces, the same colour "
    "grading, haze, sharpness, depth of field and grain. It is shown whole at its real size, never cut off. "
    "Everything else stays exactly as in image 1."
)
# область правки подсвечена на image 1 — модель видит, где и какого размера рисовать, и не путает место по словам
MARK_RULE = (
    " Image 2 is image 1 with the area to edit marked by a translucent red overlay; it is only a guide to where the "
    "edit goes. The result is based on the clean image 1: no red tint, the ground and surroundings keep their own "
    "colours."
)
# с рамкой: модель держалась за маленькую заливку выделения и рисовала машину размером с собаку — размер говорим прямо
MARK_BOX_RULE = (
    " Image 2 is image 1 with a translucent red spot where the new element stands and a thin red outline of its full "
    "size; it is only a guide: the new element fills that outline from edge to edge, it is not squeezed into the "
    "small red spot. The result is based on the clean image 1: no red tint or outline anywhere, the ground and "
    "surroundings keep their own colours."
)
EDIT_MARK = True
EDIT_MARK_OPACITY = 0.45
EDIT_GROW = 0.05  # запас по контуру выделения пользователя
# запас вокруг размеченной области (доля её размера): объекту есть куда выйти за рамку, а не обрезаться о край маски
EDIT_ROOM = 0.08
# изменение того, что уже есть (материал, цвет, порода): форма и размер прежние — запас минимальный,
# иначе модель «разращивает» объект (сакура больше дерева, которое заменяла)
EDIT_ROOM_CHANGE = 0.03
# край маски правки (доля короткой стороны кропа): жёсткий — иначе по мягкому краю Differential Diffusion
# оставляет исходник, и край нового объекта растворяется в фоне
EDIT_EDGE = 1 / 480
REMOVE_EDGE = 1 / 160
# порог изменения (0..255 по сглаженной картинке): во вклейку идёт только то, что модель действительно поменяла
CHANGE_THRESHOLD = 22
MAX_BOX_AREA = 0.6  # рамка объекта от DeepSeek — не больше этой доли кадра

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
    """Перерисовка кропа только под маской. image 1 — сам кроп, image 2 (reference) — его копия с разметкой правки."""
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
    # image 1 — кроп как есть: при удалении модель видит объект и по инструкции убирает его, а не дорисовывает «дыру»
    wf["cond"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["text", 0], "latent": ["src_lat", 0]}}
    positive = ["cond", 0]
    if reference:
        # image 2 — тот же кроп с красной разметкой (где и какого размера), только подсказка: модель воспроизводит
        # чистый image 1, и разметка не «протекает» в результат красным контуром на плитке
        wf["ref_img"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        wf["ref_lat"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["ref_img", 0], "vae": ["vae", 0]}}
        wf["cond_ref"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive, "latent": ["ref_lat", 0]}}
        positive = ["cond_ref", 0]
    image = _sampling(wf, "", positive, ["noise_lat", 0], width, height, TOTAL_STEPS, seed, model=["diffdiff", 0])
    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": f"inego/{prefix}"}}
    return wf


def _load(image: bytes, mask: bytes, grow_frac: float, shadow_down: float = 0.0) -> tuple[Image.Image, Image.Image, int]:
    """Исходник, расширенная маска пользователя (в размере исходника) и величина запаса в пикселях."""
    src = Image.open(io.BytesIO(image)).convert("RGB")
    w, h = src.size
    m = _binary(Image.open(io.BytesIO(mask)), src.size)
    bbox = m.getbbox()
    if not bbox:
        raise EmptyMask("Маска пуста — выделите область")
    # запас по контуру: пользовательская маска приблизительна
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    grow = max(4, round(grow_frac * max(bw, bh)), round(0.004 * max(w, h)))
    if shadow_down:
        m = _extend_down(m, round(shadow_down * bh), grow)
    return src, _grow(m, grow), grow


def _reconstruct(seed: "np.ndarray", allowed: "np.ndarray") -> "np.ndarray":
    """Часть allowed, связная с seed (морфологическая реконструкция)."""
    region = seed & allowed
    while True:
        grown = region.copy()
        grown[1:] |= region[:-1]
        grown[:-1] |= region[1:]
        grown[:, 1:] |= region[:, :-1]
        grown[:, :-1] |= region[:, 1:]
        grown &= allowed
        if (grown == region).all():
            return region
        region = grown


def _changed(orig: Image.Image, patch: Image.Image, seed: Image.Image, allowed: Image.Image) -> Image.Image:
    """Маска вклейки: размеченная область + всё, что модель заметно изменила и что с ней связано (без дыр).

    Так в результат попадает и часть объекта, вышедшая за разметку (капот машины, крона), а перерисованный
    запас без изменений остаётся пикселями исходника.
    """
    import numpy as np

    size = patch.size
    small = fit_size(*size, 0.25) if size[0] * size[1] > 250_000 else size
    blur = ImageFilter.GaussianBlur(1.5)
    a = np.asarray(orig.resize(small, Image.BILINEAR).filter(blur), dtype=np.int16)
    b = np.asarray(patch.resize(small, Image.BILINEAR).filter(blur), dtype=np.int16)
    diff = np.abs(a - b).max(-1) > CHANGE_THRESHOLD
    core = np.asarray(seed.resize(small, Image.BILINEAR)) > 127
    room = np.asarray(allowed.resize(small, Image.BILINEAR)) > 127
    region = _reconstruct(core, (diff & room) | core)
    # дыры внутри объекта (где его цвет совпал с фоном) — тоже объект
    border = np.zeros_like(region)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    outside = _reconstruct(border & ~region, ~region)
    region |= ~outside
    log.info("вклейка: разметка %.1f%%, с изменениями вокруг %.1f%% кропа", 100 * core.mean(), 100 * region.mean())
    return Image.fromarray((region * 255).astype(np.uint8)).resize(size, Image.BILINEAR).point(lambda v: 255 if v > 127 else 0)


class _Region:
    """Исходник, маска и кроп с контекстом вокруг неё.

    mask — что вклеивается всегда (и что подсвечивается для модели); с room модель перерисовывает ещё и запас
    вокруг, а во вклейку из запаса попадает только связное с маской изменение.
    """

    def __init__(self, src: Image.Image, mask: Image.Image, grow: int, extra: Image.Image | None = None, room: float = 0.0):
        w, h = src.size
        self.src, self.grow = src, grow
        mark = ImageChops.lighter(mask, extra) if extra is not None else mask
        gen = mark
        if room:
            x1, y1, x2, y2 = mark.getbbox()
            gen = _grow(mark, max(grow, round(room * max(x2 - x1, y2 - y1))))
        self.box = _context_box(gen.getbbox(), w, h)
        self.crop, self.mask, self.gen = src.crop(self.box), mark.crop(self.box), gen.crop(self.box)
        # выделение пользователя и рамка объекта по отдельности — для подсветки на image 1
        self.core = mask.crop(self.box)
        self.extra = extra.crop(self.box) if extra is not None else None
        self.room = bool(room)
        self.gen_size = fit_size(*self.crop.size, INPAINT_MEGAPIXELS)

    async def generate(
        self, prompt: str, seed: int, model: str, prefix: str, edge: float, reference: Image.Image | None = None
    ) -> bytes:
        """Перерисовка под маской в ComfyUI и вклейка в исходник: JPEG в полном разрешении."""
        gw, gh = self.gen_size
        # мягкий край маски: Differential Diffusion плавно сводит новое со старым
        soft = self.gen.resize((gw, gh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(max(1, round(min(gw, gh) * edge))))
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

        # вклейка: маска (а с запасом — и связные с ней изменения) с растушёвкой наружу — шов ложится на уже
        # перерисованную кромку
        patch = Image.open(io.BytesIO(out["save"])).convert("RGB").resize(self.crop.size, Image.LANCZOS)
        # целиком — только выделение пользователя: рамка объекта приблизительна и может захватить соседей
        # (собака рядом с новой машиной); из неё и запаса берутся только изменения, связанные с выделением
        keep = _changed(self.crop, patch, self.core, self.gen) if self.room else self.mask
        feather = max(2, self.grow // 2)
        alpha = ImageChops.lighter(_grow(keep, feather).filter(ImageFilter.GaussianBlur(feather / 2)), keep)
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
    src, core, grow = _load(image, mask, 0.04, SHADOW_DOWN)
    r = _Region(src, core, grow)
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


def _box_mask(box: list[float], src: Image.Image, core: Image.Image) -> Image.Image | None:
    """Рамка нового объекта в натуральную величину (размер — от DeepSeek, доли кадра).

    Размер DeepSeek оценивает по соседним объектам надёжно, а положение рамки на вертикальных кадрах путает
    (промах на сотни пикселей) — поэтому объект ставится сам: низом на нижний край выделения (точка касания
    земли), по центру выделения.
    """
    w, h = src.size
    bw, bh = round((box[2] - box[0]) * w), round((box[3] - box[1]) * h)
    if bw < 8 or bh < 8 or bw * bh > MAX_BOX_AREA * w * h:
        return None
    cx1, _, cx2, cy2 = core.getbbox()
    x1 = max(0, min(round((cx1 + cx2 - bw) / 2), w - bw))
    y1 = max(0, min(cy2 - bh, h - bh))
    m = Image.new("L", src.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((x1, y1, x1 + bw, y1 + bh), radius=round(0.12 * min(bw, bh)), fill=255)
    return m


def _mark(crop: Image.Image, core: Image.Image, extra: Image.Image | None) -> Image.Image:
    """image 1 правки: выделение — полупрозрачной заливкой, рамка объекта — только контуром.

    Заливка большой рамки «протекала» в результат (терракотовая плитка под машиной), контур такого не даёт,
    а размер модель по нему держит.
    """
    out = Image.open(io.BytesIO(_highlight(crop, core, EDIT_MARK_OPACITY))).convert("RGB")
    if extra is not None:
        width = max(2, min(crop.size) // 250)
        edge = ImageChops.subtract(_grow(extra, width), extra)
        ImageDraw.Draw(out).bitmap((0, 0), edge, fill=(255, 0, 40))
    return out


# слова-команды в начале запроса правки: задают тип явно, DeepSeek им не перечит
ADD_WORDS = ("добавь", "добавить", "поставь", "поставить", "посади", "посадить", "размести", "разместить",
             "нарисуй", "нарисовать", "установи", "установить", "add")
CHANGE_WORDS = ("измени", "изменить", "замени", "заменить", "поменяй", "поменять", "перекрась", "перекрасить",
                "сделай", "сделать", "change", "replace")
KIND_RULE = {
    "add": "The user explicitly asked to ADD a new element at the highlighted spot: kind is \"add\".",
    "change": (
        "The user asked to CHANGE what is already in the highlighted area: kind is \"change\" if the same thing only "
        "changes material, colour, species or style (its position, size and shape stay), or \"replace\" if it becomes "
        "a different kind of object (the old one disappears completely, the new one gets its own true size at the "
        "same spot)."
    ),
}


def parse_kind(request: str) -> str | None:
    """«добавь …» → add, «измени / замени / перекрась …» → change, иначе None (решает DeepSeek)."""
    words = request.strip().lower().replace("ё", "е").split(maxsplit=1)
    first = words[0].strip(",.:;!") if words else ""
    if first in ADD_WORDS:
        return "add"
    if first in CHANGE_WORDS:
        return "change"
    return None


async def describe_edit(frame: bytes, close: bytes, request: str, kind: str | None = None) -> dict:
    """План правки по всему кадру: что появится и какого реального размера, где, как выглядит кадр, рамка объекта.

    Без DeepSeek или при ошибке — запрос как есть, без рамки (рисуется в выделении).
    """
    fallback = {
        "instruction": request,
        "location": "in the marked area",
        "look": "the same light, shadows, colour grading and atmosphere as image 1",
        "kind": kind or "add",
    }
    if not DEEPSEEK_API_KEY:
        return fallback
    key = hashlib.sha256(frame + b"edit" + request.encode() + (kind or "").encode()).hexdigest()
    if key in _described:
        _described.move_to_end(key)
        return _described[key]
    messages = [
        {"role": "system", "content": EDIT_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"User request: «{request}»" + (f" {KIND_RULE[kind]}" if kind else "")},
                {"type": "text", "text": "Image 1 (whole frame, area highlighted in red):"},
                {"type": "image_url", "image_url": {"url": _data_url(frame, 1536)}},
                {"type": "text", "text": "Image 2 (close-up of the area):"},
                {"type": "image_url", "image_url": {"url": _data_url(close, 1024)}},
            ],
        },
    ]
    try:
        # рамка — это натуральный размер объекта; DeepSeek иногда её не присылает — тогда спрашиваем ещё раз,
        # иначе объект рисуется размером с выделение
        for attempt in range(2):
            data = json.loads(await _chat(messages, json_mode=True))
            if box := _box(data.get("box")):
                break
            log.warning("DeepSeek не дал рамку объекта (попытка %d): %s", attempt + 1, data.get("box"))
        result = {
            k: " ".join(str(data.get(k) or "").split()).rstrip(".") or fallback[k] for k in fallback if k != "kind"
        }
        if box:
            result["box"] = box
        said = str(data.get("kind", "")).strip().lower()
        # «добавь» — только добавление; «измени / замени» — изменение или замена, это решает DeepSeek
        if kind == "add":
            result["kind"] = "add"
        elif kind == "change":
            result["kind"] = "replace" if said == "replace" else "change"
        else:
            result["kind"] = said if said in ("change", "replace") else "add"
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
    """JPEG исходника в полном разрешении с тем, что попросил пользователь, и сид.

    Выделение задаёт место, а не размер: DeepSeek по всему кадру (масштаб соседних объектов, перспектива)
    даёт рамку объекта в натуральную величину; генерация идёт по выделению + рамке + запасу вокруг,
    во вклейку попадает всё, что изменилось и связано с ними, — объект не обрезается о край маски.
    """
    if seed is None:
        seed = random.randrange(2**48)
    src, core, grow = _load(image, mask, EDIT_GROW)
    close_box = _context_box(core.getbbox(), *src.size)
    kind = parse_kind(request)
    what = await describe_edit(
        _highlight(src, core), _highlight(src.crop(close_box), core.crop(close_box)), request.strip(), kind
    )
    # «изменить»: меняется то, что выделено, — рамка не нужна; «добавить» и «заменить» — объект в своём размере
    add = what.get("kind") != "change"
    extra = _box_mask(what["box"], src, core) if add and what.get("box") else None
    if add and what.get("box") and extra is None:
        log.warning("Рамка объекта отброшена (слишком мала или во весь кадр): %s", what["box"])
    r = _Region(src, core, grow, extra, EDIT_ROOM if add else EDIT_ROOM_CHANGE)
    prompt = EDIT_PROMPT.format(**what) + ((MARK_BOX_RULE if extra is not None else MARK_RULE) if EDIT_MARK else "")
    log.info(
        "edit %s %dx%d кроп %s → %dx%d seed=%d %s рамка=%s «%s»\n%s",
        model, *src.size, r.box, *r.gen_size, seed, what.get("kind"), what.get("box") if extra else None, request, prompt,
    )
    reference = _mark(r.crop, r.core, r.extra) if EDIT_MARK else None
    return await r.generate(prompt, seed, model, "edit", EDIT_EDGE, reference), seed
