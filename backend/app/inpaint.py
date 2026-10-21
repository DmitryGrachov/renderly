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
from pathlib import Path
import io
import json
import random
import re
import uuid
from collections import OrderedDict

import httpx
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .comfy import TOTAL_STEPS, ComfyError, _check_model, _run, _sampling, _upload, to_png
from .comfy import fit_size as _fit16
from .config import (
    COMFYUI_URL, DEFAULT_MODEL, FLUX_MODELS, FLUX_VAE, INPAINT_MEGAPIXELS, QWEN21, QWEN21_BASE_STEPS,
    QWEN21_MEGAPIXELS, QWEN21_MODEL, QWEN21_MODELS, QWEN_EDIT,
    QWEN_EDIT_MODEL, log,
)
from .prompting import DEEPSEEK_API_KEY, REALISM_EDIT, REALISM_REMOVE, _box, _chat, _data_url, llm_model, q21_tags

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
    "and lighting with correct perspective, as if it had never been there; the area shows only that continuing "
    "background. Everything else stays exactly as in image 1: same colours, light, sharpness and grain."
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
    "with its shadow and the new one gets its own true size and its own natural pose (a dog on four legs, not "
    "the upright pose of the person it replaces) standing at the same spot; then the instruction starts "
    "with what disappears. Measure "
    "the look of the whole frame: light direction and colour, shadow hardness and direction, ambient and sky "
    "colour, white balance, contrast, saturation, haze, depth of field, grain. Do not add anything the user did not "
    "ask for. The red highlight exists only for you: never mention it, a red area, ellipse or outline in any "
    "field. English. Answer with JSON: "
    '{"instruction": "<30-80 words: what appears, its real size compared with named neighbours, orientation along '
    'the scene perspective (e.g. parallel to the kerb, receding to the right), materials and colours as they look '
    'under this light>", '
    '"location": "<5-20 words: where it stands, by landmarks of image 1; never mention the red highlight, '
    'ellipse, outline or any marking — the image model does not see it>", '
    '"look": "<25-60 words: the measured light, shadows, colour grading, atmosphere and optics of the frame: '
    'where the light comes from relative to the camera, its colour, which faces are lit and which are in shade, '
    'the colour of the shade>", '
    '"subject": "<3-12 words: the edited element as it is after the edit, e.g. the white plastered walls of the '
    'house, the red car by the kerb>", '
    '"light_on_subject": "<25-60 words: how the light of this frame falls on the edited element at its spot, judged '
    'by the real brightness of the surfaces there and of neighbours facing the same way: which of its faces or parts '
    'are in direct light and how much brighter they are, the slight warm or cool cast the light adds, which are in '
    'shade and how much darker, where it darkens (under the eaves, in corners, at the ground). Describe only light '
    'and brightness, never a colour of the element itself: its own colour stays as the user asked>", '
    '"kind": "add" if a new element appears, "change" if the same thing changes material, colour, species or '
    'style, "replace" if it becomes a different kind of object, '
    '"box": [x1, y1, x2, y2] in image 1, x as thousandths of its width and y as thousandths of its height}'
)
EDIT_PROMPT = (
    "Edit only {location}: {instruction}. Integrate it as if it was photographed in the same shot: {look}. "
    "Match the camera perspective, lens and horizon of image 1; light it from the same direction and with the "
    "same colour as everything around it, with contact shadows and cast shadows of the same direction and "
    "softness as the other objects, reflections of the surrounding scene on glossy surfaces, the same colour "
    "grading, haze, sharpness, depth of field and grain. It is shown whole at its real size. The light, contrast "
    "and clarity of the scene stay exactly as in image 1, and everything else stays exactly as in image 1."
)
# область правки подсвечена на image 1 — модель видит, где и какого размера рисовать, и не путает место по словам
MARK_RULE = (
    " Image 2 is image 1 with the area to edit marked by a translucent red overlay; it is only a guide to where the "
    "edit goes. The result is based on the clean image 1: no red tint, the ground and surroundings keep their own "
    "colours."
)
# с рамкой: модель держалась за маленькую заливку выделения и рисовала машину размером с собаку — размер говорим прямо
MARK_BOX_RULE = (
    " Image 2 is image 1 with a soft translucent red spot where the base of the new element stands. The spot marks "
    "the place, not the size: the element has its true real-world size from the instruction and may extend far "
    "beyond the spot. It is only a guide: do not draw any red frame, line, pole, ring or patch. The result is based "
    "on the clean image 1, the ground and surroundings keep their own colours."
)
# второй прогон «детализации» (добавить / заменить): объект крупным планом, частичный шум
# второй прогон (добавить / заменить): как в фотошопе — вырезать объект, перегенерировать крупно, вставить обратно
REFINE_PROMPT = (
    "Re-render the newly added element in image 1 as a sharp, complete, photorealistic close-up; it is: "
    "{instruction}. Keep its pose, silhouette, size, position, colours and lighting, and fix every defect: a "
    "missing, cut-off or blurry head or face, malformed or extra fingers and hands, missing or extra limbs, "
    "semi-transparent or ghostly parts, smeared or melted details, parts cut off at an edge. It must be whole and "
    "anatomically and physically correct, with natural detail (face, hands, hair, fabric folds, leaves, "
    "bark, materials) and the same sharpness, noise and grain as the surroundings, as an unretouched photograph: "
    "natural skin texture with visible pores, real micro-texture and subtle wear. Everything else stays exactly as "
    "in image 1."
)
REFINE_STEPS = 5  # из TOTAL_STEPS: 4 почти не меняет; 5 лечит полупрозрачность и шлейфы; на 6 уже меняется цвет одежды
# прогон «переосвещения»: свет правки сводится с кадром отдельной правкой «relight» (как в img2img, где свет
# рисуется заново). Частичный шум не годится: модель редактирования воспроизводит <image1> и на 6 шагах из 8
RELIGHT_PROMPT = (
    "Relight {subject} in image 1 so it looks photographed in the same shot as the rest of the scene: {look}. "
    "The material keeps its own colour, tone and darkness exactly as in image 1: on lit faces it is the same material "
    "only brighter with a slight cast of the light, in shade the same material only darker; the light never "
    "repaints it in another colour. "
    "Its parts facing the light take the brightness and colour tint of the light falling there, parts facing away or "
    "in shadow are darker and take the colour of the ambient and sky light, with soft darkening in corners, under "
    "overhangs and at contacts, cast shadows of the same direction and softness as elsewhere in the photo, "
    "reflections of the surroundings on glossy surfaces, and the same exposure, white balance, contrast, sharpness "
    "and texture as the surrounding photo. {subject_cap} keeps its shape, edges, position, material, own colour and "
    "details; only its lighting changes."
)
Q21_RELIGHT_MASK = (
    " The region to relight is marked white in the mask <image2>; everything outside it stays exactly as in "
    "<image1>; the mask is only a guide and leaves no trace."
)
RELIGHT_BAND = 0.03  # запас вокруг правки (доля её размера): свет ложится и на кромку, шва нет
REFINE_PAD = 0.15  # поле вокруг объекта (доля его размера) — контекст для света и резкости
# запас вверх для второго прогона (доля высоты объекта): если в первом не было головы, ей должно быть где появиться
REFINE_HEADROOM = 0.15
# Qwen-Image-2.1 — родной способ локальной правки (блог Qwen, сентябрь 2026): исходник и отдельная ч/б маска двумя
# входами (<image1> — холст, <image2> — маска: белое — где править), модель перерисовывает кадр целиком и сама
# сохраняет остальное — без заморозки латента по маске. Запрос — с действия, остальное — одной общей фразой
Q21_EDIT_PROMPT = (
    "Edit <image1> only inside the region marked white in the mask <image2> ({location}): {instruction}. "
    "{size_rule}It must look photographed in the same shot: {look}. Same camera perspective, scale, light direction "
    "and colour, contact and cast shadows, colour grading, sharpness and grain as <image1>. Everything outside the "
    "marked region stays exactly as in <image1>; the mask is only a guide and leaves no trace."
)
Q21_SIZE_ADD = (
    "The marked region shows where it stands and its full size: it fills the region at its real size, standing at "
    "the bottom of it, and is shown whole, never cut off. Only the new element and its own contact shadow are added: "
    "the ground, grass, paving, walls and everything else behind and around it inside the marked region stay exactly "
    "as in <image1>. "
)
Q21_REMOVE_PROMPT = (
    "Remove {object} from <image1> completely, together with any shadow and reflection it casts; it lies in the "
    "region marked white in the mask <image2>. Fill that region with {background}, seamlessly continuing the "
    "surrounding surfaces, textures, lines and lighting with correct perspective, as if it had never been there; "
    "the region shows only that continuing background. Everything outside the marked region stays exactly as in "
    "<image1>; the mask is only a guide and leaves no trace."
)
Q21_KEEP = " Keep {keep} exactly as they are."
# дописывается к запросу переписчика: он пишет «внутри белой области — только объект» и молчит о фоне, и 2.1
# рисовала объект на белом прямоугольнике маски (предметная съёмка на белом), а SAM брал прямоугольник за объект
Q21_MASK_TRACE = (
    "The mask <image2> only marks the area: its white is not a colour, backdrop or panel and leaves no trace, "
    "no white, blank or empty area appears in the result."
)
Q21_MASK_TRACE_ADD = (
    " Inside the marked area the original background of <image1> (ground, grass, paving, walls, windows, sky) "
    "stays and continues around the new element."
)
# Qwen-Image-Edit-2511 — родной способ: инструкция по Picture 1 + Picture 2 — тот же кадр с тонким красным контуром
# области правки (2511 понимает пометки и вспомогательные линии; отдельную ч/б маску её не учили понимать)
Q2511_EDIT_PROMPT = (
    "Edit Picture 1 only inside the area outlined in red in Picture 2 ({location}): {instruction}. "
    "{size_rule}It must look photographed in the same shot: {look}. Same camera perspective, scale, light direction "
    "and colour, contact and cast shadows, colour grading, sharpness and grain as Picture 1. Picture 2 is only a "
    "guide: the result is based on the clean Picture 1 and the red outline leaves no trace. Everything outside the "
    "outlined area stays exactly as in Picture 1."
)
Q2511_SIZE_ADD = (
    "The outlined area shows where it stands and its full size: it fills the area at its real size, standing at "
    "the bottom of it, and is shown whole, never cut off. "
)
Q2511_REMOVE_PROMPT = (
    "Remove {object} from Picture 1 completely, together with any shadow and reflection it casts; it is inside the "
    "area outlined in red in Picture 2. Fill that area with {background}, seamlessly continuing the surrounding "
    "surfaces, textures, lines and lighting with correct perspective, as if it had never been there; the area shows "
    "only that continuing background. Picture 2 is only a guide: the result is based on the clean Picture 1 and "
    "the red outline leaves no trace. Everything outside the outlined area stays exactly as in Picture 1."
)
# предел совмещения результата родного режима с кропом (доля кадра): сдвиг там — единицы пикселей
ALIGN_MAX = 0.03
EXACT_EDGE = 0.8  # сглаживание края вклейки по контуру поверхности (px): только антиалиасинг, без растушёвки
# «добавить / заменить»: объект сводится со сценой по свету через Marigold-IID (InegoHarmonize) вместо прогона
# переосвещения через модель. Выключено: на границе солнца и тени средний свет вокруг объекта — не тот, что падает
# на него (женщина на дорожке у газона вышла серой и холодной вместо освещённой закатом); прогон модели живее
ADD_HARMONIZE = False
EDIT_MARK = True
EDIT_MARK_OPACITY = 0.45
EDIT_GROW = 0.05  # запас по контуру выделения пользователя
# запас вокруг размеченной области (доля её размера): объекту есть куда выйти за рамку, а не обрезаться о край маски
EDIT_ROOM = 0.08
# запас вверх (доля высоты рамки объекта): головы, кроны и верхушки фонарей выходят выше оценённой рамки
EDIT_HEADROOM = 0.25
# изменение того, что уже есть (материал, цвет, порода): форма и размер прежние — запас минимальный,
# иначе модель «разращивает» объект (сакура больше дерева, которое заменяла)
EDIT_ROOM_CHANGE = 0.03
# край маски правки (доля короткой стороны кропа): жёсткий — иначе по мягкому краю Differential Diffusion
# оставляет исходник, и край нового объекта растворяется в фоне
EDIT_EDGE = 1 / 480
REMOVE_EDGE = 1 / 160
# порог изменения (0..255 по сглаженной картинке): во вклейку идёт только то, что модель действительно поменяла
CHANGE_THRESHOLD = 22
# «мостики» между клочками изменений (доля стороны кропа): крона берёзы, ветки, решётки остаются одним объектом
CHANGE_BRIDGE = 0.015
# цвет заливки разметки на подсказке (_highlight) и допуск «тонированного исходника» при зачистке
MARK_COLOR = (255, 40, 60)
TINT_RESID = 14
# белёсая вуаль за новым объектом: структура совпадает с исходником (корреляция), но светлее и бледнее
VEIL_NCC = 0.55
VEIL_LIFT = 4  # насколько светлее исходника (0..255)
VEIL_CONTRAST = 0.9  # контраст ниже этой доли исходного
RED_WORDS = re.compile(r"красн|бордо|алый|алая|алое|вишн|кирпич|терракот|ржав|\bred\b|crimson|burgundy|brick")
# якорная зона вокруг выделения (доля стороны рамки объекта): объект, начинающийся в ней, — тот, что просили
ANCHOR_REACH = 0.25
MAX_BOX_AREA = 0.6  # рамка объекта от DeepSeek — не больше этой доли кадра

_described: OrderedDict[str, dict] = OrderedDict()
CACHE_SIZE = 128


def fit_size(w: int, h: int, megapixels: float) -> tuple[int, int]:
    """Размер генерации с пропорцией исходника, ~megapixels, стороны кратны 32 (Qwen-Image-2.1 требует 32)."""
    gw, gh = _fit16(w, h, megapixels)
    return max(256, round(gw / 32) * 32), max(256, round(gh / 32) * 32)


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
    """Маска, протянутая вниз на px (вверх — при px < 0): контактная тень у ног, колёс и основания почти всегда
    ниже обводки; голова человека и крона дерева — выше рамки объекта."""
    out = mask
    sign = 1 if px >= 0 else -1
    for d in range(step, abs(px) + step, step):
        shifted = Image.new("L", mask.size, 0)
        shifted.paste(mask, (0, sign * min(d, abs(px))))
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
    """Кроп с красной подсветкой и контуром выделенной области — для DeepSeek (и подсказка при «изменить»)."""
    tint = Image.new("RGB", crop.size, MARK_COLOR)
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
    key = hashlib.sha256(highlighted + hint.encode() + llm_model.get().encode()).hexdigest()
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


def build_qwen_workflow(
    crop: str, mask: str, width: int, height: int, prompt: str, seed: int, prefix: str,
    reference: str | None = None, denoise: float = 1.0, native: bool = False,
) -> dict:
    """То же на Qwen-Image-Edit-2511: картинки подаются в TextEncodeQwenImageEditPlus (Picture 1 — кроп,
    Picture 2 — подсказка), Lightning LoRA — 4 шага, cfg 1.

    native — родной способ модели: латент кропа без заморозки по маске, модель перерисовывает кадр целиком по
    инструкции и сама сохраняет остальное (вклейка берёт только область правки). Иначе — шум только под маской
    (второй прогон с частичным шумом)."""
    q = QWEN_EDIT
    wf: dict = {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": q["unet"]}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": q["lora"], "strength_model": 1.0}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["lora", 0], "shift": q["shift"]}},
        "cfgnorm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "diffdiff": {"class_type": "DifferentialDiffusion", "inputs": {"model": ["cfgnorm", 0], "strength": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": q["text_encoder"], "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": q["vae"]}},
        "src_img": {"class_type": "LoadImage", "inputs": {"image": crop}},
        "mask": {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}},
        "src_lat": {"class_type": "VAEEncode", "inputs": {"pixels": ["src_img", 0], "vae": ["vae", 0]}},
        "noise_lat": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["src_lat", 0], "mask": ["mask", 0]}},
    }
    images = {"image1": ["src_img", 0]}
    if reference:
        wf["ref_img"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        images["image2"] = ["ref_img", 0]
    for tag, text in (("pos", prompt), ("neg", "")):
        wf[f"text_{tag}"] = {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"clip": ["clip", 0], "prompt": text, "vae": ["vae", 0], **images},
        }
        # GGUF не от Comfy-Org: способ подачи референсов задаём явно (как в шаблоне ComfyUI для 2511)
        wf[f"ref_{tag}"] = {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": [f"text_{tag}", 0], "reference_latents_method": "index_timestep_zero"},
        }
    wf["sample"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["cfgnorm" if native else "diffdiff", 0], "positive": ["ref_pos", 0], "negative": ["ref_neg", 0],
            "latent_image": ["src_lat" if native else "noise_lat", 0],
            "seed": seed, "steps": q["steps"], "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": denoise,
        },
    }
    wf["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": f"inego/{prefix}"}}
    if native:
        for k in ("diffdiff", "noise_lat", "mask"):  # заморозка латента по маске родному режиму не нужна
            wf.pop(k)
        _align(wf)
    return wf


def build_qwen21_workflow(
    crop: str, mask: str, prompt: str, seed: int, prefix: str, reference: str | None = None, denoise: float = 1.0,
    native: bool = False, base: bool = False, texture: str | None = None,
) -> dict:
    """То же на Qwen-Image-2.1: картинки — в TextEncodeQwenImage21 (<image1> — кроп, <image2> — подсказка),
    латент <image1> (кроп кратен 32 — размер сохраняется) с шумом только под маской. Turbo — 8 шагов по своему
    расписанию; base — базовая модель, QWEN21_BASE_STEPS шагов по simple (как в шаблонах ComfyUI).
    denoise < 1 — хвост расписания (второй прогон)."""
    q = QWEN21
    sig = q["sigmas"]
    if denoise < 1:
        sig = sig[len(sig) - 1 - max(1, round((len(sig) - 1) * denoise)):]
    # в 2.1 картинки в тексте адресуются как <image1>, <image2>
    text = re.sub(r"\bimage ([12])\b", r"<image\1>", prompt)
    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": q["unet_base" if base else "unet"], "weight_dtype": "default"}},
        # KV-кэш образов — в RAM: в 16 ГБ видеопамяти рядом с энкодером (~9 ГБ) и моделью (~7 ГБ) он не помещается (OOM)
        "cache": {"class_type": "QwenImage21Cache", "inputs": {"model": ["unet", 0], "device": "cpu", "dtype": "default"}},
        "diffdiff": {"class_type": "DifferentialDiffusion", "inputs": {"model": ["cache", 0], "strength": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": q["text_encoder"], "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": q["vae"]}},
        "src_img": {"class_type": "LoadImage", "inputs": {"image": crop}},
        "mask": {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}},
        "src_lat": {"class_type": "VAEEncode", "inputs": {"pixels": ["src_img", 0], "vae": ["vae", 0]}},
    }
    images = {"images.image_1": ["src_img", 0]}
    if native:
        # родной режим: <image2> — ч/б маска (белое — где править); латент не замораживается, модель перерисовывает
        # кадр целиком и сама сохраняет остальное, а вклейка берёт только область правки
        wf["mask_img"] = {"class_type": "LoadImage", "inputs": {"image": mask}}
        images["images.image_2"] = ["mask_img", 0]
        if texture:
            # <image3> — образец материала (вкладка «Материалы»)
            wf["tex_img"] = {"class_type": "LoadImage", "inputs": {"image": texture}}
            images["images.image_3"] = ["tex_img", 0]
    elif reference:
        wf["ref_img"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        images["images.image_2"] = ["ref_img", 0]
    wf["text"] = {
        "class_type": "TextEncodeQwenImage21",
        "inputs": {"clip": ["clip", 0], "prompt": text, "negative_prompt": "", "resolution": 0, "vae": ["vae", 0], **images},
    }
    wf |= {
        # латент самой картинки (VAE 2.1): латент энкодера картинку не несёт — с частичным шумом из него
        # выходила текстурная каша вместо объекта
        "noise_lat": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["src_lat", 0], "mask": ["mask", 0]}},
        "guider": {
            "class_type": "CFGGuider",
            "inputs": {"model": ["cache" if native else "diffdiff", 0], "positive": ["text", 0], "negative": ["text", 1], "cfg": 1.0},
        },
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": (
            {
                "class_type": "BasicScheduler",
                "inputs": {"model": ["cache", 0], "scheduler": "simple", "steps": QWEN21_BASE_STEPS, "denoise": denoise},
            }
            if base else {"class_type": "ManualSigmas", "inputs": {"sigmas": ", ".join(f"{v:g}" for v in sig)}}
        ),
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sample": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                # родной режим — латент <image1> без заморозки по маске
                "sigmas": ["sigmas", 0], "latent_image": ["text", 2] if native else ["noise_lat", 0],
            },
        },
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": f"inego/{prefix}"}},
    }
    if native:
        for k in ("diffdiff", "noise_lat", "mask", "src_lat"):  # заморозка латента по маске родному режиму не нужна
            wf.pop(k)
        _align(wf)
    return wf


def _align(wf: dict) -> None:
    """Родной режим перерисовывает кроп целиком, и модель чуть сдвигает и масштабирует кадр (на единицы пикселей):
    вклейка по маске расходилась с исходником на краю выделения — облицовка «съезжала» с угла и цоколя.
    InegoAlign (как в img2img) возвращает результат на место кропа до вклейки."""
    wf["align"] = {
        "class_type": "InegoAlign",
        "inputs": {"image": ["decode", 0], "source": ["src_img", 0], "max_scale": ALIGN_MAX, "max_shift": ALIGN_MAX},
    }
    wf["save"]["inputs"]["images"] = ["align", 0]


def build_workflow(
    crop: str, mask: str, width: int, height: int, prompt: str, seed: int, model: str, prefix: str,
    reference: str | None = None, steps: int = TOTAL_STEPS, native: bool = False, texture: str | None = None,
) -> dict:
    """Перерисовка кропа только под маской. image 1 — сам кроп, image 2 (reference) — его копия с разметкой правки."""
    if model == QWEN_EDIT_MODEL:
        return build_qwen_workflow(crop, mask, width, height, prompt, seed, prefix, reference, steps / TOTAL_STEPS, native)
    if model in QWEN21_MODELS:
        # прогон с маской второй картинкой (native) — всегда Turbo: базовая принимала белое маски буквально и рисовала
        # объект на белом прямоугольнике. Базовая (qwen-image-2.1) — во втором прогоне по шуму под объектом: фактура
        return build_qwen21_workflow(
            crop, mask, prompt, seed, prefix, reference, steps / TOTAL_STEPS, native, model == QWEN21_MODEL and not native,
            texture,
        )
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
    # steps < TOTAL_STEPS — хвост расписания: стартует с латента кропа (второй прогон «детализации»)
    image = _sampling(wf, "", positive, ["noise_lat", 0], width, height, steps, seed, model=["diffdiff", 0])
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


def _shrink(mask: Image.Image, px: int) -> Image.Image:
    """Сжатие бинарной маски на ~px (обратное _grow)."""
    return ImageChops.invert(_grow(ImageChops.invert(mask), px))


def _label(m: "np.ndarray") -> tuple["np.ndarray", int]:
    """Связные области бинарной маски (4-связность) — номера 1..n."""
    import numpy as np

    lab = np.zeros(m.shape, dtype=np.int32)
    n = 0
    rest = m.copy()
    while rest.any():
        n += 1
        y, x = np.argwhere(rest)[0]
        seed = np.zeros_like(rest)
        seed[y, x] = True
        comp = _reconstruct(seed, rest)
        lab[comp] = n
        rest &= ~comp
        if n > 200:  # обрывков слишком много — это не контур, а шум
            break
    return lab, n


def _dilate(m: "np.ndarray", r: int) -> "np.ndarray":
    """Расширение бинарной маски на r пикселей (8-связность)."""
    m = m.copy()
    for _ in range(r):
        g = m.copy()
        g[1:] |= m[:-1]
        g[:-1] |= m[1:]
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        g[1:, 1:] |= m[:-1, :-1]
        g[:-1, :-1] |= m[1:, 1:]
        g[1:, :-1] |= m[:-1, 1:]
        g[:-1, 1:] |= m[1:, :-1]
        m = g
    return m


def _changed(
    orig: Image.Image, patch: Image.Image, seed: Image.Image, allowed: Image.Image, anchor: Image.Image | None = None
) -> Image.Image:
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
    # ажурные объекты (крона берёзы, ветки, решётки) по пикселям распадаются на клочки, связанные тонкими
    # ветками или вовсе разорванные небом — без «мостиков» во вклейку попадал только ствол. Изменения
    # расширяются на CHANGE_BRIDGE размера, и связность считается уже по ним
    bridged = _dilate(diff & room, max(2, round(CHANGE_BRIDGE * max(small)))) & room
    # якорь: модель ставит объект чуть в стороне от маленького выделения (ствол берёзы в 80 px от кружка) —
    # объект «свой», если начинается в якорной зоне; пустой газон в ней не вклеивается, только изменения
    start = core
    if anchor is not None:
        start = core | (bridged & (np.asarray(anchor.resize(small, Image.BILINEAR)) > 127))
    region = _reconstruct(start, bridged | core)
    # дыры внутри объекта (где его цвет совпал с фоном) — тоже объект
    border = np.zeros_like(region)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    outside = _reconstruct(border & ~region, ~region)
    region |= ~outside
    log.info("вклейка: разметка %.1f%%, с изменениями вокруг %.1f%% кропа", 100 * core.mean(), 100 * region.mean())
    return Image.fromarray((region * 255).astype(np.uint8)).resize(size, Image.BILINEAR).point(lambda v: 255 if v > 127 else 0)


def _box_mean(a: "np.ndarray", r: int) -> "np.ndarray":
    """Среднее по окну (2r+1)² через кумулятивные суммы."""
    import numpy as np

    k = 2 * r + 1
    c = np.pad(np.pad(a, r, mode="edge").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _unleak(
    patch: Image.Image, crop: Image.Image, gen: Image.Image, core: Image.Image, extra: Image.Image | None
) -> Image.Image:
    """Убирает протёкшую разметку подсказки и не трогает красные объекты.

    Модель иногда перерисовывает подсказку в результат: заливку — тонированным красным фоном (диск на газоне),
    контур — тонкими красными линиями, часто со смещением (вокруг машины, на бордюре, штрихи на газоне).
      - тонированный фон: красный, а структура совпадает с исходником (та же трава, тот же фасад);
      - контур: тонкие линии насыщенного чисто-красного оттенка, не прилегающие к красному пятну.
    Красный объект (купол парашюта, футболка, машина) — новая структура и пятно, его полосы и швы к нему
    прилегают: не трогается. Кожа и терракота оранжевее и менее насыщены. Найденное заменяется исходным.
    """
    import numpy as np

    area = np.asarray(gen) > 127
    p = np.asarray(patch, dtype=np.int16)
    o = np.asarray(crop, dtype=np.int16)
    red_p = p[..., 0] - p[..., 1:].max(-1)
    red_o = o[..., 0] - o[..., 1:].max(-1)

    # 1) тонированный исходник: покраснело, а окна, швы, травинки на месте
    lo = np.asarray(crop.convert("L"), dtype=np.float32)
    lp = np.asarray(patch.convert("L"), dtype=np.float32)
    r = max(3, round(min(lo.shape) / 160))
    mo, mp = _box_mean(lo, r), _box_mean(lp, r)
    vo = np.maximum(_box_mean(lo * lo, r) - mo * mo, 0)
    vp = np.maximum(_box_mean(lp * lp, r) - mp * mp, 0)
    ncc = (_box_mean(lo * lp, r) - mo * mp) / np.sqrt(vo * vp + 1e-3)
    # и цвет — ровно смесь исходника с цветом разметки: p ≈ o + a·(MARK − o). Полупрозрачный оранжевый купол,
    # сквозь который видно фасад, по структуре тоже похож на исходник, но в такую смесь не укладывается
    mark = np.array(MARK_COLOR, dtype=np.float32)
    of, pf = o.astype(np.float32), p.astype(np.float32)
    d = mark - of
    alpha = ((pf - of) * d).sum(-1) / ((d * d).sum(-1) + 1e-3)
    resid = np.sqrt(((pf - (of + alpha[..., None] * d)) ** 2).mean(-1))
    resid = _box_mean(resid, 2)
    tinted = (alpha > 0.15) & (alpha < 0.9) & (resid < TINT_RESID)
    tint_leak = area & tinted & (red_p - red_o > 40) & (ncc > 0.5) & (np.sqrt(vo) > 3)

    # 2) тонкие линии разметки, не прилегающие к красному пятну
    hsv = np.asarray(patch.convert("HSV"), dtype=np.int16)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    pure_red = area & ((h < 9) | (h > 235)) & (sat > 140) & (val > 60) & (red_p - red_o > 40)
    objects = np.zeros_like(area)  # красные пятна — объекты (купол, футболка) с прилегающими швами и полосами
    line_leak = np.zeros_like(area)
    if pure_red.any():
        w = max(2, min(crop.size) // 250)
        blobs = _dilate(~_dilate(~pure_red, w), w)  # размыкание: пятна шире 2w
        lines = pure_red & ~blobs
        attached = _reconstruct(_dilate(blobs, w) & lines, lines) if blobs.any() else np.zeros_like(lines)
        line_leak = lines & ~attached
        objects = _dilate(blobs | attached, w)
    # у края выделения и контура рамки разметка перерисовывается и красно-оранжевым (смешана с травой, JPEG):
    # там допуск по оттенку шире, но только для длинных линий — обрывок руки или пальцы так не выглядят
    edge_w = max(3, round(0.03 * min(crop.size)))
    near = _grow(ImageChops.subtract(_grow(core, edge_w), _shrink(core, edge_w)), 1)
    if extra is not None:
        near = ImageChops.lighter(near, ImageChops.subtract(_grow(extra, edge_w), _shrink(extra, edge_w)))
    near = np.asarray(near) > 127
    warm = area & near & ((h <= 22) | (h > 235)) & (sat > 85) & (val > 60) & (red_p - red_o > 30) & ~objects
    if warm.any():
        long_lines = np.zeros_like(warm)
        lab, n = _label(warm)
        min_len = 0.06 * min(crop.size)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if max(np.ptp(xs), np.ptp(ys)) >= min_len:
                long_lines |= lab == i
        line_leak |= long_lines
    # края тонировки, где тон смешался с тенью и смесь сошлась хуже: красноватое, связанное с ней,
    # но не заходя в красные объекты (иначе зачистка выедала край купола)
    if tint_leak.any():
        reddish = area & (red_p - red_o > 25) & (ncc > 0.3) & ~objects
        tint_leak = _reconstruct(tint_leak & ~objects, tint_leak | reddish)
    leak = tint_leak | line_leak
    if leak.sum() < 20:
        return patch
    m = Image.fromarray((leak * 255).astype(np.uint8))
    m = _grow(m, 2).filter(ImageFilter.GaussianBlur(1.5))
    log.info("зачистка разметки: %d px", int(leak.sum()))
    return Image.composite(crop, patch, m)


def _unveil(patch: Image.Image, crop: Image.Image, gen: Image.Image) -> Image.Image:
    """Снимает белёсую вуаль, которой модель заливает фон за новым объектом (фасад за парашютистом).

    Вуаль — тот же фон, только светлее и бледнее: окна и швы на месте (структура совпадает с исходником,
    корреляция высокая), но яркость выше, а контраст ниже. Там возвращаются исходные пиксели. Новый объект
    по структуре на фон не похож — остаётся; тени темнят, а не высветляют, — тоже остаются.
    """
    import numpy as np

    o = np.asarray(crop.convert("L"), dtype=np.float32)
    p = np.asarray(patch.convert("L"), dtype=np.float32)
    r = max(3, round(min(o.shape) / 160))
    mo, mp = _box_mean(o, r), _box_mean(p, r)
    vo = np.maximum(_box_mean(o * o, r) - mo * mo, 0)
    vp = np.maximum(_box_mean(p * p, r) - mp * mp, 0)
    ncc = (_box_mean(o * p, r) - mo * mp) / np.sqrt(vo * vp + 1e-3)
    so, sp = np.sqrt(vo), np.sqrt(vp)
    veil = (ncc > VEIL_NCC) & (mp - mo > VEIL_LIFT) & (sp < VEIL_CONTRAST * so) & (so > 3)
    veil &= np.asarray(gen) > 127
    if veil.sum() < 0.002 * veil.size:
        return patch
    m = Image.fromarray((veil * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2))
    log.info("снята вуаль: %.1f%% кропа", 100 * veil.mean())
    return Image.composite(crop, patch, m)


async def _extract_object(
    client: httpx.AsyncClient, patch: Image.Image, crop: Image.Image, seed: Image.Image, allowed: Image.Image
) -> tuple[Image.Image, Image.Image] | None:
    """Силуэт нового объекта и его тень (ComfyUI, InegoObjectMask: изменения у выделения → рамка → SAM 2.1).
    None — объект не найден (остаётся вклейка по изменениям)."""
    names = []
    for img in (patch, crop, seed.convert("RGB"), allowed.convert("RGB")):
        names.append(await _upload(client, to_png(img), f"{uuid.uuid4().hex}.png"))
    wf = {
        "img": {"class_type": "LoadImage", "inputs": {"image": names[0]}},
        "org": {"class_type": "LoadImage", "inputs": {"image": names[1]}},
        "seed": {"class_type": "LoadImageMask", "inputs": {"image": names[2], "channel": "red"}},
        "allowed": {"class_type": "LoadImageMask", "inputs": {"image": names[3], "channel": "red"}},
        "obj": {
            "class_type": "InegoObjectMask",
            "inputs": {"image": ["img", 0], "original": ["org", 0], "seed": ["seed", 0], "allowed": ["allowed", 0],
                       "threshold": 0.09},
        },
        "obj_img": {"class_type": "MaskToImage", "inputs": {"mask": ["obj", 0]}},
        "sh_img": {"class_type": "MaskToImage", "inputs": {"mask": ["obj", 1]}},
        "save_obj": {"class_type": "SaveImage", "inputs": {"images": ["obj_img", 0], "filename_prefix": "inego/objmask"}},
        "save_sh": {"class_type": "SaveImage", "inputs": {"images": ["sh_img", 0], "filename_prefix": "inego/objshadow"}},
    }
    try:
        out = await _run(client, wf)
    except ComfyError as e:
        log.warning("Маска объекта не получена: %s", e)
        return None
    if "save_obj" not in out:
        return None
    obj = Image.open(io.BytesIO(out["save_obj"])).convert("L").resize(crop.size, Image.BILINEAR)
    shadow = Image.open(io.BytesIO(out["save_sh"])).convert("L").resize(crop.size, Image.BILINEAR)
    obj = obj.point(lambda v: 255 if v > 127 else 0)
    if not obj.getbbox() or sum(obj.histogram()[255:]) < 0.0005 * crop.size[0] * crop.size[1]:
        return None
    return obj, shadow


def _compose_object(patch: Image.Image, crop: Image.Image, obj: Image.Image, shadow: Image.Image, edge: int) -> Image.Image:
    """Кроп исходника + объект из генерации + тень.

    Тень — мягкое затемнение: отношение сглаженных яркостей генерации и исходника (только темнее, без текстуры
    генерации), с весом по области тени, размытой на ~4% размера объекта, — края нет. Бинарная маска тени с
    резким краем давала видимое пятно на плитке. Свою тень не дорисовываем — только та, что нарисовала модель.
    """
    import numpy as np
    from scipy import ndimage

    o = np.asarray(crop, dtype=np.float32)
    g = np.asarray(patch, dtype=np.float32)
    m = np.asarray(obj) > 127
    sh = np.asarray(shadow) > 127
    ys, xs = np.nonzero(m)
    size = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)

    s_lum = max(2.0, size * 0.012)
    ratio = np.clip(
        (ndimage.gaussian_filter(g.mean(-1), s_lum) + 1) / (ndimage.gaussian_filter(o.mean(-1), s_lum) + 1), 0.25, 1.0
    )
    wgt = np.clip(ndimage.gaussian_filter(sh.astype(np.float32), max(3.0, size * 0.04)) * 1.6, 0, 1)
    dark = 1 - wgt * (1 - ratio)

    out = o * dark[..., None]
    # край — с чуть сжатой маски: пиксели фона из генерации вокруг контура (тёмный ореол) не попадают в результат
    a = ndimage.gaussian_filter(ndimage.binary_erosion(m, iterations=1).astype(np.float32), max(0.8, size * 0.0015))
    out = out * (1 - a[..., None]) + g * a[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


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
            pad = max(grow, round(room * max(x2 - x1, y2 - y1)))
            if extra is not None:
                # верх высоких объектов (голова, крона, фонарь) уходит выше рамки DeepSeek — запас вверх больше,
                # иначе голову срезает край области генерации, и вместо неё остаётся фон («призрак»)
                ex1, ey1, ex2, ey2 = extra.getbbox()
                gen = _extend_down(gen, -round(EDIT_HEADROOM * (ey2 - ey1)), pad)
            gen = _grow(gen, pad)
        self.box = _context_box(gen.getbbox(), w, h)
        self.crop, self.mask, self.gen = src.crop(self.box), mark.crop(self.box), gen.crop(self.box)
        # выделение пользователя и рамка объекта по отдельности — для подсветки на image 1
        self.core = mask.crop(self.box)
        self.extra = extra.crop(self.box) if extra is not None else None
        # якорная зона: выделение + четверть размера рамки — объект, начинающийся здесь, вклеивается целиком
        self.anchor = None
        if extra is not None:
            x1, y1, x2, y2 = extra.getbbox()
            self.anchor = _grow(mask, max(grow, round(ANCHOR_REACH * max(x2 - x1, y2 - y1)))).crop(self.box)
        self.room = bool(room)
        # смена материала: маска — точный контур поверхности (_surface), вклейка строго по нему, без запаса и
        # растушёвки наружу — размеры и примыкания остаются от исходника
        self.exact = False
        # «изменить» и материалы: свет правки — с исходника через Marigold-IID (_relight_source), а не прогоном Qwen
        self.source_light = False
        self.gen_size = fit_size(*self.crop.size, INPAINT_MEGAPIXELS)

    async def generate(
        self, prompt: str, seed: int, model: str, prefix: str, edge: float, reference: Image.Image | None = None,
        unleak: bool = False, unveil: bool = False, refine: str | None = None, native: bool = False,
        relight: str | None = None, texture: Image.Image | None = None,
    ) -> bytes:
        """Перерисовка под маской в ComfyUI и вклейка в исходник: JPEG в полном разрешении.

        relight — запрос прогона переосвещения (_relight): свет вклеенной правки сводится с кадром.

        native (Qwen-Image-2.1): маска уходит модели второй картинкой — чёткая ч/б область правки (self.mask:
        выделение + рамка объекта), а не мягкая маска шума.
        """
        gw, gh = self.gen_size
        if native and model == QWEN_EDIT_MODEL:
            # 2511: Picture 2 — кроп с контуром области; маска модели не уходит, она нужна только вклейке
            reference = _outline(self.crop, self.mask)
            soft = self.mask.resize((gw, gh), Image.BILINEAR)
        elif native:
            soft = self.mask.resize((gw, gh), Image.BILINEAR).point(lambda v: 255 if v > 127 else 0)
        else:
            # мягкий край маски: Differential Diffusion плавно сводит новое со старым
            soft = self.gen.resize((gw, gh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(max(1, round(min(gw, gh) * edge))))
        async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
            await _check_model(client, model)
            crop_name = await _upload(client, to_png(self.crop.resize((gw, gh), Image.LANCZOS)), f"{uuid.uuid4().hex}.png")
            mask_name = await _upload(client, to_png(soft.convert("RGB")), f"{uuid.uuid4().hex}.png")
            ref_name = None
            if reference is not None:
                ref_name = await _upload(client, to_png(reference.resize((gw, gh), Image.LANCZOS)), f"{uuid.uuid4().hex}.png")
            tex_name = None
            if texture is not None:
                # образец материала — своим размером (кратен 32), третьей картинкой
                tex_name = await _upload(client, to_png(texture), f"{uuid.uuid4().hex}.png")
            out = await _run(
                client,
                build_workflow(crop_name, mask_name, gw, gh, prompt, seed, model, prefix, ref_name, native=native, texture=tex_name),
            )
        if "save" not in out:
            raise ComfyError("ComfyUI не вернул результат")

        # вклейка: маска (а с запасом — и связные с ней изменения) с растушёвкой наружу — шов ложится на уже
        # перерисованную кромку
        patch = Image.open(io.BytesIO(out["save"])).convert("RGB").resize(self.crop.size, Image.LANCZOS)
        if unleak:
            patch = _unleak(patch, self.crop, self.gen, self.core, self.extra)
        feather = max(2, self.grow // 2)
        found = None
        if self.extra is not None:
            # добавить / заменить: вклеиваются только силуэт объекта (SAM 2.1) и его тень затемнением
            async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
                found = await _extract_object(client, patch, self.crop, self.core, self.gen)
        # вуаль снимается только без силуэта: с ним фон и так из исходника, а светлый объект над тёмным фоном
        # (тент зонта над окном) _unveil принимал за вуаль и вырезал в нём дыры с окном
        if unveil and not found:
            patch = _unveil(patch, self.crop, self.gen)
        harmonized = None
        if found:
            obj, shadow = found
            if ADD_HARMONIZE:
                harmonized = await _harmonize(patch, self.crop, obj, shadow)
            patch = harmonized or _compose_object(patch, self.crop, obj, shadow, max(1, feather // 2))
            keep = obj
            # _compose_object уже свёл объект и тень с исходником — вклеивается весь кроп
            alpha = Image.new("L", self.crop.size, 255)
        else:
            # целиком — только выделение пользователя: рамка объекта приблизительна и может захватить соседей
            # (собака рядом с новой машиной); из неё и запаса берутся только изменения, связанные с выделением
            if self.exact:
                keep = self.mask
                alpha = keep.filter(ImageFilter.GaussianBlur(EXACT_EDGE))
            else:
                keep = _changed(self.crop, patch, self.core, self.gen, self.anchor) if self.room else self.mask
                alpha = ImageChops.lighter(_grow(keep, feather).filter(ImageFilter.GaussianBlur(feather / 2)), keep)
        result = self.src.copy()
        result.paste(patch, self.box[:2], alpha)
        obj = Image.new("L", self.src.size, 0)
        obj.paste(keep, self.box[:2])
        if self.source_light:
            result = await _relight_source(result, self.src, obj, self.exact)
        elif relight and harmonized is None:
            # до второго прогона: он дорисовывает детали объекта и держит его цвета и свет — уже сведённые
            result = await _relight(result, obj, relight, seed + 2, model, self.exact)
        if refine and self.room:
            result = await _refine(result, obj, refine, seed + 1, model, self.gen_size[0] / self.crop.size[0])
        buf = io.BytesIO()
        result.save(buf, "JPEG", quality=95)
        return buf.getvalue()


async def _refine(result: Image.Image, obj: Image.Image, prompt: str, seed: int, model: str, scale1: float) -> Image.Image:
    """Второй прогон: как в фотошопе — объект вырезается, перегенерируется крупно отдельно и вставляется обратно.

    Первый прогон вписывает объект в сцену (место, размер, перспектива, свет), но видит его в кропе с контекстом:
    мелкому объекту достаётся мало пикселей, отсюда каша вместо лица, кривые руки, «призраки». Здесь кроп — только
    объект с полем, растянутый до ~1 Мп; модель перерисовывает его на REFINE_STEPS из TOTAL_STEPS (силуэт, поза,
    место и свет — от первого прогона) с явным списком дефектов. Лечит то, что в объекте уже есть: размытое лицо
    и руки, полупрозрачные части, смазанные шлейфы. Недостающую часть (голову) не дорисует: образец — результат
    первого прогона, и модель редактирования его воспроизводит. Вклеивается то, что перерисовано и связано с объектом.
    """
    bbox = obj.getbbox()
    if not bbox:
        return result
    w, h = result.size
    x1, y1, x2, y2 = bbox
    # модель второго прогона может перерисовать объект чуть крупнее и выше: если он упёрся в край кропа (не кадра),
    # прогон повторяется с удвоенным запасом, а если и тогда — остаётся целый объект первого прогона
    # (раньше вклеивался срезанный ровной линией верх тента зонта)
    for scale in (1, 2):
        pad = max(16, round(scale * REFINE_PAD * max(x2 - x1, y2 - y1)))
        box = (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))
        bw, bh = box[2] - box[0], box[3] - box[1]
        gw, gh = fit_size(bw, bh, INPAINT_MEGAPIXELS)
        crop, mask = result.crop(box), obj.crop(box)
        grow = max(2, round(0.02 * max(bw, bh)))
        room = _grow(_extend_down(mask, -round(scale * REFINE_HEADROOM * (y2 - y1)), grow), grow)
        soft = room.resize((gw, gh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(max(1, min(gw, gh) // 200)))
        log.info(
            "второй прогон: объект %dx%d → %dx%d (×%.1f пикселей к первому), шагов %d из %d, запас ×%d",
            bw, bh, gw, gh, (gw / bw) / scale1, REFINE_STEPS, TOTAL_STEPS, scale,
        )
        async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
            crop_name = await _upload(client, to_png(crop.resize((gw, gh), Image.LANCZOS)), f"{uuid.uuid4().hex}.png")
            mask_name = await _upload(client, to_png(soft.convert("RGB")), f"{uuid.uuid4().hex}.png")
            out = await _run(
                client, build_workflow(crop_name, mask_name, gw, gh, prompt, seed, model, "refine", steps=REFINE_STEPS)
            )
        if "save" not in out:
            return result
        refined = Image.open(io.BytesIO(out["save"])).convert("RGB").resize((bw, bh), Image.LANCZOS)
        # объект целиком + то, что перерисовано и связано с ним (дорисованная голова), — фон запаса остаётся прежним
        keep = _changed(crop, refined, mask, room)
        cut = _touches_edge(keep, box, (w, h))
        if not cut:
            break
        log.info("второй прогон: объект упёрся в край кропа (%s)%s", cut, " — повтор с большим запасом" if scale == 1 else
                 " — оставляю первый прогон")
    else:
        return result
    alpha = ImageChops.lighter(_grow(keep, grow).filter(ImageFilter.GaussianBlur(max(1, grow / 2))), keep)
    result = result.copy()
    result.paste(refined, box[:2], alpha)
    return result


async def _harmonize(patch: Image.Image, crop: Image.Image, obj: Image.Image, shadow: Image.Image) -> Image.Image | None:
    """Добавленный объект, сведённый со сценой по свету (InegoHarmonize: Marigold-IID) — кроп исходника + тень по
    падению освещённости + объект с оттенком и яркостью света сцены. None — не получилось (сборка _compose_object)."""
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        names = [
            await _upload(client, to_png(img), f"{uuid.uuid4().hex}.png")
            for img in (patch, crop, obj.convert("RGB"), shadow.convert("RGB"))
        ]
        wf = {
            "gen": {"class_type": "LoadImage", "inputs": {"image": names[0]}},
            "src": {"class_type": "LoadImage", "inputs": {"image": names[1]}},
            "obj": {"class_type": "LoadImageMask", "inputs": {"image": names[2], "channel": "red"}},
            "sh": {"class_type": "LoadImageMask", "inputs": {"image": names[3], "channel": "red"}},
            "harm": {
                "class_type": "InegoHarmonize",
                "inputs": {"generated": ["gen", 0], "source": ["src", 0], "object": ["obj", 0], "shadow": ["sh", 0]},
            },
            "save": {"class_type": "SaveImage", "inputs": {"images": ["harm", 0], "filename_prefix": "inego/harmonize"}},
        }
        try:
            out = await _run(client, wf)
        except ComfyError as e:
            log.warning("Объект не сведён со сценой по свету: %s", e)
            return None
    if "save" not in out:
        return None
    return Image.open(io.BytesIO(out["save"])).convert("RGB").resize(crop.size, Image.LANCZOS)


async def _relight_source(result: Image.Image, src: Image.Image, area: Image.Image, exact: bool) -> Image.Image:
    """Свет исходника на изменённой поверхности (InegoRelight: Marigold-IID) — для «изменить» и материалов.

    Поверхность та же, значит и свет на ней тот же, что на фото: освещённость результата заменяется освещённостью
    исходника, фактура генерации остаётся. Не получилось — результат как есть (без переосвещения)."""
    bbox = area.getbbox()
    if not bbox:
        return result
    box = _context_box(bbox, *result.size)
    crop, before, m = result.crop(box), src.crop(box), area.crop(box)
    if not exact:
        # «изменить» без точного контура: свет — и на узкую кромку вокруг, шва нет
        m = _grow(m, max(2, round(RELIGHT_BAND * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))))
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        names = [await _upload(client, to_png(img), f"{uuid.uuid4().hex}.png") for img in (crop, before, m.convert("RGB"))]
        wf = {
            "img": {"class_type": "LoadImage", "inputs": {"image": names[0]}},
            "src": {"class_type": "LoadImage", "inputs": {"image": names[1]}},
            "mask": {"class_type": "LoadImageMask", "inputs": {"image": names[2], "channel": "red"}},
            "relight": {"class_type": "InegoRelight", "inputs": {"image": ["img", 0], "source": ["src", 0], "mask": ["mask", 0]}},
            "save": {"class_type": "SaveImage", "inputs": {"images": ["relight", 0], "filename_prefix": "inego/srclight"}},
        }
        try:
            out = await _run(client, wf)
        except ComfyError as e:
            log.warning("Свет исходника не перенесён: %s", e)
            return result
    if "save" not in out:
        return result
    patch = Image.open(io.BytesIO(out["save"])).convert("RGB").resize(crop.size, Image.LANCZOS)
    result = result.copy()
    result.paste(patch, box[:2], m.filter(ImageFilter.GaussianBlur(EXACT_EDGE)))
    log.info("свет исходника на изменённой поверхности: кроп %s", box)
    return result


async def _relight(
    result: Image.Image, area: Image.Image, prompt: str, seed: int, model: str, exact: bool = False
) -> Image.Image:
    """Прогон переосвещения: свет правки сводится с кадром.

    Правку модель рисует отдельно от кадра, и вклеенное выглядит наклейкой: ровно залитая белая стена без солнца и
    тени, объект в своём свете. В img2img так не бывает — свет рисуется заново по всему кадру. Здесь кроп с правкой
    уже вклеенной уходит модели ещё раз с правкой «переосвети это как остальной кадр» (свет кадра измерил DeepSeek):
    форма, материал и цвет остаются, солнце, тень, оттенок и экспозиция сходятся с окружением. На 2.1 — родной
    режим (маска вторым входом, Turbo), иначе — шум под маской. Вклеивается только правка с узкой кромкой.
    """
    bbox = area.getbbox()
    if not bbox:
        return result
    w, h = result.size
    box = _context_box(bbox, w, h)
    crop = result.crop(box)
    band = max(4, round(RELIGHT_BAND * max(bbox[2] - bbox[0], bbox[3] - bbox[1])))
    m = _grow(area.crop(box), band)
    native = model in QWEN21_MODELS
    if native:
        prompt = q21_tags(prompt) + Q21_RELIGHT_MASK
    gw, gh = fit_size(*crop.size, QWEN21_MEGAPIXELS if native else INPAINT_MEGAPIXELS)
    log.info("переосвещение %s: кроп %s → %dx%d\n%s", model, box, gw, gh, prompt)
    mask = m.resize((gw, gh), Image.BILINEAR)
    if not native:
        mask = mask.filter(ImageFilter.GaussianBlur(max(1, round(min(gw, gh) * EDIT_EDGE))))
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        crop_name = await _upload(client, to_png(crop.resize((gw, gh), Image.LANCZOS)), f"{uuid.uuid4().hex}.png")
        mask_name = await _upload(client, to_png(mask.convert("RGB")), f"{uuid.uuid4().hex}.png")
        out = await _run(client, build_workflow(crop_name, mask_name, gw, gh, prompt, seed, model, "relight", native=native))
    if "save" not in out:
        return result
    patch = Image.open(io.BytesIO(out["save"])).convert("RGB").resize(crop.size, Image.LANCZOS)
    result = result.copy()
    # exact (смена материала) — строго по контуру поверхности, кромка не задевает соседей
    paste = area.crop(box).filter(ImageFilter.GaussianBlur(EXACT_EDGE)) if exact else m.filter(ImageFilter.GaussianBlur(band / 2))
    result.paste(patch, box[:2], paste)
    return result


async def _surface(src: Image.Image, core: Image.Image) -> Image.Image:
    """Точный контур поверхности под мазком (InegoSurfaceMask: SAM 2.1 по исходнику) в размере исходника.
    Считается по кропу с контекстом вокруг мазка — в его разрешении, а не по уменьшенному кадру."""
    box = _context_box(core.getbbox(), *src.size)
    crop, stroke = src.crop(box), core.crop(box)
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        img_name = await _upload(client, to_png(crop), f"{uuid.uuid4().hex}.png")
        stroke_name = await _upload(client, to_png(stroke.convert("RGB")), f"{uuid.uuid4().hex}.png")
        wf = {
            "img": {"class_type": "LoadImage", "inputs": {"image": img_name}},
            "stroke": {"class_type": "LoadImageMask", "inputs": {"image": stroke_name, "channel": "red"}},
            "surface": {"class_type": "InegoSurfaceMask", "inputs": {"image": ["img", 0], "stroke": ["stroke", 0], "reach": 0.04}},
            "out": {"class_type": "MaskToImage", "inputs": {"mask": ["surface", 0]}},
            "save": {"class_type": "SaveImage", "inputs": {"images": ["out", 0], "filename_prefix": "inego/surface"}},
        }
        try:
            out = await _run(client, wf)
        except ComfyError as e:
            log.warning("Контур поверхности не получен, остаётся мазок: %s", e)
            return core
    if "save" not in out:
        return core
    m = Image.open(io.BytesIO(out["save"])).convert("L").resize(crop.size, Image.BILINEAR).point(lambda v: 255 if v > 127 else 0)
    if not m.getbbox():
        return core
    full = Image.new("L", src.size, 0)
    full.paste(m, box[:2])
    log.info("контур поверхности: мазок %d px → поверхность %d px", sum(core.histogram()[128:]), sum(full.histogram()[128:]))
    return full


TEXTURE_WIDTH_SYSTEM = (
    "You see a photo of a building material used as a repeating texture (bricks, stone, tiles, boards, plaster, "
    "concrete...). Estimate the real-world width in metres that the whole image covers, from the size of its "
    "elements: a standard brick is about 0.25 m long and 0.065 m high with 10 mm joints, a facade board 0.1-0.2 m "
    "wide, a paving slab 0.3-0.6 m, a large porcelain tile 0.6-1.2 m, aggregate stones in concrete 3-15 mm. Count "
    "the elements across the image. Answer with JSON: "
    '{"material": "<what it is, 3-10 words>", "elements_across": <number>, "width_m": <number>}'
)
TEXTURE_WIDTH_DEFAULT = 1.0  # без DeepSeek или при ошибке: метр — середина типичных образцов
TEXTURE_MAX_SIDE = 2048  # образец для проекции — в своём разрешении, но не больше


async def _texture_width(texture: bytes) -> float:
    """Ширина образца в метрах (сколько реального материала на картинке) — DeepSeek по одному образцу."""
    if not DEEPSEEK_API_KEY:
        return TEXTURE_WIDTH_DEFAULT
    key = hashlib.sha256(b"texwidth" + texture + llm_model.get().encode()).hexdigest()
    if key in _described:
        return _described[key]["width_m"]
    try:
        data = json.loads(await _chat(
            [
                {"role": "system", "content": TEXTURE_WIDTH_SYSTEM},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url(texture, 1024)}}]},
            ],
            json_mode=True,
        ))
        width = float(data.get("width_m") or 0)
    except Exception as e:  # noqa: BLE001
        log.warning("Ширина образца не оценена (DeepSeek): %s", e)
        return TEXTURE_WIDTH_DEFAULT
    if not 0.02 <= width <= 20:
        return TEXTURE_WIDTH_DEFAULT
    log.info("образец: %s, %s элементов поперёк → %.2f м", data.get("material"), data.get("elements_across"), width)
    _described[key] = {"width_m": width}
    return width


async def _project_texture(src: Image.Image, surface: Image.Image, texture: bytes, width_m: float) -> bytes | None:
    """Образец на поверхность проекцией 1 в 1 (InegoTextureProject: MoGe-2 + Marigold-IID) — JPEG кадра или None,
    если поверхность не плоская или геометрия не найдена. Кроп — с контекстом вокруг поверхности, в разрешении
    исходника: MoGe-2 нужен окружающий кадр для геометрии, проекции — все пиксели."""
    if not surface.getbbox():
        return None
    box = _context_box(surface.getbbox(), *src.size)
    crop, m = src.crop(box), surface.crop(box)
    tex = Image.open(io.BytesIO(texture)).convert("RGB")
    tex.thumbnail((TEXTURE_MAX_SIDE, TEXTURE_MAX_SIDE), Image.LANCZOS)
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        names = [
            await _upload(client, to_png(img), f"{uuid.uuid4().hex}.png") for img in (crop, m.convert("RGB"), tex)
        ]
        wf = {
            "img": {"class_type": "LoadImage", "inputs": {"image": names[0]}},
            "surf": {"class_type": "LoadImageMask", "inputs": {"image": names[1], "channel": "red"}},
            "tex": {"class_type": "LoadImage", "inputs": {"image": names[2]}},
            "proj": {
                "class_type": "InegoTextureProject",
                "inputs": {"image": ["img", 0], "surface": ["surf", 0], "texture": ["tex", 0], "tex_width_m": width_m},
            },
            "applied": {"class_type": "MaskToImage", "inputs": {"mask": ["proj", 1]}},
            "save": {"class_type": "SaveImage", "inputs": {"images": ["proj", 0], "filename_prefix": "inego/texture"}},
            "save_m": {"class_type": "SaveImage", "inputs": {"images": ["applied", 0], "filename_prefix": "inego/texmask"}},
        }
        try:
            out = await _run(client, wf)
        except ComfyError as e:
            log.warning("Проекция образца не удалась: %s", e)
            return None
    if "save" not in out or "save_m" not in out:
        return None
    applied = Image.open(io.BytesIO(out["save_m"])).convert("L")
    if not applied.point(lambda v: 255 if v > 127 else 0).getbbox():
        return None
    patch = Image.open(io.BytesIO(out["save"])).convert("RGB").resize(crop.size, Image.LANCZOS)
    log.info("образец спроецирован: кроп %s, ширина образца %.2f м", box, width_m)
    result = src.copy()
    # нода уже свела край со сглаживанием; вклейка — только по поверхности
    result.paste(patch, box[:2], m.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(EXACT_EDGE)))
    buf = io.BytesIO()
    result.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def _touches_edge(keep: Image.Image, box: tuple[int, int, int, int], size: tuple[int, int]) -> str:
    """Стороны кропа (не совпадающие с краем кадра), в которые упирается вклеиваемый объект: «верх, слева»; пусто — нет."""
    import numpy as np

    k = np.asarray(keep) > 127
    hh, ww = k.shape
    band = max(2, round(0.01 * max(hh, ww)))
    need = max(4, round(0.01 * min(hh, ww)))  # больше пары пикселей шва
    sides = []
    if box[1] > 0 and k[:band].any(0).sum() >= need:
        sides.append("верх")
    if box[0] > 0 and k[:, :band].any(1).sum() >= need:
        sides.append("слева")
    if box[2] < size[0] and k[:, -band:].any(1).sum() >= need:
        sides.append("справа")
    return ", ".join(sides)


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
    if model in QWEN21_MODELS:
        r.gen_size = fit_size(*r.crop.size, QWEN21_MEGAPIXELS)
    cw, ch = r.crop.size

    what = await describe_removal(to_png(r.crop), _highlight(r.crop, r.mask), (hint or "").strip())
    prompt = REMOVE_PROMPT.format(**what) + (KEEP_PROMPT.format(keep=what["keep"]) if what.get("keep") else "")
    if shadow := what.get("shadow"):
        # тень объекта за пределами выделения — эллипс по рамке DeepSeek, иначе останется «тень без хозяина»
        r.mask = r.mask.copy()
        ImageDraw.Draw(r.mask).ellipse(
            (round(shadow[0] * cw), round(shadow[1] * ch), round(shadow[2] * cw), round(shadow[3] * ch)), fill=255
        )
    native = model in (*QWEN21_MODELS, QWEN_EDIT_MODEL)
    if native:
        tpl = Q21_REMOVE_PROMPT if model in QWEN21_MODELS else Q2511_REMOVE_PROMPT
        prompt = tpl.format(**what) + (Q21_KEEP.format(keep=what["keep"]) if what.get("keep") else "")
    if model in QWEN21_MODELS:
        request = f"удали {hint.strip()}" if hint and hint.strip() else "удали то, что отмечено"
        prompt = await rewrite_q21(r.crop, r.mask, "remove", request) or prompt
        prompt += " " + Q21_MASK_TRACE
    prompt += REALISM_REMOVE
    log.info(
        "remove %s %dx%d кроп %s → %dx%d seed=%d тень=%s\n%s",
        model, *r.src.size, r.box, *r.gen_size, seed, what.get("shadow"), prompt,
    )
    # 2511 видит красный контур — зачистка, если он протечёт
    unleak = model == QWEN_EDIT_MODEL
    return await r.generate(prompt, seed, model, "remove", REMOVE_EDGE, unleak=unleak, native=native), seed


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
    """image 2 правки (подсказка): только место — мягкое полупрозрачное пятно на выделении, без линий и рамки.

    Чёткие красные линии модель «овеществляла» (обводка → красное кольцо на газоне, контур рамки → красная
    металлическая дуга), а заливка всей рамки протекала (люди в красном, красная дорожка под ними). Размер объекта
    задаёт текст инструкции (натуральный размер относительно соседей), место для него — область генерации.
    """
    feather = max(3, min(crop.size) // 80)
    soft = core.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(Image.blend(crop, Image.new("RGB", crop.size, MARK_COLOR), EDIT_MARK_OPACITY), crop, soft)


# Официальный системный промпт переписчика правок Qwen-Image-2.1 (QwenLM/Qwen-Image-2.1, prompt_rewrite/prompts/
# system_prompt_edit.txt; Qwen Research License — некоммерческое использование). Модель обучена на запросах, которые
# пишет этот переписчик; его роль у нас играет DeepSeek — свой Qwen3.5-переписчик не ставим
Q21_REWRITE_SYSTEM = (Path(__file__).parent / "prompts" / "qwen21_edit_rewrite.txt").read_text(encoding="utf-8")
# что переписчик узнаёт о маске и задаче — инструкция пользователя идёт следом как есть
Q21_TASK = {
    "add": (
        "<image1> is the photo to edit. <image2> is a mask: its white area marks where the new element stands and how "
        "big it is (the element stands on the ground at the bottom of the white area, at its true real-world size); "
        "everything outside the white area of <image1> stays unchanged, and inside it only the new element and its "
        "own shadow appear. Edit request"
    ),
    "change": (
        "<image1> is the photo to edit. <image2> is a mask: its white area marks what to change; everything outside "
        "the white area of <image1> stays unchanged. Edit request"
    ),
    "remove": (
        "<image1> is the photo to edit. <image2> is a mask: its white area marks what to remove (with its shadow) and "
        "fill with the background that continues around it; everything outside the white area of <image1> stays "
        "unchanged. Edit request"
    ),
}


async def rewrite_q21(
    crop: Image.Image, mask: Image.Image, task: str, request: str, texture: bytes | None = None
) -> str | None:
    """Запрос для Qwen-Image-2.1 так, как его пишет официальный переписчик: DeepSeek с их системным промптом,
    картинками <image1> (кроп) и <image2> (ч/б маска) и инструкцией пользователя. None — переписать не удалось."""
    if not DEEPSEEK_API_KEY:
        return None
    instruction = f"{Q21_TASK[task]}: «{request}»"
    if texture:
        instruction = (
            "<image3> is a sample photo of the new material: the marked surface gets exactly this material at true "
            "real-world scale. " + instruction
        )
    crop_png, mask_png = to_png(crop), to_png(mask.convert("RGB"))
    key = hashlib.sha256(crop_png + mask_png + instruction.encode() + llm_model.get().encode() + (texture or b"")).hexdigest()
    if key in _described:
        _described.move_to_end(key)
        return _described[key]["prompt"]
    messages = [
        {"role": "system", "content": Q21_REWRITE_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_url(crop_png, 1536)}},
                {"type": "image_url", "image_url": {"url": _data_url(mask_png, 1024)}},
            ] + ([{"type": "image_url", "image_url": {"url": _data_url(texture, TEXTURE_SIDE)}}] if texture else []) + [
                {"type": "text", "text": instruction},
            ],
        },
    ]
    try:
        data = json.loads(await _chat(messages, json_mode=True))
        text = " ".join(str(data.get("rewritten_prompt") or data.get("rewrited_prompt") or "").split())
    except Exception as e:  # noqa: BLE001 — без переписчика остаётся шаблон
        log.warning("Переписчик Qwen-Image-2.1 (DeepSeek) недоступен: %s", e)
        return None
    if not text:
        return None
    _described[key] = {"prompt": text}
    if len(_described) > CACHE_SIZE:
        _described.popitem(last=False)
    return text


def _outline(crop: Image.Image, area: Image.Image) -> Image.Image:
    """Picture 2 для Qwen-Image-Edit-2511: кроп с тонким красным контуром области правки (без заливки)."""
    out = crop.copy()
    width = max(2, min(crop.size) // 220)
    ImageDraw.Draw(out).bitmap((0, 0), ImageChops.subtract(_grow(area, width), area), fill=(255, 0, 40))
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
    "material": (
        "The user asked to change the MATERIAL or finish of the surface in the highlighted area: kind is \"change\". "
        "It is the same surface with the same shape, size, edges and joints, only covered with the new material; "
        "nothing is added in front of it."
    ),
}
# только если в запросе есть «дерево»: оговорка в любом запросе тянула описание к дереву (образец бетона → доски)
WOOD_RULE = " In this Russian request «дерево» means wood or timber cladding, never a tree."
# смена материала — основной сценарий: DeepSeek описывает новый материал так, чтобы модель положила его в масштабе,
# по плоскости и до настоящих краёв поверхности (модели «венге» мало: ни размера доски, ни направления, ни швов)
MATERIAL_SYSTEM = (
    " This is a MATERIAL change: the highlighted surface keeps its geometry and gets the new material. Also return "
    '"material": "<40-90 words: the new material for an image model: what it is (e.g. vertical larch boards, '
    "exposed-aggregate concrete, large-format porcelain tiles), its colour with hex codes and natural variation, pattern and grain, the real size of its "
    "elements in metres and how many of them fit along a nearby reference such as the 2.1 m door height, "
    "orientation and laying pattern, joints and gaps, finish and sheen (matte, oiled, polished), "
    "how it turns corners and meets window reveals, the plinth and the eaves (trims, mitred corners). If the user "
    'named a colour, a species or a pattern, keep it exactly>", '
    '"surface": "<8-25 words: which surfaces get it, by landmarks of image 1 (e.g. the plinth wall below the front '
    'windows between the downpipe and the porch)>", '
    '"keep_in_surface": "<things set into or in front of that surface that stay as they are: windows, frames, '
    'doors, sills, downpipes, lamps, vents, signs>" or null. '
    "The user's brush stroke is rough: inside the highlight the material covers that surface completely, up to its "
    "real edges, corners and openings, and never spreads onto other surfaces or materials the stroke touches."
)
# в запрос генерации: где и как лежит новый материал
MATERIAL_PROMPT = (
    " The new material, covering {surface} inside the marked region completely, up to its real edges, corners and "
    "openings and stopping there, without spilling onto other surfaces: {material}. It lies flat on the existing surface plane, follows its perspective and vanishing lines at "
    "true real-world scale, and turns the corners with proper trims; the shape, size, edges and openings of the "
    "surface stay exactly as before.{keep}"
)
MATERIAL_KEEP = " {keep} stay exactly as they are, in front of the new material."
# образец материала (текстура, которую перетащили во вкладке «Материалы»): DeepSeek описывает материал с него
TEXTURE_RULE = (
    " Image 3 is a sample photo of the new material supplied by the user: the new material is exactly this one, "
    "whatever the request words say. First look at image 3 carefully and return "
    '"sample": "<10-25 words: what image 3 really shows: material type, colour, pattern>"; "material" describes '
    "this same material and never another one. "
    "Describe it in \"material\" from image 3: its colour with hex codes, pattern, grain, element shapes and their "
    "real size, joints and finish, as a material laid on the surface at true real-world scale, not at the scale of "
    "the photo. Anything else in image 3 (background, objects, light) is not part of the material."
)
# в запрос Qwen-2.1: образец — третьей картинкой
Q21_TEXTURE = (
    " The new material is exactly the material shown in the sample photo <image3>: the same colour, pattern, grain, "
    "element shapes, joints and surface finish, laid on the marked surface at true real-world scale (not at the "
    "scale of the photo), following its plane and perspective. <image3> is only a material sample; nothing else "
    "from it appears in the result."
)
TEXTURE_SIDE = 768  # длинная сторона образца для модели и DeepSeek: хватает фактуре, не съедает видеопамять
# материалы и поверхности: запрос про них — смена материала той же поверхности, а не новый объект.
# «дерево» здесь не в списке: «замени дерево на фонарь» — про растение; «камень на дерево» ловится по «камень»
MATERIAL_WORDS = re.compile(
    r"камен|камн|кирпич|плитк|плитн|штукатур|бетон|облицов|отделк|фасад|цокол|стен|кровл|крыш|черепиц|металл|"
    r"сайдинг|панел|мрамор|гранит|травертин|клинкер|ламел|вагонк|имитац|покрыт|материал|"
    r"(stone|brick|tiles?|plaster|stucco|concrete|cladding|facade|wall|roof|siding|panels?|marble|material)"
)
# переписчику Qwen-2.1: та же оговорка — он тоже читал «дерево» как растение
MATERIAL_NOTE = (
    " (this is a material change of the same surface: its shape, size and edges stay, only the material changes)"
)
WOOD_NOTE = " («дерево» here means wood, not a tree)"


def parse_kind(request: str) -> str | None:
    """«добавь …» → add, «измени / замени / перекрась …» → change (material — если речь о материале или
    поверхности: «камень на дерево»), иначе None (решает DeepSeek)."""
    text = request.strip().lower().replace("ё", "е")
    words = text.split(maxsplit=1)
    first = words[0].strip(",.:;!") if words else ""
    if first in ADD_WORDS:
        return "add"
    if MATERIAL_WORDS.search(text):
        return "material"
    if first in CHANGE_WORDS:
        return "change"
    return None


async def describe_edit(
    frame: bytes, close: bytes, request: str, kind: str | None = None, texture: bytes | None = None
) -> dict:
    """План правки по всему кадру: что появится и какого реального размера, где, как выглядит кадр, рамка объекта.

    Без DeepSeek или при ошибке — запрос как есть, без рамки (рисуется в выделении).
    """
    fallback = {
        "instruction": request,
        "location": "in the marked area",
        "look": "the same light, shadows, colour grading and atmosphere as image 1",
        "kind": "change" if kind == "material" else kind or "add",
        "subject": "the edited element",
        "light_on_subject": "",
    }
    if not DEEPSEEK_API_KEY:
        return fallback
    key = hashlib.sha256(
        frame + b"edit" + request.encode() + (kind or "").encode() + llm_model.get().encode() + (texture or b"")
    ).hexdigest()
    if key in _described:
        _described.move_to_end(key)
        return _described[key]
    messages = [
        {
            "role": "system",
            "content": EDIT_SYSTEM + (MATERIAL_SYSTEM if kind == "material" else "") + (TEXTURE_RULE if texture else ""),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"User request: «{request}»" + (f" {KIND_RULE[kind]}" if kind else "")
                    + (WOOD_RULE if kind == "material" and "дерев" in request.lower() else ""),
                },
                {"type": "text", "text": "Image 1 (whole frame, area highlighted in red):"},
                {"type": "image_url", "image_url": {"url": _data_url(frame, 1536)}},
                {"type": "text", "text": "Image 2 (close-up of the area):"},
                {"type": "image_url", "image_url": {"url": _data_url(close, 1024)}},
            ] + ([
                {"type": "text", "text": "Image 3 (sample photo of the new material):"},
                {"type": "image_url", "image_url": {"url": _data_url(texture, TEXTURE_SIDE)}},
            ] if texture else []),
        },
    ]
    try:
        # рамка — это натуральный размер объекта; DeepSeek иногда её не присылает — тогда спрашиваем ещё раз,
        # иначе объект рисуется размером с выделение
        for attempt in range(2):
            data = json.loads(await _chat(messages, json_mode=True))
            # смене материала рамка не нужна — второй раз не спрашиваем
            if (box := _box(data.get("box"))) or kind == "material":
                break
            log.warning("DeepSeek не дал рамку объекта (попытка %d): %s", attempt + 1, data.get("box"))
        result = {
            k: " ".join(str(data.get(k) or "").split()).rstrip(".") or fallback[k] for k in fallback if k != "kind"
        }
        if box:
            result["box"] = box
        for k in ("material", "surface", "keep_in_surface", "sample"):
            v = " ".join(str(data.get(k) or "").split()).rstrip(".")
            if kind == "material" and v and v.lower() not in ("null", "none", "nothing"):
                result[k] = v
        said = str(data.get("kind", "")).strip().lower()
        # «добавь» — только добавление; «измени / замени» — изменение или замена, это решает DeepSeek
        if kind == "add":
            result["kind"] = "add"
        elif kind == "material":
            result["kind"] = "change"
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


def _rect_mask(box: list[float], size: tuple[int, int]) -> Image.Image:
    """Рамка (доли кадра) → маска со скруглёнными углами в размере исходника."""
    w, h = size
    x1, y1, x2, y2 = round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle((x1, y1, x2, y2), radius=round(0.12 * max(1, min(x2 - x1, y2 - y1))), fill=255)
    return m


async def _plan(
    src: Image.Image, core: Image.Image, request: str, kind: str | None = None, texture: bytes | None = None
) -> tuple[dict, Image.Image | None]:
    """Разбор правки DeepSeek (кэшируется) и рамка объекта в натуральную величину у выделения (None — «изменить»).
    kind — тип, заданный режимом (вкладка «Материалы» — material); иначе — по словам запроса."""
    close_box = _context_box(core.getbbox(), *src.size)
    what = await describe_edit(
        _highlight(src, core), _highlight(src.crop(close_box), core.crop(close_box)), request.strip(),
        kind or parse_kind(request), texture,
    )
    if what.get("kind") == "change" or not what.get("box"):
        return what, None
    extra = _box_mask(what["box"], src, core)
    if extra is None:
        log.warning("Рамка объекта отброшена (слишком мала или во весь кадр): %s", what["box"])
    return what, extra


async def plan_edit(image: bytes, mask: bytes, request: str) -> dict:
    """План правки для показа на канвасе: тип и рамка объекта (доли кадра), которую пользователь может поправить."""
    src, core, _ = _load(image, mask, EDIT_GROW)
    what, extra = await _plan(src, core, request)
    box = None
    if extra is not None:
        x1, y1, x2, y2 = extra.getbbox()
        w, h = src.size
        box = [x1 / w, y1 / h, x2 / w, y2 / h]
    return {"kind": what.get("kind", "add"), "box": box}


async def run_edit(
    image: bytes, mask: bytes, request: str, seed: int | None = None, model: str = DEFAULT_MODEL,
    box: list[float] | None = None, material: bool = False, texture: bytes | None = None,
    texture_width: float | None = None,
) -> tuple[bytes, int]:
    """JPEG исходника в полном разрешении с тем, что попросил пользователь, и сид.

    Выделение задаёт место, а не размер: DeepSeek по всему кадру (масштаб соседних объектов, перспектива)
    даёт рамку объекта в натуральную величину; генерация идёт по выделению + рамке + запасу вокруг,
    во вклейку попадает всё, что изменилось и связано с ними, — объект не обрезается о край маски.
    """
    if seed is None:
        seed = random.randrange(2**48)
    src, core, grow = _load(image, mask, EDIT_GROW)
    surface = None
    if material and texture:
        # образец — проекцией 1 в 1 (InegoTextureProject); не плоская поверхность — генеративный путь ниже
        surface = await _surface(src, core)
        width = texture_width or await _texture_width(texture)
        projected = await _project_texture(src, surface, texture, width)
        if projected is not None:
            return projected, seed
        log.info("проекция образца не удалась — материал по образцу через модель")
    # «изменить»: меняется то, что выделено, — рамка не нужна; «добавить» и «заменить» — объект в своём размере
    # вкладка «Материалы»: запрос — сам материал («дерево — венге»), тип правки задан, рамки нет
    tex = None
    if material and texture:
        # образец материала, перетащенный во вкладку «Материалы»: в DeepSeek, переписчик и модель (<image3>)
        tex = Image.open(io.BytesIO(texture)).convert("RGB")
        tex.thumbnail((TEXTURE_SIDE, TEXTURE_SIDE), Image.LANCZOS)
        tex = tex.resize((max(32, round(tex.width / 32) * 32), max(32, round(tex.height / 32) * 32)), Image.LANCZOS)
        texture = to_png(tex)
        request = "Смени материал выделенной поверхности на материал с образца" + (
            f": {request.strip()}" if request and request.strip() else ""
        )
    else:
        texture = None
        if material:
            request = f"Смени материал выделенной поверхности на: {request.strip()}"
    what, extra = await _plan(src, core, request, "material" if material else None, texture)
    if what.get("sample"):
        log.info("образец материала (DeepSeek): %s", what["sample"])
    add = what.get("kind") != "change"
    if add and box:
        # рамка, поправленная пользователем на канвасе
        extra = _rect_mask(box, src.size)
    if extra is not None:
        # рамка заменяет выделение, а не склеивается с ним в полосу: выделение остаётся только внутри рамки
        # (откуда начинается объект), а если рамку отодвинули — им становится нижняя треть рамки
        inside = ImageChops.multiply(core, extra)
        if inside.getbbox():
            core = inside
        else:
            x1, y1, x2, y2 = extra.getbbox()
            core = ImageChops.multiply(extra, _rect_mask([0, (y1 + 2 * (y2 - y1) / 3) / src.size[1], 1, 1], src.size))
    if material:
        # контур поверхности под мазком по исходнику (SAM 2.1): модель правит и вклейка берёт только его
        core = surface if surface is not None else await _surface(src, core)
    r = _Region(src, core, grow, extra, 0.0 if material else EDIT_ROOM if add else EDIT_ROOM_CHANGE)
    r.exact = material
    # «изменить» и материалы: поверхность та же — свет с исходника (Marigold-IID); «добавить / заменить» —
    # новый объект со своей формой, его свет сводит прогон переосвещения через модель
    r.source_light = not add
    if model in QWEN21_MODELS:
        r.gen_size = fit_size(*r.crop.size, QWEN21_MEGAPIXELS)
    # с рамкой (добавить / заменить) — без красной разметки: её модель копировала (кольца, дуги, красные пятна и
    # дорожки на земле); место задаёт область генерации — рамка в натуральную величину, размер — текст инструкции.
    # Разметка остаётся только для «изменить» и когда рамки нет: там она показывает, что именно меняется
    native = model in (*QWEN21_MODELS, QWEN_EDIT_MODEL)
    mark = EDIT_MARK and extra is None and not native
    if model in QWEN21_MODELS:
        # 2.1: место и размер — ч/б маской вторым входом (выделение + рамка объекта); запрос — официальный
        # переписчик Qwen в исполнении DeepSeek, шаблон — запасной вариант
        material = material or parse_kind(request) == "material"
        prompt = await rewrite_q21(
            r.crop, r.mask, "add" if extra is not None else "change", request.strip() + (MATERIAL_NOTE if material else "")
            + (WOOD_NOTE if material and "дерев" in request.lower() else ""),
            texture,
        ) or Q21_EDIT_PROMPT.format(**what, size_rule=Q21_SIZE_ADD if extra is not None else "")
        prompt += " " + Q21_MASK_TRACE + (Q21_MASK_TRACE_ADD if extra is not None else "")
        # переписчик пишет, что поменять, а о свете кадра молчит — измеренный DeepSeek свет отдельной фразой
        prompt += f" It is lit like the rest of <image1>: {what['look']}."
        if tex is not None:
            prompt += Q21_TEXTURE
    elif native:
        # 2511: место и размер — красным контуром на Picture 2
        prompt = Q2511_EDIT_PROMPT.format(**what, size_rule=Q2511_SIZE_ADD if extra is not None else "")
    else:
        prompt = EDIT_PROMPT.format(**what) + (MARK_RULE if mark else "")
    if what.get("material"):
        keep = MATERIAL_KEEP.format(keep=what["keep_in_surface"][:1].upper() + what["keep_in_surface"][1:])             if what.get("keep_in_surface") else ""
        text = MATERIAL_PROMPT.format(material=what["material"], surface=what.get("surface") or "the marked surface", keep=keep)
        prompt += q21_tags(text) if model in QWEN21_MODELS else text
    prompt += REALISM_EDIT
    log.info(
        "edit %s %dx%d кроп %s → %dx%d seed=%d %s рамка=%s «%s»\n%s",
        model, *src.size, r.box, *r.gen_size, seed, what.get("kind"), what.get("box") if extra else None, request, prompt,
    )
    reference = _mark(r.crop, r.core, r.extra) if mark else None
    # зачистка протёкшей разметки — если сам запрос не про красное (красная машина пострадала бы)
    unleak = (mark or model == QWEN_EDIT_MODEL) and not RED_WORDS.search(request.lower())
    # снятие белёсой вуали — только при добавлении и замене; при «изменить» перекраска того же объекта и есть цель
    refine = REFINE_PROMPT.format(instruction=what["instruction"].rstrip(".")) if add else None
    subject = what.get("subject") or "the edited element"
    subject = subject[:1].lower() + subject[1:]
    # свет на самом элементе (какая грань на солнце и какого оттенка, какая в тени) — с ним переосвещение заметное,
    # с одним общим описанием кадра модель почти ничего не меняет
    light = what.get("light_on_subject") or what["look"]
    relight = RELIGHT_PROMPT.format(subject=subject, subject_cap=subject[:1].upper() + subject[1:], look=light)
    return await r.generate(
        prompt, seed, model, "edit", EDIT_EDGE, reference, unleak, unveil=add, refine=refine, native=native,
        relight=relight, texture=tex if model in QWEN21_MODELS else None,
    ), seed
