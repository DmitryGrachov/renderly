"""Сборка промпта для FLUX.2 [klein] и Qwen-Image-2.1 через DeepSeek (vision).

Референс (image 2) — эталон окружения и света: небо, солнце и тени, лучи, атмосфера, трава и деревья,
колористика и «фотографичность». Здание, его геометрия, материалы и камера — строго с исходника (image 1).

Два шага:
  1. analyze_reference — DeepSeek подробно разбирает референс по разделам (кэшируется по хешу картинки);
  2. DeepSeek видит исходник и референс, получает разбор и пишет описательную часть под ЭТО здание:
     с какой стороны свет, какие фасады в тени, где на этой земле трава и деревья с рефа.
Замки геометрии/материалов и блок переноса окружения добавляются в промпт всегда, даже без DeepSeek.

Qwen-Image-2.1 (family="qwen21") — по правилам официального переписчика правок (prompts/qwen21_edit_rewrite.txt):
картинки — тегами <image1>, <image2>, <image3>, у каждой названа роль (холст / донор окружения / образец неба),
запрос начинается с действия, а смена «медиума» (рендер → фото) названа целью правки явно и в полную силу —
иначе 2.1 сохраняет медиум входа как идентичность. Стиль с картинки-референса 2.1 переносит хуже, чем по тексту,
поэтому разбор референса идёт в запрос целиком. Один абзац, всё утвердительно.
"""

import re

import asyncio
import base64
import hashlib
import io
import json
import logging
from collections import OrderedDict
from contextvars import ContextVar

import httpx
from PIL import Image

from .config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_URL

# модель DeepSeek текущего запроса (переключатель Flash / Pro в шапке); по умолчанию — из DEEPSEEK_MODEL
llm_model: ContextVar[str] = ContextVar("llm_model", default=DEEPSEEK_MODEL)
from .presets import PRESETS, reference_image

log = logging.getLogger("inego")

# Ответы DeepSeek по входам: тот же исходник + те же настройки → тот же промпт,
# поэтому с фиксированным сидом картинка повторяется (DeepSeek формулирует по-разному даже при temperature 0)
_cache: OrderedDict[str, tuple[str, list[float] | None]] = OrderedDict()
CACHE_SIZE = 256
# Разборы референсов по sha256 картинки: встроенные пресеты разбираются один раз за жизнь процесса
_analysis: OrderedDict[str, dict] = OrderedDict()
ANALYSIS_CACHE_SIZE = 128

# ---------------- обязательные блоки ----------------
INTRO = (
    "Turn this draft render into a real architectural photograph, indistinguishable from an unretouched photo "
    "taken on location with a full-frame camera."
)
KEEP_GEOMETRY = (
    "Keep the exact same camera position, focal length, perspective, framing and horizon line. "
    "Keep the building geometry, massing, silhouette, proportions, floor count, window layout, every opening, "
    "edge and line exactly as in image 1; do not add, remove or move any architectural element. The roofline "
    "and the silhouette of the building against the sky stay exactly as in image 1: no added rooftop volumes, "
    "penthouses, extra floors, parapets, antennas, signs or structures on or above the roof."
)
KEEP_MATERIALS = (
    "Keep every material exactly as in image 1: the same material type, colour, texture pattern, joint and panel "
    "layout, and the same surfaces it covers; do not replace, reinterpret or recolour any material, and do not add "
    "cladding, panels, wall lights or other fixtures. Only make the existing materials look physically real."
)
SPEC_MATERIALS = (
    "Materials specified by the architect, mandatory: {spec} Apply each of them to its surface exactly as specified, "
    "following the joints, seams and panel lines already drawn in image 1. Every surface not listed keeps its material "
    "exactly as in image 1. Do not add cladding, panels, wall lights or other fixtures."
)
# Рамка переноса: что берём с референса и где проходит граница
REF_RULE = (
    "Image 2 is the master reference for everything around the building: reproduce its sky, sunlight, shadows, "
    "light rays, atmosphere, vegetation and colour grading as faithfully as possible. Take from image 2 only the "
    "environment and the light, never its buildings, architecture, materials, camera angle or composition. "
    "Vegetation from image 2 grows only on the ground and in the background areas that are open in image 1; it never "
    "stands in front of the main building, never covers its facade and never changes its silhouette."
)
# «Сохранить зону» = сохранить её расклад и суть, но перерисовать фотореально в стилистике референса
KEEP_SKY = (
    "Sky: keep the cloud layout of image 1, the same clouds with the same shapes, sizes and positions and nothing "
    "added, but render it as a real photographed sky with the cloud texture, colours, light and atmosphere of image 2."
)
KEEP_BG = (
    "Background behind the main building: keep its layout from image 1, the same neighbouring and distant buildings, "
    "tree masses and skyline with the same shapes, sizes and positions, nothing added or removed, but re-render it "
    "as real photographed surroundings in the style of image 2: flat, translucent or schematic placeholder volumes "
    "become real buildings with plain, unobtrusive facades, blob or low-detail trees become real trees of the "
    "image 2 species and season, all in the light, haze and colours of image 2."
)
KEEP_SUR = (
    "Surroundings in front of and around the main building: keep their layout from image 1, the same lawn areas, "
    "paths, roads, paving, planting beds, trees, cars and people with the same shapes, sizes and positions, nothing "
    "added or removed, but re-render them as real photographed surroundings in the style of image 2: a flat green "
    "plane becomes real lawn grass like in image 2, simple or low-poly trees and shrubs become real ones of the "
    "image 2 species and season, flat grey areas become real asphalt or paving, placeholder cars and people become "
    "real ones, all in the light, shadows and colours of image 2."
)
# Реализм — утвердительно и языком фотографа (гайды BFL по FLUX.2 и по Qwen-Image): камера, объектив, свет, фактура
# и мелкие несовершенства реального фото. Без «8k / masterpiece / ultra-detailed / flawless / render» — они тянут
# в CGI, и без отрицаний: FLUX не поддерживает негативные промпты, отрицание фокусирует модель на запрещённом
PHOTOREAL = (
    "An unretouched architectural photograph taken on location with a full-frame camera and a sharp prime lens: "
    "physically correct light falloff, bounce light and soft contact shadows, real glass with reflections and "
    "visible interiors, individual leaves and grass blades, a natural sky, true-to-life materials with real "
    "micro-texture, subtle weathering and slight unevenness, natural lens behaviour with gentle depth-of-field "
    "falloff and faint vignetting, true dynamic range and fine sensor grain."
)
# для правок Inpainting: то, что добавлено или изменено, — снято той же камерой в том же кадре
REALISM_EDIT = (
    " Everything new is a real photographed thing from the same shot, as an unretouched photograph: true-to-life "
    "materials with real micro-texture and subtle wear, people with natural skin texture, visible pores, slight "
    "asymmetry and relaxed natural poses, fabric with real folds, the same lens sharpness, depth of field, "
    "dynamic range, noise and fine sensor grain as the rest of the photo."
)
REALISM_REMOVE = (
    " The filled area reads as the same unretouched photograph: the same real surface texture, wear, "
    "noise and fine sensor grain as its surroundings."
)
FINAL_LOCK = "The building itself stays exactly as in image 1: only the world around it and the light change."

# ---------------- Qwen-Image-2.1 ----------------
# с действия и роли картинок: <image1> — холст, <image2> — донор окружения и света; смена медиума — цель правки
Q21_INTRO = (
    "Convert image 1, a draft architectural 3D render, into a real unretouched architectural photograph taken on "
    "location with a full-frame camera, changing its medium completely from computer graphics to photography. "
    "Image 1 is the canvas: its camera, framing, composition, building and layout carry over to the result. "
    "Every object of image 1 keeps its place, shape and size: the streets and paving, cars, people, neighbouring "
    "buildings, trees and shrubs all stay, each becoming its real photographed counterpart."
)
Q21_REF_ROLE = (
    "Image 2 is a real reference photograph that supplies only the environment and the light: its sky, sun, shadows, "
    "atmosphere, vegetation, colour grading and photographic look."
)
# усиление реализма: 2.1 держит медиум входа, поэтому «как снято сенсором» — отдельной фразой
Q21_PHOTOREAL = (
    " Every surface, plant, cloud and reflection reads as captured by a camera sensor, with the natural "
    "irregularity, small imperfections and optical character of a real photograph."
)
# правило для DeepSeek: как пишет официальный переписчик 2.1 — то, что остаётся, называется по типу и месту,
# не перерисовывается словами (подробное описание сохраняемого 2.1 читает как команду нарисовать заново)
Q21_BODY_RULE = """- Write it as an edit instruction from someone holding only the input images. Describe concretely only
  what changes: the light, the sky, the vegetation, how the existing surfaces look once photographed. Name what stays
  (the building, its openings, the layout) by type and position, without re-describing its appearance.
"""


def q21_tags(text: str) -> str:
    """«image 1» → «<image1>»: в Qwen-Image-2.1 картинки адресуются только тегами; один абзац без переносов."""
    return " ".join(re.sub(r"\b[Ii]mage ([1-9])\b", r"<image\1>", text).split())

# ---------------- разбор референса ----------------
ANALYSIS_FIELDS = {
    "time_and_weather": "time of day, season, weather",
    "cloud_cover": (
        "cloud cover in okta (0-8) and percent of the sky covered, cloud types, their size and where they are; "
        "for a cloudless sky write exactly 'clear sky, 0 okta, no clouds'"
    ),
    "sky": (
        "colours from zenith to horizon with hex codes, gradient shape, cloud types, their size, density, "
        "position in the frame and how they are lit, sun visible or not and where"
    ),
    "sun_and_shadows": (
        "sun elevation in degrees, sun azimuth relative to the camera (from the left/right, behind the camera, "
        "against the camera), direct light colour in Kelvin and hex, hardness, shadow length, direction, edge "
        "softness, shadow colour and density, fill/bounce light, rim light"
    ),
    "light_rays_and_atmosphere": (
        "god rays, volumetric light, haze, mist, dust, humidity, aerial perspective and how distance fades, "
        "bloom, glow around bright areas, lens flare"
    ),
    "vegetation": (
        "grass: type, length, density, condition, colour in sun and in shade with hex; trees: species or type, "
        "crown shape and size, foliage density, season, leaf colours with hex, how light passes through leaves, "
        "bark; shrubs, hedges, flowers; write 'none' if there is no vegetation"
    ),
    "ground_and_water": "paving, asphalt, soil, wet or dry, puddles, reflections, water bodies; 'none' if not visible",
    "artificial_light": "street lamps, windows, interior lights, their colour in Kelvin and intensity; 'none' if absent",
    "colour_grade": (
        "white balance in Kelvin, 4-6 dominant palette colours with hex, tint of highlights, midtones and shadows, "
        "contrast curve, black level, saturation, overall exposure"
    ),
    "photography": (
        "camera and lens feel, dynamic range, depth of field, sharpness, grain or noise, vignetting, "
        "and the concrete cues that make it look like a real photo"
    ),
    "mood": "the atmosphere in a few words",
}

ANALYSIS_SYSTEM = (
    "You are a senior director of photography and archviz colourist. Analyse the reference photo with extreme "
    "precision so that another image can be lit, graded and landscaped to be indistinguishable from it. "
    "Describe what you SEE, concretely and measurably: name colours with hex codes, colour temperature in Kelvin, "
    "angles in degrees, directions relative to the camera, species and sizes. No vague words like 'beautiful' or "
    "'nice'. Ignore buildings and architecture entirely: never describe their shape, style or materials. Never "
    "mention people, cars, animals or other objects. If a section does not apply, its value is exactly \"none\". "
    "English. Answer with JSON with exactly these keys, each value 25-60 words:\n"
    + "\n".join(f'- "{k}": {v}' for k, v in ANALYSIS_FIELDS.items())
    + '\n- "sky_box": [x1, y1, x2, y2] normalised to 0..1000 of the image width and height: the largest rectangle '
    "that contains ONLY sky (no buildings, trees, poles or wires); [0, 0, 0, 0] if no clean sky is visible"
)

# ---------------- описательная часть под конкретное здание ----------------
PROMPT_SYSTEM = """You are a prompt engineer for {model}, an instruction-based image EDITING model.
Image 1 is a draft {scene} architectural image: a viewport screenshot, a clay render or a weak render.
Image 2 is the master reference for the environment and the light. You also get a detailed analysis of image 2.
Write the descriptive part of an edit instruction that turns image 1 into a real photograph that looks as if it was
shot in the exact world, weather, light and atmosphere of image 2.
Camera and geometry locks and the full reference analysis are added to the prompt separately - do not repeat them.

Rules:
- The building is sacred: never change its geometry, silhouette, openings or proportions.
{material_rules}
- Map the light of image 2 onto THIS building: say from which side the sun comes relative to this camera, which
  facades are in direct light and which are in shade, where and how long the shadows fall on this ground, where
  the light rays and haze are.
- Map the environment of image 2 onto THIS scene: the sky of image 2 above this building, the grass, trees,
  shrubs and their season and colours of image 2 on the ground and in the background areas that are open in
  image 1. Vegetation never stands in front of the main building and never covers its facade.
- The light and colour grading must never change the material colours, only how they are lit.
{sur_rule}{model_rule}- 120-180 words, plain English, no lists, no quotes, no negative prompt.
{box_rule}Answer with JSON: {json_shape}"""

SUR_KEEP_RULE = """- The surroundings of image 1 keep their layout: nothing is added, removed or moved. Go through what is there
  and say how each element becomes real in the style of image 2 at the same place, e.g. "the flat green area in the
  foreground becomes dense lawn like in image 2", "the simple round trees on the right become <image 2 species>".
"""
SKY_KEEP_RULE = """- The sky of image 1 keeps its cloud layout: describe the same clouds rendered with the texture, colours and
  light of image 2, nothing new.
"""
BG_KEEP_RULE = """- The background behind the building keeps its layout: describe how each background element (neighbouring
  volumes, tree masses, skyline) becomes real in the style of image 2 at the same place, nothing new.
"""

BOX_RULE = """- Also locate the MAIN building: the subject of the visualisation, usually the largest and most central one,
  not the neighbouring buildings. Put into "box" [x1, y1, x2, y2], normalised to 0..1000 of the image width
  and height, tightly enclosing the whole main building from the top of the roof (chimneys, parapets,
  rooftop signs) down to the ground line.
"""

RULES_FROM_IMAGE = """- Materials are fixed by the architect. Name each facade material (walls, cladding, plinth, roof, frames,
  glazing, paving) only as you actually see it in image 1, with its colour and pattern, and say it stays as it is.
  Never invent, upgrade or "improve" a material, never take materials from image 2. If a surface is untextured,
  flat or unclear (clay or viewport render), call it "the existing <colour> <surface> finish" instead of guessing.
  You may describe realism only: fine surface texture, subtle weathering, real glass with reflections of the
  image 2 sky and visible interiors behind the glazing."""

RULES_FROM_SPEC = """- The architect specified the materials (possibly in Russian). They have the highest priority and override
  what the draft shows. First translate them into precise English archviz terms, keeping every colour,
  finish and pattern, and put that into "materials" as one sentence naming each surface and its material.
  In "prompt", describe these materials on the exact surfaces of this building where they belong.
  Surfaces the architect did not mention keep what you see in image 1: never invent materials, never take them
  from image 2. You may describe realism only: fine surface texture, subtle weathering, real glass with
  reflections of the image 2 sky and visible interiors behind the glazing."""


def _data_url(image: bytes, max_side: int = 1536) -> str:
    img = Image.open(io.BytesIO(image)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def _chat(messages: list[dict], json_mode: bool = False) -> str:
    body = {"model": llm_model.get(), "messages": messages, "temperature": 0}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{DEEPSEEK_URL}/chat/completions", headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}, json=body
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def _clean(v) -> str:
    s = " ".join(str(v or "").split()).strip()
    low = s.lower()
    return "" if not s or low.startswith(("none", "n/a", "not visible", "no vegetation")) else s.rstrip(".") + "."


async def analyze_reference(reference: bytes) -> dict:
    """Подробный разбор референса по разделам ANALYSIS_FIELDS (пустые разделы отброшены)."""
    key = hashlib.sha256(reference + llm_model.get().encode()).hexdigest()
    if key in _analysis:
        _analysis.move_to_end(key)
        return _analysis[key]
    raw = await _chat(
        [
            {"role": "system", "content": ANALYSIS_SYSTEM},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url(reference, 2048)}}]},
        ],
        json_mode=True,
    )
    data = json.loads(raw)
    result = {k: c for k in ANALYSIS_FIELDS if (c := _clean(data.get(k)))}
    if box := _box(data.get("sky_box")):
        result["_sky_box"] = box  # рамка чистого неба на референсе — для прохода замены неба
    log.info("Разбор референса %s:\n%s", key[:10], json.dumps(result, ensure_ascii=False, indent=1))
    _analysis[key] = result
    if len(_analysis) > ANALYSIS_CACHE_SIZE:
        _analysis.popitem(last=False)
    return result


async def warm_preset_analyses() -> None:
    """Разбирает встроенные референсы пресетов заранее, чтобы первый рендер не ждал DeepSeek."""
    if not DEEPSEEK_API_KEY:
        return
    for scene, presets in PRESETS.items():
        for preset in presets:
            ref = reference_image(scene, preset)
            if not ref:
                continue
            try:
                await analyze_reference(ref)
            except Exception as e:  # noqa: BLE001
                log.warning("Не удалось разобрать референс %s/%s: %s", scene, preset, e)
            await asyncio.sleep(0)


def is_clear_sky(cloud_cover: str) -> bool:
    c = cloud_cover.lower()
    return any(x in c for x in ("0 okta", "no clouds", "cloudless", "clear sky"))


def clouds_text(cloud_cover: str, image: str = "image 2") -> str:
    """Облачность — отдельной фразой: модель охотно оставляет облака черновика."""
    if is_clear_sky(cloud_cover):
        return f"Cloud cover exactly as in {image}: a completely clear, cloudless sky, not a single cloud anywhere."
    return f"Cloud cover exactly as in {image}, the same amount and density of clouds, no more and no fewer: {cloud_cover}"


# образец неба: вырезанное чистое небо референса подаётся в основной проход третьей картинкой
SKY_IMAGE = (
    "Image 3 is a close-up of the sky of image 2: the sky of the result must look exactly like image 3, with the "
    "same colours, gradient, light, cloud types, amount and density; no clouds from image 1 remain."
)


def reference_block(analysis: dict, keep_sky: bool, keep_bg: bool, keep_sur: bool) -> str:
    """Блок переноса окружения и света с референса — идёт в промпт целиком, без пересказа."""
    a = analysis
    parts = []
    if a.get("time_and_weather"):
        parts.append(f"Time and weather as in image 2: {a['time_and_weather']}")
    if a.get("sky"):
        parts.append(
            f"Sky colours, light and cloud texture as in image 2, cloud layout as in image 1: {a['sky']}" if keep_sky
            else f"Replace the sky with the sky of image 2: {a['sky']}"
        )
    if a.get("cloud_cover") and not keep_sky:
        parts.append(clouds_text(a["cloud_cover"]))
    if a.get("sun_and_shadows"):
        parts.append(f"Sunlight and shadows exactly as in image 2: {a['sun_and_shadows']}")
    if a.get("light_rays_and_atmosphere"):
        parts.append(f"Light rays and atmosphere as in image 2: {a['light_rays_and_atmosphere']}")
    if a.get("vegetation"):
        if keep_sur:
            parts.append(f"Existing plants keep their places and sizes and become the species, season and colours of image 2: {a['vegetation']}")
        else:
            parts.append(f"Grass, trees and shrubs exactly like in image 2: {a['vegetation']}")
    if a.get("ground_and_water"):
        parts.append(f"Ground surface condition as in image 2: {a['ground_and_water']}")
    if a.get("artificial_light"):
        parts.append(f"Artificial light as in image 2: {a['artificial_light']}")
    if a.get("colour_grade"):
        parts.append(f"Colour grading exactly as in image 2: {a['colour_grade']}")
    if a.get("photography"):
        parts.append(f"Photographic look of image 2: {a['photography']}")
    if a.get("mood"):
        parts.append(f"Mood: {a['mood']}")
    return " ".join(parts)


def assemble(
    body: str,
    materials: str | None,
    ref_text: str,
    has_ref: bool,
    keep_sky: bool = False,
    keep_bg: bool = False,
    keep_sur: bool = False,
    sky_image: bool = False,
    family: str = "flux",
) -> str:
    """Промпт = вступление + замки здания + рамка референса + замки зон + разбор референса
    + описательная часть под здание + [образец неба] + фотореализм + финальный замок здания.
    family="qwen21": вступление с действия и ролей картинок, теги <imageN>, один абзац."""
    lock = SPEC_MATERIALS.format(spec=materials.rstrip(".") + ".") if materials else KEEP_MATERIALS
    q21 = family == "qwen21"
    parts = [Q21_INTRO, Q21_REF_ROLE if has_ref else "", KEEP_GEOMETRY, lock] if q21 else [INTRO, KEEP_GEOMETRY, lock]
    if has_ref:
        parts.append(REF_RULE)
    if keep_sky:
        parts.append(KEEP_SKY)
    if keep_bg:
        parts.append(KEEP_BG)
    if keep_sur:
        parts.append(KEEP_SUR)
    parts += [ref_text, body.strip(), SKY_IMAGE if sky_image else "", PHOTOREAL + (Q21_PHOTOREAL if q21 else ""), FINAL_LOCK]
    text = " ".join(p for p in parts if p)
    return q21_tags(text) if q21 else text


# ---------------- перерисовка сохраняемых зон (второй проход, только по маске) ----------------
# Короткая прямая команда: модель редактирования при длинном промпте «сохрани как в image 1»
# копирует заглушки исходника (пятна краски, прозрачные блоки) вместо того, чтобы сделать их настоящими.
ZONE_INTRO = (
    "Edit only the {zones}; the building stays exactly as it is. Keep the layout: every element stays at the same "
    "place with the same shape and size, nothing is added or removed. Replace every schematic, flat, painted or "
    "translucent placeholder with its real photographed counterpart: translucent, glass-like, white or blank "
    "volumes are neighbouring buildings and become real opaque buildings of the same shape with plain, unobtrusive "
    "facades;"
)
ZONE_SKY = "a flat, painted or CG sky becomes a real photographed sky with the same cloud layout"
ZONE_BG = (
    "translucent, blank or untextured volumes become real buildings with plain, unobtrusive facades; blob or "
    "low-detail trees become real trees"
)
ZONE_SUR = (
    "flat or painted green areas become dense real lawn grass with individual blades; flat grey areas become real "
    "asphalt or paving; simple trees and shrubs become real ones; placeholder cars and people become real ones"
)
Q21_ZONE_PLANTS = (
    "Every existing tree, shrub and lawn of image 1 keeps its number, place, shape and size, and no new plants "
    "appear; only their species look, season, leaf colours, grass texture and light follow image 2."
)
ZONE_OUTRO = (
    "Everything must look like an unretouched real photograph, seamlessly blended with the building and the rest "
    "of the image."
)


async def zone_instruction(
    reference: bytes | None, keep_sky: bool, keep_bg: bool, keep_sur: bool, family: str = "flux"
) -> str:
    """Команда для второго прохода: заглушки в сохраняемых зонах → настоящие объекты в стилистике референса."""
    names, rules = [], []
    if keep_sky:
        names.append("sky")
        rules.append(ZONE_SKY)
    if keep_bg:
        names.append("background behind the building")
        rules.append(ZONE_BG)
    if keep_sur:
        names.append("ground and surroundings in front of and around the building")
        rules.append(ZONE_SUR)
    parts = [ZONE_INTRO.format(zones=", ".join(names)), "; ".join(rules) + "."]

    a = {}
    if reference is not None and DEEPSEEK_API_KEY:
        try:
            a = await analyze_reference(reference)  # из кэша: тот же разбор, что и для первого прохода
        except Exception as e:  # noqa: BLE001
            log.warning("Разбор референса для зон недоступен: %s", e)
    style = []
    if (keep_bg or keep_sur) and family == "qwen21":
        # 2.1 читает перечень растений референса («саженцы на кольях, низкие кусты») как команду их добавить
        style.append(Q21_ZONE_PLANTS)
    elif (keep_bg or keep_sur) and a.get("vegetation"):
        style.append(f"Grass, trees and shrubs exactly like in image 2: {a['vegetation']}")
    if keep_sky and a.get("sky"):
        style.append(f"Sky colours, light and cloud texture like in image 2: {a['sky']}")
    for k, label in (("sun_and_shadows", "Light and shadows"), ("light_rays_and_atmosphere", "Atmosphere"),
                     ("colour_grade", "Colour grading")):
        if a.get(k):
            style.append(f"{label} like in image 2: {a[k]}")
    if not style and reference is not None:
        style.append("Match the vegetation, sky, light and colour grading of image 2.")
    text = " ".join(parts + style + [ZONE_OUTRO])
    return q21_tags(text) if family == "qwen21" else text


def _box(raw) -> list[float] | None:
    """[x1, y1, x2, y2] в 0..1000 → доли кадра; мусор — None (тогда здание ищет SegFormer)."""
    try:
        v = [min(1.0, max(0.0, float(x) / 1000)) for x in raw]
    except (TypeError, ValueError):
        return None
    return v if len(v) == 4 and v[0] < v[2] and v[1] < v[3] else None


def fallback_body(scene: str, look: str) -> str:
    return (
        f"A photoreal {scene} with fine physical texture detail on every material, real glass with reflections "
        f"and visible interiors, photoreal landscaping and sky. Lighting and colour grading: {look}"
    )


async def build_prompt(
    image: bytes,
    scene: str,
    preset: str,
    reference: bytes | None,
    materials: str | None = None,
    keep_sky: bool = False,
    keep_bg: bool = False,
    keep_sur: bool = False,
    family: str = "flux",
) -> tuple[str, list[float] | None]:
    """Итоговый промпт для FLUX / Qwen-Image-2.1 (family) и рамка главного здания (доли кадра; только при keep_bg/keep_sur, для SAM).

    reference — референс пресета или пользователя (image 2); materials — спецификация архитектора,
    пусто — материалы берутся с изображения. При любой ошибке DeepSeek — шаблонный промпт из пресета.
    """
    p = PRESETS[scene][preset]
    look = p.get("look") or p.get("prompt") or "natural light matching image 2."
    materials = (materials or "").strip() or None
    has_ref = reference is not None

    keep = (keep_sky, keep_bg, keep_sur)
    # рамка главного здания: для зон и для замка силуэта при замене неба (экстерьер)
    need_box = keep_bg or keep_sur or scene == "exterior"

    if not DEEPSEEK_API_KEY:
        return assemble(fallback_body(scene, look), materials, "", has_ref, *keep, family=family), None

    h = hashlib.sha256(image)
    h.update(hashlib.sha256(reference or b"").digest())
    h.update(json.dumps([scene, preset, materials, keep, "v6", llm_model.get(), family]).encode())
    key = h.hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        log.info("Промпт из кэша (те же входы)")
        return _cache[key]

    try:
        analysis = await analyze_reference(reference) if has_ref else {}
        ref_text = reference_block(analysis, *keep)
        if not ref_text:
            ref_text = f"Lighting and colour grading: {look}"

        shape = {"materials": "...", "prompt": "..."} if materials else {"prompt": "..."}
        if need_box:
            shape["box"] = [0, 0, 1000, 1000]
        system = PROMPT_SYSTEM.format(
            model="Qwen-Image-2.1" if family == "qwen21" else "FLUX.2 [klein]",
            model_rule=Q21_BODY_RULE if family == "qwen21" else "",
            scene=scene,
            material_rules=RULES_FROM_SPEC if materials else RULES_FROM_IMAGE,
            sur_rule=(SKY_KEEP_RULE if keep_sky else "") + (BG_KEEP_RULE if keep_bg else "") + (SUR_KEEP_RULE if keep_sur else ""),
            box_rule=BOX_RULE if need_box else "",
            json_shape=json.dumps(shape),
        )
        text = "Analysis of image 2 (environment and light reference):\n" + (
            json.dumps({k: v for k, v in analysis.items() if not k.startswith("_")}, ensure_ascii=False, indent=1)
            if analysis else look
        )
        if materials:
            text += f"\nArchitect's material specification: {materials}"
        content = [
            {"type": "text", "text": text},
            {"type": "text", "text": "Image 1 (the draft, the building to keep):"},
            {"type": "image_url", "image_url": {"url": _data_url(image)}},
        ]
        if has_ref:
            content += [
                {"type": "text", "text": "Image 2 (environment and light reference):"},
                {"type": "image_url", "image_url": {"url": _data_url(reference, 1024)}},
            ]
        # второй попыткой — при разовом сбое (таймаут, битый JSON): шаблон намного слабее, а Qwen-Image-2.1
        # с ним почти копирует черновик
        for attempt in range(2):
            try:
                raw = await _chat([{"role": "system", "content": system}, {"role": "user", "content": content}], json_mode=True)
                data = json.loads(raw)
                data["prompt"]
                break
            except Exception as e:  # noqa: BLE001
                if attempt:
                    raise
                log.warning("DeepSeek: сбой описания, повторяю: %s", e)
        if materials:
            # переведённая DeepSeek спецификация; если не вернул — исходный текст архитектора
            materials = (data.get("materials") or "").strip() or materials
        # образец неба (image 3) — экстерьер без «Сохранить небо», если на референсе есть чистое небо
        # (только FLUX: в Qwen-Image-2.1 третья картинка не помещается в память)
        sky_image = scene == "exterior" and not keep_sky and bool(analysis.get("_sky_box")) and family == "flux"
        result = (
            assemble(data["prompt"], materials, ref_text, has_ref, *keep, sky_image=sky_image, family=family),
            _box(data.get("box")) if need_box else None,
        )
        _cache[key] = result
        if len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
        return result
    except Exception as e:  # noqa: BLE001 — генерация не должна падать из-за LLM
        log.warning("DeepSeek недоступен, использую шаблонный промпт: %s", e)
        return assemble(fallback_body(scene, look), materials, "", has_ref, *keep, family=family), None
