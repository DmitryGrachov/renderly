"""Сборка промпта для FLUX.2 [klein] через DeepSeek (vision).

DeepSeek смотрит на исходник (вьюпорт / клей / слабый рендер) и пишет описательную часть инструкции:
материалы, окружение, свет пресета. Замки камеры, геометрии и материалов добавляются всегда сверху.
"""

import base64
import io
import json
import logging

import httpx
from PIL import Image

from .config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_URL
from .presets import PRESETS

log = logging.getLogger("inego")

# Обязательные блоки — добавляются в промпт всегда, даже если DeepSeek недоступен.
INTRO = "Turn this draft render into a photorealistic architectural photograph."
KEEP_GEOMETRY = (
    "Keep the exact same camera position, focal length, perspective, framing and horizon line. "
    "Keep the building geometry, massing, proportions, floor count, window layout, every opening, "
    "edge and line exactly as in image 1; do not add, remove or move any architectural element."
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
REF_ONLY_LIGHT = "Image 2 is only a lighting reference: take its light, sky and colour grading, nothing else."
KEEP_SKY = (
    "Keep the sky of image 1: the same clouds with the same shapes and positions, no new clouds or objects; "
    "only restyle its colour, light and atmosphere to the target lighting."
)
KEEP_BG = (
    "Keep the background behind the main building as in image 1: the same neighbouring and distant buildings, "
    "trees and skyline with the same shapes, sizes and positions; only restyle their light, colour and atmosphere "
    "to the target lighting. Nothing new in the background and in the sky: no added buildings, trees or objects."
)
KEEP_SUR = (
    "Keep the surroundings in front of and around the main building as in image 1: the same ground, road and "
    "paving layout, lawn, landscaping, trees, cars and people with the same shapes, sizes and positions; do not "
    "add, remove or move anything, only restyle their light, colour and atmosphere to the target lighting."
)

LOOK_SYSTEM = (
    "You analyse reference photos for an architectural visualisation studio. Describe ONLY the lighting and mood: "
    "time of day, sun direction and hardness, sky (colour, gradient, clouds), shadows, artificial lights, white "
    "balance, colour grading, contrast, saturation and atmosphere. Never describe buildings, objects or layout. "
    "One paragraph, 50-80 words, English."
)

PROMPT_SYSTEM = """You are a prompt engineer for FLUX.2 [klein], an instruction-based image EDITING model.
The user gives you a draft {scene} architectural image: a viewport screenshot, a clay render or a weak render.
Write the descriptive part of an edit instruction that turns it into a final photorealistic architectural
visualisation, like a high-end archviz studio render or professional architectural photography.
Camera and geometry locks are added separately - do not repeat them.

Rules:
{material_rules}
- Describe the surroundings only where the draft already has them (ground, landscaping, trees, street, sky),
  and make them photoreal. Small details such as a few people or cars are fine only where there is free space.
- Apply the target lighting and colour grading described by the user. Image 2, if given, is ONLY a lighting and
  colour reference: mention "match the lighting, sky and colour grading of image 2" and never copy its buildings
  or materials. The target lighting must never change material colours, only how they are lit.
- 90-150 words, plain English, no lists, no quotes, no negative prompt.
{box_rule}Answer with JSON: {json_shape}"""

BOX_RULE = """- Also locate the MAIN building: the subject of the visualisation, usually the largest and most central one,
  not the neighbouring buildings. Put into "box" [x1, y1, x2, y2], normalised to 0..1000 of the image width
  and height, tightly enclosing the whole main building from the top of the roof (chimneys, parapets,
  rooftop signs) down to the ground line.
"""

RULES_FROM_IMAGE = """- Materials are fixed by the architect. Name each facade material (walls, cladding, plinth, roof, frames,
  glazing, paving) only as you actually see it, with its colour and pattern, and say it stays as it is.
  Never invent, upgrade or "improve" a material. If a surface is untextured, flat or unclear (clay or viewport
  render), call it "the existing <colour> <surface> finish" instead of guessing a material type.
  You may describe realism only: fine surface texture, subtle weathering, real glass with reflections and
  visible interiors behind the glazing."""

RULES_FROM_SPEC = """- The architect specified the materials (possibly in Russian). They have the highest priority and override
  what the draft shows. First translate them into precise English archviz terms, keeping every colour,
  finish and pattern, and put that into "materials" as one sentence naming each surface and its material.
  In "prompt", describe these materials on the exact surfaces of this building where they belong.
  Surfaces the architect did not mention keep what you see: name them as they are, never invent materials.
  You may describe realism only: fine surface texture, subtle weathering, real glass with reflections and
  visible interiors behind the glazing."""


def _data_url(image: bytes, max_side: int = 1536) -> str:
    img = Image.open(io.BytesIO(image)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def _chat(messages: list[dict], json_mode: bool = False) -> str:
    body = {"model": DEEPSEEK_MODEL, "messages": messages, "temperature": 0.2}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            f"{DEEPSEEK_URL}/chat/completions", headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}, json=body
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


async def describe_look(reference: bytes) -> str:
    """Описание света и колористики пользовательского референса."""
    return await _chat(
        [
            {"role": "system", "content": LOOK_SYSTEM},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url(reference)}}]},
        ]
    )


def assemble(
    body: str,
    materials: str | None,
    has_ref: bool,
    keep_sky: bool = False,
    keep_bg: bool = False,
    keep_sur: bool = False,
) -> str:
    """Промпт = вступление + замки геометрии, материалов, неба/фона/окружения + описательная часть (+ оговорка про image 2)."""
    lock = SPEC_MATERIALS.format(spec=materials.rstrip(".") + ".") if materials else KEEP_MATERIALS
    parts = [INTRO, KEEP_GEOMETRY, lock]
    if keep_sky:
        parts.append(KEEP_SKY)
    if keep_bg:
        parts.append(KEEP_BG)
    if keep_sur:
        parts.append(KEEP_SUR)
    parts.append(body.strip())
    if has_ref:
        parts.append(REF_ONLY_LIGHT)
    return " ".join(p for p in parts if p)


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
) -> tuple[str, list[float] | None]:
    """Итоговый промпт для FLUX и рамка главного здания (доли кадра; только при keep_bg/keep_sur, для SAM).

    materials — спецификация архитектора; пусто — материалы берутся с изображения.
    При любой ошибке DeepSeek — шаблонный промпт из пресета.
    """
    p = PRESETS[scene][preset]
    look = p.get("look") or p.get("prompt") or "natural daylight."
    materials = (materials or "").strip() or None
    has_ref = reference is not None

    keep = (keep_sky, keep_bg, keep_sur)
    need_box = keep_bg or keep_sur

    if not DEEPSEEK_API_KEY:
        return assemble(fallback_body(scene, look), materials, has_ref, *keep), None

    try:
        if preset == "custom" and has_ref:
            look = await describe_look(reference)
        shape = {"materials": "...", "prompt": "..."} if materials else {"prompt": "..."}
        if need_box:
            shape["box"] = [0, 0, 1000, 1000]
        system = PROMPT_SYSTEM.format(
            scene=scene,
            material_rules=RULES_FROM_SPEC if materials else RULES_FROM_IMAGE,
            box_rule=BOX_RULE if need_box else "",
            json_shape=json.dumps(shape),
        )
        text = f"Target lighting and colour grading: {look}"
        if has_ref:
            text += " A lighting reference photo is passed to FLUX as image 2."
        if materials:
            text += f"\nArchitect's material specification: {materials}"
        if keep_sky:
            text += "\nThe sky of the draft keeps its clouds and composition: describe only its new colour and light, no new clouds."
        if keep_bg:
            text += "\nThe background behind the building keeps its content: describe only its new light and colour, nothing new."
        if keep_sur:
            text += "\nThe surroundings keep their content (ground, road, landscaping, cars, people): describe only their new light and colour, nothing new."
        raw = await _chat(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": _data_url(image)}},
                    ],
                },
            ],
            json_mode=True,
        )
        data = json.loads(raw)
        if materials:
            # переведённая DeepSeek спецификация; если не вернул — исходный текст архитектора
            materials = (data.get("materials") or "").strip() or materials
        return assemble(data["prompt"], materials, has_ref, *keep), _box(data.get("box")) if need_box else None
    except Exception as e:  # noqa: BLE001 — генерация не должна падать из-за LLM
        log.warning("DeepSeek недоступен, использую шаблонный промпт: %s", e)
        return assemble(fallback_body(scene, look), materials, has_ref, *keep), None
