"""Img to Img, «Материалы»: образец материала + где его положить («на стены дома») — без масок, целиком на
Qwen-Image-Edit-2511.

Маски и их автоподбор по тексту не используются: 2511 сама находит поверхность по словам — перенос материала с
образца для неё родной (шаблон ComfyUI «…to the material in image 2»), остальной кадр она сохраняет как
edit-модель. Работает и по клей-/техническому рендеру, и по готовому: медиум кадра не меняется.

На каждый образец — один проход по всему кадру (~1 Мп, Lightning 8 шагов, как рендер 2511 в Img to Img):
  Picture 1 — кадр (после предыдущего образца), Picture 2 — образец;
  запрос — короткий (PROMPT, ~60 слов): что сохранить — первым, затем что становится материалом образца. Части
  запроса называет DeepSeek по кадру, образцу и словам пользователя (TARGET_SYSTEM).

Длинный запрос (официальный переписчик ~120 слов + хвост про образец и масштаб) 2511 почти не слушала: каменные
вставки под окнами становились кирпичом даже при «кроме каменной облицовки», которую переписчик честно вписал. С
коротким запросом камень остался на 4 из 4 прогонов (2 формулировки × 2 сида). Масштаб — не словами, а самим
образцом (_rescale).
"""

import hashlib
import io
import json
import random
import uuid

import httpx
from PIL import Image

from . import comfy
from .comfy import _check_model, _run, _upload, fit_size, to_png
from .config import COMFYUI_URL, QWEN_EDIT_MODEL, QWEN_EDIT_RENDER_MEGAPIXELS, log
from .inpaint import CACHE_SIZE, TEXTURE_SIDE, _described, decode_texture
from .prompting import DEEPSEEK_API_KEY, _chat, _data_url, llm_model

MAX_SAMPLES = 3  # образцов за раз: каждый — свой проход, кадр с каждым чуть мягче
MIN_SCALE, MAX_SCALE = 0.25, 3.0  # ползунок «Размер»: на ×4 вырезанный кусок образца мылится, кладка с артефактами

# DeepSeek по кадру, образцу и словам пользователя называет части запроса — коротко, по-английски
TARGET_SYSTEM = (
    "You prepare a short instruction for an image editing model that puts a building material on surfaces of an "
    "architectural image. Picture 1 is the image (a photo, a render, or a clay / technical draft render); Picture 2 "
    "is a photo sample of the new material; the user says in their own words (often Russian) where it goes. "
    "Identify in Picture 1 exactly the surfaces the user means and nothing more: 'walls' means the main wall finish "
    "only — claddings, stone or wood panels, plinth, trims, columns and other finishes stay unless the user names "
    "them. Name the target surfaces by their current look and place so they can be found (e.g. 'the white plaster "
    "walls of the house'). Then list the elements next to them that look like they could be confused with the "
    "target and must stay, each by colour, material and place (e.g. 'the beige stone cladding panels below the "
    "windows', 'the stone pillar at the entrance', 'the dark grey plinth'); at most five, the most at risk first. "
    "Answer with JSON: "
    '{"material": "<what Picture 2 shows, 2-6 words, e.g. red clay brick>", '
    '"target": "<the surfaces that get the material, 4-20 words>", '
    '"keep": ["<element that stays, 3-12 words>", ...]}'
)
# что сохранить — первым: в длинном запросе 2511 теряла исключения
PROMPT = (
    "Keep {keep} exactly as in Picture 1. Only {target} become the {material} from Picture 2, at true real-world "
    "scale, following each surface's perspective and the light of Picture 1. Window frames, doors, roof and "
    "everything else stay unchanged."
)
PROMPT_NOKEEP = (
    "Only {target} become the {material} from Picture 2, at true real-world scale, following each surface's "
    "perspective and the light of Picture 1. Window frames, doors, roof and everything else stay unchanged."
)


def _rescale(tex: Image.Image, scale: float) -> Image.Image:
    """Размер элементов на образце ×scale в тех же пикселях: мельче — образец мостится, крупнее — берётся его
    часть. Масштаб словами (сколько метров на образце) 2511 не слушает: при 0.6 и 5 м кирпич выходил одинаковым,
    а замощённый 3×3 образец дал заметно мельче."""
    if abs(scale - 1) < 0.01:
        return tex
    w, h = tex.size
    cw, ch = max(8, round(w / scale)), max(8, round(h / scale))
    canvas = Image.new("RGB", (max(cw, w), max(ch, h)))
    for x in range(0, canvas.width, w):
        for y in range(0, canvas.height, h):
            canvas.paste(tex, (x, y))
    return canvas.crop((0, 0, cw, ch)).resize((w, h), Image.LANCZOS)


def _sample_png(data: bytes, scale: float = 1.0) -> bytes:
    """Образец для модели: длинная сторона TEXTURE_SIDE, стороны кратны 32 (как в Inpainting)."""
    tex = _rescale(decode_texture(data), scale)
    tex.thumbnail((TEXTURE_SIDE, TEXTURE_SIDE), Image.LANCZOS)
    tex = tex.resize((max(32, round(tex.width / 32) * 32), max(32, round(tex.height / 32) * 32)), Image.LANCZOS)
    return to_png(tex)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


async def _parts(frame: bytes, sample: bytes, where: str) -> dict:
    """Части запроса от DeepSeek: material, target, keep. Без DeepSeek или при ошибке — слова пользователя как есть."""
    fallback = {"material": "material", "target": where, "keep": []}
    if not DEEPSEEK_API_KEY:
        return fallback
    key = hashlib.sha256(b"img2img-material-parts" + frame + sample + where.encode() + llm_model.get().encode()).hexdigest()
    if key in _described:
        return _described[key]
    content = []
    for name, data, side in (("Picture 1", frame, 1536), ("Picture 2", sample, TEXTURE_SIDE)):
        content += [{"type": "text", "text": f"{name}:"}, {"type": "image_url", "image_url": {"url": _data_url(data, side)}}]
    content.append({"type": "text", "text": f"Where the material goes: «{where}»"})
    try:
        data = json.loads(await _chat([{"role": "system", "content": TARGET_SYSTEM}, {"role": "user", "content": content}],
                                      json_mode=True))
    except Exception as e:  # noqa: BLE001 — без DeepSeek остаются слова пользователя
        log.warning("Разбор материала (Img to Img) недоступен: %s", e)
        return fallback
    clean = lambda v: " ".join(str(v or "").split()).rstrip(".")  # noqa: E731
    keep = [clean(k) for k in data.get("keep") or [] if clean(k)] if isinstance(data.get("keep"), list) else []
    parts = {
        "material": clean(data.get("material")) or "material",
        "target": clean(data.get("target")) or where,
        "keep": keep[:5],
    }
    _described[key] = parts
    if len(_described) > CACHE_SIZE:
        _described.popitem(last=False)
    return parts


async def run_materials(
    image: bytes, items: list[tuple[bytes, str, float]], seed: int | None = None
) -> tuple[bytes, int]:
    """JPEG кадра с материалами (в разрешении генерации, в пропорциях исходника) и сид.

    items — [(образец, где положить, размер элементов ×scale — ползунок «Размер»)], по порядку; каждый
    следующий проход идёт по результату предыдущего."""
    if seed is None:
        seed = random.randrange(2**48)
    src = Image.open(io.BytesIO(image)).convert("RGB")
    size = fit_size(*src.size, QWEN_EDIT_RENDER_MEGAPIXELS)
    frame = src.resize(size, Image.LANCZOS)
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        await _check_model(client, QWEN_EDIT_MODEL)
        for i, (data, where, scale) in enumerate(items):
            sample = _sample_png(data, scale)
            frame_png = to_png(frame)
            # что за материал — по исходному образцу: перемасштабированный — тот же материал, кэш не дробится
            p = await _parts(frame_png, _sample_png(data), where)
            first = lambda t: t[:1].lower() + t[1:]  # noqa: E731
            fmt = {"target": first(p["target"]), "material": first(p["material"])}
            prompt = PROMPT.format(keep=first(_join(p["keep"])), **fmt) if p["keep"] else PROMPT_NOKEEP.format(**fmt)
            prompt = prompt[:1].upper() + prompt[1:]
            log.info("материал %d/%d %dx%d seed=%d «%s» размер ×%.2f\n%s",
                     i + 1, len(items), *size, seed + i, where, scale, prompt)
            src_name = await _upload(client, frame_png, f"{uuid.uuid4().hex}.png")
            tex_name = await _upload(client, sample, f"{uuid.uuid4().hex}.png")
            wf = comfy.build_qwen2511_workflow(src_name, prompt, seed + i, None, samples=[tex_name])
            out = await _run(client, wf)
            frame = Image.open(io.BytesIO(out["save"])).convert("RGB").resize(size, Image.LANCZOS)
    # стороны кратны 16 — возвращаем точную пропорцию исходника
    result = frame.resize((frame.width, round(frame.width * src.height / src.width)), Image.LANCZOS)
    buf = io.BytesIO()
    result.save(buf, "JPEG", quality=95)
    return buf.getvalue(), seed
