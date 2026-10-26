"""Img to Img, «Материалы»: образец материала + где его положить («на стены дома») — без масок, на модели из
селектора в шапке.

Маски и их автоподбор по тексту не используются: модель сама находит поверхность по словам. Работает и по клей-/
техническому рендеру, и по готовому.

На каждый образец — один проход по всему кадру: картинка 1 — кадр (после предыдущего образца), картинка 2 — образец;
запрос — короткий (PROMPT, ~60 слов): что сохранить — первым, затем что становится материалом образца. Части запроса
называет DeepSeek по кадру, образцу и словам пользователя (TARGET_SYSTEM), картинки в нём — как их зовёт модель.
Свой пайплайн у каждой модели, как в рендере Img to Img:
  - Qwen-Image-Edit-2511 — TextEncodeQwenImageEditPlus (Picture 1, Picture 2), Lightning 8 шагов, ~1 Мп; перенос
    материала с образца для неё родной (шаблон ComfyUI «…to the material in image 2»);
  - Qwen-Image-2.1 — TextEncodeQwenImage21 (<image1>, <image2>), Turbo 8 шагов с чистого шума, ~1.5 Мп (со старта
    от латента кадра 2.1 почти копирует вход);
  - FLUX.2 klein 4B / 9B — ReferenceLatent-цепочка (image 1, image 2), 8 шагов полным шумом от латента кадра, ~1.5 Мп.
Кадр после каждой модели возвращается на место исходника (InegoAlign).

Длинный запрос (официальный переписчик ~120 слов + хвост про образец и масштаб) 2511 почти не слушала: каменные
вставки под окнами становились кирпичом даже при «кроме каменной облицовки», которую переписчик честно вписал. С
коротким запросом камень остался на 4 из 4 прогонов (2 формулировки × 2 сида). Масштаб — не словами, а самим
образцом (_rescale).
"""

import asyncio
import hashlib
import io
import json
import random
import uuid

import cv2
import httpx
import numpy as np
from PIL import Image

from . import comfy
from .comfy import _check_model, _run, _upload, fit_size, to_png
from .config import (
    COMFYUI_URL, FLUX_MODELS, FLUX_VAE, QWEN21, QWEN21_MODELS, QWEN21_RENDER_MEGAPIXELS, QWEN_EDIT_MODEL,
    QWEN_EDIT_RENDER_MEGAPIXELS, RENDER_MEGAPIXELS, log,
)
from .inpaint import CACHE_SIZE, TEXTURE_SIDE, _described, decode_texture
from .prompting import DEEPSEEK_API_KEY, _chat, _data_url, llm_model

MAX_SAMPLES = 3  # образцов за раз: каждый — свой проход, кадр с каждым чуть мягче
MIN_SCALE, MAX_SCALE = 0.25, 3.0  # ползунок «Размер»: на ×4 вырезанный кусок образца мылится, кладка с артефактами
REGISTER_SIDE = 1536  # подгонка к исходнику — по кадру не больше этого
REGISTER_MAX = 0.03  # подгонка больше 3% (масштаб, сдвиг) — ECC промахнулся, кадр остаётся как есть

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
# что сохранить — первым: в длинном запросе 2511 теряла исключения. {a} — кадр, {b} — образец, как их зовёт модель
# тени — отдельно: газон под грунт выходил ровно освещённым, тени деревьев пропадали, и грунт выглядел наклеенным
PROMPT = (
    "Keep {keep} exactly as in {a}. Only {target} become the {material} from {b}, at true real-world scale, "
    "following each surface's perspective and the light of {a}; every shadow that falls on them stays where it "
    "is. Window frames, doors, roof and everything else stay unchanged."
)
PROMPT_NOKEEP = (
    "Only {target} become the {material} from {b}, at true real-world scale, following each surface's perspective "
    "and the light of {a}; every shadow that falls on them stays where it is. Window frames, doors, roof and "
    "everything else stay unchanged."
)
# как модель зовёт картинки в запросе
PICTURES = {"qwen2511": ("Picture 1", "Picture 2"), "qwen21": ("<image1>", "<image2>"), "flux": ("image 1", "image 2")}


def _family(model: str) -> str:
    return "qwen2511" if model == QWEN_EDIT_MODEL else "qwen21" if model in QWEN21_MODELS else "flux"


def _frame_size(family: str, size: tuple[int, int]) -> tuple[int, int]:
    """Размер кадра генерации — как в рендере Img to Img этой моделью."""
    if family == "qwen2511":
        return fit_size(*size, QWEN_EDIT_RENDER_MEGAPIXELS)
    if family == "qwen21":
        return fit_size(*size, QWEN21_RENDER_MEGAPIXELS, step=32)
    return fit_size(*size, RENDER_MEGAPIXELS)


def _finish(wf: dict, image: list) -> dict:
    """Кадр — на место исходника (модель чуть сдвигает и масштабирует его), сохранение."""
    wf["align"] = {
        "class_type": "InegoAlign",
        "inputs": {"image": image, "source": ["src_img", 0], "max_scale": 0.08, "max_shift": 0.08},
    }
    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": ["align", 0], "filename_prefix": "inego/materials"}}
    return wf


def _qwen21_workflow(source: str, sample: str, prompt: str, seed: int) -> dict:
    """Qwen-Image-2.1: <image1> — кадр, <image2> — образец; Turbo 8 шагов с чистого шума в размере <image1>."""
    q = QWEN21
    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": q["unet"], "weight_dtype": "default"}},
        # KV-кэш образов — в RAM: в 16 ГБ видеопамяти рядом с энкодером и моделью он не помещается (OOM)
        "cache": {"class_type": "QwenImage21Cache", "inputs": {"model": ["unet", 0], "device": "cpu", "dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": q["text_encoder"], "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": q["vae"]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "src_img": {"class_type": "LoadImage", "inputs": {"image": source}},
        "tex_img": {"class_type": "LoadImage", "inputs": {"image": sample}},
        "text": {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {
                "clip": ["clip", 0], "prompt": prompt, "negative_prompt": "", "resolution": 0, "vae": ["vae", 0],
                "images.image_1": ["src_img", 0], "images.image_2": ["tex_img", 0],
            },
        },
    }
    return _finish(wf, comfy._q21_sampling(wf, "", "text", ["cache", 0], ["text", 2], 1.0, seed))


def _flux_workflow(source: str, sample: str, prompt: str, seed: int, model: str, size: tuple[int, int]) -> dict:
    """FLUX.2 klein: image 1 — кадр, image 2 — образец (ReferenceLatent-цепочкой); 8 шагов полным шумом."""
    files = FLUX_MODELS[model]
    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": files["unet"], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": files["text_encoder"], "type": "flux2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "src_img": {"class_type": "LoadImage", "inputs": {"image": source}},
        "src_lat": {"class_type": "VAEEncode", "inputs": {"pixels": ["src_img", 0], "vae": ["vae", 0]}},
        "tex_img": {"class_type": "LoadImage", "inputs": {"image": sample}},
        "tex_lat": {"class_type": "VAEEncode", "inputs": {"pixels": ["tex_img", 0], "vae": ["vae", 0]}},
        "text": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "cond": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["text", 0], "latent": ["src_lat", 0]}},
        "cond_tex": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["cond", 0], "latent": ["tex_lat", 0]}},
    }
    return _finish(wf, comfy._sampling(wf, "", ["cond_tex", 0], ["src_lat", 0], *size, comfy.TOTAL_STEPS, seed))


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


def _edges(gray: np.ndarray) -> np.ndarray:
    """Сила краёв: сравнивать кадры по ней, а не по яркости — материал и свет поменялись, края остались."""
    g = cv2.GaussianBlur(gray, (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def _register(frame: Image.Image, src: Image.Image) -> Image.Image:
    """Результат — в размере исходника и точно на нём (ECC по краям, сдвиг и растяжение по осям).

    InegoAlign ставит кадр с точностью до пикселя по сдвигу, но масштаб у него один на обе оси: у 2511 кадр
    оставался сжат по вертикали на 0.3% (~3 px на высоту). На тестовом доме ECC снизил медиану смещения краёв
    с 1.45 до 0.42 px (2511), с 0.97 до 0.41 (klein 9B); остаток — местный, модель по-своему рисует края."""
    big = frame.resize(src.size, Image.BICUBIC)
    k = min(1.0, REGISTER_SIDE / max(src.size))
    small = (round(src.width * k), round(src.height * k))
    a, b = (_edges(np.asarray(im.convert("L").resize(small, Image.BILINEAR), np.float32)) for im in (src, big))
    m = np.eye(2, 3, dtype=np.float32)
    try:
        _, m = cv2.findTransformECC(a, b, m, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error as e:
        log.warning("Подгонка к исходнику не сошлась: %s", e)
        return big
    m[:, 2] /= k  # сдвиг — в пиксели исходника
    if np.abs(m[:, :2] - np.eye(2)).max() > REGISTER_MAX or np.abs(m[:, 2]).max() > REGISTER_MAX * max(src.size):
        log.warning("Подгонка к исходнику отброшена — слишком большая: %s", np.round(m, 4).tolist())
        return big
    log.info("подгонка к исходнику: %s", np.round(m, 4).tolist())
    out = cv2.warpAffine(
        np.asarray(big), m, src.size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(out)


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
    image: bytes, items: list[tuple[bytes, str, float]], seed: int | None = None, model: str = QWEN_EDIT_MODEL
) -> tuple[bytes, int]:
    """JPEG кадра с материалами (в разрешении генерации, в пропорциях исходника) и сид.

    items — [(образец, где положить, размер элементов ×scale — ползунок «Размер»)], по порядку; каждый
    следующий проход идёт по результату предыдущего."""
    if seed is None:
        seed = random.randrange(2**48)
    src = Image.open(io.BytesIO(image)).convert("RGB")
    family = _family(model)
    size = _frame_size(family, src.size)
    a, b = PICTURES[family]
    frame = src.resize(size, Image.LANCZOS)
    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        await _check_model(client, model)  # веса на месте; прежняя модель другого семейства выгружается
        for i, (data, where, scale) in enumerate(items):
            sample = _sample_png(data, scale)
            frame_png = to_png(frame)
            # что за материал — по исходному образцу: перемасштабированный — тот же материал, кэш не дробится
            p = await _parts(frame_png, _sample_png(data), where)
            first = lambda t: t[:1].lower() + t[1:]  # noqa: E731
            fmt = {"target": first(p["target"]), "material": first(p["material"]), "a": a, "b": b}
            prompt = PROMPT.format(keep=first(_join(p["keep"])), **fmt) if p["keep"] else PROMPT_NOKEEP.format(**fmt)
            prompt = prompt[:1].upper() + prompt[1:]
            log.info("материал %s %d/%d %dx%d seed=%d «%s» размер ×%.2f\n%s",
                     model, i + 1, len(items), *size, seed + i, where, scale, prompt)
            src_name = await _upload(client, frame_png, f"{uuid.uuid4().hex}.png")
            tex_name = await _upload(client, sample, f"{uuid.uuid4().hex}.png")
            if family == "qwen2511":
                wf = comfy.build_qwen2511_workflow(src_name, prompt, seed + i, None, samples=[tex_name])
            elif family == "qwen21":
                wf = _qwen21_workflow(src_name, tex_name, prompt, seed + i)
            else:
                wf = _flux_workflow(src_name, tex_name, prompt, seed + i, model, size)
            out = await _run(client, wf)
            frame = Image.open(io.BytesIO(out["save"])).convert("RGB").resize(size, Image.LANCZOS)
    # в размер исходника и точно на него: кисть отката проявляет исходник пиксель в пиксель
    result = await asyncio.to_thread(_register, frame, src)
    buf = io.BytesIO()
    result.save(buf, "JPEG", quality=95)
    return buf.getvalue(), seed
