"""Клиент ComfyUI: img2img экстерьера на FLUX.2 [klein] 4B / 9B.

Как сохраняется геометрия:
  1. исходник подаётся как ReferenceLatent (image 1) — klein редактирует его, а не рисует с нуля;
  2. генерация стартует с латента исходника (частичный denoise по «силе»), а не с шума;
  3. кадр генерируется в тех же пропорциях, результат приводится к размеру исходника.
Референс подаётся вторым ReferenceLatent (image 2) — эталон неба, света, атмосферы и растительности.

Небо (экстерьер с референсом) — без масок и вклеек, их швы на дымке и полупрозрачных объёмах давали артефакты:
  - без «Сохранить небо» вырезанное небо референса (рамку чистого неба даёт DeepSeek) подаётся в основной проход
    третьим ReferenceLatent (image 3) — образец неба, облачность и её плотность заданы текстом;
  - в конце InegoSkyMatch мягко подгоняет цвет неба к небу референса по полосам от зенита к горизонту.

«Сохранить небо / фон / окружение» — сохраняется расклад зоны (где газон, дорога, деревья, облака),
а сама зона перерисовывается фотореально в стилистике референса:
  1. полный рендер по референсу (как без галочек) — из него берётся целевой свет и цвет каждой зоны;
  2. InegoKeepZones (SegFormer + SAM + глубина) размечает зоны, InegoZoneRestyle ставит в сохраняемые зоны
     исходник (те же объекты, дороги, облака) в цвете и свете рендера — это стартовая склейка;
  3. перерисовка только по маске зон (ZONE_RESTYLE шагов, Differential Diffusion по мягкой маске):
     формы и расположение — со склейки, фактура — как на референсе: зелёная плоскость становится газоном,
     условные деревья — настоящими. Здание вне маски не трогается.
"""

import asyncio
import io
import random
import uuid

import httpx
from PIL import Image

from .config import (
    COMFY_TIMEOUT,
    LOCAL_LIGHT,
    COMFYUI_URL,
    DEFAULT_MODEL,
    FLUX_MODELS,
    FLUX_VAE,
    RENDER_MEGAPIXELS,
    SKY_MATCH,
    ZONE_DETAIL,
    ZONE_RESTYLE,
    log,
)
from .presets import reference_image
from .prompting import analyze_reference, build_prompt, zone_instruction

REF_MEGAPIXELS = 1.0  # размер референса (image 2) во FLUX
TOTAL_STEPS = 8  # шаги полного расписания; при denoise < 1 выполняется только хвост


class ComfyError(RuntimeError):
    pass


class ModelUnavailable(ComfyError):
    pass


def fit_size(w: int, h: int, megapixels: float) -> tuple[int, int]:
    """Размер с той же пропорцией, площадью ~megapixels и сторонами кратными 16."""
    scale = (megapixels * 1_000_000 / (w * h)) ** 0.5
    return max(256, round(w * scale / 16) * 16), max(256, round(h * scale / 16) * 16)


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _crop_sky(ref: Image.Image, box: list[float]) -> Image.Image:
    """Вырезает чистое небо референса (рамка в долях кадра) и приводит к ~0.5 Мп."""
    w, h = ref.size
    crop = ref.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)))
    return crop.resize(fit_size(*crop.size, 0.5), Image.LANCZOS)


def denoise_for(strength: float) -> float:
    """«Сила» из UI (0.1–1) → доля шагов: 0.9…1.0.

    Геометрию держит ReferenceLatent, а не стартовый латент: при низком denoise сквозь рендер
    проступают черновое небо, плоская трава и заливки исходника, и окружение с референса
    почти не переносится. С 0.9 небо, растительность и атмосфера берутся с референса, здание не меняется.
    """
    return 0.9 + 0.1 * max(0.0, min(1.0, strength))


def _sampling(
    wf: dict, tag: str, positive: list, latent: list, width: int, height: int, run_steps: int, seed: int,
    model: list | None = None,
) -> list:
    """Сэмплинг FLUX.2 klein (cfg 1, euler) хвостом из run_steps шагов полного расписания. Возвращает выход VAEDecode."""
    wf |= {
        f"negative{tag}": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": positive}},
        f"guider{tag}": {
            "class_type": "CFGGuider",
            "inputs": {"model": model or ["unet", 0], "positive": positive, "negative": [f"negative{tag}", 0], "cfg": 1.0},
        },
        f"sched{tag}": {"class_type": "Flux2Scheduler", "inputs": {"steps": TOTAL_STEPS, "width": width, "height": height}},
        f"split{tag}": {"class_type": "SplitSigmas", "inputs": {"sigmas": [f"sched{tag}", 0], "step": TOTAL_STEPS - run_steps}},
        f"noise{tag}": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        f"sample{tag}": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [f"noise{tag}", 0],
                "guider": [f"guider{tag}", 0],
                "sampler": ["sampler", 0],
                "sigmas": [f"split{tag}", 1],
                "latent_image": latent,
            },
        },
        f"decode{tag}": {"class_type": "VAEDecode", "inputs": {"samples": [f"sample{tag}", 0], "vae": ["vae", 0]}},
    }
    return [f"decode{tag}", 0]


def _conditioning(
    wf: dict, tag: str, image_latent: list, reference: str | None, text: str = "text", sky: bool = False
) -> list:
    """Текст + image 1 (что редактируем) + image 2 (референс) [+ image 3 — небо референса]."""
    wf[f"cond{tag}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": [text, 0], "latent": image_latent}}
    positive = [f"cond{tag}", 0]
    if reference:
        wf[f"cond_ref{tag}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive, "latent": ["ref_lat", 0]}}
        positive = [f"cond_ref{tag}", 0]
    if sky:
        wf[f"cond_sky{tag}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive, "latent": ["sky_ref_lat", 0]}}
        positive = [f"cond_sky{tag}", 0]
    return positive


def build_workflow(
    source: str,
    width: int,
    height: int,
    prompt: str,
    denoise: float,
    seed: int,
    reference: str | None,
    keep_sky: bool = False,
    keep_bg: bool = False,
    keep_sur: bool = False,
    building_box: list[float] | None = None,
    model: str = DEFAULT_MODEL,
    zone_prompt: str = "",
    sky_ref: str | None = None,
    sky_match: bool = False,
) -> dict:
    """Workflow в API-формате ComfyUI."""
    files = FLUX_MODELS[model]
    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": files["unet"], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": files["text_encoder"], "type": "flux2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "src_img": {"class_type": "LoadImage", "inputs": {"image": source}},
        "src_lat": {"class_type": "VAEEncode", "inputs": {"pixels": ["src_img", 0], "vae": ["vae", 0]}},
        "text": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
    }
    if reference:
        wf["ref_img"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        wf["ref_lat"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["ref_img", 0], "vae": ["vae", 0]}}
    sky = bool(sky_ref and reference and not keep_sky)
    if sky:
        # небо референса крупно — образец неба для основного прохода (image 3)
        wf["sky_ref_img"] = {"class_type": "LoadImage", "inputs": {"image": sky_ref}}
        wf["sky_ref_lat"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["sky_ref_img", 0], "vae": ["vae", 0]}}

    # проход 1 — полный рендер
    positive = _conditioning(wf, "", ["src_lat", 0], reference, sky=sky)
    image = _sampling(wf, "", positive, ["src_lat", 0], width, height, max(1, round(TOTAL_STEPS * denoise)), seed)
    # Differential Diffusion — мягкий край для всех проходов по маске
    wf["diffdiff"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": ["unet", 0], "strength": 1.0}}

    if keep_sky or keep_bg or keep_sur:
        wf["zones"] = {
            "class_type": "InegoKeepZones",
            "inputs": {
                "image": ["src_img", 0],
                "keep_sky": keep_sky,
                "keep_background": keep_bg,
                "keep_surroundings": keep_sur,
                "grow": 0.002,
                "feather": 0.006,
                "building_box": ",".join(f"{v:.4f}" for v in building_box) if building_box else "",
                "refine": True,  # край зон по контуру картинки — без светлой «обводки» у здания
            },
        }
        wf["restyle"] = {
            "class_type": "InegoZoneRestyle",
            "inputs": {
                "source": ["src_img", 0],
                "render": image,
                "sky": ["zones", 0],
                "background": ["zones", 1],
                "surroundings": ["zones", 2],
                "local_light": LOCAL_LIGHT,
                "light_radius": 0.04,
                "detail": ZONE_DETAIL,
            },
        }
        wf["mask_img"] = {"class_type": "MaskToImage", "inputs": {"mask": ["zones", 3]}}
        wf["save_mask"] = {"class_type": "SaveImage", "inputs": {"images": ["mask_img", 0], "filename_prefix": "inego/mask"}}
        # проход 2 — перерисовка только сохраняемых зон: склейка — стартовый латент, шум только под маской
        # (мягкий край через Differential Diffusion), здание вне маски не меняется. image 1 — исходник, а не склейка:
        # модель редактирования воспроизводит image 1, и со склейкой заглушки (пятна, прозрачные блоки) оставались
        wf["comp_lat"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["restyle", 0], "vae": ["vae", 0]}}
        wf["zone_lat"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["comp_lat", 0], "mask": ["zones", 3]}}
        wf["text_z"] = {"class_type": "CLIPTextEncode", "inputs": {"text": zone_prompt or prompt, "clip": ["clip", 0]}}
        positive = _conditioning(wf, "_z", ["src_lat", 0], reference, text="text_z")
        image = _sampling(
            wf, "_z", positive, ["zone_lat", 0], width, height,
            max(1, round(TOTAL_STEPS * ZONE_RESTYLE)), seed + 1, model=["diffdiff", 0],
        )

    if sky_match and reference:
        # цвет неба — как на референсе (облака и их расклад не трогаются)
        wf["sky_match"] = {
            "class_type": "InegoSkyMatch",
            "inputs": {"image": image, "reference": ["ref_img", 0], "strength": SKY_MATCH, "feather": 0.006},
        }
        image = ["sky_match", 0]

    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": "inego/img2img"}}
    return wf


async def _check_model(client: httpx.AsyncClient, model: str) -> None:
    """Понятная ошибка вместо «Value not in list», если веса модели ещё не скачаны в ComfyUI."""
    files = FLUX_MODELS[model]
    for folder, name in (("diffusion_models", files["unet"]), ("text_encoders", files["text_encoder"])):
        r = await client.get(f"/models/{folder}")
        if r.status_code == 200 and name not in r.json():
            raise ModelUnavailable(f"Модель {model} ещё не загружена в ComfyUI (нет {name})")


async def _upload(client: httpx.AsyncClient, data: bytes, name: str) -> str:
    r = await client.post("/upload/image", files={"image": (name, data, "image/png")}, data={"type": "input", "subfolder": "inego"})
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


async def _run(client: httpx.AsyncClient, workflow: dict) -> dict[str, bytes]:
    """Выполняет workflow, возвращает {id SaveImage-нода: PNG}."""
    r = await client.post("/prompt", json={"prompt": workflow, "client_id": uuid.uuid4().hex})
    if r.status_code != 200:
        raise ComfyError(f"ComfyUI отклонил workflow: {r.text[:500]}")
    prompt_id = r.json()["prompt_id"]

    loop = asyncio.get_running_loop()
    deadline = loop.time() + COMFY_TIMEOUT
    while loop.time() < deadline:
        await asyncio.sleep(1)
        h = (await client.get(f"/history/{prompt_id}")).json().get(prompt_id)
        if not h:
            continue
        status = h.get("status", {})
        if status.get("status_str") == "error":
            msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
            raise ComfyError(f"Ошибка ComfyUI: {msgs[-1][1].get('exception_message') if msgs else status}")
        if status.get("completed"):
            files = {}
            for node, out in h["outputs"].items():
                img = out["images"][0]
                v = await client.get("/view", params={"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
                v.raise_for_status()
                files[node] = v.content
            return files
    raise ComfyError("ComfyUI не ответил вовремя")


async def run_img2img(
    image: bytes,
    scene: str,
    preset: str,
    strength: float,
    reference: bytes | None = None,
    materials: str | None = None,
    keep_sky: bool = False,
    keep_bg: bool = False,
    keep_sur: bool = False,
    seed: int | None = None,
    model: str = DEFAULT_MODEL,
) -> tuple[bytes, int]:
    """JPEG финального рендера (в разрешении генерации, строго в пропорциях исходника) и использованный сид.

    seed — для повторяемости: тот же сид + те же входы дают ту же картинку, поэтому её можно
    «добивать» материалами и пресетом, не меняя всю сцену. None — случайный.
    """
    if seed is None:
        seed = random.randrange(2**48)
    src = Image.open(io.BytesIO(image)).convert("RGB")
    width, height = fit_size(*src.size, RENDER_MEGAPIXELS)

    ref_bytes = reference if preset == "custom" else reference_image(scene, preset)
    prompt, box = await build_prompt(image, scene, preset, ref_bytes, materials, keep_sky, keep_bg, keep_sur)
    zone_prompt = await zone_instruction(ref_bytes, keep_sky, keep_bg, keep_sur) if (keep_sky or keep_bg or keep_sur) else ""
    if zone_prompt:
        log.info("Перерисовка зон:\n%s", zone_prompt)
    # небо референса: рамка чистого неба из разбора DeepSeek (из кэша) — образец неба (image 3) в экстерьере
    sky_box = None
    exterior = scene == "exterior"
    if ref_bytes and exterior and not keep_sky:
        try:
            sky_box = (await analyze_reference(ref_bytes)).get("_sky_box")
        except Exception as e:  # noqa: BLE001 — без образца неба генерация всё равно идёт
            log.warning("Разбор референса для неба недоступен: %s", e)
    denoise = denoise_for(strength)
    log.info(
        "img2img %s %s/%s %dx%d denoise=%.2f seed=%d keep sky=%s bg=%s sur=%s box=%s\n%s",
        model, scene, preset, width, height, denoise, seed, keep_sky, keep_bg, keep_sur, box,
        prompt + (f"\n[image 3: небо референса, рамка {sky_box}]" if sky_box else ""),
    )

    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        await _check_model(client, model)
        src_fit = src.resize((width, height), Image.LANCZOS)
        source_name = await _upload(client, to_png(src_fit), f"{uuid.uuid4().hex}.png")
        ref_name = sky_name = None
        if ref_bytes:
            ref = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
            # референс — эталон окружения: больше пикселей → точнее небо, листва и свет
            ref = ref.resize(fit_size(*ref.size, REF_MEGAPIXELS), Image.LANCZOS)
            ref_name = await _upload(client, to_png(ref), f"{uuid.uuid4().hex}.png")
            if sky_box:
                sky_name = await _upload(client, to_png(_crop_sky(ref, sky_box)), f"{uuid.uuid4().hex}.png")

        workflow = build_workflow(
            source_name, width, height, prompt, denoise, seed, ref_name, keep_sky, keep_bg, keep_sur, box, model,
            zone_prompt, sky_name, sky_match=exterior,
        )
        out = await _run(client, workflow)

    result = Image.open(io.BytesIO(out["save"])).convert("RGB")
    # стороны кратны 16 — пропорция чуть плывёт, возвращаем точную пропорцию исходника
    exact = (result.width, round(result.width * src.height / src.width))
    if result.size != exact:
        result = result.resize(exact, Image.LANCZOS)
    buf = io.BytesIO()
    result.save(buf, "JPEG", quality=95)
    return buf.getvalue(), seed

