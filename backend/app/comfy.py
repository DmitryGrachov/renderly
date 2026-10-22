"""Клиент ComfyUI: img2img экстерьера на FLUX.2 [klein] 4B / 9B и Qwen-Image-2.1 Turbo.

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

Qwen-Image-2.1 Turbo — та же логика в родном для неё виде (build_qwen21_workflow):
  - исходник — картинка <image1> в TextEncodeQwenImage21 (энкодер Qwen3-VL читает её и вставляет латентом
    в последовательность) — вместо ReferenceLatent. Референс картинкой не подаётся: на слабом холсте (клей-рендер)
    2.1 выдавала вместо правки дом с референса; окружение и свет референса идут текстом — разбором DeepSeek;
  - кадр ~1.5 Мп со сторонами кратными 32; 8 шагов Turbo по фиксированному расписанию, cfg 1;
  - старт с чистого шума (пустой латент энкодера в размере <image1>): со старта от латента исходника
    (σ ≤ 0.98) 2.1 почти копирует черновик — меняется только цвет. Цена — небольшое перекадрирование
    (здание смещается на 2–3% кадра), его снимает InegoAlign: масштаб и сдвиг по контурам до исходника.
    «Сила» здесь не применяется;
  - зоны — тем же InegoKeepZones / InegoZoneRestyle, перерисовка по маске — Differential Diffusion поверх
    QwenImage21Cache и хвост расписания; небо — тот же InegoSkyMatch.

Qwen-Image-Edit-2511 — в родном для неё виде (build_qwen2511_workflow), как в шаблоне ComfyUI:
  - картинки — в TextEncodeQwenImageEditPlus: Picture 1 — исходник, Picture 2 — вырезанное небо референса.
    Сам референс картинкой не подаётся (QWEN_EDIT_REF_IMAGE): с ним 2511 брала с референса и дом — цоколь,
    колонны, крыльцо. Свет и окружение референса описывает текстом официальный переписчик (DeepSeek видит референс);
  - Lightning LoRA, 4 шага, cfg 1, сдвиг 3.1, CFGNorm, index_timestep_zero; кадр ~1 Мп, полный шум от латента
    исходника; InegoAlign возвращает кадр на место; зоны и небо — как у FLUX и 2.1. «Сила» не применяется.
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
    QWEN21,
    QWEN21_BASE_STEPS,
    QWEN21_REFINE,
    QWEN21_MODELS,
    QWEN21_RENDER_MEGAPIXELS,
    QWEN21_ZONE_STEPS,
    QWEN_EDIT,
    QWEN_EDIT_MODEL,
    QWEN_EDIT_REF_IMAGE,
    QWEN_EDIT_RENDER_MEGAPIXELS,
    QWEN_EDIT_ZONE_DENOISE,
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


def fit_size(w: int, h: int, megapixels: float, step: int = 16) -> tuple[int, int]:
    """Размер с той же пропорцией, площадью ~megapixels и сторонами кратными step (Qwen-Image-2.1 — 32)."""
    scale = (megapixels * 1_000_000 / (w * h)) ** 0.5
    return max(256, round(w * scale / step) * step), max(256, round(h * scale / step) * step)


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


def _zones(wf: dict, image: list, keep_sky: bool, keep_bg: bool, keep_sur: bool, building_box: list[float] | None) -> None:
    """Разметка сохраняемых зон и стартовая склейка (исходник в цвете и свете рендера): restyle — склейка,
    zones.3 — маска перерисовки."""
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


def _sky_match(wf: dict, image: list) -> list:
    """Цвет неба — как на референсе (облака и их расклад не трогаются)."""
    wf["sky_match"] = {
        "class_type": "InegoSkyMatch",
        "inputs": {"image": image, "reference": ["ref_img", 0], "strength": SKY_MATCH, "feather": 0.006},
    }
    return ["sky_match", 0]


def _q21_sigmas(denoise: float) -> str:
    """Расписание Turbo (8 шагов); denoise < 1 — его хвост из round(8 * denoise) шагов."""
    sig = QWEN21["sigmas"]
    sig = sig[len(sig) - 1 - max(1, round((len(sig) - 1) * denoise)):]
    return ", ".join(f"{v:g}" for v in sig)


def _q21_sampling(
    wf: dict, tag: str, text: str, model: list, latent: list, denoise: float, seed: int, base_steps: int = 0
) -> list:
    """Сэмплинг Qwen-Image-2.1: cfg 1 (негатив не используется), euler. Turbo — фиксированное расписание на 8 шагов;
    base_steps — базовая модель: планировщик simple (сдвиг 0.69 из настроек модели), как в шаблонах ComfyUI."""
    if base_steps:
        sigmas = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": base_steps, "denoise": denoise}}
    else:
        sigmas = {"class_type": "ManualSigmas", "inputs": {"sigmas": _q21_sigmas(denoise)}}
    wf |= {
        f"guider{tag}": {
            "class_type": "CFGGuider",
            "inputs": {"model": model, "positive": [text, 0], "negative": [text, 1], "cfg": 1.0},
        },
        f"sigmas{tag}": sigmas,
        f"noise{tag}": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        f"sample{tag}": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [f"noise{tag}", 0], "guider": [f"guider{tag}", 0], "sampler": ["sampler", 0],
                "sigmas": [f"sigmas{tag}", 0], "latent_image": latent,
            },
        },
        f"decode{tag}": {"class_type": "VAEDecode", "inputs": {"samples": [f"sample{tag}", 0], "vae": ["vae", 0]}},
        # VAE 2.1 декодирует RGBA (у модели родной альфа-канал) — ноды зон и неба ждут RGB
        f"rgb{tag}": {"class_type": "SplitImageWithAlpha", "inputs": {"image": [f"decode{tag}", 0]}},
    }
    return [f"rgb{tag}", 0]


def build_qwen21_workflow(
    source: str,
    prompt: str,
    seed: int,
    reference: str | None,
    keep_sky: bool = False,
    keep_bg: bool = False,
    keep_sur: bool = False,
    building_box: list[float] | None = None,
    zone_prompt: str = "",
    sky_match: bool = False,
    refine: bool = False,
) -> dict:
    """Img to Img на Qwen-Image-2.1: <image1> — исходник (холст); reference — только для InegoSkyMatch, модели
    не подаётся (см. docstring модуля). Рендер — Turbo на 8 шагов;
    refine — финальная доводка фактуры базовой моделью (QWEN21_REFINE шума от готового кадра)."""
    q = QWEN21
    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": q["unet"], "weight_dtype": "default"}},
        # KV-кэш образов — в RAM: в 16 ГБ видеопамяти рядом с энкодером и моделью он не помещается (OOM)
        "cache": {"class_type": "QwenImage21Cache", "inputs": {"model": ["unet", 0], "device": "cpu", "dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": q["text_encoder"], "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": q["vae"]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "src_img": {"class_type": "LoadImage", "inputs": {"image": source}},
    }
    images = {"images.image_1": ["src_img", 0]}
    if reference:
        wf["ref_img"] = {"class_type": "LoadImage", "inputs": {"image": reference}}

    def encode(name: str, text: str, imgs: dict) -> str:
        # resolution 0 — кадр по <image1> (стороны уже кратны 32)
        wf[name] = {
            "class_type": "TextEncodeQwenImage21",
            "inputs": {"clip": ["clip", 0], "prompt": text, "negative_prompt": "", "resolution": 0, "vae": ["vae", 0], **imgs},
        }
        return name

    # проход 1 — полный рендер с чистого шума, затем кадр возвращается на место исходника
    image = _q21_sampling(wf, "", encode("text", prompt, images), ["cache", 0], ["text", 2], 1.0, seed)
    wf["align"] = {
        "class_type": "InegoAlign",
        "inputs": {"image": image, "source": ["src_img", 0], "max_scale": 0.08, "max_shift": 0.08},
    }
    image = ["align", 0]

    if keep_sky or keep_bg or keep_sur:
        _zones(wf, image, keep_sky, keep_bg, keep_sur, building_box)
        # проход 2 — только сохраняемые зоны: склейка — стартовый латент, шум под маской (Differential Diffusion),
        # <image1> — исходник (держит расклад).
        # Шагов — QWEN21_ZONE_STEPS из 8: на 7 (как ZONE_RESTYLE у FLUX) 2.1 подменяла деревья и добавляла кусты с референса
        wf["diffdiff"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": ["cache", 0], "strength": 1.0}}
        wf["comp_lat"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["restyle", 0], "vae": ["vae", 0]}}
        wf["zone_lat"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["comp_lat", 0], "mask": ["zones", 3]}}
        image = _q21_sampling(
            wf, "_z", encode("text_z", zone_prompt or prompt, images), ["diffdiff", 0], ["zone_lat", 0],
            QWEN21_ZONE_STEPS / 8, seed + 1,
        )

    if refine:
        # доводка: базовая модель от готового кадра (он же <image1>) с частичным шумом — композиция и здание
        # остаются, листва и трава из мазков дистиллята прорисовываются отдельными листьями и травинками
        wf["unet_r"] = {"class_type": "UNETLoader", "inputs": {"unet_name": q["unet_base"], "weight_dtype": "default"}}
        wf["cache_r"] = {"class_type": "QwenImage21Cache", "inputs": {"model": ["unet_r", 0], "device": "cpu", "dtype": "default"}}
        wf["lat_r"] = {"class_type": "VAEEncode", "inputs": {"pixels": image, "vae": ["vae", 0]}}
        image = _q21_sampling(
            wf, "_r", encode("text_r", prompt, {"images.image_1": image}), ["cache_r", 0], ["lat_r", 0],
            QWEN21_REFINE, seed + 2, QWEN21_BASE_STEPS,
        )

    if sky_match and reference:
        image = _sky_match(wf, image)
    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": "inego/img2img"}}
    return wf


def _q2511_sampling(wf: dict, tag: str, text: str, model: list, latent: list, denoise: float, seed: int) -> list:
    """Сэмплинг Qwen-Image-Edit-2511: Lightning 4 шага, euler/simple, cfg 1 (негатив — пустой текст)."""
    wf[f"sample{tag}"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model, "positive": [f"ref_pos{text}", 0], "negative": [f"ref_neg{text}", 0], "latent_image": latent,
            "seed": seed, "steps": QWEN_EDIT["steps"], "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "denoise": denoise,
        },
    }
    wf[f"decode{tag}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"sample{tag}", 0], "vae": ["vae", 0]}}
    return [f"decode{tag}", 0]


def build_qwen2511_workflow(
    source: str,
    prompt: str,
    seed: int,
    reference: str | None,
    keep_sky: bool = False,
    keep_bg: bool = False,
    keep_sur: bool = False,
    building_box: list[float] | None = None,
    zone_prompt: str = "",
    sky_ref: str | None = None,
    sky_match: bool = False,
) -> dict:
    """Img to Img на Qwen-Image-Edit-2511 (см. docstring модуля): Picture 1 — исходник, затем небо референса
    (и сам референс перед ним при QWEN_EDIT_REF_IMAGE); референс в workflow — для InegoSkyMatch."""
    q = QWEN_EDIT
    wf: dict = {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": q["unet"]}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": q["lora"], "strength_model": 1.0}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["lora", 0], "shift": q["shift"]}},
        "cfgnorm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": q["text_encoder"], "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": q["vae"]}},
        "src_img": {"class_type": "LoadImage", "inputs": {"image": source}},
        "src_lat": {"class_type": "VAEEncode", "inputs": {"pixels": ["src_img", 0], "vae": ["vae", 0]}},
    }
    images = {"image1": ["src_img", 0]}
    if reference:
        wf["ref_img"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        if QWEN_EDIT_REF_IMAGE:
            images["image2"] = ["ref_img", 0]
        if sky_ref and not keep_sky:
            wf["sky_ref_img"] = {"class_type": "LoadImage", "inputs": {"image": sky_ref}}
            images[f"image{len(images) + 1}"] = ["sky_ref_img", 0]

    def encode(tag: str, text: str) -> str:
        for kind, t in (("pos", text), ("neg", "")):
            wf[f"text_{kind}{tag}"] = {
                "class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": ["clip", 0], "prompt": t, "vae": ["vae", 0], **images},
            }
            # GGUF не от Comfy-Org: способ подачи референсов — явно, как в шаблоне ComfyUI для 2511
            wf[f"ref_{kind}{tag}"] = {
                "class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": [f"text_{kind}{tag}", 0], "reference_latents_method": "index_timestep_zero"},
            }
        return tag

    # проход 1 — полный рендер (полный шум от латента исходника), кадр возвращается на место исходника
    image = _q2511_sampling(wf, "", encode("", prompt), ["cfgnorm", 0], ["src_lat", 0], 1.0, seed)
    wf["align"] = {
        "class_type": "InegoAlign",
        "inputs": {"image": image, "source": ["src_img", 0], "max_scale": 0.08, "max_shift": 0.08},
    }
    image = ["align", 0]

    if keep_sky or keep_bg or keep_sur:
        _zones(wf, image, keep_sky, keep_bg, keep_sur, building_box)
        # проход 2 — только сохраняемые зоны: склейка — стартовый латент, шум под маской (Differential Diffusion)
        wf["diffdiff"] = {"class_type": "DifferentialDiffusion", "inputs": {"model": ["cfgnorm", 0], "strength": 1.0}}
        wf["comp_lat"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["restyle", 0], "vae": ["vae", 0]}}
        wf["zone_lat"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["comp_lat", 0], "mask": ["zones", 3]}}
        image = _q2511_sampling(
            wf, "_z", encode("_z", zone_prompt or prompt), ["diffdiff", 0], ["zone_lat", 0], QWEN_EDIT_ZONE_DENOISE, seed + 1,
        )

    if sky_match and reference:
        image = _sky_match(wf, image)
    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": "inego/img2img"}}
    return wf


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
    qwen21_refine: bool = False,
) -> dict:
    """Workflow в API-формате ComfyUI."""
    if model == QWEN_EDIT_MODEL:
        return build_qwen2511_workflow(
            source, prompt, seed, reference, keep_sky, keep_bg, keep_sur, building_box, zone_prompt, sky_ref, sky_match,
        )
    if model in QWEN21_MODELS:
        return build_qwen21_workflow(
            source, prompt, seed, reference, keep_sky, keep_bg, keep_sur, building_box, zone_prompt, sky_match,
            refine=qwen21_refine,
        )
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
        _zones(wf, image, keep_sky, keep_bg, keep_sur, building_box)
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
        image = _sky_match(wf, image)

    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": "inego/img2img"}}
    return wf


_last_family: str | None = None


async def _use_family(client: httpx.AsyncClient, family: str) -> None:
    """При переключении между FLUX klein и Qwen выгружает прежние веса из видеопамяти и RAM.

    Вместе они не помещаются (Qwen GGUF ~13 ГБ + энкодер ~9 ГБ, klein 9B ~17 ГБ при 25 ГБ RAM у Docker):
    без выгрузки ComfyUI держит обе в кэше и упирается в память.
    """
    global _last_family
    if _last_family and _last_family != family:
        await client.post("/free", json={"unload_models": True, "free_memory": True})
        log.info("ComfyUI: выгружены модели %s перед %s", _last_family, family)
    _last_family = family


async def _check_model(client: httpx.AsyncClient, model: str) -> None:
    """Понятная ошибка вместо «Value not in list», если веса модели ещё не скачаны в ComfyUI."""
    if model == QWEN_EDIT_MODEL:
        q = QWEN_EDIT
        need = (("unet_gguf", q["unet"]), ("text_encoders", q["text_encoder"]), ("vae", q["vae"]), ("loras", q["lora"]))
    elif model in QWEN21_MODELS:
        q = QWEN21
        need = (("diffusion_models", q["unet"]), ("text_encoders", q["text_encoder"]), ("vae", q["vae"]))
    else:
        files = FLUX_MODELS[model]
        need = (("diffusion_models", files["unet"]), ("text_encoders", files["text_encoder"]))
    for folder, name in need:
        r = await client.get(f"/models/{folder}")
        if r.status_code != 200:
            if folder == "unet_gguf":
                raise ModelUnavailable("В ComfyUI нет загрузчика GGUF (ComfyUI-GGUF) — пересоберите образ comfyui")
            continue
        if name not in r.json():
            raise ModelUnavailable(f"Модель {model} ещё не загружена в ComfyUI (нет {name})")
    await _use_family(client, "qwen-2511" if model == QWEN_EDIT_MODEL else "qwen-2.1" if model in QWEN21_MODELS else "flux")


async def _upload(client: httpx.AsyncClient, data: bytes, name: str) -> str:
    r = await client.post("/upload/image", files={"image": (name, data, "image/png")}, data={"type": "input", "subfolder": "inego"})
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


async def _run(client: httpx.AsyncClient, workflow: dict) -> dict[str, bytes]:
    """Выполняет workflow, возвращает {id SaveImage-нода: PNG}.

    Нехватка видеопамяти — один повтор: после OOM ComfyUI выгружает модели, и тот же workflow обычно проходит
    (Qwen-Image-2.1 на 2 Мп с длинным запросом впритык помещается в 16 ГБ).
    """
    try:
        return await _run_once(client, workflow)
    except ComfyError as e:
        if "out of memory" not in str(e).lower():
            raise
        log.warning("ComfyUI: нехватка видеопамяти — освобождаю память и повторяю")
        await client.post("/free", json={"unload_models": True, "free_memory": True})
        return await _run_once(client, workflow)


async def _run_once(client: httpx.AsyncClient, workflow: dict) -> dict[str, bytes]:
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
    family = "qwen21" if model in QWEN21_MODELS else "qwen2511" if model == QWEN_EDIT_MODEL else "flux"
    if family == "qwen21":
        width, height = fit_size(*src.size, QWEN21_RENDER_MEGAPIXELS, step=32)
    elif family == "qwen2511":
        width, height = fit_size(*src.size, QWEN_EDIT_RENDER_MEGAPIXELS)
    else:
        width, height = fit_size(*src.size, RENDER_MEGAPIXELS)

    ref_bytes = reference if preset == "custom" else reference_image(scene, preset)
    prompt, box = await build_prompt(image, scene, preset, ref_bytes, materials, keep_sky, keep_bg, keep_sur, family)
    zone_prompt = (
        await zone_instruction(ref_bytes, keep_sky, keep_bg, keep_sur, family) if (keep_sky or keep_bg or keep_sur) else ""
    )
    if zone_prompt:
        log.info("Перерисовка зон:\n%s", zone_prompt)
    # небо референса: рамка чистого неба из разбора DeepSeek (из кэша) — образец неба (image 3) в экстерьере
    sky_box = None
    exterior = scene == "exterior"
    if ref_bytes and exterior and not keep_sky and family in ("flux", "qwen2511"):
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
        # Qwen-Image-2.1 — везде Turbo: базовая модель уводила кадр (камера, газон в луг, новые дома на фоне)
        qwen21_refine = False
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
            zone_prompt, sky_name, sky_match=exterior and SKY_MATCH > 0, qwen21_refine=qwen21_refine,
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

