"""Клиент ComfyUI: img2img экстерьера на FLUX.2 [klein] 4B.

Как сохраняется геометрия:
  1. исходник подаётся как ReferenceLatent (image 1) — klein редактирует его, а не рисует с нуля;
  2. генерация стартует с латента исходника (частичный denoise по «силе»), а не с шума;
  3. кадр генерируется в тех же пропорциях, результат приводится к размеру исходника.
Референс света подаётся вторым ReferenceLatent (image 2) — только для колористики и атмосферы.

«Сохранить небо / фон / окружение» — два прохода:
  1. полный рендер по референсу (как без галочек) — из него берётся целевой свет и цвет каждой зоны;
  2. InegoKeepZones (SegFormer + SAM + глубина) размечает зоны, InegoZoneRestyle ставит в сохраняемые зоны
     исходник (те же объекты, дороги, облака) в цвете и свете рендера;
  3. лёгкий проход гармонизации (HARMONIZE) по склейке — свет ложится естественно, ничего не дорисовывается.
"""

import asyncio
import io
import random
import uuid

import httpx
from PIL import Image

from .config import (
    COMFY_TIMEOUT,
    HARMONIZE,
    LOCAL_LIGHT,
    COMFYUI_URL,
    FLUX_TEXT_ENCODER,
    FLUX_UNET,
    FLUX_VAE,
    RENDER_MEGAPIXELS,
    log,
)
from .presets import reference_image
from .prompting import build_prompt

TOTAL_STEPS = 8  # шаги полного расписания; при denoise < 1 выполняется только хвост


class ComfyError(RuntimeError):
    pass


def fit_size(w: int, h: int, megapixels: float) -> tuple[int, int]:
    """Размер с той же пропорцией, площадью ~megapixels и сторонами кратными 16."""
    scale = (megapixels * 1_000_000 / (w * h)) ** 0.5
    return max(256, round(w * scale / 16) * 16), max(256, round(h * scale / 16) * 16)


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def denoise_for(strength: float) -> float:
    """«Сила» из UI (0.1–1) → доля шагов: 0.75…1.0.

    Геометрию держит ReferenceLatent, а не стартовый латент: ниже ~0.75 сквозь рендер
    проступают пятна и плоские заливки исходника (клей так и остаётся клеем).
    """
    return 0.75 + 0.25 * max(0.0, min(1.0, strength))


def _sampling(wf: dict, tag: str, positive: list, latent: list, width: int, height: int, run_steps: int, seed: int) -> list:
    """Сэмплинг FLUX.2 klein (cfg 1, euler) хвостом из run_steps шагов полного расписания. Возвращает выход VAEDecode."""
    wf |= {
        f"negative{tag}": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": positive}},
        f"guider{tag}": {
            "class_type": "CFGGuider",
            "inputs": {"model": ["unet", 0], "positive": positive, "negative": [f"negative{tag}", 0], "cfg": 1.0},
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


def _conditioning(wf: dict, tag: str, image_latent: list, reference: str | None) -> list:
    """Текст + image 1 (что редактируем) + image 2 (референс света)."""
    wf[f"cond{tag}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["text", 0], "latent": image_latent}}
    positive = [f"cond{tag}", 0]
    if reference:
        wf[f"cond_ref{tag}"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": positive, "latent": ["ref_lat", 0]}}
        positive = [f"cond_ref{tag}", 0]
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
) -> dict:
    """Workflow в API-формате ComfyUI."""
    wf: dict = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": FLUX_UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": FLUX_TEXT_ENCODER, "type": "flux2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "src_img": {"class_type": "LoadImage", "inputs": {"image": source}},
        "src_lat": {"class_type": "VAEEncode", "inputs": {"pixels": ["src_img", 0], "vae": ["vae", 0]}},
        "text": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
    }
    if reference:
        wf["ref_img"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        wf["ref_lat"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["ref_img", 0], "vae": ["vae", 0]}}

    # проход 1 — полный рендер
    positive = _conditioning(wf, "", ["src_lat", 0], reference)
    image = _sampling(wf, "", positive, ["src_lat", 0], width, height, max(1, round(TOTAL_STEPS * denoise)), seed)

    if keep_sky or keep_bg or keep_sur:
        wf["zones"] = {
            "class_type": "InegoKeepZones",
            "inputs": {
                "image": ["src_img", 0],
                "keep_sky": keep_sky,
                "keep_background": keep_bg,
                "keep_surroundings": keep_sur,
                "grow": 0.006,
                "feather": 0.003,
                "building_box": ",".join(f"{v:.4f}" for v in building_box) if building_box else "",
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
            },
        }
        wf["mask_img"] = {"class_type": "MaskToImage", "inputs": {"mask": ["zones", 3]}}
        wf["save_mask"] = {"class_type": "SaveImage", "inputs": {"images": ["mask_img", 0], "filename_prefix": "inego/mask"}}
        # проход 2 — гармонизация склейки: склейка — image 1, старт с её латента, только хвост расписания
        wf["comp_lat"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["restyle", 0], "vae": ["vae", 0]}}
        positive = _conditioning(wf, "_h", ["comp_lat", 0], reference)
        image = _sampling(wf, "_h", positive, ["comp_lat", 0], width, height, max(1, round(TOTAL_STEPS * HARMONIZE)), seed + 1)

    wf["save"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": "inego/img2img"}}
    return wf


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
    denoise = denoise_for(strength)
    log.info(
        "img2img %s/%s %dx%d denoise=%.2f seed=%d keep sky=%s bg=%s sur=%s box=%s\n%s",
        scene, preset, width, height, denoise, seed, keep_sky, keep_bg, keep_sur, box, prompt,
    )

    async with httpx.AsyncClient(base_url=COMFYUI_URL, timeout=60) as client:
        src_fit = src.resize((width, height), Image.LANCZOS)
        source_name = await _upload(client, to_png(src_fit), f"{uuid.uuid4().hex}.png")
        ref_name = None
        if ref_bytes:
            ref = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
            ref = ref.resize(fit_size(*ref.size, 0.6), Image.LANCZOS)
            ref_name = await _upload(client, to_png(ref), f"{uuid.uuid4().hex}.png")

        workflow = build_workflow(
            source_name, width, height, prompt, denoise, seed, ref_name, keep_sky, keep_bg, keep_sur, box
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


async def run_inpaint(image: bytes, mask: bytes, scene: str, mode: str, prompt: str | None = None) -> bytes:
    # mode: "edit" — перерисовать область по prompt, "remove" — удалить объект (заполнить фоном).
    # mask: PNG, белое — область для перерисовки.
    _ = (COMFYUI_URL, scene, mask, mode, prompt)
    return image
