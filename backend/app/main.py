import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from . import comfy
from . import inpaint as inpainting
from .auth import current_user
from .auth import router as auth_router
from .config import CORS_ORIGINS, DEFAULT_MODEL, FLUX_MODELS, log
from .db import init_db
from .models import User
from .presets import PRESETS, SCENES
from .prompting import warm_preset_analyses


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    # разборы встроенных референсов DeepSeek — в фоне, сервер стартует сразу
    warm = asyncio.create_task(warm_preset_analyses())
    yield
    warm.cancel()


app = FastAPI(title="INEGO.dev API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
app.include_router(auth_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


def check_scene(scene: str) -> None:
    if scene not in SCENES:
        raise HTTPException(400, f"Unknown scene: {scene}")


@app.get("/api/presets")
async def presets(scene: str = "exterior"):
    check_scene(scene)
    return [{"id": k, "title": v["title"]} for k, v in PRESETS[scene].items()]


@app.post("/api/img2img")
async def img2img(
    image: UploadFile = File(...),
    scene: str = Form("exterior"),
    preset: str = Form(...),
    strength: float = Form(0.7),
    reference: UploadFile | None = File(None),
    # Материалы от архитектора; пусто — берутся с изображения
    materials: str | None = Form(None, max_length=600),
    # true — зона остаётся как на исходнике, false — по референсу
    keep_sky: bool = Form(False),
    keep_bg: bool = Form(False),  # фон: всё позади главного здания
    keep_sur: bool = Form(False),  # окружение: земля, озеленение, люди, машины перед зданием
    # сид генерации; пусто — случайный. Фактический сид возвращается в заголовке X-Seed
    seed: int | None = Form(None, ge=0, lt=2**48),
    # FLUX.2 klein: klein-4b | klein-9b
    model: str = Form(DEFAULT_MODEL),
    _user: User = Depends(current_user),
):
    check_scene(scene)
    if model not in FLUX_MODELS:
        raise HTTPException(400, f"Unknown model: {model}")
    if preset not in PRESETS[scene]:
        raise HTTPException(400, f"Unknown preset: {preset}")
    if preset == "custom" and reference is None:
        raise HTTPException(400, "Custom preset requires a reference image")

    try:
        result, used_seed = await comfy.run_img2img(
            await image.read(),
            scene,
            preset,
            strength,
            await reference.read() if reference else None,
            materials,
            keep_sky,
            keep_bg,
            keep_sur,
            seed,
            model,
        )
    except comfy.ModelUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except (comfy.ComfyError, httpx.HTTPError) as e:
        log.exception("img2img failed")
        raise HTTPException(502, f"Генерация не удалась: {e}") from e
    return Response(content=result, media_type="image/jpeg", headers={"X-Seed": str(used_seed)})


@app.post("/api/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    scene: str = Form("exterior"),
    mode: str = Form(...),
    # edit — что нарисовать; remove — необязательное уточнение, что именно удалить («люди», «машина»)
    prompt: str | None = Form(None, max_length=300),
    seed: int | None = Form(None, ge=0, lt=2**48),
    model: str = Form(DEFAULT_MODEL),
    _user: User = Depends(current_user),
):
    check_scene(scene)
    if model not in FLUX_MODELS:
        raise HTTPException(400, f"Unknown model: {model}")
    if mode not in ("edit", "remove"):
        raise HTTPException(400, f"Unknown mode: {mode}")
    if mode == "edit":
        raise HTTPException(501, "Режим «Добавить / изменить» пока в разработке — доступно удаление")

    try:
        result, used_seed = await inpainting.run_remove(await image.read(), await mask.read(), seed, model, prompt)
    except inpainting.EmptyMask as e:
        raise HTTPException(400, str(e)) from e
    except comfy.ModelUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except (comfy.ComfyError, httpx.HTTPError) as e:
        log.exception("inpaint failed")
        raise HTTPException(502, f"Генерация не удалась: {e}") from e
    return Response(content=result, media_type="image/jpeg", headers={"X-Seed": str(used_seed)})
