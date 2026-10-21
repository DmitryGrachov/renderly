import asyncio
import io
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from . import comfy
from . import inpaint as inpainting
from .auth import current_user
from .auth import router as auth_router
from .config import CORS_ORIGINS, DEEPSEEK_MODEL, DEEPSEEK_MODELS, DEFAULT_MODEL, IMG2IMG_MODELS, INPAINT_MODELS, log
from .db import init_db
from .models import User
from .presets import PRESETS, SCENES
from .prompting import llm_model, warm_preset_analyses


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


@app.get("/api/llm-models")
async def llm_models():
    """Модели DeepSeek для переключателя в шапке."""
    return {"models": DEEPSEEK_MODELS, "default": DEEPSEEK_MODEL}


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
    # FLUX.2 klein: klein-4b | klein-9b; Qwen-Image-2.1: qwen-image-2.1 (качество) | qwen-image-2.1-turbo
    model: str = Form(DEFAULT_MODEL),
    # DeepSeek: deepseek-flash | deepseek-v4-pro (переключатель в шапке)
    llm: str = Form(DEEPSEEK_MODEL),
    _user: User = Depends(current_user),
):
    check_scene(scene)
    if model not in IMG2IMG_MODELS:
        raise HTTPException(400, f"Модель {model} доступна только в Inpainting" if model in INPAINT_MODELS else f"Unknown model: {model}")
    if llm not in DEEPSEEK_MODELS:
        raise HTTPException(400, f"Unknown llm: {llm}")
    llm_model.set(llm)
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


def _parse_box(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    try:
        v = [float(x) for x in raw.split(",")]
    except ValueError:
        raise HTTPException(400, "box: ожидается x1,y1,x2,y2") from None
    if len(v) != 4 or not (0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1):
        raise HTTPException(400, "box: ожидается x1,y1,x2,y2 в долях кадра")
    return v


@app.post("/api/inpaint/plan")
async def inpaint_plan(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(..., max_length=300),
    llm: str = Form(DEEPSEEK_MODEL),
    _user: User = Depends(current_user),
):
    """План правки до генерации: тип и рамка объекта (доли кадра) — её показывают на канвасе, чтобы поправить."""
    if llm not in DEEPSEEK_MODELS:
        raise HTTPException(400, f"Unknown llm: {llm}")
    llm_model.set(llm)
    try:
        return await inpainting.plan_edit(await image.read(), await mask.read(), prompt)
    except inpainting.EmptyMask as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    scene: str = Form("exterior"),
    mode: str = Form(...),
    # edit — что нарисовать; material — какой материал положить на выделенную поверхность;
    # remove — необязательное уточнение, что именно удалить («люди», «машина»)
    prompt: str | None = Form(None, max_length=300),
    # рамка объекта «x1,y1,x2,y2» в долях кадра, поправленная на канвасе (добавить / заменить); пусто — от DeepSeek
    box: str | None = Form(None, max_length=100),
    # образец материала (вкладка «Материалы»): текстура, которую перетащили в окно
    texture: UploadFile | None = File(None),
    seed: int | None = Form(None, ge=0, lt=2**48),
    model: str = Form(DEFAULT_MODEL),
    # DeepSeek: deepseek-flash | deepseek-v4-pro (переключатель в шапке)
    llm: str = Form(DEEPSEEK_MODEL),
    _user: User = Depends(current_user),
):
    check_scene(scene)
    if model not in INPAINT_MODELS:
        raise HTTPException(400, f"Unknown model: {model}")
    if llm not in DEEPSEEK_MODELS:
        raise HTTPException(400, f"Unknown llm: {llm}")
    llm_model.set(llm)
    if mode not in ("edit", "material", "remove"):
        raise HTTPException(400, f"Unknown mode: {mode}")
    if mode == "edit" and not (prompt and prompt.strip()):
        raise HTTPException(400, "Опишите, что должно появиться в области")
    texture_bytes = await texture.read() if texture is not None and mode == "material" else None
    if texture_bytes is not None and len(texture_bytes) > 20 * 1024 * 1024:
        raise HTTPException(400, "Текстура больше 20 МБ")
    if texture_bytes:
        try:
            Image.open(io.BytesIO(texture_bytes)).verify()
        except Exception as e:  # noqa: BLE001 — не картинка
            raise HTTPException(400, "Текстура: нужен файл изображения (JPG, PNG, WebP)") from e
    if mode == "material" and not (prompt and prompt.strip()) and not texture_bytes:
        raise HTTPException(400, "Опишите материал или добавьте текстуру")

    try:
        if mode in ("edit", "material"):
            result, used_seed = await inpainting.run_edit(
                await image.read(), await mask.read(), prompt or "", seed, model, _parse_box(box),
                material=mode == "material", texture=texture_bytes or None,
            )
        else:
            result, used_seed = await inpainting.run_remove(await image.read(), await mask.read(), seed, model, prompt)
    except inpainting.EmptyMask as e:
        raise HTTPException(400, str(e)) from e
    except comfy.ModelUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except (comfy.ComfyError, httpx.HTTPError) as e:
        log.exception("inpaint failed")
        raise HTTPException(502, f"Генерация не удалась: {e}") from e
    return Response(content=result, media_type="image/jpeg", headers={"X-Seed": str(used_seed)})
