from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from . import comfy
from .config import CORS_ORIGINS
from .presets import PRESETS

app = FastAPI(title="INEGO.dev API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/presets")
async def presets():
    return [{"id": k, "title": v["title"]} for k, v in PRESETS.items()]


@app.post("/api/img2img")
async def img2img(
    image: UploadFile = File(...),
    preset: str = Form(...),
    strength: float = Form(0.7),
    reference: UploadFile | None = File(None),
):
    if preset not in PRESETS:
        raise HTTPException(400, f"Unknown preset: {preset}")
    if preset == "custom" and reference is None:
        raise HTTPException(400, "Custom preset requires a reference image")

    result = await comfy.run_img2img(
        await image.read(),
        preset,
        strength,
        await reference.read() if reference else None,
    )
    return Response(content=result, media_type=image.content_type or "image/png")


@app.post("/api/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    mode: str = Form(...),
    prompt: str | None = Form(None),
):
    if mode not in ("edit", "remove"):
        raise HTTPException(400, f"Unknown mode: {mode}")
    if mode == "edit" and not (prompt and prompt.strip()):
        raise HTTPException(400, "Edit mode requires a prompt")

    result = await comfy.run_inpaint(await image.read(), await mask.read(), mode, prompt)
    return Response(content=result, media_type=image.content_type or "image/png")
