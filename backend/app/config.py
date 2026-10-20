import logging
import os
import secrets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("inego")

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://inego:inego@localhost:5432/inego")

JWT_SECRET = os.getenv("JWT_SECRET") or ""
if not JWT_SECRET:
    JWT_SECRET = secrets.token_urlsafe(48)
    log.warning("JWT_SECRET не задан — сгенерирован временный, сессии сбросятся при перезапуске")
JWT_TTL_DAYS = int(os.getenv("JWT_TTL_DAYS", "14"))

# true только за HTTPS, иначе браузер не сохранит cookie
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"

# DeepSeek (OpenAI-совместимый API с поддержкой изображений). Без ключа — промпт собирается из пресета.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
# модели DeepSeek, между которыми можно переключаться из шапки сайта (поле llm в запросах)
DEEPSEEK_MODELS = [m.strip() for m in os.getenv("DEEPSEEK_MODELS", "deepseek-flash,deepseek-v4-pro").split(",") if m.strip()]

# FLUX.2 [klein] (distilled) в ComfyUI: модель выбирается в запросе, VAE общий.
# 9B закрыта лицензией BFL — ComfyUI скачает её только при заданном HF_TOKEN.
FLUX_MODELS = {
    "klein-4b": {
        "unet": os.getenv("FLUX_4B_UNET", "flux-2-klein-4b-fp8.safetensors"),
        "text_encoder": os.getenv("FLUX_4B_TEXT_ENCODER", "qwen_3_4b.safetensors"),
    },
    "klein-9b": {
        "unet": os.getenv("FLUX_9B_UNET", "flux-2-klein-9b-fp8.safetensors"),
        "text_encoder": os.getenv("FLUX_9B_TEXT_ENCODER", "qwen_3_8b_fp8mixed.safetensors"),
    },
}
DEFAULT_MODEL = "klein-4b"
# Qwen-Image-Edit-2511 (Apache 2.0, 20B) — только Inpainting: GGUF Q4_K_M + Lightning LoRA на 4 шага
QWEN_EDIT_MODEL = "qwen-edit-2511"
QWEN_EDIT = {
    "unet": os.getenv("QWEN_EDIT_UNET", "qwen-image-edit-2511-Q4_K_M.gguf"),
    "text_encoder": os.getenv("QWEN_EDIT_TEXT_ENCODER", "qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    "vae": os.getenv("QWEN_EDIT_VAE", "qwen_image_vae.safetensors"),
    "lora": os.getenv("QWEN_EDIT_LORA", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"),
    "steps": int(os.getenv("QWEN_EDIT_STEPS", "4")),
    "shift": float(os.getenv("QWEN_EDIT_SHIFT", "3.1")),
}
# Qwen-Image-2.1 Turbo (7B, сентябрь 2026; Qwen Research License — только некоммерческое использование) — только Inpainting: int8, 8 шагов по фиксированному расписанию
QWEN21_MODEL = "qwen-image-2.1"
QWEN21 = {
    "unet": os.getenv("QWEN21_UNET", "qwen_image_2.1_turbo_int8_convrot.safetensors"),
    "text_encoder": os.getenv("QWEN21_TEXT_ENCODER", "qwen3vl_8b_int8_convrot.safetensors"),
    "vae": os.getenv("QWEN21_VAE", "qwen_image_2.1_vae_bf16.safetensors"),
    # расписание Turbo из официального шаблона ComfyUI (8 шагов, cfg 1)
    "sigmas": [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0],
}
# площадь кропа для Qwen-Image-2.1 (Мп): модель родная для 2K — на ~1 Мп объект на большом рендере рисовался
# вдвое мельче итогового и выходил «пластиковым»
QWEN21_MEGAPIXELS = float(os.getenv("QWEN21_MEGAPIXELS", "2.0"))
# модели генерации, доступные в Inpainting
INPAINT_MODELS = [*FLUX_MODELS, QWEN_EDIT_MODEL, QWEN21_MODEL]
FLUX_VAE = os.getenv("FLUX_VAE", "flux2-vae.safetensors")
# Площадь кадра генерации в мегапикселях (пропорции исходника сохраняются)
RENDER_MEGAPIXELS = float(os.getenv("RENDER_MEGAPIXELS", "1.5"))
COMFY_TIMEOUT = float(os.getenv("COMFY_TIMEOUT", "600"))
# «Сохранить небо / фон / окружение»: доля шагов перерисовки сохраняемых зон по маске
# (меньше — ближе к исходнику, больше — фотореальнее и ближе к референсу, но расклад может поплыть)
# и доля локального света рендера в стартовой склейке
ZONE_RESTYLE = float(os.getenv("ZONE_RESTYLE", "0.875"))
# доля мелкой фактуры полного рендера в фоне и окружении стартовой склейки: без неё плоская зелёная
# заливка исходника так и остаётся гладкой «перекрашенной» плоскостью вместо травы
ZONE_DETAIL = float(os.getenv("ZONE_DETAIL", "0.9"))
# сила подгонки цвета неба к небу референса (0 — выкл.)
SKY_MATCH = float(os.getenv("SKY_MATCH", "0.85"))
LOCAL_LIGHT = float(os.getenv("LOCAL_LIGHT", "0.6"))
# Inpainting: площадь кропа вокруг маски в мегапикселях (мелкий объект получает полное разрешение модели)
INPAINT_MEGAPIXELS = float(os.getenv("INPAINT_MEGAPIXELS", "1.0"))
