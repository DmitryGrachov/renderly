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

# FLUX.2 [klein] 4B (distilled) в ComfyUI
FLUX_UNET = os.getenv("FLUX_UNET", "flux-2-klein-4b-fp8.safetensors")
FLUX_TEXT_ENCODER = os.getenv("FLUX_TEXT_ENCODER", "qwen_3_4b.safetensors")
FLUX_VAE = os.getenv("FLUX_VAE", "flux2-vae.safetensors")
# Площадь кадра генерации в мегапикселях (пропорции исходника сохраняются)
RENDER_MEGAPIXELS = float(os.getenv("RENDER_MEGAPIXELS", "1.5"))
COMFY_TIMEOUT = float(os.getenv("COMFY_TIMEOUT", "600"))
# «Сохранить небо / фон / окружение»: доля шагов прохода гармонизации склейки
# (больше — естественнее свет, но выше риск что-то дорисовать) и доля локального света рендера в зонах
HARMONIZE = float(os.getenv("HARMONIZE", "0.25"))
LOCAL_LIGHT = float(os.getenv("LOCAL_LIGHT", "0.6"))
