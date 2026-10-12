import logging
import os
import secrets

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
