import os

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
