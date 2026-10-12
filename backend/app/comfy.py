"""Клиент ComfyUI — заглушка.

TODO:
  1. POST {COMFYUI_URL}/upload/image — загрузить исходник (и референс)
  2. POST {COMFYUI_URL}/prompt — отправить workflow с подставленными параметрами
  3. WS  {COMFYUI_URL}/ws?clientId=... — слушать прогресс
  4. GET {COMFYUI_URL}/history/{prompt_id} + /view — забрать результат
"""

from .config import COMFYUI_URL


async def run_img2img(image: bytes, scene: str, preset: str, strength: float, reference: bytes | None = None) -> bytes:
    # Пока просто возвращаем исходник без изменений.
    _ = (COMFYUI_URL, scene, preset, strength, reference)
    return image


async def run_inpaint(image: bytes, mask: bytes, scene: str, mode: str, prompt: str | None = None) -> bytes:
    # mode: "edit" — перерисовать область по prompt, "remove" — удалить объект (заполнить фоном).
    # mask: PNG, белое — область для перерисовки.
    _ = (COMFYUI_URL, scene, mask, mode, prompt)
    return image
