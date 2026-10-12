"""Пресеты освещения. Позже каждому пресету будет соответствовать workflow/промпт ComfyUI."""

PRESETS = {
    "natural": {"title": "Естественный свет", "prompt": "soft overcast natural daylight, diffuse light"},
    "day": {"title": "День", "prompt": "bright sunny day, clear blue sky, crisp shadows"},
    "golden": {"title": "Золотой час", "prompt": "golden hour, warm low sun, long soft shadows"},
    "evening": {"title": "Вечер", "prompt": "blue hour, dusk, warm interior lights glowing in windows"},
    "custom": {"title": "Свой референс", "prompt": None},
}
