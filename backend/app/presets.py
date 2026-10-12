"""Пресеты освещения по типу сцены. Позже каждому будет соответствовать workflow/промпт ComfyUI."""

SCENES = ("interior", "exterior")

PRESETS = {
    "exterior": {
        "natural": {"title": "Естественный свет", "prompt": "soft overcast natural daylight, diffuse light"},
        "day": {"title": "День", "prompt": "bright sunny day, clear blue sky, crisp shadows"},
        "golden": {"title": "Золотой час", "prompt": "golden hour, warm low sun, long soft shadows"},
        "evening": {"title": "Вечер", "prompt": "blue hour, dusk, warm interior lights glowing in windows"},
        "custom": {"title": "Свой референс", "prompt": None},
    },
    "interior": {
        "natural": {"title": "Естественный свет", "prompt": "soft natural daylight from windows, diffuse indirect light"},
        "day": {"title": "День", "prompt": "bright direct sunlight through windows, sun patches on the floor, crisp shadows"},
        "golden": {"title": "Золотой час", "prompt": "warm golden hour sunbeams through windows, long soft shadows"},
        "evening": {"title": "Вечер", "prompt": "cozy evening, warm lamps and accent lighting on, dusk outside the windows"},
        "custom": {"title": "Свой референс", "prompt": None},
    },
}
