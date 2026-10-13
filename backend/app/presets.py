"""Пресеты освещения по типу сцены.

prompt — короткое описание света (используется без DeepSeek).
look   — подробное описание колористики и атмосферы референса из refs/<scene>/<id>.jpg;
         его DeepSeek вплетает в итоговый промпт, а сам референс уходит во FLUX как второе изображение.
"""

from pathlib import Path

SCENES = ("interior", "exterior")

REFS_DIR = Path(__file__).parent / "refs"

PRESETS = {
    "exterior": {
        "natural": {
            "title": "Естественный свет",
            "prompt": "soft overcast natural daylight, diffuse light",
            "look": (
                "Overcast daylight: bright, even, pale white-grey sky with no visible sun. Soft diffuse light from above, "
                "no hard shadows, only gentle ambient occlusion under balconies and in recesses. Neutral white balance "
                "around 5600K, true-to-life material colours, fresh light-green foliage, moderate contrast, clean "
                "documentary architectural photography look, calm and quiet European city atmosphere."
            ),
        },
        "day": {
            "title": "День",
            "prompt": "bright sunny day, clear blue sky, crisp shadows",
            "look": (
                "Bright sunny midday in summer: deep saturated cobalt-blue sky fading lighter towards the horizon with a few "
                "small white cumulus clouds low on the horizon. Strong direct sunlight from high and to the side, crisp "
                "well-defined shadows, glass reflecting the blue sky, vivid green lawn and full-leaf trees. High clarity, "
                "punchy contrast, rich saturated colours, cheerful lively atmosphere."
            ),
        },
        "golden": {
            "title": "Золотой час",
            "prompt": "golden hour, warm low sun, long soft shadows",
            "look": (
                "Golden hour shortly before sunset: clear pale sky with a soft gradient from light blue to warm peach near "
                "the horizon. Low warm sun grazing the facades, golden highlights on sunlit surfaces, long soft shadows, "
                "cool blue-grey tones in the shade. Muted natural colour grading, gentle contrast, slight warm haze, "
                "tranquil and serene atmosphere."
            ),
        },
        "evening": {
            "title": "Вечер",
            "prompt": "blue hour, dusk, warm interior lights glowing in windows",
            "look": (
                "Blue hour at dusk: clear cloudless sky in a smooth gradient from steel blue at the top to pale cyan-grey at "
                "the horizon, no sun. Soft cool ambient light on the facades, warm amber 2700K interior lighting glowing "
                "through many windows, loggias and the glazed ground floor, any existing street lamps glowing warm, subtle "
                "reflections on paving. Low contrast, cinematic cool-blue versus warm-orange colour contrast, calm "
                "evening atmosphere."
            ),
        },
        "custom": {"title": "Свой референс", "prompt": None, "look": None},
    },
    "interior": {
        "natural": {"title": "Естественный свет", "prompt": "soft natural daylight from windows, diffuse indirect light", "look": None},
        "day": {"title": "День", "prompt": "bright direct sunlight through windows, sun patches on the floor, crisp shadows", "look": None},
        "golden": {"title": "Золотой час", "prompt": "warm golden hour sunbeams through windows, long soft shadows", "look": None},
        "evening": {"title": "Вечер", "prompt": "cozy evening, warm lamps and accent lighting on, dusk outside the windows", "look": None},
        "custom": {"title": "Свой референс", "prompt": None, "look": None},
    },
}


def reference_image(scene: str, preset: str) -> bytes | None:
    """Встроенный референс пресета, если он есть."""
    path = REFS_DIR / scene / f"{preset}.jpg"
    return path.read_bytes() if path.is_file() else None
