"""Проверка карты зон зрячей моделью (DeepSeek): детекторы по тексту не понимают, есть ли на кадре то, что
у них спросили, — они находят самое похожее (на стеклянной башне «крышей» оказалась вся башня). DeepSeek видит кадр
с пронумерованными контурами зон и по каждой отвечает, что это на самом деле, какие зоны — один объект, и как
называется материал стен. Зоны — кандидаты, их края не меняются: меняются подписи, состав и объединения."""

import hashlib
import io
import os
import json
import time
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .config import log
from .prompting import DEEPSEEK_API_KEY, _chat, _data_url, llm_model

# классы, которыми DeepSeek может назвать зону, и их вид (раздел во вкладке)
CLASSES = {
    "Небо": "scene", "Газон": "scene", "Земля": "scene", "Дорога": "scene", "Тротуар": "scene", "Дорожка": "scene",
    "Ступени": "part", "Покрытие": "part", "Вода": "scene", "Забор": "scene", "Машина": "scene", "Человек": "scene",
    "Велосипед": "scene", "Столб": "scene", "Фонарь": "scene", "Скамейка": "scene", "Вывеска": "scene",
    "Навес": "part", "Здания (фон)": "scene", "Стена (фон)": "scene", "Горы": "scene", "Камни": "scene",
    "Дерево": "veg", "Деревья": "veg", "Кусты": "veg", "Цветы": "veg", "Живая изгородь": "veg",
    "Стены": "material", "Крыша": "part", "Окно": "part", "Дверь": "part", "Ворота": "part", "Балкон": "part",
    "Ограждение": "part", "Колонна": "part", "Водосток": "part", "Труба": "part", "Светильник": "part",
    "Карниз": "part", "Козырёк": "part", "Стекло": "material", "Рамы": "material", "Прочее": "scene",
}
# у таких классов номер экземпляра («Окно 3»); остальные — одна зона на класс
INSTANCES = {"Дерево", "Машина", "Человек", "Велосипед", "Столб", "Фонарь", "Скамейка", "Окно", "Дверь", "Ворота",
             "Балкон", "Колонна", "Водосток", "Труба", "Светильник"}

SYSTEM = (
    "You check an automatic segmentation of an architectural photo or render. You get the whole image for context "
    "and a contact sheet: one tile per zone, the tile number is written in the black bar above the tile; inside the "
    "tile the zone itself is shown in full colour with a bright magenta outline and everything else is dimmed. "
    "You also get the list of zones with their current labels. The labels come from detectors that match words, "
    "not meaning: they sometimes call a whole glass tower 'roof', a gutter 'railing', a dark hedge 'fence'. Judge "
    "each tile by what its bright, outlined part really is. Keep the current label unless it is clearly wrong; "
    "when unsure, keep it.\n"
    "For every zone return its true class from this list (Russian, exactly as written): "
    + ", ".join(CLASSES) +
    ". Use 'Прочее' when it fits none, and 'Удалить' when the zone is a mistake that covers nothing distinct "
    "(e.g. a sliver of another zone). Building walls are 'Стены'; for them also give the wall material in Russian "
    "in 1-3 words as it looks (e.g. 'штукатурка', 'кирпич', 'камень', 'стекло', 'металлические панели', "
    "'дерево'). Also list groups of zone numbers that are one object or one class that must be one zone "
    "(two pieces of one roof; two hedges; walls of the same material) — instances that are separate objects "
    "(two trees, two windows, two cars) stay separate.\n"
    "For every zone also give your confidence from 0 to 1 that your class is right, judging only by what you "
    "see in its tile and the whole image.\n"
    'Answer with compact JSON, no explanations: {"zones": [{"id": <number>, "class": "<class>", '
    '"confidence": <0-1>, "material": "<only for Стены>"}], "merge": [[<id>, <id>, ...]]}'
)
# смена подписи или удаление — только при такой уверенности модели: судья ошибается в обе стороны
# (на одинаковых входах то верно исправлял нарисованный газон, то называл настоящий газон «покрытием»)
CHANGE_CONFIDENCE = 0.85
TILE = 256  # сторона плитки контактного листа
# сколько DeepSeek рассуждает над листом: high (по умолчанию у модели) — ~60–100 с
ZONE_CHECK_EFFORT = os.getenv("ZONE_CHECK_EFFORT", "high")
TILE_COLS = 8

_cache: OrderedDict[str, dict] = OrderedDict()
CACHE_SIZE = 64


def _sheet(src: Image.Image, ids: np.ndarray, zones: list[dict]) -> Image.Image:
    """Контактный лист: плитка на зону — вырезка вокруг неё с запасом, зона в полном цвете с ярким контуром,
    остальное затемнено; номер — на чёрной полосе над плиткой (на общем кадре номера мелкие и тесные, и модель
    путала, какой номер к какой зоне: газон называла дорогой, светильник — окном)."""
    h, w = ids.shape
    img = np.asarray(src.resize((w, h)).convert("RGB")).astype(np.float32)
    bar = 26
    rows = (len(zones) + TILE_COLS - 1) // TILE_COLS
    sheet = Image.new("RGB", (TILE_COLS * TILE, rows * (TILE + bar)), "black")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    for n, z in enumerate(zones):
        m = ids == z["id"]
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        side = max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.6 + 24
        y0, x0 = int(max(0, cy - side / 2)), int(max(0, cx - side / 2))
        y1, x1 = int(min(h, cy + side / 2)), int(min(w, cx + side / 2))
        crop = img[y0:y1, x0:x1].copy()
        mm = m[y0:y1, x0:x1]
        crop[~mm] *= 0.3
        edge = mm ^ ndimage.binary_erosion(mm, iterations=max(1, round(0.006 * side)))
        crop[edge] = (255, 0, 255)
        tile = Image.fromarray(crop.clip(0, 255).astype(np.uint8))
        tile.thumbnail((TILE, TILE), Image.LANCZOS)
        r, c = divmod(n, TILE_COLS)
        ox, oy = c * TILE, r * (TILE + bar)
        sheet.paste(tile, (ox + (TILE - tile.width) // 2, oy + bar + (TILE - tile.height) // 2))
        d.text((ox + 6, oy + 3), str(z["id"]), fill="white", font=font)
    return sheet


async def check_zones(src: Image.Image, ids: np.ndarray, veg: np.ndarray, zones: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Исправленные (ids, veg, zones). Без DeepSeek или при сбое — как было."""
    if not DEEPSEEK_API_KEY or not zones:
        return ids, veg, zones
    if all("color" in z for z in zones):
        # кадр по вектору: цвет — маска как есть, «Материал N» с образцом цвета. Подписи DeepSeek здесь путались
        # (цоколь — «дверь», изгородь — «окно»: у одного цвета много кусков по всему кадру) — ничего не додумываем
        return ids, veg, zones
    listing = "\n".join(f"{z['id']}: {z['label']}" for z in zones)
    key = hashlib.sha256(
        ids.tobytes() + listing.encode() + src.tobytes()[:1_000_000] + llm_model.get().encode() + ZONE_CHECK_EFFORT.encode()
    ).hexdigest()
    if key in _cache:
        data = _cache[key]
    else:
        sheet = _sheet(src, np.where(veg > 0, veg, ids), zones)
        try:
            t0 = time.time()
            raw = await _chat(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Whole image:"},
                        {"type": "image_url", "image_url": {"url": _data_url(_png(src), 1280)}},
                        {"type": "text", "text": "Contact sheet (one tile per zone, number above the tile):"},
                        {"type": "image_url", "image_url": {"url": _data_url(_png(sheet), 2048)}},
                        {"type": "text", "text": "Zones (number: current label):\n" + listing},
                    ]},
                ],
                json_mode=True,
                effort=ZONE_CHECK_EFFORT,
            )
            log.info("проверка зон DeepSeek: ответ за %.1f с", time.time() - t0)
            data = json.loads(raw)
        except Exception as e:  # noqa: BLE001 — без проверки карта остаётся как есть
            log.warning("Проверка зон DeepSeek недоступна: %s", e)
            return ids, veg, zones
        _cache[key] = data
        if len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)

    verdict = {}
    for z in data.get("zones") or []:
        try:
            conf = float(z.get("confidence") or 0)
            verdict[int(z["id"])] = (
                str(z.get("class") or "").strip() if conf >= CHANGE_CONFIDENCE else "",
                str(z.get("material") or "").strip(),
            )
        except (KeyError, TypeError, ValueError):
            continue
    by_id = {z["id"]: z for z in zones}
    ids, veg = ids.copy(), veg.copy()
    changes = []
    if all("color" in z for z in zones):
        # кадр по вектору: каждый цвет — маска как есть; модель только подписывает, ничего не сливает и не удаляет
        out = []
        for z in zones:
            cls, material = verdict.get(z["id"], ("", ""))
            if material and (cls or "").startswith("Стены"):
                label = f"Стены · {material.lower()}"
            elif cls in CLASSES and cls not in ("Удалить", "Прочее"):
                label = cls + (f" · {material.lower()}" if material else "")
            else:
                label = z["label"]
            out.append({**z, "label": label, "kind": CLASSES.get(cls, z["kind"]) if cls in CLASSES else z["kind"]})
        counts: dict[str, int] = {}
        totals = {lab: sum(1 for z in out if z["label"] == lab) for lab in {z["label"] for z in out}}
        for z in out:
            if totals[z["label"]] > 1:
                counts[z["label"]] = counts.get(z["label"], 0) + 1
                z["label"] = f"{z['label']} {counts[z['label']]}"
        log.info("проверка зон DeepSeek (вектор): %s", ", ".join(z["label"] for z in out))
        return ids, veg, out

    # объединения: всё в первую зону группы (класс — по большинству площади)
    for group in data.get("merge") or []:
        try:
            g = [int(i) for i in group if int(i) in by_id]
        except (TypeError, ValueError):
            continue
        if len(g) < 2:
            continue
        if any(by_id[i]["label"].startswith("Стены") for i in g):
            # стены модель только называет: деление по альбедо надёжнее, а сливала она нестабильно (на одних и тех
            # же входах то верно, то штукатурку с камнем или белые стены с тёмными рамами)
            continue
        g.sort(key=lambda i: -int((ids == i).sum() + (veg == i).sum()))
        head = g[0]
        for i in g[1:]:
            ids[ids == i] = head
            veg[veg == i] = head
            by_id.pop(i, None)
        changes.append(f"объединены {g}")

    out = []
    for i, z in by_id.items():
        cls, material = verdict.get(i, ("", ""))
        if cls == "Удалить":
            ids[ids == i] = 0
            veg[veg == i] = 0
            changes.append(f"{i} «{z['label']}» удалена")
            continue
        if cls not in CLASSES:
            cls = None  # нет ответа или чужой класс — подпись прежняя
        if cls:
            base = cls
        else:
            # без ответа подпись прежняя; номер экземпляра снимается — его проставят заново
            base = z["label"].rstrip("0123456789 ") if z["label"].rstrip("0123456789 ") in INSTANCES else z["label"]
        if (cls or base).startswith("Стены") and material:
            base = f"Стены · {material.lower()}"  # название материала — всегда, класс стен не меняется
        if cls and not z["label"].startswith(cls):
            changes.append(f"{i} «{z['label']}» → «{base}»")
        out.append({"id": i, "label": base, "kind": CLASSES.get(cls or "", z["kind"]), "group": None})

    # одинаковые подписи «вещей» — номера экземпляров; одинаковые «пространства» — одна зона
    seen: dict[str, int] = {}
    final = []
    for z in sorted(out, key=lambda z: z["id"]):
        cls = z["label"].split(" · ")[0]
        # стены одинакового названия не сливаются: «штукатурка» белая и тёмно-серая — разные материалы под замену
        if cls not in INSTANCES and cls != "Стены" and z["label"] in seen:
            head = seen[z["label"]]
            ids[ids == z["id"]] = head
            veg[veg == z["id"]] = head
            changes.append(f"{z['id']} слита с {head} («{z['label']}»)")
            continue
        seen.setdefault(z["label"], z["id"])
        final.append(z)
    counts: dict[str, int] = {}
    totals = {lab: sum(1 for z in final if z["label"] == lab) for lab in {z["label"] for z in final}}
    for z in final:
        if totals[z["label"]] > 1:
            counts[z["label"]] = counts.get(z["label"], 0) + 1
            z["label"] = f"{z['label']} {counts[z['label']]}"
    log.info("проверка зон DeepSeek: %s", "; ".join(changes) or "без изменений")
    return ids, veg, final


def _png(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()
