"""INEGO Zone Map — карта зон кадра для выбора готовых масок (вкладка Render pass).

Слои, поверх друг друга:
  1. сцена — EoMT-L 1280 (ADE20K panoptic): небо, газон, дорога, тротуар, деревья, кусты, машины, люди, здание.
     Лучшая из сравнённых (OneFormer, Mask2Former): только она на клей-рендере отделила дом от фона;
  2. части здания — детекторы по тексту: Grounding DINO base (окна, двери, ворота, крыша, балкон — рамки точнее всех)
     и OWLv2 (колонны, водостоки, трубы — их Grounding DINO не видит); ложные — по форме и пересечениям.
     Крыша — один слой (все найденные куски вместе);
  3. по всему кадру Grounding DINO ищет то, что EoMT пропускает: живую изгородь, забор и каждое дерево отдельно;
  4. края — SAM 2.1 по рамкам;
  5. стены — по материалам, как Material ID: один слой на материал по всему зданию («Стены · материал 1» —
     вся штукатурка, «Стены · материал 2» — весь кирпич). Куски — SAM по сетке точек (однородные первыми), один
     ли материал — по альбедо Marigold-IID (цвет без света: штукатурка на солнце и в тени — одна) и фактуре.
     Мелкие детали фасада (светильники) — Grounding DINO по увеличенной вырезке здания (на весь кадр их не видно).
     Горизонтальные плоскости (пол террасы) — «Покрытие» (оси здания по MoGe-2 и линиям кадра);
  6. края карты — крапинки соседям, границы прижаты к перепадам картинки (guided filter), прямые края архитектуры —
     по прямым кадра; растительность — мягкая альфа ViTMatte: в тримапе «точно дерево» — только не похожее на небо
     (небо между ветками раздутого пятна EoMT иначе становилось деревом).

Выход (OUTPUT_NODE): PNG карты (значение пикселя — id зоны, 0 — без зоны), PNG альфы растительности, PNG id
растительности под альфой и список зон {id, label, kind, group} в ui.zones (JSON).
"""

import json
import logging
import os
import time
import uuid

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

import folder_paths

from . import (_decompose, _device, _free_vram, _manhattan_faces, _manhattan_frame, _moge, _sam, _straighten,
               _to_srgb, _to_uint8)

log = logging.getLogger("inego")

EOMT_ID = "tue-mps/ade20k_panoptic_eomt_large_1280"
GDINO_ID = "IDEA-Research/grounding-dino-base"
OWL_ID = "google/owlv2-large-patch14-ensemble"
MATTE_ID = "hustvl/vitmatte-base-composition-1k"
CACHE = f"{folder_paths.models_dir}/zones"

# части здания: детектор, запрос, порог, подпись
PARTS = [
    ("gdino", "window", 0.30, "Окно"),
    ("gdino", "door", 0.30, "Дверь"),
    ("gdino", "garage door", 0.35, "Ворота"),
    ("gdino", "roof", 0.30, "Крыша"),
    ("gdino", "balcony", 0.35, "Балкон"),
    ("owl", "column", 0.25, "Колонна"),
    ("owl", "downpipe", 0.25, "Водосток"),
    ("owl", "chimney", 0.25, "Труба"),
]
# мелкие детали фасада — по увеличенной вырезке здания: запрос, порог, подпись
DETAILS = [("lamp", 0.40, "Светильник")]
DETAIL_SIDE = 1600  # длинная сторона увеличенной вырезки здания
MAT_MIN = 0.03  # кусок материала стен — не меньше этой доли стен
FAR_DEPTH = 2.0  # здание дальше стольких глубин основного — фон (город на горизонте)
DEPTH_JUMP = 1.12  # скачок глубины между соседними пикселями, после которого здания — разные
# по всему кадру (EoMT их пропускает или сливает): запрос, порог, подпись, вид
SCENE_EXTRA = [
    ("hedge", 0.30, "Живая изгородь", "veg"),
    ("fence", 0.35, "Забор", "scene"),
    ("tree", 0.30, "Дерево", "veg"),
]
BUILDING = {"building", "house", "skyscraper", "tower"}
# классы ADE20K → подписи; «вещи» — отдельными экземплярами, остальное сливается по классу
SCENE = {
    "sky": "Небо", "grass": "Газон", "field": "Газон", "earth": "Земля", "sand": "Земля", "road": "Дорога",
    "sidewalk": "Тротуар", "path": "Дорожка", "floor": "Покрытие", "tree": "Деревья", "plant": "Кусты",
    "flower": "Цветы", "fence": "Забор", "wall": "Стена (фон)", "stairs": "Ступени", "step": "Ступени",
    "stairway": "Ступени", "car": "Машина", "truck": "Машина", "van": "Машина", "person": "Человек",
    "bicycle": "Велосипед", "minibike": "Мотоцикл", "pole": "Столб", "streetlight": "Фонарь", "bench": "Скамейка",
    "water": "Вода", "mountain": "Горы", "hill": "Холмы", "rock": "Камни", "signboard": "Вывеска",
    "awning": "Навес", "railing": "Ограждение", "column": "Колонна",
}
THINGS = {"car", "truck", "van", "person", "bicycle", "minibike", "bench", "pole", "streetlight"}
OPENINGS = {"Окно", "Дверь", "Ворота"}
ARCH = ("Крыша", "Стены", "Окно", "Дверь", "Ворота", "Колонна", "Труба", "Балкон", "Покрытие", "Ступени", "Водосток",
        "Светильник")
VEG = ("Дерево", "Деревья", "Кусты", "Цветы", "Живая изгородь")
MATTE_SIDE = 1536  # длинная сторона кропа для ViTMatte

_models: dict = {}


def _load(key, fn):
    if key not in _models:
        t = time.time()
        _models[key] = fn()
        log.info("INEGO zones: загружена модель %s за %.1f с", key, time.time() - t)
    return _models[key]


def _on_gpu(model):
    """Модель на видеокарту на время прогона, затем обратно в RAM (рядом живут Qwen и klein)."""

    class _Ctx:
        def __enter__(self):
            model.to(_device())
            return model

        def __exit__(self, *a):
            model.to("cpu")
            torch.cuda.empty_cache()

    return _Ctx()


def _scene(img: Image.Image):
    from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

    proc, model = _load("eomt", lambda: (
        AutoImageProcessor.from_pretrained(EOMT_ID, cache_dir=CACHE),
        AutoModelForUniversalSegmentation.from_pretrained(EOMT_ID, cache_dir=CACHE, torch_dtype=torch.float16).eval()))
    w, h = img.size
    inp = proc(images=img, return_tensors="pt")
    with _on_gpu(model):
        inp = {k: (v.to(_device(), torch.float16) if v.dtype.is_floating_point else v.to(_device())) if hasattr(v, "to") else v
               for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp)
        r = proc.post_process_panoptic_segmentation(out, target_sizes=[(h, w)])[0]
    seg = r["segmentation"].cpu().numpy()
    return [(model.config.id2label[s["label_id"]], seg == s["id"]) for s in r["segments_info"]]


def _detector(kind: str):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    repo = GDINO_ID if kind == "gdino" else OWL_ID
    return _load(kind, lambda: (AutoProcessor.from_pretrained(repo, cache_dir=CACHE),
                                AutoModelForZeroShotObjectDetection.from_pretrained(repo, cache_dir=CACHE).eval()))


def _detect(img: Image.Image, kind: str, query: str, thr: float):
    """Рамки по тексту (модель уже на видеокарте — см. build_zone_map)."""
    w, h = img.size
    proc, model = _detector(kind)
    with torch.no_grad():
        if kind == "gdino":
            inp = proc(images=img, text=query + ".", return_tensors="pt").to(_device())
            out = model(**inp)
            r = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=thr, text_threshold=thr,
                                                            target_sizes=[(h, w)])[0]
        else:
            inp = proc(text=[[query]], images=img, return_tensors="pt").to(_device())
            out = model(**inp)
            r = proc.post_process_grounded_object_detection(out, threshold=thr, target_sizes=[(h, w)],
                                                            text_labels=[[query]])[0]
    return list(zip(r["boxes"].tolist(), r["scores"].tolist()))


def _hedge_or_fence(img: Image.Image, b) -> tuple[str, str]:
    """Изгородь или забор: по цвету не различить (изгородь в тени почти чёрная), а детектор на «fence» находит и
    изгородь — спрашиваем его же по вырезке, какой из двух ответов увереннее."""
    pad = 0.1 * max(b[2] - b[0], b[3] - b[1])
    crop = img.crop((max(0, b[0] - pad), max(0, b[1] - pad), min(img.width, b[2] + pad), min(img.height, b[3] + pad)))
    best = {}
    for q in ("hedge", "fence"):
        found = _detect(crop, "gdino", q, 0.15)
        best[q] = max((sc for _, sc in found), default=0.0)
    return ("Живая изгородь", "veg") if best["hedge"] >= best["fence"] else ("Забор", "scene")


def _nms(dets, iou=0.5):
    keep = []
    for b, s in sorted(dets, key=lambda d: -d[1]):
        ok = True
        for kb, _ in keep:
            ix = max(0, min(b[2], kb[2]) - max(b[0], kb[0])) * max(0, min(b[3], kb[3]) - max(b[1], kb[1]))
            u = (b[2] - b[0]) * (b[3] - b[1]) + (kb[2] - kb[0]) * (kb[3] - kb[1]) - ix
            if ix / max(u, 1) > iou or ix / max(1, (b[2] - b[0]) * (b[3] - b[1])) > 0.85:
                ok = False
                break
        if ok:
            keep.append((b, s))
    return keep


def _clean_parts(found, bmask, H, W):
    """Ложные подписи: проверки формы и одна рамка на место (самая уверенная среди проёмов)."""
    if bmask.any():
        ys, xs = np.nonzero(bmask)
        bh, bw = np.ptp(ys) + 1, np.ptp(xs) + 1
    else:
        bh, bw = H, W
    out = []
    for name, b, s in found:
        w, h = b[2] - b[0], b[3] - b[1]
        if name == "Дверь" and (h > 0.75 * bh or w * h > 0.12 * bw * bh or h < 1.1 * w):
            continue  # «дверь» во весь объём или лежачая — не дверь
        if name == "Ворота":
            if h > 1.4 * w:
                name = "Дверь"  # высокая и узкая — входная дверь
            elif w * h > 0.25 * bw * bh:
                continue
        if name == "Ограждение" and h < 0.06 * bh:
            continue  # тонкая полоса — карниз или желоб, не ограждение
        out.append((name, b, s))
    keep = []
    for name, b, s in sorted(out, key=lambda f: -f[2]):
        ok = True
        for kn, kb, _ in keep:
            if name in OPENINGS and kn in OPENINGS:
                ix = max(0, min(b[2], kb[2]) - max(b[0], kb[0])) * max(0, min(b[3], kb[3]) - max(b[1], kb[1]))
                a, ka = (b[2] - b[0]) * (b[3] - b[1]), (kb[2] - kb[0]) * (kb[3] - kb[1])
                if ix / (a + ka - ix) > 0.4 or ix > 0.7 * min(a, ka):
                    ok = False  # тот же проём другим классом или проём внутри проёма
                    break
        if ok:
            keep.append((name, b, s))
    return keep


def _sam_run(rgb, **prompts):
    proc, model = _sam()
    with _on_gpu(model):
        inp = proc(images=rgb, return_tensors="pt", **prompts).to(_device())
        with torch.no_grad():
            out = model(**inp, multimask_output=True)
        ms = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0].numpy() > 0
        sc = out.iou_scores[0].float().cpu().numpy()
    return ms, sc


def _sam_boxes(rgb, boxes):
    """Маски SAM 2.1 по рамкам одним прогоном: из уверенных вариантов — лучше заполняющий рамку, обрезанный ею."""
    if not boxes:
        return []
    h, w = rgb.shape[:2]
    ms, sc = _sam_run(rgb, input_boxes=[[list(b) for b in boxes]])
    res = []
    for i, b in enumerate(boxes):
        pad = 0.04 * max(b[2] - b[0], b[3] - b[1])
        clip = np.zeros((h, w), bool)
        clip[max(0, int(b[1] - pad)):int(b[3] + pad) + 1, max(0, int(b[0] - pad)):int(b[2] + pad) + 1] = True
        area = (b[2] - b[0]) * (b[3] - b[1])
        good = [j for j in range(sc.shape[1]) if sc[i, j] >= sc[i].max() - 0.05]
        j = max(good, key=lambda j: min((ms[i, j] & clip).sum(), area) - 0.5 * (ms[i, j] & ~clip).sum())
        res.append(ms[i, j] & clip)
    return res


def _geometry(rgb):
    """Точки, нормали и камера MoGe-2 (один прогон на карту)."""
    src = rgb.astype(np.float32) / 255
    model = _moge()
    with _on_gpu(model):
        with torch.no_grad():
            g = model.infer(torch.from_numpy(src).permute(2, 0, 1).to(_device()))
    return (g["points"].float().cpu().numpy(), g["normal"].float().cpu().numpy(),
            g["intrinsics"].float().cpu().numpy())


def _far_buildings(bmask, P):
    """Здания далеко за основным (город на горизонте): EoMT называет их тем же «building», и они уходили в стены.
    Основное здание — крупнейший связный кусок вместе со всем, что примыкает к нему без скачка глубины (длинный
    дом, уходящий вглубь, — тоже он: глубина вдоль стены растёт плавно). Остальное дальше FAR_DEPTH его глубины
    — фон."""
    depth = np.linalg.norm(np.where(np.isfinite(P), P, np.inf), axis=-1)
    ok = bmask & np.isfinite(depth)
    # разрыв: соседние пиксели здания с глубиной, отличающейся больше чем на DEPTH_JUMP, не связаны
    ld = np.log(np.where(ok, depth, 1.0))
    jump = np.zeros_like(ok)
    jump[1:] |= np.abs(ld[1:] - ld[:-1]) > np.log(DEPTH_JUMP)
    jump[:-1] |= np.abs(ld[1:] - ld[:-1]) > np.log(DEPTH_JUMP)
    jump[:, 1:] |= np.abs(ld[:, 1:] - ld[:, :-1]) > np.log(DEPTH_JUMP)
    jump[:, :-1] |= np.abs(ld[:, 1:] - ld[:, :-1]) > np.log(DEPTH_JUMP)
    lab, n = ndimage.label(ok & ~jump)
    if not n:
        return np.zeros_like(bmask)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    main_id = int(np.argmax(sizes)) + 1
    main_depth = float(np.median(depth[lab == main_id]))
    meds = ndimage.median(np.where(ok, depth, 0), lab, range(1, n + 1))
    far_ids = [k + 1 for k in range(n) if k + 1 != main_id and meds[k] > FAR_DEPTH * main_depth]
    far = np.isin(lab, far_ids)
    far = ndimage.binary_opening(far, iterations=1)
    if far.any():
        log.info("INEGO zones: здания за скачком глубины дальше %.1f× основного — фон (%.1f%% кадра)",
                 FAR_DEPTH, 100 * far.mean())
    return far


def _faces(rgb, walls, geo):
    """Грани стен по осям здания; [(маска, горизонтальная)]."""
    if walls.sum() < 0.01 * walls.size:
        return [(walls, False)]
    src = rgb.astype(np.float32) / 255
    P, Nn, K = geo
    H, W = walls.shape
    Kp = K.copy()
    Kp[0] *= W
    Kp[1] *= H
    valid = walls & np.isfinite(P).all(-1) & np.isfinite(Nn).all(-1)
    gray = (np.clip(src @ np.array([0.299, 0.587, 0.114], np.float32), 0, 1) * 255).astype(np.uint8)
    try:
        axes = _manhattan_frame(gray, Nn, walls, Kp)
        built = _manhattan_faces(axes, P, Nn, valid, walls)
    except Exception as e:  # noqa: BLE001 — без граней стены остаются одной зоной
        log.warning("INEGO zones: грани не построены: %s", e)
        built = None
    if built is None:
        return [(walls, False)]
    owner, planes = built
    # горизонтальные плоскости (пол террасы, козырёк) — не стены
    return [(owner == i, abs(float(np.dot(n, axes[0]))) > np.cos(np.radians(35))) for i, (n, _) in enumerate(planes)]


def _is_roof(m: np.ndarray, bmask: np.ndarray) -> bool:
    """Крыша — верх здания: на высотке «roof» обводит всю башню, и SAM по рамке отдаёт здание целиком. Не крыша,
    если зона закрывает больше половины здания под собой (в тех же столбцах кадра) или почти всю его высоту."""
    if not m.any() or not bmask.any():
        return bool(m.any())
    cols = m.any(0)
    below = bmask[:, cols]
    if not below.any():
        return True
    ys = np.nonzero(m)[0]
    by = np.nonzero(below.any(1))[0]
    covers = (m[:, cols] & below).sum() / below.sum()
    tall = np.ptp(ys) / max(1, np.ptp(by))
    ok = covers < 0.5 and tall < 0.8  # двускатная крыша дома — до ~60% высоты здания
    if not ok:
        log.info("INEGO zones: «крыша» закрывает %.0f%% здания под собой и %.0f%% его высоты — не крыша",
                 100 * covers, 100 * tall)
    return ok


def _features(rgb):
    """Признаки материала на пиксель: альбедо Marigold-IID в Lab (цвет без света — штукатурка на солнце и в тени
    одна) и фактура — локальный перепад яркости, делённый на местную яркость (тень её не меняет): камень и кирпич —
    высокая, штукатурка — низкая."""
    import cv2

    alb = _decompose(rgb)[0]
    alb = (np.clip(_to_srgb(np.clip(alb, 0, 1)), 0, 1) * 255).astype(np.uint8)
    if alb.shape[:2] != rgb.shape[:2]:
        alb = np.asarray(Image.fromarray(alb).resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))
    lab = cv2.cvtColor(alb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    Y = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * 100 / 255
    tex = np.abs(Y - ndimage.uniform_filter(Y, 5)) / (ndimage.uniform_filter(Y, 15) + 8) * 100
    tex = ndimage.uniform_filter(tex, max(5, rgb.shape[1] // 120))
    return np.stack([a, b, L / 2, tex / 2], -1)


def _inhomogeneity(feat, m):
    v = feat[m][:: max(1, int(m.sum()) // 4000)]
    return float(v.std(0).sum())


def _split_materials(rgb, feat, walls):
    """Куски материала стен: SAM по сетке точек; однородные куски первыми (маска «вся стена» с кирпичом и
    штукатуркой разнородна и проигрывает); остаток — куску с похожим материалом (иначе граница шла дугой там, где
    стыка нет)."""
    total = walls.sum()
    if total < 400:
        return [walls]
    ys, xs = np.nonzero(walls)
    inner = ndimage.binary_erosion(walls, iterations=max(1, round(0.02 * np.sqrt(total))))
    if inner.sum() < 50:
        inner = walls
    iy, ix = np.nonzero(inner)
    pts = set()
    for gy in np.linspace(0.05, 0.95, 9):
        for gx in np.linspace(0.05, 0.95, 9):
            ty, tx = ys.min() + gy * np.ptp(ys), xs.min() + gx * np.ptp(xs)
            k = int(np.argmin((iy - ty) ** 2 + (ix - tx) ** 2))
            pts.add((float(ix[k]), float(iy[k])))
    pts = sorted(pts)
    ms, sc = _sam_run(rgb, input_points=[[[list(p)] for p in pts]], input_labels=[[[1] for _ in pts]])
    ms = ms.reshape(len(pts), -1, *walls.shape)
    sc = sc.reshape(len(pts), -1)
    cands = []
    for i in range(len(ms)):
        for j in range(ms.shape[1]):
            inw = ms[i, j] & walls
            if inw.sum() >= MAT_MIN * total and inw.sum() > 0.5 * ms[i, j].sum() and sc[i, j] > 0.7:
                cands.append(inw)
    cands.sort(key=lambda m: _inhomogeneity(feat, m) - 2.0 * np.log(m.sum() / total))
    taken = np.zeros_like(walls)
    segs = []
    for m in cands:
        new = m & ~taken
        if new.sum() >= 0.7 * m.sum() and new.sum() >= MAT_MIN * total:
            segs.append(new)
            taken |= new
    if len(segs) <= 1:
        return [walls]
    fs = ndimage.gaussian_filter(feat, (2, 2, 0))
    cost = []
    for m in segs:
        mu, sd = fs[m].mean(0), fs[m].std(0) + 1.0
        cost.append((((fs - mu) / sd) ** 2).sum(-1))
    lab = np.argmin(np.stack(cost), 0) + 1
    for i, m in enumerate(segs, 1):
        lab[m] = i
    sig = max(1.5, 0.004 * max(walls.shape))
    votes = np.stack([ndimage.gaussian_filter(((lab == i) & walls).astype(np.float32), sig) for i in range(1, len(segs) + 1)])
    lab = np.where(walls, np.argmax(votes, 0) + 1, 0)
    return [lab == i for i in range(1, len(segs) + 1)]


def _material_groups(feat, segs):
    """Один материал — одна группа: альбедо (цвет без света) и фактура."""
    med = [np.median(feat[m], 0) for m in segs]
    group = list(range(len(segs)))

    def root(i):
        while group[i] != i:
            i = group[i]
        return i

    for i in range(len(segs)):
        for j in range(i):
            (ai, bi, li, ti), (aj, bj, lj, tj) = med[i], med[j]
            # фактура разделяет, только если разница заметная: на гладкой штукатурке мерка неустойчива (рамы окон,
            # тени и рёбра в куске), и одна штукатурка распадалась на несколько материалов
            if (abs(ai - aj) < 5 and abs(bi - bj) < 6 and abs(li - lj) * 2 < 12
                    and (abs(ti - tj) < 2.5 or max(ti, tj) + 0.5 < 1.4 * (min(ti, tj) + 0.5))):
                group[root(i)] = root(j)
    log.info("INEGO zones: куски стен (a, b, L/2, фактура/2): %s → группы %s",
             [tuple(round(float(v), 1) for v in m) for m in med], [root(i) for i in range(len(segs))])
    return [root(i) for i in range(len(segs))]


def _details(img: Image.Image, bmask: np.ndarray):
    """Мелкие детали фасада (светильники): Grounding DINO по увеличенной вырезке здания — на весь кадр они
    в несколько пикселей и не находятся. Рамки — в координатах кадра."""
    if not bmask.any():
        return []
    ys, xs = np.nonzero(bmask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    k = min(3.0, DETAIL_SIDE / max(x1 - x0, y1 - y0))
    if k <= 1.05:
        return []
    crop = img.crop((x0, y0, x1, y1)).resize((round((x1 - x0) * k), round((y1 - y0) * k)), Image.LANCZOS)
    out = []
    for q, thr, name in DETAILS:
        for b, sc in _nms(_detect(crop, "gdino", q, thr)):
            box = [x0 + b[0] / k, y0 + b[1] / k, x0 + b[2] / k, y0 + b[3] / k]
            if (box[2] - box[0]) * (box[3] - box[1]) < 0.02 * (x1 - x0) * (y1 - y0):
                out.append((name, box, sc))
    return out


def _fill(out, hole):
    """Пиксели hole — ближайшей зоне не из hole."""
    if hole.any() and (~hole).any():
        _, (iy, ix) = ndimage.distance_transform_edt(hole, return_indices=True)
        out = np.where(hole, out[iy, ix], out)
    return out


def _finish(idmap, rgb):
    """Крапинки и мелкие дырки — соседям; границы прижимаются к перепадам картинки (guided filter по всем зонам
    сразу, только в узкой полосе у границ — середина зон не плывёт)."""
    import kornia

    H, W = idmap.shape
    small = max(30, int(0.0003 * H * W))
    out = idmap.copy()
    for i in np.unique(idmap):
        if i == 0:
            continue
        lab, n = ndimage.label(idmap == i)
        if n <= 1:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        main = int(np.argmax(sizes)) + 1
        out[np.isin(lab, [k for k in range(1, n + 1) if k != main and sizes[k - 1] < small])] = -1
    # мелкие куски без зоны — тоже крапинки; крупные остаются без зоны (0)
    lab, n = ndimage.label(out == 0)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        out[np.isin(lab, [k + 1 for k in range(n) if sizes[k] < small])] = -1
    out = _fill(out, out == -1)
    ids = np.unique(out)
    r = max(2, round(0.004 * min(H, W)))
    guide = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(_device()) / 255
    idx = torch.from_numpy(np.searchsorted(ids, out)).to(_device())
    onehot = torch.nn.functional.one_hot(idx, len(ids)).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        soft = torch.cat([kornia.filters.guided_blur(guide, onehot[:, c:c + 16], 2 * r + 1, 1e-4)
                          for c in range(0, len(ids), 16)], 1)
        snapped = ids[soft[0].argmax(0).cpu().numpy()]
    del guide, onehot, soft
    torch.cuda.empty_cache()
    edge = np.zeros((H, W), bool)
    edge[1:] |= out[1:] != out[:-1]
    edge[:, 1:] |= out[:, 1:] != out[:, :-1]
    return np.where(ndimage.binary_dilation(edge, iterations=2 * r), snapped, out)


def _straighten_arch(idmap, labels, rgb):
    """Прямые края архитектуры — по прямым кадра (_straighten): срезанное отдаётся соседям."""
    out = idmap.copy()
    for i, name in labels.items():
        if not name.startswith(ARCH):
            continue
        m = idmap == i
        if m.sum() < 200:
            continue
        st = _straighten(m, rgb)
        out[st & ~m & (out == idmap)] = i
        out[m & ~st & (out == i)] = -1
    return _fill(out, out == -1)


QUAD = ("Окно", "Дверь", "Ворота", "Светильник", "Колонна", "Труба", "Стекло", "Балкон")
QUAD_IOU = 0.82  # кусок заменяется четырёхугольником, если совпадает с ним не хуже этого


def _quad(m: np.ndarray):
    """Четырёхугольник (в перспективе) по контуру куска: выпуклая оболочка, упрощённая до 4 вершин. None — кусок
    не четырёхугольник (арка, круглое окно, Г-образный)."""
    import cv2

    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        poly = cv2.approxPolyDP(hull, eps * peri, True)
        if len(poly) == 4:
            q = np.zeros(m.shape, np.uint8)
            cv2.fillPoly(q, [poly], 1)
            q = q.astype(bool)
            iou = (q & m).sum() / max(1, (q | m).sum())
            return q if iou >= QUAD_IOU else None
        if len(poly) < 4:
            return None
    return None


def _quadify(idmap, labels):
    """Примитивы: окна, двери, стёкла (каждое стекло отдельно), светильники, колонны — точные четырёхугольники
    по своему контуру, а не волнистое пятно. Добирает пиксели только у стен, рам и «без зоны»; лишнее отдаёт
    соседям."""
    out = idmap.copy()
    soft = [j for j, n in labels.items() if n.startswith(("Стены", "Рамы"))] + [0]
    for i, name in labels.items():
        if not name.startswith(QUAD):
            continue
        lab, n = ndimage.label(idmap == i)
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() < 40:
                continue
            q = _quad(m)
            if q is None:
                continue
            out[q & ~m & np.isin(out, soft)] = i
            out[m & ~q & (out == i)] = -1
    return _fill(out, out == -1)


def _matte(idmap, labels, rgb):
    """Растительность: ViTMatte по тримапу вокруг зоны — мягкая альфа (ветки на фоне неба). «Точно дерево» — пятно
    без похожего на небо (иначе небо между ветками раздутого пятна EoMT считалось деревом), неизвестность — пятно
    и полоса вокруг. Возвращает карту (зона берёт пиксели с альфой > 0.5 у неба и отдаёт с < 0.5), альфу и id
    растительности под альфой (у каждого пикселя — зона с наибольшей альфой)."""
    import cv2
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    H, W = idmap.shape
    out = idmap.copy()
    alpha = np.zeros((H, W), np.float32)
    vegid = np.zeros((H, W), np.int32)
    veg = [i for i, n in labels.items() if n.startswith(VEG) and (idmap == i).sum() >= 400]
    if not veg:
        return out, alpha, vegid
    proc, model = _load("matte", lambda: (VitMatteImageProcessor.from_pretrained(MATTE_ID, cache_dir=CACHE),
                                          VitMatteForImageMatting.from_pretrained(MATTE_ID, cache_dir=CACHE).eval()))
    sky_ids = [j for j, n in labels.items() if n == "Небо"]
    open_ids = sky_ids + [j for j, n in labels.items() if n == "Стена (фон)"] + [0]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    sky = np.isin(idmap, sky_ids)
    skyish = np.zeros((H, W), bool)
    if sky.sum() > 500:
        # «похоже на небо»: близко к цвету ближайшего настоящего неба (небо — градиент и по высоте, и к солнцу:
        # у горизонта за деревом оно светлее и теплее, чем в среднем по строке). Небо сглажено — без шума
        sig = max(2.0, 0.01 * max(H, W))
        wsum = ndimage.gaussian_filter(sky.astype(np.float32), sig) + 1e-6
        smooth = np.stack([ndimage.gaussian_filter(lab[..., c] * sky, sig) / wsum for c in range(3)], -1)
        _, (iy, ix) = ndimage.distance_transform_edt(~sky, return_indices=True)
        skyish = np.linalg.norm(lab - smooth[iy, ix], axis=-1) < 12
    with _on_gpu(model):
        for i in veg:
            m = idmap == i
            ys, xs = np.nonzero(m)
            size = max(np.ptp(ys), np.ptp(xs)) + 1
            grow = max(6, round(0.05 * size))
            fg = ndimage.binary_erosion(m, iterations=1) & ~ndimage.binary_dilation(skyish, iterations=2)
            fg = ndimage.binary_opening(fg, iterations=1)
            bg = ~ndimage.binary_dilation(m, iterations=grow)
            y0, y1 = max(0, ys.min() - grow - 8), min(H, ys.max() + grow + 9)
            x0, x1 = max(0, xs.min() - grow - 8), min(W, xs.max() + grow + 9)
            crop = rgb[y0:y1, x0:x1]
            tri = np.full(crop.shape[:2], 128, np.uint8)
            tri[fg[y0:y1, x0:x1]] = 255
            tri[bg[y0:y1, x0:x1]] = 0
            ch, cw = crop.shape[:2]
            k = min(1.0, MATTE_SIDE / max(ch, cw))
            ci, ti = Image.fromarray(crop), Image.fromarray(tri)
            if k < 1:
                ci = ci.resize((round(cw * k), round(ch * k)), Image.LANCZOS)
                ti = ti.resize(ci.size, Image.NEAREST)
            inp = proc(images=ci, trimaps=ti, return_tensors="pt").to(_device())
            with torch.no_grad():
                a = model(**inp).alphas[0, 0].float().cpu().numpy()[: ci.size[1], : ci.size[0]]
            a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((cw, ch), Image.BILINEAR)) / 255.0
            a = np.where(tri == 255, 1.0, np.where(tri == 0, 0.0, a))
            full = np.zeros((H, W), np.float32)
            full[y0:y1, x0:x1] = a
            vegid[full > alpha] = i
            alpha = np.maximum(alpha, full)
            unknown = ~fg & ~bg
            out[unknown & (full > 0.5) & np.isin(out, open_ids)] = i
            give = unknown & (full < 0.5) & (out == i)
            out = _fill(np.where(give, -1, out), give)
    vegid[alpha < 0.02] = 0
    return out, alpha, vegid


def _schema_glass(schema: np.ndarray, area: np.ndarray | None) -> np.ndarray:
    """Стекло со схемы: чисто голубое (оттенок циан, насыщенное и светлое); только в здании; без крошек."""
    import cv2

    hsv = cv2.cvtColor(schema, cv2.COLOR_RGB2HSV)
    glass = (np.abs(hsv[..., 0].astype(np.int16) - 90) <= 10) & (hsv[..., 1] > 100) & (hsv[..., 2] > 100)
    if area is not None:
        glass &= area
    lab, n = ndimage.label(glass)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        glass = np.isin(lab, [k + 1 for k in range(n) if sizes[k] >= 20])
    return glass


SCHEMA_DE = 12.0  # цвета схемы ближе этого (ΔE) — один цвет
SCHEMA_MIN = 0.0002  # пятно схемы мельче этой доли кадра — соседям
PART_COVER = 0.5  # пятно — часть (окно, дверь, светильник…), если её рамка покрывает хотя бы такую долю пятна


def _schema_regions(schema: np.ndarray, area: np.ndarray):
    """Пятна схемы в здании: цвета сводятся к палитре (k-means в Lab, близкие слиты), пятно — связный кусок
    одного цвета; крошки — соседям. Края — ровно как нарисовала модель. (метки пятен (H, W), цвет пятна)."""
    import cv2

    H, W = area.shape
    lab = cv2.cvtColor(schema, cv2.COLOR_RGB2LAB).astype(np.float32)
    X = lab[area]
    if len(X) < 100:
        return np.zeros((H, W), np.int32), {}
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = min(12, len(X))
    _, cl, cen = cv2.kmeans(X[:: max(1, len(X) // 200000)], k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    # близкие центры — один цвет
    root = list(range(k))
    for i in range(k):
        for j in range(i):
            if np.linalg.norm(cen[i] - cen[j]) < SCHEMA_DE:
                a, b = i, j
                while root[a] != a:
                    a = root[a]
                while root[b] != b:
                    b = root[b]
                root[a] = b
    for i in range(k):
        while root[root[i]] != root[i]:
            root[i] = root[root[i]]
    d = ((X[:, None, :] - cen[None]) ** 2).sum(-1)
    col = np.full((H, W), -1, np.int32)
    col[area] = np.array(root)[d.argmin(1)]
    # пятна: связные куски одного цвета; крошки — ближайшему пятну
    reg = np.zeros((H, W), np.int32)
    colour = {}
    n = 0
    small = max(20, int(SCHEMA_MIN * H * W))
    for c in np.unique(col[area]):
        lab_c, nc = ndimage.label(col == c)
        sizes = ndimage.sum(np.ones_like(lab_c), lab_c, range(1, nc + 1))
        for kk in range(1, nc + 1):
            if sizes[kk - 1] < small:
                reg[lab_c == kk] = -1
                continue
            n += 1
            reg[lab_c == kk] = n
            colour[n] = int(c)
    hole = (reg == -1) | (area & (reg == 0))
    if hole.any() and (reg > 0).any():
        reg = _fill(np.where(hole, -1, reg), hole)
    reg[~area] = 0
    return reg, colour


def _schema_building(rgb, schema, area, glass, part_masks, roof, floor, feat):
    """Здание по схеме: пятна схемы с её ровными краями, имена — от стекла, рамок частей, крыши и альбедо.
    [подпись, маска, вид]."""
    H, W = area.shape
    reg, colour = _schema_regions(schema, area)
    out = []
    walls_by_colour: dict[int, np.ndarray] = {}
    claimed = {name: np.zeros((H, W), bool) for name, _ in part_masks}
    for r in range(1, reg.max() + 1):
        m = reg == r
        if not m.any():
            continue
        if (m & glass).sum() > 0.5 * m.sum():
            out.append(["Стекло", m, "material"])
            continue
        hit = None
        for name, pm in part_masks:
            if (m & pm).sum() >= PART_COVER * m.sum():
                hit = name
                break
        if hit:
            claimed[hit] |= m
            continue
        if roof is not None and (m & roof).sum() >= PART_COVER * m.sum():
            out.append(["Крыша", m, "part"])
            continue
        if floor is not None and (m & floor).sum() >= 0.6 * m.sum():
            out.append(["Покрытие", m, "part"])
            continue
        c = colour.get(r, 0)
        walls_by_colour[c] = walls_by_colour.get(c, np.zeros((H, W), bool)) | m
    # стекло, крыша, покрытие — по одному слою
    merged = {}
    rest = []
    for z in out:
        if z[0] in ("Стекло", "Крыша", "Покрытие"):
            if z[0] in merged:
                merged[z[0]][1] |= z[1]
            else:
                merged[z[0]] = [z[0], z[1].copy(), z[2]]
        else:
            rest.append(z)
    out = rest + list(merged.values())
    # окна без стекла — рамы; остальные части — своими именами (пятна схемы внутри их рамок)
    for name, m in claimed.items():
        if m.sum() > 30:
            out.append(["Рамы" if name == "Окно" else name, m, "material" if name == "Окно" else "part"])
    # стены — слой на цвет схемы; если в одном цвете два материала (камень и штукатурка одним цветом) — по альбедо
    # куски: цвета схемы (крупный цвет ещё и по альбедо — камень и штукатурка одним цветом), затем группировка
    # по альбедо всех кусков сразу: палитра схемы дробная, один материал — один слой
    pieces = []
    for m in sorted(walls_by_colour.values(), key=lambda m: -m.sum()):
        pieces += _split_materials(rgb, feat, m) if m.sum() > 0.01 * H * W else [m]
    pieces = [p for p in pieces if p.any()]
    groups = _material_groups(feat, pieces) if len(pieces) > 1 else [0] * len(pieces)
    mats = {}
    for g, p in zip(groups, pieces):
        mats[g] = mats.get(g, np.zeros((H, W), bool)) | p
    walls = sorted(mats.values(), key=lambda m: -m.sum())
    # мелкие группы (обрезки по краям пятен схемы) — к соседнему материалу с самой длинной общей границей
    total = sum(int(m.sum()) for m in walls) or 1
    big = [m for m in walls if m.sum() >= 0.02 * total] or walls[:1]
    for m in walls:
        if any(m is b for b in big):
            continue
        ring = ndimage.binary_dilation(m, iterations=2) & ~m
        best = max(big, key=lambda b: int((ring & b).sum()))
        best |= m
    walls = big
    for n, m in enumerate(walls, 1):
        out.insert(0, ["Стены" if len(walls) == 1 else f"Стены · материал {n}", m, "material"])
    # части, которым схема не дала своего пятна (светильник того же цвета, что стена), — по рамке детектора поверх
    for name, pm in part_masks:
        if name != "Окно" and not (claimed[name] & pm).any() and pm.sum() > 30:
            out.append([name, pm, "part"])
    return out


def build_zone_map(rgb: np.ndarray, schema: np.ndarray | None = None):
    """(карта id (H, W) int32, альфа растительности (H, W) float32, id растительности под альфой (H, W) int32,
    [{id, label, kind, group}])."""
    img = Image.fromarray(rgb)
    H, W = rgb.shape[:2]
    t0 = time.time()
    segs = _scene(img)
    bmask = np.zeros((H, W), bool)
    for lab, m in segs:
        if lab in BUILDING:
            bmask |= m
    geo = _geometry(rgb)
    far = _far_buildings(bmask, geo[0]) if bmask.any() else np.zeros((H, W), bool)
    bmask &= ~far
    near = ndimage.binary_dilation(bmask, iterations=max(3, round(0.02 * max(H, W)))) if bmask.any() else None
    found, extra = [], []
    for kind in ("gdino", "owl"):
        with _on_gpu(_detector(kind)[1]):
            for _, q, thr, name in (p for p in PARTS if p[0] == kind):
                for b, s in _nms(_detect(img, kind, q, thr)):
                    if near is not None:
                        x0, y0, x1, y1 = (int(v) for v in b)
                        box = near[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
                        if box.size and box.mean() < 0.6:
                            continue  # не на главном здании
                        if (b[2] - b[0]) * (b[3] - b[1]) > 0.6 * bmask.sum() and name != "Крыша":
                            continue
                    found.append((name, b, s))
            if kind == "gdino":
                for q, thr, name, zkind in SCENE_EXTRA:
                    for b, s in _nms(_detect(img, kind, q, thr)):
                        if (b[2] - b[0]) * (b[3] - b[1]) > 0.5 * H * W:
                            continue  # рамка на полкадра — не отдельный объект
                        if name in ("Живая изгородь", "Забор"):
                            name, zkind = _hedge_or_fence(img, b)
                        extra.append((name, b, zkind))
                found += _details(img, bmask)
    found = _clean_parts(found, bmask, H, W)
    masks = _sam_boxes(rgb, [f[1] for f in found] + [e[1] for e in extra])
    part_masks, extra_masks = masks[: len(found)], masks[len(found):]

    zones = []  # [подпись, маска, вид]
    if far.sum() > 0.001 * H * W:
        zones.append(["Здания (фон)", far, "scene"])
    for lab, m in segs:
        if lab in BUILDING:
            continue
        name = SCENE.get(lab, lab)
        kind = "veg" if name.startswith(VEG) else "scene"
        if lab not in THINGS:
            same = next((z for z in zones if z[0] == name), None)
            if same is not None:
                same[1] |= m
                continue
        zones.append([name, m.copy(), kind])
    # изгородь, заборы, отдельные деревья — поверх сцены, не на здании
    # «зелёное»: зелёный канал выше красного и синего (тёмная изгородь в тени почти чёрная — по Lab не видно)
    g16 = rgb.astype(np.int16)
    green = (g16[..., 1] > g16[..., 0] + 2) & (g16[..., 1] > g16[..., 2] + 2)
    bushes = np.zeros((H, W), bool)
    for z in zones:
        if z[0] == "Кусты":
            bushes |= z[1]
    for (name, b, zkind), m in zip(extra, extra_masks):
        m = m & ~bmask
        if m.sum() < 0.0005 * H * W:
            continue
        if name == "Забор" and green[m].mean() > 0.5:
            name, zkind = "Живая изгородь", "veg"  # зелёный «забор» — это изгородь
        if name == "Живая изгородь" and (m & bushes).sum() > 0.6 * m.sum():
            continue  # отдельный куст, его уже размечает сцена
        if name != "Дерево":
            same = next((z for z in zones if z[0] == name), None)
            if same is not None:
                same[1] |= m
                continue
        zones.append([name, m, zkind])

    dil = ndimage.binary_dilation(bmask, iterations=max(2, round(0.01 * max(H, W)))) if bmask.any() else None
    roof = np.zeros((H, W), bool)
    parts = []
    for (name, b, s), m in zip(found, part_masks):
        if name == "Крыша":
            if _is_roof(m, bmask):
                roof |= m  # крыша — один слой
            continue
        if dil is not None:
            m = m & dil
        if name.startswith(QUAD) and m.sum() > 30:
            m = _quad(m) if _quad(m) is not None else m  # примитив: ровный четырёхугольник по контуру
        if m.sum() > 30:
            parts.append([name, m, "part"])
    # порядок слоёв (позже — поверх): крыша, рамы, стекло, остальные части; мелкие детали (светильники) — последними,
    # чтобы их не перекрывали рамки окон
    parts.sort(key=lambda p: (p[0] != "Крыша", -int(p[1].sum())))
    if roof.sum() > 30:
        parts.insert(0, ["Крыша", roof, "part"])
    if schema is not None:
        # стекло — со схемы, одним слоем-материалом; рамы и импосты — окна без стекла, тоже одним слоем; двери —
        # без своего остекления
        glass = _schema_glass(schema, dil)
        # каждое стекло — ровный четырёхугольник
        gl, gn = ndimage.label(glass)
        for k in range(1, gn + 1):
            q = _quad(gl == k)
            if q is not None:
                glass = (glass & (gl != k)) | q
        windows = np.zeros((H, W), bool)
        rest = []
        for p in parts:
            if p[0] == "Окно":
                windows |= p[1]
            else:
                if p[0] in ("Дверь", "Ворота"):
                    p[1] = p[1] & ~glass
                rest.append(p)
        frames = windows & ~glass
        mats = []
        if frames.sum() > 30:
            mats.append(["Рамы", frames, "material"])
        if glass.sum() > 30:
            mats.append(["Стекло", glass, "material"])
        # после крыши — рамы и стекло, поверх — двери, светильники и прочие части
        parts = rest[:1] + mats + rest[1:] if rest and rest[0][0] == "Крыша" else mats + rest
        log.info("INEGO zones: со схемы стекло %.1f%% кадра, рамы %.1f%%", 100 * glass.mean(), 100 * frames.mean())
    occupied = np.zeros((H, W), bool)
    for p in parts:
        occupied |= p[1]
    exact: set[str] = set()
    if schema is not None:
        # здание целиком по схеме: её края ровные, наши (SAM, детекторы, прижатие к фото) волнились
        open_mask = np.zeros((H, W), bool)
        for z in zones:
            if z[0] in ("Небо",) or z[2] == "veg":
                open_mask |= z[1]
        area = (dil if dil is not None else bmask) & ~open_mask
        floor = np.zeros((H, W), bool)
        for f, horizontal in _faces(rgb, bmask & ~occupied, geo):
            if horizontal:
                floor |= f
        part_masks = [("Окно", windows)] + [(p[0], p[1]) for p in parts if p[0] not in ("Крыша", "Стекло", "Рамы")]
        bzones = _schema_building(rgb, schema, area, glass, part_masks, roof if roof.sum() > 30 else None,
                                  floor, _features(rgb))
        exact = {z[0] for z in bzones}
        zones += bzones
        parts = []
    # стены — без частей; горизонтальные плоскости — покрытие; остальное — по материалам, слой на материал
    walls = bmask & ~occupied if schema is None else np.zeros((H, W), bool)
    floor = np.zeros((H, W), bool)
    if schema is None:
        for f, horizontal in _faces(rgb, walls, geo):
            if horizontal:
                floor |= f
        if floor.sum() > 0.002 * H * W:
            zones.append(["Покрытие", floor, "part"])
    walls &= ~floor
    if schema is None:
        feat = _features(rgb)
        pieces = _split_materials(rgb, feat, walls)
        groups = _material_groups(feat, pieces) if len(pieces) > 1 else [0]
        mats = {}
        for g, m in zip(groups, pieces):
            mats[g] = mats.get(g, np.zeros((H, W), bool)) | m
        mats = sorted(mats.values(), key=lambda m: -m.sum())
        for n, m in enumerate(mats, 1):
            zones.append(["Стены" if len(mats) == 1 else f"Стены · материал {n}", m, "material"])
    zones += parts

    # карта: позже — поверх
    idmap = np.zeros((H, W), np.int32)
    labels, kinds, counts = {}, {}, {}
    for i, (name, m, kind) in enumerate(zones, 1):
        counts[name] = counts.get(name, 0) + 1
        idmap[m] = i
        labels[i] = name if sum(1 for z in zones if z[0] == name) == 1 else f"{name} {counts[name]}"
        kinds[i] = kind
    pre = idmap.copy()
    exact_ids = np.array([i for i, n in labels.items() if n.split(" · ")[0].rstrip("0123456789 ") in exact
                          or n.rstrip("0123456789 ") in exact or n in exact] + [-99])
    idmap = _finish(idmap, rgb)
    free = {i: n for i, n in labels.items() if i not in set(exact_ids.tolist())}
    idmap = _straighten_arch(idmap, free, rgb)
    idmap = _quadify(idmap, free)
    # края здания — как в схеме: ни прижатие к фото, ни выпрямление их не трогают
    keep = np.isin(pre, exact_ids) | np.isin(idmap, exact_ids)
    idmap = np.where(keep, pre, idmap)
    idmap, alpha, vegid = _matte(idmap, labels, rgb)
    present = set(np.unique(idmap).tolist()) | set(np.unique(vegid).tolist())
    out = [{"id": i, "label": labels[i], "kind": kinds[i], "group": None} for i in labels if i in present]
    log.info("INEGO zones: %d зон за %.1f с: %s", len(out), time.time() - t0, ", ".join(z["label"] for z in out))
    return idmap, alpha, vegid, out


class InegoZoneMap:
    """Карта зон кадра (см. docstring модуля). Выход — файлы в output/inego и список зон в ui.zones."""

    @classmethod
    def INPUT_TYPES(cls):
        # schema — схема материалов (кадр плоскими цветами от модели генерации): из неё стекло
        return {"required": {"image": ("IMAGE",)}, "optional": {"schema": ("IMAGE",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "inego"

    def run(self, image, schema=None):
        _free_vram()
        rgb = _to_uint8(image[0])
        sch = None
        if schema is not None:
            sch = np.asarray(Image.fromarray(_to_uint8(schema[0])[..., :3]).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
        try:
            idmap, alpha, vegid, zones = build_zone_map(rgb, sch)
        finally:
            # ~4.5 ГБ моделей карты не держим в RAM рядом с Qwen и klein (25 ГБ у контейнера): следующая карта
            # загрузит их заново за ~10 с
            _models.clear()
            torch.cuda.empty_cache()
        if idmap.max() > 255:
            raise ValueError("Слишком много зон для 8-битной карты")
        folder = os.path.join(folder_paths.get_output_directory(), "inego")
        os.makedirs(folder, exist_ok=True)
        tag = uuid.uuid4().hex
        files = []
        for name, data in (("map", idmap.astype(np.uint8)), ("alpha", (alpha * 255 + 0.5).astype(np.uint8)),
                           ("veg", vegid.astype(np.uint8))):
            fn = f"zones_{tag}_{name}.png"
            Image.fromarray(data, "L").save(os.path.join(folder, fn))
            files.append({"filename": fn, "subfolder": "inego", "type": "output"})
        return {"ui": {"images": files, "zones": [json.dumps(zones, ensure_ascii=False)]}}
