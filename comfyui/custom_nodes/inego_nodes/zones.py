"""INEGO Zone Map — карта зон кадра для выбора готовых масок (вкладка Render pass).

Слои, поверх друг друга:
  1. сцена — EoMT-L 1280 (ADE20K panoptic): небо, газон, дорога, тротуар, деревья, кусты, машины, люди, здание.
     Лучшая из сравнённых (OneFormer, Mask2Former): только она на клей-рендере отделила дом от фона;
  2. части здания — детекторы по тексту: Grounding DINO base (окна, двери, ворота, крыша, балкон — рамки точнее всех)
     и OWLv2 (колонны, водостоки, трубы, ограждения — их Grounding DINO не видит); ложные — по форме и пересечениям;
  3. края частей — SAM 2.1 по рамкам;
  4. стены = здание − части; грани — по осям здания (MoGe-2 + линии кадра, как в InegoTextureProject),
     горизонтальные плоскости — «Покрытие»; внутри — куски отделки (SAM по сетке точек, однородные первыми),
     одинаковая отделка на разных гранях — одна группа («Отделка А» — весь камень, «Отделка Б» — вся штукатурка);
  5. края карты — крапинки соседям, границы прижаты к перепадам картинки (guided filter), прямые края архитектуры —
     по прямым кадра; растительность — мягкая альфа ViTMatte (ветки на фоне неба, жёсткая карта их рвёт).

Выход (OUTPUT_NODE): PNG карты (значение пикселя — id зоны, 0 — без зоны), PNG альфы растительности
и список зон {id, label, kind, group} в ui.zones (JSON).
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

from . import _device, _free_vram, _manhattan_faces, _manhattan_frame, _moge, _sam, _straighten, _to_uint8

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
    ("owl", "balcony railing", 0.22, "Ограждение"),
]
BUILDING = {"building", "house", "skyscraper", "tower"}
# классы ADE20K → подписи; «вещи» — отдельными экземплярами, остальное сливается по классу
SCENE = {
    "sky": "Небо", "grass": "Газон", "field": "Газон", "earth": "Земля", "sand": "Земля", "road": "Дорога",
    "sidewalk": "Тротуар", "path": "Дорожка", "floor": "Покрытие", "tree": "Дерево", "plant": "Кусты",
    "flower": "Цветы", "fence": "Забор", "wall": "Стена (фон)", "stairs": "Ступени", "step": "Ступени",
    "stairway": "Ступени", "car": "Машина", "truck": "Машина", "van": "Машина", "person": "Человек",
    "bicycle": "Велосипед", "minibike": "Мотоцикл", "pole": "Столб", "streetlight": "Фонарь", "bench": "Скамейка",
    "water": "Вода", "mountain": "Горы", "hill": "Холмы", "rock": "Камни", "signboard": "Вывеска",
    "awning": "Навес", "railing": "Ограждение", "column": "Колонна",
}
THINGS = {"car", "truck", "van", "person", "bicycle", "minibike", "tree", "bench", "pole", "streetlight"}
OPENINGS = {"Окно", "Дверь", "Ворота"}
ARCH = ("Крыша", "Отделка", "Окно", "Дверь", "Ворота", "Колонна", "Труба", "Балкон", "Покрытие", "Ступени",
        "Ограждение", "Водосток")
VEG = ("Дерево", "Кусты", "Цветы")
MAT_MIN = 0.03  # кусок отделки — не меньше этой доли стен
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


def _faces(rgb, walls):
    """Грани стен по осям здания; [(маска, горизонтальная)]."""
    if walls.sum() < 0.01 * walls.size:
        return [(walls, False)]
    src = rgb.astype(np.float32) / 255
    model = _moge()
    with _on_gpu(model):
        with torch.no_grad():
            g = model.infer(torch.from_numpy(src).permute(2, 0, 1).to(_device()))
    P = g["points"].float().cpu().numpy()
    Nn = g["normal"].float().cpu().numpy()
    K = g["intrinsics"].float().cpu().numpy()
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


def _features(rgb):
    """Признаки отделки на пиксель: цветность a, b (Lab), яркость/3 и фактура (локальный перепад яркости,
    усреднённый по окрестности: камень и кирпич — высокий, штукатурка — низкий)."""
    import cv2

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    tex = ndimage.uniform_filter(np.abs(L - ndimage.uniform_filter(L, 5)), max(5, rgb.shape[1] // 120))
    return np.stack([a, b, L / 3, tex * 3], -1)


def _inhomogeneity(feat, m):
    v = feat[m][:: max(1, int(m.sum()) // 4000)]
    return float(v.std(0).sum())


def _split_materials(rgb, feat, walls):
    """Куски отделки: SAM по сетке точек; однородные куски первыми (маска «вся стена» с камнем и штукатуркой
    разнородна и проигрывает); остаток — куску с похожей отделкой (иначе граница шла дугой там, где стыка нет)."""
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
            m = ms[i, j]
            inw = m & walls
            if inw.sum() >= MAT_MIN * total and inw.sum() > 0.5 * m.sum() and sc[i, j] > 0.7:
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
    f = ndimage.gaussian_filter(feat, (2, 2, 0))
    cost = []
    for m in segs:
        mu, sd = f[m].mean(0), f[m].std(0) + 1.0
        cost.append((((f - mu) / sd) ** 2).sum(-1))
    lab = np.argmin(np.stack(cost), 0) + 1
    for i, m in enumerate(segs, 1):
        lab[m] = i
    sig = max(1.5, 0.004 * max(walls.shape))
    votes = np.stack([ndimage.gaussian_filter(((lab == i) & walls).astype(np.float32), sig) for i in range(1, len(segs) + 1)])
    lab = np.where(walls, np.argmax(votes, 0) + 1, 0)
    return [lab == i for i in range(1, len(segs) + 1)]


def _material_groups(feat, segs):
    """Одинаковая отделка на разных гранях — одна группа: цветность и фактура строго, яркость (свет граней разный)
    — с допуском."""
    feats = []
    for m in segs:
        a, b, L3, t3 = (float(np.median(feat[..., c][m])) for c in range(4))
        feats.append((L3 * 3, a, b, t3 / 3 + 0.3))
    group = list(range(len(segs)))

    def root(i):
        while group[i] != i:
            i = group[i]
        return i

    for i in range(len(segs)):
        for j in range(i):
            fi, fj = feats[i], feats[j]
            if (abs(fi[1] - fj[1]) < 4 and abs(fi[2] - fj[2]) < 5 and abs(fi[0] - fj[0]) < 22
                    and max(fi[3], fj[3]) / min(fi[3], fj[3]) < 1.35):
                group[root(i)] = root(j)
    return [root(i) for i in range(len(segs))]


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


def _matte(idmap, labels, rgb):
    """Растительность: ViTMatte по тримапу вокруг зоны — мягкая альфа (ветки на фоне неба). В карте зона берёт
    пиксели с альфой > 0.5 у неба и фона и отдаёт с альфой < 0.5 соседям."""
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    H, W = idmap.shape
    out = idmap.copy()
    alpha = np.zeros((H, W), np.float32)
    veg = [i for i, n in labels.items() if n.startswith(VEG) and (idmap == i).sum() >= 400]
    if not veg:
        return out, alpha
    proc, model = _load("matte", lambda: (VitMatteImageProcessor.from_pretrained(MATTE_ID, cache_dir=CACHE),
                                          VitMatteForImageMatting.from_pretrained(MATTE_ID, cache_dir=CACHE).eval()))
    open_ids = [j for j, n in labels.items() if n in ("Небо", "Стена (фон)")] + [0]
    with _on_gpu(model):
        for i in veg:
            m = idmap == i
            ys, xs = np.nonzero(m)
            size = max(np.ptp(ys), np.ptp(xs)) + 1
            grow = max(6, round(0.04 * size))
            shrink = max(1, round(0.004 * size))  # ствол и толстые ветки тонкие — сильное сжатие их съедает
            fg = ndimage.binary_erosion(m, iterations=shrink)
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
            alpha = np.maximum(alpha, full)
            unknown = ~fg & ~bg
            out[unknown & (full > 0.5) & np.isin(out, open_ids)] = i
            give = unknown & (full < 0.5) & (out == i)
            out = _fill(np.where(give, -1, out), give)
    return out, alpha


def build_zone_map(rgb: np.ndarray):
    """(карта id (H, W) int32, альфа растительности (H, W) float32, [{id, label, kind, group}])."""
    img = Image.fromarray(rgb)
    H, W = rgb.shape[:2]
    t0 = time.time()
    segs = _scene(img)
    bmask = np.zeros((H, W), bool)
    for lab, m in segs:
        if lab in BUILDING:
            bmask |= m
    near = ndimage.binary_dilation(bmask, iterations=max(3, round(0.02 * max(H, W)))) if bmask.any() else None
    found = []
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
    found = _clean_parts(found, bmask, H, W)
    masks = _sam_boxes(rgb, [f[1] for f in found])

    zones = []  # [подпись, маска, вид, группа]
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
        zones.append([name, m.copy(), kind, None])
    parts = []
    dil = ndimage.binary_dilation(bmask, iterations=max(2, round(0.01 * max(H, W)))) if bmask.any() else None
    for (name, b, s), m in zip(found, masks):
        if name != "Крыша" and dil is not None:
            m = m & dil
        if m.sum() > 30:
            parts.append([name, m, "part", None])
    occupied = np.zeros((H, W), bool)
    for p in parts:
        occupied |= p[1]
    walls = bmask & ~occupied
    fs = _faces(rgb, walls)
    vert = np.zeros((H, W), bool)
    for f, horizontal in fs:
        if horizontal:
            zones.append(["Покрытие", f, "part", None])
        else:
            vert |= f
    feat = _features(rgb)
    segs_w, owners = [], []
    for m in _split_materials(rgb, feat, vert):
        for k, (f, horizontal) in enumerate(fs):
            if not horizontal and (m & f).sum() >= 0.004 * vert.sum():
                segs_w.append(m & f)
                owners.append(k)
    groups = _material_groups(feat, segs_w) if segs_w else []
    order = sorted(set(groups), key=lambda g: -sum(segs_w[i].sum() for i in range(len(groups)) if groups[i] == g))
    letter = {g: "АБВГДЕЖЗИК"[n] if n < 10 else str(n + 1) for n, g in enumerate(order)}
    nfaces = len(set(owners))
    for i, m in enumerate(segs_w):
        g = letter[groups[i]]
        zones.append([f"Отделка {g}" + (f" · грань {owners[i] + 1}" if nfaces > 1 else ""), m, "finish", g])
    zones += parts

    # карта: позже — поверх
    idmap = np.zeros((H, W), np.int32)
    labels, meta, counts = {}, {}, {}
    for i, (name, m, kind, group) in enumerate(zones, 1):
        counts[name] = counts.get(name, 0) + 1
        idmap[m] = i
        labels[i] = name if sum(1 for z in zones if z[0] == name) == 1 else f"{name} {counts[name]}"
        meta[i] = (kind, group)
    idmap = _finish(idmap, rgb)
    idmap = _straighten_arch(idmap, labels, rgb)
    idmap, alpha = _matte(idmap, labels, rgb)
    present = set(np.unique(idmap).tolist())
    out = [{"id": i, "label": labels[i], "kind": meta[i][0], "group": meta[i][1]} for i in labels if i in present]
    log.info("INEGO zones: %d зон за %.1f с: %s", len(out), time.time() - t0, ", ".join(z["label"] for z in out))
    return idmap, alpha, out


class InegoZoneMap:
    """Карта зон кадра (см. docstring модуля). Выход — файлы в output/inego и список зон в ui.zones."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "inego"

    def run(self, image):
        _free_vram()
        rgb = _to_uint8(image[0])
        try:
            idmap, alpha, zones = build_zone_map(rgb)
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
        for name, data in (("map", idmap.astype(np.uint8)), ("alpha", (alpha * 255 + 0.5).astype(np.uint8))):
            fn = f"zones_{tag}_{name}.png"
            Image.fromarray(data, "L").save(os.path.join(folder, fn))
            files.append({"filename": fn, "subfolder": "inego", "type": "output"})
        return {"ui": {"images": files, "zones": [json.dumps(zones, ensure_ascii=False)]}}
