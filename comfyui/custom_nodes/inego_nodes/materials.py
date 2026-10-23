"""Render pass: кадр по элементам (нод InegoMaterialGroups).

Qwen-Image-Edit-2511 перерисовывает кадр плоской иллюстрацией без света (схема, бэкенд backend/app/materials.py):
каждый элемент — заливка своего цвета, между элементами — тонкие контурные линии. Схема — как раскраска: элемент —
связная заливка между линиями и перепадами цвета; линии и перепады делятся пополам между соседними заливками.
Заливки собираются в элементы только по самой схеме, фото в нарезке не участвует (см. _group). Края —
прямые и чёткие из самой иллюстрации: у SAM маска в 256², края плыли, а узкую доску карниза он не видел.
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

from . import InegoAlign, _device, _free_vram, _models, _sam, _to_uint8

log = logging.getLogger("inego")

MAX_SIDE = 2048  # карта строится в этом размере по длинной стороне; фронтенд растягивает её на кадр
# граница заливок: перепад цвета через пиксель (ΔE Lab, |p+1 − p−1|) больше EDGE_CUT — переход сглажен на 2–3 px,
# и соседний шаг между похожими заливками (кремовая стена и бежевый камень) по отдельности порога не берёт
EDGE_CUT = 6.0
# контурная линия: темнее закрытия окном LINE_SIZE (линии 2511 в размере карты — 4–6 px) на LINE_CUT по светлоте
LINE_SIZE = 7
LINE_CUT = 15.0
# раздел линий водоразделом: компактность — рост от заливки дорожает с расстоянием, и тонкая тёмная заливка (водосток,
# ветка) не растекается по контурам на десятки пикселей, разрезая стену
LINE_REACH = 6.0
# SAM 2.1 по фото (_sam_refine): элемент флета — подсказка (рамка + точка), граница — по фото
SAM_PAD = 0.25  # кроп вокруг элемента: его рамка с запасом в такую долю её стороны
SAM_CROP_MIN = 160  # и не меньше стольких px — у мелких (фонарь, водосток) маска SAM 256² ложится на сам элемент
SAM_CROP_MAX = 640  # кроп больше — у SAM в нём 2.5+ px кадра на клетку маски, край зубчатый: элемент (стена,
# кровля, газон) остаётся с флета, его прямые точнее
SAM_IOU = 0.5  # вариант SAM берётся, если совпадает с элементом флета хотя бы так; иначе остаётся флет
POLY_EPS = 1.0  # контур элемента → многоугольник: отклонение от пиксельного контура схемы до стольких px
POLY_EPS_SAM = 2.0  # то же для карты после SAM (в px карты): дрожание маски SAM в пару px — в прямую
ELEMENT_MIN = 0.00002  # заливка мельче этой доли кадра — обрывок (кусок толстой рамы, листик), целиком уходит к соседу
# сборка заливок в элементы (_group) — только по схеме, фото не участвует
SAME_CUT = 5.0  # один цвет на схеме: ΔE Lab медиан заливок меньше
ENCLOSE_BORDER = 0.75  # окружённая: такая доля границы — с одним соседом
ENCLOSE_MAX = 0.01  # и он не крупнее этой доли кадра (рама — да, стена — нет: фонарь на стене остаётся собой)
OPENING_MAX = 0.012  # проём: всё, что целиком внутри одного элемента и не крупнее этой доли кадра, — один элемент
SKY_CRUMB = 0.0005  # обрывки (листья) целиком внутри неба мельче этой доли кадра — небо
# альбедо (InegoIntrinsicAlbedo): Intrinsic v2 (Careaga & Aksoy, «Colorful Diffuse Intrinsic Image Decomposition in
# the Wild», TOG 2024; только академическое использование). Веса и бэкбоны — через torch.hub в volume
INTRINSIC_DIR = os.path.join(folder_paths.models_dir, "intrinsic")


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 (..., 3) → CIE Lab (D65)."""
    c = rgb.astype(np.float32) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)



def _fills(schema: np.ndarray) -> np.ndarray:
    """Заливки схемы [H, W] (1..N): связные области между контурными линиями и перепадами цвета. Линии и перепады
    достаются соседним заливкам водоразделом по перепаду цвета: граница ложится на самый резкий шаг — прямой край
    «стена → чёрная рама», а не середину линии (толщина линии 2511 гуляет, середина виляла). Обрывки (кусок толстой
    рамы между линиями, листик) — целиком соседней заливке ближайшего цвета: попиксельно их рвало между стеной и
    стеклом, и рама шла зубцами."""
    from skimage.segmentation import watershed

    # медиана 3×3 гасит шум и сжатие заливок, линии и края не сдвигает
    L = _lab(ndimage.median_filter(schema, size=(3, 3, 1)))
    grad = np.zeros(schema.shape[:2], np.float32)
    grad[:, 1:-1] = np.linalg.norm(L[:, 2:] - L[:, :-2], axis=2)
    grad[1:-1] = np.maximum(grad[1:-1], np.linalg.norm(L[2:] - L[:-2], axis=2))
    Y = L[..., 0]
    line = ndimage.grey_closing(Y, size=(LINE_SIZE, LINE_SIZE)) - Y > LINE_CUT
    region, n = ndimage.label(~(line | (grad > EDGE_CUT)))
    size = np.bincount(region.ravel(), minlength=n + 1)
    crumb = size < ELEMENT_MIN * region.size
    crumb[0] = False
    bits = np.where(crumb[region], region, 0)
    region[crumb[region]] = 0
    region = watershed(grad, region, compactness=LINE_REACH)
    for k, sl in enumerate(ndimage.find_objects(bits), 1):
        if sl is None:
            continue
        sl = tuple(slice(max(t.start - 3, 0), t.stop + 3) for t in sl)
        m = bits[sl] == k
        near = np.unique(region[sl][ndimage.binary_dilation(m, iterations=3) & ~m])
        if len(near):
            c = np.median(L[sl][m], 0)
            dist = [np.linalg.norm(np.median(L[sl][region[sl] == j], 0) - c) for j in near]
            region[sl][m] = near[int(np.argmin(dist))]
    _, region = np.unique(region, return_inverse=True)
    return region.reshape(schema.shape[:2]).astype(np.int32) + 1


def _group(region: np.ndarray, schema: np.ndarray) -> np.ndarray:
    """Заливки → элементы только по схеме: стена, кровля, карниз, водосток, окно, газон.

    1. соседние заливки одного цвета — один элемент (плоскости стены, разрезанные контуром ребра; скаты кровли);
       несоседние не сводятся никогда — одинаковый цвет у 2511 бывает у разных материалов (камень и плитка,
       кровля и мощение), а живая изгородь слева и куст спереди — разные элементы;
    2. небо — заливка, которой больше всего в верхнем ряду; куски его цвета между ветками (их окружает то, что
       касается неба) — тоже небо. Стекло в раме небесного цвета остаётся окну: рама неба не касается;
    3. проём: заливки, вместе целиком лежащие внутри одного элемента (дыра в его маске) и вместе не крупнее
       OPENING_MAX, — один элемент: окно со стёклами, переплётом и панелями, дверь с филёнками. У 2511 рама часто —
       чёрная контурная линия, а не заливка, и без этого окно рассыпается на стёкла. Внутри неба мелкие дыры
       (листья) — само небо;
    4. некрупная заливка, на ENCLOSE_BORDER своей границы окружённая некрупной, — её часть (стёкла в раме, когда
       окно касается и стены, и цоколя); стена в раму не уходит. Мелкий обрывок, граничащий больше всего с небом
       (лист на ветке), — небо."""
    n = int(region.max())
    N = region.size
    idx = np.arange(n + 1)
    area = np.bincount(region.ravel(), minlength=n + 1).astype(np.float64)
    L = _lab(schema)
    col = np.nan_to_num(np.stack([ndimage.median(L[..., i], region, idx) for i in range(3)], 1))
    parent = np.arange(n + 1)

    def root(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def union(x, y):
        x, y = root(x), root(y)
        if x != y:
            parent[y] = x

    def same(i, j):
        return np.linalg.norm(col[i] - col[j]) < SAME_CUT

    a = np.concatenate([region[:, :-1].ravel(), region[:-1].ravel()])
    b = np.concatenate([region[:, 1:].ravel(), region[1:].ravel()])
    d = a != b
    key, cnt = np.unique(np.minimum(a[d], b[d]).astype(np.int64) * (n + 1) + np.maximum(a[d], b[d]), return_counts=True)
    a, b = key // (n + 1), key % (n + 1)
    for i, j in zip(a, b):
        if same(i, j):
            union(i, j)

    def borders():
        g = np.array([root(k) for k in range(n + 1)])
        out: dict = {}
        for x, y, c in zip(g[a], g[b], cnt):
            if x != y:
                out.setdefault(x, {}).setdefault(y, 0)
                out[x][y] += c
                out.setdefault(y, {}).setdefault(x, 0)
                out[y][x] += c
        return g, out

    sky = root(np.bincount(region[0]).argmax())
    g, border = borders()
    touches_sky = {x for x, nb in border.items() if sky in nb}
    for x, nb in border.items():
        y, c = max(nb.items(), key=lambda t: t[1])
        if x != sky and same(x, sky) and c >= ENCLOSE_BORDER * sum(nb.values()) and y in touches_sky:
            union(sky, x)
    g = np.array([root(k) for k in range(n + 1)])
    garea = np.bincount(g, weights=area, minlength=n + 1)
    owner = g[region]
    for y in np.flatnonzero(garea > OPENING_MAX * N):
        ys, xs = np.nonzero(owner == y)
        sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        m = owner[sl] == y
        holes, k = ndimage.label(ndimage.binary_fill_holes(m) & ~m)
        for h, hs in enumerate(ndimage.find_objects(holes), 1):
            inside = owner[sl][hs][holes[hs] == h]
            if y == root(sky) and inside.size < SKY_CRUMB * N:
                for x in np.unique(inside):
                    union(sky, x)
            elif y != root(sky) and inside.size < OPENING_MAX * N:
                inside = np.unique(inside)
                for x in inside[1:]:
                    union(inside[0], x)
    for _ in range(3):
        g, border = borders()
        garea = np.bincount(g, weights=area, minlength=n + 1)
        changed = False
        for x, nb in border.items():
            y, c = max(nb.items(), key=lambda t: t[1])
            tot = sum(nb.values())
            enclosed = c >= ENCLOSE_BORDER * tot and max(garea[x], garea[y]) < ENCLOSE_MAX * N
            crumb = y == root(sky) and c >= 0.5 * tot and garea[x] < SKY_CRUMB * N  # лист на ветке — небо
            if (enclosed or crumb) and x != root(sky) and root(x) != root(y):
                union(y, x)
                changed = True
        if not changed:
            break
    return np.array([root(k) for k in range(n + 1)])[region]


def _transfer(owner: np.ndarray, size: tuple[int, int], align, eps: float = POLY_EPS) -> np.ndarray:
    """Элементы схемы → карта кадра [H, W] 1 в 1: каждый элемент — многоугольник по его контуру на схеме (стороны —
    прямые, лесенка пикселей и дрожание в пиксель сглажены), перенесённый тем же масштабом и сдвигом, что совмещают
    схему с кадром, и закрашенный в размере карты. Щели в пиксель между соседями — ближайшему элементу."""
    import cv2

    h, w = owner.shape
    H, W = size
    s, tx, ty = align or (1.0, 0.0, 0.0)
    out = np.zeros((H, W), np.int32)
    # крупные первыми: контур мелкого внутри крупного ложится поверх, дыры крупного не закрашиваются (even-odd)
    for k in np.argsort(-np.bincount(owner.ravel())):
        if k == 0:
            continue
        m = (owner == k).astype(np.uint8)
        if not m.any():
            break
        cs, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        polys = []
        for c in cs:
            c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64) + 0.5
            if len(c) < 3:
                continue
            c[:, 0] = (s * c[:, 0] / w + tx) * W
            c[:, 1] = (s * c[:, 1] / h + ty) * H
            polys.append(np.round(c * 16).astype(np.int32))  # 4 бита дробной части — без лишней лесенки
        if polys:
            cv2.fillPoly(out, polys, int(k), lineType=cv2.LINE_8, shift=4)
    if (out == 0).any():
        _, (iy, ix) = ndimage.distance_transform_edt(out == 0, return_indices=True)
        out = out[iy, ix]
    return out


def _sam_refine(rgb: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Элементы флета → границы по фото (SAM 2.1). 2511 перерисовывает кадр не пиксель в пиксель: окно на флете
    уже или сдвинуто на несколько px, и перенесённый контур ложится мимо рамы на фото. Флет знает, какие элементы
    есть и где, SAM — где их край на фото: для каждого элемента — кроп вокруг него (у мелких маска 256² покрывает
    сам элемент, а не весь кадр), подсказка — рамка элемента и его самая глубокая точка, из трёх вариантов SAM —
    ближайший к элементу флета. Крупные (кроп больше SAM_CROP_MAX) остаются с флета: на кадр целиком маска SAM
    грубая, край зубчатый. Пиксель достаётся элементу с самым уверенным ответом SAM; где SAM не уверен ни в
    ком — элементу флета."""
    processor, model = _sam()
    H, W = flat.shape
    best = np.zeros((H, W), np.float32)  # логит победителя; 0 — порог «SAM уверен»
    out = flat.copy()
    dev = _device()
    model.to(dev)
    used = 0
    try:
        for k in np.argsort(-np.bincount(flat.ravel())):
            if k == 0:
                continue
            m = flat == k
            if not m.any():
                break
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            pad = max(SAM_PAD * max(y1 - y0, x1 - x0), (SAM_CROP_MIN - min(y1 - y0, x1 - x0)) / 2, 8)
            cy0, cy1 = int(max(0, y0 - pad)), int(min(H, y1 + pad))
            cx0, cx1 = int(max(0, x0 - pad)), int(min(W, x1 + pad))
            if max(cy1 - cy0, cx1 - cx0) > SAM_CROP_MAX:
                continue
            mc = m[cy0:cy1, cx0:cx1]
            dist = ndimage.distance_transform_edt(np.pad(mc, 1))[1:-1, 1:-1]
            py, px = np.unravel_index(np.argmax(dist), dist.shape)
            box = [float(x0 - cx0), float(y0 - cy0), float(x1 - cx0), float(y1 - cy0)]
            inputs = processor(images=rgb[cy0:cy1, cx0:cx1], input_boxes=[[box]], input_points=[[[[float(px), float(py)]]]],
                               input_labels=[[[1]]], return_tensors="pt").to(dev)
            with torch.no_grad():
                res = model(**inputs, multimask_output=True)
            logit = processor.post_process_masks(res.pred_masks.float().cpu(), inputs["original_sizes"], binarize=False)[0][0].numpy()
            pos = logit > 0
            inter = (pos & mc).sum((1, 2))
            iou = inter / np.maximum((pos | mc).sum((1, 2)), 1)
            i = int(np.argmax(iou))
            if iou[i] < SAM_IOU:
                continue
            used += 1
            lg = logit[i]
            # своё по флету, но не по фото — соседям (ниже, ближайшему элементу)
            out[cy0:cy1, cx0:cx1][mc & ~pos[i] & (out[cy0:cy1, cx0:cx1] == k)] = 0
            win = lg > best[cy0:cy1, cx0:cx1]
            out[cy0:cy1, cx0:cx1][win] = k
            best[cy0:cy1, cx0:cx1][win] = lg[win]
    finally:
        model.to("cpu")
        torch.cuda.empty_cache()
    log.info("INEGO materials: SAM уточнил %d элементов из %d", used, len(np.unique(flat)))
    # крошки, отрезанные от элемента чужой маской, — соседям; заметные куски остаются элементу
    for k in np.unique(out):
        comp, n = ndimage.label(out == k)
        if n > 1:
            sizes = np.bincount(comp.ravel())
            sizes[0] = 0
            out[(sizes[comp] < ELEMENT_MIN * 5 * out.size) & (comp != np.argmax(sizes)) & (comp > 0)] = 0
    if (out == 0).any():
        _, (iy, ix) = ndimage.distance_transform_edt(out == 0, return_indices=True)
        out = out[iy, ix]
    return out


def build_material_map(rgb: np.ndarray, schema: np.ndarray, align=None) -> tuple[np.ndarray, list[dict], list[list]]:
    """Карта элементов [H, W] кадра (1..N, uint16) и список элементов по убыванию площади. Схема режется в своём
    размере и без совмещения; align — (s, tx, ty) InegoAlign.transform: перенос схемы на место кадра. Края
    элементов уточняет по фото SAM 2.1 (_sam_refine)."""
    t0 = time.time()
    region = _fills(schema)
    owner = _transfer(_group(region, schema), rgb.shape[:2], align)
    # границы — по фото (SAM), затем снова многоугольники: стороны прямые
    owner = _transfer(_sam_refine(rgb, owner), rgb.shape[:2], None, POLY_EPS_SAM)
    # номера — по убыванию площади
    area = np.bincount(owner.ravel())
    order = [k for k in np.argsort(-area) if k and area[k]][:65535]
    remap = np.zeros(len(area), np.int32)
    remap[order] = np.arange(1, len(order) + 1)
    idmap = remap[owner].astype(np.uint16)
    n = len(order)
    cnt = np.maximum(np.bincount(idmap.ravel(), minlength=n + 1), 1)
    col = np.stack([np.bincount(idmap.ravel(), weights=rgb[..., i].ravel(), minlength=n + 1) for i in range(3)], 1)
    col = (col / cnt[:, None]).astype(int)
    groups = [{"id": k, "label": f"Элемент {k}", "area": round(float(cnt[k]) / idmap.size, 5),
               "color": "#%02x%02x%02x" % tuple(col[k])} for k in range(1, n + 1)]
    log.info("INEGO materials: %d заливок → %d элементов за %.1f с", int(region.max()), n, time.time() - t0)
    return idmap, groups, []


def _intrinsic():
    if "intrinsic" not in _models:
        from intrinsic.pipeline import load_models

        torch.hub.set_dir(f"{INTRINSIC_DIR}/hub")
        # бэкбоны (ResNeXt WSL, EfficientNet) модель берёт через torch.hub.load, а он без терминала спрашивает
        # «доверять ли репозиторию» и падает на EOF — на время загрузки вопрос снят
        check = torch.hub._check_repo_is_trusted
        torch.hub._check_repo_is_trusted = lambda *a, **k: None
        try:
            _models["intrinsic"] = load_models("v2", device="cpu")
        finally:
            torch.hub._check_repo_is_trusted = check
    return _models["intrinsic"]


def intrinsic_albedo(rgb: np.ndarray) -> np.ndarray:
    """Альбедо кадра (sRGB uint8, размер кадра): тени и солнце сняты, детали (рамы, стёкла, фонари) на месте —
    в отличие от Marigold-IID (768 px), Intrinsic считает в высоком разрешении."""
    from intrinsic.pipeline import run_pipeline

    models = _intrinsic()
    dev = _device()
    for m in models.values():
        if isinstance(m, torch.nn.Module):
            m.to(dev)
    try:
        with torch.no_grad():
            alb = run_pipeline(models, rgb.astype(np.float32) / 255, device=dev)["hr_alb"]
    finally:
        for m in models.values():
            if isinstance(m, torch.nn.Module):
                m.to("cpu")
        torch.cuda.empty_cache()
    alb = _to_srgb_np(alb / max(float(np.quantile(alb, 0.995)), 1e-6))
    out = Image.fromarray((alb * 255 + 0.5).astype(np.uint8))
    return np.asarray(out.resize((rgb.shape[1], rgb.shape[0]), Image.LANCZOS))


def _to_srgb_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


class InegoIntrinsicAlbedo:
    """Кадр без света для схемы материалов: Qwen-Image-2.1 по фото переносил в схему тень под навесом и солнце на
    фронтоне (стена серо-бежевая, фронтон кремовый), по альбедо Intrinsic штукатурка — одного цвета."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "inego"

    def run(self, image):
        _free_vram()
        t0 = time.time()
        try:
            alb = intrinsic_albedo(_to_uint8(image[0])[..., :3])
        finally:
            _models.pop("intrinsic", None)
        log.info("INEGO intrinsic: альбедо %dx%d за %.1f с", alb.shape[1], alb.shape[0], time.time() - t0)
        return (torch.from_numpy(alb.astype(np.float32) / 255)[None],)


def _placed(schema: np.ndarray, size: tuple[int, int], align) -> Image.Image:
    """Схема в размере кадра на его месте (тот же перенос, что у карты)."""
    H, W = size
    s, tx, ty = align or (1.0, 0.0, 0.0)
    im = Image.fromarray(schema).resize((W, H), Image.LANCZOS)
    # PIL берёт обратное отображение: точка кадра X → точка схемы (X − tx·W) / s
    return im.transform((W, H), Image.AFFINE, (1 / s, 0, -tx * W / s, 0, 1 / s, -ty * H / s), Image.BICUBIC)


class InegoMaterialGroups:
    """Кадр по элементам (см. docstring модуля): image — кадр, schema — его плоская иллюстрация (2511, как есть —
    совмещение с кадром нод считает сам и переносит уже готовые многоугольники). Выход — карта в output/inego и список элементов в ui.groups."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "schema": ("IMAGE",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "inego"

    def run(self, image, schema):
        rgb = _to_uint8(image[0])[..., :3]
        if max(rgb.shape[:2]) > MAX_SIDE:
            im = Image.fromarray(rgb)
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            rgb = np.asarray(im)
        # схема режется в своём размере (~1 Мп) как есть, на кадр переносятся уже многоугольники элементов
        sch = _to_uint8(schema[0])[..., :3]
        align = InegoAlign.transform(schema[0][..., :3], image[0][..., :3])
        _free_vram()  # 2511 из того же workflow — из видеопамяти, рядом SAM
        try:
            idmap, groups, pairs = build_material_map(rgb, sch, align)
        finally:
            _models.pop("sam", None)
        folder = os.path.join(folder_paths.get_output_directory(), "inego")
        os.makedirs(folder, exist_ok=True)
        fn = f"materials_{uuid.uuid4().hex}.png"
        # карта — RGB PNG: id = R + 256·G (элементов бывает больше 255)
        rgbmap = np.zeros((*idmap.shape, 3), np.uint8)
        rgbmap[..., 0] = idmap & 255
        rgbmap[..., 1] = idmap >> 8
        Image.fromarray(rgbmap, "RGB").save(os.path.join(folder, fn))
        files = [{"filename": fn, "subfolder": "inego", "type": "output"}]
        # сама плоская схема на месте кадра — во вкладке её можно показать рядом с кадром
        fn = fn.replace(".png", "_schema.jpg")
        _placed(sch, rgb.shape[:2], align).save(os.path.join(folder, fn), quality=92)
        files.append({"filename": fn, "subfolder": "inego", "type": "output"})
        return {"ui": {"images": files,
                       "groups": [json.dumps(groups, ensure_ascii=False)], "pairs": [json.dumps(pairs)]}}
