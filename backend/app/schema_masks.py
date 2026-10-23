"""Маски по схеме 2511: каждая область схемы — своя маска, контуры между ними — отдельная маска (разделитель).

Схема — мультяшная раскраска: плоские заливки и тёмные контуры. Режется как ведром в Photoshop: заливка от «клика»
идёт по пикселям, близким к цвету клика, и упирается в линию. Цвет нужен, потому что 2511 проводит контур не везде
(небо и изгородь делит только цвет); через короткие разрывы линии заливка не протекает — барьер на первом проходе
расширен (как «закрывать разрывы» в Clip Studio), на втором узкие элементы (рамы) доливаются по тонкой линии.
Одна заливка — одна область, поэтому границы без линии или смены цвета внутри области не бывает.

1. линия — узкий тёмный штрих (black-hat по яркости); рамы и тёмные заливки шире и остаются областями;
2. клик — в самой дальней от барьера и залитого точке; цвет сравнивается в OKLab с пониженным весом яркости
   (лёгкий градиент на карнизе не дробит его, а серые крыша, дорожка и рамы различаются: в HSL у серого тон
   не определён);
3. полоска у линии, съеденная расширением барьера, и мелочь отходят ближайшей области; линия делится между
   соседями по своей оси — под разделителем у каждого пикселя тоже есть область;
4. края — многоугольники; разделитель — сама линия схемы пиксель в пиксель, поверх областей.
"""

import cv2
import numpy as np
from scipy import ndimage

L_WEIGHT = 0.5  # вес яркости OKLab против цветности
TOLERANCE = 6.0  # допуск заливки от цвета клика по каждой оси OKLab×100 (яркость — с весом L_WEIGHT)
LINE_BH = 20  # ядро линии на столько темнее окружения в окне 5×5 (L из Lab OpenCV, 0…255)
# проходы заливки: (расширение барьера, px; клик не ближе стольких px к барьеру). Первый закрывает разрывы
# линии до ~2·3 px, второй доливает узкие элементы — протечь ему некуда, залитое тоже барьер
PASSES = ((3, 6.0), (1, 2.0))
MIN_REGION = 150  # области меньше (px кадра) — к соседям
POLY_EPS = 1.0  # допуск упрощения многоугольников (px кадра)
# доводка маски по фото (snap_to_photo): схема совмещена с кадром с точностью ~1% — край ищется в полосе ± этой
# доли кадра; разошлось с маской больше SNAP_MAX её площади — GrabCut промахнулся, маска остаётся как была
SNAP_BAND = 0.008
SNAP_MAX = 0.35
# выпрямление края по прямым фото (_straighten, как у InegoSurfaceMask): полоса вокруг прямой — доля размера маски;
# отрезок не короче этой доли; не больше стольких прямых; длинный отрезок (не короче STRAIGHT_LONG), от которого
# край отходит дальше узкой полосы, — с полосой до STRAIGHT_FAR
STRAIGHT_TOL = 0.012
STRAIGHT_MIN_SEG = 0.05
STRAIGHT_MAX_SEGS = 40
STRAIGHT_LONG = 0.08
STRAIGHT_FAR = 0.02

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)


def oklab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.cbrt(c @ _M1.T) @ _M2.T


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def line_mask(rgb: np.ndarray) -> np.ndarray:
    """Нарисованные контуры: узкий (до ~4 px) тёмный штрих — black-hat с окном 5×5. Рамы окон и тёмные заливки
    шире (рама 6–8 px против 2–3 px у линии), в окно не помещаются и остаются областями. Порога по яркости нет:
    штрих на серой отмостке или крыше светлее чёрного и с ним рвался в пунктир."""
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, _disk(2))
    core = (bh > LINE_BH).astype(np.uint8)
    # сглаженные края штриха — рядом с ядром и заметно темнее окружения
    line = (core | ((cv2.dilate(core, _disk(1)) > 0) & (bh > LINE_BH // 2))).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(line, connectivity=8)
    return np.where(st[lbl, cv2.CC_STAT_AREA] >= 12, line, 0).astype(np.uint8)


def _fill(okw: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Заливки ведром: клик в самой дальней от барьера точке, заливка с допуском от цвета клика."""
    h, w = line.shape
    lbl = np.zeros((h, w), np.int32)
    tol = (TOLERANCE,) * 3
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    nxt = 1
    for k, (grow, min_dist) in enumerate(PASSES):
        wall = cv2.dilate(line, _disk(grow)) if grow else line
        if k:
            _creep(lbl, okw, wall, PASSES[k - 1][0] - grow + 1)
        mask = np.zeros((h + 2, w + 2), np.uint8)  # floodFill не заходит туда, где маска не 0
        while True:
            mask[1:-1, 1:-1] = ((wall > 0) | (lbl != 0)).astype(np.uint8)
            dist = cv2.distanceTransform(1 - mask[1:-1, 1:-1], cv2.DIST_L2, 3)
            y, x = np.unravel_index(int(dist.argmax()), dist.shape)
            if dist[y, x] < min_dist:
                break
            cv2.floodFill(okw, mask, (int(x), int(y)), 0, tol, tol, flags)
            got = mask[1:-1, 1:-1] == 255
            if got.sum() < 4:  # клик в одиночный пиксель шума — не область, но больше не кликать сюда
                lbl[y, x] = -1
                continue
            lbl[got] = nxt
            nxt += 1
    lbl[lbl < 0] = 0
    return lbl


def _creep(lbl: np.ndarray, okw: np.ndarray, wall: np.ndarray, steps: int) -> None:
    """Залитое дорастает до нового, более тонкого барьера по пикселям своего цвета — полоска у линии, которую съело
    прошлое расширение барьера, иначе зальётся следующим проходом отдельной каёмкой вдоль каждой линии."""
    n = lbl.max() + 1
    cnt = np.maximum(np.bincount(lbl[lbl > 0], minlength=n), 1)[:, None]
    mean = np.stack([np.bincount(lbl[lbl > 0], okw[..., i][lbl > 0], n) for i in range(3)], 1) / cnt
    for _ in range(steps):
        d = cv2.dilate(np.maximum(lbl, 0).astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        cand = (lbl == 0) & (wall == 0) & (d > 0)
        ys, xs = np.nonzero(cand)
        ok = (np.abs(okw[ys, xs] - mean[d[ys, xs]]) <= TOLERANCE).all(1)
        lbl[ys[ok], xs[ok]] = d[ys[ok], xs[ok]]


def _absorb(lbl: np.ndarray) -> np.ndarray:
    """Мелочь и всё незалитое (полоска у линии, сама линия) — ближайшей области: линия делится по своей оси."""
    area = np.bincount(lbl.ravel())
    lbl = np.where(area[lbl] < MIN_REGION, 0, lbl)
    idx = ndimage.distance_transform_edt(lbl == 0, return_distances=False, return_indices=True)
    lbl = lbl[idx[0], idx[1]]
    _, lbl = np.unique(lbl, return_inverse=True)
    return lbl.reshape(idx.shape[1:]).astype(np.int32) + 1


def _polygons(lbl: np.ndarray) -> np.ndarray:
    """Каждая область — многоугольник(и) с дырами; щели после упрощения — к ближайшей области."""
    h, w = lbl.shape
    out = np.zeros((h, w), np.int32)
    ids, area = np.unique(lbl, return_counts=True)
    for i in ids[np.argsort(-area)]:  # крупные первыми, мелкие поверх
        cs, hier = cv2.findContours((lbl == i).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        m = np.zeros((h, w), np.uint8)
        polys = [cv2.approxPolyDP(c, POLY_EPS, True) for c in cs]
        cv2.fillPoly(m, [p for p, hh in zip(polys, hier[0]) if hh[3] < 0], 1)
        cv2.fillPoly(m, [p for p, hh in zip(polys, hier[0]) if hh[3] >= 0], 0)
        out[m > 0] = i
    if (out == 0).any():
        idx = ndimage.distance_transform_edt(out == 0, return_distances=False, return_indices=True)
        out = out[idx[0], idx[1]]
    return out


def _divider(line: np.ndarray) -> np.ndarray:
    """Разделитель — сама нарисованная линия схемы, пиксель в пиксель: любая перестройка (граница соседних областей
    заданной толщины) отходит от штриха и теряет линии внутри одной области (ветки на фоне неба). Только чистка:
    дырки в штрихе в 1 px закрываются, точки шума отбрасываются."""
    d = cv2.morphologyEx(line, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    return (d > 0) & (st[lbl, cv2.CC_STAT_AREA] >= 30)


def segment(schema_rgb: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Схема (RGB) → (id области на пиксель кадра size=(w, h), 1…N; маска разделителя того же размера).

    Режется в размере кадра: на родном ~1 Мп 2511 линии 1–2 px и рвутся в пунктир, через который текут заливки;
    бикубика их не рвёт."""
    if schema_rgb.shape[1::-1] != tuple(size):
        schema_rgb = cv2.resize(schema_rgb, size, interpolation=cv2.INTER_CUBIC)
    sm = cv2.bilateralFilter(schema_rgb, 7, 40, 7)  # гасит шум, края держит
    line = line_mask(sm)
    okw = np.ascontiguousarray(oklab(sm) * np.float32([L_WEIGHT, 1, 1]) * 100)
    lbl = _absorb(_fill(okw, line))
    return _polygons(lbl), _divider(line)


def _straighten(m: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Прямые края маски — по прямым фото (по мотивам _straighten из InegoSurfaceMask). Кандидаты — отрезки LSD
    вдоль края маски, только с фото: усреднённая прямая самой маски (как у InegoSurfaceMask) на ступенчатом крае
    шла диагональю через все ступени — угол стены у лестницы оставался незалитым. Сначала ближайший к краю; уже
    выпрямленное следующие не перекраивают. В полосе край ставится ровно по прямой, сторона «внутри» — где маски
    больше. Кривые края (кусты, арки) и края без прямой на фото не трогаются."""
    ys, xs = np.nonzero(m)
    if not len(ys):
        return m
    size = max(np.ptp(ys), np.ptp(xs)) + 1
    tol = max(2.5, STRAIGHT_TOL * size)
    far = max(2 * tol, STRAIGHT_FAR * size)
    H, W = m.shape
    edge = m ^ ndimage.binary_erosion(m)
    dist = ndimage.distance_transform_edt(~edge)
    det = cv2.createLineSegmentDetector()
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    segs = []
    found = det.detect(gray)[0]
    if found is not None:
        for x1, y1, x2, y2 in np.asarray(found).reshape(-1, 4):
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < max(12.0, STRAIGHT_MIN_SEG * size):
                continue
            t = np.linspace(0, 1, 16)
            px = np.clip(np.round(x1 + t * (x2 - x1)).astype(int), 0, W - 1)
            py = np.clip(np.round(y1 + t * (y2 - y1)).astype(int), 0, H - 1)
            dd = dist[py, px]
            if (dd <= tol).mean() >= 0.75:
                width = tol
            elif length >= STRAIGHT_LONG * size and np.percentile(dd, 80) <= far:
                width = min(far, float(np.percentile(dd, 90)) + 2)
            else:
                continue  # не вдоль края маски: шов внутри, соседняя поверхность
            key = float(np.median(dd))
            segs.append((key, -length, width, x1, y1, x2, y2))
    if not segs:
        return m
    out = m.copy()
    done = np.zeros_like(m)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for _, neg_len, width, x1, y1, x2, y2 in sorted(segs)[:STRAIGHT_MAX_SEGS]:
        length = -neg_len
        d = np.array([x2 - x1, y2 - y1], np.float32) / length
        along = (xx - x1) * d[0] + (yy - y1) * d[1]
        across = (xx - x1) * -d[1] + (yy - y1) * d[0]
        band = (np.abs(across) <= width) & (along >= 0) & (along <= length)
        if band.sum() < 10:
            continue
        pos = m[band & (across > 0)].mean() if (band & (across > 0)).any() else 0
        neg = m[band & (across < 0)].mean() if (band & (across < 0)).any() else 0
        if abs(pos - neg) < 0.4:
            continue  # прямая не разделяет маску и фон — это не край
        inside = (across > 0) if pos > neg else (across < 0)
        new = band & ~done
        out[new] = inside[new]
        done |= band
    return out


def snap_to_photo(photo_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Маска по схеме → та же маска с краем по фото. Схема 2511 совмещена с кадром не пиксель в пиксель (InegoAlign,
    ~1%): у столба край маски уходил на соседнюю стену, а у основания недоходил — на фото оставалась полоска старого
    камня. В полосе ± SNAP_BAND вокруг края край ставит GrabCut по цветам фото: глубина маски — объект, всё за полосой —
    не объект. Край выпрямляется по прямым фото и
    многоугольником, как в схеме. mask и ответ — bool в размере фото."""
    if not mask.any():
        return mask
    h, w = mask.shape
    k = max(4, round(SNAP_BAND * max(h, w)))
    ys, xs = np.nonzero(mask)
    y1, y2 = max(0, ys.min() - 2 * k), min(h, ys.max() + 2 * k + 1)
    x1, x2 = max(0, xs.min() - 2 * k), min(w, xs.max() + 2 * k + 1)
    m = mask[y1:y2, x1:x2].astype(np.uint8)
    inner = cv2.erode(m, _disk(k)) > 0
    outer = cv2.dilate(m, _disk(k)) > 0
    if not inner.any():  # элемент тоньше полосы — глубины нет, держимся маски
        inner = cv2.erode(m, _disk(1)) > 0
        if not inner.any():
            return mask
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[outer] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[inner] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    img = np.ascontiguousarray(cv2.cvtColor(photo_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
    try:
        cv2.grabCut(img, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    # модель цвета — одна на кроп. Пробовались и не прошли проверку на реальных местах: модели по кускам (лезли на
    # светлую ступеньку и подоконник), запрет убирать «спорное» из схемы (схема сама залезала на штукатурку у окна),
    # сдвиг маски по краям фото (ошибок формы схемы не правит). Остаток правится в редакторе кистью
    out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    # мелкие зубцы и дырки края; куски, не связанные с маской, — мимо
    out = cv2.morphologyEx(cv2.morphologyEx(out, cv2.MORPH_OPEN, _disk(1)), cv2.MORPH_CLOSE, _disk(2))
    out = ((out > 0) & outer) | inner
    n, lbl = cv2.connectedComponents(out.astype(np.uint8), connectivity=8)
    keep = np.unique(lbl[inner])
    out = np.isin(lbl, keep[keep > 0])
    changed = (out ^ (m > 0)).sum() / max(1, m.sum())
    if changed > SNAP_MAX:
        return mask
    # где фото не дало чёткого края (серая ступенька у серого цоколя), край волной — по прямым фото
    # (кромки ступеней, угол стены)
    out = _straighten(out, photo_rgb[y1:y2, x1:x2]) | inner
    poly = np.zeros(m.shape, np.uint8)
    cs, hier = cv2.findContours(out.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if cs:
        polys = [cv2.approxPolyDP(c, POLY_EPS * 1.5, True) for c in cs]
        cv2.fillPoly(poly, [p for p, hh in zip(polys, hier[0]) if hh[3] < 0], 1)
        cv2.fillPoly(poly, [p for p, hh in zip(polys, hier[0]) if hh[3] >= 0], 0)
    res = mask.copy()
    res[y1:y2, x1:x2] = poly > 0
    return res
