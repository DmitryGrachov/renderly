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
