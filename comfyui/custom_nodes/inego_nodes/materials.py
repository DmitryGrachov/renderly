"""Render pass: кадр по материалам (нод InegoMaterialGroups).

Что одним материалом — решает Materialistic (Sharma et al., SIGGRAPH 2023): по точке он даёт карту «где такой же
материал», не глядя на свет, тени и блики. По кадру ставится сетка точек; точки с взаимным сходством выше
MAT_CUT — один материал (средняя связь), пиксель уходит материалу с наибольшим средним сходством.

Края — от SAM 2.1: Materialistic работает в квадрате 512 и края у него мыльные. SAM в полном разрешении режет кадр
на куски (сетка точек, уверенные маски, без дублей); кусок целиком уходит материалу с наибольшим сходством внутри
него. Заодно стена в тени и на солнце — один кусок SAM, и Materialistic её уже не делит.
"""

import importlib.util
import json
import logging
import os
import sys
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage

import folder_paths

from . import _device, _free_vram, _models, _sam, _to_uint8

log = logging.getLogger("inego")

# код модели — репозиторий Materialistic (клонируется в Dockerfile), веса — в volume моделей (entrypoint.sh)
MAT_REPO = os.getenv("MATERIALISTIC_REPO", "/opt/materialistic")
MAT_DIR = os.path.join(folder_paths.models_dir, "materialistic")
MAT_SIZE = 512  # вход модели фиксирован: 512×512
MAT_GRID = 16  # сетка точек-образцов материала (16×16)
MAT_CUT = 0.5  # точки одного материала: среднее взаимное сходство выше
MAT_MIN = 0.3  # пиксель с наибольшим сходством ниже — без материала
MAT_BATCH = 4  # точек за проход головы: у неё 256 каналов в 512² на точку
COLOUR_CUT = 3.0  # по схеме: куски одной группы Materialistic — один материал, если цвет ближе (ΔE Lab со светлотой ×COLOUR_L)
# светлота весит вчетверо меньше оттенка: схема держит остатки тени (короб карниза темнее кровли на ΔL 6 при том же
# оттенке), а другой материал почти всегда другого оттенка (серые ступени и кремовая штукатурка — Δb 4). При весе 0.5
# и пороге 4.5 короб (3.05) лежал у самого порога и от прогона к прогону то сливался с кровлей, то нет; сейчас — 1.85
# против 4.5 у ступеней
COLOUR_L = 0.25
COLOUR_SMALL = 0.0003  # кусок меньше этой доли кадра своей группы по цвету не заводит
COLOUR_SAME = 1.8  # группы с таким близким цветом на схеме (та же мера) — кандидаты на слияние (подтверждает фото или DMS)
# дорезка кусков SAM по схеме: область дальше PIECE_CUT от цвета куска (та же мера) и не меньше PIECE_MIN пикселей
PIECE_CUT = 6.0
PIECE_MIN = 60
# спорные пары для судьи (DeepSeek): цвет на схеме ближе ASK_CUT (та же мера), группы не меньше ASK_MIN кадра
ASK_CUT = 6.0
ASK_MIN = 0.001
ASK_MAX = 16
COLOUR_GROUP_MIN = 0.002  # мелкие куски вдали от цветов группы: своя группа, если вместе не меньше этой доли кадра
SAM_ROWS = 32  # сетка точек SAM по высоте кадра (по ширине — в пропорции)
SAM_IOU = 0.80  # маска SAM в дело: предсказанный IoU не ниже
SAM_STABLE = 0.85  # и устойчивость к порогу (площадь при logit>1 / при logit>-1)
SAM_DUP = 0.7  # маски с IoU выше — дубли, берётся уверенная
MAX_SIDE = 2048  # карта строится в этом размере по длинной стороне; фронтенд растягивает её на кадр
# альбедо до схемы (InegoIntrinsicAlbedo): Intrinsic v2 (Careaga & Aksoy, «Colorful Diffuse Intrinsic Image
# Decomposition in the Wild», TOG 2024; только академическое использование). Веса и бэкбоны — через torch.hub в volume
INTRINSIC_DIR = os.path.join(folder_paths.models_dir, "intrinsic")


def _materialistic():
    if "materialistic" not in _models:
        spec = importlib.util.spec_from_file_location("materialistic_transformer", f"{MAT_REPO}/src/models/transformer.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # бэкбон DINO ViT-B/8 модель тянет через torch.hub — кэш в volume моделей, а не в контейнере
        torch.hub.set_dir(f"{MAT_DIR}/hub")
        # код DINO импортирует свои модули utils и vision_transformer, а utils в sys.modules уже у ComfyUI
        names = ("utils", "vision_transformer")
        saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
        try:
            net = mod.Transformer()
        finally:
            for n in names:
                sys.modules.pop(n, None)
            sys.modules.update(saved)
        sd = torch.load(f"{MAT_DIR}/model.ckpt", map_location="cpu", weights_only=False)["state_dict"]
        net.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("net.")})
        _models["materialistic"] = net.eval()
    return _models["materialistic"]


def _point_maps(net, rgb: np.ndarray, pts: np.ndarray) -> torch.Tensor:
    """Карты сходства Materialistic (N, 512, 512) на видеокарте: по карте на точку-образец."""
    S = MAT_SIZE
    img = np.asarray(Image.fromarray(rgb).resize((S, S), Image.BICUBIC)).astype(np.float32) / 255
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = ((torch.from_numpy(img).permute(2, 0, 1) - mean) / std).to(_device())
    # Transformer.forward на каждую точку заново гонит DINO по кадру: пачка из 8 копий ViT-B/8 на 512² забивала
    # 16 ГБ, и драйвер уходил в системную память (~4 мин). Здесь DINO и признаки кадра — один раз, по точкам —
    # только голова сравнения (cross-attention + fusion), как в forward
    maps = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        net.dino_model(x[None])
        ctx = []
        for i in range(1, 5):
            c = getattr(net, f"spatial_transform_{i}")(net.activations[str(i)])
            ctx.append(c.permute(0, 2, 3, 1).reshape(1, -1, 256))
        for b in range(0, len(pts), MAT_BATCH):
            p = torch.from_numpy(pts[b:b + MAT_BATCH])
            B = len(p)
            ca = []
            for i, side in zip(range(1, 5), (256, 128, 64, 64)):
                o = getattr(net, f"cross_attention_{i}")(ctx[i - 1].expand(B, -1, -1), p)[0]
                ca.append(o.permute(0, 2, 1).reshape(B, -1, side, side))
            path = net.fusion_4(ca[3])
            path = net.fusion_3(ca[2], path)
            path = net.fusion_2(ca[1], path)
            path = net.fusion_1(ca[0], path)
            maps.append(torch.sigmoid(net.out_conv(path.permute(0, 2, 3, 1)).reshape(B, S, S)).float())
    net.activations.clear()
    return torch.cat(maps)


def _material_maps(*images: np.ndarray) -> np.ndarray:
    """Средние карты сходства материалов (K, 512, 512). Несколько кадров одной сцены (схема и фото) — сходство
    по наименьшему: один материал, только если согласны оба. Схема даёт цвет без света, фото — фактуру (листва куста
    против травы газона: на схеме они одного зелёного); при среднем схема фото перевешивала."""
    net = _materialistic().to(_device())
    S = MAT_SIZE
    step = S / MAT_GRID
    pts = np.array([[int((i + 0.5) * step), int((j + 0.5) * step)] for i in range(MAT_GRID) for j in range(MAT_GRID)])
    try:
        maps = _point_maps(net, images[0], pts)  # (N, S, S) на видеокарте
        for im in images[1:]:
            maps = torch.minimum(maps, _point_maps(net, im, pts))
        t = torch.from_numpy(pts).to(maps.device)
        A = maps[:, t[:, 0], t[:, 1]].cpu().numpy()  # A[i, j] — сходство точки j с материалом точки i
        sim = (A + A.T) / 2
        np.fill_diagonal(sim, 1)
        n = len(pts)
        lab = fcluster(linkage(1 - sim[np.triu_indices(n, 1)], "average"), 1 - MAT_CUT, "distance") - 1
        lab = torch.from_numpy(lab).to(maps.device)
        return torch.stack([maps[lab == k].mean(0) for k in range(int(lab.max()) + 1)]).cpu().numpy()
    finally:
        net.to("cpu")
        net.activations.clear()


def _sam_pieces(rgb: np.ndarray) -> np.ndarray:
    """Разбиение кадра на куски SAM 2.1 [H, W] (1..N): мелкие куски поверх крупных, непокрытое — по связности."""
    processor, model = _sam()
    h, w = rgb.shape[:2]
    model.to(_device())
    # отбор и дубли — по низкому разрешению SAM (256²) на видеокарте; в полный размер — только оставленные маски
    logits, scores = [], []
    try:
        inputs = processor(images=rgb, return_tensors="pt").to(_device())
        with torch.no_grad():
            emb = model.get_image_embeddings(inputs["pixel_values"])
        cols = max(1, round(SAM_ROWS * w / h))
        pts = [[(c + 0.5) * w / cols, (r + 0.5) * h / SAM_ROWS] for r in range(SAM_ROWS) for c in range(cols)]
        for b in range(0, len(pts), 64):
            chunk = pts[b:b + 64]
            pi = processor(images=rgb, input_points=[[[p] for p in chunk]], input_labels=[[[1] for _ in chunk]],
                           return_tensors="pt").to(_device())
            with torch.no_grad():
                out = model(image_embeddings=emb, input_points=pi["input_points"], input_labels=pi["input_labels"],
                            multimask_output=True)
            logits.append(out.pred_masks[0].flatten(0, 1).float())  # (n·3, 256, 256)
            scores.append(out.iou_scores[0].flatten().float())
    finally:
        model.to("cpu")
    L = torch.cat(logits)
    s = torch.cat(scores)
    lo = (L > -1).flatten(1).sum(1)
    stable = (L > 1).flatten(1).sum(1) / lo.clamp(min=1)
    m = (L > 0).flatten(1)
    area = m.sum(1)
    ok = (s >= SAM_IOU) & (stable >= SAM_STABLE) & (area >= max(1, 30 * m.shape[1] / (h * w)))
    idx = torch.nonzero(ok).flatten()
    idx = idx[torch.argsort(s[idx], descending=True)]
    mf = m[idx].half()
    inter = mf @ mf.T
    union = area[idx][:, None] + area[idx][None, :] - inter
    dup = ((inter / union.clamp(min=1)) > SAM_DUP).cpu().numpy()
    taken = np.zeros(len(idx), bool)  # уже оставленные маски
    keep = []
    for i in range(len(idx)):
        if not (dup[i] & taken).any():
            keep.append(i)
            taken[i] = True
    kept = idx[keep]
    kept = kept[torch.argsort(area[kept], descending=True)]  # крупные снизу, мелкие поверх
    piece = np.zeros((h, w), np.int32)
    for b in range(0, len(kept), 32):
        full = processor.post_process_masks(L[kept[b:b + 32]][None, :, None], inputs["original_sizes"])[0]
        for n, mm in enumerate(full[:, 0].cpu().numpy(), b + 1):
            piece[mm] = n
    rest, _ = ndimage.label(piece == 0)
    piece[rest > 0] = rest[rest > 0] + len(kept)
    return piece


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 (..., 3) → CIE Lab (D65)."""
    c = rgb.astype(np.float32) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _split_by_colour(lab: np.ndarray, piece: np.ndarray, schema: np.ndarray, refl: np.ndarray, confirm) -> tuple[np.ndarray, int]:
    """Группы Materialistic по схеме делятся по цвету: на плоской заливке без фактуры он склеивает газон с изгородью
    и водостоки со стеной, а цвет схемы и есть материал. Куски одной группы с близким цветом (медиана; ΔE Lab, светлота
    с весом COLOUR_L, ниже COLOUR_CUT, средняя связь) — один материал; мелкие куски (кромки с переходным цветом) своих групп не заводят —
    уходят к ближайшему по цвету материалу своей группы."""
    P = len(lab)
    pid = piece.ravel()
    cnt = np.bincount(pid, minlength=P).astype(np.float64)
    L = _lab(schema)
    # цвет куска — медиана, не среднее: на плоской заливке это сам цвет заливки, а в среднее большого куска стены
    # затекали кромки и остатки рам — серела штукатурка и склеивалась с серыми ступенями (ΔE 6 при пороге 9)
    col = np.stack([ndimage.median(L[..., i], piece, np.arange(P)) for i in range(3)], 1)
    col = np.nan_to_num(col)
    col[:, 0] *= COLOUR_L
    small = cnt < COLOUR_SMALL * pid.size
    out = np.zeros(P, np.int32)

    def cluster(ids: np.ndarray) -> list[np.ndarray]:
        sub = fcluster(linkage(col[ids], "average"), COLOUR_CUT, "distance") if len(ids) > 1 else np.ones(1, int)
        return [ids[sub == s] for s in np.unique(sub)]

    n = 0
    for k in np.unique(lab[lab > 0]):
        idx = np.flatnonzero(lab == k)
        big = idx[~small[idx]]
        if len(big) == 0:
            big = idx
        members = cluster(big)
        rest = np.setdiff1d(idx, big)
        if len(rest) and members:
            # мелкие куски вдали от всех цветов группы — не кромки, а свой материал, порезанный SAM на мелочь
            # (кусты с листвой на схеме): вместе по площади заметные — своя группа, иначе к ближайшему цвету
            cent = np.stack([np.average(col[m], axis=0, weights=cnt[m]) for m in members])
            far = rest[np.linalg.norm(col[rest, None] - cent[None], axis=2).min(1) > COLOUR_CUT]
            if len(far):
                members += [m for m in cluster(far) if cnt[m].sum() >= COLOUR_GROUP_MIN * pid.size]
        cent = np.stack([np.average(col[m], axis=0, weights=cnt[m]) for m in members])
        out[idx] = n + 1 + np.linalg.norm(col[idx, None] - cent[None], axis=2).argmin(1)
        for s, m in enumerate(members):
            out[m] = n + 1 + s
        n += len(members)
    # и наоборот — сборка: группы с одним цветом заливки на схеме (полная связь, ниже COLOUR_SAME) сливаются, если
    # это подтверждает фото (confirm: сходство Materialistic по фото не ниже MAT_CUT) или обе — стекло (DMS46: по фото
    # в каждом окне отражается своё). Materialistic раскладывал одинаковые серые панели дома по двум группам (одну —
    # со стеклом, другую — с кирпичом), а без подтверждения по одному цвету схемы склеивались куст с газоном (один
    # зелёный) и водосток с мощением (один серый)
    w = np.bincount(out, weights=cnt, minlength=n + 1)
    glass = np.bincount(out, weights=np.bincount(pid, weights=refl.ravel(), minlength=P), minlength=n + 1) / np.maximum(w, 1)
    alive = np.flatnonzero(w[1:] > 0) + 1
    if len(alive) > 1:
        cent = np.stack([np.bincount(out, weights=cnt * col[:, i], minlength=n + 1) for i in range(3)], 1)
        cent = cent[alive] / w[alive][:, None]
        same = fcluster(linkage(cent, "complete"), COLOUR_SAME, "distance")
        parent = np.arange(n + 1)

        def root(g):
            while parent[g] != g:
                g = parent[g]
            return g

        for c in np.unique(same):
            gs = alive[same == c]
            if len(gs) < 2:
                continue
            sim = confirm(out, gs)
            for i in range(len(gs)):
                for j in range(i + 1, len(gs)):
                    both_glass = glass[gs[i]] > DMS_GLASS and glass[gs[j]] > DMS_GLASS
                    if both_glass or sim[i, j] >= MAT_CUT:
                        parent[root(gs[j])] = root(gs[i])
        out = np.array([root(g) for g in range(n + 1)])[out]
    return out, n


def _split_pieces_by_colour(piece: np.ndarray, schema: np.ndarray) -> np.ndarray:
    """Дорезка кусков SAM по схеме: связная область куска, цветом далёкая от его медианы (та же мера, дальше
    PIECE_CUT) и не меньше PIECE_MIN пикселей, — свой кусок. Сетка точек SAM (шаг ~36 px) пропускает мелкие стёкла
    многоэтажки: они оставались внутри куска кирпичной стены и уходили в кирпич, хотя на схеме — голубые на красном."""
    P = int(piece.max()) + 1
    L = _lab(schema)
    L[..., 0] *= COLOUR_L
    med = np.nan_to_num(np.stack([ndimage.median(L[..., i], piece, np.arange(P)) for i in range(3)], 1))
    far = np.linalg.norm(L - med[piece], axis=2) > PIECE_CUT
    comp, n = ndimage.label(far)
    if not n:
        return piece
    size = np.bincount(comp.ravel(), minlength=n + 1)
    keep = size >= PIECE_MIN
    keep[0] = False
    out = piece.copy()
    sel = keep[comp]
    out[sel] = P + comp[sel]  # компонента не пересекает куски: на границе кусков цвет обычно близок к одному из них
    # плотная перенумерация
    _, out = np.unique(out, return_inverse=True)
    return out.reshape(piece.shape).astype(np.int32)


def _photo_similarity(net, rgb: np.ndarray, owner: np.ndarray, gids: np.ndarray) -> np.ndarray:
    """Сходство групп по фото (len×len, симметричное): точка-образец в глубине группы i — среднее сходство
    Materialistic по пикселям группы j. Группа, не видная в 512², ни с кем не сходна."""
    S = MAT_SIZE
    own = np.asarray(Image.fromarray(owner.astype(np.int32)).resize((S, S), Image.NEAREST))
    masks = [own == g for g in gids]
    ok = [m.any() for m in masks]
    sim = np.zeros((len(gids), len(gids)))
    if sum(ok) < 2:
        return sim
    pts = np.array([np.unravel_index(ndimage.distance_transform_edt(m).argmax(), m.shape) for m, o in zip(masks, ok) if o])
    maps = _point_maps(net, rgb, pts)
    idx = [i for i, o in enumerate(ok) if o]
    for a, i in enumerate(idx):
        for j in idx:
            sim[i, j] = maps[a][torch.from_numpy(masks[j]).to(maps.device)].mean().item()
    return (sim + sim.T) / 2


def build_material_map(rgb: np.ndarray, schema: np.ndarray | None = None) -> tuple[np.ndarray, list[dict], list[list]]:
    """Карта материалов [H, W] (1..K, 0 — без материала), список групп по убыванию площади и спорные пары.

    schema — тот же кадр плоской заливкой без света (Qwen-Image-2.1, совмещён с исходником): материалы и куски
    ищутся по ней — без теней и бликов водостоки, фонари и рамы — сплошные пятна своего цвета; карта — на кадр."""
    look = rgb if schema is None else schema
    t0 = time.time()
    h, w = rgb.shape[:2]
    # материалы — по схеме и фото вместе: схема без света, фото с фактурой
    mean = _material_maps(rgb) if schema is None else _material_maps(schema, rgb)
    K = len(mean)
    t1 = time.time()
    piece = _sam_pieces(look)
    if schema is not None:
        piece = _split_pieces_by_colour(piece, schema)
    t2 = time.time()
    P = int(piece.max()) + 1
    # среднее сходство каждого материала по каждому куску — на видеокарте
    dev = _device()
    M = F.interpolate(torch.from_numpy(mean).to(dev)[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    pid = torch.from_numpy(piece).to(dev).flatten()
    score = torch.zeros(K, P, device=dev).index_add_(1, pid, M.flatten(1))
    score /= torch.bincount(pid, minlength=P).clamp(min=1)
    best, lab = score.max(0)
    lab = (lab + 1).masked_fill(best < MAT_MIN, 0).cpu().numpy()
    del M, score
    if schema is not None:
        refl = reflective_mask(rgb)
        _models.pop("dms", None)
        net = _materialistic().to(dev)
        try:
            lab, K = _split_by_colour(lab, piece, schema, refl, lambda out, gs: _photo_similarity(net, rgb, out[piece], gs))
        finally:
            net.to("cpu")
    owner = lab[piece]
    # номера — по убыванию площади, пустые группы выпадают
    area = np.bincount(owner.ravel(), minlength=K + 1)
    order = [k for k in np.argsort(-area) if k and area[k]]
    remap = np.zeros(K + 1, np.int32)
    groups = []
    for n, k in enumerate(order[:255], 1):
        remap[k] = n
        c = np.median(rgb[owner == k], axis=0).astype(int)
        groups.append({"id": n, "label": f"Материал {n}", "area": round(float(area[k]) / owner.size, 5),
                       "color": "#%02x%02x%02x" % tuple(c)})
    idmap = remap[owner].astype(np.uint8)
    pairs = _disputed_pairs(idmap, schema) if schema is not None else []
    log.info("INEGO materials: %d групп, %d кусков SAM, %d спорных пар за %.1f с (Materialistic %.1f, SAM %.1f)",
             len(groups), P - 1, len(pairs), time.time() - t0, t1 - t0, t2 - t1)
    return idmap, groups, pairs


def _disputed_pairs(idmap: np.ndarray, schema: np.ndarray) -> list[list]:
    """Спорные пары для судьи (DeepSeek в бэкенде): группы остались раздельными, а цвет на схеме близкий (та же мера,
    ниже ASK_CUT) — одинаковые панели, разложенные Materialistic по разным группам, стёкла в тени. Мелкие группы
    (меньше ASK_MIN кадра) не спрашиваются; пары — от самых близких по цвету, не больше ASK_MAX."""
    n = int(idmap.max())
    if n < 2:
        return []
    ids = idmap.ravel()
    L = _lab(schema).reshape(-1, 3)
    L[:, 0] *= COLOUR_L
    w = np.bincount(ids, minlength=n + 1).astype(np.float64)
    cent = np.stack([np.bincount(ids, weights=L[:, i], minlength=n + 1) for i in range(3)], 1) / np.maximum(w, 1)[:, None]
    big = [g for g in range(1, n + 1) if w[g] >= ASK_MIN * ids.size]
    out = []
    for i, a in enumerate(big):
        for b in big[i + 1:]:
            d = float(np.linalg.norm(cent[a] - cent[b]))
            if d < ASK_CUT:
                out.append([a, b, round(d, 2)])
    return sorted(out, key=lambda p: p[2])[:ASK_MAX]


DMS_DIR = os.path.join(folder_paths.models_dir, "zones", "dms46")
# DMS46 (Upchurch & Niu, Apple, ECCV 2022): класс материала на пиксель. Нужен для одного: найти стекло — в нём
# отражается своё в каждом окне, и Materialistic по фото разводит одинаковые стёкла по разным группам
DMS_REFLECTIVE = {20, 27, 48}  # Glass, Mirror, Water — номера в полной таксономии DMS (taxonomy.json авторов)
# сторона кадра для DMS: у авторов 512 — стекло нашлось только в большом окне; на 1024 — во всех окнах и двери,
# на 1536 — ложные пятна на мощении. Группа — стекло, если стекла по DMS в ней больше DMS_GLASS (пятна DMS мельче стёкол)
DMS_SIDE = 1024
DMS_GLASS = 0.3


def _dms():
    if "dms" not in _models:
        _models["dms"] = torch.jit.load(f"{DMS_DIR}/DMS46_v1.pt", map_location="cpu").eval()
    return _models["dms"]


def reflective_mask(rgb: np.ndarray) -> np.ndarray:
    """[H, W] bool: стекло, зеркало или вода по DMS46 (нормировка как в inference.py авторов, сторона — DMS_SIDE)."""
    # выход модели — номер в списке DMS46 (46 классов), это номера в полной таксономии
    dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 34,
             35, 36, 37, 38, 39, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 56]
    refl = torch.tensor([i in DMS_REFLECTIVE for i in dms46])
    h, w = rgb.shape[:2]
    sc = DMS_SIDE / max(h, w)
    img = np.asarray(Image.fromarray(rgb).resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS))
    x = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
    x = (x - torch.tensor([123.675, 116.28, 103.53])[:, None, None]) / torch.tensor([58.395, 57.12, 57.375])[:, None, None]
    net = _dms().to(_device())
    try:
        with torch.no_grad():
            pred = net(x[None].to(_device()))[0][0, 0].cpu()
    finally:
        net.to("cpu")
    m = refl[pred.long()].numpy().astype(np.uint8) * 255
    return np.asarray(Image.fromarray(m).resize((w, h), Image.NEAREST)) > 0


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


class InegoMaterialGroups:
    """Кадр по материалам (см. docstring модуля). Выход — карта в output/inego и список групп в ui.groups."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}, "optional": {"schema": ("IMAGE",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "inego"

    def run(self, image, schema=None):
        _free_vram()
        rgb = _to_uint8(image[0])[..., :3]
        if max(rgb.shape[:2]) > MAX_SIDE:
            im = Image.fromarray(rgb)
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            rgb = np.asarray(im)
        sch = None
        try:
            if schema is not None:
                sch = np.asarray(Image.fromarray(_to_uint8(schema[0])[..., :3]).resize((rgb.shape[1], rgb.shape[0]), Image.LANCZOS))
            idmap, groups, pairs = build_material_map(rgb, sch)
        finally:
            # Materialistic и SAM (~1.5 ГБ) не держим в RAM рядом с Qwen и klein
            _models.pop("materialistic", None)
            _models.pop("sam", None)
            torch.cuda.empty_cache()
        folder = os.path.join(folder_paths.get_output_directory(), "inego")
        os.makedirs(folder, exist_ok=True)
        fn = f"materials_{uuid.uuid4().hex}.png"
        Image.fromarray(idmap, "L").save(os.path.join(folder, fn))
        files = [{"filename": fn, "subfolder": "inego", "type": "output"}]
        if sch is not None:
            # сама плоская схема — во вкладке её можно показать рядом с кадром
            fn = fn.replace(".png", "_schema.jpg")
            Image.fromarray(sch).save(os.path.join(folder, fn), quality=92)
            files.append({"filename": fn, "subfolder": "inego", "type": "output"})
        return {"ui": {"images": files,
                       "groups": [json.dumps(groups, ensure_ascii=False)], "pairs": [json.dumps(pairs)]}}
