"""INEGO: «сохранить небо / фон / окружение» для img2img экстерьера.

InegoKeepZones — разметка исходника на зоны:
- небо — SegFormer (ADE20k), класс sky;
- главное здание — SAM 2.1 по рамке (её даёт DeepSeek, см. backend/app/prompting.py); нет рамки —
  крупнейшая (с поправкой на центр кадра) связная область архитектурных классов SegFormer;
- фон — прочая застройка, горы и холмы, деревья позади здания;
- окружение — земля, дороги, газоны, люди, машины, малые формы, деревья и кусты перед зданием
  («позади / перед» для деревьев, стен и оград решает глубина Depth Anything V2).

InegoZoneRestyle — сохраняемые зоны берутся с исходника (те же объекты, формы, облака), а их цвет
и свет переносятся с полного рендера той же зоны (статистики LAB + мягкий низкочастотный свет).
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

import folder_paths

log = logging.getLogger("inego")

SEGFORMER_ID = "nvidia/segformer-b4-finetuned-ade-512-512"
SAM_ID = "facebook/sam2.1-hiera-large"
ARCH = {
    "wall", "building", "house", "skyscraper", "windowpane", "door", "screen door", "column", "railing",
    "bannister", "awning", "canopy", "stairs", "step", "stairway", "tower", "hovel", "ceiling", "booth",
}
SKY = {"sky"}
# в фон — именно постройки и рельеф; ступени, ограждения, навесы, киоски у земли — окружение
FAR_ALWAYS = {
    "building", "house", "skyscraper", "windowpane", "door", "screen door", "column", "tower", "hovel",
    "ceiling", "mountain", "hill",
}
# фон или окружение — по глубине: деревья, а ещё стены и ограды (так SegFormer часто видит живую изгородь)
VEGETATION = {"tree", "palm", "plant", "wall", "fence"}
DEPTH_ID = "depth-anything/Depth-Anything-V2-Base-hf"
# дерево — фон, только если оно явно дальше самой дальней части главного здания (квантиль её «близости»)
BACKGROUND_QUANTILE = 0.05
BOX_PAD = 0.03  # рамка от LLM приблизительная — расширяем, SAM сам найдёт контур

_models: dict = {}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _segformer():
    if "segformer" not in _models:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        cache = f"{folder_paths.models_dir}/segformer"
        _models["segformer"] = (
            SegformerImageProcessor.from_pretrained(SEGFORMER_ID, cache_dir=cache),
            SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_ID, cache_dir=cache).eval(),
        )
    return _models["segformer"]


def _sam():
    if "sam" not in _models:
        from transformers import Sam2Model, Sam2Processor

        cache = f"{folder_paths.models_dir}/sam2"
        _models["sam"] = (
            Sam2Processor.from_pretrained(SAM_ID, cache_dir=cache),
            Sam2Model.from_pretrained(SAM_ID, cache_dir=cache).eval(),
        )
    return _models["sam"]


def _depth_model():
    if "depth" not in _models:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        cache = f"{folder_paths.models_dir}/depth"
        _models["depth"] = (
            AutoImageProcessor.from_pretrained(DEPTH_ID, cache_dir=cache),
            AutoModelForDepthEstimation.from_pretrained(DEPTH_ID, cache_dir=cache).eval(),
        )
    return _models["depth"]


def _nearness(rgb: np.ndarray) -> np.ndarray:
    """Относительная близость [H, W] (больше — ближе к камере)."""
    processor, model = _depth_model()
    h, w = rgb.shape[:2]
    model.to(_device())
    try:
        inputs = processor(images=rgb, return_tensors="pt").to(_device())
        with torch.no_grad():
            depth = model(**inputs).predicted_depth
        return F.interpolate(depth[:, None], size=(h, w), mode="bicubic", align_corners=False)[0, 0].float().cpu().numpy()
    finally:
        model.to("cpu")


def _split_background(labels, names, rest, building, rgb):
    """Делит «всё кроме главного здания и неба» на фон и окружение."""
    ids = lambda group: [i for i, n in names.items() if n in group]  # noqa: E731
    background = rest & np.isin(labels, ids(FAR_ALWAYS))
    trees = rest & np.isin(labels, ids(VEGETATION))
    if trees.any():
        near = _nearness(rgb)
        ref = near[building] if building.any() else near[rest]
        threshold = np.quantile(ref, BACKGROUND_QUANTILE)
        comp, n = ndimage.label(trees)
        medians = np.asarray(ndimage.median(near, comp, index=np.arange(1, n + 1)))
        background |= np.isin(comp, np.flatnonzero(medians < threshold) + 1)
    return background, rest & ~background


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.clamp(0, 1).numpy() * 255).astype(np.uint8)


def _segment(rgb: np.ndarray, sky_prob: bool = False):
    """Карта классов ADE20k [H, W]; с sky_prob — ещё и вероятность неба [H, W] (для мягкой маски неба)."""
    processor, model = _segformer()
    h, w = rgb.shape[:2]
    model.to(_device())
    names = {int(k): v.split(";")[0].strip().lower() for k, v in model.config.id2label.items()}
    try:
        inputs = processor(images=rgb, return_tensors="pt").to(_device())
        with torch.no_grad():
            logits = model(**inputs).logits
        labels = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
        prob = None
        if sky_prob:
            sky_ids = [i for i, n in names.items() if n in SKY]
            p = logits.softmax(1)[:, sky_ids].sum(1, keepdim=True)
            prob = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    finally:
        model.to("cpu")
    return (labels, names, prob) if sky_prob else (labels, names)


def _clean_sky(prob: np.ndarray) -> np.ndarray:
    """Цельная маска неба из вероятности: без пятен и клочьев.

    По argmax полупрозрачные объёмы и дымка рвутся на клочья «небо / не небо» — плашка неба вклеивалась
    пятнами. Сглаженная вероятность даёт цельные области; мелкие островки и дыры убираются.
    """
    h, w = prob.shape
    area = h * w
    p = ndimage.gaussian_filter(prob.astype(np.float32), sigma=min(h, w) * 0.012)
    m = p > 0.5
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.where(sizes >= 0.003 * area)[0])
    holes, n = ndimage.label(~m)
    if n:
        sizes = ndimage.sum(~m, holes, range(1, n + 1))
        m |= np.isin(holes, 1 + np.where(sizes < 0.003 * area)[0])
    return m


def _sam_building(rgb: np.ndarray, box: list[float]) -> np.ndarray | None:
    """Маска главного здания по рамке (нормированной 0..1)."""
    processor, model = _sam()
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    px = [
        max(0.0, x1 - BOX_PAD) * w, max(0.0, y1 - BOX_PAD) * h,
        min(1.0, x2 + BOX_PAD) * w, min(1.0, y2 + BOX_PAD) * h,
    ]
    model.to(_device())
    try:
        inputs = processor(images=rgb, input_boxes=[[px]], return_tensors="pt").to(_device())
        with torch.no_grad():
            out = model(**inputs, multimask_output=True)
        masks = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0].numpy() > 0
        scores = out.iou_scores[0, 0].float().cpu().numpy()
    finally:
        model.to("cpu")
    # из уверенных вариантов берём самый полный: SAM любит отдавать отдельно фасад или кровлю
    good = [i for i in range(len(scores)) if scores[i] >= scores.max() - 0.08]
    best = max(good, key=lambda i: masks[i].sum())
    log.info("INEGO SAM: scores=%s, выбран %d (%.1f%% кадра)", np.round(scores, 3), best, 100 * masks[best].mean())
    m = ndimage.binary_fill_holes(masks[best])
    return m if m.any() else None


def _segformer_building(arch: np.ndarray) -> np.ndarray:
    h, w = arch.shape
    k = max(3, round(min(h, w) * 0.01))
    closed = ndimage.binary_closing(arch, structure=np.ones((k, k)), iterations=2)
    comp, n = ndimage.label(closed)
    if n == 0:
        return closed
    yy, xx = np.mgrid[0:h, 0:w]
    best, best_score = 0, -1.0
    for i in range(1, n + 1):
        m = comp == i
        cy, cx = yy[m].mean() / h, xx[m].mean() / w
        # ближе к центру кадра — вероятнее объект визуализации, а не соседняя застройка
        score = m.sum() * (1.0 - 0.6 * min(1.0, ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5 / 0.5))
        if score > best_score:
            best, best_score = i, score
    return ndimage.binary_fill_holes(comp == best)


def _parse_box(box: str) -> list[float] | None:
    try:
        v = [float(x) for x in box.split(",")]
    except ValueError:
        return None
    if len(v) != 4 or not (0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1):
        return None
    return v


def _feather(m: np.ndarray, sigma: float) -> np.ndarray:
    m = m.astype(np.float32)
    return ndimage.gaussian_filter(m, sigma=sigma) if sigma else m


def _refine(mask: np.ndarray, rgb: np.ndarray, radius: float = 0.006, eps: float = 1e-3) -> np.ndarray:
    """Край маски по настоящему контуру картинки (guided filter).

    SegFormer размечает в 512 px и растягивает — край грубый: у здания остаётся полоска старого неба
    («обводка»). Маску чуть расширяем, а guided filter прижимает её к реальной границе по цвету и яркости.
    """
    import kornia

    h, w = mask.shape
    r = max(2, round(min(h, w) * radius))
    grown = ndimage.binary_dilation(mask, iterations=max(1, r // 2)) if mask.dtype == bool else mask
    guide = torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 255.0
    m = torch.from_numpy(grown.astype(np.float32))[None, None]
    out = kornia.filters.guided_blur(guide, m, 2 * r + 1, eps)[0, 0].clamp(0, 1).numpy()
    # подтянуть переход: guided filter оставляет полупрозрачную кромку
    out = np.clip((out - 0.5) * 1.6 + 0.5, 0, 1)
    return out.astype(np.float32)


class InegoKeepZones:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "keep_sky": ("BOOLEAN", {"default": False}),
                "keep_background": ("BOOLEAN", {"default": False}),
                "keep_surroundings": ("BOOLEAN", {"default": False}),
                "grow": ("FLOAT", {"default": 0.006, "min": 0.0, "max": 0.1, "step": 0.001}),
                "feather": ("FLOAT", {"default": 0.003, "min": 0.0, "max": 0.05, "step": 0.001}),
            },
            "optional": {
                # рамка главного здания "x1,y1,x2,y2" в долях кадра; пусто — ищем по SegFormer
                "building_box": ("STRING", {"default": ""}),
                # край маски по контуру картинки (guided filter) вместо размытия — для замены неба
                "refine": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("sky", "background", "surroundings", "keep")
    FUNCTION = "run"
    CATEGORY = "inego"

    def run(self, image, keep_sky, keep_background, keep_surroundings, grow, feather, building_box="", refine=False):
        h, w = image.shape[1:3]
        empty = torch.zeros((1, h, w))
        if not (keep_sky or keep_background or keep_surroundings):
            return (empty, empty, empty, empty)

        rgb = _to_uint8(image[0])
        split = keep_background or keep_surroundings
        if refine and keep_sky and not split:
            # только небо: цельная маска из сглаженной вероятности, без клочьев на дымке и полупрозрачных объёмах
            labels, names, prob = _segment(rgb, sky_prob=True)
            sky = _clean_sky(prob)
        else:
            labels, names = _segment(rgb)
            sky = np.isin(labels, [i for i, n in names.items() if n in SKY])
        ids = lambda group: [i for i, n in names.items() if n in group]  # noqa: E731

        building = None
        box = _parse_box(building_box)
        if building is None and split and box:
            building = _sam_building(rgb, box)
        if building is None:
            building = _segformer_building(np.isin(labels, ids(ARCH)))
        if split and not building.any():
            log.warning("INEGO: здание не найдено — сохраняемых зон нет")
            return (empty, empty, empty, empty)
        # здание не бывает небом: SegFormer иногда помечает небом стекло с отражением неба
        sky &= ~building

        zero = np.zeros((h, w), bool)
        background = surroundings = zero
        if split:
            background, surroundings = _split_background(labels, names, ~building & ~sky, building, rgb)
            log.info("INEGO: фон %.1f%%, окружение %.1f%% кадра", 100 * background.mean(), 100 * surroundings.mean())

        # здание с небольшим запасом по контуру остаётся за рендером — шов уходит в сохраняемые зоны
        g = round(min(h, w) * grow)
        band = ndimage.binary_dilation(building, iterations=g) if g else building
        zones = [
            sky if keep_sky else zero,
            background if keep_background else zero,
            surroundings if keep_surroundings else zero,
        ]
        f = min(h, w) * feather
        # медианный фильтр убирает точечную рябь сегментации (стекло, отражения) — иначе в зонах пятна
        k = max(3, round(min(h, w) * 0.01) | 1)
        zones = [ndimage.median_filter(z.astype(np.uint8), size=k).astype(bool) if z.any() else z for z in zones]
        if refine:
            out = [torch.from_numpy(_refine(z & ~band, rgb) if z.any() else z.astype(np.float32))[None] for z in zones]
        else:
            out = [torch.from_numpy(_feather(z & ~band, f))[None] for z in zones]
        keep = torch.clamp(out[0] + out[1] + out[2], 0, 1)
        return (*out, keep)


class InegoZoneRestyle:
    """Сохраняемые зоны — с исходника, но в цвете и свете полного рендера (по каждой зоне отдельно)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "render": ("IMAGE",),
                "sky": ("MASK",),
                "background": ("MASK",),
                "surroundings": ("MASK",),
                # доля локального (низкочастотного) света рендера: градиенты неба, тени, засветки
                "local_light": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "light_radius": ("FLOAT", {"default": 0.04, "min": 0.005, "max": 0.2, "step": 0.005}),
            },
            "optional": {
                # доля мелкой фактуры рендера (травинки, листва, асфальт) в фоне и окружении:
                # плоские заливки исходника получают настоящую текстуру, крупный расклад остаётся с исходника
                "detail": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.5, "step": 0.05}),
                "detail_radius": ("FLOAT", {"default": 0.006, "min": 0.001, "max": 0.05, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "inego"

    def run(self, source, render, sky, background, surroundings, local_light, light_radius, detail=0.0, detail_radius=0.006):
        import kornia

        src = source[0].permute(2, 0, 1)[None].float()
        ren = render[0].permute(2, 0, 1)[None].float()
        if ren.shape[-2:] != src.shape[-2:]:
            ren = F.interpolate(ren, size=src.shape[-2:], mode="bilinear", align_corners=False)
        src_lab, ren_lab = kornia.color.rgb_to_lab(src.clamp(0, 1)), kornia.color.rgb_to_lab(ren.clamp(0, 1))
        h, w = src.shape[-2:]
        sigma = max(1.0, min(h, w) * light_radius)
        k = int(sigma * 3) * 2 + 1

        # высокочастотная фактура рендера (рендер минус его размытие)
        high = None
        if detail:
            ds = max(1.0, min(h, w) * detail_radius)
            dk = int(ds * 3) * 2 + 1
            high = ren_lab - kornia.filters.gaussian_blur2d(ren_lab, (dk, dk), (ds, ds))

        out = ren_lab.clone()
        weight_total = torch.zeros((1, 1, h, w))
        for idx, zone in enumerate((sky, background, surroundings)):
            m = zone[0][None, None].float()
            if m.shape[-2:] != (h, w):
                m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
            core = m > 0.5
            if core.sum() < 64:
                continue
            # 1) глобально: среднее и разброс каждого канала LAB зоны — как в рендере
            sel_s, sel_r = src_lab[:, :, core[0, 0]], ren_lab[:, :, core[0, 0]]
            mean_s, std_s = sel_s.mean(-1)[..., None, None], sel_s.std(-1)[..., None, None] + 1e-3
            mean_r, std_r = sel_r.mean(-1)[..., None, None], sel_r.std(-1)[..., None, None] + 1e-3
            styled = (src_lab - mean_s) / std_s * std_r + mean_r
            # 2) локально: низкочастотный свет рендера внутри зоны (нормированное размытие по маске)
            if local_light:
                blur = lambda x: kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma))  # noqa: E731
                denom = blur(m) + 1e-4
                low_r, low_s = blur(ren_lab * m) / denom, blur(styled * m) / denom
                styled = styled + local_light * (low_r - low_s)
            # 3) фактура: в фоне и окружении (не в небе — иначе облака задвоятся) плоские заливки исходника
            #    получают мелкую текстуру рендера; дальше её дорисовывает перерисовка зон
            if high is not None and idx > 0:
                src_flat = kornia.filters.gaussian_blur2d(src_lab, (dk, dk), (ds, ds))
                src_high = src_lab - src_flat
                styled = styled - src_high * 0.5 + detail * high
            out = out * (1 - m) + styled * m
            weight_total += m

        rgb = kornia.color.lab_to_rgb(out).clamp(0, 1)
        log.info("INEGO restyle: перенесён цвет в %.1f%% кадра", 100 * weight_total.clamp(0, 1).mean().item())
        return (rgb[0].permute(1, 2, 0)[None],)


def _sky_mask(rgb: np.ndarray) -> np.ndarray:
    """Цельная маска неба SegFormer (по сглаженной вероятности, без ряби и клочьев)."""
    _, _, prob = _segment(rgb, sky_prob=True)
    return _clean_sky(prob)


class InegoSkyMatch:
    """Цвет неба результата — как у неба референса: градиент от зенита к горизонту по полосам высоты (LAB).

    Облака и их расклад не трогаются: цвет считается только по чистому небу (без облаков), а яркие пиксели
    облаков получают ослабленный сдвиг — иначе белые облака окрашиваются в цвет неба (сиреневые, серые).
    """

    BANDS = 6

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
                "feather": ("FLOAT", {"default": 0.006, "min": 0.0, "max": 0.05, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "inego"

    @classmethod
    def _profile(cls, lab: torch.Tensor, sky: np.ndarray) -> np.ndarray | None:
        """Средний LAB чистого неба (без облаков: нижние 70% яркости полосы) по полосам высоты сверху вниз."""
        rows = np.where(sky.any(1))[0]
        if sky.sum() < 0.02 * sky.size or rows.size < cls.BANDS:
            return None
        top, bottom = rows.min(), rows.max() + 1
        edges = np.linspace(top, bottom, cls.BANDS + 1).astype(int)
        means = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = np.zeros_like(sky)
            m[a:b] = sky[a:b]
            if m.sum() < 32:
                means.append(None)
                continue
            sel = lab[:, torch.from_numpy(m)]
            clear = sel[:, sel[0] <= torch.quantile(sel[0], 0.7)]
            means.append(clear.mean(-1).numpy())
        valid = [x for x in means if x is not None]
        if not valid:
            return None
        return np.stack([x if x is not None else valid[0] for x in means])

    def run(self, image, reference, strength, feather):
        import kornia

        img = image[0].permute(2, 0, 1)[None].float().clamp(0, 1)
        ref = reference[0].permute(2, 0, 1)[None].float().clamp(0, 1)
        sky_i = _sky_mask(_to_uint8(image[0]))
        sky_r = _sky_mask(_to_uint8(reference[0]))
        lab_i = kornia.color.rgb_to_lab(img)[0]
        lab_r = kornia.color.rgb_to_lab(ref)[0]
        prof_i, prof_r = self._profile(lab_i, sky_i), self._profile(lab_r, sky_r)
        if prof_i is None or prof_r is None:
            log.info("INEGO sky match: неба мало (результат %.1f%%, референс %.1f%%) — пропуск",
                     100 * sky_i.mean(), 100 * sky_r.mean())
            return (image,)

        mi, mr = prof_i, prof_r
        h, w = sky_i.shape
        rows = np.where(sky_i.any(1))[0]
        top, bottom = rows.min(), rows.max() + 1
        # поправка для каждой строки: интерполяция средних полос по относительной высоте в небе
        centers = (np.arange(self.BANDS) + 0.5) / self.BANDS
        rel = np.clip((np.arange(h) - top) / max(1, bottom - top), 0, 1)
        mean_i_row = np.stack([np.interp(rel, centers, mi[:, c]) for c in range(3)])
        mean_r_row = np.stack([np.interp(rel, centers, mr[:, c]) for c in range(3)])
        mi_t = torch.from_numpy(mean_i_row).float()[:, :, None]
        mr_t = torch.from_numpy(mean_r_row).float()[:, :, None]
        # облака — пиксели заметно ярче чистого неба своей полосы: им сдвиг ослаблен, они остаются белыми
        cloud = ((lab_i[0:1] - mi_t[0:1] - 8.0) / 15.0).clamp(0, 1)
        matched = lab_i + (mr_t - mi_t) * (1 - 0.85 * cloud)

        # край — по настоящему контуру, иначе у здания остаётся полоска неба старого цвета
        m = _refine(sky_i, _to_uint8(image[0])) if feather else sky_i.astype(np.float32)
        # небо, просвечивающее сквозь редкие кроны и ветки: SegFormer относит всю крону к дереву, и небо внутри
        # оставалось старого цвета — пятно вокруг дерева. Рядом с небом подкрашиваем ещё и по цвету: пиксели,
        # близкие к чистому небу своей полосы; ветки, листва и фасады по цвету отличаются и не задеваются
        near = ndimage.binary_dilation(sky_i, iterations=max(1, round(min(h, w) * 0.04)))
        diff = lab_i - mi_t
        dist = torch.sqrt((0.5 * diff[0]) ** 2 + diff[1] ** 2 + diff[2] ** 2)
        by_colour = torch.exp(-(dist / 12.0) ** 2).numpy() * near
        m = np.maximum(m, by_colour.astype(np.float32))
        m = torch.from_numpy(m).float()[None] * strength
        out = lab_i * (1 - m) + matched * m
        rgb = kornia.color.lab_to_rgb(out[None]).clamp(0, 1)[0]
        log.info("INEGO sky match: небо %.1f%% кадра, L %.0f→%.0f (верх) / %.0f→%.0f (низ)",
                 100 * sky_i.mean(), mi[0, 0], mr[0, 0], mi[-1, 0], mr[-1, 0])
        return (rgb.permute(1, 2, 0)[None],)


def _edge_map(img: torch.Tensor, w: int) -> np.ndarray:
    """Карта контуров (модуль градиента яркости) шириной w, с обрезанными выбросами и нулевым средним."""
    x = img.permute(2, 0, 1)[None].float()
    h = max(8, round(w * x.shape[2] / x.shape[3]))
    g = F.interpolate(x.mean(1, keepdim=True), size=(h, w), mode="area")[0, 0].numpy()
    g = ndimage.gaussian_filter(g, 1.0)
    m = np.hypot(ndimage.sobel(g, 1), ndimage.sobel(g, 0))
    m = np.minimum(m, np.percentile(m, 99))  # яркие края не перевешивают остальные
    return (m - m.mean()).astype(np.float32)


def _xcorr(a: np.ndarray, b: np.ndarray, max_dx: int, max_dy: int) -> tuple[float, int, int]:
    """Сдвиг b относительно a по пику нормированной взаимной корреляции (FFT), в пределах max_dx / max_dy."""
    H, W = a.shape
    c = np.fft.irfft2(np.fft.rfft2(a, s=(2 * H, 2 * W)) * np.conj(np.fft.rfft2(b, s=(2 * H, 2 * W))), s=(2 * H, 2 * W))
    c = np.roll(np.roll(c, max_dy, 0), max_dx, 1)[: 2 * max_dy + 1, : 2 * max_dx + 1]
    i = np.unravel_index(np.argmax(c), c.shape)
    return float(c[i]) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6), int(i[0]) - max_dy, int(i[1]) - max_dx


class InegoAlign:
    """Совмещает результат с исходником: масштаб + сдвиг по контурам.

    Qwen-Image-2.1 при сильной правке (рендер → фото с референсом) слегка перекадрирует сцену: здание сдвигается
    на 2–3% кадра и чуть меняет масштаб. Ракурс тот же, поэтому хватает подобия без поворота: по картам контуров
    ищется масштаб (грубо, затем точно) и сдвиг (пик взаимной корреляции), результат возвращается на место
    исходника. Открывшийся край заполняется несдвинутым результатом. Если совпадение не лучше, чем без сдвига, —
    кадр не трогается.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source": ("IMAGE",),
                "max_scale": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.3, "step": 0.01}),
                "max_shift": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.3, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "inego"

    @staticmethod
    def _search(src: np.ndarray, out_img: torch.Tensor, w: int, scales, max_shift: float):
        H, W = src.shape
        best = (-1.0, 1.0, 0, 0)
        for s in scales:
            b_full = _edge_map(out_img, max(8, round(w * s)))
            b = np.zeros_like(src)
            hh, ww = min(H, b_full.shape[0]), min(W, b_full.shape[1])
            b[:hh, :ww] = b_full[:hh, :ww]
            # результат, увеличенный в s раз от левого верхнего угла, сдвинутый на (dx, dy), — поверх исходника
            score, dy, dx = _xcorr(src, b, max(1, round(W * (max_shift + abs(s - 1)))), max(1, round(H * (max_shift + abs(s - 1)))))
            # штраф за большое преобразование: новые деревья и кусты результата иначе перетягивают оценку
            # к краю диапазона (масштаб 1.085 вместо 1.025, здание двоится)
            score -= 0.5 * (abs(s - 1) + abs(dx) / W + abs(dy) / H)
            if score > best[0]:
                best = (score, float(s), dx, dy)
        return best

    def run(self, image, source, max_scale, max_shift):
        out = image[0][..., :3]
        src_img = source[0][..., :3]
        coarse_w, fine_w = 512, 1024
        a = _edge_map(src_img, coarse_w)
        _, s0, _, _ = self._search(a, out, coarse_w, np.arange(1 - max_scale, 1 + max_scale + 1e-6, 0.005), max_shift)
        if abs(s0 - 1) > max_scale - 0.006:
            # масштаб упёрся в край диапазона — оценка ненадёжна, лучше несовмещённый кадр, чем двоящееся здание
            log.info("INEGO align: масштаб %.3f на краю диапазона — кадр не трогаю", s0)
            return (image,)
        a = _edge_map(src_img, fine_w)
        score, s, dx, dy = self._search(a, out, fine_w, np.arange(s0 - 0.005, s0 + 0.0051, 0.001), max_shift)
        base, _, _ = _xcorr(a, _edge_map(out, fine_w)[: a.shape[0], : a.shape[1]], 0, 0)
        H, W = out.shape[:2]
        tx, ty = dx / fine_w, dy / a.shape[0]  # доли кадра
        if score <= base * 1.02 or (abs(s - 1) < 0.002 and abs(tx) * W < 1 and abs(ty) * H < 1):
            log.info("INEGO align: сдвига нет (совпадение %.3f, со сдвигом %.3f)", base, score)
            return (image,)
        # точка результата p_out переходит в s * p_out + t; обратное отображение для grid_sample
        ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
        sx = (xs - tx * W) / s
        sy = (ys - ty * H) / s
        grid = torch.stack([(sx + 0.5) / W * 2 - 1, (sy + 0.5) / H * 2 - 1], -1)[None]
        img = out.permute(2, 0, 1)[None].float()
        warped = F.grid_sample(img, grid, mode="bicubic", padding_mode="border", align_corners=False)
        inside = ((sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1)).float()
        # мягкий шов у открывшегося края: там остаётся несдвинутый результат
        inside = torch.from_numpy(ndimage.gaussian_filter(inside.numpy(), max(1.0, 0.004 * min(H, W)))).clamp(0, 1)
        res = (warped * inside + img * (1 - inside))[0].clamp(0, 1).permute(1, 2, 0)[None]
        log.info("INEGO align: масштаб %.3f, сдвиг %+.1f%% / %+.1f%% кадра (совпадение %.3f → %.3f)",
                 s, 100 * tx, 100 * ty, base, score)
        return (res,)


class InegoObjectMask:
    """Маска нового объекта после генерации (Inpainting «добавить / заменить») и его тени.

    Вклейка «всего, что изменилось рядом» забирала и побочные правки модели: высветленную траву вокруг человека,
    газон, перерисованный в плитку. Здесь: изменения рядом с выделением → их рамка → SAM 2.1 даёт точный силуэт
    объекта (точка-подсказка — центр выделения). Тень — отдельно: связанные с объектом пиксели, которые стали
    заметно темнее исходника (не светлее и не другого цвета).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # результат генерации (кроп)
                "original": ("IMAGE",),  # тот же кроп до генерации
                "seed": ("MASK",),  # выделение пользователя (откуда начинается объект)
                "allowed": ("MASK",),  # где модели разрешено было рисовать
                "threshold": ("FLOAT", {"default": 0.09, "min": 0.01, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("object", "shadow")
    FUNCTION = "run"
    CATEGORY = "inego"

    def run(self, image, original, seed, allowed, threshold):
        img = image[0].float().numpy()
        org = original[0].float().numpy()
        h, w = img.shape[:2]

        def fit(m):
            m = m[0].float()[None, None]
            if m.shape[-2:] != (h, w):
                m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
            return m[0, 0].numpy() > 0.5

        core, room = fit(seed), fit(allowed)
        empty = torch.zeros((1, h, w))
        sigma = max(1.0, min(h, w) / 400)
        a = ndimage.gaussian_filter(img, sigma=(sigma, sigma, 0))
        b = ndimage.gaussian_filter(org, sigma=(sigma, sigma, 0))
        diff = (np.abs(a - b).max(-1) > threshold) & room
        # изменения, связанные с выделением (с «мостиками» — крона, ветки)
        bridged = ndimage.binary_dilation(diff, iterations=max(1, round(min(h, w) * 0.006)))
        lab, n = ndimage.label(bridged | core)
        ids = np.unique(lab[core & (lab > 0)])
        region = np.isin(lab, ids) & (diff | core)
        if region.sum() < 64:
            log.info("INEGO object: изменений рядом с выделением нет")
            return (empty, empty)
        ys, xs = np.nonzero(region)
        box = [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]
        cy, cx = ndimage.center_of_mass(core) if core.any() else ((ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2)

        rgb = _to_uint8(image[0])

        def segment(bx):
            """SAM 2.1 по рамке (пиксели) и точке — центру выделения; самый уверенный вариант, совпадение с
            изменениями — только при почти равной уверенности (вариант «объект + кусок перерисованного фона»
            иначе выигрывал)."""
            processor, model = _sam()
            model.to(_device())
            try:
                inputs = processor(
                    images=rgb, input_boxes=[[[float(v) for v in bx]]], input_points=[[[[float(cx), float(cy)]]]],
                    input_labels=[[[1]]], return_tensors="pt",
                ).to(_device())
                with torch.no_grad():
                    out = model(**inputs, multimask_output=True)
                ms = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0].numpy() > 0
                sc = out.iou_scores[0, 0].float().cpu().numpy()
            finally:
                model.to("cpu")
            good = [i for i in range(len(sc)) if sc[i] >= sc.max() - 0.02]
            best = max(good, key=lambda i: (ms[i] & region).sum() / max(1, (ms[i] | region).sum()))
            return ms[best], float(sc[best])

        mask1, score = segment([box[0] * w, box[1] * h, box[2] * w, box[3] * h])
        # второй проход с подрезанной рамкой: рамка по изменениям растягивается на фон, который модель тоже
        # перерисовала (стенд за спиной выше головы), и SAM включал его в объект. Тонкие отростки маски срезаются
        # размыканием; верх и бока рамки — по основному силуэту, низ — как был (стопы и колёса не подрезаются)
        m1 = ndimage.binary_fill_holes(mask1)
        if m1.any():
            ys1 = np.nonzero(m1)[0]
            k = max(3, round((ys1.max() - ys1.min()) * 0.05))
            core_body = ndimage.binary_opening(m1, structure=np.ones((k, k), bool))
            lab_b, nb = ndimage.label(core_body)
            if nb:
                sizes = ndimage.sum(core_body, lab_b, range(1, nb + 1))
                body = lab_b == 1 + int(np.argmax(sizes))
                by, bx_ = np.nonzero(body)
                pad = k
                b2 = [float(max(0, bx_.min() - pad)), float(max(0, by.min() - pad)), float(min(w, bx_.max() + 1 + pad)), float(ys1.max() + 1)]
                if (b2[2] - b2[0]) * (b2[3] - b2[1]) < 0.92 * (box[2] - box[0]) * w * (box[3] - box[1]) * h:
                    mask2, score2 = segment(b2)
                    if score2 >= score - 0.05 and mask2.sum() > 0.5 * m1.sum():
                        mask1, score = mask2, score2
        masks, scores, best = [mask1], np.array([score]), 0
        obj = ndimage.binary_fill_holes(masks[best]) & room
        oy, ox = np.nonzero(obj)
        if not oy.size:
            return (empty, empty)
        top, bottom = oy.min(), oy.max()
        height = bottom - top + 1
        # тонкие края, которые SAM не взял (носок, подошва), — только у ног: в верхней части к объекту иначе
        # «прилипал» перерисованный моделью фон (стенд за головой)
        feet = np.zeros_like(obj)
        feet[int(top + 0.7 * height):] = True
        obj |= region & feet & ndimage.binary_dilation(obj, iterations=max(1, round(min(h, w) * 0.004)))

        # тень — только на земле у ног и ниже, без тонкой полосы вдоль контура объекта (на краях генерация и исходник
        # расходятся всегда — это давало тёмную обводку) и без нитевидных обрывков
        la, lb = a.mean(-1), b.mean(-1)
        band = ndimage.binary_dilation(obj, iterations=max(2, round(height * 0.02)))
        darker = (lb - la > threshold * 0.35) & room & ~band
        darker[: int(top + 0.75 * height)] = False
        k = max(2, round(height * 0.012))
        darker = ndimage.binary_opening(darker, structure=np.ones((k, k), bool))
        near = ndimage.binary_dilation(obj, iterations=max(3, round(height * 0.06)))
        lab2, _ = ndimage.label(darker | near)
        sh_ids = np.unique(lab2[near & (lab2 > 0)])
        shadow = np.isin(lab2, sh_ids) & darker
        log.info(
            "INEGO object: SAM score %.2f, объект %.1f%%, тень %.1f%% кропа (изменений рядом было %.1f%%)",
            scores[best], 100 * obj.mean(), 100 * shadow.mean(), 100 * region.mean(),
        )
        return (torch.from_numpy(obj.astype(np.float32))[None], torch.from_numpy(shadow.astype(np.float32))[None])


class InegoSurfaceMask:
    """Точный контур поверхности под мазком пользователя — для смены материала (Inpainting «Материалы»).

    Мазок кистью грубый: задевает угол стены, подоконник, отлив, и модель перекрашивала их тоже — облицовка меняла
    размер и «съезжала» с примыканий. Здесь SAM 2.1 по исходнику (точки внутри мазка + его рамка) даёт контур
    самой поверхности: окна, отливы и всё, что в неё врезано, остаются дырами. Берётся вариант, лучше всего
    совпадающий с мазком, и он подрезается мазком с запасом (поверхность за пределами мазка не трогается).
    Если SAM промахнулся (покрыл меньше половины мазка или расползся далеко за него) — возвращается сам мазок.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # исходный кроп
                "stroke": ("MASK",),  # мазок пользователя
                "reach": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.3, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "inego"

    def run(self, image, stroke, reach):
        rgb = _to_uint8(image[0])
        h, w = rgb.shape[:2]
        m = stroke[0].float()[None, None]
        if m.shape[-2:] != (h, w):
            m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
        s = m[0, 0].numpy() > 0.5
        if s.sum() < 64:
            return (stroke,)
        ys, xs = np.nonzero(s)
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        # точки-кандидаты — по сетке внутри мазка, подальше от его края (край мазка часто уже на соседней поверхности).
        # Каждая точка — отдельный запрос: все точки в одном запросе SAM склеивал в одну маску всё, на что они
        # попали (колонна + дверь и стена за ней кусками)
        inner = ndimage.binary_erosion(s, iterations=max(1, round(0.12 * min(bw, bh))))
        if inner.sum() < 16:
            inner = s
        iy, ix = np.nonzero(inner)
        pts = set()
        for gy in np.linspace(0.1, 0.9, 4):
            for gx in np.linspace(0.1, 0.9, 4):
                ty, tx = iy.min() + gy * (iy.max() - iy.min()), ix.min() + gx * (ix.max() - ix.min())
                k = int(np.argmin((iy - ty) ** 2 + (ix - tx) ** 2))
                pts.add((float(ix[k]), float(iy[k])))
        pts = sorted(pts)

        processor, model = _sam()
        model.to(_device())
        try:
            inputs = processor(
                images=rgb, input_points=[[[list(p)] for p in pts]], input_labels=[[[1] for _ in pts]],
                return_tensors="pt",
            ).to(_device())
            with torch.no_grad():
                out = model(**inputs, multimask_output=True)
            ms = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0].numpy() > 0
            sc = out.iou_scores[0].float().cpu().numpy()
        finally:
            model.to("cpu")
        ms = ms.reshape(-1, h, w)
        sc = sc.reshape(-1)
        near = ndimage.binary_dilation(s, iterations=max(2, round(reach * max(bw, bh))))
        ssum = s.sum()

        def rate(i):
            # мазок грубый и обычно шире поверхности: важно, чтобы контур лежал внутри мазка (не вылезал на соседей),
            # а не чтобы покрыл весь мазок. Колонна в широком мазке — треть мазка, но вся внутри; «колонна + стена
            # за ней» покрывает больше, но наполовину снаружи
            inside = (ms[i] & near).sum() / max(1, ms[i].sum())
            return ((ms[i] & s).sum() / ssum) * inside ** 2 + 0.1 * sc[i]

        best = max(range(len(sc)), key=rate)
        inside = (ms[best] & near).sum() / max(1, ms[best].sum())
        surface = _clean_surface(ms[best] & near, s)
        covered = (surface & s).sum() / ssum
        log.info(
            "INEGO surface: %d точек × 3 варианта, SAM score %.2f, внутри мазка %.0f%% контура, мазок покрыт на %.0f%%",
            len(pts), sc[best], 100 * inside, 100 * covered,
        )
        if inside < 0.7 or covered < 0.1:
            log.info("INEGO surface: SAM промахнулся — остаётся мазок")
            return (stroke,)
        surface = _straighten(surface, rgb)
        return (torch.from_numpy(surface.astype(np.float32))[None],)


def _straighten(m: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Прямые края маски поверхности — по прямым кадра. SAM на мелком участке кадра даёт рваный край («лесенка»
    под подоконником), а у архитектуры края прямые: подоконник, угол колонны, линия земли.

    Кандидаты — отрезки LSD вдоль края маски: с фото (настоящая граница) и с размытой маски (её усреднённая прямая
    там, где на фото контраста нет). Близкий к краю отрезок — с узкой полосой; длинный, от которого «лесенка»
    отходит дальше (до STRAIGHT_FAR), — с полосой на её глубину. Сначала ближайший к краю (из пачки параллельных
    линий подоконника — его нижняя кромка, а не верх), прямые фото — раньше прямых маски; уже выпрямленное
    следующие не перекраивают. В полосе край ставится ровно по прямой, сторона «внутри» — где маски больше: зубцы
    срезаются, ямки заливаются. Кривые края (арки) не трогаются: прямых вдоль них нет."""
    import cv2

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
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    soft = (ndimage.gaussian_filter(m.astype(np.float32), max(1.0, tol / 2)) * 255).astype(np.uint8)
    segs = []
    for source, found in ((0, det.detect(gray)[0]), (1, det.detect(soft)[0])):
        if found is None:
            continue
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
            # приоритет: ближе к краю; прямые маски — после прямых фото (они лишь среднее «лесенки»)
            key = float(np.median(dd)) + (far / 2 if source else 0)
            segs.append((key, -length, width, x1, y1, x2, y2))
    if not segs:
        return m
    out = m.copy()
    done = np.zeros_like(m)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fixed = 0
    for _, neg_len, width, x1, y1, x2, y2 in sorted(segs)[:STRAIGHT_MAX_SEGS]:
        length = -neg_len
        d = np.array([x2 - x1, y2 - y1], np.float32) / length
        nrm = np.array([-d[1], d[0]], np.float32)
        along = (xx - x1) * d[0] + (yy - y1) * d[1]
        across = (xx - x1) * nrm[0] + (yy - y1) * nrm[1]
        band = (np.abs(across) <= width) & (along >= 0) & (along <= length)
        if band.sum() < 10:
            continue
        # где маска — по ту сторону прямой
        pos = m[band & (across > 0)].mean() if (band & (across > 0)).any() else 0
        neg = m[band & (across < 0)].mean() if (band & (across < 0)).any() else 0
        if abs(pos - neg) < 0.4:
            continue  # прямая не разделяет маску и фон — это не край
        inside = (across > 0) if pos > neg else (across < 0)
        new = band & ~done
        out[new] = inside[new]
        done |= band
        fixed += 1
    log.info("INEGO surface: край выпрямлен по %d прямым (полоса %.1f…%.1f px)", fixed, tol, far)
    return out


def _clean_surface(m: np.ndarray, stroke: np.ndarray) -> np.ndarray:
    """Маска поверхности без «крапинок»: только куски, связанные с мазком и не мельче 2% маски; дырки мельче 1%
    маски заливаются (текстура камня, швы), крупные — окна, двери — остаются."""
    lab, n = ndimage.label(m)
    if not n:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    on_stroke = ndimage.sum(stroke, lab, range(1, n + 1))
    keep = [i + 1 for i in range(n) if on_stroke[i] > 0 and sizes[i] >= 0.02 * m.sum()]
    m = np.isin(lab, keep)
    holes = ndimage.binary_fill_holes(m) & ~m
    hl, hn = ndimage.label(holes)
    if hn:
        hs = ndimage.sum(holes, hl, range(1, hn + 1))
        m |= np.isin(hl, [i + 1 for i in range(hn) if hs[i] < 0.01 * m.sum()])
    return ndimage.binary_opening(m, iterations=1)


# выпрямление края маски поверхности (_straighten): полоса вокруг прямой — доля размера поверхности; отрезок не
# короче этой доли; не больше стольких прямых
STRAIGHT_TOL = 0.012
STRAIGHT_MIN_SEG = 0.05
STRAIGHT_MAX_SEGS = 40
# длинный отрезок (не короче этой доли), от которого «лесенка» SAM отходит дальше узкой полосы — до этой доли размера
STRAIGHT_LONG = 0.08
STRAIGHT_FAR = 0.02

MOGE_ID = "Ruicheng/moge-2-vitl-normal"
IID_ID = "prs-eth/marigold-iid-lighting-v1-1"
# грани: нормали ближе этого угла — одна грань; грань мельче этой доли поверхности — отбрасывается
FACE_MERGE_DEG = 20
FACE_MIN = 0.05
# грань не плоская (арка, цилиндр, волна): средний угол нормалей грани от её средней нормали больше этого.
# Разброс самих точек не годится: на узкой грани под острым углом (0.5 м) шум глубины MoGe сравним с шириной
PLANE_TOL_DEG = 14
# сглаживание освещённости внутри грани (доля высоты поверхности): швы старого материала не просвечивают
SHADE_SMOOTH = 0.006
# предфильтр образца — доля шага выборки: 1.0 — без муара, но вдвое мягче; 0.6 — резко, муара на кладке нет
TEX_PREFILTER = 0.6
# переосвещение правки по исходнику (InegoRelight): сглаживание карт освещённости (доля высоты поверхности)
# и пределы поправки света
RELIGHT_SMOOTH = 0.01
RELIGHT_MIN, RELIGHT_MAX = 0.15, 6.0
# сведение добавленного объекта (InegoHarmonize): оттенок света — полностью в этих пределах по каналу, яркость —
# в степени HARM_GAIN_POW (у объекта свои грани к свету) и в пределах
HARM_TINT_MIN, HARM_TINT_MAX = 0.8, 1.25
HARM_GAIN_POW = 0.5
HARM_GAIN_MIN, HARM_GAIN_MAX = 0.7, 1.4
# уточнение нормали стены по её краям — не больше этого угла (больше — края не те: окно, тень, обрезка)
REFINE_MAX_DEG = 25
# соседняя грань с MoGe-углом 90° ± этот допуск к уточнённой — ставится ровно под прямой угол (параллельная — так же)
SNAP_DEG = 20
# точка схода горизонталей грани (LSD + RANSAC): отрезок — не короче этой доли грани и не круче этого угла к
# горизонтали кадра; отрезок «указывает» на точку при отклонении меньше VP_TOL; поддержка — суммарная длина
# согласных отрезков, не меньше этой доли размера грани
VP_MIN_SEG = 0.025
VP_MAX_TILT = 35
VP_TOL = 1.5
VP_MIN_SUPPORT = 0.3
ASSIGN_PLANE_REL = 0.03  # «на плоскости грани»: отклонение точки меньше этой доли расстояния до камеры
SNAP_SUPPORT = 1.5
# уточнение грани по точке схода — только при такой опоре: согласных отрезков и их длины / размер грани
VP_MIN_INLIERS = 6
REFINE_MIN_SUPPORT = 1.0  # поддержка (длина согласных отрезков / размер грани) опорной грани для «под прямым углом»


def _moge():
    if "moge" not in _models:
        from moge.model.v2 import MoGeModel

        _models["moge"] = MoGeModel.from_pretrained(MOGE_ID, cache_dir=f"{folder_paths.models_dir}/moge").eval()
    return _models["moge"]


def _iid():
    if "iid" not in _models:
        from diffusers import MarigoldIntrinsicsPipeline

        _models["iid"] = MarigoldIntrinsicsPipeline.from_pretrained(
            IID_ID, torch_dtype=torch.float16, cache_dir=f"{folder_paths.models_dir}/marigold"
        )
    return _models["iid"]


def _free_vram():
    """Своим моделям нужно место: ComfyUI выгружает свои (Qwen ~16 ГБ), иначе OOM."""
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as e:  # noqa: BLE001
        log.warning("INEGO texture: не удалось освободить видеопамять: %s", e)


def _decompose(rgb: np.ndarray) -> np.ndarray:
    """Marigold-IID lighting: (3, H, W, 3) — альбедо, диффузная освещённость, остаток (линейные)."""
    from PIL import Image

    pipe = _iid().to(_device())
    try:
        out = pipe(Image.fromarray(rgb), num_inference_steps=4, processing_resolution=768, ensemble_size=1)
    finally:
        pipe.to("cpu")
        # кэш видеопамяти torch иначе остаётся за нами, и следующая Qwen-генерация падает с OOM
        torch.cuda.empty_cache()
    return np.asarray(out.prediction, dtype=np.float32)


def _to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


class InegoTextureProject:
    """Образец текстуры на поверхность 1 в 1 — проекцией, без генерации (Inpainting «Материалы» с образцом).

    Генеративная модель переносит только характер материала, а крапинки, кирпичи и волокна рисует свои. Здесь
    пиксели образца переносятся как есть:
      1) MoGe-2 по кадру — точки сцены в метрах и нормали; поверхность (контур SAM) делится на плоские грани
         по нормалям, к каждой подгоняется плоскость;
      2) каждый пиксель грани — точка на её плоскости в метрах (луч камеры ∩ плоскость): горизонталь вдоль
         грани (сшита на стыках граней), вертикаль общая — ряды кладки переходят через угол; образец
         повторяется с шагом tex_width_m, мелкость — уменьшенной копией образца (без муара);
      3) свет — с исходника: Marigold-IID раскладывает кадр на альбедо × освещённость + остаток; новый цвет =
         образец (как альбедо, общий множитель — по разложению самого образца) × освещённость + остаток.
    Не плоская поверхность или грани не найдены — пустая маска на выходе (бэкенд берёт генеративный путь).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # кроп исходника
                "surface": ("MASK",),  # точный контур поверхности (InegoSurfaceMask)
                "texture": ("IMAGE",),  # образец
                "tex_width_m": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 50.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "applied")
    FUNCTION = "run"
    CATEGORY = "inego"

    def run(self, image, surface, texture, tex_width_m):
        import cv2

        src = image[0][..., :3].float().cpu().numpy()
        H, W = src.shape[:2]
        m = surface[0].float()[None, None]
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        S = m[0, 0].numpy() > 0.5
        fail = (image, torch.zeros((1, H, W)))
        if S.sum() < 256:
            return fail
        tex = texture[0][..., :3].float().cpu().numpy()
        _free_vram()

        # 1) геометрия
        model = _moge().to(_device())
        try:
            with torch.no_grad():
                g = model.infer(torch.from_numpy(src).permute(2, 0, 1).to(_device()))
        finally:
            model.to("cpu")
            torch.cuda.empty_cache()
        P = g["points"].float().cpu().numpy()
        N = g["normal"].float().cpu().numpy()
        K = g["intrinsics"].float().cpu().numpy()
        valid = S & np.isfinite(P).all(-1) & np.isfinite(N).all(-1)
        if valid.sum() < 256:
            log.info("INEGO texture: геометрии под поверхностью нет")
            return fail

        # грани — кластеры нормалей
        n = N[valid] / (np.linalg.norm(N[valid], axis=1, keepdims=True) + 1e-6)
        rng = np.random.default_rng(0)
        c = n[rng.choice(len(n), 3, replace=False)]
        for _ in range(30):
            lab = np.argmax(n @ c.T, 1)
            c = np.stack([n[lab == i].mean(0) if (lab == i).any() else c[i] for i in range(3)])
            c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-6
        keep = []
        for i in np.argsort([-(lab == i).sum() for i in range(3)]):
            if (lab == i).mean() < FACE_MIN:
                continue
            if any(np.degrees(np.arccos(np.clip(c[i] @ c[j], -1, 1))) < FACE_MERGE_DEG for j in keep):
                continue
            keep.append(i)
        centers = c[keep]
        lab_full = np.full((H, W), -1)
        lab_full[valid] = np.argmax(n @ centers.T, 1)
        faces = []
        for i in range(len(centers)):
            fm = ndimage.binary_opening(lab_full == i, iterations=3)
            fl, k = ndimage.label(fm)
            if k:
                fm = fl == 1 + int(np.argmax(ndimage.sum(fm, fl, range(1, k + 1))))
            if fm.sum() >= FACE_MIN * S.sum():
                faces.append(fm)
        if not faces:
            log.info("INEGO texture: граней не найдено")
            return fail

        # 2) плоскости и координаты на них (метры)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        Kp = K.copy()
        Kp[0] *= W
        Kp[1] *= H
        rays = np.stack([(xx + 0.5 - Kp[0, 2]) / Kp[0, 0], (yy + 0.5 - Kp[1, 2]) / Kp[1, 1], np.ones_like(xx)], -1)
        # нормали граней — средние нормали MoGe (надёжнее подгонки по точкам на узких гранях)
        normals = []
        for i, fm in enumerate(faces):
            fn = N[fm & valid]
            fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-6)
            nf = fn.mean(0)
            nf /= np.linalg.norm(nf) + 1e-6
            spread = float(np.degrees(np.arccos(np.clip(fn @ nf, -1, 1))).mean())
            if spread > PLANE_TOL_DEG:
                log.info("INEGO texture: грань %d не плоская (нормали ±%.0f°) — генеративный путь", i, spread)
                return fail
            if nf @ (-np.median(P[fm & valid], 0)) < 0:
                nf = -nf
            normals.append(nf)
        up = _scene_up(normals, N, valid_all=np.isfinite(N).all(-1) & ~S)
        # стены: горизонталь — по точке схода верхнего и нижнего края грани на кадре (подоконник, земля, карниз).
        # MoGe ошибался в повороте грани на 10-15°, и ряды кладки сходились круче подоконника. Сначала крупные
        # грани; узкую (край короткий), стоящую к уточнённой примерно под прямым углом, — ставим ровно под 90°
        walls = [i for i in range(len(faces)) if abs(normals[i] @ up) < 0.5]
        gray = (np.clip(src @ np.array([0.299, 0.587, 0.114], np.float32), 0, 1) * 255).astype(np.uint8)
        refined, support = {}, {}
        for i in sorted(walls, key=lambda i: -faces[i].sum()):
            r = _refine_normal(faces[i], normals[i], up, Kp, gray)
            if r is not None:
                refined[i], support[i] = r
        # опора для «под прямым углом» — только надёжно уточнённая грань (много согласных отрезков): от грани,
        # уточнённой по пяти коротким штрихам, ошибка уходила на соседнюю
        strong = [i for i in refined if support[i] >= SNAP_SUPPORT]
        if strong:
            ref_i = max(strong, key=lambda i: support[i] * faces[i].sum())
            ref_n = refined[ref_i]
            for i in walls:
                if i in strong:
                    continue
                ang = float(np.degrees(np.arccos(np.clip(abs(normals[i] @ ref_n), -1, 1))))
                if abs(ang - 90) < SNAP_DEG:
                    n_new = np.cross(ref_n, up)
                    refined[i] = (n_new if n_new @ normals[i] > 0 else -n_new).astype(np.float32)
                    log.info("INEGO texture: грань %d — под прямым углом к грани %d (MoGe: %.0f°)", i, ref_i, ang)
                elif ang < SNAP_DEG:
                    refined[i] = ref_n if ref_n @ normals[i] > 0 else -ref_n
        elif len(walls) >= 2:
            # надёжной опоры нет: соседние стены, которые MoGe ставит примерно под прямым углом (колонна, угол
            # дома), — ровно под 90°, симметрично от их биссектрисы
            a_i, b_i = sorted(walls, key=lambda i: -faces[i].sum())[:2]
            a, b = refined.get(a_i, normals[a_i]), refined.get(b_i, normals[b_i])
            ang = float(np.degrees(np.arccos(np.clip(abs(normals[a_i] @ normals[b_i]), -1, 1))))
            if abs(ang - 90) < SNAP_DEG:
                ha, hb = a - (a @ up) * up, b - (b @ up) * up
                ha /= np.linalg.norm(ha) + 1e-6
                hb /= np.linalg.norm(hb) + 1e-6
                mid = ha + hb
                mid /= np.linalg.norm(mid) + 1e-6
                side = np.cross(up, mid)
                side /= np.linalg.norm(side) + 1e-6
                c45 = np.float32(np.sqrt(0.5))
                for i, h in ((a_i, ha), (b_i, hb)):
                    sgn = 1.0 if h @ side > 0 else -1.0
                    refined[i] = (c45 * mid + sgn * c45 * side).astype(np.float32)
                log.info("INEGO texture: грани %d и %d — ровно под прямым углом (MoGe: %.0f°)", a_i, b_i, ang)
        owner = _assign_faces(faces, [refined.get(i, normals[i]) for i in range(len(faces))], P, N, valid, S)
        UV = np.zeros((H, W, 2), np.float32)
        for i, fm in enumerate(faces):
            pts = P[fm & valid]
            nf = refined.get(i, normals[i])
            # плоскость проходит через точки грани: смещение — медиана по ним
            cen = nf * float(np.median(pts @ nf))
            v = up - (up @ nf) * nf
            if np.linalg.norm(v) < 0.3:  # горизонтальная поверхность (пол, терраса): «верх» — от камеры вглубь
                v = np.array([0, 0, 1], np.float32) - nf[2] * nf
            v /= np.linalg.norm(v)
            u = np.cross(v, nf)
            u /= np.linalg.norm(u)
            if u[0] < 0:
                u = -u
            d = -(nf @ cen)
            sel = owner == i
            den = rays[sel] @ nf
            t = -d / np.where(np.abs(den) < 1e-6, 1e-6, den)
            X = rays[sel] * t[:, None]
            UV[sel, 0] = X @ u
            UV[sel, 1] = X @ v
            log.info("INEGO texture: грань %d — нормаль %s, %.2f × %.2f м", i, np.round(nf, 2),
                     np.ptp(UV[sel, 0]), np.ptp(UV[sel, 1]))
        # сшивка горизонтали на стыках: от крупной грани к соседним
        order = sorted(range(len(faces)), key=lambda i: -(owner == i).sum())
        done = {order[0]}
        for _ in range(len(faces)):
            for i in order:
                if i in done:
                    continue
                for j in list(done):
                    ej = ndimage.binary_dilation(owner == i, iterations=2) & (owner == j)
                    ei = ndimage.binary_dilation(owner == j, iterations=2) & (owner == i)
                    if ej.sum() > 10 and ei.sum() > 10:
                        UV[owner == i, 0] += UV[ej, 0].mean() - UV[ei, 0].mean()
                        done.add(i)
                        break

        # 3) образец на гранях: повтор, мелкость — уменьшенной копией (по грани)
        th, tw = tex.shape[:2]
        ppm = tw / tex_width_m
        proj = np.zeros_like(src)
        gx = np.hypot(np.gradient(UV[..., 0], axis=1), np.gradient(UV[..., 1], axis=0))
        for i in range(len(faces)):
            sel = owner == i
            # предфильтр слабее шага выборки (TEX_PREFILTER): уменьшенная копия + билинейная выборка сглаживали
            # дважды — кирпич выходил «мыльным»; выборка бикубическая
            lvl = max(1.0, TEX_PREFILTER * float(np.median(gx[sel])) * ppm)
            ts = cv2.resize(tex, (max(8, round(tw / lvl)), max(8, round(th / lvl))), interpolation=cv2.INTER_AREA)
            p = ppm / lvl
            mx = (UV[..., 0] * p) % ts.shape[1]
            my = (-UV[..., 1] * p) % ts.shape[0]
            wrap = cv2.copyMakeBorder(ts, 4, 4, 4, 4, cv2.BORDER_WRAP)
            r = cv2.remap(wrap, (mx + 4).astype(np.float32), (my + 4).astype(np.float32), cv2.INTER_CUBIC)
            proj[sel] = np.clip(r[sel], 0, 1)

        # 4) свет исходника
        iid = _decompose((np.clip(src, 0, 1) * 255).astype(np.uint8))
        alb, shd, rsd = iid[0], iid[1], iid[2]
        sig = max(1.0, SHADE_SMOOTH * np.ptp(np.nonzero(S)[0]))
        for i in range(len(faces)):
            fm = (owner == i).astype(np.float32)
            den = ndimage.gaussian_filter(fm, sig) + 1e-4
            for ch in range(3):
                shd[..., ch] = np.where(owner == i, ndimage.gaussian_filter(shd[..., ch] * fm, sig) / den, shd[..., ch])
        I = _to_lin(src)
        cal = float(np.median((I[S].mean(1) + 1e-4) / ((alb * shd + rsd)[S].mean(1) + 1e-4)))
        t_alb = float(_decompose((np.clip(tex, 0, 1) * 255).astype(np.uint8))[0].mean())
        k = t_alb / max(1e-4, float(_to_lin(tex).mean()))
        out = _to_srgb(cal * (_to_lin(proj) * k * shd + rsd))
        log.info("INEGO texture: %d граней, образец %.2f м, калибровка %.2f, альбедо образца ×%.2f", len(faces),
                 tex_width_m, cal, k)

        out = _match_grain(out, src, S)

        alpha = ndimage.gaussian_filter(S.astype(np.float32), 0.8)[..., None]
        res = src * (1 - alpha) + out * alpha
        return (torch.from_numpy(res.astype(np.float32))[None], torch.from_numpy(S.astype(np.float32))[None])


class InegoRelight:
    """Свет исходника на изменённой поверхности (Inpainting «изменить» и материалы текстом), без генерации.

    Геометрия поверхности при смене материала или цвета та же, значит и свет на ней тот же, что на фото. Модель
    же рисует новый материал в своём свете: белая стена — плоской заливкой, венге — с синей тенью. Marigold-IID
    раскладывает и исходник, и результат на альбедо × освещённость; освещённость результата заменяется
    освещённостью исходника. Меняется только отношение гладких карт освещённости, поэтому вся мелкая фактура
    генерации (волокна, зерно, швы) остаётся в полном разрешении. Заменяет прогон переосвещения через Qwen:
    секунды вместо минуты, свет совпадает с фото, цвет материала не уходит.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # кроп с вклеенной правкой
                "source": ("IMAGE",),  # тот же кроп до правки
                "mask": ("MASK",),  # изменённая поверхность
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "inego"

    def run(self, image, source, mask):
        gen = image[0][..., :3].float().cpu().numpy()
        src = source[0][..., :3].float().cpu().numpy()
        H, W = gen.shape[:2]
        m = mask[0].float()[None, None]
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
        S = m[0, 0].numpy() > 0.5
        if S.sum() < 256 or src.shape != gen.shape:
            return (image,)
        _free_vram()
        d_src = _decompose((np.clip(src, 0, 1) * 255).astype(np.uint8))
        d_gen = _decompose((np.clip(gen, 0, 1) * 255).astype(np.uint8))
        # гладкие карты освещённости по поверхности (нормированная свёртка — соседние поверхности не подмешиваются):
        # швы старого материала и фактура нового в них не попадают
        sig = max(1.5, RELIGHT_SMOOTH * np.ptp(np.nonzero(S)[0]))
        fm = S.astype(np.float32)
        den = ndimage.gaussian_filter(fm, sig) + 1e-4

        def smooth(x):
            return np.stack([ndimage.gaussian_filter(x[..., c] * fm, sig) / den for c in range(3)], -1)

        s_src, s_gen = smooth(d_src[1]), smooth(d_gen[1])
        # общий уровень: разложения — каждое со своим множителем, сводим их по фото
        cal_src = float(np.median((_to_lin(src)[S].mean(1) + 1e-4) / ((d_src[0] * d_src[1] + d_src[2])[S].mean(1) + 1e-4)))
        cal_gen = float(np.median((_to_lin(gen)[S].mean(1) + 1e-4) / ((d_gen[0] * d_gen[1] + d_gen[2])[S].mean(1) + 1e-4)))
        ratio = np.clip((cal_src * s_src + 1e-4) / (cal_gen * s_gen + 1e-4), RELIGHT_MIN, RELIGHT_MAX)
        out = _to_srgb(_to_lin(gen) * ratio)
        log.info("INEGO relight: поверхность %.1f%%, свет ×%.2f..%.2f (медиана %.2f)", 100 * S.mean(),
                 float(ratio[S].min()), float(ratio[S].max()), float(np.median(ratio[S])))
        alpha = ndimage.gaussian_filter(fm, 0.8)[..., None]
        res = gen * (1 - alpha) + out * alpha
        return (torch.from_numpy(res.astype(np.float32))[None],)


class InegoHarmonize:
    """Сборка добавленного объекта с исходником по свету (Inpainting «добавить / заменить»): кроп исходника +
    тень + объект из генерации, сведённый по оттенку и яркости со светом сцены.

    Модель рисует объект в своём свете — теплее или холоднее сцены, ярче или темнее, — и он выглядит вклеенным.
    Marigold-IID раскладывает исходник и генерацию на альбедо × освещённость:
      - объект: свет сцены вокруг него (кольцо по исходнику) против света, в котором его нарисовала модель
        (освещённость генерации на объекте). Оттенок света сводится полностью, яркость — частично (HARM_GAIN_POW):
        у объекта свои грани, повёрнутые к свету иначе, чем земля и стена рядом;
      - тень: там, где освещённость генерации ниже исходной, — затемнение исходника на это отношение. Трава и
        плитка под тенью остаются пикселями фото (раньше тень бралась по яркости, и в неё попадало всё, что модель
        перекрасила на газоне).
    Край объекта — как в _compose_object бэкенда: по чуть сжатой маске, без тёмного ореола фона генерации.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated": ("IMAGE",),  # кроп генерации
                "source": ("IMAGE",),  # тот же кроп исходника
                "object": ("MASK",),
                "shadow": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "inego"

    def run(self, generated, source, object, shadow):  # noqa: A002 — имя входа ноды
        g = generated[0][..., :3].float().cpu().numpy()
        o = source[0][..., :3].float().cpu().numpy()
        H, W = o.shape[:2]

        def fit(mk):
            mk = mk[0].float()[None, None]
            if mk.shape[-2:] != (H, W):
                mk = F.interpolate(mk, size=(H, W), mode="bilinear", align_corners=False)
            return mk[0, 0].numpy() > 0.5

        if g.shape != o.shape:
            return (generated,)
        m, sh = fit(object), fit(shadow)
        if m.sum() < 64:
            return (source,)
        ys, xs = np.nonzero(m)
        size = max(np.ptp(ys) + 1, np.ptp(xs) + 1)
        _free_vram()
        d_o = _decompose((np.clip(o, 0, 1) * 255).astype(np.uint8))
        d_g = _decompose((np.clip(g, 0, 1) * 255).astype(np.uint8))

        def calib(img, d, region):
            return float(np.median((_to_lin(img)[region].mean(1) + 1e-4) / ((d[0] * d[1] + d[2])[region].mean(1) + 1e-4)))

        everywhere = np.ones_like(m)
        cal_o, cal_g = calib(o, d_o, everywhere), calib(g, d_g, everywhere)
        shd_o, shd_g = cal_o * d_o[1], cal_g * d_g[1]

        # тень: отношение гладких освещённостей генерации и исходника (только темнее), вес — по области тени
        s_lum = max(2.0, size * 0.012)
        lo = ndimage.gaussian_filter(shd_o.mean(-1), s_lum)
        lg = ndimage.gaussian_filter(shd_g.mean(-1), s_lum)
        ratio = np.clip((lg + 1e-4) / (lo + 1e-4), 0.2, 1.0)
        wgt = np.clip(ndimage.gaussian_filter(sh.astype(np.float32), max(3.0, size * 0.04)) * 1.6, 0, 1)
        out = _to_lin(o) * (1 - wgt * (1 - ratio))[..., None]

        # объект: свет сцены вокруг против света, в котором он нарисован
        core = ndimage.binary_erosion(m, iterations=max(1, round(size * 0.02)))
        if core.sum() < 32:
            core = m
        ring = ndimage.binary_dilation(m, iterations=max(4, round(size * 0.15))) & ~ndimage.binary_dilation(
            m, iterations=max(2, round(size * 0.03))) & ~ndimage.binary_dilation(sh, iterations=2)
        lin_g = _to_lin(g)
        if ring.sum() > 64:
            scene = np.median(shd_o[ring], 0)
            drawn = np.median(shd_g[core], 0)
            lum = np.array([0.2126, 0.7152, 0.0722], np.float32)
            tint = np.clip((scene / (scene @ lum + 1e-4)) / (drawn / (drawn @ lum + 1e-4) + 1e-4), HARM_TINT_MIN, HARM_TINT_MAX)
            gain = float(np.clip(((scene @ lum) / (drawn @ lum + 1e-4)) ** HARM_GAIN_POW, HARM_GAIN_MIN, HARM_GAIN_MAX))
            lin_g = lin_g * (tint * gain)[None, None, :]
            log.info("INEGO harmonize: оттенок света ×%s, яркость ×%.2f", np.round(tint, 2), gain)

        # край — с чуть сжатой маски: пиксели фона из генерации вокруг контура (тёмный ореол) не попадают
        a = ndimage.gaussian_filter(ndimage.binary_erosion(m, iterations=1).astype(np.float32), max(0.8, size * 0.0015))
        out = out * (1 - a[..., None]) + lin_g * a[..., None]
        return (torch.from_numpy(_to_srgb(out).astype(np.float32))[None],)


def _scene_up(normals: list, N: np.ndarray, valid_all: np.ndarray) -> np.ndarray:
    """Вертикаль сцены в координатах камеры (OpenCV: y вниз). Камера бывает чуть наклонена — «строго вниз по
    кадру» разворачивал рисунок. Две стены под углом: вертикаль — их общее ребро (ошибки нормалей по курсу
    сокращаются); иначе — нормаль земли рядом; иначе — ось кадра."""
    up = np.array([0, -1, 0], np.float32)
    walls = [n for n in normals if abs(n @ up) < 0.5]
    for a in range(len(walls)):
        for b in range(a + 1, len(walls)):
            if np.degrees(np.arccos(np.clip(abs(walls[a] @ walls[b]), -1, 1))) > 30:
                e = np.cross(walls[a], walls[b])
                e /= np.linalg.norm(e) + 1e-6
                e = e if e @ up > 0 else -e
                if e @ up > np.cos(np.radians(15)):
                    return e.astype(np.float32)
    g = N[valid_all]
    if len(g):
        g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-6)
        ground = g[g @ up > np.cos(np.radians(20))]
        if len(ground) > 0.02 * len(g):
            e = ground.mean(0)
            return (e / (np.linalg.norm(e) + 1e-6)).astype(np.float32)
    return up


def _assign_faces(faces: list, normals: list, P: np.ndarray, N: np.ndarray, valid: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Чья грань каждый пиксель поверхности: та, с чьей нормалью совпадает его нормаль и на чьей плоскости лежит его
    3D-точка. Раньше спорные пиксели доставались ближайшей на кадре грани, и полосу под окном резало по диагонали
    (граница равных расстояний) — на одной полосе две плоскости, доски с изломом. Метки сглаживаются, мелкие
    островки уходят соседям; пиксели без геометрии — ближайшей грани."""
    H, W = S.shape
    if len(faces) == 1:
        owner = np.zeros((H, W), int)
        owner[~S] = -1
        return owner
    Pn = np.where(np.isfinite(P), P, 0)
    depth = np.linalg.norm(Pn, axis=-1) + 1e-6
    Nn = np.where(np.isfinite(N), N, 0)
    Nn = Nn / (np.linalg.norm(Nn, axis=-1, keepdims=True) + 1e-6)
    cos_tol = 1 - np.cos(np.radians(FACE_MERGE_DEG))
    costs = []
    for fm, nf in zip(faces, normals):
        off = float(np.median(P[fm & valid] @ nf))
        dist = np.abs(Pn @ nf - off) / (ASSIGN_PLANE_REL * depth)
        costs.append(dist + (1 - Nn @ nf) / cos_tol)
    owner = np.argmin(np.stack(costs), 0)
    # пиксели без геометрии — ближайшей грани по уже размеченным
    if (S & ~valid).any():
        known = S & valid
        _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
        owner = np.where(known, owner, owner[iy, ix])
    # сглаживание: метка — та, чья доля в окрестности больше
    sig = max(1.5, 0.01 * max(H, W))
    onehot = np.stack([ndimage.gaussian_filter(((owner == i) & S).astype(np.float32), sig) for i in range(len(faces))])
    owner = np.argmax(onehot, 0)
    owner[~S] = -1
    return owner


def _horizontal_vp(gray: np.ndarray, fm: np.ndarray) -> np.ndarray | None:
    """Точка схода горизонталей грани (однородная) по отрезкам на исходнике: LSD внутри грани и по её краю —
    подоконник, линия земли, верх колонны и швы старого материала (у камня, кирпича, сайдинга они горизонтальны).
    RANSAC: точка, на которую указывает больше всего длины отрезков; уточнение — наименьшие квадраты по ним.
    Г-образной грани (колонна + полоса под окном) так же хватает, как прямоугольной. None — отрезков мало."""
    import cv2

    ys, xs = np.nonzero(fm)
    size = max(np.ptp(ys), np.ptp(xs)) + 1
    region = ndimage.binary_dilation(fm, iterations=max(3, round(size * 0.01)))
    det = cv2.createLineSegmentDetector()
    # отрезки фото и края самой маски грани: контур SAM точен, его верх и низ — подоконник и земля, даже когда на
    # фото между камнем и плиткой контраста нет и LSD их не видит
    found = [f for f in (det.detect(gray)[0], det.detect((fm * 255).astype(np.uint8))[0]) if f is not None]
    if not found:
        return None
    segs = []
    for x1, y1, x2, y2 in np.concatenate([np.asarray(f).reshape(-1, 4) for f in found]):
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < max(10.0, VP_MIN_SEG * size):
            continue
        if np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))) > VP_MAX_TILT:
            continue  # вертикали и крутые диагонали — не горизонтали грани
        t = np.linspace(0, 1, 7)
        px = np.clip(np.round(x1 + t * (x2 - x1)).astype(int), 0, fm.shape[1] - 1)
        py = np.clip(np.round(y1 + t * (y2 - y1)).astype(int), 0, fm.shape[0] - 1)
        if region[py, px].mean() < 0.8:
            continue
        line = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        line /= np.hypot(line[0], line[1]) + 1e-9
        segs.append((line, np.array([(x1 + x2) / 2, (y1 + y2) / 2]), np.array([x2 - x1, y2 - y1]) / length, length))
    if len(segs) < 3:
        return None
    lines = np.stack([s[0] for s in segs])
    mids = np.stack([s[1] for s in segs])
    dirs = np.stack([s[2] for s in segs])
    lens = np.array([s[3] for s in segs])

    def inliers(vp):
        # угол между отрезком и направлением от его середины к точке схода
        if abs(vp[2]) > 1e-9:
            to = vp[:2] / vp[2] - mids
        else:
            to = np.tile(vp[:2], (len(mids), 1))
        to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-9
        return np.degrees(np.arccos(np.clip(np.abs((to * dirs).sum(1)), 0, 1))) < VP_TOL

    rng = np.random.default_rng(0)
    p = lens / lens.sum()
    best, best_score = None, 0.0
    for _ in range(400):
        i, j = rng.choice(len(segs), 2, replace=False, p=p)
        vp = np.cross(lines[i], lines[j])
        if np.linalg.norm(vp) < 1e-12:
            continue
        inl = inliers(vp)
        score = lens[inl].sum()
        if score > best_score:
            best, best_score = inl, score
    if best is None or best.sum() < 3 or best_score < VP_MIN_SUPPORT * size:
        return None
    _, _, vt = np.linalg.svd(lines[best] * lens[best, None])
    log.info("INEGO texture: точка схода по %d отрезкам (%.0f px)", int(best.sum()), best_score)
    return vt[-1], best_score / size, int(best.sum())


def _refine_normal(
    fm: np.ndarray, nf: np.ndarray, up: np.ndarray, Kp: np.ndarray, gray: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Нормаль стены, уточнённая по точке схода её горизонталей на кадре (_horizontal_vp): u = K⁻¹·VP — направление
    горизонтали грани. Вертикаль сцены для этого не нужна (вариант через одну прямую и вертикаль ошибался на
    десятки градусов от наклона камеры в пару градусов). Точка не найдена или дальше REFINE_MAX_DEG от MoGe — None."""
    u0 = np.cross(up, nf)
    u0 /= np.linalg.norm(u0) + 1e-6
    found = _horizontal_vp(gray, fm)
    if found is None:
        log.info("INEGO texture: горизонталей у грани мало — нормаль MoGe")
        return None
    vp, support, count = found
    if count < VP_MIN_INLIERS or support < REFINE_MIN_SUPPORT:
        # узкая грань колонны: 4 коротких штриха поворачивали её на 15-25°, и две стороны колонны сходились под 41°
        log.info("INEGO texture: опора точки схода слабая (%d отрезков, %.1f) — нормаль MoGe", count, support)
        return None
    u = np.linalg.inv(Kp) @ vp
    if np.linalg.norm(u) < 1e-12:
        return None
    u /= np.linalg.norm(u)
    u = u if u @ u0 > 0 else -u
    u -= (u @ up) * up
    u /= np.linalg.norm(u) + 1e-6
    n_new = np.cross(u, up)
    n_new = n_new if n_new @ nf > 0 else -n_new
    ang = float(np.degrees(np.arccos(np.clip(n_new @ nf, -1, 1))))
    if ang > REFINE_MAX_DEG:
        log.info("INEGO texture: горизонтали грани расходятся с MoGe на %.0f° — оставляю нормаль MoGe", ang)
        return None
    log.info("INEGO texture: нормаль грани уточнена по точке схода на %.1f°", ang)
    return n_new.astype(np.float32), support


def _match_grain(out: np.ndarray, src: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Зерно сенсора как на фото: на подставленной текстуре его нет, и она выглядела «чистой CG» на фоне
    шумного кадра. Уровень — по мелкой составляющей исходника на соседних гладких местах (MAD), добавляется
    недостающая до него доля, в основном по яркости."""
    def hf_std(img, m):
        hf = img - ndimage.gaussian_filter(img, (1.0, 1.0, 0))
        v = hf[m]
        return 1.4826 * np.median(np.abs(v - np.median(v, 0)), 0) if len(v) else np.zeros(3)

    ring = ndimage.binary_dilation(S, iterations=12) & ~ndimage.binary_dilation(S, iterations=3)
    want = hf_std(src, ring) if ring.sum() > 500 else hf_std(src, S)
    have = hf_std(out, S)
    add = np.sqrt(np.clip(want ** 2 - have ** 2, 0, None))
    if add.max() < 1 / 255:
        return out
    rng = np.random.default_rng(0)
    lum = rng.standard_normal(out.shape[:2]).astype(np.float32)
    chroma = rng.standard_normal(out.shape).astype(np.float32)
    noise = ndimage.gaussian_filter(0.85 * lum[..., None] + 0.35 * chroma, (0.6, 0.6, 0))
    noise /= noise.std() + 1e-6
    log.info("INEGO texture: зерно +%s", np.round(add * 255, 1))
    return np.clip(out + noise * add, 0, 1)


NODE_CLASS_MAPPINGS = {"InegoKeepZones": InegoKeepZones, "InegoZoneRestyle": InegoZoneRestyle, "InegoSkyMatch": InegoSkyMatch, "InegoObjectMask": InegoObjectMask, "InegoAlign": InegoAlign, "InegoSurfaceMask": InegoSurfaceMask, "InegoTextureProject": InegoTextureProject, "InegoRelight": InegoRelight, "InegoHarmonize": InegoHarmonize}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InegoHarmonize": "INEGO Harmonize (added object: light tint, exposure and shadow from Marigold-IID)",
    "InegoRelight": "INEGO Relight (source lighting onto an edited surface, Marigold-IID)",
    "InegoTextureProject": "INEGO Texture Project (sample texture 1:1 onto planar faces, MoGe-2 + Marigold-IID)",
    "InegoSurfaceMask": "INEGO Surface Mask (exact surface under a stroke, SAM 2.1)",
    "InegoKeepZones": "INEGO Keep Zones (sky / background / surroundings)",
    "InegoZoneRestyle": "INEGO Zone Restyle (source content, render light)",
    "InegoSkyMatch": "INEGO Sky Match (sky colour from reference)",
    "InegoObjectMask": "INEGO Object Mask (new object + shadow, SAM 2.1)",
    "InegoAlign": "INEGO Align (match result framing to source)",
}
