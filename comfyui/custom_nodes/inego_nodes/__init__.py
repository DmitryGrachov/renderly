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


NODE_CLASS_MAPPINGS = {"InegoKeepZones": InegoKeepZones, "InegoZoneRestyle": InegoZoneRestyle, "InegoSkyMatch": InegoSkyMatch, "InegoObjectMask": InegoObjectMask, "InegoAlign": InegoAlign}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InegoKeepZones": "INEGO Keep Zones (sky / background / surroundings)",
    "InegoZoneRestyle": "INEGO Zone Restyle (source content, render light)",
    "InegoSkyMatch": "INEGO Sky Match (sky colour from reference)",
    "InegoObjectMask": "INEGO Object Mask (new object + shadow, SAM 2.1)",
    "InegoAlign": "INEGO Align (match result framing to source)",
}
