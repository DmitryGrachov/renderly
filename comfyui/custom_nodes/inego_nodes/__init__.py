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


def _segment(rgb: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    """Карта классов ADE20k [H, W]."""
    processor, model = _segformer()
    h, w = rgb.shape[:2]
    model.to(_device())
    try:
        inputs = processor(images=rgb, return_tensors="pt").to(_device())
        with torch.no_grad():
            logits = model(**inputs).logits
        labels = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
    finally:
        model.to("cpu")
    names = {int(k): v.split(";")[0].strip().lower() for k, v in model.config.id2label.items()}
    return labels, names


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
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("sky", "background", "surroundings", "keep")
    FUNCTION = "run"
    CATEGORY = "inego"

    def run(self, image, keep_sky, keep_background, keep_surroundings, grow, feather, building_box=""):
        h, w = image.shape[1:3]
        empty = torch.zeros((1, h, w))
        if not (keep_sky or keep_background or keep_surroundings):
            return (empty, empty, empty, empty)

        rgb = _to_uint8(image[0])
        labels, names = _segment(rgb)
        ids = lambda group: [i for i, n in names.items() if n in group]  # noqa: E731
        sky = np.isin(labels, ids(SKY))
        split = keep_background or keep_surroundings

        building = None
        box = _parse_box(building_box)
        if split and box:
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


NODE_CLASS_MAPPINGS = {"InegoKeepZones": InegoKeepZones, "InegoZoneRestyle": InegoZoneRestyle}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InegoKeepZones": "INEGO Keep Zones (sky / background / surroundings)",
    "InegoZoneRestyle": "INEGO Zone Restyle (source content, render light)",
}
