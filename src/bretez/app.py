from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

from bretez.config import DINO_PATCH_SIZE, MAP_PIXELS_PER_FEATURE


EPS = 1e-8
CACHE_VERSION = "1"
DEFAULT_FEATURES_PATH = "features.pt"
DISTANCE_OVERLAY_ALPHA = 0.72
DISTANCE_OVERLAY_ALPHA_MIN = 0.08
DISTANCE_COLORMAP = "magma_r"
DISTANCE_DISPLAY_MAX = 0.85
DISTANCE_DISPLAY_GAMMA = 0.65


@dataclass(slots=True)
class FeatureStore:
    path: Path
    features: torch.Tensor
    flat_features: torch.Tensor
    preview_rgb: np.ndarray
    map_rgb: np.ndarray | None
    map_scale_x: float
    map_scale_y: float
    preview_step: int
    cache_dir: Path
    cache_key: str
    device: torch.device
    distance_chunk_size: int
    _norms: torch.Tensor | None = None
    _last_distance_xy: tuple[int, int] | None = None
    _last_distance: np.ndarray | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        cache_dir: str | Path = ".bretez_cache",
        max_preview_pixels: int = 1_500_000,
        pca_sample: int = 50_000,
        map_image_path: str | Path | None = None,
        include_map: bool = True,
        device: str = "auto",
        distance_chunk_size: int = 16_384,
    ) -> "FeatureStore":
        feature_path = Path(path).expanduser().resolve()
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        features = _load_feature_tensor(feature_path)
        width, height, channels = _feature_shape(features)
        flat_features = features.reshape(width * height, channels)
        preview_step = _preview_step(width, height, max_preview_pixels)
        resolved_cache_dir = Path(cache_dir).expanduser().resolve()
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)

        cache_key = _feature_cache_key(feature_path)
        preview_key = _preview_cache_key(cache_key, preview_step, pca_sample)
        preview_path = resolved_cache_dir / f"{feature_path.stem}-{preview_key}-preview.npz"
        if preview_path.exists():
            preview_rgb = np.load(preview_path)["rgb"]
        else:
            preview_rgb = _build_preview_rgb(features, preview_step, pca_sample)
            np.savez_compressed(preview_path, rgb=preview_rgb)

        map_rgb = None
        map_scale_x = 1.0
        map_scale_y = 1.0
        if include_map:
            map_rgb, map_scale_x, map_scale_y = _load_map_rgb(map_image_path, width, height)

        return cls(
            path=feature_path,
            features=features,
            flat_features=flat_features,
            preview_rgb=preview_rgb,
            map_rgb=map_rgb,
            map_scale_x=map_scale_x,
            map_scale_y=map_scale_y,
            preview_step=preview_step,
            cache_dir=resolved_cache_dir,
            cache_key=cache_key,
            device=_select_device(device),
            distance_chunk_size=distance_chunk_size,
        )

    @property
    def width(self) -> int:
        return int(self.features.shape[0])

    @property
    def height(self) -> int:
        return int(self.features.shape[1])

    @property
    def channels(self) -> int:
        return int(self.features.shape[2])

    @property
    def preview_width(self) -> int:
        return int(self.preview_rgb.shape[1])

    @property
    def preview_height(self) -> int:
        return int(self.preview_rgb.shape[0])

    @property
    def norms(self) -> torch.Tensor:
        if self._norms is not None:
            return self._norms

        norms_path = self.cache_dir / f"{self.path.stem}-{self.cache_key}-norms.npy"
        if norms_path.exists():
            self._norms = torch.from_numpy(np.load(norms_path)).float().clamp_min_(EPS)
            return self._norms

        norms = torch.empty(self.flat_features.shape[0], dtype=torch.float32)
        for start in range(0, self.flat_features.shape[0], self.distance_chunk_size):
            end = min(start + self.distance_chunk_size, self.flat_features.shape[0])
            chunk = torch.nan_to_num(self.flat_features[start:end].float())
            norms[start:end] = torch.linalg.vector_norm(chunk, dim=1)
        norms.clamp_min_(EPS)
        np.save(norms_path, norms.numpy())
        self._norms = norms
        return norms

    def clamp_xy(self, x_value: Any, y_value: Any) -> tuple[int, int]:
        x = int(round(float(x_value)))
        y = int(round(float(y_value)))
        return min(max(x, 0), self.width - 1), min(max(y, 0), self.height - 1)

    def point_from_event(self, evt: gr.SelectData) -> tuple[int, int]:
        return self.point_from_feature_event(evt)

    def point_from_feature_event(self, evt: gr.SelectData) -> tuple[int, int]:
        image_x, image_y = _event_image_xy(evt)
        return self.clamp_xy(image_x * self.preview_step, image_y * self.preview_step)

    def point_from_map_event(self, evt: gr.SelectData) -> tuple[int, int]:
        if self.map_rgb is None:
            return self.point_from_feature_event(evt)
        image_x, image_y = _event_image_xy(evt)
        return self.clamp_xy(image_x / self.map_scale_x, image_y / self.map_scale_y)

    def render_view(self, selected: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        map_image = self.map_image(selected)
        feature_image = self.feature_image(selected)
        overlay_image, status = self.distance_overlay_image(selected)
        return map_image, feature_image, overlay_image, status

    def map_image(self, selected: tuple[int, int] | None = None) -> np.ndarray:
        if self.map_rgb is None:
            image = np.full((self.preview_height, self.preview_width, 3), 245, dtype=np.uint8)
            self.draw_feature_marker(image, selected, color=(255, 255, 255), outline=(0, 0, 0))
        else:
            image = self.map_rgb.copy()
            self.draw_map_marker(image, selected, color=(255, 255, 255), outline=(0, 0, 0))
        return image

    def feature_image(self, selected: tuple[int, int] | None = None) -> np.ndarray:
        image = self.preview_rgb.copy()
        self.draw_feature_marker(image, selected, color=(255, 255, 255), outline=(0, 0, 0))
        return image

    def distance_overlay_image(self, selected: tuple[int, int] | None = None) -> tuple[np.ndarray, str]:
        if selected is None:
            return self.map_image(), "No feature selected"

        x, y = selected
        distance = self.cached_cosine_distance(x, y)
        if self.map_rgb is None:
            distance_image = np.swapaxes(distance[:: self.preview_step, :: self.preview_step], 0, 1)
            image = _blend_distance_over_map(self.map_image(), distance_image, DISTANCE_OVERLAY_ALPHA)
            self.draw_feature_marker(image, selected, color=(255, 255, 255), outline=(0, 0, 0))
        else:
            distance_image = _resize_feature_values_to_map(distance, self.map_rgb.shape[:2], self.map_scale_x, self.map_scale_y)
            image = _blend_distance_over_map(self.map_rgb, distance_image, DISTANCE_OVERLAY_ALPHA)
            self.draw_map_marker(image, selected, color=(255, 255, 255), outline=(0, 0, 0))
        status = (
            f"Selected {x}, {y}"
            f"{self.map_value_status(selected)}"
            f" | min {float(np.nanmin(distance)):.4f}"
            f" | mean {float(np.nanmean(distance)):.4f}"
            f" | max {float(np.nanmax(distance)):.4f}"
        )
        return image, status

    def map_value_status(self, selected: tuple[int, int]) -> str:
        if self.map_rgb is None:
            return ""
        x, y = selected
        map_x, map_y = self.feature_to_map_xy(x, y)
        red, green, blue = (int(value) for value in self.map_rgb[map_y, map_x])
        return f" | map RGB {red}, {green}, {blue}"

    def draw_feature_marker(
        self,
        image: np.ndarray,
        selected: tuple[int, int] | None,
        *,
        color: tuple[int, int, int],
        outline: tuple[int, int, int],
    ) -> None:
        if selected is None:
            return
        x, y = selected
        marker_x = int(round(x / self.preview_step))
        marker_y = int(round(y / self.preview_step))
        if marker_x < 0 or marker_y < 0 or marker_x >= image.shape[1] or marker_y >= image.shape[0]:
            return

        radius = max(5, min(image.shape[:2]) // 24)
        _draw_cross(image, marker_x, marker_y, radius + 2, outline)
        _draw_cross(image, marker_x, marker_y, radius, color)

    def draw_map_marker(
        self,
        image: np.ndarray,
        selected: tuple[int, int] | None,
        *,
        color: tuple[int, int, int],
        outline: tuple[int, int, int],
    ) -> None:
        if selected is None:
            return
        marker_x, marker_y = self.feature_to_map_xy(*selected)
        radius = max(5, min(image.shape[:2]) // 24)
        _draw_cross(image, marker_x, marker_y, radius + 2, outline)
        _draw_cross(image, marker_x, marker_y, radius, color)

    def feature_to_map_xy(self, x: int, y: int) -> tuple[int, int]:
        if self.map_rgb is None:
            return int(round(x / self.preview_step)), int(round(y / self.preview_step))
        map_x = int(round((x + 0.5) * self.map_scale_x - 0.5))
        map_y = int(round((y + 0.5) * self.map_scale_y - 0.5))
        return min(max(map_x, 0), self.map_rgb.shape[1] - 1), min(max(map_y, 0), self.map_rgb.shape[0] - 1)

    def cached_cosine_distance(self, x: int, y: int) -> np.ndarray:
        if self._last_distance_xy == (x, y) and self._last_distance is not None:
            return self._last_distance
        distance = self.cosine_distance(x, y)
        self._last_distance_xy = (x, y)
        self._last_distance = distance
        return distance

    def cosine_distance(self, x: int, y: int) -> np.ndarray:
        target_index = x * self.height + y
        target = torch.nan_to_num(self.flat_features[target_index].float())
        target_norm = self.norms[target_index].clamp_min(EPS)
        cosine = torch.empty(self.flat_features.shape[0], dtype=torch.float32)

        if self.device.type == "cuda":
            target_device = target.to(self.device)
            target_norm_device = target_norm.to(self.device)
            for start in range(0, self.flat_features.shape[0], self.distance_chunk_size):
                end = min(start + self.distance_chunk_size, self.flat_features.shape[0])
                chunk = torch.nan_to_num(self.flat_features[start:end].float()).to(self.device)
                denom = self.norms[start:end].to(self.device) * target_norm_device
                cosine[start:end] = ((chunk @ target_device) / denom.clamp_min(EPS)).cpu()
        else:
            for start in range(0, self.flat_features.shape[0], self.distance_chunk_size):
                end = min(start + self.distance_chunk_size, self.flat_features.shape[0])
                chunk = torch.nan_to_num(self.flat_features[start:end].float())
                denom = self.norms[start:end] * target_norm
                cosine[start:end] = (chunk @ target) / denom.clamp_min(EPS)

        distance = 1.0 - cosine.clamp(-1.0, 1.0)
        return distance.reshape(self.width, self.height).numpy()


def create_app(
    *,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    cache_dir: str | Path = ".bretez_cache",
    max_preview_pixels: int = 1_500_000,
    pca_sample: int = 50_000,
    map_image_path: str | Path | None = None,
    include_map: bool = True,
    device: str = "auto",
    distance_chunk_size: int = 16_384,
) -> gr.Blocks:
    store = FeatureStore.load(
        features_path,
        cache_dir=cache_dir,
        max_preview_pixels=max_preview_pixels,
        pca_sample=pca_sample,
        map_image_path=map_image_path,
        include_map=include_map,
        device=device,
        distance_chunk_size=distance_chunk_size,
    )

    initial_map, initial_feature, initial_overlay, initial_status = store.render_view()

    def select_map_feature(_x_value: Any, _y_value: Any, evt: gr.SelectData):
        selected = store.point_from_map_event(evt)
        map_image, feature_image, overlay_image, status = store.render_view(selected)
        return map_image, feature_image, overlay_image, status, selected[0], selected[1]

    def select_feature(_x_value: Any, _y_value: Any, evt: gr.SelectData):
        selected = store.point_from_feature_event(evt)
        map_image, feature_image, overlay_image, status = store.render_view(selected)
        return map_image, feature_image, overlay_image, status, selected[0], selected[1]

    def measure_feature(x_value: Any, y_value: Any):
        selected = store.clamp_xy(x_value, y_value)
        map_image, feature_image, overlay_image, status = store.render_view(selected)
        return map_image, feature_image, overlay_image, status, selected[0], selected[1]

    with gr.Blocks(title="Bretez Feature Viewer", fill_width=True) as app:
        gr.Markdown("# Bretez Feature Viewer")
        status = gr.Markdown(value=initial_status)

        with gr.Row(equal_height=True):
            map_image = gr.Image(
                value=initial_map,
                label="Original map",
                type="numpy",
                interactive=False,
                format="jpeg",
                height=620,
                buttons=["download", "fullscreen"],
            )
            feature_image = gr.Image(
                value=initial_feature,
                label="Features",
                type="numpy",
                interactive=False,
                height=620,
                buttons=["download", "fullscreen"],
            )
            distance_image = gr.Image(
                value=initial_overlay,
                label="Map + cosine distance",
                type="numpy",
                interactive=False,
                format="jpeg",
                height=620,
                buttons=["download", "fullscreen"],
            )

        with gr.Row():
            x_input = gr.Number(label="Selected x", value=0, precision=0, minimum=0, maximum=store.width - 1)
            y_input = gr.Number(label="Selected y", value=0, precision=0, minimum=0, maximum=store.height - 1)
            measure = gr.Button("Measure", variant="primary")

        select_outputs = [map_image, feature_image, distance_image, status, x_input, y_input]
        map_image.select(select_map_feature, inputs=[x_input, y_input], outputs=select_outputs)
        feature_image.select(select_feature, inputs=[x_input, y_input], outputs=select_outputs)
        distance_image.select(select_map_feature, inputs=[x_input, y_input], outputs=select_outputs)

        measure.click(
            measure_feature,
            inputs=[x_input, y_input],
            outputs=select_outputs,
        )

    return app


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the Bretez feature viewer.")
    parser.add_argument("--features", default=DEFAULT_FEATURES_PATH, help="Path to a saved feature tensor.")
    parser.add_argument("--cache-dir", default=".bretez_cache", help="Directory for preview and norm caches.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for Gradio.")
    parser.add_argument("--port", default=7860, type=int, help="Port for Gradio.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share URL.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Device for distance maps.")
    parser.add_argument("--preview-pixels", default=1_500_000, type=int, help="Maximum preview pixels.")
    parser.add_argument("--pca-sample", default=50_000, type=int, help="Feature samples for PCA preview.")
    parser.add_argument("--map-image", default=None, help="Path to the original map image. Defaults to the loader image.")
    parser.add_argument("--no-map", action="store_true", help="Do not load or show the original map image.")
    parser.add_argument("--distance-chunk-size", default=16_384, type=int, help="Rows per cosine chunk.")
    args = parser.parse_args(argv)

    app = create_app(
        features_path=args.features,
        cache_dir=args.cache_dir,
        max_preview_pixels=args.preview_pixels,
        pca_sample=args.pca_sample,
        map_image_path=args.map_image,
        include_map=not args.no_map,
        device=args.device,
        distance_chunk_size=args.distance_chunk_size,
    )
    app.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


def _load_feature_tensor(path: Path) -> torch.Tensor:
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": True}
    try:
        data = torch.load(path, mmap=True, **kwargs)
    except TypeError:
        data = torch.load(path, **kwargs)

    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, dict):
        tensor = next(
            (value for key, value in data.items() if key in {"features", "feature", "data", "tensor"} and isinstance(value, torch.Tensor)),
            None,
        )
        if tensor is None:
            raise TypeError("Feature file dictionary must contain a tensor named features, feature, data, or tensor.")
    else:
        raise TypeError(f"Expected a tensor feature file, got {type(data).__name__}.")

    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    _feature_shape(tensor)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.detach().cpu()


def _feature_shape(features: torch.Tensor) -> tuple[int, int, int]:
    if features.ndim != 3:
        raise ValueError(f"Expected feature tensor shape (width, height, channels), got {tuple(features.shape)}.")
    width, height, channels = (int(dim) for dim in features.shape)
    if min(width, height, channels) <= 0:
        raise ValueError(f"Feature tensor dimensions must be positive, got {tuple(features.shape)}.")
    return width, height, channels


def _build_preview_rgb(features: torch.Tensor, step: int, pca_sample: int) -> np.ndarray:
    preview_features = features[::step, ::step, :]
    width, height, channels = _feature_shape(preview_features)
    flat = preview_features.reshape(width * height, channels)
    valid = torch.isfinite(flat).all(dim=1)
    if not bool(valid.any()):
        return np.zeros((height, width, 3), dtype=np.uint8)

    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    sample_count = min(max(int(pca_sample), 3), int(valid_indices.numel()))
    if valid_indices.numel() > sample_count:
        generator = torch.Generator(device="cpu").manual_seed(0)
        valid_indices = valid_indices[torch.randperm(valid_indices.numel(), generator=generator)[:sample_count]]

    sample = torch.nan_to_num(flat[valid_indices].float())
    sample = F.normalize(sample, dim=1, eps=EPS)
    mean = sample.mean(dim=0, keepdim=True)
    components = _pca_components(sample - mean, channels)

    projected = torch.empty((flat.shape[0], 3), dtype=torch.float32)
    chunk_size = 16_384
    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        chunk = F.normalize(torch.nan_to_num(flat[start:end].float()), dim=1, eps=EPS)
        projected[start:end] = (chunk - mean) @ components

    low = torch.quantile(projected, 0.01, dim=0)
    high = torch.quantile(projected, 0.99, dim=0)
    rgb = ((projected - low) / (high - low).clamp_min(EPS)).clamp(0, 1)
    rgb = (rgb.reshape(width, height, 3).numpy() * 255).round().astype(np.uint8)
    return np.swapaxes(rgb, 0, 1)


def _pca_components(centered_sample: torch.Tensor, channels: int) -> torch.Tensor:
    if centered_sample.shape[0] >= 3 and channels >= 3:
        try:
            _, _, components = torch.pca_lowrank(centered_sample, q=3, center=False, niter=4)
            if components.shape == (channels, 3):
                return components[:, :3]
        except RuntimeError:
            pass

    random_basis = torch.randn((channels, min(channels, 3)))
    components, _ = torch.linalg.qr(random_basis, mode="reduced")
    if components.shape[1] < 3:
        components = F.pad(components, (0, 3 - components.shape[1]))
    return components[:, :3]


def _load_map_rgb(
    map_image_path: str | Path | None,
    feature_width: int,
    feature_height: int,
) -> tuple[np.ndarray, float, float]:
    image, _image_key = _load_map_image(map_image_path)
    scale_x, scale_y = _infer_map_scale(image.size[0], image.size[1], feature_width, feature_height)
    return np.asarray(image.convert("RGB")).copy(), scale_x, scale_y


def _load_map_image(map_image_path: str | Path | None) -> tuple[Image.Image, str]:
    if map_image_path is not None:
        path = Path(map_image_path).expanduser().resolve()
        image = Image.open(path)
        return image, _path_cache_key(path)

    from bretez.loader import load_original_image

    image = load_original_image()
    filename = getattr(image, "filename", None)
    if filename:
        return image, _path_cache_key(Path(filename).expanduser().resolve())
    size_key = f"loader-image|{image.size[0]}|{image.size[1]}|{image.mode}"
    return image, hashlib.sha256(size_key.encode("utf-8")).hexdigest()[:16]


def _map_crop_box(image_size: tuple[int, int], feature_width: int, feature_height: int) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    scale_x, scale_y = _infer_map_scale(image_width, image_height, feature_width, feature_height)
    crop_width = min(image_width, max(1, int(round(feature_width * scale_x))))
    crop_height = min(image_height, max(1, int(round(feature_height * scale_y))))
    return 0, 0, crop_width, crop_height


def _infer_map_scale(image_width: int, image_height: int, feature_width: int, feature_height: int) -> tuple[float, float]:
    for scale in (MAP_PIXELS_PER_FEATURE, DINO_PATCH_SIZE):
        if _scale_matches_map(image_width, image_height, feature_width, feature_height, scale):
            return float(scale), float(scale)

    x_scale = image_width / feature_width
    y_scale = image_height / feature_height
    common_scale = math.floor(min(x_scale, y_scale))
    if common_scale > 0 and _scale_matches_map(image_width, image_height, feature_width, feature_height, common_scale):
        return float(common_scale), float(common_scale)
    return x_scale, y_scale


def _scale_matches_map(image_width: int, image_height: int, feature_width: int, feature_height: int, scale: int) -> bool:
    crop_width = feature_width * scale
    crop_height = feature_height * scale
    if crop_width > image_width or crop_height > image_height:
        return False
    return crop_width / image_width >= 0.95 and crop_height / image_height >= 0.95


def _resize_feature_values_to_map(distance: np.ndarray, map_shape: tuple[int, int], scale_x: float, scale_y: float) -> np.ndarray:
    map_height, map_width = map_shape
    feature_width, feature_height = distance.shape
    coverage_width = min(map_width, max(1, int(round(feature_width * scale_x))))
    coverage_height = min(map_height, max(1, int(round(feature_height * scale_y))))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    feature_image = Image.fromarray(np.swapaxes(distance, 0, 1).astype(np.float32), mode="F")
    coverage = np.asarray(feature_image.resize((coverage_width, coverage_height), resampling))

    if coverage.shape == (map_height, map_width):
        return coverage

    full_distance = np.full((map_height, map_width), np.nan, dtype=np.float32)
    full_distance[:coverage_height, :coverage_width] = coverage
    return full_distance


def _distance_to_rgb(distance: np.ndarray) -> np.ndarray:
    normalized = _distance_display_values(distance)
    return (colormaps[DISTANCE_COLORMAP](normalized)[..., :3] * 255).round().astype(np.uint8)


def _distance_display_values(distance: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.nan_to_num(distance, nan=DISTANCE_DISPLAY_MAX, posinf=DISTANCE_DISPLAY_MAX, neginf=0.0), 0.0, DISTANCE_DISPLAY_MAX)
    return (clipped / DISTANCE_DISPLAY_MAX) ** DISTANCE_DISPLAY_GAMMA


def _blend_distance_over_map(map_image: np.ndarray, distance: np.ndarray, alpha: float) -> np.ndarray:
    if map_image.shape[:2] != distance.shape[:2]:
        height = min(map_image.shape[0], distance.shape[0])
        width = min(map_image.shape[1], distance.shape[1])
        map_image = map_image[:height, :width]
        distance = distance[:height, :width]

    blended = np.empty_like(map_image)
    chunk_rows = 256
    for start in range(0, map_image.shape[0], chunk_rows):
        end = min(start + chunk_rows, map_image.shape[0])
        distance_chunk = distance[start:end]
        distance_rgb = _distance_to_rgb(distance_chunk)
        valid = np.isfinite(distance_chunk)[..., None]
        normalized = _distance_display_values(distance_chunk)[..., None]
        distance_alpha = DISTANCE_OVERLAY_ALPHA_MIN + ((alpha - DISTANCE_OVERLAY_ALPHA_MIN) * (1.0 - normalized))
        distance_alpha = np.where(valid, distance_alpha, 0.0)
        blended_chunk = (map_image[start:end].astype(np.float32) * (1.0 - distance_alpha)) + (
            distance_rgb.astype(np.float32) * distance_alpha
        )
        blended[start:end] = blended_chunk.clip(0, 255).round().astype(np.uint8)

    return blended


def _draw_cross(image: np.ndarray, x: int, y: int, radius: int, color: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    image[y, x0:x1] = color
    image[y0:y1, x] = color


def _event_image_xy(evt: gr.SelectData) -> tuple[int, int]:
    value = getattr(evt, "index", None)
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"Expected an image click with x/y index, got {value!r}.")
    return int(round(float(value[0]))), int(round(float(value[1])))


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device)


def _preview_step(width: int, height: int, max_pixels: int) -> int:
    if max_pixels <= 0:
        return 1
    return max(1, math.ceil(math.sqrt((width * height) / max_pixels)))


def _feature_cache_key(path: Path) -> str:
    stat = path.stat()
    payload = f"{CACHE_VERSION}|{path}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _path_cache_key(path: Path) -> str:
    stat = path.stat()
    payload = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _preview_cache_key(cache_key: str, preview_step: int, pca_sample: int) -> str:
    payload = f"{cache_key}|step={preview_step}|sample={pca_sample}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    main()