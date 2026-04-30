from __future__ import annotations

import argparse
import io
import math
import threading
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from bretez.app import FeatureStore, _distance_to_rgb
from bretez.classifier import DEFAULT_CLASSIFIER_PATH, DEFAULT_PREDICTION_IMAGE_PATH, load_prediction_layer, train_classifier
from bretez.config import INPUT_IMAGE_DOWNSCALE_FACTOR
from bretez.extractor import DEFAULT_FEATURES_PATH, extract_features
from bretez.jobs import JobManager
from bretez.state import DEFAULT_PROJECT_PATH, ProjectStore


TILE_SIZE = 256


class EntityPayload(BaseModel):
    model_config = ConfigDict(extra="allow")


class ViewportPayload(BaseModel):
    x: float
    y: float
    scale: float = Field(gt=0)
    layer: str = "map"


class SelectionPayload(BaseModel):
    kind: str | None = None
    id: str | None = None


class FeatureExtractionPayload(BaseModel):
    output_path: str = str(DEFAULT_FEATURES_PATH)
    map_image_path: str | None = None
    downscale_factor: int = INPUT_IMAGE_DOWNSCALE_FACTOR
    model_name: str | None = None
    device: str = "auto"
    batch_size: int = 32


class ClassifierTrainingPayload(BaseModel):
    features_path: str | None = None
    output_path: str = str(DEFAULT_CLASSIFIER_PATH)
    prediction_image_path: str = str(DEFAULT_PREDICTION_IMAGE_PATH)
    epochs: int = 80
    learning_rate: float = 0.03
    max_samples_per_class: int = 20_000
    max_samples_per_annotation: int = 4_000
    device: str = "auto"


@dataclass(slots=True)
class RasterLayer:
    id: str
    name: str
    image: Image.Image
    world_width: int
    world_height: int
    tile_size: int = TILE_SIZE
    mime_type: str = "image/jpeg"

    @property
    def max_zoom(self) -> int:
        return max(0, math.ceil(math.log2(max(self.world_width, self.world_height) / self.tile_size)))

    @property
    def format(self) -> str:
        return "PNG" if self.mime_type == "image/png" else "JPEG"

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "width": self.world_width,
            "height": self.world_height,
            "tileSize": self.tile_size,
            "maxZoom": self.max_zoom,
            "mimeType": self.mime_type,
        }

    def tile(self, z: int, x: int, y: int) -> bytes:
        z = min(max(int(z), 0), self.max_zoom)
        downsample = 2 ** (self.max_zoom - z)
        world_tile_size = self.tile_size * downsample
        left = x * world_tile_size
        upper = y * world_tile_size
        right = left + world_tile_size
        lower = upper + world_tile_size

        image_box = self._world_box_to_image_box((left, upper, right, lower))
        crop = self.image.crop(image_box)
        if crop.size != (self.tile_size, self.tile_size):
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            crop = crop.resize((self.tile_size, self.tile_size), resampling)
        if self.format == "JPEG" and crop.mode != "RGB":
            crop = crop.convert("RGB")
        buffer = io.BytesIO()
        save_kwargs: dict[str, Any] = {"format": self.format}
        if self.format == "JPEG":
            save_kwargs.update({"quality": 88, "optimize": True})
        crop.save(buffer, **save_kwargs)
        return buffer.getvalue()

    def _world_box_to_image_box(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        left, upper, right, lower = box
        image_width, image_height = self.image.size
        scale_x = image_width / max(1, self.world_width)
        scale_y = image_height / max(1, self.world_height)
        return (
            int(math.floor(left * scale_x)),
            int(math.floor(upper * scale_y)),
            int(math.ceil(right * scale_x)),
            int(math.ceil(lower * scale_y)),
        )


class MappingAssets:
    def __init__(self, layers: dict[str, RasterLayer], feature_store: FeatureStore | None = None) -> None:
        self.layers = layers
        self.feature_store = feature_store

    @property
    def primary_layer_id(self) -> str:
        if "map" in self.layers:
            return "map"
        if "features" in self.layers:
            return "features"
        return "blank"

    @property
    def primary_layer(self) -> RasterLayer:
        return self.layers[self.primary_layer_id]

    def metadata(self) -> dict[str, Any]:
        primary = self.primary_layer
        return {
            "tileSize": primary.tile_size,
            "primaryLayer": self.primary_layer_id,
            "width": primary.world_width,
            "height": primary.world_height,
            "layers": [layer.metadata() for layer in self.layers.values()],
            "computed": {"distanceTiles": self.feature_store is not None},
        }


def create_app(
    *,
    project_path: str | Path | None = DEFAULT_PROJECT_PATH,
    features_path: str | Path | None = "features.pt",
    map_image_path: str | Path | None = None,
    cache_dir: str | Path = ".bretez_cache",
    device: str = "auto",
    tile_size: int = TILE_SIZE,
    load_default_map: bool = True,
    classifier_image_path: str | Path = DEFAULT_PREDICTION_IMAGE_PATH,
) -> FastAPI:
    store = ProjectStore(project_path)
    jobs = JobManager()
    assets_lock = threading.RLock()
    paths: dict[str, str | Path | None] = {"features": features_path, "map_image": map_image_path, "classifier_image": classifier_image_path}
    assets_ref = {
        "assets": load_assets(
            features_path=paths["features"],
            map_image_path=paths["map_image"],
            cache_dir=cache_dir,
            device=device,
            tile_size=tile_size,
            load_default_map=load_default_map,
            classifier_image_path=paths["classifier_image"],
        )
    }

    def current_assets() -> MappingAssets:
        with assets_lock:
            return assets_ref["assets"]

    def reload_assets(*, features: str | Path | None | object = None, classifier_image: str | Path | None | object = None) -> dict[str, Any]:
        if features is not None:
            paths["features"] = features  # type: ignore[assignment]
        if classifier_image is not None:
            paths["classifier_image"] = classifier_image  # type: ignore[assignment]
        loaded = load_assets(
            features_path=paths["features"],
            map_image_path=paths["map_image"],
            cache_dir=cache_dir,
            device=device,
            tile_size=tile_size,
            load_default_map=load_default_map,
            classifier_image_path=paths["classifier_image"],
        )
        with assets_lock:
            assets_ref["assets"] = loaded
        metadata = loaded.metadata()
        store.ensure_runtime_metadata(
            {
                "width": metadata["width"],
                "height": metadata["height"],
                "tile_size": metadata["tileSize"],
                "primary_layer": metadata["primaryLayer"],
                "layers": metadata["layers"],
            }
        )
        return metadata

    reload_assets()

    app = FastAPI(title="Bretez Mapping Server")
    static_path = files("bretez") / "static"
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(static_path / "index.html"))

    @app.get("/api/assets")
    def get_assets() -> dict[str, Any]:
        return current_assets().metadata()

    @app.get("/api/project")
    def get_project() -> dict[str, Any]:
        return store.read()

    @app.put("/api/project")
    def put_project(project: dict[str, Any]) -> dict[str, Any]:
        return store.replace_project(project)

    @app.get("/api/summary")
    def get_summary() -> dict[str, Any]:
        return store.summary()

    @app.post("/api/viewport")
    def save_viewport(payload: ViewportPayload) -> dict[str, Any]:
        viewport = store.set_viewport(payload.model_dump())
        return {"project": store.read(), "viewport": viewport}

    @app.post("/api/selection")
    def save_selection(payload: SelectionPayload) -> dict[str, Any]:
        selection = payload.model_dump() if payload.kind and payload.id else None
        return {"project": store.read(), "selection": store.set_selection(selection)}

    @app.post("/api/undo")
    def undo() -> dict[str, Any]:
        return {"project": store.undo()}

    @app.post("/api/redo")
    def redo() -> dict[str, Any]:
        return {"project": store.redo()}

    @app.get("/api/jobs")
    def list_jobs() -> dict[str, Any]:
        return {"jobs": jobs.list()}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> JSONResponse:
        return _json_response(lambda: jobs.get(job_id))

    @app.post("/api/extract-features")
    def start_feature_extraction(payload: FeatureExtractionPayload) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            kwargs = payload.model_dump()
            if not kwargs.get("model_name"):
                kwargs.pop("model_name")
            result = extract_features(**kwargs)
            result["assets"] = reload_assets(features=result["output_path"])
            return result

        return {"job": jobs.start("extract_features", run)}

    @app.post("/api/train-classifier")
    def start_classifier_training(payload: ClassifierTrainingPayload) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            features_for_training = payload.features_path or str(paths["features"] or DEFAULT_FEATURES_PATH)
            result = train_classifier(
                project_path=project_path or DEFAULT_PROJECT_PATH,
                features_path=features_for_training,
                output_path=payload.output_path,
                prediction_image_path=payload.prediction_image_path,
                epochs=payload.epochs,
                learning_rate=payload.learning_rate,
                max_samples_per_class=payload.max_samples_per_class,
                max_samples_per_annotation=payload.max_samples_per_annotation,
                device=payload.device,
            )
            result["assets"] = reload_assets(features=features_for_training, classifier_image=result["prediction_image_path"])
            return result

        return {"job": jobs.start("train_classifier", run)}

    @app.post("/api/{collection}")
    def add_entity(collection: str, payload: EntityPayload) -> JSONResponse:
        def create() -> dict[str, Any]:
            entity = store.add_entity(collection, payload.model_dump())
            return {"project": store.read(), "entity": entity}

        return _json_response(create)

    @app.patch("/api/{collection}/{entity_id}")
    def update_entity(collection: str, entity_id: str, payload: EntityPayload) -> JSONResponse:
        def update() -> dict[str, Any]:
            entity = store.update_entity(collection, entity_id, payload.model_dump())
            return {"project": store.read(), "entity": entity}

        return _json_response(update)

    @app.delete("/api/{collection}/{entity_id}")
    def delete_entity(collection: str, entity_id: str) -> JSONResponse:
        def delete() -> dict[str, Any]:
            entity = store.delete_entity(collection, entity_id)
            return {"project": store.read(), "entity": entity}

        return _json_response(delete)

    @app.get("/api/tiles/{layer_id}/{z}/{x}/{y}")
    def tile(layer_id: str, z: int, x: int, y: int) -> Response:
        layer = current_assets().layers.get(layer_id)
        if layer is None:
            raise HTTPException(status_code=404, detail=f"Unknown layer {layer_id!r}.")
        return Response(layer.tile(z, x, y), media_type=layer.mime_type)

    @app.get("/api/tiles/distance/{world_x}/{world_y}/{z}/{x}/{y}")
    def distance_tile(world_x: float, world_y: float, z: int, x: int, y: int) -> Response:
        assets = current_assets()
        if assets.feature_store is None:
            raise HTTPException(status_code=404, detail="No feature tensor is loaded.")
        layer = distance_layer(assets, world_x, world_y, tile_size)
        return Response(layer.tile(z, x, y), media_type=layer.mime_type)

    return app


def load_assets(
    *,
    features_path: str | Path | None,
    map_image_path: str | Path | None,
    cache_dir: str | Path,
    device: str,
    tile_size: int,
    load_default_map: bool,
    classifier_image_path: str | Path | None,
) -> MappingAssets:
    feature_store: FeatureStore | None = None
    layers: dict[str, RasterLayer] = {}

    if features_path is not None and Path(features_path).expanduser().exists():
        feature_store = FeatureStore.load(features_path, cache_dir=cache_dir, include_map=False, device=device)

    map_image = load_map_image(map_image_path, load_default_map=load_default_map)
    if map_image is not None:
        layers["map"] = RasterLayer("map", "Map", map_image, map_image.width, map_image.height, tile_size=tile_size)

    if feature_store is not None:
        feature_image = Image.fromarray(feature_store.preview_rgb, mode="RGB")
        world_width = layers["map"].world_width if "map" in layers else feature_store.preview_width
        world_height = layers["map"].world_height if "map" in layers else feature_store.preview_height
        layers["features"] = RasterLayer(
            "features",
            "DINO features",
            feature_image,
            world_width,
            world_height,
            tile_size=tile_size,
            mime_type="image/png",
        )

    classifier_image = load_prediction_layer(classifier_image_path) if classifier_image_path else None
    if classifier_image is not None:
        world_width = layers["map"].world_width if "map" in layers else classifier_image.width
        world_height = layers["map"].world_height if "map" in layers else classifier_image.height
        layers["classifier"] = RasterLayer(
            "classifier",
            "Classifier prediction",
            classifier_image,
            world_width,
            world_height,
            tile_size=tile_size,
            mime_type="image/png",
        )

    if not layers:
        blank = Image.new("RGB", (4096, 4096), "#f3f4f6")
        layers["blank"] = RasterLayer("blank", "Blank map", blank, blank.width, blank.height, tile_size=tile_size)

    return MappingAssets(layers, feature_store)


def load_map_image(map_image_path: str | Path | None, *, load_default_map: bool) -> Image.Image | None:
    if map_image_path is None:
        if not load_default_map:
            return None
        from bretez.loader import load_original_image

        return load_original_image().convert("RGB")
    path = Path(map_image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Map image not found: {path}")
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def distance_layer(assets: MappingAssets, world_x: float, world_y: float, tile_size: int) -> RasterLayer:
    store = assets.feature_store
    if store is None:
        raise HTTPException(status_code=404, detail="No feature tensor is loaded.")

    primary = assets.primary_layer
    feature_x = int(round((float(world_x) / max(1, primary.world_width)) * (store.width - 1)))
    feature_y = int(round((float(world_y) / max(1, primary.world_height)) * (store.height - 1)))
    feature_x, feature_y = store.clamp_xy(feature_x, feature_y)
    distance = store.cached_cosine_distance(feature_x, feature_y)
    rgb = _distance_to_rgb(distance)
    image = Image.fromarray(np.swapaxes(rgb, 0, 1), mode="RGB")
    return RasterLayer("distance", "Feature distance", image, primary.world_width, primary.world_height, tile_size=tile_size, mime_type="image/png")


def _json_response(factory: Any) -> JSONResponse:
    try:
        result = factory()
        return JSONResponse(result)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the Bretez mapping server.")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT_PATH), help="Project JSON path. Defaults to .bretez/project.json.")
    parser.add_argument("--features", default="features.pt", help="Feature tensor path. Use an empty value to skip features.")
    parser.add_argument("--map-image", default=None, help="Map image path for tiled viewing.")
    parser.add_argument("--no-default-map", action="store_true", help="Do not load the default Turgot sheet when --map-image is omitted.")
    parser.add_argument("--cache-dir", default=".bretez_cache", help="Directory for feature preview caches.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the HTTP server.")
    parser.add_argument("--port", default=8000, type=int, help="HTTP port.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Device for computed feature overlays.")
    parser.add_argument("--tile-size", default=TILE_SIZE, type=int, help="Tile edge size in pixels.")
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app(
        project_path=args.project,
        features_path=args.features or None,
        map_image_path=args.map_image,
        cache_dir=args.cache_dir,
        device=args.device,
        tile_size=args.tile_size,
        load_default_map=not args.no_default_map,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
