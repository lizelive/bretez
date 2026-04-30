from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageOps

from bretez.backbone import SAT_MODEL_SMALL, Backbone
from bretez.config import INPUT_IMAGE_DOWNSCALE_FACTOR
from bretez.loader import downscale_image, load_image
from bretez.state import now_iso


logger = logging.getLogger(__name__)
DEFAULT_FEATURES_PATH = Path("features.pt")


def extract_features(
    *,
    output_path: str | Path = DEFAULT_FEATURES_PATH,
    map_image_path: str | Path | None = None,
    downscale_factor: int = INPUT_IMAGE_DOWNSCALE_FACTOR,
    model_name: str = SAT_MODEL_SMALL,
    device: str = "auto",
    batch_size: int = 32,
) -> dict[str, Any]:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image = _load_extraction_image(map_image_path, downscale_factor)
    logger.info("Extracting DINO features from image size %s", image.size)
    model = Backbone(model_name=model_name, device=device, batch_size=batch_size)
    features = model.process_image(image)
    torch.save(features, output)

    metadata = {
        "output_path": str(output),
        "created_at": now_iso(),
        "model_name": model_name,
        "device": str(model.device),
        "batch_size": batch_size,
        "downscale_factor": downscale_factor,
        "source_image": str(Path(map_image_path).expanduser().resolve()) if map_image_path else "bretez.loader.load_image",
        "image_width": image.size[0],
        "image_height": image.size[1],
        "feature_width": int(features.shape[0]),
        "feature_height": int(features.shape[1]),
        "channels": int(features.shape[2]),
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def _load_extraction_image(map_image_path: str | Path | None, downscale_factor: int) -> Image.Image:
    if map_image_path is None:
        return load_image(downscale_factor)

    image_path = Path(map_image_path).expanduser()
    if not image_path.exists():
        raise FileNotFoundError(f"Map image not found: {image_path}")
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return downscale_image(image.convert("RGB"), downscale_factor)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract DINOv3 map features for Bretez.")
    parser.add_argument("--output", default=str(DEFAULT_FEATURES_PATH), help="Output feature tensor path.")
    parser.add_argument("--map-image", default=None, help="Optional local map image. Defaults to the configured Turgot sheet.")
    parser.add_argument("--downscale", default=INPUT_IMAGE_DOWNSCALE_FACTOR, type=int, help="Image downscale factor before extraction.")
    parser.add_argument("--model", default=SAT_MODEL_SMALL, help="Hugging Face DINOv3 model id.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Extraction device.")
    parser.add_argument("--batch-size", default=32, type=int, help="Tile batch size for large-image extraction.")
    args = parser.parse_args(argv)

    logging.basicConfig(filename="bretez.log", level=logging.INFO)
    result = extract_features(
        output_path=args.output,
        map_image_path=args.map_image,
        downscale_factor=args.downscale,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
