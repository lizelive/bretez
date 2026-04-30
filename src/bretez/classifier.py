from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from bretez.app import _load_feature_tensor
from bretez.state import DEFAULT_PROJECT_PATH, ProjectStore, now_iso


DEFAULT_CLASSIFIER_PATH = Path(".bretez") / "classifier.pt"
DEFAULT_PREDICTION_IMAGE_PATH = Path(".bretez") / "classifier.png"


def train_classifier(
    *,
    project_path: str | Path = DEFAULT_PROJECT_PATH,
    features_path: str | Path = "features.pt",
    output_path: str | Path = DEFAULT_CLASSIFIER_PATH,
    prediction_image_path: str | Path = DEFAULT_PREDICTION_IMAGE_PATH,
    epochs: int = 80,
    learning_rate: float = 0.03,
    max_samples_per_class: int = 20_000,
    max_samples_per_annotation: int = 4_000,
    validation_fraction: float = 0.15,
    seed: int = 0,
    device: str = "auto",
) -> dict[str, Any]:
    project = ProjectStore(project_path).read()
    feature_file = Path(features_path).expanduser().resolve()
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature tensor not found: {feature_file}")

    features = _load_feature_tensor(feature_file)
    feature_width, feature_height, channels = (int(value) for value in features.shape)
    class_ids = [item["id"] for item in project.get("classifications", [])]
    colors = {item["id"]: item.get("color", "#0f766e") for item in project.get("classifications", [])}
    label_to_index = {label: index for index, label in enumerate(class_ids)}
    samples, labels, label_counts = _annotation_samples(
        project=project,
        features=features,
        label_to_index=label_to_index,
        max_samples_per_class=max_samples_per_class,
        max_samples_per_annotation=max_samples_per_annotation,
        seed=seed,
    )
    used_class_indices = sorted(set(int(value) for value in labels.tolist()))
    if len(used_class_indices) < 2:
        raise ValueError("Training needs annotations from at least two different classes.")

    train_device = _select_device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(samples.shape[0], generator=generator)
    validation_count = max(1, int(round(samples.shape[0] * validation_fraction))) if samples.shape[0] >= 10 else 0
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]
    if train_indices.numel() == 0:
        train_indices = permutation
        validation_indices = torch.empty(0, dtype=torch.long)

    mean = samples[train_indices].mean(dim=0)
    std = samples[train_indices].std(dim=0).clamp_min(1e-6)
    train_x = ((samples[train_indices] - mean) / std).to(train_device)
    train_y = labels[train_indices].to(train_device)
    model = torch.nn.Linear(channels, len(class_ids)).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    for _epoch in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    metrics = _metrics(model, samples, labels, train_indices, validation_indices, mean, std, train_device)
    classifier = {
        "schema_version": 1,
        "created_at": now_iso(),
        "features_path": str(feature_file),
        "feature_shape": [feature_width, feature_height, channels],
        "world_width": int(project.get("map", {}).get("width") or feature_width),
        "world_height": int(project.get("map", {}).get("height") or feature_height),
        "class_ids": class_ids,
        "colors": colors,
        "mean": mean.cpu(),
        "std": std.cpu(),
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "label_counts": label_counts,
        "metrics": metrics,
    }

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier, output)
    prediction_path = Path(prediction_image_path).expanduser().resolve()
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    render_prediction_image(features, classifier, prediction_path, device=device)

    metadata = {
        "classifier_path": str(output),
        "prediction_image_path": str(prediction_path),
        "features_path": str(feature_file),
        "classes": [class_ids[index] for index in used_class_indices],
        "label_counts": label_counts,
        "metrics": metrics,
        "trained_at": classifier["created_at"],
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def render_prediction_image(
    features: torch.Tensor,
    classifier: dict[str, Any],
    prediction_image_path: str | Path,
    *,
    device: str = "auto",
    chunk_size: int = 16_384,
) -> None:
    train_device = _select_device(device)
    feature_width, feature_height, channels = (int(value) for value in features.shape)
    model = torch.nn.Linear(channels, len(classifier["class_ids"])).to(train_device)
    model.load_state_dict(classifier["state_dict"])
    model.eval()
    mean = classifier["mean"].to(train_device)
    std = classifier["std"].to(train_device)
    flat = features.reshape(feature_width * feature_height, channels).float()
    predictions = torch.empty(flat.shape[0], dtype=torch.long)
    with torch.inference_mode():
        for start in range(0, flat.shape[0], chunk_size):
            end = min(start + chunk_size, flat.shape[0])
            batch = torch.nan_to_num(flat[start:end]).to(train_device)
            logits = model((batch - mean) / std)
            predictions[start:end] = logits.argmax(dim=1).cpu()

    color_table = np.array([_hex_to_rgb(classifier["colors"].get(class_id, "#0f766e")) for class_id in classifier["class_ids"]], dtype=np.uint8)
    rgb = color_table[predictions.reshape(feature_width, feature_height).numpy()]
    Image.fromarray(np.swapaxes(rgb, 0, 1), mode="RGB").save(prediction_image_path)


def load_prediction_layer(path: str | Path) -> Image.Image | None:
    prediction_path = Path(path).expanduser()
    if not prediction_path.exists():
        return None
    return Image.open(prediction_path).convert("RGB")


def _annotation_samples(
    *,
    project: dict[str, Any],
    features: torch.Tensor,
    label_to_index: dict[str, int],
    max_samples_per_class: int,
    max_samples_per_annotation: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    feature_width, feature_height, _channels = (int(value) for value in features.shape)
    map_info = project.get("map", {})
    world_width = float(map_info.get("width") or feature_width)
    world_height = float(map_info.get("height") or feature_height)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    per_class: dict[str, list[torch.Tensor]] = {}
    per_class_labels: dict[str, list[torch.Tensor]] = {}

    for annotation in project.get("annotations", []):
        if annotation.get("visible") is False:
            continue
        class_id = annotation.get("classification_id")
        if class_id not in label_to_index:
            continue
        box = _annotation_feature_box(annotation, world_width, world_height, feature_width, feature_height)
        x0, y0, x1, y1 = box
        area = (x1 - x0) * (y1 - y0)
        if area <= 0:
            continue
        sample_count = min(area, max(1, int(max_samples_per_annotation)))
        xs = torch.randint(x0, x1, (sample_count,), generator=generator)
        ys = torch.randint(y0, y1, (sample_count,), generator=generator)
        class_samples = torch.nan_to_num(features[xs, ys].float())
        per_class.setdefault(class_id, []).append(class_samples)
        per_class_labels.setdefault(class_id, []).append(torch.full((sample_count,), label_to_index[class_id], dtype=torch.long))

    sampled_features: list[torch.Tensor] = []
    sampled_labels: list[torch.Tensor] = []
    label_counts: dict[str, int] = {}
    for class_id, chunks in per_class.items():
        class_features = torch.cat(chunks, dim=0)
        class_labels = torch.cat(per_class_labels[class_id], dim=0)
        if class_features.shape[0] > max_samples_per_class:
            indices = torch.randperm(class_features.shape[0], generator=generator)[:max_samples_per_class]
            class_features = class_features[indices]
            class_labels = class_labels[indices]
        label_counts[class_id] = int(class_features.shape[0])
        sampled_features.append(class_features)
        sampled_labels.append(class_labels)

    if not sampled_features:
        raise ValueError("No classified rectangle annotations overlap the feature grid.")
    return torch.cat(sampled_features, dim=0), torch.cat(sampled_labels, dim=0), label_counts


def _annotation_feature_box(
    annotation: dict[str, Any],
    world_width: float,
    world_height: float,
    feature_width: int,
    feature_height: int,
) -> tuple[int, int, int, int]:
    left = float(annotation.get("x", 0.0))
    top = float(annotation.get("y", 0.0))
    right = left + float(annotation.get("width", 0.0))
    bottom = top + float(annotation.get("height", 0.0))
    x0 = max(0, min(feature_width, math.floor((left / world_width) * feature_width)))
    y0 = max(0, min(feature_height, math.floor((top / world_height) * feature_height)))
    x1 = max(0, min(feature_width, math.ceil((right / world_width) * feature_width)))
    y1 = max(0, min(feature_height, math.ceil((bottom / world_height) * feature_height)))
    return x0, y0, x1, y1


def _metrics(
    model: torch.nn.Linear,
    samples: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, float | None]:
    with torch.inference_mode():
        train_accuracy = _accuracy(model, samples, labels, train_indices, mean, std, device)
        validation_accuracy = _accuracy(model, samples, labels, validation_indices, mean, std, device) if validation_indices.numel() else None
    return {"train_accuracy": train_accuracy, "validation_accuracy": validation_accuracy}


def _accuracy(
    model: torch.nn.Linear,
    samples: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> float:
    logits = model(((samples[indices] - mean) / std).to(device))
    predictions = logits.argmax(dim=1).cpu()
    return float((predictions == labels[indices]).float().mean().item())


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.strip().lstrip("#")
    if len(value) != 6:
        return 15, 118, 110
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a Bretez pixel classifier from saved annotations.")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT_PATH), help="Bretez project JSON path.")
    parser.add_argument("--features", default="features.pt", help="Feature tensor path.")
    parser.add_argument("--output", default=str(DEFAULT_CLASSIFIER_PATH), help="Output classifier path.")
    parser.add_argument("--prediction-image", default=str(DEFAULT_PREDICTION_IMAGE_PATH), help="Output prediction PNG path.")
    parser.add_argument("--epochs", default=80, type=int, help="Training epochs.")
    parser.add_argument("--learning-rate", default=0.03, type=float, help="Optimizer learning rate.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Training device.")
    args = parser.parse_args(argv)
    result = train_classifier(
        project_path=args.project,
        features_path=args.features,
        output_path=args.output,
        prediction_image_path=args.prediction_image,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
