from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from mcp.server.fastmcp import FastMCP

from bretez.classifier import DEFAULT_CLASSIFIER_PATH, DEFAULT_PREDICTION_IMAGE_PATH, train_classifier as run_train_classifier
from bretez.config import INPUT_IMAGE_DOWNSCALE_FACTOR
from bretez.extractor import DEFAULT_FEATURES_PATH, extract_features as run_extract_features
from bretez.state import DEFAULT_PROJECT_PATH, ProjectStore


def build_mcp(project_path: str | Path = DEFAULT_PROJECT_PATH) -> FastMCP:
    store = ProjectStore(project_path)
    mcp = FastMCP("bretez")

    @mcp.tool()
    def project_summary() -> dict[str, Any]:
        """Return project paths, map metadata, entity counts, and undo/redo counts."""
        return store.summary()

    @mcp.tool()
    def get_project() -> dict[str, Any]:
        """Return the complete saved Bretez project document."""
        return store.read()

    @mcp.tool()
    def list_classifications() -> list[dict[str, Any]]:
        """List the available flat pixel classification labels and colors."""
        return store.read().get("classifications", [])

    @mcp.tool()
    def list_entities(collection: str) -> list[dict[str, Any]]:
        """List entities from annotations, vertices, lines, faces, or constraints."""
        project = store.read()
        if collection not in {"annotations", "vertices", "lines", "faces", "constraints"}:
            raise ValueError("collection must be annotations, vertices, lines, faces, or constraints")
        return project[collection]

    @mcp.tool()
    def add_rectangle(
        classification_id: str,
        x: float,
        y: float,
        width: float,
        height: float,
        label: str = "",
    ) -> dict[str, Any]:
        """Add a classified rectangular map annotation and save it immediately."""
        return store.add_entity(
            "annotations",
            {"classification_id": classification_id, "x": x, "y": y, "width": width, "height": height, "label": label},
        )

    @mcp.tool()
    def add_vertex(
        u: float,
        v: float,
        northing: float | None = None,
        easting: float | None = None,
        accuracy: float | None = None,
        altitude: float | None = None,
    ) -> dict[str, Any]:
        """Add a vertex with optional geocoordinates and altitude."""
        geocoords = {"northing": northing, "easting": easting, "accuracy": accuracy}
        geocoords = {key: value for key, value in geocoords.items() if value is not None}
        return store.add_entity("vertices", {"u": u, "v": v, "geocoords": geocoords, "altitude": altitude})

    @mcp.tool()
    def add_line(vertex_ids: list[str], vertical: bool = False, horizontal: bool = False, length: float | None = None) -> dict[str, Any]:
        """Add a line from existing vertex ids."""
        if len(vertex_ids) < 2:
            raise ValueError("A line needs at least two vertex ids.")
        return store.add_entity("lines", {"vertex_ids": vertex_ids, "vertical": vertical, "horizontal": horizontal, "length": length})

    @mcp.tool()
    def add_face(
        vertex_ids: list[str],
        classification_id: str = "building",
        roof: bool = False,
        wall: bool = False,
        door: bool = False,
        window: bool = False,
    ) -> dict[str, Any]:
        """Add a polygonal face from existing vertex ids."""
        if len(vertex_ids) < 3:
            raise ValueError("A face needs at least three vertex ids.")
        return store.add_entity(
            "faces",
            {
                "vertex_ids": vertex_ids,
                "classification_id": classification_id,
                "roof": roof,
                "wall": wall,
                "door": door,
                "window": window,
            },
        )

    @mcp.tool()
    def add_constraint(constraint_type: str, subject_ids: list[str], data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Add a saved geometric or semantic constraint for vertices, lines, or faces."""
        return store.add_entity("constraints", {"constraint_type": constraint_type, "subject_ids": subject_ids, "data": data or {}})

    @mcp.tool()
    def update_entity(collection: str, entity_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Patch an entity in annotations, vertices, lines, faces, or constraints."""
        return store.update_entity(collection, entity_id, patch)

    @mcp.tool()
    def delete_entity(collection: str, entity_id: str) -> dict[str, Any]:
        """Delete an entity and save the project immediately."""
        return store.delete_entity(collection, entity_id)

    @mcp.tool()
    def save_viewport(x: float, y: float, scale: float, layer: str = "map") -> dict[str, Any]:
        """Save the current web map viewport."""
        return store.set_viewport({"x": x, "y": y, "scale": scale, "layer": layer})

    @mcp.tool()
    def undo() -> dict[str, Any]:
        """Undo the last saved editing action."""
        return store.undo()

    @mcp.tool()
    def redo() -> dict[str, Any]:
        """Redo the last undone editing action."""
        return store.redo()

    @mcp.tool()
    def extract_features(
        output_path: str = str(DEFAULT_FEATURES_PATH),
        map_image_path: str | None = None,
        downscale_factor: int = INPUT_IMAGE_DOWNSCALE_FACTOR,
        device: str = "auto",
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Extract DINOv3 features for the default or provided map image and save them."""
        return run_extract_features(
            output_path=output_path,
            map_image_path=map_image_path,
            downscale_factor=downscale_factor,
            device=device,
            batch_size=batch_size,
        )

    @mcp.tool()
    def train_classifier(
        features_path: str = str(DEFAULT_FEATURES_PATH),
        output_path: str = str(DEFAULT_CLASSIFIER_PATH),
        prediction_image_path: str = str(DEFAULT_PREDICTION_IMAGE_PATH),
        epochs: int = 80,
        learning_rate: float = 0.03,
        device: str = "auto",
    ) -> dict[str, Any]:
        """Train a pixel classifier from saved rectangle annotations and render a prediction layer."""
        return run_train_classifier(
            project_path=store.path,
            features_path=features_path,
            output_path=output_path,
            prediction_image_path=prediction_image_path,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )

    return mcp


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Bretez MCP server over stdio.")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT_PATH), help="Project JSON path. Defaults to .bretez/project.json.")
    args = parser.parse_args(argv)
    build_mcp(args.project).run()


if __name__ == "__main__":
    main()
