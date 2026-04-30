from __future__ import annotations

import copy
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1
DEFAULT_PROJECT_PATH = Path(".bretez") / "project.json"
EVENT_LOG_NAME = "events.jsonl"


CLASSIFICATIONS: list[dict[str, str]] = [
    {"id": "border", "name": "Border", "group": "Off-map", "color": "#9ca3af"},
    {"id": "label", "name": "Label", "group": "Off-map", "color": "#f97316"},
    {"id": "road", "name": "Road", "group": "Ground", "color": "#78716c"},
    {"id": "dirt", "name": "Dirt", "group": "Ground", "color": "#a16207"},
    {"id": "water", "name": "Water", "group": "Water", "color": "#0284c7"},
    {"id": "field", "name": "Field", "group": "Vegetation", "color": "#84cc16"},
    {"id": "garden", "name": "Garden", "group": "Vegetation", "color": "#22c55e"},
    {"id": "lawn", "name": "Lawn", "group": "Vegetation", "color": "#65a30d"},
    {"id": "tree", "name": "Tree", "group": "Vegetation", "color": "#15803d"},
    {"id": "bush", "name": "Bush", "group": "Vegetation", "color": "#16a34a"},
    {"id": "hedge", "name": "Hedge", "group": "Vegetation", "color": "#166534"},
    {"id": "building", "name": "Building", "group": "Structures", "color": "#475569"},
    {"id": "roof", "name": "Roof", "group": "Structures", "color": "#b45309"},
    {"id": "roof_gable", "name": "Roof gable", "group": "Structures", "color": "#c2410c"},
    {"id": "roof_hip", "name": "Roof hip", "group": "Structures", "color": "#dc2626"},
    {"id": "roof_mansard", "name": "Roof mansard", "group": "Structures", "color": "#be123c"},
    {"id": "roof_shed", "name": "Roof shed", "group": "Structures", "color": "#ea580c"},
    {"id": "roof_conical", "name": "Roof conical", "group": "Structures", "color": "#9333ea"},
    {"id": "roof_flat", "name": "Roof flat", "group": "Structures", "color": "#f59e0b"},
    {"id": "roof_dormer", "name": "Roof dormer", "group": "Structures", "color": "#e11d48"},
    {"id": "wall", "name": "Wall", "group": "Structures", "color": "#64748b"},
    {"id": "window", "name": "Window", "group": "Structures", "color": "#38bdf8"},
    {"id": "door", "name": "Door", "group": "Structures", "color": "#92400e"},
    {"id": "chimney", "name": "Chimney", "group": "Structures", "color": "#7f1d1d"},
    {"id": "windmill", "name": "Windmill", "group": "Structures", "color": "#6d28d9"},
    {"id": "bridge", "name": "Bridge", "group": "Structures", "color": "#0f766e"},
    {"id": "stairs", "name": "Stairs", "group": "Structures", "color": "#f43f5e"},
    {"id": "fence", "name": "Fence", "group": "Structures", "color": "#71717a"},
    {"id": "boat", "name": "Boat", "group": "Objects", "color": "#2563eb"},
    {"id": "rock", "name": "Rock", "group": "Objects", "color": "#52525b"},
    {"id": "log", "name": "Log", "group": "Objects", "color": "#854d0e"},
]


ENTITY_COLLECTIONS: dict[str, str] = {
    "annotations": "ann",
    "vertices": "v",
    "lines": "line",
    "faces": "face",
    "constraints": "con",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def project_path(path: str | Path | None = None) -> Path:
    return Path(path or DEFAULT_PROJECT_PATH).expanduser().resolve()


class ProjectStore:
    """Durable project state with snapshot history and an append-only event log."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = project_path(path)
        self.event_log_path = self.path.with_name(EVENT_LOG_NAME)
        self._lock = threading.RLock()
        self.project = self._load_or_create()

    def read(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.project)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            history = self.project.get("history", {})
            return {
                "path": str(self.path),
                "schema_version": self.project.get("schema_version"),
                "created_at": self.project.get("created_at"),
                "updated_at": self.project.get("updated_at"),
                "map": copy.deepcopy(self.project.get("map", {})),
                "counts": {name: len(self.project.get(name, [])) for name in ENTITY_COLLECTIONS},
                "undo_count": len(history.get("past", [])),
                "redo_count": len(history.get("future", [])),
                "event_log": str(self.event_log_path),
            }

    def ensure_runtime_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            current = self.project.setdefault("map", {})
            changed = False
            for key, value in metadata.items():
                if current.get(key) != value:
                    current[key] = value
                    changed = True
            if changed:
                self.project["updated_at"] = now_iso()
                self._save({"type": "runtime_metadata", "metadata": metadata})
            return copy.deepcopy(current)

    def add_entity(self, collection: str, values: dict[str, Any], label: str | None = None) -> dict[str, Any]:
        self._validate_collection(collection)
        created: dict[str, Any] = {}

        def mutate(project: dict[str, Any]) -> None:
            entity = self._prepare_entity(collection, values)
            project[collection].append(entity)
            created.update(entity)

        self._mutate(label or f"Add {collection[:-1]}", mutate)
        return copy.deepcopy(created)

    def update_entity(self, collection: str, entity_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        self._validate_collection(collection)
        updated: dict[str, Any] = {}

        def mutate(project: dict[str, Any]) -> None:
            entity = self._find_entity(project, collection, entity_id)
            for key, value in patch.items():
                if key not in {"id", "created_at"}:
                    entity[key] = value
            entity["updated_at"] = now_iso()
            updated.update(entity)

        self._mutate(f"Update {collection[:-1]}", mutate)
        return copy.deepcopy(updated)

    def delete_entity(self, collection: str, entity_id: str) -> dict[str, Any]:
        self._validate_collection(collection)
        deleted: dict[str, Any] = {}

        def mutate(project: dict[str, Any]) -> None:
            entities = project[collection]
            for index, entity in enumerate(entities):
                if entity.get("id") == entity_id:
                    deleted.update(entities.pop(index))
                    self._cleanup_references(project, collection, entity_id)
                    return
            raise KeyError(f"No {collection[:-1]} with id {entity_id!r}.")

        self._mutate(f"Delete {collection[:-1]}", mutate)
        return copy.deepcopy(deleted)

    def set_viewport(self, viewport: dict[str, Any]) -> dict[str, Any]:
        saved: dict[str, Any] = {}

        def mutate(project: dict[str, Any]) -> None:
            project["viewport"] = {
                "x": float(viewport.get("x", project.get("viewport", {}).get("x", 0.0))),
                "y": float(viewport.get("y", project.get("viewport", {}).get("y", 0.0))),
                "scale": float(viewport.get("scale", project.get("viewport", {}).get("scale", 1.0))),
                "layer": str(viewport.get("layer", project.get("viewport", {}).get("layer", "map"))),
            }
            saved.update(project["viewport"])

        self._mutate("Save viewport", mutate, include_history=False)
        return copy.deepcopy(saved)

    def set_selection(self, selection: dict[str, Any] | None) -> dict[str, Any] | None:
        saved = copy.deepcopy(selection) if selection else None

        def mutate(project: dict[str, Any]) -> None:
            project["selection"] = saved

        self._mutate("Save selection", mutate, include_history=False)
        return copy.deepcopy(saved)

    def replace_project(self, project: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            before = self._snapshot()
            replacement = self._normalize_project(copy.deepcopy(project))
            replacement["history"] = {"past": [], "future": []}
            self.project = replacement
            after = self._snapshot()
            action = self._action("Replace project", before, after)
            self.project["history"]["past"].append(action)
            self.project["updated_at"] = now_iso()
            self._save({"type": "replace_project", "action": self._action_summary(action)})
            return copy.deepcopy(self.project)

    def undo(self) -> dict[str, Any]:
        with self._lock:
            history = self.project.setdefault("history", {"past": [], "future": []})
            if not history["past"]:
                return self.read()
            action = history["past"].pop()
            history["future"].append(action)
            self._restore_snapshot(action["before"])
            self.project["updated_at"] = now_iso()
            self._save({"type": "undo", "action": self._action_summary(action)})
            return copy.deepcopy(self.project)

    def redo(self) -> dict[str, Any]:
        with self._lock:
            history = self.project.setdefault("history", {"past": [], "future": []})
            if not history["future"]:
                return self.read()
            action = history["future"].pop()
            history["past"].append(action)
            self._restore_snapshot(action["after"])
            self.project["updated_at"] = now_iso()
            self._save({"type": "redo", "action": self._action_summary(action)})
            return copy.deepcopy(self.project)

    def _load_or_create(self) -> dict[str, Any]:
        with self._lock:
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as file:
                    return self._normalize_project(json.load(file))

            project = self._default_project()
            self.project = project
            self._save({"type": "create_project"})
            return project

    def _default_project(self) -> dict[str, Any]:
        timestamp = now_iso()
        return {
            "schema_version": SCHEMA_VERSION,
            "id": new_id("project"),
            "name": "Bretez map",
            "created_at": timestamp,
            "updated_at": timestamp,
            "map": {
                "width": 4096,
                "height": 4096,
                "tile_size": 256,
                "primary_layer": "map",
                "source": None,
            },
            "classifications": copy.deepcopy(CLASSIFICATIONS),
            "annotations": [],
            "vertices": [],
            "lines": [],
            "faces": [],
            "constraints": [],
            "viewport": {"x": 0.0, "y": 0.0, "scale": 0.25, "layer": "map"},
            "selection": None,
            "history": {"past": [], "future": []},
        }

    def _normalize_project(self, project: dict[str, Any]) -> dict[str, Any]:
        normalized = self._default_project()
        normalized.update(project)
        normalized["schema_version"] = SCHEMA_VERSION
        normalized.setdefault("id", new_id("project"))
        normalized.setdefault("created_at", now_iso())
        normalized.setdefault("updated_at", normalized["created_at"])
        normalized["classifications"] = project.get("classifications") or copy.deepcopy(CLASSIFICATIONS)
        normalized["history"] = project.get("history") or {"past": [], "future": []}
        normalized["history"].setdefault("past", [])
        normalized["history"].setdefault("future", [])
        for collection in ENTITY_COLLECTIONS:
            normalized[collection] = list(project.get(collection) or [])
        return normalized

    def _prepare_entity(self, collection: str, values: dict[str, Any]) -> dict[str, Any]:
        timestamp = now_iso()
        entity = copy.deepcopy(values)
        entity.setdefault("id", new_id(ENTITY_COLLECTIONS[collection]))
        entity.setdefault("created_at", timestamp)
        entity["updated_at"] = timestamp
        if collection == "annotations":
            entity.setdefault("classification_id", "building")
            entity.setdefault("label", "")
            entity.setdefault("visible", True)
            entity.setdefault("locked", False)
            entity.setdefault("properties", {})
            for key in ("x", "y", "width", "height"):
                entity[key] = float(entity.get(key, 0.0))
        elif collection == "vertices":
            entity["u"] = float(entity.get("u", 0.0))
            entity["v"] = float(entity.get("v", 0.0))
            entity.setdefault("geocoords", {})
            entity.setdefault("altitude", None)
            entity.setdefault("properties", {})
        elif collection == "lines":
            entity["vertex_ids"] = list(entity.get("vertex_ids", entity.get("verts", [])))
            entity.setdefault("vertical", False)
            entity.setdefault("horizontal", False)
            entity.setdefault("length", None)
            entity.setdefault("properties", {})
        elif collection == "faces":
            entity["vertex_ids"] = list(entity.get("vertex_ids", entity.get("verts", [])))
            entity.setdefault("classification_id", "building")
            entity.setdefault("roof", False)
            entity.setdefault("wall", False)
            entity.setdefault("door", False)
            entity.setdefault("window", False)
            entity.setdefault("properties", {})
        elif collection == "constraints":
            entity.setdefault("constraint_type", entity.get("type", "note"))
            entity.setdefault("subject_ids", [])
            entity.setdefault("data", {})
            entity.setdefault("properties", {})
        return entity

    def _mutate(self, label: str, mutator: Callable[[dict[str, Any]], None], *, include_history: bool = True) -> dict[str, Any]:
        with self._lock:
            before = self._snapshot()
            mutator(self.project)
            self.project["updated_at"] = now_iso()
            after = self._snapshot()
            if before == after:
                return copy.deepcopy(self.project)
            event: dict[str, Any] = {"type": "mutation", "label": label}
            if include_history:
                action = self._action(label, before, after)
                history = self.project.setdefault("history", {"past": [], "future": []})
                history["past"].append(action)
                history["future"].clear()
                event["action"] = self._action_summary(action)
            self._save(event)
            return copy.deepcopy(self.project)

    def _snapshot(self) -> dict[str, Any]:
        return {key: copy.deepcopy(value) for key, value in self.project.items() if key != "history"}

    def _restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        history = self.project.setdefault("history", {"past": [], "future": []})
        self.project = self._normalize_project(copy.deepcopy(snapshot))
        self.project["history"] = history

    def _action(self, label: str, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        return {"id": new_id("act"), "timestamp": now_iso(), "label": label, "before": before, "after": after}

    def _action_summary(self, action: dict[str, Any]) -> dict[str, Any]:
        return {"id": action["id"], "timestamp": action["timestamp"], "label": action["label"]}

    def _save(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self.project, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary_path, self.path)

        event_record = {"timestamp": now_iso(), **event}
        with self.event_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event_record, sort_keys=True) + "\n")

    def _validate_collection(self, collection: str) -> None:
        if collection not in ENTITY_COLLECTIONS:
            valid = ", ".join(sorted(ENTITY_COLLECTIONS))
            raise KeyError(f"Unknown collection {collection!r}. Expected one of: {valid}.")

    def _find_entity(self, project: dict[str, Any], collection: str, entity_id: str) -> dict[str, Any]:
        for entity in project[collection]:
            if entity.get("id") == entity_id:
                return entity
        raise KeyError(f"No {collection[:-1]} with id {entity_id!r}.")

    def _cleanup_references(self, project: dict[str, Any], collection: str, entity_id: str) -> None:
        if collection == "vertices":
            project["lines"] = [line for line in project["lines"] if entity_id not in line.get("vertex_ids", [])]
            project["faces"] = [face for face in project["faces"] if entity_id not in face.get("vertex_ids", [])]
        elif collection == "lines":
            for constraint in project["constraints"]:
                constraint["subject_ids"] = [subject for subject in constraint.get("subject_ids", []) if subject != entity_id]

        project["constraints"] = [
            constraint for constraint in project["constraints"] if entity_id not in constraint.get("subject_ids", [])
        ]
