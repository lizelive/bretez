# Bretez

Bretez digitizes old maps with DINOv3 features, a saved mapping workspace, a
tiled web UI, and an MCP server for AI agents.

## Feature extraction

```bash
bretez-extract --output features.pt
```

This downloads the configured map sheet, extracts DINOv3 features, and saves
`features.pt`. The older `bretez` command is kept as an alias for feature
extraction.

## Mapping server

```bash
bretez-map --features features.pt --project .bretez/project.json
```

Open `http://127.0.0.1:8000` for the mapping UI. By default the server loads the
configured Turgot sheet as the map layer. Add `--map-image path/to/map.jpg` to
use a local raster, or `--no-default-map` for a lightweight feature-only or blank
session.

The server saves every edit immediately:

- `.bretez/project.json` stores the complete current project, including
	annotations, vertices, lines, faces, constraints, viewport, selection, and
	undo/redo history.
- `.bretez/events.jsonl` stores an append-only journal of project creation,
	server metadata updates, edits, undo, redo, and viewport saves.

## Web UI

The UI supports tiled zoom and pan, classification rectangles, vertices, lines,
faces, selection, deletion, property editing, undo/redo, and real-time saves.
The Model controls can start feature extraction, train a classifier, and refresh
available tile layers without leaving the browser.

## Classifier training

Draw rectangle annotations for at least two classes, then train from the saved
project and DINO features:

```bash
bretez-train-classifier --project .bretez/project.json --features features.pt
```

Training saves `.bretez/classifier.pt` and renders `.bretez/classifier.png`,
which the mapping server exposes as a `Classifier prediction` tile layer.

## MCP server

```bash
bretez-mcp --project .bretez/project.json
```

The MCP server exposes tools for agents to inspect the project, list
classifications, add or update annotations, add vertices, lines, faces, add
constraints, save viewport state, extract features, train the classifier, and
undo or redo edits. It uses the same project file and event log as the web
server.