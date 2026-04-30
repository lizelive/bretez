const canvas = document.querySelector("#mapCanvas");
const ctx = canvas.getContext("2d");

const app = {
  project: null,
  assets: null,
  layer: "map",
  tool: "select",
  classification: "building",
  view: { x: 0, y: 0, scale: 0.25 },
  dpr: window.devicePixelRatio || 1,
  cache: new Map(),
  selected: null,
  drag: null,
  pendingLine: null,
  pendingFace: [],
  lastPointer: { x: 0, y: 0 },
  viewportTimer: null,
};

const elements = {
  layerSelect: document.querySelector("#layerSelect"),
  classSelect: document.querySelector("#classSelect"),
  saveState: document.querySelector("#saveState"),
  stats: document.querySelector("#stats"),
  coordinateReadout: document.querySelector("#coordinateReadout"),
  zoomReadout: document.querySelector("#zoomReadout"),
  selectionReadout: document.querySelector("#selectionReadout"),
  propertiesForm: document.querySelector("#propertiesForm"),
  propertiesSave: document.querySelector("#propertiesSave"),
  historyList: document.querySelector("#historyList"),
  undoButton: document.querySelector("#undoButton"),
  redoButton: document.querySelector("#redoButton"),
  deleteButton: document.querySelector("#deleteButton"),
  fitButton: document.querySelector("#fitButton"),
  extractButton: document.querySelector("#extractButton"),
  trainButton: document.querySelector("#trainButton"),
  refreshAssetsButton: document.querySelector("#refreshAssetsButton"),
  jobStatus: document.querySelector("#jobStatus"),
};

init();

async function init() {
  setSaveState("Loading");
  const [assets, project] = await Promise.all([api("/api/assets"), api("/api/project")]);
  app.assets = assets;
  app.project = project;
  app.layer = project.viewport?.layer || assets.primaryLayer;
  app.view = {
    x: Number(project.viewport?.x || 0),
    y: Number(project.viewport?.y || 0),
    scale: Number(project.viewport?.scale || 0.25),
  };
  app.selected = project.selection;
  setupControls();
  resizeCanvas();
  if (!project.viewport || project.viewport.scale === 0.25) fitMap();
  render();
  renderUi();
  setSaveState("Saved");
}

function setupControls() {
  populateLayerSelect();
  elements.layerSelect.addEventListener("change", () => {
    app.layer = elements.layerSelect.value;
    saveViewportSoon();
    render();
  });

  elements.classSelect.innerHTML = "";
  for (const classification of app.project.classifications) {
    const option = document.createElement("option");
    option.value = classification.id;
    option.textContent = `${classification.group} / ${classification.name}`;
    elements.classSelect.append(option);
  }
  elements.classSelect.value = app.classification;
  elements.classSelect.addEventListener("change", () => {
    app.classification = elements.classSelect.value;
  });

  document.querySelectorAll(".tool-button").forEach((button) => {
    button.addEventListener("click", () => setTool(button.dataset.tool));
  });
  elements.undoButton.addEventListener("click", undo);
  elements.redoButton.addEventListener("click", redo);
  elements.deleteButton.addEventListener("click", deleteSelected);
  elements.fitButton.addEventListener("click", fitMap);
  elements.propertiesSave.addEventListener("click", saveProperties);
  elements.extractButton.addEventListener("click", startFeatureExtraction);
  elements.trainButton.addEventListener("click", startClassifierTraining);
  elements.refreshAssetsButton.addEventListener("click", refreshAssets);

  canvas.addEventListener("pointerdown", pointerDown);
  canvas.addEventListener("pointermove", pointerMove);
  canvas.addEventListener("pointerup", pointerUp);
  canvas.addEventListener("pointerleave", pointerUp);
  canvas.addEventListener("dblclick", doubleClick);
  canvas.addEventListener("wheel", wheel, { passive: false });
  window.addEventListener("resize", () => {
    resizeCanvas();
    render();
  });
  window.addEventListener("keydown", keydown);
  window.addEventListener("keyup", keyup);
}

function populateLayerSelect() {
  const currentLayer = app.layer;
  elements.layerSelect.innerHTML = "";
  for (const layer of app.assets.layers) {
    const option = document.createElement("option");
    option.value = layer.id;
    option.textContent = layer.name;
    elements.layerSelect.append(option);
  }
  app.layer = app.assets.layers.some((layer) => layer.id === currentLayer) ? currentLayer : app.assets.primaryLayer;
  elements.layerSelect.value = app.layer;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${body}`);
  }
  return response.json();
}

function setTool(tool) {
  app.tool = tool;
  app.pendingLine = null;
  app.pendingFace = tool === "face" ? app.pendingFace : [];
  document.querySelectorAll(".tool-button").forEach((button) => button.classList.toggle("is-active", button.dataset.tool === tool));
  render();
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  app.dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * app.dpr));
  canvas.height = Math.max(1, Math.round(rect.height * app.dpr));
  ctx.setTransform(app.dpr, 0, 0, app.dpr, 0, 0);
}

function activeLayer() {
  return app.assets.layers.find((layer) => layer.id === app.layer) || app.assets.layers[0];
}

function fitMap() {
  const layer = activeLayer();
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / layer.width, rect.height / layer.height) * 0.94;
  app.view.scale = clamp(scale, 0.01, 24);
  app.view.x = (layer.width - rect.width / app.view.scale) / 2;
  app.view.y = (layer.height - rect.height / app.view.scale) / 2;
  saveViewportSoon();
  render();
}

function render() {
  if (!app.project || !app.assets) return;
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#d9dee5";
  ctx.fillRect(0, 0, rect.width, rect.height);
  drawTiles();
  drawFaces();
  drawAnnotations();
  drawLines();
  drawVertices();
  drawDraft();
  elements.zoomReadout.textContent = `${Math.round(app.view.scale * 100)}%`;
}

function renderUi() {
  renderStats();
  renderInspector();
  renderHistory();
}

function drawTiles() {
  const layer = activeLayer();
  const rect = canvas.getBoundingClientRect();
  const maxZoom = layer.maxZoom;
  const ideal = Math.round(maxZoom + Math.log2(Math.max(app.view.scale, 0.001)));
  const z = clamp(ideal, 0, maxZoom);
  const downsample = 2 ** (maxZoom - z);
  const tileWorld = layer.tileSize * downsample;
  const firstX = Math.floor(app.view.x / tileWorld) - 1;
  const firstY = Math.floor(app.view.y / tileWorld) - 1;
  const lastX = Math.ceil((app.view.x + rect.width / app.view.scale) / tileWorld) + 1;
  const lastY = Math.ceil((app.view.y + rect.height / app.view.scale) / tileWorld) + 1;

  for (let y = firstY; y <= lastY; y += 1) {
    for (let x = firstX; x <= lastX; x += 1) {
      if (x < 0 || y < 0) continue;
      const url = `/api/tiles/${layer.id}/${z}/${x}/${y}`;
      const image = getTile(url);
      if (!image.complete) continue;
      const screen = worldToScreen(x * tileWorld, y * tileWorld);
      const size = tileWorld * app.view.scale;
      ctx.drawImage(image, screen.x, screen.y, size, size);
    }
  }
}

function getTile(url) {
  if (app.cache.has(url)) return app.cache.get(url);
  const image = new Image();
  image.onload = render;
  image.src = url;
  app.cache.set(url, image);
  return image;
}

function drawAnnotations() {
  for (const annotation of app.project.annotations) {
    if (annotation.visible === false) continue;
    const color = classificationColor(annotation.classification_id);
    const point = worldToScreen(annotation.x, annotation.y);
    const width = annotation.width * app.view.scale;
    const height = annotation.height * app.view.scale;
    ctx.fillStyle = withAlpha(color, 0.22);
    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected("annotations", annotation.id) ? 3 : 1.5;
    ctx.fillRect(point.x, point.y, width, height);
    ctx.strokeRect(point.x, point.y, width, height);
  }
}

function drawVertices() {
  for (const vertex of app.project.vertices) {
    const point = worldToScreen(vertex.u, vertex.v);
    ctx.beginPath();
    ctx.arc(point.x, point.y, isSelected("vertices", vertex.id) ? 6 : 4, 0, Math.PI * 2);
    ctx.fillStyle = isPending(vertex.id) ? "#b45309" : "#ffffff";
    ctx.strokeStyle = isSelected("vertices", vertex.id) ? "#0f766e" : "#172033";
    ctx.lineWidth = isSelected("vertices", vertex.id) ? 3 : 1.5;
    ctx.fill();
    ctx.stroke();
  }
}

function drawLines() {
  for (const line of app.project.lines) {
    const vertices = line.vertex_ids.map(findVertex).filter(Boolean);
    if (vertices.length < 2) continue;
    ctx.beginPath();
    vertices.forEach((vertex, index) => {
      const point = worldToScreen(vertex.u, vertex.v);
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.strokeStyle = isSelected("lines", line.id) ? "#0f766e" : "#172033";
    ctx.lineWidth = isSelected("lines", line.id) ? 4 : 2;
    ctx.stroke();
  }
}

function drawFaces() {
  for (const face of app.project.faces) {
    const vertices = face.vertex_ids.map(findVertex).filter(Boolean);
    if (vertices.length < 3) continue;
    ctx.beginPath();
    vertices.forEach((vertex, index) => {
      const point = worldToScreen(vertex.u, vertex.v);
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.closePath();
    const color = classificationColor(face.classification_id);
    ctx.fillStyle = withAlpha(color, 0.18);
    ctx.strokeStyle = isSelected("faces", face.id) ? "#0f766e" : color;
    ctx.lineWidth = isSelected("faces", face.id) ? 3 : 1.5;
    ctx.fill();
    ctx.stroke();
  }
}

function drawDraft() {
  if (app.drag?.type === "rectangle") {
    const start = worldToScreen(app.drag.start.x, app.drag.start.y);
    const current = worldToScreen(app.drag.current.x, app.drag.current.y);
    ctx.setLineDash([6, 5]);
    ctx.strokeStyle = classificationColor(app.classification);
    ctx.lineWidth = 2;
    ctx.strokeRect(start.x, start.y, current.x - start.x, current.y - start.y);
    ctx.setLineDash([]);
  }
  if (app.pendingLine) {
    const vertex = findVertex(app.pendingLine);
    if (vertex) {
      const start = worldToScreen(vertex.u, vertex.v);
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(app.lastPointer.x, app.lastPointer.y);
      ctx.strokeStyle = "#b45309";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
  if (app.pendingFace.length) {
    ctx.beginPath();
    app.pendingFace.map(findVertex).filter(Boolean).forEach((vertex, index) => {
      const point = worldToScreen(vertex.u, vertex.v);
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.lineTo(app.lastPointer.x, app.lastPointer.y);
    ctx.strokeStyle = "#b45309";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function pointerDown(event) {
  canvas.setPointerCapture(event.pointerId);
  const screen = eventScreen(event);
  const world = screenToWorld(screen.x, screen.y);
  app.lastPointer = screen;

  if (event.button === 1 || event.button === 2 || app.tool === "pan") {
    app.drag = { type: "pan", screen, view: { ...app.view } };
    return;
  }

  if (app.tool === "rectangle") {
    app.drag = { type: "rectangle", start: world, current: world };
    return;
  }

  if (app.tool === "select") {
    const hit = hitTest(screen.x, screen.y);
    select(hit);
    if (hit && (hit.kind === "annotations" || hit.kind === "vertices")) {
      app.drag = { type: "move", hit, startWorld: world, original: structuredClone(hit.entity) };
    }
    return;
  }
}

function pointerMove(event) {
  const screen = eventScreen(event);
  const world = screenToWorld(screen.x, screen.y);
  app.lastPointer = screen;
  elements.coordinateReadout.textContent = `${Math.round(world.x)}, ${Math.round(world.y)}`;

  if (!app.drag) {
    render();
    return;
  }
  if (app.drag.type === "pan") {
    app.view.x = app.drag.view.x - (screen.x - app.drag.screen.x) / app.view.scale;
    app.view.y = app.drag.view.y - (screen.y - app.drag.screen.y) / app.view.scale;
    saveViewportSoon();
  } else if (app.drag.type === "rectangle") {
    app.drag.current = world;
  } else if (app.drag.type === "move") {
    moveDraftEntity(app.drag, world);
  }
  render();
}

async function pointerUp(event) {
  const screen = eventScreen(event);
  const world = screenToWorld(screen.x, screen.y);
  if (!app.drag) {
    if (event.button === 0) await clickWorld(world, screen);
    return;
  }

  const drag = app.drag;
  app.drag = null;
  if (drag.type === "rectangle") {
    await createRectangle(drag.start, world);
  } else if (drag.type === "move") {
    await persistMovedEntity(drag);
  }
  render();
}

async function clickWorld(world, screen) {
  if (app.tool === "vertex") {
    const vertex = await addEntity("vertices", { u: world.x, v: world.y });
    select({ kind: "vertices", id: vertex.id, entity: vertex });
  } else if (app.tool === "line") {
    const vertex = await vertexAtOrCreate(world, screen);
    if (!app.pendingLine) {
      app.pendingLine = vertex.id;
    } else if (app.pendingLine !== vertex.id) {
      const line = await addEntity("lines", { vertex_ids: [app.pendingLine, vertex.id] });
      app.pendingLine = null;
      select({ kind: "lines", id: line.id, entity: line });
    }
  } else if (app.tool === "face") {
    const vertex = await vertexAtOrCreate(world, screen);
    if (!app.pendingFace.includes(vertex.id)) app.pendingFace.push(vertex.id);
    if (app.pendingFace.length >= 3 && vertex.id === app.pendingFace[0]) await finishFace();
  }
  render();
}

async function doubleClick() {
  if (app.tool === "face") await finishFace();
}

async function finishFace() {
  const vertexIds = [...new Set(app.pendingFace)];
  if (vertexIds.length < 3) return;
  const face = await addEntity("faces", { vertex_ids: vertexIds, classification_id: app.classification });
  app.pendingFace = [];
  select({ kind: "faces", id: face.id, entity: face });
}

function wheel(event) {
  event.preventDefault();
  const screen = eventScreen(event);
  const before = screenToWorld(screen.x, screen.y);
  const factor = Math.exp(-event.deltaY * 0.001);
  app.view.scale = clamp(app.view.scale * factor, 0.01, 24);
  app.view.x = before.x - screen.x / app.view.scale;
  app.view.y = before.y - screen.y / app.view.scale;
  saveViewportSoon();
  render();
}

function keydown(event) {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    if (event.shiftKey) redo();
    else undo();
  } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
    event.preventDefault();
    redo();
  } else if (event.key === "Delete" || event.key === "Backspace") {
    if (app.selected) {
      event.preventDefault();
      deleteSelected();
    }
  } else if (event.key === "Enter" && app.tool === "face") {
    finishFace();
  } else if (event.key === "Escape") {
    app.pendingLine = null;
    app.pendingFace = [];
    app.drag = null;
    render();
  } else if (event.code === "Space" && !event.repeat) {
    canvas.dataset.previousTool = app.tool;
    setTool("pan");
  }
}

function keyup(event) {
  if (event.code === "Space" && canvas.dataset.previousTool) {
    setTool(canvas.dataset.previousTool);
    delete canvas.dataset.previousTool;
  }
}

async function createRectangle(start, end) {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  const width = Math.abs(end.x - start.x);
  const height = Math.abs(end.y - start.y);
  if (width < 4 || height < 4) return;
  const annotation = await addEntity("annotations", { x, y, width, height, classification_id: app.classification });
  select({ kind: "annotations", id: annotation.id, entity: annotation });
}

async function vertexAtOrCreate(world, screen) {
  const hit = hitTest(screen.x, screen.y, ["vertices"]);
  if (hit) return hit.entity;
  return addEntity("vertices", { u: world.x, v: world.y });
}

async function addEntity(collection, payload) {
  setSaveState("Saving");
  const result = await api(`/api/${collection}`, { method: "POST", body: JSON.stringify(payload) });
  const entity = result.entity;
  await refreshProject();
  setSaveState("Saved");
  return entity;
}

async function updateEntity(collection, id, patch) {
  setSaveState("Saving");
  const result = await api(`/api/${collection}/${id}`, { method: "PATCH", body: JSON.stringify(patch) });
  await refreshProject();
  setSaveState("Saved");
  return result.entity;
}

async function deleteSelected() {
  if (!app.selected) return;
  setSaveState("Saving");
  await api(`/api/${app.selected.kind}/${app.selected.id}`, { method: "DELETE" });
  app.selected = null;
  await saveSelection(null);
  await refreshProject();
  setSaveState("Saved");
}

async function undo() {
  setSaveState("Saving");
  const result = await api("/api/undo", { method: "POST", body: "{}" });
  app.project = result.project;
  app.selected = app.project.selection;
  renderUi();
  render();
  setSaveState("Saved");
}

async function redo() {
  setSaveState("Saving");
  const result = await api("/api/redo", { method: "POST", body: "{}" });
  app.project = result.project;
  app.selected = app.project.selection;
  renderUi();
  render();
  setSaveState("Saved");
}

async function startFeatureExtraction() {
  try {
    setJobStatus("Starting feature extraction");
    const result = await api("/api/extract-features", { method: "POST", body: "{}" });
    pollJob(result.job.id);
  } catch (error) {
    setJobStatus(error.message);
    console.error(error);
  }
}

async function startClassifierTraining() {
  try {
    setJobStatus("Starting classifier training");
    const result = await api("/api/train-classifier", { method: "POST", body: "{}" });
    pollJob(result.job.id);
  } catch (error) {
    setJobStatus(error.message);
    console.error(error);
  }
}

async function pollJob(jobId) {
  const job = await api(`/api/jobs/${jobId}`);
  setJobStatus(`${job.type}: ${job.status} - ${job.message}`);
  if (job.status === "completed") {
    await refreshAssets();
    if (job.result?.prediction_image_path) {
      app.layer = "classifier";
      elements.layerSelect.value = app.layer;
      saveViewportSoon();
    }
    render();
    return;
  }
  if (job.status === "failed") {
    console.error(job.error);
    return;
  }
  window.setTimeout(() => pollJob(jobId), 1600);
}

async function refreshAssets() {
  app.assets = await api("/api/assets");
  populateLayerSelect();
  app.cache.clear();
  render();
}

async function refreshProject() {
  app.project = await api("/api/project");
  if (app.selected) {
    const entity = findEntity(app.selected.kind, app.selected.id);
    app.selected = entity ? { ...app.selected, entity } : null;
  }
  renderUi();
  render();
}

function moveDraftEntity(drag, world) {
  const dx = world.x - drag.startWorld.x;
  const dy = world.y - drag.startWorld.y;
  const entity = findEntity(drag.hit.kind, drag.hit.id);
  if (!entity) return;
  if (drag.hit.kind === "annotations") {
    entity.x = drag.original.x + dx;
    entity.y = drag.original.y + dy;
  } else if (drag.hit.kind === "vertices") {
    entity.u = drag.original.u + dx;
    entity.v = drag.original.v + dy;
  }
}

async function persistMovedEntity(drag) {
  const entity = findEntity(drag.hit.kind, drag.hit.id);
  if (!entity) return;
  if (drag.hit.kind === "annotations") await updateEntity("annotations", entity.id, { x: entity.x, y: entity.y });
  if (drag.hit.kind === "vertices") await updateEntity("vertices", entity.id, { u: entity.u, v: entity.v });
}

async function select(hit) {
  app.selected = hit ? { kind: hit.kind, id: hit.id, entity: hit.entity } : null;
  await saveSelection(app.selected);
  renderUi();
  render();
}

async function saveSelection(selection) {
  const body = selection ? { kind: selection.kind, id: selection.id } : {};
  app.project.selection = selection ? { kind: selection.kind, id: selection.id } : null;
  await api("/api/selection", { method: "POST", body: JSON.stringify(body) });
}

function saveViewportSoon() {
  clearTimeout(app.viewportTimer);
  app.viewportTimer = setTimeout(async () => {
    setSaveState("Saving");
    await api("/api/viewport", {
      method: "POST",
      body: JSON.stringify({ ...app.view, layer: app.layer }),
    });
    setSaveState("Saved");
  }, 180);
}

function renderInspector() {
  const form = elements.propertiesForm;
  form.innerHTML = "";
  const selected = app.selected ? { ...app.selected, entity: findEntity(app.selected.kind, app.selected.id) } : null;
  if (!selected?.entity) {
    elements.selectionReadout.textContent = "No selection";
    form.append(emptyMessage("Nothing selected"));
    return;
  }
  elements.selectionReadout.textContent = `${singular(selected.kind)} ${selected.id}`;
  form.dataset.kind = selected.kind;
  form.dataset.id = selected.id;
  if (selected.kind === "annotations") renderAnnotationForm(form, selected.entity);
  else if (selected.kind === "vertices") renderVertexForm(form, selected.entity);
  else if (selected.kind === "lines") renderLineForm(form, selected.entity);
  else if (selected.kind === "faces") renderFaceForm(form, selected.entity);
  else if (selected.kind === "constraints") renderConstraintForm(form, selected.entity);
}

function renderAnnotationForm(form, entity) {
  form.append(inputField("label", "Label", entity.label || ""));
  form.append(selectField("classification_id", "Class", entity.classification_id));
  form.append(row(inputField("x", "X", entity.x, "number"), inputField("y", "Y", entity.y, "number")));
  form.append(row(inputField("width", "Width", entity.width, "number"), inputField("height", "Height", entity.height, "number")));
  form.append(checkboxField("visible", "Visible", entity.visible !== false));
  form.append(checkboxField("locked", "Locked", Boolean(entity.locked)));
  form.append(textField("properties", "Properties", JSON.stringify(entity.properties || {}, null, 2)));
}

function renderVertexForm(form, entity) {
  form.append(row(inputField("u", "U", entity.u, "number"), inputField("v", "V", entity.v, "number")));
  form.append(row(inputField("northing", "Northing", entity.geocoords?.northing ?? "", "number"), inputField("easting", "Easting", entity.geocoords?.easting ?? "", "number")));
  form.append(row(inputField("accuracy", "Accuracy", entity.geocoords?.accuracy ?? "", "number"), inputField("altitude", "Altitude", entity.altitude ?? "", "number")));
  form.append(textField("properties", "Properties", JSON.stringify(entity.properties || {}, null, 2)));
}

function renderLineForm(form, entity) {
  form.append(textField("vertex_ids", "Vertices", JSON.stringify(entity.vertex_ids || [], null, 2)));
  form.append(checkboxField("vertical", "Vertical", Boolean(entity.vertical)));
  form.append(checkboxField("horizontal", "Horizontal", Boolean(entity.horizontal)));
  form.append(inputField("length", "Length", entity.length ?? "", "number"));
  form.append(textField("properties", "Properties", JSON.stringify(entity.properties || {}, null, 2)));
}

function renderFaceForm(form, entity) {
  form.append(selectField("classification_id", "Class", entity.classification_id));
  form.append(textField("vertex_ids", "Vertices", JSON.stringify(entity.vertex_ids || [], null, 2)));
  form.append(row(checkboxField("roof", "Roof", Boolean(entity.roof)), checkboxField("wall", "Wall", Boolean(entity.wall))));
  form.append(row(checkboxField("door", "Door", Boolean(entity.door)), checkboxField("window", "Window", Boolean(entity.window))));
  form.append(textField("properties", "Properties", JSON.stringify(entity.properties || {}, null, 2)));
}

function renderConstraintForm(form, entity) {
  form.append(inputField("constraint_type", "Type", entity.constraint_type || ""));
  form.append(textField("subject_ids", "Subjects", JSON.stringify(entity.subject_ids || [], null, 2)));
  form.append(textField("data", "Data", JSON.stringify(entity.data || {}, null, 2)));
}

async function saveProperties(event) {
  event.preventDefault();
  const form = elements.propertiesForm;
  const kind = form.dataset.kind;
  const id = form.dataset.id;
  if (!kind || !id) return;
  try {
    const patch = formPatch(kind, new FormData(form));
    await updateEntity(kind, id, patch);
  } catch (error) {
    setSaveState("Invalid JSON");
    console.error(error);
  }
}

function formPatch(kind, data) {
  const patch = {};
  for (const [key, value] of data.entries()) {
    if (["x", "y", "width", "height", "u", "v", "length", "altitude"].includes(key)) patch[key] = blankNumber(value);
    else if (["properties", "vertex_ids", "subject_ids", "data"].includes(key)) patch[key] = value ? JSON.parse(value) : key.endsWith("ids") ? [] : {};
    else if (["northing", "easting", "accuracy"].includes(key)) {
      patch.geocoords ||= {};
      const number = blankNumber(value);
      if (number !== null) patch.geocoords[key] = number;
    } else patch[key] = value;
  }
  for (const checkbox of elements.propertiesForm.querySelectorAll("input[type='checkbox']")) {
    patch[checkbox.name] = checkbox.checked;
  }
  if (kind === "vertices") {
    patch.geocoords ||= {};
  }
  return patch;
}

function renderStats() {
  const counts = [
    ["Annotations", app.project.annotations.length],
    ["Vertices", app.project.vertices.length],
    ["Lines", app.project.lines.length],
    ["Faces", app.project.faces.length],
    ["Constraints", app.project.constraints.length],
    ["Undo", app.project.history?.past?.length || 0],
  ];
  elements.stats.innerHTML = counts.map(([label, value]) => `<div class="stat"><strong>${value}</strong>${label}</div>`).join("");
}

function renderHistory() {
  const actions = [...(app.project.history?.past || [])].slice(-8).reverse();
  elements.historyList.innerHTML = actions.map((action) => `<li>${escapeHtml(action.label || action.id)}</li>`).join("");
}

function hitTest(screenX, screenY, onlyKinds = null) {
  const kinds = onlyKinds || ["vertices", "lines", "annotations", "faces"];
  if (kinds.includes("vertices")) {
    for (let index = app.project.vertices.length - 1; index >= 0; index -= 1) {
      const vertex = app.project.vertices[index];
      const point = worldToScreen(vertex.u, vertex.v);
      if (distance(point.x, point.y, screenX, screenY) < 10) return { kind: "vertices", id: vertex.id, entity: vertex };
    }
  }
  if (kinds.includes("lines")) {
    for (let index = app.project.lines.length - 1; index >= 0; index -= 1) {
      const line = app.project.lines[index];
      const vertices = line.vertex_ids.map(findVertex).filter(Boolean);
      for (let i = 1; i < vertices.length; i += 1) {
        if (distanceToSegment(screenX, screenY, worldToScreen(vertices[i - 1].u, vertices[i - 1].v), worldToScreen(vertices[i].u, vertices[i].v)) < 8) {
          return { kind: "lines", id: line.id, entity: line };
        }
      }
    }
  }
  if (kinds.includes("annotations")) {
    const world = screenToWorld(screenX, screenY);
    for (let index = app.project.annotations.length - 1; index >= 0; index -= 1) {
      const annotation = app.project.annotations[index];
      if (world.x >= annotation.x && world.x <= annotation.x + annotation.width && world.y >= annotation.y && world.y <= annotation.y + annotation.height) {
        return { kind: "annotations", id: annotation.id, entity: annotation };
      }
    }
  }
  if (kinds.includes("faces")) {
    const world = screenToWorld(screenX, screenY);
    for (let index = app.project.faces.length - 1; index >= 0; index -= 1) {
      const face = app.project.faces[index];
      const vertices = face.vertex_ids.map(findVertex).filter(Boolean);
      if (pointInPolygon(world, vertices)) return { kind: "faces", id: face.id, entity: face };
    }
  }
  return null;
}

function findEntity(kind, id) {
  return app.project?.[kind]?.find((entity) => entity.id === id) || null;
}

function findVertex(id) {
  return findEntity("vertices", id);
}

function worldToScreen(x, y) {
  return { x: (x - app.view.x) * app.view.scale, y: (y - app.view.y) * app.view.scale };
}

function screenToWorld(x, y) {
  return { x: app.view.x + x / app.view.scale, y: app.view.y + y / app.view.scale };
}

function eventScreen(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function classificationColor(id) {
  return app.project.classifications.find((classification) => classification.id === id)?.color || "#0f766e";
}

function withAlpha(hex, alpha) {
  const clean = hex.replace("#", "");
  const red = parseInt(clean.slice(0, 2), 16);
  const green = parseInt(clean.slice(2, 4), 16);
  const blue = parseInt(clean.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function isSelected(kind, id) {
  return app.selected?.kind === kind && app.selected?.id === id;
}

function isPending(id) {
  return app.pendingLine === id || app.pendingFace.includes(id);
}

function inputField(name, label, value, type = "text") {
  const wrapper = document.createElement("label");
  wrapper.textContent = label;
  const input = document.createElement("input");
  input.name = name;
  input.type = type;
  input.step = type === "number" ? "any" : undefined;
  input.value = value ?? "";
  wrapper.append(input);
  return wrapper;
}

function selectField(name, label, value) {
  const wrapper = document.createElement("label");
  wrapper.textContent = label;
  const select = document.createElement("select");
  select.name = name;
  for (const classification of app.project.classifications) {
    const option = document.createElement("option");
    option.value = classification.id;
    option.textContent = `${classification.group} / ${classification.name}`;
    select.append(option);
  }
  select.value = value || app.classification;
  wrapper.append(select);
  return wrapper;
}

function checkboxField(name, label, checked) {
  const wrapper = document.createElement("label");
  wrapper.className = "checkbox-row";
  const span = document.createElement("span");
  span.textContent = label;
  const input = document.createElement("input");
  input.name = name;
  input.type = "checkbox";
  input.checked = checked;
  wrapper.append(span, input);
  return wrapper;
}

function textField(name, label, value) {
  const wrapper = document.createElement("label");
  wrapper.textContent = label;
  const textarea = document.createElement("textarea");
  textarea.name = name;
  textarea.value = value || "";
  wrapper.append(textarea);
  return wrapper;
}

function row(...children) {
  const wrapper = document.createElement("div");
  wrapper.className = "row";
  wrapper.append(...children);
  return wrapper;
}

function emptyMessage(text) {
  const node = document.createElement("p");
  node.className = "field-name";
  node.textContent = text;
  return node;
}

function blankNumber(value) {
  if (value === "" || value === null || value === undefined) return null;
  return Number(value);
}

function singular(kind) {
  return kind.endsWith("s") ? kind.slice(0, -1) : kind;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function distance(x1, y1, x2, y2) {
  return Math.hypot(x1 - x2, y1 - y2);
}

function distanceToSegment(px, py, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (dx === 0 && dy === 0) return distance(px, py, a.x, a.y);
  const t = clamp(((px - a.x) * dx + (py - a.y) * dy) / (dx * dx + dy * dy), 0, 1);
  return distance(px, py, a.x + t * dx, a.y + t * dy);
}

function pointInPolygon(point, vertices) {
  let inside = false;
  for (let i = 0, j = vertices.length - 1; i < vertices.length; j = i, i += 1) {
    const xi = vertices[i].u;
    const yi = vertices[i].v;
    const xj = vertices[j].u;
    const yj = vertices[j].v;
    const intersect = yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function setSaveState(text) {
  elements.saveState.textContent = text;
}

function setJobStatus(text) {
  elements.jobStatus.textContent = text;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]);
}
