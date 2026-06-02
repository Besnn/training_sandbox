const state = {
  models: [],
  classes: [],
  model: null,
  rows: [],
  filtered: [],
  selected: null,
  mode: "annotated",
  labels: [],
  centroidLabels: [],
};

const els = {
  modelSelect: document.querySelector("#modelSelect"),
  subtitle: document.querySelector("#subtitle"),
  stats: document.querySelector("#stats"),
  caseList: document.querySelector("#caseList"),
  searchInput: document.querySelector("#searchInput"),
  filterFp: document.querySelector("#filterFp"),
  filterFn: document.querySelector("#filterFn"),
  filterMismatch: document.querySelector("#filterMismatch"),
  filterTp: document.querySelector("#filterTp"),
  filterIssues: document.querySelector("#filterIssues"),
  onlySelected: document.querySelector("#onlySelected"),
  rawMode: document.querySelector("#rawMode"),
  annotatedMode: document.querySelector("#annotatedMode"),
  compareMode: document.querySelector("#compareMode"),
  showGt: document.querySelector("#showGt"),
  showDetections: document.querySelector("#showDetections"),
  showTp: document.querySelector("#showTp"),
  showFp: document.querySelector("#showFp"),
  showFn: document.querySelector("#showFn"),
  showCentroids: document.querySelector("#showCentroids"),
  showLabels: document.querySelector("#showLabels"),
  fitImage: document.querySelector("#fitImage"),
  confSlider: document.querySelector("#confSlider"),
  confValue: document.querySelector("#confValue"),
  imageArea: document.querySelector("#imageArea"),
  rawFrame: document.querySelector("#rawFrame"),
  annotatedFrame: document.querySelector("#annotatedFrame"),
  rawImage: document.querySelector("#rawImage"),
  annotatedImage: document.querySelector("#annotatedImage"),
  overlayCanvas: document.querySelector("#overlayCanvas"),
  caseTitle: document.querySelector("#caseTitle"),
  caseMeta: document.querySelector("#caseMeta"),
  labelTable: document.querySelector("#labelTable"),
  caseDetails: document.querySelector("#caseDetails"),
};

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function imageUrl(kind, params) {
  const query = new URLSearchParams(params);
  return `/asset/${kind}?${query.toString()}`;
}

function labelUrl(image, kind) {
  const query = new URLSearchParams({ image, kind });
  return `/api/labels?${query.toString()}`;
}

function inspectionAssetUrl(path) {
  const query = new URLSearchParams({ model: state.model.id, path });
  return `/asset/inspection?${query.toString()}`;
}

function rawImageUrl(row) {
  if (row.rawImage && !row.rawImage.startsWith("/")) {
    return inspectionAssetUrl(row.rawImage);
  }
  return imageUrl("raw", { image: row.image });
}

function annotatedImageUrl(row) {
  if (!row.annotatedImage) {
    return "";
  }
  if (row.annotatedImage.includes("/")) {
    return inspectionAssetUrl(row.annotatedImage);
  }
  return imageUrl("annotated", {
    model: state.model.id,
    image: row.annotatedImage,
  });
}

function modelLabel(model) {
  return model.group ? `${model.group}/${model.name}` : model.name;
}

function badge(label, value, className) {
  return `<span class="badge ${className}">${label} ${value}</span>`;
}

function renderStats() {
  const totals = state.model?.totals || {};
  const items = [
    ["Cases", totals.images || 0],
    ["FP", totals.fp || 0],
    ["FN", totals.fn || 0],
    ["Mismatch", totals.mismatch || 0],
  ];
  els.stats.innerHTML = items.map(([label, value]) => (
    `<div class="stat"><span>${label}</span><strong>${value}</strong></div>`
  )).join("");
}

function passesFilters(row) {
  const query = els.searchInput.value.trim().toLowerCase();
  if (query && !row.image.toLowerCase().includes(query)) {
    return false;
  }
  if (els.onlySelected.checked && state.selected && row.index !== state.selected.index) {
    return false;
  }

  const wantsFp = els.filterFp.checked && row.fp > 0;
  const wantsFn = els.filterFn.checked && row.fn > 0;
  const wantsMismatch = els.filterMismatch.checked && row.mismatch > 0;
  const wantsTp = els.filterTp.checked && row.tp > 0;
  const wantsIssues = els.filterIssues.checked && row.labelIssues;

  if (!els.filterFp.checked && !els.filterFn.checked && !els.filterMismatch.checked && !els.filterTp.checked && !els.filterIssues.checked) {
    return true;
  }
  return wantsFp || wantsFn || wantsMismatch || wantsTp || wantsIssues;
}

function applyFilters() {
  state.filtered = state.rows.filter(passesFilters);
  renderCaseList();
}

function renderCaseList() {
  if (!state.filtered.length) {
    els.caseList.innerHTML = '<div class="empty">No cases match the current filters.</div>';
    return;
  }

  els.caseList.innerHTML = state.filtered.map((row) => {
    const active = state.selected && row.index === state.selected.index ? " active" : "";
    const issueBadge = row.labelIssues ? badge("issues", 1, "issue") : "";
    return `
      <button class="case-card${active}" type="button" data-index="${row.index}">
        <strong>${row.image}</strong>
        <div class="badges">
          ${badge("FP", row.fp, "fp")}
          ${badge("FN", row.fn, "fn")}
          ${badge("MM", row.mismatch, "mismatch")}
          ${badge("TP", row.tp, "")}
          ${issueBadge}
        </div>
      </button>
    `;
  }).join("");
}

function renderDetails() {
  const row = state.selected;
  if (!row) {
    els.caseDetails.innerHTML = '<div class="empty">No case selected.</div>';
    return;
  }

  const arch = modelArchitecture();
  const fpBestLabel = arch === "fomo" ? "FP best distances" : "FP best IoUs";
  const fpBestValue = arch === "fomo"
    ? (row.fpBestDistances || "none")
    : (row.fpBestIous || "none");
  const items = [
    ["Image", row.image],
    ["Annotated", row.annotatedImage],
    ["Architecture", arch],
    ["GT count", row.gtCount],
    ["False positives", row.fp],
    ["False negatives", row.fn],
    ["Class mismatches", row.mismatch],
    ["True positives", row.tp],
    [fpBestLabel, fpBestValue],
    ["Label issues", row.labelIssues || "none"],
  ];
  els.caseDetails.innerHTML = items.map(([key, value]) => (
    `<div class="table-row"><div class="key">${key}</div><div>${escapeHtml(String(value))}</div></div>`
  )).join("");
}

function renderLabelTable() {
  const all = [
    ...state.labels.map((label) => ({ ...label, source: "OBB" })),
    ...state.centroidLabels.map((label) => ({ ...label, source: "centroid" })),
  ];

  if (!all.length) {
    els.labelTable.innerHTML = '<div class="empty">No labels found for this image.</div>';
    return;
  }

  els.labelTable.innerHTML = all.map((label) => {
    let value = "";
    if (label.type === "obb") {
      value = `${label.source} ${label.className}, ${label.points.length} points`;
    } else if (label.type === "centroid") {
      value = `${label.source} ${label.className}, x=${format(label.x)}, y=${format(label.y)}`;
    } else {
      value = `${label.source} ${label.raw || ""}`;
    }
    return `<div class="table-row"><div class="key">line ${label.line}</div><div>${escapeHtml(value)}</div></div>`;
  }).join("");
}

function setMode(mode) {
  state.mode = mode;
  for (const [button, value] of [
    [els.rawMode, "raw"],
    [els.annotatedMode, "annotated"],
    [els.compareMode, "compare"],
  ]) {
    button.classList.toggle("active", mode === value);
  }
  renderImageMode();
}

function renderImageMode() {
  els.imageArea.classList.toggle("compare", state.mode === "compare");
  els.rawFrame.classList.toggle("hidden", state.mode === "annotated");
  els.annotatedFrame.classList.toggle("hidden", state.mode === "raw");
  els.rawFrame.classList.toggle("fit", els.fitImage.checked);
  els.annotatedFrame.classList.toggle("fit", els.fitImage.checked);
  drawOverlay();
}

async function selectRow(index) {
  const row = state.rows.find((item) => item.index === index);
  if (!row) return;

  state.selected = row;
  state.labels = [];
  state.centroidLabels = [];

  els.caseTitle.textContent = row.image;
  els.caseMeta.textContent = `FP ${row.fp} · FN ${row.fn} · mismatch ${row.mismatch} · TP ${row.tp}`;
  els.rawImage.src = rawImageUrl(row);
  const annotatedSrc = annotatedImageUrl(row);
  els.annotatedImage.src = annotatedSrc || "";
  els.annotatedFrame.classList.toggle("missing", !annotatedSrc);

  const [obb, centroid] = await Promise.all([
    api(labelUrl(row.image, "obb")),
    api(labelUrl(row.image, "centroid")).catch(() => ({ labels: [] })),
  ]);
  state.labels = obb.labels || [];
  state.centroidLabels = centroid.labels || [];

  renderCaseList();
  renderDetails();
  renderLabelTable();
  drawOverlay();
}

function drawOverlay() {
  const canvas = els.overlayCanvas;
  const img = els.rawImage;
  const ctx = canvas.getContext("2d");
  if (!img.complete || !img.naturalWidth) {
    return;
  }

  const displayWidth = img.clientWidth || img.naturalWidth;
  const displayHeight = img.clientHeight || img.naturalHeight;
  canvas.width = displayWidth;
  canvas.height = displayHeight;
  canvas.style.width = `${displayWidth}px`;
  canvas.style.height = `${displayHeight}px`;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx._labelBoxes = [];

  const sx = displayWidth;
  const sy = displayHeight;
  const row = state.selected;

  if (row && (row.groundTruth?.length || row.detections?.length)) {
    drawInspectionOverlay(ctx, row, displayWidth, displayHeight, img.naturalWidth, img.naturalHeight);
  } else if (els.showGt.checked) {
    for (const label of state.labels) {
      if (label.type !== "obb") continue;
      drawPolygon(ctx, label.points, sx, sy, "#22a06b", label.className);
    }
  }

  if (els.showCentroids.checked) {
    for (const label of state.centroidLabels) {
      if (label.type !== "centroid") continue;
      drawCentroid(ctx, label.x * sx, label.y * sy, "#1864c9", label.className);
    }
  }
}

function drawInspectionOverlay(ctx, row, displayWidth, displayHeight, naturalWidth, naturalHeight) {
  const sx = displayWidth / naturalWidth;
  const sy = displayHeight / naturalHeight;
  const arch = modelArchitecture();

  if (els.showGt.checked) {
    for (const gt of row.groundTruth || []) {
      if (!shouldShowGroundTruth(gt.status)) continue;
      const color = gt.status === "FN" ? "#e87916" : gt.status === "CLASS_MISMATCH" ? "#b7791f" : "#22a06b";
      if (arch === "fomo") {
        drawInspectionCentroid(ctx, gt.centroid, sx, sy, color, gtOverlayLabel(gt, row, arch), false);
      } else {
        drawPixelPolygon(ctx, gt.polygon || [], sx, sy, color, gtOverlayLabel(gt, row, arch), false);
      }
    }
  }

  if (els.showDetections.checked) {
    for (const det of row.detections || []) {
      if (Number(det.score || 0) < currentConfidence()) continue;
      if (!shouldShowDetection(det.status)) continue;
      const color = detectionColor(det.status);
      if (arch === "fomo") {
        drawInspectionCentroid(ctx, det.centroid, sx, sy, color, detectionOverlayLabel(det, arch), true);
      } else {
        drawPixelPolygon(ctx, det.polygon || [], sx, sy, color, detectionOverlayLabel(det, arch), true);
      }
    }
  }
}

function modelArchitecture() {
  const explicit = state.model?.architecture;
  if (explicit) return explicit;
  for (const row of state.rows) {
    for (const layer of [...(row.detections || []), ...(row.groundTruth || [])]) {
      if (layer.centroid) return "fomo";
      if (layer.polygon) return "yolo-obb";
    }
  }
  return "yolo-obb";
}

function drawInspectionCentroid(ctx, centroid, sx, sy, color, label, dashed) {
  if (!centroid) return;
  const x = Number(centroid.x || 0) * sx;
  const y = Number(centroid.y || 0) * sy;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = `${color}33`;
  ctx.lineWidth = 3;
  ctx.setLineDash(dashed ? [6, 4] : []);
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(x - 14, y);
  ctx.lineTo(x + 14, y);
  ctx.moveTo(x, y - 14);
  ctx.lineTo(x, y + 14);
  ctx.stroke();
  if (els.showLabels.checked) {
    drawTag(ctx, label, x + 10, y - 10, color);
  }
  ctx.restore();
}

function currentConfidence() {
  return Number(els.confSlider.value || 0);
}

function updateConfidenceLabel() {
  els.confValue.textContent = currentConfidence().toFixed(2);
}

function shouldShowGroundTruth(status) {
  if (status === "TP") return els.showTp.checked;
  if (status === "FN") return els.showFn.checked;
  if (status === "CLASS_MISMATCH") return els.showFp.checked || els.showFn.checked;
  return true;
}

function shouldShowDetection(status) {
  if (status === "TP") return els.showTp.checked;
  if (status === "FP") return els.showFp.checked;
  if (status === "CLASS_MISMATCH") return els.showFp.checked;
  return true;
}

function gtOverlayLabel(gt, row, arch = modelArchitecture()) {
  const match = Number.isInteger(gt.matchIndex) ? row.detections?.[gt.matchIndex] : null;
  const metricLabel = arch === "fomo" ? "dist" : "IoU";
  if (!match) {
    return `GT ${gt.className} same ${metricLabel} -- conf --`;
  }
  const metric = arch === "fomo"
    ? sameClassDistance(match, gt.matchIndex ?? null)
    : sameClassIoU(match, gt.matchIndex ?? null);
  const conf = Number(match.score || 0).toFixed(2);
  return `GT ${gt.className} same ${metricLabel} ${metric} conf ${conf}`;
}

function detectionOverlayLabel(det, arch = modelArchitecture()) {
  const conf = Number(det.score || 0).toFixed(2);
  if (arch === "fomo") {
    return `${det.status} ${det.className} conf ${conf} same dist ${sameClassDistance(det)}`;
  }
  return `${det.status} ${det.className} conf ${conf} same IoU ${sameClassIoU(det)}`;
}

function sameClassIoU(det, preferredGtIndex = null) {
  const ious = det.ious || [];
  if (Number.isInteger(preferredGtIndex)) {
    const preferred = ious.find((item) => item.gtIndex === preferredGtIndex);
    if (preferred) {
      return Number(preferred.iou || 0).toFixed(2);
    }
  }
  if (ious.length) {
    const best = ious.reduce((acc, item) => (
      Number(item.iou || 0) > Number(acc.iou || 0) ? item : acc
    ), ious[0]);
    return Number(best.iou || 0).toFixed(2);
  }
  return "0.00";
}

function sameClassDistance(det, preferredGtIndex = null) {
  const distances = det.distances || [];
  if (Number.isInteger(preferredGtIndex)) {
    const preferred = distances.find((item) => item.gtIndex === preferredGtIndex);
    if (preferred) {
      return Number(preferred.distance || 0).toFixed(2);
    }
  }
  if (distances.length) {
    const best = distances.reduce((acc, item) => (
      Number(item.distance || 0) < Number(acc.distance || 0) ? item : acc
    ), distances[0]);
    return Number(best.distance || 0).toFixed(2);
  }
  return "--";
}

function detectionColor(status) {
  if (status === "TP") return "#2563eb";
  if (status === "FP") return "#dc2626";
  if (status === "CLASS_MISMATCH") return "#b7791f";
  return "#7c3aed";
}

function drawPixelPolygon(ctx, points, sx, sy, color, label, dashed) {
  if (!points.length) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.fillStyle = `${color}22`;
  ctx.setLineDash(dashed ? [8, 5] : []);
  ctx.beginPath();
  ctx.moveTo(points[0].x * sx, points[0].y * sy);
  for (const point of points.slice(1)) {
    ctx.lineTo(point.x * sx, point.y * sy);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.setLineDash([]);
  if (els.showLabels.checked) {
    drawTag(ctx, label, points[0].x * sx, points[0].y * sy, color);
  }
  ctx.restore();
}

function drawPolygon(ctx, points, width, height, color, label) {
  if (!points.length) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.fillStyle = "rgba(34, 160, 107, 0.12)";
  ctx.beginPath();
  ctx.moveTo(points[0].x * width, points[0].y * height);
  for (const point of points.slice(1)) {
    ctx.lineTo(point.x * width, point.y * height);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  if (els.showLabels.checked) {
    drawTag(ctx, label, points[0].x * width, points[0].y * height, color);
  }
  ctx.restore();
}

function drawCentroid(ctx, x, y, color, label) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, 8, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 14, y);
  ctx.lineTo(x + 14, y);
  ctx.moveTo(x, y - 14);
  ctx.lineTo(x, y + 14);
  ctx.stroke();
  if (els.showLabels.checked) {
    drawTag(ctx, label, x + 10, y - 10, color);
  }
  ctx.restore();
}

function drawTag(ctx, text, x, y, color) {
  ctx.save();
  ctx.font = "12px system-ui, sans-serif";
  const metrics = ctx.measureText(text);
  const w = metrics.width + 10;
  const h = 20;
  const [tx, ty] = placeLabel(ctx, x, y, w, h);
  ctx.fillStyle = color;
  ctx.fillRect(tx, ty - h, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, tx + 5, ty - 6);
  ctx.restore();
}

function placeLabel(ctx, anchorX, anchorY, width, height) {
  const margin = 4;
  const boxes = ctx._labelBoxes || [];
  const candidates = [
    [anchorX + 8, anchorY - 8],
    [anchorX + 8, anchorY + height + 12],
    [anchorX - width - 8, anchorY - 8],
    [anchorX - width - 8, anchorY + height + 12],
    [anchorX - width / 2, anchorY - 20],
    [anchorX - width / 2, anchorY + height + 20],
    [anchorX + 22, anchorY + height / 2],
    [anchorX - width - 22, anchorY + height / 2],
  ];

  for (const candidate of candidates) {
    const box = labelBox(ctx, candidate[0], candidate[1], width, height, margin);
    if (!boxes.some((other) => boxesOverlap(box, other))) {
      boxes.push(box);
      ctx._labelBoxes = boxes;
      return [box.x, box.y + height];
    }
  }

  const step = height + 4;
  for (let row = 0; row < Math.max(1, Math.floor(ctx.canvas.height / step)); row += 1) {
    for (const x of [margin, ctx.canvas.width - width - margin]) {
      const box = labelBox(ctx, x, margin + row * step + height, width, height, margin);
      if (!boxes.some((other) => boxesOverlap(box, other))) {
        boxes.push(box);
        ctx._labelBoxes = boxes;
        return [box.x, box.y + height];
      }
    }
  }

  const fallback = labelBox(ctx, anchorX, anchorY, width, height, margin);
  boxes.push(fallback);
  ctx._labelBoxes = boxes;
  return [fallback.x, fallback.y + height];
}

function labelBox(ctx, x, baselineY, width, height, margin) {
  const left = Math.max(margin, Math.min(x, ctx.canvas.width - width - margin));
  const top = Math.max(margin, Math.min(baselineY - height, ctx.canvas.height - height - margin));
  return {
    x: left,
    y: top,
    w: width,
    h: height,
  };
}

function boxesOverlap(a, b) {
  const pad = 3;
  return !(
    a.x + a.w + pad < b.x ||
    b.x + b.w + pad < a.x ||
    a.y + a.h + pad < b.y ||
    b.y + b.h + pad < a.y
  );
}

async function loadModel(id) {
  const payload = await api(`/api/model?id=${encodeURIComponent(id)}`);
  state.model = payload.model;
  state.rows = payload.rows;
  state.selected = null;
  const conf = Number(payload.thresholds?.confidence ?? 0);
  els.confSlider.value = String(Math.max(0, Math.min(1, conf)));
  updateConfidenceLabel();
  const archLabel = state.model?.architecture
    ? ` · ${state.model.architecture}`
    : "";
  els.subtitle.textContent = `${modelLabel(state.model)}${archLabel}`;
  renderStats();
  applyFilters();
  if (state.rows.length) {
    await selectRow(state.rows[0].index);
  }
}

async function init() {
  const payload = await api("/api/models");
  state.models = payload.models;
  state.classes = payload.classes;
  els.modelSelect.innerHTML = state.models.map((model) => {
    const tag = model.architecture ? ` [${model.architecture}]` : "";
    return `<option value="${escapeHtml(model.id)}">${escapeHtml(modelLabel(model) + tag)}</option>`;
  }).join("");

  if (!state.models.length) {
    els.caseList.innerHTML = '<div class="empty">No error_inspection reports found.</div>';
    return;
  }

  await loadModel(state.models[0].id);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function format(value) {
  return Number(value).toFixed(4);
}

els.modelSelect.addEventListener("change", () => loadModel(els.modelSelect.value));
els.caseList.addEventListener("click", (event) => {
  const card = event.target.closest(".case-card");
  if (card) {
    selectRow(Number(card.dataset.index));
  }
});

for (const input of [els.searchInput, els.filterFp, els.filterFn, els.filterMismatch, els.filterTp, els.filterIssues, els.onlySelected]) {
  input.addEventListener("input", applyFilters);
}

els.rawMode.addEventListener("click", () => setMode("raw"));
els.annotatedMode.addEventListener("click", () => setMode("annotated"));
els.compareMode.addEventListener("click", () => setMode("compare"));
for (const input of [els.showGt, els.showDetections, els.showTp, els.showFp, els.showFn, els.showCentroids, els.showLabels, els.fitImage]) {
  input.addEventListener("input", renderImageMode);
}
els.confSlider.addEventListener("input", () => {
  updateConfidenceLabel();
  drawOverlay();
});

els.rawImage.addEventListener("load", drawOverlay);
window.addEventListener("resize", drawOverlay);

init().catch((error) => {
  els.caseList.innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
});
