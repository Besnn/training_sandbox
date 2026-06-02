const state = {
  models: [],
  model: null,
  records: [],
  filtered: [],
  selected: null,
  classes: [],
  classEnabled: new Map(),
  statusEnabled: new Map(),
};

const statusNames = ["TP", "FP", "FN", "CLASS_MISMATCH", "DET"];
const colors = {
  TP: "#2563eb",
  FP: "#dc2626",
  FN: "#e87916",
  CLASS_MISMATCH: "#b7791f",
  DET: "#7c3aed",
};

const els = {
  modelSelect: document.querySelector("#modelSelect"),
  subtitle: document.querySelector("#subtitle"),
  searchInput: document.querySelector("#searchInput"),
  confSlider: document.querySelector("#confSlider"),
  confValue: document.querySelector("#confValue"),
  classToggles: document.querySelector("#classToggles"),
  statusToggles: document.querySelector("#statusToggles"),
  showLabels: document.querySelector("#showLabels"),
  fitImage: document.querySelector("#fitImage"),
  stats: document.querySelector("#stats"),
  imageList: document.querySelector("#imageList"),
  imageTitle: document.querySelector("#imageTitle"),
  imageMeta: document.querySelector("#imageMeta"),
  legend: document.querySelector("#legend"),
  imageFrame: document.querySelector("#imageFrame"),
  rawImage: document.querySelector("#rawImage"),
  overlay: document.querySelector("#overlay"),
};

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function assetUrl(record) {
  const query = new URLSearchParams({
    model: state.model.id,
    path: record.rawImage || "",
    image: record.image,
  });
  return `/asset/raw?${query.toString()}`;
}

async function loadModel(id) {
  const payload = await api(`/api/model?id=${encodeURIComponent(id)}`);
  state.model = payload.model;
  state.records = payload.records;
  state.classes = payload.classes?.length ? payload.classes : collectClasses(payload.records);
  state.selected = null;

  const conf = Number(payload.thresholds?.confidence ?? 0);
  els.confSlider.value = String(Math.max(0, Math.min(1, conf)));
  updateConfidence();

  state.classEnabled = new Map(state.classes.map((name) => [name, true]));
  state.statusEnabled = new Map(statusNames.map((name) => [name, true]));
  renderMenus();
  renderLegend();
  els.subtitle.textContent = `${state.model.name} · ${state.model.images} images · ${state.model.detections} detections`;
  applyFilters();
  if (state.filtered.length) selectRecord(state.filtered[0].index);
}

function collectClasses(records) {
  return [...new Set(records.flatMap((record) => (
    record.detections || []
  ).map((det) => det.className || String(det.classId))))].sort();
}

function renderMenus() {
  els.classToggles.innerHTML = state.classes.map((name) => `
    <label><input type="checkbox" data-class="${escapeHtml(name)}" checked> ${escapeHtml(name)}</label>
  `).join("");
  els.statusToggles.innerHTML = statusNames.map((name) => `
    <label><input type="checkbox" data-status="${name}" checked> ${statusLabel(name)}</label>
  `).join("");
}

function renderLegend() {
  els.legend.innerHTML = statusNames.map((name) => `
    <span class="legend-item"><span class="legend-swatch" style="background:${colors[name]}"></span>${statusLabel(name)}</span>
  `).join("");
}

function statusLabel(status) {
  return status === "CLASS_MISMATCH" ? "Mismatch" : status;
}

function currentConf() {
  return Number(els.confSlider.value || 0);
}

function updateConfidence() {
  els.confValue.textContent = currentConf().toFixed(2);
}

function visibleDetections(record) {
  const conf = currentConf();
  return (record.detections || []).filter((det) => {
    const cls = det.className || String(det.classId);
    const status = det.status || "DET";
    return Number(det.score || 0) >= conf &&
      state.classEnabled.get(cls) !== false &&
      state.statusEnabled.get(status) !== false;
  });
}

function passesSearch(record) {
  const query = els.searchInput.value.trim().toLowerCase();
  return !query || record.image.toLowerCase().includes(query);
}

function applyFilters() {
  state.filtered = state.records.filter((record) => passesSearch(record) && visibleDetections(record).length > 0);
  renderStats();
  renderList();
  if (state.selected && !state.filtered.some((record) => record.index === state.selected.index)) {
    state.selected = null;
    if (state.filtered.length) selectRecord(state.filtered[0].index);
  } else {
    drawOverlay();
  }
}

function renderStats() {
  const detections = state.filtered.reduce((sum, record) => sum + visibleDetections(record).length, 0);
  els.stats.innerHTML = [
    ["Images", state.filtered.length],
    ["Shown", detections],
    ["All", state.records.reduce((sum, record) => sum + (record.detections || []).length, 0)],
  ].map(([label, value]) => `<div class="stat"><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function renderList() {
  if (!state.filtered.length) {
    els.imageList.innerHTML = '<div class="empty">No images match the current detection filters.</div>';
    return;
  }
  els.imageList.innerHTML = state.filtered.map((record) => {
    const active = state.selected?.index === record.index ? " active" : "";
    const count = visibleDetections(record).length;
    return `
      <button class="image-card${active}" data-index="${record.index}" type="button">
        <strong>${escapeHtml(record.image)}</strong>
        <span>${count} visible / ${(record.detections || []).length} total detections</span>
      </button>
    `;
  }).join("");
}

function selectRecord(index) {
  const record = state.records.find((item) => item.index === index);
  if (!record) return;
  state.selected = record;
  els.imageTitle.textContent = record.image;
  els.imageMeta.textContent = `${visibleDetections(record).length} visible detections`;
  els.rawImage.src = assetUrl(record);
  renderList();
  drawOverlay();
}

function drawOverlay() {
  const img = els.rawImage;
  const canvas = els.overlay;
  const ctx = canvas.getContext("2d");
  if (!state.selected || !img.complete || !img.naturalWidth) return;

  const displayWidth = img.clientWidth || img.naturalWidth;
  const displayHeight = img.clientHeight || img.naturalHeight;
  canvas.width = displayWidth;
  canvas.height = displayHeight;
  canvas.style.width = `${displayWidth}px`;
  canvas.style.height = `${displayHeight}px`;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx._labelBoxes = [];

  const sx = displayWidth / img.naturalWidth;
  const sy = displayHeight / img.naturalHeight;
  for (const det of visibleDetections(state.selected)) {
    drawDetection(ctx, det, sx, sy);
  }
  els.imageMeta.textContent = `${visibleDetections(state.selected).length} visible detections`;
}

function drawDetection(ctx, det, sx, sy) {
  const status = det.status || "DET";
  const color = colors[status] || colors.DET;
  const points = det.polygon || [];
  if (!points.length) return;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = `${color}22`;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(points[0].x * sx, points[0].y * sy);
  for (const point of points.slice(1)) ctx.lineTo(point.x * sx, point.y * sy);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  if (els.showLabels.checked) {
    const conf = Number(det.score || 0).toFixed(2);
    const iou = sameClassIoU(det);
    drawTag(ctx, `${statusLabel(status)} ${det.className} conf ${conf} IoU ${iou}`, points[0].x * sx, points[0].y * sy, color);
  }
  ctx.restore();
}

function sameClassIoU(det) {
  const ious = det.ious || [];
  if (!ious.length) return "0.00";
  return Number(Math.max(...ious.map((item) => Number(item.iou || 0)))).toFixed(2);
}

function drawTag(ctx, text, x, y, color) {
  ctx.save();
  ctx.font = "12px system-ui, sans-serif";
  const width = ctx.measureText(text).width + 10;
  const height = 20;
  const [tx, ty] = placeLabel(ctx, x, y, width, height);
  ctx.fillStyle = color;
  ctx.fillRect(tx, ty - height, width, height);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, tx + 5, ty - 6);
  ctx.restore();
}

function placeLabel(ctx, anchorX, anchorY, width, height) {
  const boxes = ctx._labelBoxes || [];
  const candidates = [
    [anchorX + 8, anchorY - 8],
    [anchorX + 8, anchorY + height + 12],
    [anchorX - width - 8, anchorY - 8],
    [anchorX - width - 8, anchorY + height + 12],
    [anchorX - width / 2, anchorY - 20],
    [anchorX - width / 2, anchorY + height + 20],
  ];
  for (const candidate of candidates) {
    const box = labelBox(ctx, candidate[0], candidate[1], width, height);
    if (!boxes.some((other) => boxesOverlap(box, other))) {
      boxes.push(box);
      ctx._labelBoxes = boxes;
      return [box.x, box.y + height];
    }
  }
  const fallback = labelBox(ctx, anchorX, anchorY, width, height);
  boxes.push(fallback);
  ctx._labelBoxes = boxes;
  return [fallback.x, fallback.y + height];
}

function labelBox(ctx, x, baselineY, width, height) {
  const margin = 4;
  return {
    x: Math.max(margin, Math.min(x, ctx.canvas.width - width - margin)),
    y: Math.max(margin, Math.min(baselineY - height, ctx.canvas.height - height - margin)),
    w: width,
    h: height,
  };
}

function boxesOverlap(a, b) {
  const pad = 3;
  return !(a.x + a.w + pad < b.x || b.x + b.w + pad < a.x || a.y + a.h + pad < b.y || b.y + b.h + pad < a.y);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

els.modelSelect.addEventListener("change", () => loadModel(els.modelSelect.value));
els.searchInput.addEventListener("input", applyFilters);
els.confSlider.addEventListener("input", () => {
  updateConfidence();
  applyFilters();
});
els.showLabels.addEventListener("input", drawOverlay);
els.fitImage.addEventListener("input", () => {
  els.imageFrame.classList.toggle("fit", els.fitImage.checked);
  drawOverlay();
});
els.classToggles.addEventListener("input", (event) => {
  const input = event.target.closest("input[data-class]");
  if (!input) return;
  state.classEnabled.set(input.dataset.class, input.checked);
  applyFilters();
});
els.statusToggles.addEventListener("input", (event) => {
  const input = event.target.closest("input[data-status]");
  if (!input) return;
  state.statusEnabled.set(input.dataset.status, input.checked);
  applyFilters();
});
els.imageList.addEventListener("click", (event) => {
  const card = event.target.closest(".image-card");
  if (card) selectRecord(Number(card.dataset.index));
});
els.rawImage.addEventListener("load", drawOverlay);
window.addEventListener("resize", drawOverlay);

async function init() {
  const payload = await api("/api/models");
  state.models = payload.models;
  els.modelSelect.innerHTML = state.models.map((model) => (
    `<option value="${escapeHtml(model.id)}">${escapeHtml(model.name)}</option>`
  )).join("");
  if (!state.models.length) {
    els.imageList.innerHTML = '<div class="empty">No inspection.json folders found.</div>';
    return;
  }
  await loadModel(state.models[0].id);
}

init().catch((error) => {
  els.imageList.innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
});
