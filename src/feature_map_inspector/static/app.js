const $ = (id) => document.getElementById(id);

const fileInput  = $("file");
const drop       = $("drop");
const dropText   = $("dropText");
const runBtn     = $("run");
const statusEl   = $("status");
const mapsEl     = $("maps");
const tileRange  = $("tile");
const activeOnly = $("activeOnly");
const familySel  = $("family");
const layerSel   = $("layer");

let selectedFile = null;
let lastTiles = [];

// Populate (or repopulate) the Layer dropdown from the LAYERS_BY_FAMILY constant
// embedded in the HTML by Jinja.
function updateLayerOptions() {
  const family = familySel.value;
  const layers = LAYERS_BY_FAMILY[family] || {};
  const prev   = layerSel.value;
  layerSel.innerHTML = "";
  Object.entries(layers).forEach(([name, label]) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = label;
    if (name === prev) opt.selected = true;
    layerSel.appendChild(opt);
  });
}
function updateBypassVisibility() {
  const show = familySel.value === "stn-fomo";
  $("bypassChk").style.display = show ? "" : "none";
  if (!show) $("bypassStn").checked = false;
}
familySel.addEventListener("change", () => { updateLayerOptions(); updateBypassVisibility(); });
updateLayerOptions();      // populate on page load
updateBypassVisibility();  // sync bypass checkbox visibility

function setFile(f) {
  if (!f || !f.type.startsWith("image/")) return;
  selectedFile = f;
  dropText.textContent = f.name;
  drop.classList.add("has-file");
  runBtn.disabled = false;
}

fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragover"); })
);
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

tileRange.addEventListener("input", () => {
  mapsEl.style.setProperty("--tile", tileRange.value + "px");
});
mapsEl.style.setProperty("--tile", tileRange.value + "px");

activeOnly.addEventListener("change", applyActiveFilter);
function applyActiveFilter() {
  const hide = activeOnly.checked;
  document.querySelectorAll(".tile").forEach((t) => {
    const inactive = t.classList.contains("inactive");
    t.style.display = hide && inactive ? "none" : "";
  });
}

async function run() {
  if (!selectedFile) return;
  runBtn.disabled = true;
  statusEl.textContent = "running inference…";

  const fd = new FormData();
  fd.append("image",      selectedFile);
  fd.append("family",     $("family").value);
  fd.append("precision",  $("precision").value);
  fd.append("layer",      $("layer").value);
  fd.append("colormap",   $("colormap").value);
  fd.append("norm",       $("norm").value);
  fd.append("tile",       tileRange.value);
  fd.append("bypass_stn", $("bypassStn").checked ? "1" : "0");

  const t0 = performance.now();
  try {
    const res = await fetch("/api/feature_maps", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "request failed");
    render(data);
    const ms = (performance.now() - t0).toFixed(0);
    statusEl.textContent =
      `${data.family} · ${data.precision} · ${data.layer} · ${data.num_maps} maps · ${ms} ms`;
  } catch (err) {
    statusEl.textContent = "error: " + err.message;
  } finally {
    runBtn.disabled = false;
  }
}
runBtn.addEventListener("click", run);

function render(data) {
  $("meta").hidden = false;
  $("bypassBanner").hidden = !data.bypass_stn;
  $("preview").src = "data:image/png;base64," + data.input_preview;

  // Detection heatmaps (final per-class output), clickable into the compare view.
  const heat = $("heatmaps");
  heat.innerHTML = "";
  heatmapItems = (data.class_heatmaps || []).map((c) => ({
    png: c.png,
    title: c.name,
    stats: `max prob ${c.max}  ·  mean ${c.mean}`,
  }));
  heatmapItems.forEach((h, i) => {
    const div = document.createElement("div");
    div.className = "heatmap";
    div.innerHTML =
      `<img src="data:image/png;base64,${h.png}" alt="${h.title}" />` +
      `<div class="hcap"><b>${h.title}</b><span>${data.class_heatmaps[i].max.toFixed(
        2
      )}</span></div>`;
    div.addEventListener("click", () => openItems(heatmapItems, i, "heatmap"));
    heat.appendChild(div);
  });

  const scores = $("scores");
  scores.innerHTML = "";
  data.class_scores.forEach((c) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${c.name}</span><span class="v">${c.max.toFixed(3)}</span>`;
    scores.appendChild(li);
  });

  $("layerInfo").innerHTML =
    `<b>${data.layer}</b><br>${data.layer_label}<br>` +
    `grid ${data.grid}×${data.grid} · range [${data.global_min}, ${data.global_max}]`;

  mapsEl.innerHTML = "";
  lastTiles = data.tiles;
  featureItems = data.tiles.map((t) => ({
    png: t.png,
    title: `channel ${t.idx}`,
    stats: `min ${t.min}  ·  max ${t.max}  ·  mean ${t.mean}  ·  ${
      t.active ? "active" : "inactive"
    }`,
    active: t.active,
  }));
  data.tiles.forEach((t, i) => {
    const div = document.createElement("div");
    div.className = "tile" + (t.active ? "" : " inactive");
    div.dataset.idx = i;
    div.innerHTML =
      `<img src="data:image/png;base64,${t.png}" alt="ch ${t.idx}" />` +
      `<div class="cap">#${t.idx} ${t.max.toFixed(2)}</div>`;
    div.addEventListener("click", () => openItems(featureItems, i, "feature"));
    mapsEl.appendChild(div);
  });
  applyActiveFilter();

  renderRotationDiff(data, data.bypass_stn);
}

/* ---- "most affected by the STN warp" (add_7 vs grid_sampler) ---------- */
let rotationDiffItems = [];

function renderRotationDiff(data, bypassActive) {
  const sec    = $("rotationDiff");
  const grid   = $("rotationDiffGrid");
  const dispEl = $("rdDisplacement");
  grid.innerHTML = "";
  rotationDiffItems = [];

  const rd = data.rotation_diff;
  if (!rd || !rd.items || !rd.items.length) {
    sec.hidden = true;
    dispEl.hidden = true;
    dispEl.innerHTML = "";
    return;
  }
  sec.hidden = false;

  if (rd.displacement_png) {
    dispEl.hidden = false;
    const dispCaption = bypassActive
      ? `<b>|δ(x)|</b> — the STN's <em>intended</em> warp for this image ` +
        `(from layer <code>${rd.grid_layer}</code>), in output-pixel units. ` +
        `<b>This correction was NOT applied</b> — the identity grid was used instead, ` +
        `so |Δ| in the cards below is ~0.`
      : `<b>|δ(x)|</b> — the STN's actual per-image warp displacement field for this input ` +
        `(reconstructed as <code>actual sampling grid − identity grid</code> from layer ` +
        `<code>${rd.grid_layer}</code>, in output-pixel units). Identical for every channel — ` +
        `multiplying it by each channel's own |∇before| gives the first-order/Taylor prediction of |Δ| ` +
        `shown as the 4th tile in every card below.`;
    dispEl.innerHTML =
      `<figure><img src="data:image/png;base64,${rd.displacement_png}" alt="warp displacement field" />` +
      `<figcaption>${dispCaption}</figcaption></figure>`;
  } else {
    dispEl.hidden = true;
    dispEl.innerHTML = "";
  }

  const corrBits = [];
  if (rd.mean_grad_corr != null)
    corrBits.push(`|∇before| alone → mean corr <b>${rd.mean_grad_corr}</b>`);
  if (rd.mean_pred_corr != null)
    corrBits.push(`|∇before|·|δ(x)| → mean corr <b>${rd.mean_pred_corr}</b>`);

  const bypassNote = bypassActive
    ? `<span class="rd-bypass-note">STN bypassed — grid_sampler received an identity grid, ` +
      `so before ≈ after and |Δ| ≈ 0. The displacement field shows what the STN <em>would have</em> applied.</span><br>`
    : "";
  $("rdSubtitle").innerHTML =
    bypassNote +
    `${rd.before_layer} (before the warp) → ${rd.after_layer} (after) · ` +
    `top ${rd.items.length} channels ranked by mean |Δ| per pixel · ` +
    `before/after share one colour scale, |Δ| and the prediction have their own<br>` +
    `<span class="rd-hint">small-warp / Taylor model: |Δ(x)| ≈ |∇before(x)| · |δ(x)| — ` +
    `Pearson correlation with the actual |Δ|, pooled across these channels: ` +
    (corrBits.length ? corrBits.join("  ·  ") : "n/a") +
    `</span>`;

  rd.items.forEach((it) => {
    const predStats = (it.pred_corr != null)
      ? `corr(|Δ|, prediction) = <b>${it.pred_corr}</b>` +
        (it.grad_corr != null && it.grad_corr !== it.pred_corr
          ? `  (vs. ${it.grad_corr} for |∇before| alone — displacement-weighting ` +
            `${it.pred_corr > it.grad_corr ? "helps" : "doesn't help"} here)`
          : "")
      : "corr: n/a (flat channel)";
    const quad = [
      { png: it.before_png, title: `channel ${it.idx} — before (${rd.before_layer})`,
        stats: `shared range [${it.shared_min}, ${it.shared_max}]` },
      { png: it.after_png, title: `channel ${it.idx} — after (${rd.after_layer})`,
        stats: `shared range [${it.shared_min}, ${it.shared_max}]` },
      { png: it.diff_png, title: `channel ${it.idx} — |Δ| (actual, before vs after)`,
        stats: `mean |Δ| ${it.diff_mean}  ·  max |Δ| ${it.diff_max}` },
      { png: it.pred_png, title: `channel ${it.idx} — |∇before|·|δ(x)|  (predicted |Δ|, first-order/Taylor model)`,
        stats: predStats },
    ];
    const base = rotationDiffItems.length;
    rotationDiffItems.push(...quad);

    const card = document.createElement("div");
    card.className = "rd-card";
    card.innerHTML =
      `<div class="rd-head"><span>channel <b>#${it.idx}</b></span>` +
      `<span>mean |Δ| ${it.diff_mean} · max |Δ| ${it.diff_max}` +
      (it.pred_corr != null ? ` · corr w/ prediction ${it.pred_corr}` : "") +
      `</span></div>` +
      `<div class="rd-quad">` +
      `<figure><img src="data:image/png;base64,${it.before_png}" alt="before"><figcaption>before</figcaption></figure>` +
      `<figure><img src="data:image/png;base64,${it.after_png}" alt="after"><figcaption>after</figcaption></figure>` +
      `<figure><img src="data:image/png;base64,${it.diff_png}" alt="diff"><figcaption>|Δ| actual</figcaption></figure>` +
      `<figure><img src="data:image/png;base64,${it.pred_png}" alt="predicted diff"><figcaption>|Δ| predicted</figcaption></figure>` +
      `</div>`;
    card.querySelectorAll("figure").forEach((fig, j) =>
      fig.addEventListener("click", () => openItems(rotationDiffItems, base + j, "rotation"))
    );
    grid.appendChild(card);
  });
}

/* ---- selection + comparison (generic over feature maps & heatmaps) ----- */
let featureItems = [];
let heatmapItems = [];
let cmpList = [];   // currently-open list of items
let cmpKind = "feature";
let selectedIdx = -1;
let origImg = null; // preloaded original-input Image
let mapImg = null; // preloaded selected item Image
let currentMode = "overlay";

function openItems(list, i, kind) {
  cmpList = list;
  cmpKind = kind;
  selectedIdx = i;
  // highlight the selected feature tile in the grid (heatmaps aren't in the grid)
  document.querySelectorAll(".tile").forEach((t) =>
    t.classList.toggle("selected", kind === "feature" && Number(t.dataset.idx) === i)
  );
  openCompare(list[i]);
}

function openCompare(item) {
  // Redraw whenever EITHER image finishes decoding, so the overlay never
  // renders the map alone on black (the source of the dimming bug).
  origImg = new Image();
  mapImg = new Image();
  origImg.onload = onImgReady;
  mapImg.onload = onImgReady;
  origImg.src = $("preview").src; // 480x480 input preview
  mapImg.src = "data:image/png;base64," + item.png;

  $("cmpTitle").textContent = item.title;
  $("cmpStats").textContent = item.stats;
  $("sideOrig").src = origImg.src;
  $("sideMap").src = mapImg.src;
  $("soloMap").src = mapImg.src;
  $("compare").hidden = false;
  setMode(currentMode);
}

function onImgReady() {
  if (currentMode === "overlay") drawOverlay();
}

function drawOverlay() {
  const c = $("overlayCanvas");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  if (origImg && origImg.complete && origImg.naturalWidth)
    ctx.drawImage(origImg, 0, 0, c.width, c.height);
  if (mapImg && mapImg.complete && mapImg.naturalWidth) {
    ctx.globalAlpha = Number($("opacity").value) / 100;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(mapImg, 0, 0, c.width, c.height);
    ctx.globalAlpha = 1;
  }
}

function setMode(mode) {
  currentMode = mode;
  document
    .querySelectorAll("#cmpModes button")
    .forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $("overlayCanvas").hidden = mode !== "overlay";
  $("opacityWrap").hidden = mode !== "overlay";
  $("sideView").hidden = mode !== "side";
  $("soloView").hidden = mode !== "solo";
  if (mode === "overlay") drawOverlay();
}

function stepSelection(delta) {
  if (!cmpList.length) return;
  let i = selectedIdx;
  // honor the "active only" filter when stepping through feature maps
  const hideInactive = cmpKind === "feature" && activeOnly.checked;
  for (let n = 0; n < cmpList.length; n++) {
    i = (i + delta + cmpList.length) % cmpList.length;
    if (!hideInactive || cmpList[i].active) break;
  }
  openItems(cmpList, i, cmpKind);
}

$("opacity").addEventListener("input", drawOverlay);
$("cmpModes").addEventListener("click", (e) => {
  if (e.target.dataset.mode) setMode(e.target.dataset.mode);
});
$("cmpClose").addEventListener("click", () => ($("compare").hidden = true));
$("compare").addEventListener("click", (e) => {
  if (e.target.id === "compare") $("compare").hidden = true;
});
document.addEventListener("keydown", (e) => {
  if ($("compare").hidden) return;
  if (e.key === "Escape") $("compare").hidden = true;
  else if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); stepSelection(1); }
  else if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); stepSelection(-1); }
});
