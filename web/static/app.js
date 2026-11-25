/**
 * app.js - Scientific Volume Rendering Client Controller
 * Course: IS F311 Computer Graphics
 * Handles mouse orbit navigation, progressive rendering, 2D cross-section scrubbing,
 * parameter controls, and dataset inspection.
 */

// Application State
const state = {
  datasetKey: "brain_mri",
  method: 0,        // 0: Levoy (1988), 1: Drebin (1988), 2: MIP
  compositing: 1,   // 0: Back-to-Front, 1: Front-to-Back (Early Termination)
  stepSize: 0.5,
  width: 320,
  height: 320,
  isInteracting: false,

  // Camera Orbit
  camera: {
    azimuth: 45.0,
    elevation: 25.0,
    distance: 2.5,
    target: [0.0, 0.0, 0.0],
    fov: 40.0,
  },

  // Levoy (1988) Parameters
  levoy: {
    iso1: 0.32,
    thick1: 1.5,
    alpha1: 0.75,
    color1: [0.92, 0.76, 0.65],
    iso2: 0.60,
    thick2: 2.0,
    alpha2: 0.90,
    color2: [0.95, 0.95, 0.90],
  },

  // Drebin (1988) Parameters
  drebin: {
    smin: 0.15,
    sp1: 0.30,
    sp2: 0.55,
    smax: 0.85,
  },

  // Clipping & Cutaways
  clipping: {
    enabled: false,
    depth: 0.0,
    wedgeCut: false,
  },

  // Lighting
  lighting: {
    ka: 0.20,
    kd: 0.65,
    ks: 0.35,
    shininess: 30.0,
  },

  // 2D Orthogonal Slice
  slice: {
    axis: "axial",
    pct: 0.5,
  },

  lastRenderDataUrl: "",
};

// DOM References
const renderView = document.getElementById("renderView");
const sliceView = document.getElementById("sliceView");
const canvasContainer = document.getElementById("canvasContainer");
const loadingIndicator = document.getElementById("loadingIndicator");

const datasetSelect = document.getElementById("datasetSelect");
const compositingSelect = document.getElementById("compositingSelect");
const statusVolume = document.getElementById("statusVolume");
const statusFps = document.getElementById("statusFps");
const statusDim = document.getElementById("statusDim");
const camInfo = document.getElementById("camInfo");

// Modal Elements
const btnDatasetInfo = document.getElementById("btnDatasetInfo");
const datasetModal = document.getElementById("datasetModal");
const btnModalClose = document.getElementById("btnModalClose");
const modalDatasetKey = document.getElementById("modalDatasetKey");
const modalTitle = document.getElementById("modalTitle");
const modalSubject = document.getElementById("modalSubject");
const modalCreator = document.getElementById("modalCreator");
const modalModality = document.getElementById("modalModality");
const modalShape = document.getElementById("modalShape");
const modalStructures = document.getElementById("modalStructures");
const modalSignificance = document.getElementById("modalSignificance");
const modalRecommended = document.getElementById("modalRecommended");

// Algorithm Selection
const btnLevoy = document.getElementById("btnLevoy");
const btnDrebin = document.getElementById("btnDrebin");
const btnMIP = document.getElementById("btnMIP");
const levoySection = document.getElementById("levoySection");
const drebinSection = document.getElementById("drebinSection");

const btnResetCamera = document.getElementById("btnResetCamera");
const btnViewFront = document.getElementById("btnViewFront");
const btnViewSide = document.getElementById("btnViewSide");
const btnViewTop = document.getElementById("btnViewTop");

// Slice Controls
const sliceSlider = document.getElementById("sliceSlider");
const slicePosText = document.getElementById("slicePosText");
const sliceAxisLabel = document.getElementById("sliceAxisLabel");
const sliceBtns = document.querySelectorAll(".btn-slice");

// Parameter Sliders
const stepSlider = document.getElementById("stepSlider");
const stepVal = document.getElementById("stepVal");

const iso1Slider = document.getElementById("iso1Slider");
const iso1Val = document.getElementById("iso1Val");
const thick1Slider = document.getElementById("thick1Slider");
const thick1Val = document.getElementById("thick1Val");
const alpha1Slider = document.getElementById("alpha1Slider");
const alpha1Val = document.getElementById("alpha1Val");

const iso2Slider = document.getElementById("iso2Slider");
const iso2Val = document.getElementById("iso2Val");
const thick2Slider = document.getElementById("thick2Slider");
const thick2Val = document.getElementById("thick2Val");
const alpha2Slider = document.getElementById("alpha2Slider");
const alpha2Val = document.getElementById("alpha2Val");

const drebinMinSlider = document.getElementById("drebinMinSlider");
const drebinMinVal = document.getElementById("drebinMinVal");
const drebinPeakSlider = document.getElementById("drebinPeakSlider");
const drebinPeakVal = document.getElementById("drebinPeakVal");
const drebinMaxSlider = document.getElementById("drebinMaxSlider");
const drebinMaxVal = document.getElementById("drebinMaxVal");

const clipPlaneToggle = document.getElementById("clipPlaneToggle");
const clipPlaneGroup = document.getElementById("clipPlaneGroup");
const clipDSlider = document.getElementById("clipDSlider");
const clipDVal = document.getElementById("clipDVal");
const wedgeCutToggle = document.getElementById("wedgeCutToggle");

const kdSlider = document.getElementById("kdSlider");
const kdVal = document.getElementById("kdVal");
const ksSlider = document.getElementById("ksSlider");
const ksVal = document.getElementById("ksVal");
const shineSlider = document.getElementById("shineSlider");
const shineVal = document.getElementById("shineVal");

// Dataset Catalog & Presets
let datasetsCatalog = {};

async function loadDatasetsCatalog() {
  try {
    const res = await fetch("/api/datasets");
    if (res.ok) {
      const data = await res.json();
      data.datasets.forEach(ds => {
        datasetsCatalog[ds.key] = ds;
      });
      updateDatasetModalContent(state.datasetKey);
    }
  } catch (err) {
    console.error("Failed to load datasets catalog:", err);
  }
}

function updateDatasetModalContent(key) {
  const meta = datasetsCatalog[key];
  if (!meta) return;

  if (modalDatasetKey) modalDatasetKey.textContent = key.toUpperCase();
  if (modalTitle) modalTitle.textContent = meta.name || key;
  if (modalSubject) modalSubject.textContent = meta.subject || "Not specified";
  if (modalCreator) modalCreator.textContent = meta.creator || "Scientific Archive";
  if (modalModality) modalModality.textContent = meta.modality || "Volumetric CT / MRI";
  if (modalShape) modalShape.textContent = `${meta.shape || ""} (${meta.spacing || "isotropic"})`;
  if (modalStructures) modalStructures.textContent = meta.structures || "Internal and surface structures.";
  if (modalSignificance) modalSignificance.textContent = meta.significance || "Volume rendering evaluation.";
  if (modalRecommended) modalRecommended.textContent = meta.recommended || "Levoy (1988) or Drebin (1988)";

  if (statusVolume) statusVolume.textContent = meta.name || key;
  if (statusDim) statusDim.textContent = meta.shape ? meta.shape.replace(/ voxels.*/, "") : "";
}

const DATASET_PRESETS = {
  brain_mri: {
    method: 0,
    iso1: 0.32, thick1: 1.5, alpha1: 0.75,
    iso2: 0.60, thick2: 2.0, alpha2: 0.90,
  },
  bonsai: {
    method: 1,
    drebinMin: 0.12, drebinPeak: 0.35, drebinMax: 0.75,
  },
  engine: {
    method: 0,
    iso1: 0.35, thick1: 1.5, alpha1: 0.95,
    iso2: 0.70, thick2: 2.0, alpha2: 0.95,
  },
  avm: {
    method: 0,
    iso1: 0.22, thick1: 1.5, alpha1: 0.85,
    iso2: 0.55, thick2: 2.0, alpha2: 0.95,
  },
  brain_flair: {
    method: 0,
    iso1: 0.28, thick1: 1.5, alpha1: 0.75,
    iso2: 0.55, thick2: 2.0, alpha2: 0.90,
  },
  marschner_lobb: {
    method: 0,
    iso1: 0.50, thick1: 1.5, alpha1: 0.95,
    iso2: 0.80, thick2: 2.0, alpha2: 0.90,
  },
};

// Rendering Dispatcher
let isRendering = false;
let pendingRender = false;

async function triggerRender(interactive = false) {
  if (isRendering) {
    pendingRender = true;
    return;
  }
  isRendering = true;
  if (loadingIndicator) loadingIndicator.classList.add("active");

  const width = interactive ? 200 : 320;
  const height = interactive ? 200 : 320;
  const step = interactive ? Math.max(state.stepSize, 0.8) : state.stepSize;

  const payload = {
    dataset_key: state.datasetKey,
    method: state.method,
    compositing: state.compositing,
    step_size: step,
    width: width,
    height: height,
    azimuth: state.camera.azimuth,
    elevation: state.camera.elevation,
    distance: state.camera.distance,
    target_x: state.camera.target[0],
    target_y: state.camera.target[1],
    target_z: state.camera.target[2],
    fov_y: state.camera.fov,
    levoy_iso1: state.levoy.iso1,
    levoy_thick1: state.levoy.thick1,
    levoy_alpha1: state.levoy.alpha1,
    levoy_color1: state.levoy.color1,
    levoy_iso2: state.levoy.iso2,
    levoy_thick2: state.levoy.thick2,
    levoy_alpha2: state.levoy.alpha2,
    levoy_color2: state.levoy.color2,
    drebin_smin: state.drebin.smin,
    drebin_sp1: state.drebin.sp1,
    drebin_sp2: state.drebin.sp2,
    drebin_smax: state.drebin.smax,
    clip_enabled: state.clipping.enabled,
    clip_plane_d: state.clipping.depth,
    wedge_cut: state.clipping.wedgeCut,
    ka: state.lighting.ka,
    kd: state.lighting.kd,
    ks: state.lighting.ks,
    shininess: state.lighting.shininess,
  };

  try {
    const res = await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.ok) {
      const data = await res.json();
      if (renderView) renderView.src = data.image;
      state.lastRenderDataUrl = data.image;
      if (statusFps) statusFps.textContent = `${data.render_time_ms} ms (${data.fps} fps)`;
    }
  } catch (err) {
    console.error("Render dispatch error:", err);
  } finally {
    isRendering = false;
    if (loadingIndicator) loadingIndicator.classList.remove("active");
    if (pendingRender) {
      pendingRender = false;
      triggerRender(state.isInteracting);
    }
  }
}

// 2D Orthogonal Slice Retrieval
async function fetchSlice() {
  try {
    const res = await fetch(`/api/slice?axis=${state.slice.axis}&index_pct=${state.slice.pct}`);
    if (res.ok) {
      const data = await res.json();
      if (sliceView) sliceView.src = data.image;
      if (slicePosText) slicePosText.textContent = `${data.index} / ${data.max_index}`;
    }
  } catch (err) {
    console.error("Slice fetch error:", err);
  }
}

// HUD Camera Coordinates
function updateCameraInfo() {
  if (camInfo) {
    camInfo.textContent = `Az: ${Math.round(state.camera.azimuth)}° | El: ${Math.round(state.camera.elevation)}° | Dist: ${state.camera.distance.toFixed(2)}`;
  }
}

// Mouse Navigation (Orbit & Zoom)
let isDragging = false;
let startX = 0;
let startY = 0;
let mouseButton = 0;

if (canvasContainer) {
  canvasContainer.addEventListener("mousedown", (e) => {
    e.preventDefault();
    isDragging = true;
    state.isInteracting = true;
    startX = e.clientX;
    startY = e.clientY;
    mouseButton = e.button;
  });

  canvasContainer.addEventListener("contextmenu", (e) => e.preventDefault());

  canvasContainer.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.08 : 0.92;
    state.camera.distance = Math.max(0.5, Math.min(6.0, state.camera.distance * factor));
    updateCameraInfo();
    triggerRender(false);
  });
}

window.addEventListener("mousemove", (e) => {
  if (!isDragging) return;

  const dx = e.clientX - startX;
  const dy = e.clientY - startY;
  startX = e.clientX;
  startY = e.clientY;

  if (mouseButton === 0) {
    // Left drag: Orbit
    state.camera.azimuth = (state.camera.azimuth + dx * 0.75) % 360.0;
    state.camera.elevation = Math.max(-85.0, Math.min(85.0, state.camera.elevation - dy * 0.75));
    updateCameraInfo();
    triggerRender(true);
  } else if (mouseButton === 2) {
    // Right drag: Zoom
    state.camera.distance = Math.max(0.5, Math.min(6.0, state.camera.distance + dy * 0.01));
    updateCameraInfo();
    triggerRender(true);
  }
});

window.addEventListener("mouseup", () => {
  if (isDragging) {
    isDragging = false;
    state.isInteracting = false;
    triggerRender(false);
  }
});

if (btnResetCamera) {
  btnResetCamera.addEventListener("click", () => {
    state.camera.azimuth = 45.0;
    state.camera.elevation = 25.0;
    state.camera.distance = 2.5;
    state.camera.target = [0.0, 0.0, 0.0];
    updateCameraInfo();
    triggerRender(false);
  });
}

if (btnViewFront) {
  btnViewFront.addEventListener("click", () => {
    state.camera.azimuth = 0.0;
    state.camera.elevation = 0.0;
    state.camera.distance = 2.5;
    updateCameraInfo();
    triggerRender(false);
  });
}

if (btnViewSide) {
  btnViewSide.addEventListener("click", () => {
    state.camera.azimuth = 90.0;
    state.camera.elevation = 0.0;
    state.camera.distance = 2.5;
    updateCameraInfo();
    triggerRender(false);
  });
}

if (btnViewTop) {
  btnViewTop.addEventListener("click", () => {
    state.camera.azimuth = 0.0;
    state.camera.elevation = 89.9;
    state.camera.distance = 2.5;
    updateCameraInfo();
    triggerRender(false);
  });
}

// Algorithm Switching
function setMethod(methodIdx) {
  state.method = methodIdx;
  [btnLevoy, btnDrebin, btnMIP].forEach((btn, idx) => {
    if (btn) btn.classList.toggle("active", idx === methodIdx);
  });

  if (methodIdx === 0) {
    if (levoySection) levoySection.classList.remove("hidden");
    if (drebinSection) drebinSection.classList.add("hidden");
  } else if (methodIdx === 1) {
    if (levoySection) levoySection.classList.add("hidden");
    if (drebinSection) drebinSection.classList.remove("hidden");
  } else {
    if (levoySection) levoySection.classList.add("hidden");
    if (drebinSection) drebinSection.classList.add("hidden");
  }
  triggerRender(false);
}

if (btnLevoy) btnLevoy.addEventListener("click", () => setMethod(0));
if (btnDrebin) btnDrebin.addEventListener("click", () => setMethod(1));
if (btnMIP) btnMIP.addEventListener("click", () => setMethod(2));

// Dataset Selector
if (datasetSelect) {
  datasetSelect.addEventListener("change", async (e) => {
    const key = e.target.value;
    state.datasetKey = key;
    updateDatasetModalContent(key);

    if (DATASET_PRESETS[key]) {
      const p = DATASET_PRESETS[key];
      if (p.method !== undefined) setMethod(p.method);
      if (p.iso1 !== undefined) {
        state.levoy.iso1 = p.iso1;
        if (iso1Slider) iso1Slider.value = p.iso1;
        if (iso1Val) iso1Val.textContent = p.iso1;
      }
    }

    try {
      const res = await fetch("/api/load_dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: key }),
      });
      if (res.ok) {
        await fetchSlice();
        await triggerRender(false);
      }
    } catch (err) {
      console.error("Dataset load error:", err);
    }
  });
}

// Compositing Selector
if (compositingSelect) {
  compositingSelect.addEventListener("change", (e) => {
    state.compositing = parseInt(e.target.value, 10);
    triggerRender(false);
  });
}

// Slider Binding Helper
function bindSlider(slider, display, onChange) {
  if (!slider) return;
  slider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    if (display) display.textContent = val;
    onChange(val);
    triggerRender(true);
  });
  slider.addEventListener("change", () => triggerRender(false));
}

bindSlider(stepSlider, stepVal, (v) => state.stepSize = v);
bindSlider(iso1Slider, iso1Val, (v) => state.levoy.iso1 = v);
bindSlider(thick1Slider, thick1Val, (v) => state.levoy.thick1 = v);
bindSlider(alpha1Slider, alpha1Val, (v) => state.levoy.alpha1 = v);
bindSlider(iso2Slider, iso2Val, (v) => state.levoy.iso2 = v);
bindSlider(thick2Slider, thick2Val, (v) => state.levoy.thick2 = v);
bindSlider(alpha2Slider, alpha2Val, (v) => state.levoy.alpha2 = v);

bindSlider(drebinMinSlider, drebinMinVal, (v) => state.drebin.smin = v);
bindSlider(drebinPeakSlider, drebinPeakVal, (v) => state.drebin.sp1 = v);
bindSlider(drebinMaxSlider, drebinMaxVal, (v) => state.drebin.smax = v);

bindSlider(kdSlider, kdVal, (v) => state.lighting.kd = v);
bindSlider(ksSlider, ksVal, (v) => state.lighting.ks = v);
bindSlider(shineSlider, shineVal, (v) => state.lighting.shininess = v);

// Cutaway & Clipping Toggles
if (clipPlaneToggle) {
  clipPlaneToggle.addEventListener("change", (e) => {
    state.clipping.enabled = e.target.checked;
    if (clipPlaneGroup) clipPlaneGroup.style.display = e.target.checked ? "block" : "none";
    triggerRender(false);
  });
}

if (clipDSlider) {
  clipDSlider.addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    if (clipDVal) clipDVal.textContent = v;
    state.clipping.depth = v;
    triggerRender(true);
  });
  clipDSlider.addEventListener("change", () => triggerRender(false));
}

if (wedgeCutToggle) {
  wedgeCutToggle.addEventListener("change", (e) => {
    state.clipping.wedgeCut = e.target.checked;
    triggerRender(false);
  });
}

// Slice Navigation
const AXIS_MAP = {
  axial: "Axial (Z-axis)",
  coronal: "Coronal (Y-axis)",
  sagittal: "Sagittal (X-axis)",
};

sliceBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    sliceBtns.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.slice.axis = btn.dataset.axis;
    if (sliceAxisLabel) {
      sliceAxisLabel.textContent = AXIS_MAP[btn.dataset.axis] || btn.dataset.axis;
    }
    fetchSlice();
  });
});

if (sliceSlider) {
  sliceSlider.addEventListener("input", (e) => {
    state.slice.pct = parseFloat(e.target.value) / 100.0;
    fetchSlice();
  });
}

// Dataset Info Modal Logic
if (btnDatasetInfo && datasetModal) {
  btnDatasetInfo.addEventListener("click", () => {
    datasetModal.classList.remove("hidden");
  });
}

if (btnModalClose && datasetModal) {
  btnModalClose.addEventListener("click", () => {
    datasetModal.classList.add("hidden");
  });
}

if (datasetModal) {
  datasetModal.addEventListener("click", (e) => {
    if (e.target === datasetModal) {
      datasetModal.classList.add("hidden");
    }
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !datasetModal.classList.contains("hidden")) {
      datasetModal.classList.add("hidden");
    }
  });
}

// Bootstrap Initialization
window.addEventListener("DOMContentLoaded", async () => {
  updateCameraInfo();
  await loadDatasetsCatalog();
  await fetchSlice();
  await triggerRender(false);
});
