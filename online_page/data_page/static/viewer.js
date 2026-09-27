import * as THREE from "/static/vendor/three.module.js?v=20260901-5";
import { OrbitControls } from "/static/vendor/OrbitControls.js?v=20260901-5";

const DATASET_LABELS = { humanml3d: "HumanML3D", sonic: "SONIC", motionx: "MotionX" };
const CHAINS = [
  [0, 2, 5, 8, 11],
  [0, 1, 4, 7, 10],
  [0, 3, 6, 9, 12, 15],
  [9, 14, 17, 19, 21],
  [9, 13, 16, 18, 20],
];
const EDGES = CHAINS.flatMap((chain) => chain.slice(0, -1).map((joint, index) => [joint, chain[index + 1]]));

const state = {
  summary: null,
  dataset: "humanml3d",
  query: "",
  filter: "all",
  samples: [],
  total: 0,
  selectedId: null,
  detail: null,
  positions: null,
  frames: 0,
  fps: 20,
  currentFrame: 0,
  frameCursor: 0,
  floorOffset: 0,
  playing: false,
  speed: 1,
  loop: true,
  lastTimestamp: performance.now(),
  viewMode: "skeleton",
  textMode: "base",
  textQuery: "",
  textSerial: 0,
  listSerial: 0,
  selectionSerial: 0,
  smplhAvailable: false,
  smplReady: false,
  smplFailed: false,
};

const elements = {
  serverDot: document.getElementById("serverDot"),
  serverStatus: document.getElementById("serverStatus"),
  resultCount: document.getElementById("resultCount"),
  datasetButtons: [...document.querySelectorAll(".dataset-button")],
  datasetTotal: document.getElementById("datasetTotal"),
  datasetDuration: document.getElementById("datasetDuration"),
  sampleSearch: document.getElementById("sampleSearch"),
  assetFilter: document.getElementById("assetFilter"),
  sampleList: document.getElementById("sampleList"),
  loadMore: document.getElementById("loadMore"),
  activeDataset: document.getElementById("activeDataset"),
  activeSample: document.getElementById("activeSample"),
  sampleMeta: document.getElementById("sampleMeta"),
  meshState: document.getElementById("meshState"),
  viewButtons: [...document.querySelectorAll(".view-button")],
  viewport: document.getElementById("viewport"),
  viewportEmpty: document.getElementById("viewportEmpty"),
  viewportBadges: document.getElementById("viewportBadges"),
  cameraButtons: [...document.querySelectorAll("[data-camera]")],
  resetCamera: document.getElementById("resetCamera"),
  prevFrame: document.getElementById("prevFrame"),
  playToggle: document.getElementById("playToggle"),
  nextFrame: document.getElementById("nextFrame"),
  frameCount: document.getElementById("frameCount"),
  frameSlider: document.getElementById("frameSlider"),
  speedSelect: document.getElementById("speedSelect"),
  loopToggle: document.getElementById("loopToggle"),
  motionFacts: document.getElementById("motionFacts"),
  sourceVideo: document.getElementById("sourceVideo"),
  videoWrap: document.querySelector(".video-wrap"),
  videoEmpty: document.getElementById("videoEmpty"),
  videoState: document.getElementById("videoState"),
  syncNote: document.getElementById("syncNote"),
  textTabs: [...document.querySelectorAll(".text-tab")],
  textPanel: document.querySelector(".text-panel"),
  textState: document.getElementById("textState"),
  textSearch: document.getElementById("textSearch"),
  copyText: document.getElementById("copyText"),
  expandText: document.getElementById("expandText"),
  textContent: document.getElementById("textContent"),
  evidenceList: document.getElementById("evidenceList"),
  toast: document.getElementById("toast"),
};

const numberFormatter = new Intl.NumberFormat("zh-CN");
let toastTimer;
let searchTimer;
let textSearchTimer;

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove("is-visible"), 3400);
}

async function requestJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function updateDatasetMeta() {
  if (!state.summary) return;
  const dataset = state.summary.datasets[state.dataset];
  elements.datasetTotal.textContent = numberFormatter.format(dataset.motion_files);
  elements.datasetDuration.textContent = `${dataset.duration_hours.toFixed(2)} HOURS`;
  elements.activeDataset.textContent = DATASET_LABELS[state.dataset].toUpperCase();
}

function assetChip(label, className = "") {
  const chip = document.createElement("i");
  chip.className = className;
  chip.textContent = label;
  return chip;
}

function sampleRow(sample) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `sample-row${sample.id === state.selectedId ? " is-active" : ""}`;
  button.dataset.dataset = state.dataset;
  const title = document.createElement("strong");
  title.textContent = sample.id;
  const meta = document.createElement("span");
  meta.textContent = `${sample.frames} frames · ${(sample.frames / sample.fps).toFixed(2)}s`;
  const chips = document.createElement("div");
  chips.className = "asset-chips";
  chips.append(assetChip("MOTION"));
  if (sample.has_video) chips.append(assetChip("VIDEO", "is-video"));
  if (sample.has_caption || sample.has_clip_caption) chips.append(assetChip("TEXT", "is-text"));
  if (sample.has_cot) chips.append(assetChip("COT", "is-cot"));
  button.append(title, meta, chips);
  button.addEventListener("click", () => selectSample(sample.id));
  return button;
}

function renderSamples(append = false, appendFrom = 0) {
  if (!append) elements.sampleList.replaceChildren();
  if (!state.samples.length) {
    const empty = document.createElement("div");
    empty.className = "list-state";
    const title = document.createElement("strong");
    title.textContent = state.query || state.filter !== "all" ? "No matching records" : "No indexed records";
    const text = document.createElement("span");
    text.textContent = state.query || state.filter !== "all" ? "Change the search or asset filter." : "Build the server index to browse samples.";
    empty.append(title, text);
    elements.sampleList.append(empty);
  } else {
    for (const sample of state.samples.slice(append ? appendFrom : 0)) elements.sampleList.append(sampleRow(sample));
  }
  elements.resultCount.textContent = numberFormatter.format(state.total);
  elements.loadMore.disabled = state.samples.length >= state.total;
  elements.loadMore.textContent = state.samples.length >= state.total ? `${numberFormatter.format(state.total)} records` : "Load more";
}

async function loadSamples({ reset = true, autoSelect = false } = {}) {
  const dataset = state.dataset;
  const query = state.query;
  const filter = state.filter;
  const serial = ++state.listSerial;
  const offset = reset ? 0 : state.samples.length;
  const params = new URLSearchParams({ dataset, q: query, limit: "50", offset: String(offset) });
  if (filter === "video") params.set("has_video", "1");
  if (filter === "text") params.set("has_text", "1");
  if (filter === "cot") params.set("has_cot", "1");
  elements.loadMore.disabled = true;
  try {
    const payload = await requestJson(`/api/samples?${params}`);
    if (serial !== state.listSerial || dataset !== state.dataset || query !== state.query || filter !== state.filter) return;
    if (reset) state.samples = payload.items;
    else state.samples.push(...payload.items);
    state.total = payload.total;
    renderSamples(!reset, offset);
    if (autoSelect && !state.selectedId && state.samples[0]) await selectSample(state.samples[0].id);
  } catch (error) {
    if (serial !== state.listSerial) return;
    elements.sampleList.replaceChildren();
    const failed = document.createElement("div");
    failed.className = "list-state";
    failed.textContent = "Server index unavailable.";
    elements.sampleList.append(failed);
    showToast(`Unable to read sample index: ${error.message}`);
  }
}

function clearSample() {
  state.selectionSerial += 1;
  state.selectedId = null;
  state.detail = null;
  state.positions = null;
  state.frames = 0;
  state.currentFrame = 0;
  state.frameCursor = 0;
  state.textMode = "base";
  state.textQuery = "";
  state.textSerial += 1;
  state.smplReady = false;
  state.smplFailed = false;
  setPlaying(false);
  elements.activeSample.textContent = "Select a sample";
  elements.sampleMeta.textContent = "Choose server data from the left panel.";
  elements.meshState.textContent = "No motion";
  elements.viewportEmpty.hidden = false;
  elements.frameSlider.max = "0";
  elements.frameSlider.value = "0";
  elements.frameCount.textContent = "000 / 000";
  elements.sourceVideo.removeAttribute("src");
  elements.sourceVideo.load();
  elements.videoWrap.classList.add("is-empty");
  elements.videoState.textContent = "NO SOURCE";
  elements.textState.textContent = "NO SAMPLE";
  elements.textSearch.value = "";
  renderEmptyText();
  renderMotionFacts();
  renderEvidence();
  skeletonGroup.visible = false;
  if (smplMesh) smplMesh.visible = false;
}

async function selectDataset(dataset) {
  if (dataset === state.dataset) return;
  state.dataset = dataset;
  state.query = "";
  state.filter = "all";
  elements.sampleSearch.value = "";
  elements.assetFilter.value = "all";
  elements.datasetButtons.forEach((button) => {
    const active = button.dataset.dataset === dataset;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  clearSample();
  updateDatasetMeta();
  await loadSamples({ autoSelect: true });
}

function findFloor(positions) {
  let floor = Infinity;
  for (let index = 1; index < positions.length; index += 3) floor = Math.min(floor, positions[index]);
  return Number.isFinite(floor) ? floor : 0;
}

async function selectSample(sampleId) {
  if (sampleId === state.selectedId && state.positions) return;
  const dataset = state.dataset;
  const serial = ++state.selectionSerial;
  state.selectedId = sampleId;
  state.detail = null;
  state.positions = null;
  state.frames = 0;
  state.currentFrame = 0;
  state.frameCursor = 0;
  state.textQuery = "";
  state.textSerial += 1;
  elements.textSearch.value = "";
  state.smplReady = false;
  state.smplFailed = false;
  setPlaying(false);
  renderSamples();
  elements.activeSample.textContent = sampleId;
  elements.sampleMeta.textContent = "Loading motion, media and text…";
  elements.meshState.textContent = "Decoding 263D";
  elements.viewportEmpty.hidden = true;
  skeletonGroup.visible = false;
  if (smplMesh) smplMesh.visible = false;
  try {
    const [detail, response] = await Promise.all([
      requestJson(`/api/sample/${dataset}/${encodeURIComponent(sampleId)}`),
      fetch(`/api/motion/${dataset}/${encodeURIComponent(sampleId)}`),
    ]);
    if (!response.ok) throw new Error(await response.text());
    const buffer = await response.arrayBuffer();
    if (serial !== state.selectionSerial || dataset !== state.dataset || sampleId !== state.selectedId) return;
    const frames = Number(response.headers.get("X-Motion-Frames"));
    const joints = Number(response.headers.get("X-Motion-Joints"));
    if (!frames || joints !== 22 || buffer.byteLength !== frames * 22 * 3 * 4) throw new Error("Motion binary size mismatch");
    state.detail = detail;
    state.positions = new Float32Array(buffer);
    state.frames = frames;
    state.fps = Number(response.headers.get("X-Motion-Fps")) || detail.fps;
    state.floorOffset = findFloor(state.positions);
    elements.sampleMeta.textContent = `${frames} frames · ${(frames / state.fps).toFixed(2)}s · ${state.fps} FPS · float32 (T,263)`;
    elements.frameSlider.max = String(Math.max(0, frames - 1));
    updateVideo();
    updateTextTabs();
    renderMotionFacts();
    renderEvidence();
    updateFrame(0);
    resetCamera();
    if (state.viewMode === "surface") queueSmplFrame(0);
  } catch (error) {
    if (serial !== state.selectionSerial || dataset !== state.dataset || sampleId !== state.selectedId) return;
    elements.viewportEmpty.hidden = false;
    elements.viewportEmpty.querySelector("strong").textContent = "Unable to load motion";
    elements.viewportEmpty.querySelector("p").textContent = error.message;
    elements.meshState.textContent = "Load failed";
    showToast(`Unable to load ${sampleId}: ${error.message}`);
  }
}

function updateVideo() {
  const hasVideo = Boolean(state.detail?.has_video);
  elements.videoWrap.classList.toggle("is-empty", !hasVideo);
  elements.sourceVideo.removeAttribute("src");
  if (hasVideo) {
    elements.sourceVideo.src = state.detail.video_url;
    elements.sourceVideo.load();
    elements.videoState.textContent = "SOURCE VIDEO";
    elements.syncNote.textContent = "Sample IDs are paired. Video and motion remain independently controlled because clip zero-point is not independently verified.";
  } else {
    elements.sourceVideo.load();
    elements.videoState.textContent = "NO SOURCE";
    elements.syncNote.textContent = state.dataset === "motionx" ? "This motion has no indexed source video." : "This dataset does not include a source-video pool in the current server snapshot.";
  }
}

function renderEmptyText() {
  elements.textContent.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "media-empty";
  const title = document.createElement("strong");
  title.textContent = "No text selected";
  const description = document.createElement("span");
  description.textContent = "Select a motion record to inspect its annotations.";
  empty.append(title, description);
  elements.textContent.append(empty);
}

function textForMode(mode) {
  if (!state.detail) return null;
  if (mode === "clip") return state.detail.clip_caption;
  if (mode === "cot") return state.detail.cot;
  return state.detail.caption;
}

function updateTextTabs() {
  const availability = { base: Boolean(state.detail?.caption), clip: Boolean(state.detail?.clip_caption), cot: Boolean(state.detail?.cot) };
  const baseLabels = { humanml3d: "Sequence", sonic: "Events", motionx: "Frames" };
  elements.textTabs.forEach((button) => {
    const mode = button.dataset.text;
    if (mode === "base") button.textContent = baseLabels[state.dataset];
    button.disabled = !availability[mode];
  });
  if (!availability[state.textMode]) state.textMode = ["base", "clip", "cot"].find((mode) => availability[mode]) || "base";
  elements.textTabs.forEach((button) => {
    const active = button.dataset.text === state.textMode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  renderText(textForMode(state.textMode));
}

function frameLineParts(line) {
  const match = line.match(/^\[Frame\s+(\d+)\s*,\s*([0-9.]+)s\]\s*(.*)$/i);
  if (!match) return null;
  return { frame: Number(match[1]), time: Number(match[2]), text: match[3] };
}

function appendLineCaption(caption) {
  for (const line of caption.lines) {
    const parsed = frameLineParts(line);
    if (!parsed) {
      const paragraph = document.createElement("p");
      paragraph.textContent = line;
      elements.textContent.append(paragraph);
      continue;
    }
    const row = document.createElement("div");
    row.className = "frame-text-row";
    const jump = document.createElement("button");
    jump.type = "button";
    jump.className = "frame-jump";
    jump.textContent = `Frame ${parsed.frame} · ${parsed.time.toFixed(2)}s`;
    jump.addEventListener("click", () => {
      setPlaying(false);
      state.frameCursor = Math.max(0, Math.min(state.frames - 1, parsed.frame));
      updateFrame(state.frameCursor);
    });
    const paragraph = document.createElement("p");
    paragraph.textContent = parsed.text;
    row.append(jump, paragraph);
    elements.textContent.append(row);
  }
  if (caption.next_offset !== null && caption.next_offset !== undefined) {
    const load = document.createElement("button");
    load.type = "button";
    load.className = "text-load-more";
    load.textContent = `Load more · ${numberFormatter.format(caption.lines.length)} / ${numberFormatter.format(caption.count)}`;
    load.addEventListener("click", () => loadBaseText({ reset: false }));
    elements.textContent.append(load);
  }
}

function clipSections(description) {
  const sections = [];
  let current = { title: "Overview", lines: [] };
  const flush = () => {
    if (current.title || current.lines.length) sections.push(current);
  };
  for (const rawLine of description.split("\n")) {
    const clean = rawLine.trim().replaceAll("**", "").replace(/^[-]\s*/, "").trim();
    if (!clean) continue;
    const heading = /^(Overall Action Overview|Actors:?$|Temporal Motion Breakdown|Key Action Summary|\[[0-9:.]+[–-][0-9:.]+\])[:]?$/i.test(clean);
    if (heading) {
      if (current.lines.length || current.title !== "Overview") flush();
      current = { title: clean.replace(/:$/, ""), lines: [] };
    } else current.lines.push(clean);
  }
  flush();
  return sections;
}

function appendClipCaption(caption) {
  const query = state.textQuery.toLowerCase();
  for (const section of clipSections(caption.description)) {
    const haystack = `${section.title}\n${section.lines.join("\n")}`.toLowerCase();
    if (query && !haystack.includes(query.toLowerCase())) continue;
    const details = document.createElement("details");
    details.className = "clip-section";
    details.open = !/^\[/.test(section.title);
    const summary = document.createElement("summary");
    summary.textContent = section.title;
    const body = document.createElement("div");
    for (const line of section.lines) {
      const paragraph = document.createElement("p");
      paragraph.textContent = line;
      body.append(paragraph);
    }
    details.append(summary, body);
    elements.textContent.append(details);
  }
}

function renderText(caption) {
  elements.textContent.replaceChildren();
  if (!caption) {
    elements.textState.textContent = "NO TEXT";
    renderEmptyText();
    return;
  }
  elements.textState.textContent = caption.kind === "cot" ? `${caption.count} SEGMENTS` : caption.kind === "clip" ? "CLIP CAPTION" : `${caption.lines?.length || caption.count} / ${caption.count}`;
  if (caption.kind === "events") {
    for (const event of caption.events) {
      if (state.textQuery && !event.description.toLowerCase().includes(state.textQuery.toLowerCase())) continue;
      const row = document.createElement("div");
      row.className = "text-event";
      const time = document.createElement("time");
      time.textContent = `${Number(event.start_time).toFixed(2)}–${Number(event.end_time).toFixed(2)}s`;
      const text = document.createElement("span");
      text.textContent = event.description;
      row.append(time, text);
      elements.textContent.append(row);
    }
  } else if (caption.kind === "lines") {
    if (!caption.lines.length) {
      const empty = document.createElement("div");
      empty.className = "media-empty";
      const title = document.createElement("strong");
      title.textContent = "No matching text";
      const description = document.createElement("span");
      description.textContent = "Try another search term.";
      empty.append(title, description);
      elements.textContent.append(empty);
      return;
    }
    appendLineCaption(caption);
  } else if (caption.kind === "clip") {
    appendClipCaption(caption);
  } else if (caption.kind === "cot") {
    const summary = document.createElement("div");
    summary.className = "cot-summary";
    const summaryTitle = document.createElement("strong");
    summaryTitle.textContent = "Summary";
    const summaryText = document.createElement("p");
    summaryText.textContent = caption.sample_summary;
    summary.append(summaryTitle, summaryText);
    elements.textContent.append(summary);
    for (const segment of caption.segments) {
      const searchable = `${segment.time_range} ${segment.cot_type} ${segment.think} ${segment.answer}`.toLowerCase();
      if (state.textQuery && !searchable.includes(state.textQuery.toLowerCase())) continue;
      const details = document.createElement("details");
      details.className = "cot-segment";
      const title = document.createElement("summary");
      title.textContent = `${segment.time_range} · ${segment.cot_type || "fusion"}`;
      const reasoningTitle = document.createElement("strong");
      reasoningTitle.textContent = "Reasoning";
      const reasoning = document.createElement("p");
      reasoning.textContent = segment.think;
      const answerTitle = document.createElement("strong");
      answerTitle.textContent = "Answer";
      const answer = document.createElement("p");
      answer.textContent = segment.answer;
      details.append(title, reasoningTitle, reasoning, answerTitle, answer);
      elements.textContent.append(details);
    }
    const final = document.createElement("div");
    final.className = "cot-summary";
    const finalTitle = document.createElement("strong");
    finalTitle.textContent = "Final Answer";
    const finalText = document.createElement("p");
    finalText.textContent = caption.final_answer;
    final.append(finalTitle, finalText);
    elements.textContent.append(final);
  }
}

async function loadBaseText({ reset = false } = {}) {
  if (!state.detail || state.detail.caption?.kind !== "lines") return;
  const dataset = state.dataset;
  const sampleId = state.selectedId;
  const serial = ++state.textSerial;
  const current = state.detail.caption;
  const offset = reset ? 0 : current.next_offset;
  if (offset === null || offset === undefined) return;
  const params = new URLSearchParams({ offset: String(offset), limit: "40", q: state.textQuery });
  try {
    const payload = await requestJson(`/api/text/${dataset}/${encodeURIComponent(sampleId)}?${params}`);
    if (serial !== state.textSerial || dataset !== state.dataset || sampleId !== state.selectedId || state.textMode !== "base") return;
    if (reset) state.detail.caption = payload;
    else state.detail.caption = { ...payload, lines: [...current.lines, ...payload.lines] };
    renderText(state.detail.caption);
  } catch (error) {
    if (serial !== state.textSerial) return;
    showToast(`Unable to load text: ${error.message}`);
  }
}

function captionAsText(caption) {
  if (!caption) return "";
  if (caption.kind === "lines") return caption.lines.join("\n");
  if (caption.kind === "events") return caption.events.map((event) => `[${event.start_time}–${event.end_time}s] ${event.description}`).join("\n");
  if (caption.kind === "clip") return caption.description;
  if (caption.kind === "cot") {
    return [
      `Summary\n${caption.sample_summary}`,
      ...caption.segments.map((segment) => `[${segment.time_range}] ${segment.cot_type}\nReasoning: ${segment.think}\nAnswer: ${segment.answer}`),
      `Final Answer\n${caption.final_answer}`,
    ].join("\n\n");
  }
  return "";
}

function replaceDefinitionList(entries) {
  elements.evidenceList.replaceChildren();
  for (const [label, value] of entries) {
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    row.append(dt, dd);
    elements.evidenceList.append(row);
  }
}

function renderEvidence() {
  const textKinds = state.detail ? [state.detail.caption, state.detail.clip_caption, state.detail.cot].filter(Boolean).length : 0;
  replaceDefinitionList([
    ["Motion", state.positions ? "Decoded · 22 joints" : "Waiting"],
    ["Video", state.detail?.has_video ? "Original media" : state.detail ? "Unavailable" : "Waiting"],
    ["SMPL-H", state.smplReady ? "Derived mesh · ready" : state.smplFailed ? "Failed · skeleton shown" : state.smplhAvailable ? "Configured" : "Unavailable"],
    ["Text", state.detail ? `${textKinds} annotation type${textKinds === 1 ? "" : "s"}` : "Waiting"],
  ]);
}

function renderMotionFacts() {
  const facts = [
    ["Representation", "263D"],
    ["Frames", state.frames ? numberFormatter.format(state.frames) : "—"],
    ["Duration", state.frames ? `${(state.frames / state.fps).toFixed(2)} s` : "—"],
    ["Render", state.smplReady ? "SMPL-H mesh" : state.positions ? "22-joint skeleton" : "—"],
  ];
  elements.motionFacts.replaceChildren();
  for (const [label, value] of facts) {
    const row = document.createElement("div");
    const labelNode = document.createElement("span");
    const valueNode = document.createElement("strong");
    labelNode.textContent = label;
    valueNode.textContent = value;
    row.append(labelNode, valueNode);
    elements.motionFacts.append(row);
  }
}

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x07111f, 7, 18);
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x07111f, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
elements.viewport.prepend(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;
controls.minDistance = 1.2;
controls.maxDistance = 12;

scene.add(new THREE.HemisphereLight(0xcce5ff, 0x17212e, 2.25));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
keyLight.position.set(3.5, 6, 4.5);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x7ea2ff, 1.1);
rimLight.position.set(-4, 2.5, -3);
scene.add(rimLight);

const grid = new THREE.GridHelper(12, 24, 0x28425f, 0x15273a);
grid.material.transparent = true;
grid.material.opacity = 0.58;
scene.add(grid);
const floor = new THREE.Mesh(new THREE.CircleGeometry(6, 72), new THREE.MeshStandardMaterial({ color: 0x0a1420, roughness: 1, metalness: 0, transparent: true, opacity: 0.52 }));
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.012;
scene.add(floor);

const skeletonGroup = new THREE.Group();
scene.add(skeletonGroup);
const jointGeometry = new THREE.SphereGeometry(0.045, 14, 10);
const jointMaterial = new THREE.MeshStandardMaterial({ color: 0x74a0ff, roughness: 0.42, metalness: 0.08 });
const jointMeshes = Array.from({ length: 22 }, (_, index) => {
  const material = index === 0 ? new THREE.MeshStandardMaterial({ color: 0xff9a61, roughness: 0.4 }) : jointMaterial;
  const mesh = new THREE.Mesh(index === 15 ? new THREE.SphereGeometry(0.09, 16, 12) : jointGeometry, material);
  skeletonGroup.add(mesh);
  return mesh;
});
const linePositions = new Float32Array(EDGES.length * 6);
const lineGeometry = new THREE.BufferGeometry();
lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
const boneLines = new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({ color: 0xd5e5f5, transparent: true, opacity: 0.9 }));
skeletonGroup.add(boneLines);
skeletonGroup.visible = false;

let smplGeometry = null;
let smplMesh = null;
let smplTopologyPromise = null;
let meshRequestedFrame = null;
let meshLoading = false;
let meshSerial = 0;
let lastQueuedFrame = -1;
const pointA = new THREE.Vector3();
const pointB = new THREE.Vector3();

function jointAt(frame, joint, target) {
  const offset = (frame * 22 + joint) * 3;
  const rootOffset = frame * 22 * 3;
  target.set(
    state.positions[offset] - state.positions[rootOffset],
    state.positions[offset + 1] - state.floorOffset,
    state.positions[offset + 2] - state.positions[rootOffset + 2],
  );
  return target;
}

function updateFrame(frame) {
  if (!state.positions || !state.frames) return;
  state.currentFrame = Math.max(0, Math.min(state.frames - 1, Math.round(frame)));
  let offset = 0;
  jointMeshes.forEach((mesh, joint) => jointAt(state.currentFrame, joint, mesh.position));
  for (const [a, b] of EDGES) {
    pointA.copy(jointMeshes[a].position);
    pointB.copy(jointMeshes[b].position);
    linePositions[offset++] = pointA.x;
    linePositions[offset++] = pointA.y;
    linePositions[offset++] = pointA.z;
    linePositions[offset++] = pointB.x;
    linePositions[offset++] = pointB.y;
    linePositions[offset++] = pointB.z;
  }
  lineGeometry.attributes.position.needsUpdate = true;
  skeletonGroup.visible = state.viewMode === "skeleton" || (state.viewMode === "surface" && !state.smplReady);
  if (smplMesh) smplMesh.visible = state.viewMode === "surface" && state.smplReady;
  elements.frameSlider.value = String(state.currentFrame);
  const width = Math.max(3, String(state.frames).length);
  elements.frameCount.textContent = `${String(state.currentFrame + 1).padStart(width, "0")} / ${String(state.frames).padStart(width, "0")}`;
  if (state.viewMode === "surface" && state.smplhAvailable) {
    const stride = state.playing ? Math.max(1, Math.round(state.fps / 5)) : 1;
    if (!state.playing || Math.abs(state.currentFrame - lastQueuedFrame) >= stride) {
      lastQueuedFrame = state.currentFrame;
      queueSmplFrame(state.currentFrame);
    }
  }
}

async function ensureSmplMesh() {
  if (smplGeometry) return;
  if (!smplTopologyPromise) {
    smplTopologyPromise = (async () => {
      const response = await fetch("/api/smplh/topology");
      if (!response.ok) throw new Error(await response.text());
      const faces = Number(response.headers.get("X-SMPLH-Faces"));
      const indices = new Uint32Array(await response.arrayBuffer());
      if (!faces || indices.length !== faces * 3) throw new Error("SMPL-H topology mismatch");
      smplGeometry = new THREE.BufferGeometry();
      smplGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
      smplMesh = new THREE.Mesh(smplGeometry, new THREE.MeshStandardMaterial({ color: 0xc5d8e8, roughness: 0.5, metalness: 0.02, side: THREE.DoubleSide }));
      smplMesh.visible = false;
      scene.add(smplMesh);
    })();
  }
  return smplTopologyPromise;
}

function queueSmplFrame(frame) {
  if (!state.smplhAvailable || !state.selectedId || state.viewMode !== "surface") return;
  meshRequestedFrame = Math.max(0, Math.min(state.frames - 1, Math.round(frame)));
  elements.meshState.textContent = state.smplReady ? "Updating SMPL-H" : "Loading SMPL-H";
  if (!meshLoading) processSmplQueue();
}

async function processSmplQueue() {
  meshLoading = true;
  while (meshRequestedFrame !== null && state.selectedId) {
    const frame = meshRequestedFrame;
    meshRequestedFrame = null;
    const dataset = state.dataset;
    const sampleId = state.selectedId;
    const serial = ++meshSerial;
    try {
      await ensureSmplMesh();
      const response = await fetch(`/api/smplh/mesh/${dataset}/${encodeURIComponent(sampleId)}/${frame}`);
      if (!response.ok) throw new Error(await response.text());
      const count = Number(response.headers.get("X-SMPLH-Vertices"));
      const vertices = new Float32Array(await response.arrayBuffer());
      if (!count || vertices.length !== count * 3) throw new Error("SMPL-H vertex mismatch");
      if (sampleId !== state.selectedId || dataset !== state.dataset || serial !== meshSerial) continue;
      smplGeometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
      smplGeometry.computeVertexNormals();
      smplGeometry.computeBoundingSphere();
      state.smplReady = true;
      state.smplFailed = false;
      smplMesh.visible = state.viewMode === "surface";
      skeletonGroup.visible = state.viewMode === "skeleton";
      elements.meshState.textContent = `SMPL-H · frame ${frame + 1}`;
      renderMotionFacts();
      renderEvidence();
    } catch (error) {
      state.smplReady = false;
      state.smplFailed = true;
      if (smplMesh) smplMesh.visible = false;
      skeletonGroup.visible = true;
      elements.meshState.textContent = "SMPL-H failed · skeleton fallback";
      renderMotionFacts();
      renderEvidence();
      showToast(`SMPL-H unavailable: ${error.message}`);
    }
  }
  meshLoading = false;
}

function resetCamera() {
  camera.position.set(2.8, 2.25, 3.7);
  controls.target.set(0, 0.95, 0);
  controls.update();
}

function setCameraPreset(preset) {
  if (preset === "front") camera.position.set(0, 1.55, 4.4);
  else if (preset === "side") camera.position.set(4.4, 1.55, 0);
  else if (preset === "top") camera.position.set(0.01, 5.4, 0.01);
  controls.target.set(0, 0.95, 0);
  controls.update();
}
resetCamera();

function setPlaying(value) {
  state.playing = Boolean(value && state.frames);
  state.lastTimestamp = performance.now();
  elements.playToggle.classList.toggle("is-playing", state.playing);
  elements.playToggle.setAttribute("aria-label", state.playing ? "Pause" : "Play");
  if (!state.frames) elements.meshState.textContent = "No motion";
  else if (state.playing) elements.meshState.textContent = state.viewMode === "surface" ? "Playing · SMPL-H ~5 FPS" : "Playing · skeleton";
  else if (state.viewMode === "surface" && state.smplReady) elements.meshState.textContent = `SMPL-H · frame ${state.currentFrame + 1}`;
  else elements.meshState.textContent = "Paused";
}

function animate(timestamp) {
  requestAnimationFrame(animate);
  const delta = Math.min(0.1, (timestamp - state.lastTimestamp) / 1000 || 0);
  state.lastTimestamp = timestamp;
  if (state.playing && state.frames) {
    state.frameCursor += delta * state.fps * state.speed;
    if (state.frameCursor >= state.frames) {
      if (state.loop) state.frameCursor = 0;
      else {
        state.frameCursor = state.frames - 1;
        setPlaying(false);
      }
    }
    updateFrame(state.frameCursor);
  }
  controls.update();
  renderer.render(scene, camera);
}

const resizeObserver = new ResizeObserver(() => {
  const width = Math.max(1, elements.viewport.clientWidth);
  const height = Math.max(1, elements.viewport.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
});
resizeObserver.observe(elements.viewport);
requestAnimationFrame(animate);

elements.datasetButtons.forEach((button) => button.addEventListener("click", () => selectDataset(button.dataset.dataset)));
elements.sampleSearch.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.query = elements.sampleSearch.value.trim();
    clearSample();
    loadSamples({ autoSelect: true });
  }, 220);
});
elements.assetFilter.addEventListener("change", () => {
  state.filter = elements.assetFilter.value;
  clearSample();
  loadSamples({ autoSelect: true });
});
elements.loadMore.addEventListener("click", () => loadSamples({ reset: false }));
elements.viewButtons.forEach((button) => button.addEventListener("click", () => {
  if (button.disabled) return;
  state.viewMode = button.dataset.mode;
  elements.viewButtons.forEach((item) => {
    const active = item.dataset.mode === state.viewMode;
    item.classList.toggle("is-active", active);
    item.setAttribute("aria-pressed", String(active));
  });
  if (state.positions) updateFrame(state.currentFrame);
  if (state.viewMode === "surface" && state.positions) queueSmplFrame(state.currentFrame);
  else if (state.positions) elements.meshState.textContent = "Skeleton · deterministic decode";
}));
elements.cameraButtons.forEach((button) => button.addEventListener("click", () => setCameraPreset(button.dataset.camera)));
elements.resetCamera.addEventListener("click", resetCamera);
elements.playToggle.addEventListener("click", () => setPlaying(!state.playing));
elements.prevFrame.addEventListener("click", () => {
  setPlaying(false);
  state.frameCursor = Math.max(0, state.currentFrame - 1);
  updateFrame(state.frameCursor);
});
elements.nextFrame.addEventListener("click", () => {
  setPlaying(false);
  state.frameCursor = Math.min(Math.max(0, state.frames - 1), state.currentFrame + 1);
  updateFrame(state.frameCursor);
});
elements.frameSlider.addEventListener("input", () => {
  setPlaying(false);
  state.frameCursor = Number(elements.frameSlider.value);
  updateFrame(state.frameCursor);
});
elements.speedSelect.addEventListener("change", () => { state.speed = Number(elements.speedSelect.value); });
elements.loopToggle.addEventListener("change", () => { state.loop = elements.loopToggle.checked; });
elements.textTabs.forEach((button) => button.addEventListener("click", () => {
  if (button.disabled) return;
  state.textMode = button.dataset.text;
  state.textQuery = "";
  state.textSerial += 1;
  elements.textSearch.value = "";
  updateTextTabs();
  if (state.textMode === "base" && state.detail?.caption?.kind === "lines" && state.detail.caption.query) loadBaseText({ reset: true });
}));
elements.textSearch.addEventListener("input", () => {
  clearTimeout(textSearchTimer);
  textSearchTimer = setTimeout(() => {
    state.textQuery = elements.textSearch.value.trim();
    const caption = textForMode(state.textMode);
    if (state.textMode === "base" && caption?.kind === "lines") loadBaseText({ reset: true });
    else renderText(caption);
  }, 220);
});
elements.expandText.addEventListener("click", () => {
  const expanded = elements.textPanel.classList.toggle("is-expanded");
  elements.expandText.textContent = expanded ? "Close" : "Expand";
  if (expanded) elements.textSearch.focus();
});
elements.copyText.addEventListener("click", async () => {
  const text = captionAsText(textForMode(state.textMode));
  if (!text) {
    showToast("No text to copy.");
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    showToast("Text copied.");
  } catch (error) {
    showToast(`Copy failed: ${error.message}`);
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && elements.textPanel.classList.contains("is-expanded")) {
    elements.textPanel.classList.remove("is-expanded");
    elements.expandText.textContent = "Expand";
    elements.expandText.focus();
    return;
  }
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || tag === "VIDEO") return;
  if (event.code === "Space") {
    event.preventDefault();
    setPlaying(!state.playing);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    elements.prevFrame.click();
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    elements.nextFrame.click();
  }
});

async function boot() {
  try {
    state.summary = await requestJson("/api/summary");
    state.smplhAvailable = Boolean(state.summary.smplh?.available);
    const surfaceButton = elements.viewButtons.find((button) => button.dataset.mode === "surface");
    if (surfaceButton) {
      surfaceButton.disabled = !state.smplhAvailable;
      surfaceButton.title = state.smplhAvailable ? "Render derived neutral SMPL-H" : "SMPL-H is unavailable on this server";
    }
    updateDatasetMeta();
    elements.serverDot.classList.add("is-ready");
    const indexed = Object.values(state.summary.indexed_samples || {}).reduce((sum, value) => sum + value, 0);
    elements.serverStatus.textContent = `${numberFormatter.format(indexed)} indexed`;
    renderEvidence();
    await loadSamples({ autoSelect: true });
  } catch (error) {
    elements.serverDot.classList.add("is-error");
    elements.serverStatus.textContent = "Server unavailable";
    showToast(`Viewer initialization failed: ${error.message}`);
  }
}

boot();
