import * as THREE from "/static/vendor/three.module.js?v=20260901-5";
import { OrbitControls } from "/static/vendor/OrbitControls.js?v=20260901-5";

const DATASET_LABELS = {
  humanml3d: "HumanML3D",
  sonic: "SONIC",
  motionx: "MotionX",
};

const DATASET_EXAMPLES = {
  humanml3d: {
    id: "SEQUENCE CAPTION",
    text: "The person is walk in the street backwards.",
    tags: ["Motion", "Text", "20 FPS"],
  },
  sonic: {
    id: "hook_jump_off_wall_R_003__A304_M",
    text: "Runs forward and delivers a hook punch by jumping off the wall.",
    tags: ["3 Events", "Mirrored", "30 FPS"],
  },
  motionx: {
    id: "11191119210129",
    text: "A woman performs a high-energy aerobic dance routine on a poolside terrace.",
    tags: ["Video", "Frame Text", "Clip / CoT"],
  },
};

const CHAINS = [
  [0, 2, 5, 8, 11],
  [0, 1, 4, 7, 10],
  [0, 3, 6, 9, 12, 15],
  [9, 14, 17, 19, 21],
  [9, 13, 16, 18, 20],
];

const EDGES = [];
for (const chain of CHAINS) {
  for (let index = 0; index < chain.length - 1; index += 1) {
    const edge = [chain[index], chain[index + 1]];
    if (!EDGES.some(([a, b]) => a === edge[0] && b === edge[1])) EDGES.push(edge);
  }
}

const state = {
  dataset: "humanml3d",
  summary: null,
  samples: [],
  total: 0,
  offset: 0,
  query: "",
  selectedId: null,
  detail: null,
  positions: null,
  frames: 0,
  fps: 20,
  currentFrame: 0,
  frameCursor: 0,
  playing: false,
  speed: 1,
  lastTimestamp: 0,
  viewMode: "skeleton",
  annotationMode: "base",
  loop: true,
  floorOffset: 0,
  smplhAvailable: false,
  smplMeshFallback: false,
  smplMeshReady: false,
  selectionSerial: 0,
  listSerial: 0,
};

const elements = {
  indexStatus: document.getElementById("indexStatus"),
  statusDot: document.querySelector(".status-dot"),
  statGrid: document.getElementById("statGrid"),
  taskMatrix: document.getElementById("taskMatrix"),
  taskMethod: document.getElementById("taskMethod"),
  datasetProfiles: document.getElementById("datasetProfiles"),
  lengthPlot: document.getElementById("lengthPlot"),
  datasetTabs: [...document.querySelectorAll(".dataset-tab")],
  datasetKicker: document.getElementById("datasetKicker"),
  datasetCount: document.getElementById("datasetCount"),
  datasetDescription: document.getElementById("datasetDescription"),
  sampleSearch: document.getElementById("sampleSearch"),
  sampleList: document.getElementById("sampleList"),
  loadMore: document.getElementById("loadMore"),
  activeDatasetLabel: document.getElementById("activeDatasetLabel"),
  activeSampleId: document.getElementById("activeSampleId"),
  activeSampleMeta: document.getElementById("activeSampleMeta"),
  evidenceBadges: document.getElementById("evidenceBadges"),
  viewerStatus: document.getElementById("viewerStatus"),
  modeButtons: [...document.querySelectorAll(".mode-button")],
  viewerGrid: document.getElementById("viewerGrid"),
  viewport: document.getElementById("viewport"),
  viewportEmpty: document.getElementById("viewportEmpty"),
  videoCard: document.getElementById("videoCard"),
  video: document.getElementById("motionVideo"),
  videoEmpty: document.getElementById("videoEmpty"),
  playToggle: document.getElementById("playToggle"),
  prevFrame: document.getElementById("prevFrame"),
  nextFrame: document.getElementById("nextFrame"),
  frameCounter: document.getElementById("frameCounter"),
  frameSlider: document.getElementById("frameSlider"),
  speedSelect: document.getElementById("speedSelect"),
  loopToggle: document.getElementById("loopToggle"),
  resetView: document.getElementById("resetView"),
  cameraButtons: [...document.querySelectorAll("[data-camera]")],
  captionCount: document.getElementById("captionCount"),
  captionContent: document.getElementById("captionContent"),
  annotationTabs: [...document.querySelectorAll(".annotation-tab")],
  assetFacts: document.getElementById("assetFacts"),
  scaleChart: document.getElementById("scaleChart"),
  qualityList: document.getElementById("qualityList"),
  toast: document.getElementById("toast"),
};

const numberFormatter = new Intl.NumberFormat("zh-CN");
let toastTimer;
let searchTimer;

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove("is-visible"), 3200);
}

async function requestJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function createStat(label, value, unit) {
  const card = document.createElement("article");
  card.className = "stat-card";
  const labelNode = document.createElement("span");
  labelNode.textContent = label;
  const valueNode = document.createElement("strong");
  valueNode.textContent = value;
  const unitNode = document.createElement("small");
  unitNode.textContent = unit;
  card.append(labelNode, valueNode, unitNode);
  return card;
}

function renderSummary() {
  const totals = state.summary.totals;
  const datasets = state.summary.datasets;
  const totalHours = Object.values(datasets).reduce((sum, item) => sum + item.duration_hours, 0);
  elements.statGrid.replaceChildren(
    createStat("动作序列", numberFormatter.format(totals.motion_files), "NPY"),
    createStat("源视频", numberFormatter.format(totals.video_files), "MP4"),
    createStat("动作时长", totalHours.toFixed(2), "HOURS"),
    createStat("数据体积", totals.size_gb.toFixed(1), "GB"),
  );

  const keys = ["humanml3d", "sonic", "motionx"];
  const maxMotions = Math.max(...keys.map((key) => datasets[key].motion_files));
  const maxHours = Math.max(...keys.map((key) => datasets[key].duration_hours));
  elements.scaleChart.replaceChildren();
  elements.datasetProfiles.replaceChildren();
  elements.lengthPlot.replaceChildren();
  elements.qualityList.replaceChildren();
  for (const key of keys) {
    const item = datasets[key];

    const profile = document.createElement("article");
    profile.className = `dataset-profile profile-${key}`;
    const profileHeader = document.createElement("header");
    const profileTag = document.createElement("span");
    profileTag.textContent = key === "motionx" ? "VIDEO + MOTION + TEXT" : "MOTION + TEXT";
    const profileFps = document.createElement("span");
    profileFps.textContent = `${item.fps} FPS`;
    profileHeader.append(profileTag, profileFps);
    const profileTitle = document.createElement("h3");
    profileTitle.textContent = item.label;
    const profileDescription = document.createElement("p");
    profileDescription.textContent = item.coverage;
    const profileStats = document.createElement("dl");
    profileStats.className = "profile-stats";
    for (const [labelText, valueText] of [
      ["motions", numberFormatter.format(item.motion_files)],
      ["videos", numberFormatter.format(item.video_files)],
      ["duration", `${item.duration_hours.toFixed(2)} h`],
      ["storage", `${item.size_gb.toFixed(3)} GB`],
    ]) {
      const wrapper = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = labelText;
      const dd = document.createElement("dd");
      dd.textContent = valueText;
      wrapper.append(dt, dd);
      profileStats.append(wrapper);
    }
    const example = DATASET_EXAMPLES[key];
    const sample = document.createElement("div");
    sample.className = "profile-example";
    const sampleHeader = document.createElement("div");
    const sampleLabel = document.createElement("span");
    sampleLabel.textContent = "SAMPLE";
    const sampleId = document.createElement("code");
    sampleId.textContent = example.id;
    sampleHeader.append(sampleLabel, sampleId);
    const sampleText = document.createElement("p");
    sampleText.textContent = `“${example.text}”`;
    const sampleTags = document.createElement("div");
    sampleTags.className = "profile-example-tags";
    for (const tag of example.tags) {
      const chip = document.createElement("span");
      chip.textContent = tag;
      sampleTags.append(chip);
    }
    sample.append(sampleHeader, sampleText, sampleTags);
    profile.append(profileHeader, profileTitle, profileDescription, profileStats, sample);
    elements.datasetProfiles.append(profile);

    const row = document.createElement("div");
    row.className = "scale-row";
    const label = document.createElement("span");
    label.textContent = item.label;
    const bars = document.createElement("div");
    bars.className = "scale-bars";
    const motionTrack = document.createElement("div");
    motionTrack.className = "scale-track";
    const motionFill = document.createElement("div");
    motionFill.className = "scale-fill";
    motionFill.style.width = `${Math.max(2, (item.motion_files / maxMotions) * 100)}%`;
    motionTrack.append(motionFill);
    const durationTrack = document.createElement("div");
    durationTrack.className = "scale-track";
    const durationFill = document.createElement("div");
    durationFill.className = "scale-fill secondary";
    durationFill.style.width = `${Math.max(2, (item.duration_hours / maxHours) * 100)}%`;
    durationTrack.append(durationFill);
    bars.append(motionTrack, durationTrack);
    const value = document.createElement("span");
    value.className = "scale-value";
    value.textContent = `${compactNumber(item.motion_files)} · ${item.duration_hours.toFixed(1)}h`;
    row.append(label, bars, value);
    elements.scaleChart.append(row);

    const stats = item.frame_stats;
    const maxFrame = Math.max(...keys.map((datasetKey) => datasets[datasetKey].frame_stats.max));
    const logPosition = (frame) => (Math.log10(frame + 1) / Math.log10(maxFrame + 1)) * 100;
    const lengthRow = document.createElement("div");
    lengthRow.className = "length-row";
    const lengthLabel = document.createElement("div");
    lengthLabel.className = "length-label";
    const lengthName = document.createElement("strong");
    lengthName.textContent = item.label;
    const lengthMeta = document.createElement("small");
    lengthMeta.textContent = `median ${stats.median} · p95 ${stats.p95}`;
    lengthLabel.append(lengthName, lengthMeta);
    const lengthTrack = document.createElement("div");
    lengthTrack.className = "length-track";
    const range = document.createElement("span");
    range.className = "length-range";
    range.style.left = `${logPosition(stats.min)}%`;
    range.style.width = `${Math.max(1, logPosition(stats.max) - logPosition(stats.min))}%`;
    const median = document.createElement("span");
    median.className = "length-marker median";
    median.style.left = `${logPosition(stats.median)}%`;
    median.setAttribute("aria-label", `${item.label} median ${stats.median} frames`);
    const p95 = document.createElement("span");
    p95.className = "length-marker p95";
    p95.style.left = `${logPosition(stats.p95)}%`;
    p95.setAttribute("aria-label", `${item.label} p95 ${stats.p95} frames`);
    lengthTrack.append(range, median, p95);
    lengthRow.append(lengthLabel, lengthTrack);
    elements.lengthPlot.append(lengthRow);

    const note = document.createElement("article");
    note.className = "quality-item";
    const title = document.createElement("strong");
    title.textContent = item.label;
    const description = document.createElement("p");
    description.textContent = item.quality;
    note.append(title, description);
    elements.qualityList.append(note);
  }
  const axis = document.createElement("div");
  axis.className = "length-axis";
  axis.innerHTML = "<span>1</span><span>10</span><span>100</span><span>1K</span><span>5.4K frames</span>";
  elements.lengthPlot.append(axis);
  renderTaskTaxonomy();
  updateDatasetSummary();
}

function modalityLabel(modality) {
  return { motion: "MOTION", video: "VIDEO", text: "TEXT" }[modality] || modality.toUpperCase();
}

function compactNumber(value) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 1 : 2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(value >= 100_000 ? 1 : 2)}K`;
  return numberFormatter.format(value);
}

function metricNode(label, value, sublabel, kind) {
  const node = document.createElement("div");
  node.className = `task-metric metric-${kind}`;
  const labelNode = document.createElement("span");
  labelNode.textContent = label;
  const valueNode = document.createElement("strong");
  valueNode.textContent = value;
  const sublabelNode = document.createElement("small");
  sublabelNode.textContent = sublabel;
  node.append(labelNode, valueNode, sublabelNode);
  return node;
}

function formatPercent(value) {
  if (value === 100) return "100%";
  return `${value.toFixed(value >= 99 ? 3 : 2)}%`;
}

function renderTaskTaxonomy() {
  const taxonomy = state.summary.task_taxonomy;
  if (!taxonomy || !Array.isArray(taxonomy.tasks)) {
    elements.taskMatrix.replaceChildren(Object.assign(document.createElement("div"), {
      className: "task-placeholder",
      textContent: "任务口径统计尚未生成。",
    }));
    return;
  }

  elements.taskMatrix.replaceChildren();
  taxonomy.tasks.forEach((task, index) => {
    const article = document.createElement("article");
    article.className = `task-row task-${task.dataset}`;

    const indexNode = document.createElement("span");
    indexNode.className = "task-index";
    indexNode.textContent = String(index + 1).padStart(2, "0");

    const identity = document.createElement("div");
    identity.className = "task-identity";
    const route = document.createElement("div");
    route.className = "modality-route";
    task.modalities.forEach((modality, modalityIndex) => {
      const chip = document.createElement("span");
      chip.className = `modality-chip modality-${modality}`;
      chip.textContent = modalityLabel(modality);
      route.append(chip);
      if (modalityIndex < task.modalities.length - 1) {
        const connector = document.createElement("i");
        connector.textContent = "↔";
        route.append(connector);
      }
    });
    const name = document.createElement("h3");
    name.textContent = task.name;
    const variant = document.createElement("p");
    variant.textContent = `${DATASET_LABELS[task.dataset]} · ${task.variant}`;
    identity.append(route, name, variant);

    const metrics = document.createElement("div");
    metrics.className = "task-metrics";
    metrics.append(
      metricNode("记录", numberFormatter.format(task.annotation_records), task.annotation_unit, "record"),
      metricNode("配对", numberFormatter.format(task.paired_samples), task.pair_unit, "pair"),
      metricNode("时长", task.duration_hours.toFixed(2), task.duration_basis || "hours", "time"),
    );

    const coverage = document.createElement("div");
    coverage.className = "task-coverage";
    const coverageTop = document.createElement("div");
    const coverageLabel = document.createElement("span");
    coverageLabel.textContent = `覆盖 · ${task.coverage_basis}`;
    const coverageValue = document.createElement("strong");
    coverageValue.textContent = formatPercent(task.coverage_percent);
    coverageTop.append(coverageLabel, coverageValue);
    const track = document.createElement("div");
    track.className = "coverage-track";
    const fill = document.createElement("span");
    fill.style.width = `${Math.min(100, task.coverage_percent)}%`;
    track.append(fill);
    coverage.append(coverageTop, track);

    const notes = document.createElement("div");
    notes.className = "task-notes";
    const detail = document.createElement("p");
    detail.textContent = task.detail;
    const gap = document.createElement("small");
    gap.textContent = task.mirror ? `${task.mirror} · ${task.gap}` : task.gap;
    notes.append(detail, gap);

    article.append(indexNode, identity, metrics, coverage, notes);
    elements.taskMatrix.append(article);
  });

  elements.taskMethod.replaceChildren();
  const methodTitle = document.createElement("strong");
  methodTitle.textContent = "Duration";
  const methodText = document.createElement("p");
  methodText.textContent = taxonomy.duration_definition;
  const stamp = document.createElement("span");
  stamp.textContent = `AUDITED ${taxonomy.generated_at}`;
  elements.taskMethod.append(methodTitle, methodText, stamp);
}

function updateDatasetSummary() {
  if (!state.summary) return;
  const item = state.summary.datasets[state.dataset];
  elements.datasetKicker.textContent = item.label;
  elements.datasetCount.textContent = numberFormatter.format(item.motion_files);
  elements.datasetDescription.textContent = `${item.duration_hours.toFixed(2)} 小时 · ${item.length} · ${item.coverage}`;
  elements.activeDatasetLabel.textContent = `${item.label.toUpperCase()} · MOTION`;
}

function sampleRow(item) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `sample-row${item.id === state.selectedId ? " is-active" : ""}`;
  button.dataset.sampleId = item.id;
  const title = document.createElement("strong");
  title.textContent = item.id;
  const meta = document.createElement("span");
  const dot = document.createElement("i");
  dot.className = `media-dot${item.has_video ? "" : " no-video"}`;
  dot.setAttribute("aria-hidden", "true");
  meta.append(dot, `${item.frames} 帧 · ${(item.frames / item.fps).toFixed(1)} 秒${item.has_caption ? " · 有标注" : ""}`);
  button.append(title, meta);
  button.addEventListener("click", () => selectSample(item.id));
  return button;
}

function renderSamples(append = false, appendFrom = 0) {
  if (!append) elements.sampleList.replaceChildren();
  if (!state.samples.length) {
    const empty = document.createElement("div");
    empty.className = "sample-placeholder";
    empty.textContent = state.query ? "没有找到匹配的样本" : "没有可显示的样本";
    elements.sampleList.append(empty);
  } else {
    for (const item of state.samples.slice(append ? appendFrom : 0)) elements.sampleList.append(sampleRow(item));
  }
  elements.loadMore.disabled = state.samples.length >= state.total;
  elements.loadMore.textContent = state.samples.length >= state.total ? `共 ${numberFormatter.format(state.total)} 条` : "加载更多";
}

async function loadSamples({ reset = true } = {}) {
  const dataset = state.dataset;
  const query = state.query;
  const listSerial = ++state.listSerial;
  const nextOffset = reset ? 0 : state.samples.length;
  const params = new URLSearchParams({
    dataset,
    q: query,
    limit: "40",
    offset: String(nextOffset),
  });
  elements.loadMore.disabled = true;
  try {
    const payload = await requestJson(`/api/samples?${params}`);
    if (listSerial !== state.listSerial || dataset !== state.dataset || query !== state.query) return;
    if (reset) state.samples = payload.items;
    else state.samples.push(...payload.items);
    state.total = payload.total;
    state.offset = payload.offset;
    renderSamples(!reset, nextOffset);
  } catch (error) {
    if (listSerial !== state.listSerial || dataset !== state.dataset || query !== state.query) return;
    showToast(`样本索引读取失败：${error.message}`);
  }
}

function clearSelection() {
  state.selectionSerial += 1;
  state.selectedId = null;
  state.detail = null;
  state.positions = null;
  state.frames = 0;
  state.currentFrame = 0;
  state.frameCursor = 0;
  state.annotationMode = "base";
  state.smplMeshFallback = false;
  state.smplMeshReady = false;
  meshErrorShown = false;
  lastQueuedFrame = -1;
  setPlaying(false);
  elements.activeSampleId.textContent = "Select a sample";
  elements.activeSampleMeta.textContent = "帧数、时长与表示信息会显示在这里";
  elements.evidenceBadges.replaceChildren();
  elements.viewerStatus.textContent = "等待选择";
  elements.viewportEmpty.hidden = false;
  elements.viewportEmpty.querySelector("strong").textContent = "从左侧选择样本";
  elements.viewportEmpty.querySelector("span:last-child").textContent = "载入后可旋转、平移、缩放与逐帧播放";
  elements.frameSlider.max = "0";
  elements.frameSlider.value = "0";
  elements.frameCounter.textContent = "000 / 000";
  elements.captionCount.textContent = "—";
  elements.captionContent.replaceChildren(Object.assign(document.createElement("p"), { className: "muted", textContent: "选择样本后显示 caption 或 temporal events。" }));
  elements.annotationTabs.forEach((button) => {
    const active = button.dataset.annotation === "base";
    button.disabled = button.dataset.annotation !== "base";
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  elements.video.removeAttribute("src");
  elements.video.load();
  updateVideoPanel();
  skeletonGroup.visible = false;
  surfaceGroup.visible = false;
  if (smplMesh) smplMesh.visible = false;
  surfaceProxyGroup.visible = true;
}

async function selectDataset(dataset) {
  if (dataset === state.dataset) return;
  state.dataset = dataset;
  document.body.dataset.dataset = dataset;
  state.query = "";
  elements.sampleSearch.value = "";
  elements.datasetTabs.forEach((button) => {
    const active = button.dataset.dataset === dataset;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  clearSelection();
  updateDatasetSummary();
  await loadSamples();
}

function renderCaption(caption) {
  elements.captionContent.replaceChildren();
  if (!caption) {
    elements.captionCount.textContent = "无标注";
    elements.captionContent.append(Object.assign(document.createElement("p"), { className: "muted", textContent: "这个样本没有匹配的 caption。" }));
    return;
  }
  elements.captionCount.textContent = caption.kind === "cot" ? `${caption.count} segments` : caption.kind === "clip" ? "1 clip" : `${caption.count} 条`;
  if (caption.kind === "events") {
    for (const event of caption.events) {
      const row = document.createElement("div");
      row.className = "caption-event";
      const time = document.createElement("time");
      time.textContent = `${Number(event.start_time).toFixed(2)}–${Number(event.end_time).toFixed(2)}s`;
      const description = document.createElement("span");
      description.textContent = event.description;
      row.append(time, description);
      elements.captionContent.append(row);
    }
  } else if (caption.kind === "lines") {
    for (const line of caption.lines) {
      const paragraph = document.createElement("p");
      paragraph.textContent = line;
      elements.captionContent.append(paragraph);
    }
  } else if (caption.kind === "clip") {
    for (const rawLine of caption.description.split("\n")) {
      const line = rawLine.trim();
      if (!line) continue;
      const clean = line.replaceAll("**", "").trim();
      const node = document.createElement(/^Overall|^Actors:?$|^Temporal|^Key Action/i.test(clean) ? "h4" : "p");
      if (node.tagName === "H4") node.className = "clip-heading";
      node.textContent = clean.replace(/^[-]\s*/, "");
      elements.captionContent.append(node);
    }
  } else if (caption.kind === "cot") {
    const summary = document.createElement("div");
    summary.className = "cot-summary";
    const summaryLabel = document.createElement("strong");
    summaryLabel.textContent = "Sample summary";
    const summaryText = document.createElement("p");
    summaryText.textContent = caption.sample_summary;
    summary.append(summaryLabel, summaryText);
    elements.captionContent.append(summary);
    caption.segments.forEach((segment) => {
      const details = document.createElement("details");
      details.className = "cot-segment";
      const title = document.createElement("summary");
      title.textContent = `${segment.time_range} · ${segment.cot_type || "fusion"}`;
      const thinkLabel = document.createElement("strong");
      thinkLabel.textContent = "Reasoning";
      const think = document.createElement("p");
      think.className = "cot-think";
      think.textContent = segment.think;
      const answerLabel = document.createElement("strong");
      answerLabel.textContent = "Aligned answer";
      const answer = document.createElement("p");
      answer.textContent = segment.answer;
      details.append(title, thinkLabel, think, answerLabel, answer);
      elements.captionContent.append(details);
    });
    const final = document.createElement("div");
    final.className = "cot-summary";
    const finalLabel = document.createElement("strong");
    finalLabel.textContent = "Final answer";
    const finalText = document.createElement("p");
    finalText.textContent = caption.final_answer;
    final.append(finalLabel, finalText);
    elements.captionContent.append(final);
  }
}

function annotationForMode(mode) {
  if (!state.detail) return null;
  if (mode === "clip") return state.detail.clip_caption;
  if (mode === "cot") return state.detail.cot;
  return state.detail.caption;
}

function renderActiveAnnotation() {
  renderCaption(annotationForMode(state.annotationMode));
}

function updateAnnotationTabs() {
  const baseLabels = { humanml3d: "Sequence text", sonic: "Temporal events", motionx: "Frame text" };
  const availability = {
    base: Boolean(state.detail?.caption),
    clip: Boolean(state.detail?.clip_caption),
    cot: Boolean(state.detail?.cot),
  };
  elements.annotationTabs.forEach((button) => {
    const mode = button.dataset.annotation;
    if (mode === "base") button.textContent = baseLabels[state.dataset];
    button.disabled = !availability[mode];
  });
  if (!availability[state.annotationMode]) {
    state.annotationMode = ["base", "clip", "cot"].find((mode) => availability[mode]) || "base";
  }
  elements.annotationTabs.forEach((button) => {
    const active = button.dataset.annotation === state.annotationMode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  renderActiveAnnotation();
}

function renderEvidence() {
  elements.evidenceBadges.replaceChildren();
  const badges = [];
  if (state.detail?.has_video) badges.push(["原始媒体", "evidence-source"]);
  badges.push(["确定性解码 · 22 joints", "evidence-decoded"]);
  if (state.smplhAvailable && state.smplMeshReady && !state.smplMeshFallback) badges.push(["派生近似 · SMPL-H", "evidence-derived"]);
  else if (state.smplhAvailable && !state.smplMeshFallback) badges.push(["已配置 · SMPL-H 待加载", "evidence-derived"]);
  else badges.push(["回退显示 · 关节表面代理", "evidence-derived"]);
  for (const [label, className] of badges) {
    const badge = document.createElement("span");
    badge.className = `evidence ${className}`;
    badge.textContent = label;
    elements.evidenceBadges.append(badge);
  }
}

function renderAssetFacts() {
  const facts = [
    ["源表示", `float32 (${state.frames},263)`],
    ["解码输出", `${state.frames} × 22 × 3`],
    ["时间", `${state.fps} FPS · ${(state.frames / state.fps).toFixed(2)}s`],
    ["坐标", "Y-up"],
    ["骨架", "22 joints · 21 bones"],
    ["网格", state.smplhAvailable ? "6,890V · 13,776F · 播放约 5 FPS" : "关节表面代理"],
  ];
  elements.assetFacts.replaceChildren();
  for (const [label, value] of facts) {
    const wrapper = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    wrapper.append(dt, dd);
    elements.assetFacts.append(wrapper);
  }
}

function updateVideoPanel() {
  const showVideo = state.dataset === "motionx";
  elements.videoCard.classList.toggle("is-hidden", !showVideo);
  elements.viewerGrid.classList.toggle("has-video", showVideo);
  if (!showVideo) return;
  const hasVideo = Boolean(state.detail?.has_video);
  elements.videoCard.classList.toggle("is-empty", !hasVideo);
  if (hasVideo) {
    elements.video.src = state.detail.video_url;
    elements.video.load();
  }
}

async function selectSample(sampleId) {
  if (sampleId === state.selectedId && state.positions) return;
  const dataset = state.dataset;
  const selectionSerial = ++state.selectionSerial;
  state.selectedId = sampleId;
  state.detail = null;
  state.positions = null;
  state.frames = 0;
  state.currentFrame = 0;
  state.frameCursor = 0;
  state.smplMeshFallback = false;
  state.smplMeshReady = false;
  setPlaying(false);
  renderSamples();
  elements.activeSampleId.textContent = sampleId;
  elements.activeSampleMeta.textContent = "正在恢复 263D 动作…";
  elements.viewerStatus.textContent = "解码动作";
  elements.viewportEmpty.hidden = true;
  skeletonGroup.visible = false;
  surfaceGroup.visible = false;
  if (smplMesh) smplMesh.visible = false;
  surfaceProxyGroup.visible = true;
  elements.video.removeAttribute("src");
  elements.video.load();
  updateVideoPanel();
  try {
    const [detail, response] = await Promise.all([
      requestJson(`/api/sample/${dataset}/${encodeURIComponent(sampleId)}`),
      fetch(`/api/motion/${dataset}/${encodeURIComponent(sampleId)}`),
    ]);
    if (!response.ok) throw new Error(await response.text());
    const buffer = await response.arrayBuffer();
    if (selectionSerial !== state.selectionSerial || dataset !== state.dataset || sampleId !== state.selectedId) return;
    const frames = Number(response.headers.get("X-Motion-Frames"));
    const joints = Number(response.headers.get("X-Motion-Joints"));
    if (joints !== 22 || buffer.byteLength !== frames * joints * 3 * 4) throw new Error("动作二进制尺寸不匹配");
    state.detail = detail;
    state.positions = new Float32Array(buffer);
    state.frames = frames;
    state.fps = Number(response.headers.get("X-Motion-Fps")) || detail.fps;
    state.currentFrame = 0;
    state.frameCursor = 0;
    state.floorOffset = findFloor(state.positions);
    elements.activeSampleMeta.textContent = `${frames} 帧 · ${(frames / state.fps).toFixed(2)} 秒 · ${state.fps} FPS · 263D → 22 joints`;
    elements.frameSlider.max = String(Math.max(0, frames - 1));
    elements.frameSlider.value = "0";
    updateAnnotationTabs();
    renderEvidence();
    renderAssetFacts();
    updateVideoPanel();
    updateFrame(0);
    if (state.viewMode === "surface") queueSmplFrame(0);
    resetCamera();
    elements.viewerStatus.textContent = "已就绪 · 暂停";
  } catch (error) {
    if (selectionSerial !== state.selectionSerial || dataset !== state.dataset || sampleId !== state.selectedId) return;
    elements.viewportEmpty.hidden = false;
    elements.viewportEmpty.querySelector("strong").textContent = "动作载入失败";
    elements.viewerStatus.textContent = "加载失败";
    showToast(`无法载入 ${sampleId}：${error.message}`);
  }
}

function findFloor(positions) {
  let minimum = Infinity;
  for (let index = 1; index < positions.length; index += 3) minimum = Math.min(minimum, positions[index]);
  return Number.isFinite(minimum) ? minimum : 0;
}

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x060c15, 7, 18);
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x060c15, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
elements.viewport.prepend(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 1, 0);
controls.minDistance = 1.2;
controls.maxDistance = 12;

const ambient = new THREE.HemisphereLight(0x9bc7ff, 0x101522, 2.1);
scene.add(ambient);
const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
keyLight.position.set(3, 5, 4);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x52d8c5, 1.5);
rimLight.position.set(-4, 2, -3);
scene.add(rimLight);

const grid = new THREE.GridHelper(8, 24, 0x2f5f76, 0x182a3d);
grid.material.transparent = true;
grid.material.opacity = 0.72;
scene.add(grid);

const skeletonGroup = new THREE.Group();
const surfaceGroup = new THREE.Group();
const surfaceProxyGroup = new THREE.Group();
surfaceGroup.add(surfaceProxyGroup);
scene.add(skeletonGroup, surfaceGroup);

const jointGeometry = new THREE.SphereGeometry(0.035, 10, 8);
const jointMaterials = [0x52d8c5, 0x6ba9ff, 0xff8068, 0xf2bd62, 0xc58cff].map((color) => new THREE.MeshStandardMaterial({ color, roughness: 0.42 }));
const jointMeshes = Array.from({ length: 22 }, (_, index) => {
  const chainIndex = CHAINS.findIndex((chain) => chain.includes(index));
  const mesh = new THREE.Mesh(jointGeometry, jointMaterials[Math.max(0, chainIndex)]);
  skeletonGroup.add(mesh);
  return mesh;
});

const linePositions = new Float32Array(EDGES.length * 2 * 3);
const lineGeometry = new THREE.BufferGeometry();
lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
const lineMaterial = new THREE.LineBasicMaterial({ color: 0xd9e9f7, transparent: true, opacity: 0.9 });
const boneLines = new THREE.LineSegments(lineGeometry, lineMaterial);
skeletonGroup.add(boneLines);

const surfaceMaterial = new THREE.MeshStandardMaterial({ color: 0x6ba9ff, roughness: 0.55, metalness: 0.03 });
const torsoMaterial = new THREE.MeshStandardMaterial({ color: 0x52d8c5, roughness: 0.5, metalness: 0.02 });
const cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 10, 1, false);
const surfaceBones = EDGES.map(([a, b]) => {
  const torso = (a === 0 && b === 3) || (a >= 3 && a <= 12 && b <= 15);
  const mesh = new THREE.Mesh(cylinderGeometry, torso ? torsoMaterial : surfaceMaterial);
  mesh.userData.edge = [a, b];
  surfaceProxyGroup.add(mesh);
  return mesh;
});
const surfaceJointGeometry = new THREE.SphereGeometry(1, 14, 10);
const surfaceJoints = Array.from({ length: 22 }, (_, index) => {
  const mesh = new THREE.Mesh(surfaceJointGeometry, index === 15 ? torsoMaterial : surfaceMaterial);
  surfaceProxyGroup.add(mesh);
  return mesh;
});

skeletonGroup.visible = false;
surfaceGroup.visible = false;
let smplMesh = null;
let smplGeometry = null;
let smplTopologyPromise = null;
let meshRequestedFrame = null;
let meshLoading = false;
let meshRequestSerial = 0;
let lastQueuedFrame = -1;
let meshErrorShown = false;
const pointA = new THREE.Vector3();
const pointB = new THREE.Vector3();
const direction = new THREE.Vector3();
const midpoint = new THREE.Vector3();
const up = new THREE.Vector3(0, 1, 0);

function jointAt(frame, joint, target) {
  const offset = (frame * 22 + joint) * 3;
  const rootOffset = frame * 22 * 3;
  const rootX = state.positions[rootOffset];
  const rootZ = state.positions[rootOffset + 2];
  target.set(
    state.positions[offset] - rootX,
    state.positions[offset + 1] - state.floorOffset,
    state.positions[offset + 2] - rootZ,
  );
  return target;
}

function radiusForJoint(index) {
  if (index === 15) return 0.14;
  if ([0, 3, 6, 9, 12].includes(index)) return 0.085;
  if ([1, 2, 4, 5, 7, 8].includes(index)) return 0.07;
  return 0.048;
}

function radiusForEdge(a, b) {
  if ((a === 0 && b === 3) || [3, 6, 9, 12].includes(a)) return 0.105;
  if ([0, 1, 2, 4, 5].includes(a)) return 0.075;
  return 0.052;
}

function updateFrame(frame) {
  if (!state.positions || !state.frames) return;
  state.currentFrame = Math.max(0, Math.min(state.frames - 1, Math.round(frame)));
  let lineOffset = 0;
  for (let index = 0; index < jointMeshes.length; index += 1) {
    jointAt(state.currentFrame, index, jointMeshes[index].position);
    surfaceJoints[index].position.copy(jointMeshes[index].position);
    const radius = radiusForJoint(index);
    surfaceJoints[index].scale.setScalar(radius);
  }
  EDGES.forEach(([a, b], edgeIndex) => {
    pointA.copy(jointMeshes[a].position);
    pointB.copy(jointMeshes[b].position);
    linePositions[lineOffset++] = pointA.x;
    linePositions[lineOffset++] = pointA.y;
    linePositions[lineOffset++] = pointA.z;
    linePositions[lineOffset++] = pointB.x;
    linePositions[lineOffset++] = pointB.y;
    linePositions[lineOffset++] = pointB.z;

    direction.subVectors(pointB, pointA);
    const length = Math.max(0.001, direction.length());
    midpoint.addVectors(pointA, pointB).multiplyScalar(0.5);
    const mesh = surfaceBones[edgeIndex];
    mesh.position.copy(midpoint);
    mesh.quaternion.setFromUnitVectors(up, direction.normalize());
    const radius = radiusForEdge(a, b);
    mesh.scale.set(radius, length, radius);
  });
  lineGeometry.attributes.position.needsUpdate = true;
  skeletonGroup.visible = state.viewMode === "skeleton";
  surfaceGroup.visible = state.viewMode === "surface";
  elements.frameSlider.value = String(state.currentFrame);
  const width = Math.max(3, String(state.frames).length);
  elements.frameCounter.textContent = `${String(state.currentFrame + 1).padStart(width, "0")} / ${String(state.frames).padStart(width, "0")}`;
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
      const buffer = await response.arrayBuffer();
      const indices = new Uint32Array(buffer);
      if (!faces || indices.length !== faces * 3) throw new Error("SMPL-H topology size mismatch");
      smplGeometry = new THREE.BufferGeometry();
      smplGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
      const material = new THREE.MeshStandardMaterial({
        color: 0x9fc8e8,
        roughness: 0.52,
        metalness: 0.02,
        side: THREE.DoubleSide,
      });
      smplMesh = new THREE.Mesh(smplGeometry, material);
      smplMesh.visible = false;
      surfaceGroup.add(smplMesh);
    })();
  }
  return smplTopologyPromise;
}

function queueSmplFrame(frame) {
  if (!state.smplhAvailable || !state.selectedId || state.viewMode !== "surface") return;
  meshRequestedFrame = Math.max(0, Math.min(state.frames - 1, Math.round(frame)));
  if (!meshLoading) processSmplQueue();
}

async function processSmplQueue() {
  meshLoading = true;
  while (meshRequestedFrame !== null && state.selectedId) {
    const frame = meshRequestedFrame;
    meshRequestedFrame = null;
    const dataset = state.dataset;
    const sampleId = state.selectedId;
    const serial = ++meshRequestSerial;
    try {
      await ensureSmplMesh();
      const response = await fetch(`/api/smplh/mesh/${dataset}/${encodeURIComponent(sampleId)}/${frame}`);
      if (!response.ok) throw new Error(await response.text());
      const count = Number(response.headers.get("X-SMPLH-Vertices"));
      const buffer = await response.arrayBuffer();
      const vertices = new Float32Array(buffer);
      if (vertices.length !== count * 3) throw new Error("SMPL-H vertex size mismatch");
      if (sampleId !== state.selectedId || dataset !== state.dataset || serial !== meshRequestSerial) continue;
      smplGeometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
      smplGeometry.computeVertexNormals();
      smplGeometry.computeBoundingSphere();
      smplMesh.visible = true;
      surfaceProxyGroup.visible = false;
      state.smplMeshReady = true;
      if (state.smplMeshFallback) {
        state.smplMeshFallback = false;
        renderEvidence();
      }
    } catch (error) {
      surfaceProxyGroup.visible = true;
      if (smplMesh) smplMesh.visible = false;
      state.smplMeshFallback = true;
      state.smplMeshReady = false;
      renderEvidence();
      if (state.viewMode === "surface") elements.viewerStatus.textContent = "关节表面代理 · 网格不可用";
      if (!meshErrorShown) {
        showToast(`SMPL-H 网格暂时不可用，已显示关节表面代理：${error.message}`);
        meshErrorShown = true;
      }
    }
  }
  meshLoading = false;
}

function resetCamera() {
  camera.position.set(2.8, 2.25, 3.7);
  controls.target.set(0, 0.95, 0);
  controls.update();
}
resetCamera();

function setCameraPreset(preset) {
  const target = new THREE.Vector3(0, 0.95, 0);
  if (preset === "front") camera.position.set(0, 1.55, 4.4);
  else if (preset === "side") camera.position.set(4.4, 1.55, 0);
  else if (preset === "top") camera.position.set(0.01, 5.4, 0.01);
  controls.target.copy(target);
  controls.update();
}

function setPlaying(value) {
  state.playing = Boolean(value && state.frames);
  state.lastTimestamp = performance.now();
  elements.playToggle.classList.toggle("is-playing", state.playing);
  elements.playToggle.setAttribute("aria-label", state.playing ? "暂停动作" : "播放动作");
  elements.viewerStatus.textContent = state.playing ? "播放中" : state.frames ? "已就绪 · 暂停" : "等待选择";
  if (state.dataset === "motionx" && state.detail?.has_video && state.detail?.sync_verified) {
    if (state.playing) elements.video.play().catch(() => {});
    else elements.video.pause();
  }
}

function animate(timestamp) {
  requestAnimationFrame(animate);
  const deltaSeconds = Math.min(0.1, (timestamp - state.lastTimestamp) / 1000 || 0);
  state.lastTimestamp = timestamp;
  if (state.playing && state.frames) {
    let nextFrame;
    if (state.dataset === "motionx" && state.detail?.has_video && state.detail?.sync_verified && !elements.video.paused) {
      nextFrame = elements.video.currentTime * state.fps;
      state.frameCursor = nextFrame;
    } else {
      state.frameCursor += deltaSeconds * state.fps * state.speed;
      nextFrame = state.frameCursor;
    }
    if (nextFrame >= state.frames) {
      if (state.loop) {
        if (state.dataset === "motionx" && state.detail?.has_video && state.detail?.sync_verified) elements.video.currentTime = 0;
        nextFrame = 0;
        state.frameCursor = 0;
      } else {
        nextFrame = state.frames - 1;
        state.frameCursor = nextFrame;
        setPlaying(false);
      }
    }
    updateFrame(nextFrame);
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

elements.datasetTabs.forEach((button) => button.addEventListener("click", () => selectDataset(button.dataset.dataset)));
elements.modeButtons.forEach((button) => button.addEventListener("click", () => {
  state.viewMode = button.dataset.viewMode;
  elements.modeButtons.forEach((item) => {
    const active = item.dataset.viewMode === state.viewMode;
    item.classList.toggle("is-active", active);
    item.setAttribute("aria-pressed", String(active));
  });
  if (state.positions) updateFrame(state.currentFrame);
  if (state.viewMode === "surface") queueSmplFrame(state.currentFrame);
}));
elements.sampleSearch.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.query = elements.sampleSearch.value.trim();
    loadSamples();
  }, 220);
});
elements.loadMore.addEventListener("click", () => loadSamples({ reset: false }));
elements.playToggle.addEventListener("click", () => setPlaying(!state.playing));
elements.prevFrame.addEventListener("click", () => {
  setPlaying(false);
  state.frameCursor = Math.max(0, state.currentFrame - 1);
  updateFrame(state.frameCursor);
  if (state.dataset === "motionx" && state.detail?.has_video && state.detail?.sync_verified) elements.video.currentTime = state.currentFrame / state.fps;
});
elements.nextFrame.addEventListener("click", () => {
  setPlaying(false);
  state.frameCursor = Math.min(Math.max(0, state.frames - 1), state.currentFrame + 1);
  updateFrame(state.frameCursor);
  if (state.dataset === "motionx" && state.detail?.has_video && state.detail?.sync_verified) elements.video.currentTime = state.currentFrame / state.fps;
});
elements.frameSlider.addEventListener("input", () => {
  setPlaying(false);
  state.frameCursor = Number(elements.frameSlider.value);
  updateFrame(state.frameCursor);
  if (state.dataset === "motionx" && state.detail?.has_video && state.detail?.sync_verified) elements.video.currentTime = state.currentFrame / state.fps;
});
elements.speedSelect.addEventListener("change", () => {
  state.speed = Number(elements.speedSelect.value);
  if (state.detail?.sync_verified) elements.video.playbackRate = state.speed;
});
elements.loopToggle.addEventListener("change", () => {
  state.loop = elements.loopToggle.checked;
  elements.video.loop = state.loop;
});
elements.resetView.addEventListener("click", resetCamera);
elements.cameraButtons.forEach((button) => button.addEventListener("click", () => setCameraPreset(button.dataset.camera)));
elements.annotationTabs.forEach((button) => button.addEventListener("click", () => {
  if (button.disabled) return;
  state.annotationMode = button.dataset.annotation;
  updateAnnotationTabs();
}));
elements.video.addEventListener("play", () => {
  if (state.frames && state.detail?.sync_verified) {
    state.playing = true;
    elements.playToggle.classList.add("is-playing");
    elements.viewerStatus.textContent = "播放中";
  }
});
elements.video.addEventListener("pause", () => {
  if (state.detail?.sync_verified) {
    state.playing = false;
    elements.playToggle.classList.remove("is-playing");
    if (state.frames) elements.viewerStatus.textContent = "已就绪 · 暂停";
  }
});

const sectionLinks = [...document.querySelectorAll(".docs-sidebar a[href^='#']")];
const sections = sectionLinks.map((link) => document.querySelector(link.getAttribute("href"))).filter(Boolean);
if ("IntersectionObserver" in window) {
  const sectionObserver = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    sectionLinks.forEach((link) => link.classList.toggle("is-current", link.getAttribute("href") === `#${visible.target.id}`));
  }, { rootMargin: "-18% 0px -68% 0px", threshold: [0, 0.1, 0.5] });
  sections.forEach((section) => sectionObserver.observe(section));
}
elements.video.addEventListener("seeked", () => {
  if (state.frames && state.detail?.sync_verified) {
    state.frameCursor = Math.min(state.frames - 1, elements.video.currentTime * state.fps);
    updateFrame(state.frameCursor);
  }
});

async function boot() {
  try {
    elements.video.loop = state.loop;
    state.summary = await requestJson("/api/summary");
    state.smplhAvailable = Boolean(state.summary.smplh?.available);
    const surfaceButton = elements.modeButtons.find((button) => button.dataset.viewMode === "surface");
    if (surfaceButton && !state.smplhAvailable) surfaceButton.textContent = "表面代理";
    renderSummary();
    await loadSamples();
    elements.indexStatus.textContent = `${numberFormatter.format(Object.values(state.summary.indexed_samples).reduce((sum, value) => sum + value, 0))} 条样本已索引`;
    elements.statusDot.classList.add("is-ready");
    const first = state.samples[0];
    if (first) await selectSample(first.id);
  } catch (error) {
    elements.indexStatus.textContent = "索引不可用";
    showToast(`页面初始化失败：${error.message}`);
  }
}

boot();
