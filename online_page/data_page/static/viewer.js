import * as THREE from "/static/vendor/three.module.js?v=20260901-5";
import { OrbitControls } from "/static/vendor/OrbitControls.js?v=20260901-5";
import { PAGE_SIZE, analyzeMotion, displayedJoint, meshTranslation, adjacentSample, readLocation, locationQuery } from "/static/viewer-state.mjs?v=20261020-2";
import { requestData } from "/static/viewer-request.mjs?v=20261020-3";

const DATASET_LABELS = { humanml3d: "HumanML3D", sonic: "SONIC", motionx: "Motion-X" };
const CHAINS = [[0,2,5,8,11], [0,1,4,7,10], [0,3,6,9,12,15], [9,14,17,19,21], [9,13,16,18,20]];
const EDGES = CHAINS.flatMap(chain => chain.slice(0,-1).map((joint,i) => [joint,chain[i+1]]));
const state = {
  ...readLocation(location.search), summary: null, samples: [], total: 0, listLoading: false,
  detail: null, positions: null, analysis: null, frames: 0, fps: 20, currentFrame: 0, frameCursor: 0,
  floorOffset: 0, playing: false, speed: 1, loop: true, lastTimestamp: performance.now(),
  viewMode: "skeleton", textMode: "base", textQuery: "", textSerial: 0, listSerial: 0, selectionSerial: 0,
  smplhAvailable: false, smplReady: false, smplFailed: false, meshFrame: 0, follow: false, showPath: true,
};
const ids = ["serverDot","serverStatus","resultCount","sampleSearch","assetFilter","sampleList","clearSearch","randomSample","prevPage","nextPage","pageNumber","pageTotal","activeDataset","activeSample","sampleMeta","prevSample","nextSample","copyLink","meshState","viewport","viewportEmpty","rootReadout","rootNote","rootDistance","rootMode","showTrajectory","followRoot","fitMotion","resetCamera","prevFrame","playToggle","nextFrame","frameCount","timeCount","frameSlider","speedSelect","loopToggle","sourceVideo","videoEmpty","videoState","videoPanel","syncNote","textState","textSearch","copyText","expandText","textContent","evidenceList","toast"];
const elements = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
elements.datasetButtons = [...document.querySelectorAll(".dataset-button")];
elements.viewButtons = [...document.querySelectorAll(".view-button")];
elements.cameraButtons = [...document.querySelectorAll("[data-camera]")];
elements.textTabs = [...document.querySelectorAll(".text-tab")];
elements.videoWrap = document.querySelector(".video-wrap");
elements.textPanel = document.querySelector(".text-panel");
const numberFormatter = new Intl.NumberFormat("zh-CN");
let toastTimer, searchTimer, textSearchTimer, listController, selectionController;

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove("is-visible"), 4200);
}
async function requestJson(url, signal) {
  return (await requestData(url, { signal })).body;
}
function syncLocation() { history.replaceState(null, "", location.pathname + locationQuery(state)); }
function updateDatasetMeta() {
  elements.activeDataset.textContent = DATASET_LABELS[state.dataset];
  elements.datasetButtons.forEach(button => {
    const active = button.dataset.dataset === state.dataset;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  for (const option of elements.assetFilter.options) {
    option.disabled = state.dataset !== "motionx" && ["video", "cot"].includes(option.value);
  }
}
function updateNavigation() {
  const index = state.samples.findIndex(sample => sample.id === state.selectedId);
  const busy = state.listLoading;
  elements.prevSample.disabled = busy || !adjacentSample(state.page, index, -1, state.total);
  elements.nextSample.disabled = busy || !adjacentSample(state.page, index, 1, state.total);
  elements.prevPage.disabled = busy || !state.page;
  elements.nextPage.disabled = busy || (state.page + 1) * PAGE_SIZE >= state.total;
  elements.randomSample.disabled = busy || !state.total;
  elements.pageNumber.disabled = busy || !state.total;
  elements.pageNumber.value = String(state.page + 1);
  elements.pageNumber.max = String(Math.max(1, Math.ceil(state.total / PAGE_SIZE)));
  elements.pageTotal.textContent = `/ ${numberFormatter.format(Math.max(1, Math.ceil(state.total / PAGE_SIZE)))} 页`;
  elements.copyLink.disabled = !state.detail;
  for (const key of ["playToggle", "prevFrame", "nextFrame", "frameSlider"]) elements[key].disabled = !state.positions;
  elements.sampleList.setAttribute("aria-busy", String(busy));
}
function sampleRow(sample) {
  const button = document.createElement("button");
  button.type = "button"; button.className = "sample-row"; button.dataset.id = sample.id;
  button.title = sample.id;
  const title = document.createElement("strong"); title.textContent = sample.id;
  const meta = document.createElement("span"); meta.textContent = `${(sample.frames / sample.fps).toFixed(1)} s · ${numberFormatter.format(sample.frames)} 帧`;
  const chips = document.createElement("div"); chips.className = "asset-chips";
  for (const [available, label, cls] of [[sample.has_video,"视频","is-video"],[sample.has_caption || sample.has_clip_caption,"文本",""],[sample.has_cot,"CoT",""]]) {
    if (available) { const chip = document.createElement("i"); chip.textContent = label; chip.className = cls; chips.append(chip); }
  }
  button.append(title,meta,chips);
  button.addEventListener("click", () => selectSample(sample.id));
  return button;
}
function highlightSelection(scroll = false) {
  for (const row of elements.sampleList.querySelectorAll(".sample-row")) {
    const active = row.dataset.id === state.selectedId;
    row.classList.toggle("is-active", active);
    row.setAttribute("aria-current", String(active));
    if (active && scroll) row.scrollIntoView({ block: "nearest" });
  }
}
function listMessage(title, hint, retry = false) {
  elements.sampleList.replaceChildren();
  const box = document.createElement("div"); box.className = "list-state";
  const strong = document.createElement("strong"); strong.textContent = title;
  const text = document.createElement("span"); text.textContent = hint; box.append(strong,text);
  if (retry) { const button = document.createElement("button"); button.type = "button"; button.textContent = "重新加载"; button.addEventListener("click", typeof retry === "function" ? retry : () => loadSamples()); box.append(button); }
  elements.sampleList.append(box);
}
function clearSample() {
  selectionController?.abort();
  state.selectionSerial++; state.textSerial++;
  state.selectedId = null; state.detail = null; state.positions = null; state.analysis = null;
  state.frames = 0; state.currentFrame = 0; state.frameCursor = 0;
  state.textQuery = ""; state.textMode = "base"; state.smplReady = false; state.smplFailed = false;
  pendingMesh = null;
  setPlaying(false);
  skeletonGroup.visible = false; pathLine.visible = false; progressLine.visible = false; rootMarker.visible = false;
  if (smplMesh) smplMesh.visible = false;
  elements.activeSample.textContent = "选择样本"; elements.sampleMeta.textContent = "从左侧选择一条动作。";
  elements.viewportEmpty.hidden = false;
  elements.viewportEmpty.querySelector(".retry-motion")?.remove();
  elements.viewportEmpty.querySelector("strong").textContent = "选择一条动作";
  elements.viewportEmpty.querySelector("p").textContent = "动作与根节点轨迹会显示在这里。";
  elements.frameSlider.value = "0"; elements.frameSlider.max = "0";
  elements.frameCount.textContent = "0 / 0 帧"; elements.timeCount.textContent = "0.00 / 0.00 s";
  elements.rootReadout.textContent = ""; elements.rootDistance.textContent = "—";
  elements.textSearch.value = ""; elements.textState.textContent = "—";
  elements.sourceVideo.removeAttribute("src"); elements.sourceVideo.load();
  elements.videoWrap.classList.add("is-empty"); elements.videoPanel.open = false;
  elements.videoState.textContent = "无视频"; elements.syncNote.textContent = "该样本没有关联视频。";
  elements.textTabs.forEach(button => { button.disabled = true; });
  renderEmptyText(); renderEvidence(); updateNavigation();
}
async function loadSamples({ page = state.page, pick = 0, wantedId = null } = {}) {
  clearSample(); listController?.abort(); listController = new AbortController();
  const serial = ++state.listSerial, dataset = state.dataset, query = state.query, filter = state.filter;
  state.listLoading = true; state.samples = []; state.page = Math.max(0,page);
  listMessage("正在读取样本…", ""); updateNavigation();
  const params = new URLSearchParams({ dataset, q: query, limit: String(PAGE_SIZE), offset: String(state.page * PAGE_SIZE) });
  if (filter !== "all") params.set({ video:"has_video", text:"has_text", cot:"has_cot" }[filter], "1");
  try {
    const payload = await requestJson(`/api/samples?${params}`, listController.signal);
    if (serial !== state.listSerial) return;
    state.total = payload.total;
    const lastPage = Math.max(0, Math.ceil(state.total / PAGE_SIZE) - 1);
    if (state.page > lastPage) return loadSamples({ page: lastPage, wantedId });
    state.samples = payload.items; state.listLoading = false;
    elements.resultCount.textContent = `${numberFormatter.format(state.total)} 条`;
    elements.sampleList.replaceChildren(...state.samples.map(sampleRow));
    elements.sampleList.scrollTop = 0;
    if (!state.samples.length) { listMessage("没有匹配的样本", "试试其他 ID，或清空搜索和筛选。"); syncLocation(); }
    else {
      let id = state.samples.some(sample => sample.id === wantedId) ? wantedId : null;
      if (wantedId && !id && query !== wantedId) {
        state.query = wantedId; elements.sampleSearch.value = wantedId;
        return loadSamples({ page: 0, wantedId });
      }
      id ||= state.samples[pick === "last" ? state.samples.length - 1 : Math.min(pick,state.samples.length - 1)].id;
      await selectSample(id);
    }
  } catch (error) {
    if (serial !== state.listSerial || error.name === "AbortError") return;
    state.listLoading = false; state.total = 0; state.samples = [];
    elements.resultCount.textContent = "加载失败";
    listMessage("样本列表暂时不可用", `${error.message}。请重试。`, true);
  } finally { if (serial === state.listSerial) updateNavigation(); }
}
async function selectDataset(dataset) {
  if (dataset === state.dataset) return;
  clearTimeout(searchTimer); state.dataset = dataset; state.query = ""; state.filter = "all";
  elements.sampleSearch.value = ""; elements.assetFilter.value = "all";
  updateDatasetMeta(); await loadSamples({ page:0 });
}
async function selectSample(sampleId) {
  if (sampleId === state.selectedId && state.positions) return;
  clearSample(); selectionController = new AbortController();
  const serial = state.selectionSerial, dataset = state.dataset, signal = selectionController.signal;
  state.selectedId = sampleId;
  elements.activeSample.textContent = sampleId;
  elements.sampleMeta.textContent = "正在加载动作与标注…";
  elements.viewportEmpty.querySelector("strong").textContent = "正在加载动作…";
  elements.viewportEmpty.querySelector("p").textContent = "";
  elements.meshState.textContent = "加载中";
  highlightSelection(true); updateNavigation(); syncLocation();
  try {
    const [detail,response] = await Promise.all([
      requestJson(`/api/sample/${dataset}/${encodeURIComponent(sampleId)}`,signal),
      requestData(`/api/motion/${dataset}/${encodeURIComponent(sampleId)}`,{signal,kind:"buffer"}),
    ]);
    const buffer = response.body;
    if (serial !== state.selectionSerial) return;
    const frames = Number(response.headers.get("X-Motion-Frames"));
    if (!frames || Number(response.headers.get("X-Motion-Joints")) !== 22 || buffer.byteLength !== frames * 22 * 3 * 4) throw new Error("动作数据尺寸不正确");
    const positions = new Float32Array(buffer), analysis = analyzeMotion(positions);
    state.detail = detail; state.positions = positions; state.analysis = analysis;
    state.frames = frames; state.fps = Number(response.headers.get("X-Motion-Fps")) || detail.fps;
    state.floorOffset = analysis.floor;
    elements.sampleMeta.textContent = `${numberFormatter.format(frames)} 帧 · ${(frames/state.fps).toFixed(2)} s · ${state.fps} FPS`;
    elements.frameSlider.max = String(frames-1); elements.viewportEmpty.hidden = true;
    elements.rootDistance.textContent = `水平轨迹 ${analysis.pathLength.toFixed(2)} m · 净位移 ${analysis.netDistance.toFixed(2)} m`;
    setTrajectory(analysis.trajectory); updateVideo(); updateTextTabs(); renderEvidence();
    updateFrame(0); resetCamera(); setPlaying(false); updateNavigation();
  } catch (error) {
    if (serial !== state.selectionSerial || error.name === "AbortError") return;
    state.positions = null; state.frames = 0; state.smplReady = false;
    skeletonGroup.visible = false; pathLine.visible = false; progressLine.visible = false; rootMarker.visible = false;
    if (smplMesh) smplMesh.visible = false;
    elements.viewportEmpty.hidden = false;
    elements.viewportEmpty.querySelector("strong").textContent = "动作加载失败";
    elements.viewportEmpty.querySelector("p").textContent = `${error.message}。`;
    const retry = document.createElement("button");
    retry.type = "button"; retry.className = "retry-motion"; retry.textContent = "重新加载这条动作";
    retry.addEventListener("click", () => selectSample(sampleId));
    elements.viewportEmpty.append(retry);
    elements.meshState.textContent = "加载失败"; elements.sampleMeta.textContent = "请重新选择样本重试。";
    updateNavigation();
    showToast("动作或标注读取失败，请重试。");
  }
}
async function navigateSample(delta) {
  const target = adjacentSample(state.page, state.samples.findIndex(s => s.id === state.selectedId),delta,state.total);
  if (!target || state.listLoading) return;
  if (target.page !== state.page) await loadSamples({ page:target.page, pick:target.index });
  else await selectSample(state.samples[target.index].id);
}
function updateVideo() {
  const hasVideo = Boolean(state.detail?.has_video);
  elements.videoWrap.classList.toggle("is-empty", !hasVideo); elements.videoPanel.open = hasVideo;
  elements.sourceVideo.removeAttribute("src");
  if (hasVideo) elements.sourceVideo.src = state.detail.video_url;
  elements.sourceVideo.load(); elements.videoState.textContent = hasVideo ? "可播放" : "无视频";
  elements.syncNote.textContent = hasVideo ? "视频按样本 ID 关联，时间起点尚未核实；与动作独立播放。" : "该样本没有关联视频。";
}
function renderEmptyText() {
  const text = document.createElement("p"); text.className = "empty-note";
  text.textContent = state.detail ? "该样本没有此类文本。" : "选择样本后查看标注。";
  elements.textContent.replaceChildren(text);
}
function textForMode(mode) {
  return state.detail ? ({base:state.detail.caption, clip:state.detail.clip_caption, cot:state.detail.cot}[mode]) : null;
}
function updateTextTabs() {
  const available = {base:Boolean(state.detail?.caption),clip:Boolean(state.detail?.clip_caption),cot:Boolean(state.detail?.cot)};
  if (!available[state.textMode]) state.textMode = ["base","clip","cot"].find(mode => available[mode]) || "base";
  elements.textTabs.forEach(button => {
    const mode = button.dataset.text, active = mode === state.textMode;
    button.hidden = !available[mode] && mode !== "base"; button.disabled = !available[mode];
    if (mode === "base") button.textContent = {humanml3d:"描述",sonic:"事件",motionx:"逐帧"}[state.dataset];
    button.classList.toggle("is-active",active); button.setAttribute("aria-pressed",String(active));
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
    jump.textContent = `第 ${parsed.frame} 帧 · ${parsed.time.toFixed(2)}s`;
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
    load.textContent = `加载更多 · ${numberFormatter.format(caption.lines.length)} / ${numberFormatter.format(caption.count)}`;
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
    elements.textState.textContent = "无文本";
    renderEmptyText();
    return;
  }
  elements.textState.textContent = caption.kind === "cot" ? `${caption.count} 段` : caption.kind === "clip" ? "片段描述" : `${caption.lines?.length || caption.count} / ${caption.count}`;
  if (caption.kind === "events") {
    for (const event of caption.events) {
      if (state.textQuery && !event.description.toLowerCase().includes(state.textQuery.toLowerCase())) continue;
      const row = document.createElement("div");
      row.className = "text-event";
      const time = document.createElement("button");
      time.type = "button";
      time.textContent = `${Number(event.start_time).toFixed(2)}–${Number(event.end_time).toFixed(2)}s`;
      time.title = "跳到事件起点";
      time.addEventListener("click", () => seek(Math.round(Number(event.start_time) * state.fps)));
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
      title.textContent = "没有匹配的文本";
      const description = document.createElement("span");
      description.textContent = "试试其他关键词。";
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
    summaryTitle.textContent = "概述";
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
      reasoningTitle.textContent = "推理";
      const reasoning = document.createElement("p");
      reasoning.textContent = segment.think;
      const answerTitle = document.createElement("strong");
      answerTitle.textContent = "答案";
      const answer = document.createElement("p");
      answer.textContent = segment.answer;
      details.append(title, reasoningTitle, reasoning, answerTitle, answer);
      elements.textContent.append(details);
    }
    const final = document.createElement("div");
    final.className = "cot-summary";
    const finalTitle = document.createElement("strong");
    finalTitle.textContent = "最终答案";
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
    showToast(`文本加载失败：${error.message}`);
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


function renderEvidence() {
  const kinds = state.detail ? [state.detail.caption,state.detail.clip_caption,state.detail.cot].filter(Boolean).length : 0;
  const entries = [["动作",state.positions ? "263D → 22 关节" : "—"],["Root",state.positions ? "位移 / 高度 / 转向" : "—"],["文本",state.detail ? `${kinds} 类标注` : "—"],["人体网格",state.smplFailed ? "暂不可用，显示骨架" : state.smplReady ? "SMPL-H 重建" : state.smplhAvailable ? "可按需加载" : "不可用"]];
  elements.evidenceList.replaceChildren();
  for (const [label,value] of entries) {
    const row = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent=label; dd.textContent=value; row.append(dt,dd); elements.evidenceList.append(row);
  }
}

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38,1,0.01,100);
const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:false, powerPreference:"high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1,2));
renderer.setClearColor(0xf6f8fb,1); renderer.outputColorSpace=THREE.SRGBColorSpace;
renderer.domElement.setAttribute("aria-label","三维动作画布");
renderer.domElement.tabIndex=0;
elements.viewport.prepend(renderer.domElement);
const controls = new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true; controls.dampingFactor=.08; controls.screenSpacePanning=true;
controls.minDistance=.8; controls.maxDistance=100;
scene.add(new THREE.HemisphereLight(0xffffff,0xc3cbd8,2.4));
const keyLight = new THREE.DirectionalLight(0xffffff,2.5); keyLight.position.set(4,7,5); scene.add(keyLight);
const grid = new THREE.GridHelper(12,24,0xb6c2d3,0xe0e6ee); scene.add(grid);
const skeletonGroup = new THREE.Group(); scene.add(skeletonGroup);
const jointGeometry = new THREE.SphereGeometry(.034,12,8);
const jointMaterial = new THREE.MeshStandardMaterial({color:0x44699e,roughness:.7});
const jointMeshes = Array.from({length:22},(_,index) => {
  const material = index === 0 ? new THREE.MeshStandardMaterial({color:0xe18c46,roughness:.6}) : jointMaterial;
  const mesh = new THREE.Mesh(index === 15 ? new THREE.SphereGeometry(.075,16,12) : jointGeometry,material);
  skeletonGroup.add(mesh); return mesh;
});
const boneGeometry = new THREE.CylinderGeometry(.014,.014,1,8);
const boneMaterial = new THREE.MeshStandardMaterial({color:0x567caf,roughness:.75});
const boneMeshes = EDGES.map(() => { const mesh = new THREE.Mesh(boneGeometry,boneMaterial); skeletonGroup.add(mesh); return mesh; });
skeletonGroup.visible=false;
const pathGeometry = new THREE.BufferGeometry();
const pathLine = new THREE.Line(pathGeometry,new THREE.LineBasicMaterial({color:0xbacbe3}));
const progressGeometry = new THREE.BufferGeometry();
const progressLine = new THREE.Line(progressGeometry,new THREE.LineBasicMaterial({color:0x3569d4}));
pathLine.frustumCulled=false; progressLine.frustumCulled=false; pathLine.visible=false; progressLine.visible=false;
scene.add(pathLine,progressLine);
const rootMarker = new THREE.Mesh(new THREE.RingGeometry(.065,.085,24),new THREE.MeshBasicMaterial({color:0xd2843f,side:THREE.DoubleSide}));
rootMarker.rotation.x=-Math.PI/2; rootMarker.visible=false; scene.add(rootMarker);
const up = new THREE.Vector3(0,1,0), direction = new THREE.Vector3(), followAnchor = new THREE.Vector3();
let smplGeometry, smplMesh, smplTopologyPromise, pendingMesh=null, meshBusy=false, lastQueuedFrame=-1;

function setTrajectory(positions) {
  pathGeometry.dispose(); progressGeometry.dispose();
  pathGeometry.setAttribute("position",new THREE.BufferAttribute(positions,3));
  progressGeometry.setAttribute("position",new THREE.BufferAttribute(positions.slice(),3));
  pathGeometry.computeBoundingSphere(); progressGeometry.computeBoundingSphere();
}
function meshPosition() {
  if (smplMesh && state.positions) smplMesh.position.set(...meshTranslation(state.positions,state.meshFrame,state.rootMode));
}
function updateFrame(frame) {
  if (!state.positions) return;
  state.currentFrame = Math.max(0,Math.min(state.frames-1,Math.round(frame)));
  jointMeshes.forEach((mesh,joint) => displayedJoint(state.positions,state.currentFrame,joint,state.floorOffset,state.rootMode,mesh.position));
  EDGES.forEach(([a,b],index) => {
    const first=jointMeshes[a].position, second=jointMeshes[b].position, mesh=boneMeshes[index];
    direction.subVectors(second,first); const length=direction.length();
    mesh.position.copy(first).add(second).multiplyScalar(.5); mesh.scale.set(1,Math.max(length,.0001),1);
    if (length>1e-8) mesh.quaternion.setFromUnitVectors(up,direction.multiplyScalar(1/length));
  });
  skeletonGroup.visible = state.viewMode === "skeleton" || !state.smplReady;
  if (smplMesh) smplMesh.visible = state.viewMode === "surface" && state.smplReady;
  const showPath = state.rootMode === "world" && state.showPath;
  pathLine.visible=showPath; progressLine.visible=showPath; rootMarker.visible=showPath;
  progressGeometry.setDrawRange(0,state.currentFrame+1);
  const root = jointMeshes[0].position;
  rootMarker.position.set(root.x,.025,root.z);
  if (state.follow) {
    direction.set(root.x,0,root.z).sub(followAnchor);
    camera.position.add(direction); controls.target.add(direction);
  }
  followAnchor.set(root.x,0,root.z);
  meshPosition();
  const raw = state.currentFrame*66;
  const x=state.positions[raw]-state.positions[0], z=state.positions[raw+2]-state.positions[2];
  elements.rootReadout.textContent=`Root ΔX ${x.toFixed(2)} · ΔZ ${z.toFixed(2)} m`;
  elements.frameSlider.value=String(state.currentFrame);
  elements.frameCount.textContent=`${state.currentFrame+1} / ${numberFormatter.format(state.frames)} 帧`;
  elements.timeCount.textContent=`${(state.currentFrame/state.fps).toFixed(2)} / ${((state.frames-1)/state.fps).toFixed(2)} s`;
  if (state.viewMode === "surface" && state.smplhAvailable && !state.smplFailed) {
    const stride = state.playing ? Math.max(1,Math.round(state.fps/5)) : 1;
    if (!state.playing || Math.abs(state.currentFrame-lastQueuedFrame)>=stride) queueSmplFrame(state.currentFrame);
  }
}
async function ensureSmplMesh() {
  if (smplGeometry) return;
  if (!smplTopologyPromise) smplTopologyPromise = (async () => {
    const response = await requestData("/api/smplh/topology",{kind:"buffer",timeoutMs:45000,retries:1});
    const indices=new Uint32Array(response.body);
    if (indices.length!==Number(response.headers.get("X-SMPLH-Faces"))*3) throw new Error("网格结构不完整");
    smplGeometry=new THREE.BufferGeometry(); smplGeometry.setIndex(new THREE.BufferAttribute(indices,1));
    smplMesh=new THREE.Mesh(smplGeometry,new THREE.MeshStandardMaterial({color:0xa6bbd7,roughness:.8,side:THREE.DoubleSide}));
    smplMesh.visible=false; scene.add(smplMesh);
  })().catch(error => { smplTopologyPromise=null; throw error; });
  return smplTopologyPromise;
}
function queueSmplFrame(frame) {
  if (!state.positions || !state.smplhAvailable || state.viewMode!=="surface") return;
  lastQueuedFrame=frame;
  pendingMesh={frame,dataset:state.dataset,id:state.selectedId,selection:state.selectionSerial};
  if (!state.smplReady) elements.meshState.textContent="正在加载人体网格…";
  if (!meshBusy) processSmplQueue();
}
async function processSmplQueue() {
  meshBusy=true;
  while (pendingMesh) {
    const request=pendingMesh; pendingMesh=null;
    const current=() => request.selection===state.selectionSerial && state.viewMode==="surface";
    try {
      await ensureSmplMesh();
      if (!current()) continue;
      const response=await requestData(`/api/smplh/mesh/${request.dataset}/${encodeURIComponent(request.id)}/${request.frame}`,{kind:"buffer",timeoutMs:45000,retries:1});
      const vertices=new Float32Array(response.body);
      if (!current()) continue;
      if (!vertices.length || vertices.length!==Number(response.headers.get("X-SMPLH-Vertices"))*3) throw new Error("网格数据尺寸不正确");
      const attribute=smplGeometry.getAttribute("position");
      if (attribute?.array.length===vertices.length) {
        attribute.array.set(vertices); attribute.needsUpdate=true;
      } else smplGeometry.setAttribute("position",new THREE.BufferAttribute(vertices,3));
      smplGeometry.computeVertexNormals(); smplGeometry.computeBoundingSphere();
      state.smplReady=true; state.smplFailed=false; state.meshFrame=request.frame;
      meshPosition(); smplMesh.visible=true; skeletonGroup.visible=false;
      elements.meshState.textContent=state.playing ? "人体网格 · 约 5 FPS" : `人体网格 · 第 ${request.frame+1} 帧`;
      renderEvidence();
    } catch (error) {
      if (!current()) continue;
      state.smplReady=false; state.smplFailed=true; pendingMesh=null;
      if (smplMesh) smplMesh.visible=false;
      skeletonGroup.visible=Boolean(state.positions);
      elements.meshState.textContent="网格暂不可用 · 已显示骨架";
      renderEvidence(); showToast("人体网格加载失败，已保留骨架。可重新点击“人体网格”重试。");
    }
  }
  meshBusy=false;
}
function resetCamera(preset="perspective") {
  const a=state.analysis;
  let cx=0, cz=0, height=a ? Math.max(1.6,a.height) : 2, radius=1.6;
  if (a) {
    const bounds=a[state.follow ? "inplace" : state.rootMode];
    cx=(bounds.minX+bounds.maxX)/2; cz=(bounds.minZ+bounds.maxZ)/2;
    radius=Math.max(1.3,Math.hypot(bounds.maxX-bounds.minX,height,bounds.maxZ-bounds.minZ)/2);
    if (state.follow) { cx=jointMeshes[0].position.x; cz=jointMeshes[0].position.z; }
    const world=a[state.rootMode];
    const gridSize=Math.max(8,Math.ceil(Math.max(world.maxX-world.minX,world.maxZ-world.minZ)+4));
    grid.scale.setScalar(gridSize/12); grid.position.set((world.minX+world.maxX)/2,0,(world.minZ+world.maxZ)/2);
  }
  const vertical=THREE.MathUtils.degToRad(camera.fov), horizontal=2*Math.atan(Math.tan(vertical/2)*camera.aspect);
  const distance=radius/Math.sin(Math.min(vertical,horizontal)/2)*1.12;
  controls.target.set(cx,height*.46,cz);
  const offset=preset==="front" ? [0,.12,1] : preset==="side" ? [1,.12,0] : preset==="top" ? [0,1,.001] : [.75,.5,1];
  camera.position.copy(new THREE.Vector3(...offset).normalize().multiplyScalar(distance).add(controls.target));
  controls.maxDistance=Math.max(20,distance*8); camera.far=Math.max(100,distance*20); camera.updateProjectionMatrix();
  controls.update();
  if (state.positions) followAnchor.set(jointMeshes[0].position.x,0,jointMeshes[0].position.z);
}
function updateRootMode() {
  elements.rootMode.value=state.rootMode;
  elements.showTrajectory.disabled=state.rootMode==="inplace";
  elements.rootNote.textContent=state.rootMode==="world" ? "以首帧为原点，保留 root 平移、高度变化与转向。" : "原地查看：只隐藏水平位移，保留高度变化与转向。";
  // A display-mode change must not be mistaken for camera-follow displacement.
  if (state.positions) {
    displayedJoint(state.positions,state.currentFrame,0,state.floorOffset,state.rootMode,direction);
    followAnchor.set(direction.x,0,direction.z);
    updateFrame(state.currentFrame);
  }
  resetCamera(); syncLocation();
}
function setPlaying(value) {
  state.playing=Boolean(value && state.frames); state.lastTimestamp=performance.now();
  elements.playToggle.classList.toggle("is-playing",state.playing);
  elements.playToggle.setAttribute("aria-label",state.playing ? "暂停" : "播放");
  if (!state.frames) elements.meshState.textContent="等待样本";
  else if (state.viewMode==="skeleton") elements.meshState.textContent=state.playing ? "骨架 · 播放中" : "骨架 · 已暂停";
  else if (state.smplReady) elements.meshState.textContent=state.playing ? "人体网格 · 约 5 FPS" : `人体网格 · 第 ${state.meshFrame+1} 帧`;
}
function seek(frame) {
  setPlaying(false); state.frameCursor=Math.max(0,Math.min(state.frames-1,frame)); updateFrame(state.frameCursor);
}
function animate(timestamp) {
  requestAnimationFrame(animate);
  const delta=Math.min(.1,(timestamp-state.lastTimestamp)/1000 || 0); state.lastTimestamp=timestamp;
  if (state.playing && state.frames) {
    state.frameCursor+=delta*state.fps*state.speed;
    if (state.frameCursor>=state.frames) {
      if (state.loop) state.frameCursor%=state.frames;
      else { state.frameCursor=state.frames-1; setPlaying(false); }
    }
    const frame=Math.min(state.frames-1,Math.floor(state.frameCursor));
    if (frame!==state.currentFrame) updateFrame(frame);
  }
  controls.update(); renderer.render(scene,camera);
}
let firstSize=true;
new ResizeObserver(() => {
  const width=Math.max(1,elements.viewport.clientWidth), height=Math.max(1,elements.viewport.clientHeight);
  renderer.setSize(width,height,false); camera.aspect=width/height; camera.updateProjectionMatrix();
  if (firstSize) { resetCamera(); firstSize=false; }
}).observe(elements.viewport);
resetCamera(); requestAnimationFrame(animate);

function refreshFilter() {
  state.query=elements.sampleSearch.value.trim(); state.filter=elements.assetFilter.value;
  loadSamples({page:0});
}
elements.datasetButtons.forEach(button => button.addEventListener("click",() => selectDataset(button.dataset.dataset)));
elements.sampleSearch.addEventListener("input",() => { clearTimeout(searchTimer); searchTimer=setTimeout(refreshFilter,250); });
elements.sampleSearch.addEventListener("keydown",event => { if (event.key==="Enter") { clearTimeout(searchTimer); refreshFilter(); } });
elements.assetFilter.addEventListener("change",() => { clearTimeout(searchTimer); refreshFilter(); });
elements.clearSearch.addEventListener("click",() => { clearTimeout(searchTimer); elements.sampleSearch.value=""; elements.assetFilter.value="all"; refreshFilter(); elements.sampleSearch.focus(); });
elements.prevPage.addEventListener("click",() => loadSamples({page:state.page-1}));
elements.nextPage.addEventListener("click",() => loadSamples({page:state.page+1}));
elements.pageNumber.addEventListener("change",() => { const value=Number(elements.pageNumber.value); if (Number.isInteger(value)) loadSamples({page:Math.max(0,Math.min(Math.ceil(state.total/PAGE_SIZE)-1,value-1))}); else updateNavigation(); });
elements.prevSample.addEventListener("click",() => navigateSample(-1));
elements.nextSample.addEventListener("click",() => navigateSample(1));
elements.randomSample.addEventListener("click",() => {
  if (!state.total || state.listLoading) return;
  const index=Math.floor(Math.random()*state.total), page=Math.floor(index/PAGE_SIZE);
  if (page===state.page) selectSample(state.samples[index%PAGE_SIZE].id);
  else loadSamples({page,pick:index%PAGE_SIZE});
});
elements.copyLink.addEventListener("click",async () => { try { await navigator.clipboard.writeText(location.href); showToast("已复制当前样本链接。"); } catch { showToast("无法访问剪贴板，可直接复制地址栏中的链接。"); } });
elements.rootMode.addEventListener("change",() => { state.rootMode=elements.rootMode.value; updateRootMode(); });
elements.showTrajectory.addEventListener("change",() => { state.showPath=elements.showTrajectory.checked; updateFrame(state.currentFrame); });
elements.followRoot.addEventListener("change",() => { state.follow=elements.followRoot.checked; resetCamera(); });
elements.fitMotion.addEventListener("click",() => { state.follow=false; elements.followRoot.checked=false; resetCamera(); });
elements.cameraButtons.forEach(button => button.addEventListener("click",() => resetCamera(button.dataset.camera)));
elements.resetCamera.addEventListener("click",() => resetCamera());
elements.viewButtons.forEach(button => button.addEventListener("click",() => {
  if (button.disabled) return;
  state.viewMode=button.dataset.mode; state.smplFailed=false;
  elements.viewButtons.forEach(item => { const active=item===button; item.classList.toggle("is-active",active); item.setAttribute("aria-pressed",String(active)); });
  updateFrame(state.currentFrame); setPlaying(state.playing);
}));
elements.playToggle.addEventListener("click",() => { if (!state.playing && state.currentFrame===state.frames-1) seek(0); setPlaying(!state.playing); });
elements.prevFrame.addEventListener("click",() => seek(state.currentFrame-1));
elements.nextFrame.addEventListener("click",() => seek(state.currentFrame+1));
elements.frameSlider.addEventListener("input",() => seek(Number(elements.frameSlider.value)));
elements.speedSelect.addEventListener("change",() => { state.speed=Number(elements.speedSelect.value); });
elements.loopToggle.addEventListener("change",() => { state.loop=elements.loopToggle.checked; });
elements.textTabs.forEach(button => button.addEventListener("click",() => {
  if (button.disabled) return;
  state.textMode=button.dataset.text; state.textQuery=""; state.textSerial++; elements.textSearch.value="";
  updateTextTabs();
  if (state.textMode==="base" && state.detail?.caption?.kind==="lines" && state.detail.caption.query) loadBaseText({reset:true});
}));
elements.textSearch.addEventListener("input",() => {
  clearTimeout(textSearchTimer); textSearchTimer=setTimeout(() => {
    state.textQuery=elements.textSearch.value.trim();
    const caption=textForMode(state.textMode);
    if (state.textMode==="base" && caption?.kind==="lines") loadBaseText({reset:true}); else renderText(caption);
  },220);
});
function expandText(value) {
  elements.textPanel.classList.toggle("is-expanded",value); elements.expandText.textContent=value ? "收起" : "展开";
  if (value) elements.textSearch.focus(); else elements.expandText.focus();
}
elements.expandText.addEventListener("click",() => expandText(!elements.textPanel.classList.contains("is-expanded")));
elements.copyText.addEventListener("click",async () => {
  const text=captionAsText(textForMode(state.textMode));
  if (!text) return showToast("没有可复制的文本。");
  try { await navigator.clipboard.writeText(text); showToast("已复制当前已加载的文本。"); } catch { showToast("复制失败，请选择文本后手动复制。"); }
});
document.addEventListener("keydown",event => {
  if (event.key==="Escape" && elements.textPanel.classList.contains("is-expanded")) { expandText(false); return; }
  if (event.target.closest("input,select,textarea,video,[contenteditable=true]") || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.code==="Space" && !event.target.closest("button,summary,a")) { event.preventDefault(); elements.playToggle.click(); }
  else if (event.key==="ArrowLeft") { event.preventDefault(); seek(state.currentFrame-1); }
  else if (event.key==="ArrowRight") { event.preventDefault(); seek(state.currentFrame+1); }
  else if (event.key.toLowerCase()==="j") navigateSample(1);
  else if (event.key.toLowerCase()==="k") navigateSample(-1);
  else if (event.key==="/") { event.preventDefault(); elements.sampleSearch.focus(); }
});
async function boot() {
  const wantedId=state.selectedId;
  elements.serverDot.classList.remove("is-error","is-ready"); elements.serverStatus.textContent="连接中";
  elements.sampleSearch.value=state.query; elements.assetFilter.value=state.filter;
  updateDatasetMeta(); updateRootMode(); updateNavigation();
  try {
    state.summary=await requestJson("/api/summary"); state.smplhAvailable=Boolean(state.summary.smplh?.available);
    const surface=elements.viewButtons.find(button => button.dataset.mode==="surface");
    surface.disabled=!state.smplhAvailable; surface.title=state.smplhAvailable ? "按需加载 SMPL-H 重建" : "服务器暂无人体网格";
    elements.serverDot.classList.add("is-ready"); elements.serverStatus.textContent="已连接 Aistation";
    renderEvidence(); await loadSamples({wantedId});
  } catch (error) {
    elements.serverDot.classList.add("is-error"); elements.serverStatus.textContent="连接失败";
    listMessage("服务暂时不可用",`${error.message}。`,() => boot()); showToast("无法连接数据服务，可以点击“重新加载”。");
  }
}
boot();
