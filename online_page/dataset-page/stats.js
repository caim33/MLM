const DATASET_LABELS = {
  humanml3d: "HumanML3D",
  sonic: "SONIC",
  motionx: "MotionX",
};

const numberFormatter = new Intl.NumberFormat("zh-CN");

const elements = {
  indexStatus: document.getElementById("indexStatus"),
  statusDot: document.querySelector(".status-dot"),
  statGrid: document.getElementById("statGrid"),
  taskMatrix: document.getElementById("taskMatrix"),
  taskMethod: document.getElementById("taskMethod"),
  scaleChart: document.getElementById("scaleChart"),
  qualityList: document.getElementById("qualityList"),
};

async function requestJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
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

function modalityLabel(modality) {
  return { motion: "MOTION", video: "VIDEO", text: "TEXT" }[modality] || modality.toUpperCase();
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

function renderTasks(taxonomy) {
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

  const methodTitle = document.createElement("strong");
  methodTitle.textContent = "时长口径";
  const methodText = document.createElement("p");
  methodText.textContent = taxonomy.duration_definition;
  const stamp = document.createElement("span");
  stamp.textContent = `AUDITED ${taxonomy.generated_at}`;
  elements.taskMethod.replaceChildren(methodTitle, methodText, stamp);
}

function renderSummary(summary) {
  const totals = summary.totals;
  elements.statGrid.replaceChildren(
    createStat("动作序列", numberFormatter.format(totals.motion_files), "NPY"),
    createStat("视频", numberFormatter.format(totals.video_files), "MP4"),
    createStat("数据体积", totals.size_gb.toFixed(1), "GB"),
    createStat("统一表示", `${totals.feature_dims}D`, `${totals.joints} joints`),
  );

  const datasets = summary.datasets;
  const maxSize = Math.max(...Object.values(datasets).map((item) => item.size_gb));
  elements.scaleChart.replaceChildren();
  elements.qualityList.replaceChildren();

  for (const key of ["humanml3d", "sonic", "motionx"]) {
    const item = datasets[key];
    const row = document.createElement("div");
    row.className = "scale-row";
    const label = document.createElement("span");
    label.textContent = item.label;
    const track = document.createElement("div");
    track.className = "scale-track";
    const fill = document.createElement("div");
    fill.className = "scale-fill";
    fill.style.width = `${Math.max(2, (item.size_gb / maxSize) * 100)}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "scale-value";
    value.textContent = `${item.size_gb.toFixed(3)} GB`;
    row.append(label, track, value);
    elements.scaleChart.append(row);

    const note = document.createElement("article");
    note.className = "quality-item";
    const title = document.createElement("strong");
    title.textContent = item.label;
    const description = document.createElement("p");
    description.textContent = item.quality;
    note.append(title, description);
    elements.qualityList.append(note);
  }
}

async function boot() {
  try {
    const [summary, taxonomy] = await Promise.all([
      requestJson("./data/summary.json"),
      requestJson("./data/task_stats.json"),
    ]);
    renderSummary(summary);
    renderTasks(taxonomy);
    elements.indexStatus.textContent = `统计快照 · ${numberFormatter.format(Object.values(summary.indexed_samples).reduce((sum, value) => sum + value, 0))} 条样本`;
    elements.statusDot.classList.add("is-ready");
  } catch (error) {
    elements.indexStatus.textContent = "统计快照载入失败";
    elements.taskMatrix.textContent = `无法读取本地统计文件：${error.message}`;
  }
}

boot();
