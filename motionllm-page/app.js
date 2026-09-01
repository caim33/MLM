(function () {
  const papers = Array.isArray(window.MOTION_PAPERS) ? window.MOTION_PAPERS : [];
  const grid = document.getElementById("paperGrid");
  const search = document.getElementById("paperSearch");
  const filterRow = document.getElementById("filterRow");
  const sort = document.getElementById("paperSort");
  const resultCount = document.getElementById("resultCount");
  const emptyState = document.getElementById("emptyState");
  const dialog = document.getElementById("figureDialog");
  const dialogImage = document.getElementById("dialogImage");
  const dialogTitle = document.getElementById("dialogTitle");
  const dialogCaption = document.getElementById("dialogCaption");
  let activeFilter = "all";

  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  const categoryNames = {
    direct: "直接相关",
    representation: "Motion 表示",
    rl: "RL / GRPO",
    rubric: "Rubric / Judge",
    evaluation: "评估治理",
    adjacent: "邻近工作"
  };

  function badgeClass(category) {
    if (category === "direct") return "direct";
    if (category === "rl" || category === "rubric") return "rl";
    if (category === "evaluation") return "eval";
    return "";
  }

  function linksMarkup(paper) {
    const links = [];
    if (paper.paperUrl) links.push(`<a href="${escapeHtml(paper.paperUrl)}" target="_blank" rel="noreferrer">论文 ↗</a>`);
    if (paper.projectUrl) links.push(`<a href="${escapeHtml(paper.projectUrl)}" target="_blank" rel="noreferrer">官方项目 ↗</a>`);
    if (paper.codeUrl) links.push(`<a href="${escapeHtml(paper.codeUrl)}" target="_blank" rel="noreferrer">代码 ↗</a>`);
    if (paper.localPdf) links.push(`<a href="${escapeHtml(paper.localPdf)}">本地 PDF</a>`);
    return links.join("");
  }

  function listMarkup(items) {
    if (!Array.isArray(items) || !items.length) return "";
    return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  }

  function paperCard(paper) {
    const categories = (paper.categories || []).map((category) =>
      `<span class="badge ${badgeClass(category)}">${escapeHtml(categoryNames[category] || category)}</span>`
    ).join("");
    const figure = paper.figure || "assets/figures/placeholder.svg";
    const caption = paper.figureCaption || "论文主图";
    return `
      <article class="paper-card" data-id="${escapeHtml(paper.id)}">
        <button class="paper-figure" type="button" data-figure="${escapeHtml(figure)}" data-title="${escapeHtml(paper.title)}" data-caption="${escapeHtml(caption)}" aria-label="放大 ${escapeHtml(paper.title)} 主图">
          <img src="${escapeHtml(figure)}" alt="${escapeHtml(paper.title)}：${escapeHtml(caption)}" loading="lazy">
        </button>
        <div class="paper-body">
          <div class="paper-meta">${categories}<span>${escapeHtml(paper.year)}</span><span>·</span><span>${escapeHtml(paper.venue || "Preprint")}</span><span>·</span><span>相关度 ${escapeHtml(paper.relevance)}/5</span></div>
          <h3>${escapeHtml(paper.title)}</h3>
          <p class="paper-one-line">${escapeHtml(paper.oneLine)}</p>
          <p class="paper-impact"><b>对你的价值：</b>${escapeHtml(paper.projectValue)}</p>
          <details class="paper-details">
            <summary>展开精读笔记</summary>
            <div class="detail-block"><strong>机制</strong><p>${escapeHtml(paper.method)}</p></div>
            <div class="detail-block"><strong>证据</strong><p>${escapeHtml(paper.evidence)}</p></div>
            <div class="detail-block"><strong>局限 / 不要误读</strong><p>${escapeHtml(paper.limitations)}</p></div>
            <div class="detail-block"><strong>可执行 Know-how</strong>${listMarkup(paper.knowHow)}</div>
            <div class="detail-block"><strong>主图出处</strong><p>${escapeHtml(caption)}</p></div>
            <div class="paper-links">${linksMarkup(paper)}</div>
          </details>
        </div>
      </article>`;
  }

  function searchableText(paper) {
    return [paper.title, paper.oneLine, paper.projectValue, paper.method, paper.evidence,
      paper.limitations, ...(paper.categories || []), ...(paper.knowHow || [])]
      .join(" ").toLocaleLowerCase("zh-CN");
  }

  function render() {
    const query = search.value.trim().toLocaleLowerCase("zh-CN");
    const visible = papers.filter((paper) => {
      const filterMatch = activeFilter === "all" || (paper.categories || []).includes(activeFilter);
      const searchMatch = !query || searchableText(paper).includes(query);
      return filterMatch && searchMatch;
    });
    const mode = sort.value;
    visible.sort((a, b) => {
      if (mode === "year-desc") return Number(b.year) - Number(a.year) || Number(b.relevance) - Number(a.relevance);
      if (mode === "year-asc") return Number(a.year) - Number(b.year) || Number(b.relevance) - Number(a.relevance);
      return Number(b.relevance) - Number(a.relevance) || Number(b.year) - Number(a.year);
    });
    grid.innerHTML = visible.map(paperCard).join("");
    resultCount.textContent = `显示 ${visible.length} / ${papers.length} 篇`;
    emptyState.hidden = visible.length > 0;
  }

  filterRow.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;
    activeFilter = button.dataset.filter;
    filterRow.querySelectorAll("button").forEach((item) => item.classList.toggle("active", item === button));
    render();
  });
  search.addEventListener("input", render);
  sort.addEventListener("change", render);

  grid.addEventListener("click", (event) => {
    const button = event.target.closest("[data-figure]");
    if (!button) return;
    dialogImage.src = button.dataset.figure;
    dialogImage.alt = button.dataset.title;
    dialogTitle.textContent = button.dataset.title;
    dialogCaption.textContent = button.dataset.caption;
    dialog.showModal();
  });
  document.getElementById("dialogClose").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });

  const root = document.documentElement;
  const storedTheme = localStorage.getItem("motion-atlas-theme");
  if (storedTheme) root.dataset.theme = storedTheme;
  document.getElementById("themeToggle").addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem("motion-atlas-theme", root.dataset.theme);
  });

  document.getElementById("paperCount").textContent = papers.length;
  document.getElementById("figureCount").textContent = papers.filter((paper) => paper.figure).length;
  document.getElementById("directCount").textContent = papers.filter((paper) => (paper.categories || []).includes("direct")).length;
  render();
})();
