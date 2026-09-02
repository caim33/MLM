(function () {
  const root = document.documentElement;
  const storedTheme = localStorage.getItem("motion-atlas-theme");
  if (storedTheme) root.dataset.theme = storedTheme;

  const toggle = document.getElementById("themeToggle");
  toggle?.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem("motion-atlas-theme", root.dataset.theme);
  });

  const body = document.body;
  const menuButton = document.getElementById("guideMenuButton");
  const closeButton = document.getElementById("guideSidebarClose");
  const backdrop = document.getElementById("guideSidebarBackdrop");
  const navSearch = document.getElementById("guideNavSearch");
  const mobileLayout = window.matchMedia("(max-width: 820px)");

  const closeSidebar = () => body.classList.remove("guide-sidebar-open");
  menuButton?.addEventListener("click", () => {
    if (mobileLayout.matches) {
      body.classList.toggle("guide-sidebar-open");
      return;
    }
    body.classList.toggle("guide-sidebar-collapsed");
  });
  closeButton?.addEventListener("click", closeSidebar);
  backdrop?.addEventListener("click", closeSidebar);
  window.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      if (mobileLayout.matches) body.classList.add("guide-sidebar-open");
      navSearch?.focus();
    }
    if (event.key === "Escape") closeSidebar();
  });

  const sidebarNav = document.querySelector(".guide-sidebar nav");
  const sidebarLinks = [...(sidebarNav?.querySelectorAll("a") || [])];
  const navGroups = [...(sidebarNav?.querySelectorAll(".guide-nav-group") || [])];
  navSearch?.addEventListener("input", () => {
    const query = navSearch.value.trim().toLocaleLowerCase("zh-CN");
    sidebarLinks.forEach((link) => {
      link.hidden = Boolean(query) && !link.textContent.toLocaleLowerCase("zh-CN").includes(query);
    });
    navGroups.forEach((group) => { group.hidden = Boolean(query); });
  });
  sidebarLinks.forEach((link) => link.addEventListener("click", () => {
    if (mobileLayout.matches) closeSidebar();
  }));

  document.querySelectorAll("pre").forEach((pre) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-code";
    button.textContent = "复制";
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector("code")?.textContent || "");
        button.textContent = "已复制";
      } catch (error) {
        button.textContent = "复制失败";
        console.warn("Clipboard write failed", error);
      }
      window.setTimeout(() => { button.textContent = "复制"; }, 1200);
    });
    pre.appendChild(button);
  });

  const links = [...document.querySelectorAll(".guide-sidebar nav a[href^='#']")];
  const tocLinks = [...document.querySelectorAll(".guide-toc a[href^='#']")];
  const sections = links.map((link) => document.querySelector(link.getAttribute("href"))).filter(Boolean);
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`));
    tocLinks.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`));
  }, { rootMargin: "-18% 0px -68%", threshold: [0, .2, .6] });
  sections.forEach((section) => observer.observe(section));
})();
