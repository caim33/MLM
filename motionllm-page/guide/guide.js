(function () {
  const root = document.documentElement;
  const storedTheme = localStorage.getItem("motion-atlas-theme");
  if (storedTheme) root.dataset.theme = storedTheme;

  const toggle = document.getElementById("themeToggle");
  toggle?.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem("motion-atlas-theme", root.dataset.theme);
  });

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
  const sections = links.map((link) => document.querySelector(link.getAttribute("href"))).filter(Boolean);
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`));
  }, { rootMargin: "-18% 0px -68%", threshold: [0, .2, .6] });
  sections.forEach((section) => observer.observe(section));
})();
