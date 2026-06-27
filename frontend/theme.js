// Shared light/dark theme handling for every page. Loaded in <head> (no defer)
// so the saved theme is applied before first paint — no flash of the wrong mode.
(function () {
  const KEY = "xg-theme";
  const root = document.documentElement;
  const saved = localStorage.getItem(KEY);
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = saved || (prefersDark ? "dark" : "light");

  function syncButton(btn) {
    const dark = root.dataset.theme === "dark";
    btn.textContent = dark ? "☀ Light" : "☾ Dark";
    btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
  }

  document.addEventListener("DOMContentLoaded", function () {
    const btn = document.getElementById("theme-toggle");
    if (!btn) return;
    syncButton(btn);
    btn.addEventListener("click", function () {
      root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
      localStorage.setItem(KEY, root.dataset.theme);
      syncButton(btn);
    });
  });
})();
