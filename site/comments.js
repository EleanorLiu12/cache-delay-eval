(() => {
  "use strict";

  const host = document.querySelector("#comments-thread");
  if (!host) return;

  const currentTheme = () =>
    document.documentElement.dataset.theme === "dark" ? "dark" : "light";

  const setFrameTheme = () => {
    const frame = document.querySelector("iframe.giscus-frame");
    if (!frame?.contentWindow) return;
    frame.contentWindow.postMessage(
      { giscus: { setConfig: { theme: currentTheme() } } },
      "https://giscus.app",
    );
  };

  const script = document.createElement("script");
  script.src = "https://giscus.app/client.js";
  script.async = true;
  script.crossOrigin = "anonymous";
  script.dataset.repo = "EleanorLiu12/cache-delay-eval";
  script.dataset.repoId = "R_kgDOUPuuJw";
  script.dataset.category = "General";
  script.dataset.categoryId = "DIC_kwDOUPuuJ84DFUIu";
  script.dataset.mapping = "specific";
  script.dataset.term = "week-1-2-results";
  script.dataset.strict = "1";
  script.dataset.reactionsEnabled = "1";
  script.dataset.emitMetadata = "0";
  script.dataset.inputPosition = "top";
  script.dataset.theme = currentTheme();
  script.dataset.lang = "en";
  script.dataset.loading = "lazy";
  script.onload = () => host.querySelector(".comments-loading")?.remove();
  script.onerror = () => {
    const loading = host.querySelector(".comments-loading");
    if (!loading) return;
    loading.innerHTML =
      'Comments could not be loaded. <a href="https://github.com/EleanorLiu12/cache-delay-eval/discussions">Open GitHub Discussions.</a>';
  };
  host.append(script);

  const themeObserver = new MutationObserver(setFrameTheme);
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
})();
