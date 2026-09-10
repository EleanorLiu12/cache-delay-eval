#!/usr/bin/env python3
"""Render docs/experimental-plan.md as the site's experiment-plan page."""

from __future__ import annotations

from html import escape
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "experimental-plan.md"
OUTPUT = ROOT / "site" / "experimental-plan.html"
DOWNLOAD = ROOT / "site" / "assets" / "experimental-plan.md"


def inline_markup(value: str) -> str:
    rendered = escape(value, quote=False)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    return rendered


def slugify(value: str, used: dict[str, int]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "section"
    used[base] = used.get(base, 0) + 1
    return base if used[base] == 1 else f"{base}-{used[base]}"


def render_markdown(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    lines = markdown.splitlines()
    blocks: list[str] = []
    sections: list[tuple[str, str]] = []
    used_slugs: dict[str, int] = {}
    index = 0

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            language_class = f' class="language-{escape(language)}"' if language else ""
            blocks.append(f"<pre><code{language_class}>{escape(chr(10).join(code_lines))}</code></pre>")
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            label = heading.group(2).strip()
            anchor = slugify(label, used_slugs)
            if level == 2:
                sections.append((anchor, label))
            blocks.append(f'<h{level} id="{anchor}">{inline_markup(label)}</h{level}>')
            index += 1
            continue

        if (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and re.match(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$", lines[index + 1])
        ):
            headers = [cell.strip() for cell in stripped.strip("|").split("|")]
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            header_html = "".join(f"<th>{inline_markup(cell)}</th>" for cell in headers)
            row_html = "".join(
                "<tr>" + "".join(f"<td>{inline_markup(cell)}</td>" for cell in row) + "</tr>"
                for row in rows
            )
            blocks.append(
                '<div class="table-wrap"><table><thead><tr>'
                + header_html
                + "</tr></thead><tbody>"
                + row_html
                + "</tbody></table></div>"
            )
            continue

        if stripped.startswith("- "):
            items: list[str] = []
            while index < len(lines):
                current = lines[index].strip()
                if not current.startswith("- "):
                    break
                item_parts = [current[2:].strip()]
                index += 1
                while index < len(lines):
                    continuation = lines[index]
                    continuation_stripped = continuation.strip()
                    if not continuation_stripped or continuation_stripped.startswith("- "):
                        break
                    if re.match(r"^(#{1,3})\s+", continuation_stripped) or continuation_stripped.startswith("|"):
                        break
                    item_parts.append(continuation_stripped)
                    index += 1
                items.append(" ".join(item_parts))
                if index < len(lines) and not lines[index].strip():
                    index += 1
                    break
            blocks.append("<ul>" + "".join(f"<li>{inline_markup(item)}</li>" for item in items) + "</ul>")
            continue

        paragraph = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if (
                candidate.startswith(("#", "- ", "```", "|"))
                or re.match(r"^(#{1,3})\s+", candidate)
            ):
                break
            paragraph.append(candidate)
            index += 1
        blocks.append(f"<p>{inline_markup(' '.join(paragraph))}</p>")

    return "\n          ".join(blocks), sections


def sidebar() -> str:
    return """<aside class="week-sidebar" aria-label="Experiment plan and project weeks">
        <div class="week-sidebar-inner">
          <a class="plan-link is-active" href="#top" aria-current="page">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6.75 3.75h7.5l3 3v13.5H6.75z"></path>
              <path d="M14.25 3.75v3h3M9.5 11h5M9.5 14.5h5"></path>
            </svg>
            <span>
              <strong>Experiment Plan</strong>
              <small>Questions, stages, and protocol</small>
            </span>
          </a>
          <nav class="week-nav" aria-label="Project weeks">
            <a class="week-link" href="index.html">
              <span>Week 1–2</span>
              <small>Calibration results</small>
            </a>
            <span class="week-link is-planned" aria-disabled="true">
              <span>Week 3–4</span>
              <small>Emulator · Planned</small>
            </span>
            <span class="week-link is-planned" aria-disabled="true">
              <span>Week 5–6</span>
              <small>Routing study · Planned</small>
            </span>
            <span class="week-link is-planned" aria-disabled="true">
              <span>Week 7–8</span>
              <small>Final evaluation · Planned</small>
            </span>
          </nav>
          <div class="milestone-card">
            <span class="status-dot" aria-hidden="true"></span>
            <div>
              <strong>Current milestone</strong>
              <small>Week 1–2 complete</small>
            </div>
          </div>
        </div>
      </aside>"""


def render_page(markdown: str) -> str:
    body, sections = render_markdown(markdown)
    toc = "".join(f'<li><a href="#{anchor}">{inline_markup(label)}</a></li>' for anchor, label in sections)
    toc_html = (
        '<nav class="plan-toc" aria-label="Plan contents">'
        '<strong>On this page</strong><ol>' + toc + "</ol></nav>"
    )
    body = body.replace("</h1>", "</h1>\n          " + toc_html, 1)

    return f"""<!doctype html>
<!-- Generated by scripts/sync_experimental_plan.py from docs/experimental-plan.md. -->
<html lang="en" data-theme="light">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="Experimental plan for evaluating stale-state-aware prefix-cache routing in distributed LLM serving.">
    <meta name="theme-color" content="#102338">
    <title>Cache Delay Evaluation · Experiment Plan</title>
    <script>
      (() => {{
        try {{
          const saved = localStorage.getItem("cache-delay-theme");
          const preferred = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
          document.documentElement.dataset.theme = saved || preferred;
        }} catch (_) {{
          document.documentElement.dataset.theme = "light";
        }}
      }})();
    </script>
    <link rel="stylesheet" href="styles.css">
  </head>
  <body>
    <a class="skip-link" href="#plan-content">Skip to plan</a>
    <header class="site-header">
      <div class="topbar-inner">
        <a class="wordmark" href="index.html" aria-label="Cache Delay Evaluation home">
          <span class="wordmark-mark" aria-hidden="true">CD</span>
          <span class="wordmark-copy">
            <strong>Cache Delay Evaluation</strong>
            <small>Distributed LLM routing study</small>
          </span>
        </a>
        <div class="topbar-actions">
          <a class="repository-link" href="https://github.com/EleanorLiu12/cache-delay-eval">Repository</a>
          <button class="theme-toggle" type="button" aria-label="Switch to dark mode" aria-pressed="false" title="Switch to dark mode">
            <svg class="theme-icon theme-icon-sun" viewBox="0 0 24 24" aria-hidden="true">
              <circle cx="12" cy="12" r="4"></circle>
              <path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.66 6.34l1.41-1.41"></path>
            </svg>
            <svg class="theme-icon theme-icon-moon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M20.3 15.7A8.5 8.5 0 0 1 8.3 3.7 8.5 8.5 0 1 0 20.3 15.7Z"></path>
            </svg>
          </button>
        </div>
      </div>
    </header>

    <div class="workspace">
      {sidebar()}

      <div class="content-panel">
        <div class="plan-toolbar">
          <span>Full experimental roadmap</span>
          <a href="assets/experimental-plan.md" download>Download source Markdown <span aria-hidden="true">↓</span></a>
        </div>
        <main id="top" class="plan-main">
          <article class="plan-document shell" id="plan-content">
            <p class="eyebrow">Research design · Living document</p>
          {body}
          </article>
        </main>
        <footer>
          <div class="shell footer-inner">
            <p>Cache Delay Evaluation · Experimental roadmap</p>
            <p>Source synchronized from docs/experimental-plan.md</p>
          </div>
        </footer>
      </div>
    </div>
    <script>
      (() => {{
        const button = document.querySelector(".theme-toggle");
        const themeMeta = document.querySelector('meta[name="theme-color"]');
        const syncButton = () => {{
          const isDark = document.documentElement.dataset.theme === "dark";
          button.setAttribute("aria-pressed", String(isDark));
          button.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
          button.title = isDark ? "Switch to light mode" : "Switch to dark mode";
          themeMeta.content = isDark ? "#09131f" : "#102338";
        }};
        button.addEventListener("click", () => {{
          const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
          document.documentElement.dataset.theme = next;
          try {{ localStorage.setItem("cache-delay-theme", next); }} catch (_) {{}}
          syncButton();
        }});
        syncButton();
      }})();
    </script>
  </body>
</html>
"""


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    OUTPUT.write_text(render_page(markdown), encoding="utf-8")
    DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, DOWNLOAD)
    print(f"Synced {SOURCE.relative_to(ROOT)} -> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
