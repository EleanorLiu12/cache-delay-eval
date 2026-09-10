(() => {
  "use strict";

  const repository = "EleanorLiu12/cache-delay-eval";
  const sourcePath = "docs/experimental-plan.md";
  const sourceBranch = "main";
  const sourceUrl = `https://github.com/${repository}/blob/${sourceBranch}/${sourcePath}`;
  const article = document.querySelector("#plan-content");
  const sourceStatus = document.querySelector("#plan-source-status");

  const escapeHtml = (value) =>
    value
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const inlineMarkup = (value) => {
    const protectedTokens = [];
    let rendered = value.replace(/`([^`]+)`/g, (_, code) => {
      const token = `\u0000TOKEN${protectedTokens.length}\u0000`;
      protectedTokens.push(`<code>${escapeHtml(code)}</code>`);
      return token;
    });
    rendered = rendered.replace(/\[([^\]]+)]\((https?:\/\/[^)\s]+|#[^)\s]+)\)/g, (_, label, href) => {
      const token = `\u0000TOKEN${protectedTokens.length}\u0000`;
      const external = href.startsWith("http");
      const attributes = external ? ' target="_blank" rel="noopener noreferrer"' : "";
      protectedTokens.push(`<a href="${escapeHtml(href)}"${attributes}>${escapeHtml(label)}</a>`);
      return token;
    });
    rendered = escapeHtml(rendered)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>");
    protectedTokens.forEach((token, index) => {
      rendered = rendered.replace(`\u0000TOKEN${index}\u0000`, token);
    });
    return rendered;
  };

  const slugify = (value, used) => {
    const base = value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "section";
    const count = (used.get(base) || 0) + 1;
    used.set(base, count);
    return count === 1 ? base : `${base}-${count}`;
  };

  const isTableDivider = (value) => /^\s*\|(?:\s*:?-+:?\s*\|)+\s*$/.test(value);
  const isBlockStart = (value) =>
    /^(#{1,6})\s+/.test(value) ||
    /^```/.test(value) ||
    /^[-*+]\s+/.test(value) ||
    /^\d+\.\s+/.test(value) ||
    /^>\s?/.test(value) ||
    value.startsWith("|");

  const renderMarkdown = (markdown) => {
    const lines = markdown.replace(/\r\n?/g, "\n").split("\n");
    const blocks = [];
    const sections = [];
    const usedSlugs = new Map();
    let index = 0;

    while (index < lines.length) {
      const stripped = lines[index].trim();

      if (!stripped) {
        index += 1;
        continue;
      }

      if (stripped.startsWith("```")) {
        const language = stripped.slice(3).trim();
        const codeLines = [];
        index += 1;
        while (index < lines.length && !lines[index].trim().startsWith("```")) {
          codeLines.push(lines[index]);
          index += 1;
        }
        if (index < lines.length) index += 1;
        const languageClass = language ? ` class="language-${escapeHtml(language)}"` : "";
        blocks.push(`<pre><code${languageClass}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        continue;
      }

      const heading = stripped.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        const level = heading[1].length;
        const label = heading[2].trim();
        const anchor = slugify(label, usedSlugs);
        if (level === 2) sections.push({ anchor, label });
        blocks.push(`<h${level} id="${anchor}">${inlineMarkup(label)}</h${level}>`);
        index += 1;
        continue;
      }

      if (stripped.startsWith("|") && index + 1 < lines.length && isTableDivider(lines[index + 1])) {
        const headers = stripped.slice(1, -1).split("|").map((cell) => cell.trim());
        const rows = [];
        index += 2;
        while (index < lines.length && lines[index].trim().startsWith("|")) {
          rows.push(lines[index].trim().slice(1, -1).split("|").map((cell) => cell.trim()));
          index += 1;
        }
        const headerHtml = headers.map((cell) => `<th>${inlineMarkup(cell)}</th>`).join("");
        const rowHtml = rows
          .map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkup(cell)}</td>`).join("")}</tr>`)
          .join("");
        blocks.push(`<div class="table-wrap"><table><thead><tr>${headerHtml}</tr></thead><tbody>${rowHtml}</tbody></table></div>`);
        continue;
      }

      const listMatch = stripped.match(/^([-*+]|\d+\.)\s+(.+)$/);
      if (listMatch) {
        const ordered = /\d+\./.test(listMatch[1]);
        const items = [];
        while (index < lines.length) {
          const current = lines[index].trim().match(ordered ? /^\d+\.\s+(.+)$/ : /^[-*+]\s+(.+)$/);
          if (!current) break;
          const itemParts = [current[1]];
          index += 1;
          while (index < lines.length) {
            const continuation = lines[index].trim();
            if (!continuation || isBlockStart(continuation)) break;
            itemParts.push(continuation);
            index += 1;
          }
          items.push(itemParts.join(" "));
          if (index < lines.length && !lines[index].trim()) {
            index += 1;
            break;
          }
        }
        const tag = ordered ? "ol" : "ul";
        blocks.push(`<${tag}>${items.map((item) => `<li>${inlineMarkup(item)}</li>`).join("")}</${tag}>`);
        continue;
      }

      if (stripped.startsWith(">")) {
        const quote = [];
        while (index < lines.length && lines[index].trim().startsWith(">")) {
          quote.push(lines[index].trim().replace(/^>\s?/, ""));
          index += 1;
        }
        blocks.push(`<blockquote><p>${inlineMarkup(quote.join(" "))}</p></blockquote>`);
        continue;
      }

      const paragraph = [stripped];
      index += 1;
      while (index < lines.length) {
        const candidate = lines[index].trim();
        if (!candidate || isBlockStart(candidate)) break;
        paragraph.push(candidate);
        index += 1;
      }
      blocks.push(`<p>${inlineMarkup(paragraph.join(" "))}</p>`);
    }

    const contents = sections
      .map(({ anchor, label }) => `<li><a href="#${anchor}">${inlineMarkup(label)}</a></li>`)
      .join("");
    const toc = `<nav class="plan-toc" aria-label="Plan contents"><strong>On this page</strong><ol>${contents}</ol></nav>`;
    const body = blocks.join("\n").replace("</h1>", `</h1>${toc}`);
    return `<p class="eyebrow">Research design · Live repository document</p>${body}`;
  };

  const decodeContent = (content) => {
    const binary = atob(content.replace(/\s/g, ""));
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    return new TextDecoder("utf-8").decode(bytes);
  };

  const showError = () => {
    article.innerHTML = `
      <p class="eyebrow">Repository source unavailable</p>
      <h1>Experiment Plan</h1>
      <div class="plan-error" role="alert">
        <strong>The current plan could not be loaded.</strong>
        <p>No saved copy is shown, so this page can never silently display an older plan.</p>
        <a href="${sourceUrl}">Open the authoritative file on GitHub <span aria-hidden="true">↗</span></a>
      </div>`;
    article.setAttribute("aria-busy", "false");
    sourceStatus.textContent = "Live source temporarily unavailable";
  };

  const loadPlan = async () => {
    const nonce = Date.now();
    const apiUrl = `https://api.github.com/repos/${repository}/contents/${sourcePath}?ref=${sourceBranch}&fresh=${nonce}`;
    try {
      const response = await fetch(apiUrl, {
        cache: "no-store",
        headers: {
          Accept: "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
        },
      });
      if (!response.ok) throw new Error(`GitHub returned ${response.status}`);
      const source = await response.json();
      if (source.encoding !== "base64" || !source.content) throw new Error("Unexpected source response");
      article.innerHTML = renderMarkdown(decodeContent(source.content));
      article.setAttribute("aria-busy", "false");
      article.dataset.sourceSha = source.sha;
      sourceStatus.textContent = `Live from main · ${source.sha.slice(0, 7)}`;
      sourceStatus.title = `Exact repository blob ${source.sha}`;
    } catch (error) {
      console.error("Unable to load the experimental plan", error);
      showError();
    }
  };

  loadPlan();
})();
