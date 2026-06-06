/* =========================================================================
   QuantLab shared runtime. Loaded by every page (home + lessons).

   Responsibilities:
     1. Build the sidebar table of contents from QUANTLAB_MANIFEST.
     2. Wire Prev / Next navigation based on the current lesson id.
     3. Turn every <textarea class="src"> inside a .code-block into a Monaco
        editor with Copy + Download buttons (graceful <pre> fallback offline).
     4. Track per-lesson completion in localStorage and reflect it everywhere.

   Lessons stay clean: they only write semantic markup. All behavior is here,
   so adding a lesson never means copy-pasting JavaScript.
   ========================================================================= */
(function () {
  "use strict";

  var MANIFEST = window.QUANTLAB_MANIFEST || [];
  var STORAGE_KEY = "quantlab.progress.v1";
  var MONACO_VS = "https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs";

  // ---- tiny helpers ------------------------------------------------------
  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === "class") node.className = attrs[k];
      else if (k === "html") node.innerHTML = attrs[k];
      else node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) {
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  }
  function basePrefix() {
    // Lessons live in /lessons/, the home page at repo root. Resolve relative
    // links so the site works whether opened from / or /lessons/.
    return document.body.getAttribute("data-root") === "lessons" ? "../" : "";
  }
  function lessonHref(id) { return basePrefix() + "lessons/" + id + ".html"; }

  // ---- progress (localStorage) ------------------------------------------
  function loadProgress() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
    catch (e) { return {}; }
  }
  function saveProgress(p) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(p)); } catch (e) {}
  }
  function isDone(id) { return !!loadProgress()[id]; }
  function setDone(id, done) {
    var p = loadProgress();
    if (done) p[id] = Date.now(); else delete p[id];
    saveProgress(p);
  }

  // ---- sidebar -----------------------------------------------------------
  function buildSidebar() {
    var sidebar = document.querySelector(".sidebar");
    if (!sidebar) return;
    var currentId = document.body.getAttribute("data-lesson-id");
    var progress = loadProgress();

    sidebar.appendChild(el("a", { class: "brand", href: basePrefix() + "index.html" }, ["QuantLab"]));
    sidebar.appendChild(el("div", { class: "tagline" }, ["Master quantitative trading in Python"]));

    var lastModule = null;
    MANIFEST.forEach(function (lesson) {
      if (lesson.module !== lastModule) {
        sidebar.appendChild(el("div", { class: "module-label" }, [lesson.module]));
        lastModule = lesson.module;
      }
      var link = el("a", {
        class: "lesson-link" + (lesson.id === currentId ? " active" : ""),
        href: lessonHref(lesson.id),
      });
      link.appendChild(el("span", { class: "done" }, [progress[lesson.id] ? "✓" : ""]));
      link.appendChild(el("span", {}, [lesson.title +
        (lesson.status === "soon" ? "  • soon" : "")]));
      sidebar.appendChild(link);
    });
  }

  // ---- prev / next -------------------------------------------------------
  function buildLessonNav() {
    var host = document.querySelector("[data-lesson-nav]");
    if (!host) return;
    var currentId = document.body.getAttribute("data-lesson-id");
    var idx = MANIFEST.findIndex(function (l) { return l.id === currentId; });
    if (idx === -1) return;

    var prev = MANIFEST[idx - 1];
    var next = MANIFEST[idx + 1];

    var prevA = el("a", { class: "prev" + (prev ? "" : " disabled"),
      href: prev ? lessonHref(prev.id) : "#" });
    prevA.appendChild(el("span", { class: "dir" }, ["← Previous"]));
    prevA.appendChild(el("span", { class: "title" }, [prev ? prev.title : "Start of course"]));

    var nextA = el("a", { class: "next" + (next ? "" : " disabled"),
      href: next ? lessonHref(next.id) : "#" });
    nextA.appendChild(el("span", { class: "dir" }, ["Next →"]));
    nextA.appendChild(el("span", { class: "title" }, [next ? next.title : "End of course"]));

    host.appendChild(prevA);
    host.appendChild(nextA);
  }

  // ---- mark-as-done control ---------------------------------------------
  function buildProgressControl() {
    var host = document.querySelector("[data-progress]");
    if (!host) return;
    var id = document.body.getAttribute("data-lesson-id");
    var btn = el("button", { class: "mark-done" });
    function render() {
      var done = isDone(id);
      btn.textContent = done ? "✓ Completed" : "Mark this lesson complete";
      btn.classList.toggle("is-done", done);
    }
    btn.addEventListener("click", function () { setDone(id, !isDone(id)); render(); buildJumpRefresh(); });
    render();
    host.appendChild(btn);
  }
  function buildJumpRefresh() {
    // Refresh sidebar check marks without a full reload.
    var sidebar = document.querySelector(".sidebar");
    if (sidebar) { sidebar.innerHTML = ""; buildSidebar(); }
  }

  // ---- code blocks: Monaco with graceful fallback ------------------------
  var editorQueue = [];
  function prepareCodeBlocks() {
    var blocks = document.querySelectorAll(".code-block");
    blocks.forEach(function (block, i) {
      var src = block.querySelector("textarea.src, pre.src");
      if (!src) return;
      var code = (src.value !== undefined ? src.value : src.textContent).replace(/\s+$/, "");
      var filename = block.getAttribute("data-filename") || ("snippet_" + i + ".py");
      var lang = block.getAttribute("data-lang") || "python";

      // Toolbar with Copy + Download.
      var copyBtn = el("button", {}, ["Copy"]);
      var dlBtn = el("button", {}, ["Download .py"]);
      var toolbar = el("div", { class: "code-toolbar" }, [
        el("span", { class: "filename" }, [filename]),
        el("div", { class: "btns" }, [copyBtn, dlBtn]),
      ]);

      copyBtn.addEventListener("click", function () {
        navigator.clipboard.writeText(code).then(function () {
          copyBtn.textContent = "Copied!";
          setTimeout(function () { copyBtn.textContent = "Copy"; }, 1200);
        });
      });
      dlBtn.addEventListener("click", function () {
        var blob = new Blob([code], { type: "text/x-python" });
        var url = URL.createObjectURL(blob);
        var a = el("a", { href: url, download: filename });
        document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(url);
      });

      // Replace the raw source with toolbar + an .editor mount point that
      // initially shows a styled <pre> (this IS the offline fallback).
      var editorDiv = el("div", { class: "editor" });
      var fallbackPre = el("pre", {}, [code]);
      editorDiv.appendChild(fallbackPre);

      block.innerHTML = "";
      block.appendChild(toolbar);
      block.appendChild(editorDiv);

      editorQueue.push({ editorDiv: editorDiv, fallbackPre: fallbackPre, code: code, lang: lang });
    });
  }

  function mountMonaco() {
    if (!editorQueue.length) return;
    // Load the AMD loader, then Monaco. If anything fails (offline), we simply
    // keep the <pre> fallbacks — the lesson is still fully readable & copyable.
    var loaderScript = document.createElement("script");
    loaderScript.src = MONACO_VS + "/loader.js";
    loaderScript.onerror = function () { /* keep fallbacks */ };
    loaderScript.onload = function () {
      if (!window.require) return;
      window.require.config({ paths: { vs: MONACO_VS } });
      window.require(["vs/editor/editor.main"], function () {
        defineTheme();
        editorQueue.forEach(function (item) {
          item.fallbackPre.remove();
          var lineCount = item.code.split("\n").length;
          item.editorDiv.style.height = Math.min(Math.max(lineCount * 19 + 22, 90), 640) + "px";
          monaco.editor.create(item.editorDiv, {
            value: item.code,
            language: item.lang,
            theme: "quantlab-dark",
            automaticLayout: true,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            fontSize: 13.5,
            fontFamily: "SF Mono, JetBrains Mono, Fira Code, Consolas, monospace",
            lineNumbers: "on",
            renderLineHighlight: "none",
            scrollbar: { alwaysConsumeMouseWheel: false },
            tabSize: 4,
          });
        });
      });
    };
    document.head.appendChild(loaderScript);
  }

  function defineTheme() {
    monaco.editor.defineTheme("quantlab-dark", {
      base: "vs-dark", inherit: true, rules: [],
      colors: { "editor.background": "#0b0f15" },
    });
  }

  // ---- mobile menu -------------------------------------------------------
  function buildMobileToggle() {
    var toggle = el("button", { class: "menu-toggle" }, ["☰"]);
    toggle.addEventListener("click", function () {
      var sb = document.querySelector(".sidebar");
      if (sb) sb.classList.toggle("open");
    });
    document.body.appendChild(toggle);
  }

  // ---- boot --------------------------------------------------------------
  document.addEventListener("DOMContentLoaded", function () {
    buildSidebar();
    buildLessonNav();
    buildProgressControl();
    buildMobileToggle();
    prepareCodeBlocks();
    mountMonaco();
  });
})();
