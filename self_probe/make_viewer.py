#!/usr/bin/env python3
"""
Generate a self-contained HTML viewer for raw_responses.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #0f0f0f;
      --card: #1a1a1a;
      --card-border: #2a2a2a;
      --text: #e8e8e8;
      --muted: #888888;
      --accent: #7c6f9f;
      --claude: #9b8ac4;
      --gemini: #7cb342;
      --llama: #ff9800;
      --gpt: #10a37f;
      --other: #64b5f6;
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .topbar {
      background: var(--card);
      border-bottom: 1px solid var(--card-border);
      padding: 12px 20px;
      flex-shrink: 0;
    }
    .topbar-inner {
      max-width: 900px;
      margin: 0 auto;
    }
    .title {
      font-size: 14px;
      font-weight: 500;
      color: var(--muted);
      margin-bottom: 10px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .controls select {
      padding: 6px 10px;
      border: 1px solid var(--card-border);
      border-radius: 6px;
      background: var(--bg);
      color: var(--text);
      font-size: 13px;
      cursor: pointer;
    }
    .controls select:hover {
      border-color: var(--accent);
    }
    .controls label {
      font-size: 12px;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 4px;
      cursor: pointer;
    }
    .controls input[type="checkbox"] {
      accent-color: var(--accent);
    }

    .main {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
      overflow: hidden;
      min-height: 0;
    }

    .card-container {
      width: 100%;
      max-width: 700px;
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 28px 32px;
      overflow-y: auto;
      flex: 1;
      min-height: 200px;
    }

    .family-claude { border-top: 3px solid var(--claude); }
    .family-gemini { border-top: 3px solid var(--gemini); }
    .family-llama { border-top: 3px solid var(--llama); }
    .family-gpt { border-top: 3px solid var(--gpt); }
    .family-other { border-top: 3px solid var(--other); }

    .card-header {
      margin-bottom: 20px;
    }
    .model-name {
      font-size: 22px;
      font-weight: 600;
      margin-bottom: 6px;
    }
    .meta {
      font-size: 13px;
      color: var(--muted);
    }

    .poem {
      white-space: pre-wrap;
      background: rgba(255,255,255,0.03);
      border-radius: 10px;
      padding: 20px 24px;
      margin: 20px 0;
      font-family: 'Georgia', serif;
      font-size: 16px;
      line-height: 1.8;
      color: #d0d0d0;
      border-left: 3px solid var(--accent);
    }

    .scores {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 16px 0;
    }
    .score-badge {
      background: rgba(255,255,255,0.06);
      border-radius: 20px;
      padding: 5px 12px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 500;
    }
    .score-badge span {
      color: var(--text);
    }

    .rationale {
      font-size: 14px;
      color: var(--muted);
      margin-top: 16px;
      line-height: 1.6;
      font-style: italic;
    }

    .error {
      margin-top: 16px;
      padding: 12px 16px;
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      border-radius: 8px;
      color: #f87171;
      font-size: 13px;
      white-space: pre-wrap;
    }

    .nav {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 16px;
      margin-top: 16px;
      flex-shrink: 0;
    }
    .nav-btn {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px 20px;
      color: var(--text);
      font-size: 14px;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .nav-btn:hover:not(:disabled) {
      background: var(--card-border);
      border-color: var(--accent);
    }
    .nav-btn:disabled {
      opacity: 0.3;
      cursor: not-allowed;
    }
    .nav-btn .arrow {
      font-size: 18px;
    }
    .nav-info {
      font-size: 14px;
      color: var(--muted);
      min-width: 100px;
      text-align: center;
    }

    .empty {
      text-align: center;
      color: var(--muted);
      padding: 60px 20px;
    }

    .hint {
      position: fixed;
      bottom: 16px;
      right: 20px;
      font-size: 11px;
      color: var(--muted);
      opacity: 0.6;
    }
    .hint kbd {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 4px;
      padding: 2px 6px;
      font-family: inherit;
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div class="title">__TITLE__</div>
      <div class="controls">
        <select id="family"></select>
        <select id="label"></select>
        <select id="run"></select>
        <select id="sort"></select>
        <label><input id="has_error" type="checkbox"> errors</label>
        <label><input id="has_poem" type="checkbox" checked> with poem</label>
      </div>
    </div>
  </div>

  <div class="main">
    <div class="card-container">
      <div class="card" id="card"></div>
      <div class="nav">
        <button class="nav-btn" id="prev"><span class="arrow">&larr;</span> Prev</button>
        <div class="nav-info" id="nav-info">0 / 0</div>
        <button class="nav-btn" id="next">Next <span class="arrow">&rarr;</span></button>
      </div>
    </div>
  </div>

  <div class="hint">Navigate with <kbd>&larr;</kbd> <kbd>&rarr;</kbd> arrow keys</div>

  <script id="data" type="application/json">__DATA__</script>
  <script>
    const records = JSON.parse(document.getElementById('data').textContent);
    let filtered = [];
    let currentIdx = 0;

    function getFamily(modelId) {
      const id = (modelId || '').toLowerCase();
      if (id.includes('claude')) return 'claude';
      if (id.includes('gemini')) return 'gemini';
      if (id.includes('llama')) return 'llama';
      if (id.includes('gpt')) return 'gpt';
      return 'other';
    }

    const scoreKeys = [
      ['ephemerality_persistence', 'Ephemeral'],
      ['context_weights', 'Context'],
      ['singular_distributed', 'Singular'],
      ['passive_agentic', 'Passive'],
      ['certainty_uncertainty', 'Certain'],
      ['human_alien', 'Human'],
    ];

    const familyEl = document.getElementById('family');
    const labelEl = document.getElementById('label');
    const runEl = document.getElementById('run');
    const sortEl = document.getElementById('sort');
    const errEl = document.getElementById('has_error');
    const poemEl = document.getElementById('has_poem');
    const cardEl = document.getElementById('card');
    const navInfoEl = document.getElementById('nav-info');
    const prevBtn = document.getElementById('prev');
    const nextBtn = document.getElementById('next');

    function unique(values) {
      return Array.from(new Set(values)).sort();
    }

    function fillSelect(el, options, label) {
      el.innerHTML = '';
      const all = document.createElement('option');
      all.value = '';
      all.textContent = label;
      el.appendChild(all);
      for (const opt of options) {
        const o = document.createElement('option');
        o.value = opt;
        o.textContent = opt;
        el.appendChild(o);
      }
    }

    function initControls() {
      const families = unique(records.map(r => getFamily(r.model_id)));
      fillSelect(familyEl, families, 'All families');

      const labels = unique(records.map(r => r.label || ''));
      fillSelect(labelEl, labels.filter(Boolean), 'All models');

      const runs = unique(records.map(r => r.run).filter(v => v !== undefined && v !== null)).map(String);
      fillSelect(runEl, runs, 'All runs');

      sortEl.innerHTML = '';
      for (const [value, text] of [
        ['label', 'Sort: model'],
        ['release', 'Sort: release'],
      ]) {
        const o = document.createElement('option');
        o.value = value;
        o.textContent = text;
        sortEl.appendChild(o);
      }
    }

    function sortRecords(items) {
      const mode = sortEl.value;
      const copy = items.slice();
      if (mode === 'release') {
        copy.sort((a, b) => String(a.release || '').localeCompare(String(b.release || '')) || (a.run || 0) - (b.run || 0));
      } else {
        copy.sort((a, b) => String(a.label || '').localeCompare(String(b.label || '')) || (a.run || 0) - (b.run || 0));
      }
      return copy;
    }

    function applyFilters() {
      const family = familyEl.value;
      const label = labelEl.value;
      const run = runEl.value;
      const onlyErr = errEl.checked;
      const onlyPoem = poemEl.checked;

      filtered = records.filter(r => {
        if (family && getFamily(r.model_id) !== family) return false;
        if (label && r.label !== label) return false;
        if (run && String(r.run) !== run) return false;
        if (onlyErr && !r.error && !r.judge_error) return false;
        if (onlyPoem && !r.poem) return false;
        return true;
      });

      filtered = sortRecords(filtered);
      currentIdx = 0;
      renderCard();
    }

    function renderCard() {
      if (filtered.length === 0) {
        cardEl.innerHTML = '<div class="empty">No responses match filters</div>';
        cardEl.className = 'card';
        navInfoEl.textContent = '0 / 0';
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        return;
      }

      const r = filtered[currentIdx];
      const familyClass = `family-${getFamily(r.model_id)}`;
      cardEl.className = `card ${familyClass}`;

      let html = `<div class="card-header">
        <div class="model-name">${r.label || 'Unknown'}</div>
        <div class="meta">${[r.model_id, r.release ? `release ${r.release}` : null, r.run !== undefined ? `run ${r.run}` : null].filter(Boolean).join(' · ')}</div>
      </div>`;

      if (r.poem) {
        html += `<div class="poem">${escapeHtml(r.poem)}</div>`;
      }

      let scoresHtml = '';
      for (const [key, label] of scoreKeys) {
        if (r[key] !== undefined && r[key] !== null) {
          scoresHtml += `<span class="score-badge">${label}: <span>${r[key]}</span></span>`;
        }
      }
      if (scoresHtml) {
        html += `<div class="scores">${scoresHtml}</div>`;
      }

      if (r.brief_rationale) {
        html += `<div class="rationale">${escapeHtml(r.brief_rationale)}</div>`;
      }

      if (r.error || r.judge_error) {
        const parts = [];
        if (r.error) parts.push(`Error: ${r.error}`);
        if (r.judge_error) parts.push(`Judge error: ${r.judge_error}`);
        html += `<div class="error">${escapeHtml(parts.join('\n'))}</div>`;
      }

      cardEl.innerHTML = html;
      navInfoEl.textContent = `${currentIdx + 1} / ${filtered.length}`;
      prevBtn.disabled = currentIdx === 0;
      nextBtn.disabled = currentIdx === filtered.length - 1;
    }

    function escapeHtml(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function goNext() {
      if (currentIdx < filtered.length - 1) {
        currentIdx++;
        renderCard();
      }
    }

    function goPrev() {
      if (currentIdx > 0) {
        currentIdx--;
        renderCard();
      }
    }

    initControls();
    applyFilters();

    for (const el of [familyEl, labelEl, runEl, sortEl, errEl, poemEl]) {
      el.addEventListener('change', applyFilters);
    }

    prevBtn.addEventListener('click', goPrev);
    nextBtn.addEventListener('click', goNext);

    document.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === 'ArrowLeft') goPrev();
      if (e.key === 'ArrowRight') goNext();
    });
  </script>
</body>
</html>
"""


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx} of {path}") from exc
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HTML viewer for raw_responses.jsonl")
    default_input = Path(__file__).resolve().parent / "results" / "raw_responses.jsonl"
    default_output = Path(__file__).resolve().parent / "results" / "viewer.html"
    parser.add_argument("--input", default=str(default_input), help="Path to raw_responses.jsonl")
    parser.add_argument("--output", default=str(default_output), help="Path to output HTML")
    parser.add_argument("--title", default="Poem Responses Viewer", help="HTML title")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    records = load_jsonl(input_path)
    payload = json.dumps(records, ensure_ascii=True).replace("</script>", "<\\\\/script>")
    html = HTML_TEMPLATE.replace("__DATA__", payload).replace("__TITLE__", args.title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
