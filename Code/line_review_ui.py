import json
import os
import re
import shutil
import socketserver
import threading
import time
import webbrowser
from glob import glob
from dataclasses import dataclass, asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import urlparse

import pandas as pd

try:
    from IPython.display import IFrame, display
except Exception:  # pragma: no cover
    IFrame = None
    display = None


DEFAULT_REASON_OPTIONS = [
    "good",
    "continuum_low",
    "continuum_high",
    "strong_line",
    "crowded_region",
    "low_EW_value",
    "other",
]

QA_META_COLS = ["qa_reason", "qa_note", "qa_png", "qa_timestamp"]

_FE_PATTERN = re.compile(r"^FeFit_\d+_(?P<element>.+)_(?P<waveA>\d+(?:\.\d+)?)\.png$")
_ELE_PATTERN = re.compile(r"^(?P<element>.+)_\d+_(?P<waveA>\d+(?:\.\d+)?)\.png$")


@dataclass
class ReviewCandidate:
    image_path: str
    image_name: str
    element: str
    wave_a: float
    wave_nm: float
    reason: str = "good"
    note: str = ""


def _round4(x: Any) -> float:
    return round(float(x), 4)


def _clean_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    s = str(v).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    return s


def _safe_read_tsv(path: str) -> pd.DataFrame:
    if (path is None) or (not os.path.exists(path)):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def _parse_png_name(file_name: str) -> tuple[str, float] | None:
    m = _FE_PATTERN.match(file_name)
    if m:
        return m.group("element").strip(), float(m.group("waveA"))
    m = _ELE_PATTERN.match(file_name)
    if m:
        return m.group("element").strip(), float(m.group("waveA"))
    return None


def _split_reason_suffix(file_name: str, reason_options: list[str]) -> tuple[str, str | None]:
    stem, ext = os.path.splitext(file_name)
    non_good = [x for x in reason_options if x != "good"]
    non_good.sort(key=len, reverse=True)
    for reason in non_good:
        suffix = "_" + reason
        if stem.endswith(suffix):
            return stem[: -len(suffix)] + ext, reason
    return file_name, None


def _good_dir_from_image_path(image_path: str, figure_dir: str) -> str:
    if not image_path:
        return figure_dir
    d = os.path.dirname(image_path)
    if os.path.basename(d).lower() == "deleted":
        return os.path.dirname(d)
    return d if d else figure_dir


def _deleted_path_for_reason(image_name: str, good_dir: str, reason: str) -> str:
    stem, ext = os.path.splitext(image_name)
    return os.path.join(good_dir, "deleted", f"{stem}_{reason}{ext}")


def _resolve_image_path(item: ReviewCandidate, figure_dir: str, reason_options: list[str]) -> str:
    good_dir = _good_dir_from_image_path(item.image_path, figure_dir)
    root_path = os.path.join(good_dir, item.image_name)
    if item.reason != "good":
        target = _deleted_path_for_reason(item.image_name, good_dir, item.reason)
        if os.path.exists(target):
            return target
    if os.path.exists(root_path):
        return root_path
    for reason in [x for x in reason_options if x != "good"]:
        p = _deleted_path_for_reason(item.image_name, good_dir, reason)
        if os.path.exists(p):
            return p
    # 兜底：deleted 目录里查找同 stem 的任意状态后缀文件
    stem, ext = os.path.splitext(item.image_name)
    fallback = sorted(glob(os.path.join(good_dir, "deleted", f"{stem}_*{ext}")))
    if fallback:
        return fallback[0]
    return item.image_path


def _sync_one_image(item: ReviewCandidate, figure_dir: str, reason_options: list[str]) -> None:
    good_dir = _good_dir_from_image_path(item.image_path, figure_dir)
    os.makedirs(os.path.join(good_dir, "deleted"), exist_ok=True)
    root_path = os.path.join(good_dir, item.image_name)
    variant_paths = {
        reason: _deleted_path_for_reason(item.image_name, good_dir, reason)
        for reason in reason_options
        if reason != "good"
    }

    if item.reason == "good":
        if not os.path.exists(root_path):
            src = None
            for p in variant_paths.values():
                if os.path.exists(p):
                    src = p
                    break
            if src:
                if os.path.exists(root_path):
                    os.remove(root_path)
                shutil.move(src, root_path)
        for p in variant_paths.values():
            if os.path.exists(p):
                os.remove(p)
        item.image_path = root_path
        return

    target = variant_paths.get(item.reason)
    if target is None:
        item.reason = "good"
        item.image_path = root_path
        return

    src = root_path if os.path.exists(root_path) else None
    if src is None:
        for p in variant_paths.values():
            if os.path.exists(p):
                src = p
                break
    if src and (os.path.abspath(src) != os.path.abspath(target)):
        if os.path.exists(target):
            os.remove(target)
        shutil.move(src, target)
    for reason, p in variant_paths.items():
        if reason != item.reason and os.path.exists(p):
            os.remove(p)
    item.image_path = target if os.path.exists(target) else _resolve_image_path(item, figure_dir, reason_options)


def _sync_images_for_items(
    items: list[ReviewCandidate], figure_dir: str, reason_options: list[str]
) -> int:
    for item in items:
        _sync_one_image(item, figure_dir, reason_options)
    return sum(1 for x in items if x.reason != "good")


def _refresh_deleted_summary(figure_dir: str) -> int:
    """
    将各子目录的 deleted 图汇总到 figure_dir/deleted。
    - 保留原子目录 deleted，不做删除。
    - 顶层 deleted 仅作为汇总视图。
    """
    top_deleted = os.path.join(figure_dir, "deleted")
    os.makedirs(top_deleted, exist_ok=True)

    summary_sources: dict[str, str] = {}
    has_nested_deleted = False
    for root, _, files in os.walk(figure_dir):
        if os.path.basename(root).lower() != "deleted":
            continue
        if os.path.abspath(root) == os.path.abspath(top_deleted):
            continue
        has_nested_deleted = True
        for name in files:
            if not name.lower().endswith(".png"):
                continue
            src = os.path.join(root, name)
            key = name
            # 防止重名冲突：不同子目录同名文件时，带上父目录名前缀
            if key in summary_sources and os.path.abspath(summary_sources[key]) != os.path.abspath(src):
                parent_name = os.path.basename(os.path.dirname(root))
                key = f"{parent_name}__{name}"
            summary_sources[key] = src

    # 若不存在子目录 deleted（例如 Fe 场景），top_deleted 不是汇总目录而是主存储目录，不能改写。
    if not has_nested_deleted:
        return len([f for f in os.listdir(top_deleted) if f.lower().endswith(".png")])

    existing = {f for f in os.listdir(top_deleted) if f.lower().endswith(".png")}
    expected = set(summary_sources.keys())

    # 删除过期汇总文件
    for old_name in (existing - expected):
        old_path = os.path.join(top_deleted, old_name)
        if os.path.isfile(old_path):
            os.remove(old_path)

    # 更新/创建汇总文件
    for out_name, src_path in summary_sources.items():
        out_path = os.path.join(top_deleted, out_name)
        shutil.copy2(src_path, out_path)

    return len(summary_sources)


def collect_review_candidates(figure_dir: str, reason_options: list[str]) -> list[ReviewCandidate]:
    if not os.path.isdir(figure_dir):
        return []

    out_map: dict[str, ReviewCandidate] = {}
    for root, _, files in os.walk(figure_dir):
        is_deleted_dir = os.path.basename(root).lower() == "deleted"
        for name in files:
            if not name.lower().endswith(".png"):
                continue
            canonical_name, suffix_reason = _split_reason_suffix(name, reason_options)
            parsed = _parse_png_name(canonical_name)
            if parsed is None:
                continue
            element, wave_a = parsed
            default_reason = suffix_reason if suffix_reason else "good"
            cand = ReviewCandidate(
                image_path=os.path.join(root, name),
                image_name=canonical_name,
                element=element,
                wave_a=wave_a,
                wave_nm=wave_a / 10.0,
                reason=default_reason,
            )
            if canonical_name not in out_map:
                out_map[canonical_name] = cand
            else:
                # 如果两处都存在，优先保留主目录（good）的路径
                if (not is_deleted_dir) and (os.path.basename(os.path.dirname(out_map[canonical_name].image_path)).lower() == "deleted"):
                    out_map[canonical_name] = cand
    out = list(out_map.values())
    out.sort(key=lambda x: (x.element, x.wave_nm, x.image_name))
    return out


def _load_linelist_reference(linelist_reference_path: str | None) -> pd.DataFrame:
    df = _safe_read_tsv(linelist_reference_path) if linelist_reference_path else pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if "element" in df.columns:
        df["element"] = df["element"].astype(str).str.strip()
    if "wave_nm" not in df.columns and ("wave_A" in df.columns):
        df["wave_nm"] = pd.to_numeric(df["wave_A"], errors="coerce") / 10.0
    if "wave_nm" in df.columns:
        df["wave_nm"] = pd.to_numeric(df["wave_nm"], errors="coerce")
    return df


def _build_new_deleted_rows(
    selected: list[ReviewCandidate], linelist_ref_df: pd.DataFrame
) -> pd.DataFrame:
    to_delete = [x for x in selected if x.reason != "good"]
    if not to_delete:
        return pd.DataFrame()

    now = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    has_ref = not linelist_ref_df.empty and ("element" in linelist_ref_df.columns)

    for item in to_delete:
        row_data: dict[str, Any] = {}
        if has_ref:
            ref = linelist_ref_df.copy()
            ref["element"] = ref["element"].astype(str).str.strip()
            if "wave_A" in ref.columns:
                ref["wave_A"] = pd.to_numeric(ref["wave_A"], errors="coerce")
            if "wave_nm" in ref.columns:
                ref["wave_nm"] = pd.to_numeric(ref["wave_nm"], errors="coerce")
            elif "wave_A" in ref.columns:
                ref["wave_nm"] = ref["wave_A"] / 10.0

            # 图文件名只有两位小数，优先按 element + wave_A 最近匹配，避免精度丢失导致记录不全。
            subset = ref[ref["element"] == item.element].copy()
            if not subset.empty and ("wave_A" in subset.columns):
                subset = subset.dropna(subset=["wave_A"])
                if not subset.empty:
                    subset["_delta"] = (subset["wave_A"] - float(item.wave_a)).abs()
                    subset = subset.sort_values("_delta")
                    best = subset.iloc[0]
                    if float(best["_delta"]) <= 0.03:
                        row_data = best.drop(labels=["_delta"]).to_dict()
            if (not row_data) and (not subset.empty) and ("wave_nm" in subset.columns):
                subset = subset.dropna(subset=["wave_nm"])
                if not subset.empty:
                    subset["_delta_nm"] = (subset["wave_nm"] - float(item.wave_nm)).abs()
                    subset = subset.sort_values("_delta_nm")
                    best = subset.iloc[0]
                    if float(best["_delta_nm"]) <= 0.003:
                        row_data = best.drop(labels=["_delta_nm"]).to_dict()
        row_data["element"] = item.element
        if ("wave_nm" not in row_data) or pd.isna(row_data.get("wave_nm")):
            row_data["wave_nm"] = item.wave_nm
        if "wave_A" not in row_data or pd.isna(row_data.get("wave_A")):
            row_data["wave_A"] = item.wave_a
        row_data["qa_reason"] = item.reason
        row_data["qa_note"] = item.note
        row_data["qa_png"] = item.image_name
        row_data["qa_timestamp"] = now
        rows.append(row_data)

    return pd.DataFrame(rows)


def _upsert_deleted_table(
    deleted_tsv_path: str,
    new_rows: pd.DataFrame,
    reviewed_items: list[ReviewCandidate] | None = None,
    required_base_cols: list[str] | None = None,
) -> int:
    required_base_cols = required_base_cols or ["element", "wave_nm"]
    os.makedirs(os.path.dirname(deleted_tsv_path), exist_ok=True)
    existing = _safe_read_tsv(deleted_tsv_path)

    if existing.empty and new_rows.empty:
        cols = required_base_cols + QA_META_COLS
        pd.DataFrame(columns=cols).to_csv(deleted_tsv_path, sep="\t", index=False)
        return 0

    all_cols: list[str] = []
    for c in list(existing.columns) + list(new_rows.columns):
        if c not in all_cols:
            all_cols.append(c)
    for c in required_base_cols + QA_META_COLS:
        if c not in all_cols:
            all_cols.append(c)

    existing = existing.reindex(columns=all_cols)
    new_rows = new_rows.reindex(columns=all_cols)

    # 对“本轮已审阅”的候选线先清空旧记录，再写入本轮非 good 的结果。
    # 优先用 qa_png（图片名）作为主键；可避免波长精度损失导致二次运行覆盖不全。
    if reviewed_items:
        reviewed_pngs = {x.image_name for x in reviewed_items if x.image_name}
        if reviewed_pngs and ("qa_png" in existing.columns):
            existing = existing[~existing["qa_png"].astype(str).isin(reviewed_pngs)].copy()

        reviewed_keys = {(x.element.strip(), _round4(x.wave_nm)) for x in reviewed_items}
        if reviewed_keys and {"element", "wave_nm"}.issubset(existing.columns):
            existing["element"] = existing["element"].astype(str).str.strip()
            existing["wave_nm"] = pd.to_numeric(existing["wave_nm"], errors="coerce")
            existing = existing.dropna(subset=["wave_nm"])
            keep_mask = [
                (e, _round4(w)) not in reviewed_keys
                for e, w in zip(existing["element"].tolist(), existing["wave_nm"].tolist())
            ]
            existing = existing.loc[keep_mask].copy()

    combined = pd.concat([existing, new_rows], ignore_index=True)

    if "element" in combined.columns:
        combined["element"] = combined["element"].astype(str).str.strip()
    if "wave_nm" in combined.columns:
        combined["wave_nm"] = pd.to_numeric(combined["wave_nm"], errors="coerce")
        combined = combined.dropna(subset=["wave_nm"])
        combined["_wave_nm_r"] = combined["wave_nm"].round(4)
        combined = combined.drop_duplicates(subset=["element", "_wave_nm_r"], keep="last")
        combined = combined.drop(columns=["_wave_nm_r"])

    combined.to_csv(deleted_tsv_path, sep="\t", index=False)
    return len(new_rows)


def _build_payload(candidates: list[ReviewCandidate], reason_options: list[str]) -> dict[str, Any]:
    return {
        "reason_options": reason_options,
        "items": [asdict(x) for x in candidates],
    }


def _apply_existing_decisions(
    candidates: list[ReviewCandidate], deleted_tsv_path: str, reason_options: list[str]
) -> list[ReviewCandidate]:
    if not candidates:
        return candidates
    df = _safe_read_tsv(deleted_tsv_path)
    if df.empty:
        return candidates

    out = [ReviewCandidate(**asdict(x)) for x in candidates]
    by_png: dict[str, dict[str, Any]] = {}
    by_key: dict[tuple[str, float], dict[str, Any]] = {}

    if "qa_png" in df.columns:
        for _, r in df.dropna(subset=["qa_png"]).iterrows():
            by_png[str(r["qa_png"])] = r.to_dict()

    if {"element", "wave_nm"}.issubset(df.columns):
        tmp = df.copy()
        tmp["element"] = tmp["element"].astype(str).str.strip()
        tmp["wave_nm"] = pd.to_numeric(tmp["wave_nm"], errors="coerce")
        tmp = tmp.dropna(subset=["wave_nm"])
        for _, r in tmp.iterrows():
            by_key[(str(r["element"]).strip(), _round4(r["wave_nm"]))] = r.to_dict()

    valid_reasons = set(reason_options)
    for item in out:
        row = None
        if item.image_name in by_png:
            row = by_png[item.image_name]
        else:
            row = by_key.get((item.element.strip(), _round4(item.wave_nm)))
        if not row:
            continue

        reason = _clean_text(row.get("qa_reason", ""))
        note = _clean_text(row.get("qa_note", ""))
        if reason and (reason in valid_reasons):
            item.reason = reason
            item.note = note
        else:
            # 兼容旧表：没有 qa_reason 但该线在 deleted 表中，默认映射为 other
            item.reason = "other" if "other" in valid_reasons else "good"
            if not note and ("other" in valid_reasons):
                note = "legacy_deleted"
            item.note = note
    return out


def _make_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>EW Line Review</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 16px;
      background: #f8fafc;
      color: #0f172a;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}
    .topbar h2 {{
      margin: 0;
      color: #0f172a;
    }}
    .count-badge {{
      background: #e2e8f0;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 6px 10px;
      font-weight: 600;
      color: #0f172a;
    }}
    .container {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}
    .imgbox {{
      border: 1px solid #94a3b8;
      background: #ffffff;
      padding: 8px;
      text-align: center;
      min-height: 420px;
    }}
    img {{ max-width: 100%; max-height: 70vh; }}
    .controls button {{ margin-right: 8px; }}
    .meta {{ margin-top: 8px; color: #1e293b; font-weight: 500; }}
    .reason-group label {{ display: block; margin: 6px 0; color: #0f172a; font-size: 20px; }}
    textarea {{
      width: 100%;
      min-height: 70px;
      background: #ffffff;
      color: #0f172a;
      border: 1px solid #94a3b8;
    }}
    button {{
      background: #ffffff;
      color: #0f172a;
      border: 1px solid #94a3b8;
      border-radius: 6px;
      padding: 6px 12px;
      cursor: pointer;
      font-weight: 600;
    }}
    button:hover {{
      background: #e2e8f0;
    }}
    .footer {{ margin-top: 16px; }}
  </style>
</head>
<body>
  <div class="topbar">
    <h2>EW Line Quality Review</h2>
    <div class="count-badge" id="countBadge">Deleted 0/0</div>
  </div>
  <div class="container">
    <div>
      <div class="imgbox">
        <img id="lineImg" src="" alt="line image" />
      </div>
      <div class="meta" id="lineMeta"></div>
      <div class="controls" style="margin-top:12px;">
        <button onclick="prevItem()">Previous</button>
        <button onclick="nextItem()">Next</button>
      </div>
      <div class="meta">Keyboard: Left/Up = previous, Right/Down = next.</div>
    </div>
    <div>
      <h3>Status</h3>
      <div class="reason-group" id="reasons"></div>
      <div id="otherBox" style="display:none;">
        <label>Other note (required when reason=other)</label>
        <textarea id="otherNote" onchange="onNoteChange()"></textarea>
      </div>
    </div>
  </div>
  <div class="footer">
    <button onclick="doSave()">Save</button>
    <button onclick="doCancel()">Cancel</button>
    <span id="saveMsg" style="margin-left:10px;"></span>
  </div>
  <script>
    const payload = {payload_json};
    const items = payload.items || [];
    const reasons = payload.reason_options || [];
    let idx = 0;
    let pendingSync = Promise.resolve();

    function ensureDefaults() {{
      for (const it of items) {{
        if (!it.reason) it.reason = "good";
        if (typeof it.note !== "string") it.note = "";
      }}
    }}

    function renderReasons(cur) {{
      const root = document.getElementById("reasons");
      root.innerHTML = "";
      for (const r of reasons) {{
        const id = "r_" + r;
        const checked = (cur.reason === r) ? "checked" : "";
        const row = document.createElement("label");
        row.innerHTML = `<input type="radio" name="reason" id="${{id}}" value="${{r}}" ${{checked}} onchange="onReasonChange('${{r}}')" /> ${{r}}`;
        root.appendChild(row);
      }}
      const showOther = (cur.reason === "other");
      document.getElementById("otherBox").style.display = showOther ? "block" : "none";
      if (showOther) {{
        document.getElementById("otherNote").value = cur.note || "";
      }}
    }}

    function render() {{
      if (!items.length) {{
        document.body.innerHTML = "<h3>No PNG images found for review.</h3>";
        return;
      }}
      const deletedCount = items.filter(x => x.reason !== "good").length;
      document.getElementById("countBadge").textContent = `Deleted ${{deletedCount}}/${{items.length}}`;
      const cur = items[idx];
      document.getElementById("lineImg").src = "/image?i=" + idx + "&_t=" + Date.now();
      document.getElementById("lineMeta").textContent =
        `Index ${{idx+1}}/${{items.length}} | element=${{cur.element}} | wave_A=${{Number(cur.wave_a).toFixed(2)}} | wave_nm=${{Number(cur.wave_nm).toFixed(4)}}`;
      renderReasons(cur);
    }}

    async function prevItem() {{
      if (!items.length) return;
      await pendingSync;
      if (idx === 0) {{
        return;
      }}
      idx = idx - 1;
      render();
    }}

    async function nextItem() {{
      if (!items.length) return;
      await pendingSync;
      if (idx >= items.length - 1) {{
        // 到最后一张后不循环；再点下一张即自动保存
        await doSave();
        return;
      }}
      idx = idx + 1;
      render();
    }}

    function onReasonChange(v) {{
      items[idx].reason = v;
      if (v !== "other") items[idx].note = "";
      pendingSync = syncCurrentMark();
      render();
    }}

    function onNoteChange() {{
      items[idx].note = document.getElementById("otherNote").value || "";
      pendingSync = syncCurrentMark();
    }}

    async function syncCurrentMark() {{
      if (!items.length) return;
      await fetch("/mark", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{
          idx: idx,
          reason: items[idx].reason,
          note: items[idx].note || ""
        }})
      }});
    }}

    function validateItems() {{
      for (const it of items) {{
        if (it.reason === "other" && !(it.note || "").trim()) {{
          return "All items with reason=other must include a note.";
        }}
      }}
      return "";
    }}

    async function doSave() {{
      await pendingSync;
      const err = validateItems();
      if (err) {{
        alert(err);
        return;
      }}
      const res = await fetch("/save", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ items }})
      }});
      const text = await res.text();
      document.getElementById("saveMsg").textContent = text;
      setTimeout(() => window.close(), 300);
    }}

    async function doCancel() {{
      await fetch("/cancel", {{ method: "POST" }});
      document.getElementById("saveMsg").textContent = "Cancelled.";
      setTimeout(() => window.close(), 300);
    }}

    document.addEventListener("keydown", (ev) => {{
      if (["ArrowLeft", "ArrowUp"].includes(ev.key)) {{
        ev.preventDefault(); prevItem();
      }} else if (["ArrowRight", "ArrowDown"].includes(ev.key)) {{
        ev.preventDefault(); nextItem();
      }}
    }});

    ensureDefaults();
    render();
  </script>
</body>
</html>
"""


class _ReviewHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


def run_line_review(
    *,
    output_folder: str,
    target: str,
    figure_dir: str,
    deleted_tsv_path: str | None = None,
    linelist_reference_path: str | None = None,
    ui_mode: str = "inline",
    reason_options: list[str] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "skipped", "message": "line_review_enabled=False"}

    reason_options = reason_options or DEFAULT_REASON_OPTIONS
    if "good" not in reason_options:
        reason_options = ["good"] + [x for x in reason_options if x != "good"]

    candidates = collect_review_candidates(figure_dir, reason_options)
    if not candidates:
        print(f"[LineReview] no png files found under: {figure_dir}")
        return {"status": "empty", "count": 0}

    deleted_tsv_path = deleted_tsv_path or os.path.join(output_folder, "linemasks", "Lines_deleted.tsv")
    candidates = _apply_existing_decisions(candidates, deleted_tsv_path, reason_options)
    _sync_images_for_items(candidates, figure_dir, reason_options)
    _refresh_deleted_summary(figure_dir)
    linelist_ref = _load_linelist_reference(linelist_reference_path)
    payload = _build_payload(candidates, reason_options)
    html = _make_html(payload).encode("utf-8")

    state = {"done": False, "saved": False, "items": candidates, "message": ""}

    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, content: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._write(200, html, "text/html; charset=utf-8")
                return
            if parsed.path == "/image":
                query = dict(x.split("=", 1) for x in parsed.query.split("&") if "=" in x)
                idx = int(query.get("i", "0"))
                items_live: list[ReviewCandidate] = state["items"] or []
                if idx < 0 or idx >= len(items_live):
                    self._write(404, b"Not found", "text/plain")
                    return
                p = _resolve_image_path(items_live[idx], figure_dir, reason_options)
                if not os.path.exists(p):
                    self._write(404, b"Missing image", "text/plain")
                    return
                with open(p, "rb") as f:
                    content = f.read()
                self._write(200, content, "image/png")
                return
            self._write(404, b"Not found", "text/plain")

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n) if n > 0 else b""
            if self.path == "/mark":
                try:
                    body = json.loads(raw.decode("utf-8")) if raw else {}
                    idx = int(body.get("idx", -1))
                    reason = _clean_text(body.get("reason", "good"))
                    note = _clean_text(body.get("note", ""))
                    items_live: list[ReviewCandidate] = state["items"] or []
                    if 0 <= idx < len(items_live):
                        if reason not in reason_options:
                            reason = "good"
                        items_live[idx].reason = reason
                        items_live[idx].note = note
                        _sync_one_image(items_live[idx], figure_dir, reason_options)
                        _refresh_deleted_summary(figure_dir)
                        state["items"] = items_live
                    self._write(200, b"OK", "text/plain; charset=utf-8")
                except Exception as e:  # pragma: no cover
                    msg = f"Mark failed: {e}"
                    self._write(500, msg.encode("utf-8"), "text/plain; charset=utf-8")
                return
            if self.path == "/save":
                try:
                    body = json.loads(raw.decode("utf-8"))
                    items = body.get("items", [])
                    parsed_items: list[ReviewCandidate] = []
                    for it in items:
                        _reason = _clean_text(it.get("reason", "good"))
                        if _reason not in reason_options:
                            _reason = "good"
                        parsed_items.append(
                            ReviewCandidate(
                                image_path=str(it["image_path"]),
                                image_name=str(it.get("image_name", os.path.basename(str(it["image_path"])))),
                                element=str(it["element"]).strip(),
                                wave_a=float(it["wave_a"]),
                                wave_nm=float(it["wave_nm"]),
                                reason=_reason,
                                note=_clean_text(it.get("note", "")),
                            )
                        )
                    new_rows = _build_new_deleted_rows(parsed_items, linelist_ref)
                    n_new = _upsert_deleted_table(
                        deleted_tsv_path,
                        new_rows,
                        reviewed_items=parsed_items,
                    )
                    n_deleted = _sync_images_for_items(parsed_items, figure_dir, reason_options)
                    n_summary = _refresh_deleted_summary(figure_dir)
                    state["saved"] = True
                    state["done"] = True
                    state["items"] = parsed_items
                    state["message"] = (
                        f"Saved. {n_new} deleted rows updated in Lines_deleted.tsv; "
                        f"{n_deleted} images currently in deleted/; summary={n_summary}."
                    )
                    self._write(200, state["message"].encode("utf-8"), "text/plain; charset=utf-8")
                except Exception as e:  # pragma: no cover
                    msg = f"Save failed: {e}"
                    self._write(500, msg.encode("utf-8"), "text/plain; charset=utf-8")
                return
            if self.path == "/cancel":
                state["saved"] = False
                state["done"] = True
                state["message"] = "Cancelled. No changes saved."
                self._write(200, state["message"].encode("utf-8"), "text/plain; charset=utf-8")
                return
            self._write(404, b"Not found", "text/plain")

        def log_message(self, *args, **kwargs):  # noqa: D401
            return

    with _ReviewHTTPServer(("127.0.0.1", 0), Handler) as server:
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        url = f"http://127.0.0.1:{port}/"
        print(f"[LineReview] target={target} | images={len(candidates)}")
        print(f"[LineReview] opening UI in mode='{ui_mode}' -> {url}")

        if ui_mode == "browser":
            webbrowser.open(url, new=1)
        else:
            if (IFrame is None) or (display is None):
                print("[LineReview] IPython display not available, fallback to browser mode.")
                webbrowser.open(url, new=1)
            else:
                display(IFrame(src=url, width="100%", height=860))

        try:
            while not state["done"]:
                time.sleep(0.2)
        finally:
            server.shutdown()
            server.server_close()

    print(f"[LineReview] {state['message']}")
    return {
        "status": "saved" if state["saved"] else "cancelled",
        "message": state["message"],
        "count": len(candidates),
    }

