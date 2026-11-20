"""
08_make_photo_review.py

PURPOSE
    Build the manual photo-authenticity review planned in MI2 ("1 Unknown"):
    draw a random sample of 100 listings from the final analysis data and write
    a self-contained HTML page that shows each listing's main photo with buttons
    to flag it as OK, a computer rendering, virtual staging, heavily edited, or
    not a photo of the space. The reviewer opens the page in a web browser,
    flags every photo, and clicks "Download results" to save
    photo_review_results.csv into OUTPUT/tables/.

INPUT
    DATA/final/analysis_data.csv

OUTPUT
    OUTPUT/tables/photo_review_sample.csv   the 100 sampled listing IDs (seed 42)
    OUTPUT/photo_review.html                the review page (photos load from Airbnb)

NOTES
    - Photos are not saved anywhere; the page loads them from Airbnb's servers
      using picture_url, so it needs an internet connection.
    - Run from the top-level repository folder:  python SCRIPTS/08_make_photo_review.py
"""

import json
from pathlib import Path

import pandas as pd

N_REVIEW = 100   # number of photos to review (from the MI2 plan)
SEED = 42        # same seed as the rest of the pipeline

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "SCRIPTS" else Path(".")
DATA = ROOT / "DATA" / "final" / "analysis_data.csv"
OUT_TABLES = ROOT / "OUTPUT" / "tables"
OUT_HTML = ROOT / "OUTPUT" / "photo_review.html"

# Load the final data and draw a simple random sample of listings
df = pd.read_csv(DATA, dtype={"id": str}, low_memory=False)
sample = df.sample(n=N_REVIEW, random_state=SEED)[
    ["id", "picture_url", "borough_group", "room_type", "price"]
].reset_index(drop=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)
sample[["id", "borough_group", "room_type", "price"]].to_csv(OUT_TABLES / "photo_review_sample.csv", index=False)


def resized(url: str) -> str:
    """Request Airbnb's 720px-wide version of the photo (same as script 01)."""
    if "/pictures/" in url and "/im/pictures/" not in url:
        url = url.replace("/pictures/", "/im/pictures/", 1)
    return url + ("&" if "?" in url else "?") + "im_w=720"


# Records embedded in the page as JSON
records = [
    {"n": i + 1, "id": r.id, "url": resized(r.picture_url), "orig": r.picture_url,
     "borough": r.borough_group, "room": r.room_type, "price": round(float(r.price))}
    for i, r in sample.iterrows()
]

# The review page: plain HTML/CSS/JS, no external libraries
HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Photo authenticity review (100 listings)</title>
<style>
  :root { --ink:#1f2328; --muted:#57606a; --line:#d0d7de; --bg:#f6f8fa; --accent:#3f7a8c; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color:var(--ink); background:var(--bg); }
  header { position:sticky; top:0; background:#fff; border-bottom:1px solid var(--line); padding:12px 16px; z-index:2; }
  header h1 { font-size:17px; margin:0 0 6px; }
  .bar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; font-size:14px; }
  .progress { flex:1; min-width:140px; height:8px; background:#e6e9ec; border-radius:4px; overflow:hidden; }
  .progress div { height:100%; background:var(--accent); width:0; }
  button { font:inherit; border:1px solid var(--line); background:#fff; border-radius:8px; padding:8px 12px; cursor:pointer; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  main { max-width:760px; margin:0 auto; padding:16px; }
  .help { font-size:14px; color:var(--muted); line-height:1.5; background:#fff; border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-bottom:16px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:12px; margin-bottom:18px; overflow:hidden; }
  .card.done { border-color:var(--accent); }
  .imgwrap { background:#eee; min-height:200px; display:flex; align-items:center; justify-content:center; }
  .imgwrap img { width:100%; display:block; }
  .meta { padding:10px 14px 0; font-size:13px; color:var(--muted); }
  .choices { display:flex; flex-wrap:wrap; gap:8px; padding:10px 14px 14px; }
  .choices button.sel { background:var(--ink); color:#fff; border-color:var(--ink); }
  .choices button.sel.ok { background:#2e7d4f; border-color:#2e7d4f; }
  textarea { width:calc(100% - 28px); margin:0 14px 14px; font:inherit; font-size:13px; padding:6px 8px; border:1px solid var(--line); border-radius:6px; }
  #summary { font-size:14px; margin-top:8px; }
</style></head><body>
<header>
  <h1>Photo authenticity review</h1>
  <div class="bar">
    <span id="count">0 / 100 reviewed</span>
    <div class="progress"><div id="prog"></div></div>
    <button class="primary" id="dl">Download results</button>
  </div>
  <div id="summary"></div>
</header>
<main>
  <div class="help">
    For each photo, pick the label that fits best. Most photos will be <b>OK</b>.
    <br><b>Rendering</b>: computer-generated, not a real photo. <b>Virtual staging</b>: real room with furniture digitally added.
    <b>Heavily edited</b>: obvious filters, warped walls, fake sky/view, extreme HDR.
    <b>Not the space</b>: logo, map, floor plan, building exterior from far away, or unrelated image.
    <b>Won't load</b>: the image doesn't appear.
    <br>Your choices are saved in this browser as you go. When finished, click <b>Download results</b> and put the CSV in <code>OUTPUT/tables/</code>.
  </div>
  <div id="list"></div>
</main>
<script>
const DATA = __DATA__;
const LABELS = [["ok","OK"],["rendering","Rendering"],["virtual_staging","Virtual staging"],
                ["heavy_edit","Heavily edited"],["not_space","Not the space"],["no_load","Won't load"]];
const KEY = "photo_review_v1";
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }

const list = document.getElementById("list");
DATA.forEach(d => {
  const card = document.createElement("div"); card.className = "card"; card.id = "c" + d.n;
  card.innerHTML = `
    <div class="imgwrap"><img loading="lazy" alt="Listing ${d.n} main photo" src="${d.url}"></div>
    <div class="meta">#${d.n} · ${d.borough} · ${d.room} · $${d.price}/night · <a href="https://www.airbnb.com/rooms/${d.id}" target="_blank" rel="noopener">listing</a></div>
    <div class="choices"></div>
    <textarea rows="1" placeholder="Optional note"></textarea>`;
  const img = card.querySelector("img");
  img.onerror = () => { if (img.src !== d.orig) { img.src = d.orig; } };   // fall back to the original URL
  const box = card.querySelector(".choices");
  LABELS.forEach(([k, txt]) => {
    const b = document.createElement("button"); b.textContent = txt; b.dataset.k = k;
    if (k === "ok") b.classList.add("ok");
    b.onclick = () => { state[d.id] = Object.assign(state[d.id] || {}, {label: k}); save(); render(); };
    box.appendChild(b);
  });
  const note = card.querySelector("textarea");
  note.value = (state[d.id] && state[d.id].note) || "";
  note.oninput = () => { state[d.id] = Object.assign(state[d.id] || {}, {note: note.value}); save(); };
  list.appendChild(card);
});

function render() {
  let done = 0; const counts = {};
  DATA.forEach(d => {
    const s = state[d.id]; const card = document.getElementById("c" + d.n);
    card.querySelectorAll(".choices button").forEach(b => b.classList.toggle("sel", !!s && s.label === b.dataset.k));
    card.classList.toggle("done", !!(s && s.label));
    if (s && s.label) { done++; counts[s.label] = (counts[s.label] || 0) + 1; }
  });
  document.getElementById("count").textContent = `${done} / ${DATA.length} reviewed`;
  document.getElementById("prog").style.width = (100 * done / DATA.length) + "%";
  const flagged = ["rendering","virtual_staging","heavy_edit","not_space"].reduce((a, k) => a + (counts[k] || 0), 0);
  const loaded = done - (counts["no_load"] || 0);
  document.getElementById("summary").textContent = done
    ? `Flagged: ${flagged} of ${loaded} loaded photos (${loaded ? (100 * flagged / loaded).toFixed(1) : 0}%)` : "";
}
render();

document.getElementById("dl").onclick = () => {
  const esc = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const rows = [["review_n","id","borough_group","room_type","price","label","note"].join(",")];
  DATA.forEach(d => { const s = state[d.id] || {};
    rows.push([d.n, esc(d.id), esc(d.borough), esc(d.room), d.price, esc(s.label || ""), esc(s.note || "")].join(",")); });
  const blob = new Blob([rows.join("\\n")], {type: "text/csv"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "photo_review_results.csv"; document.body.appendChild(a); a.click(); a.remove();
};
</script></body></html>
"""

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUT_HTML.write_text(HTML.replace("__DATA__", json.dumps(records)), encoding="utf-8")
print(f"Sampled {len(sample)} listings (seed {SEED})")
print(f"Wrote {OUT_TABLES / 'photo_review_sample.csv'} and {OUT_HTML}")
