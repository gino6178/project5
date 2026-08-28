"""Two-question A/B human evaluation (plan section 15.3).

The plan insists the two questions be asked *separately*:

  1. which result better preserves the reference room's layout / design style?
  2. which result better suits the target room and looks more usable?

Collapsing them into one "which is better?" makes the study uninterpretable,
because a method can win by being a good room while abandoning the reference,
or by copying the reference into a room it does not fit.  This module builds a
self-contained HTML instrument that keeps them apart, randomises left/right per
trial, records per-trial timing, and writes an answer key so the responses can
be scored without trusting the page.
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
from dataclasses import dataclass, field

import numpy as np

from ..core.scene import Scene
from ..render.topdown import figure_comparison
from ..render.scene3d import render_scene3d

__all__ = ["Trial", "build_study", "score_responses"]


@dataclass
class Trial:
    case_id: str
    reference: Scene
    target_empty: Scene            # target room with no furniture
    a: Scene
    b: Scene
    method_a: str
    method_b: str
    note: str = ""


def _png_b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _panel(scene: Scene, path: str, title: str, three_d: bool) -> str:
    if three_d:
        render_scene3d(scene, path, title=None, figsize=3.4, dpi=120)
    else:
        figure_comparison([(title, scene)], path, per_panel=3.4, labels=True)
    return _png_b64(path)


def build_study(trials: list[Trial], out_dir: str, title: str = "ReRoom study",
                three_d: bool = False, seed: int = 0) -> dict:
    """Write ``study.html`` plus ``key.json``; returns a small manifest."""
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "_img")
    os.makedirs(img_dir, exist_ok=True)
    rng = random.Random(seed)

    items = []
    key = []
    for t in trials:
        flip = rng.random() < 0.5
        left, right = (t.b, t.a) if flip else (t.a, t.b)
        lname, rname = (t.method_b, t.method_a) if flip else (t.method_a, t.method_b)
        base = os.path.join(img_dir, t.case_id)
        items.append({
            "id": t.case_id,
            "ref": _panel(t.reference, base + "_ref.png", "reference room", three_d),
            "empty": _panel(t.target_empty, base + "_empty.png", "target room", False),
            "left": _panel(left, base + "_L.png", "result A", three_d),
            "right": _panel(right, base + "_R.png", "result B", three_d),
            "note": t.note,
        })
        key.append({"id": t.case_id, "left": lname, "right": rname})

    html = _HTML.replace("__TITLE__", title).replace(
        "__DATA__", json.dumps(items)).replace("__N__", str(len(items)))
    html_path = os.path.join(out_dir, "study.html")
    with open(html_path, "w") as fh:
        fh.write(html)
    with open(os.path.join(out_dir, "key.json"), "w") as fh:
        json.dump(key, fh, indent=1)
    return {"html": html_path, "n_trials": len(items),
            "key": os.path.join(out_dir, "key.json")}


def score_responses(responses_path: str, key_path: str) -> dict:
    """Turn raw responses into per-method win rates for both questions."""
    with open(responses_path) as fh:
        resp = json.load(fh)
    with open(key_path) as fh:
        key = {k["id"]: k for k in json.load(fh)}
    out: dict = {"preservation": {}, "suitability": {}, "n": 0, "n_raters": 0}
    raters = set()
    for r in resp:
        k = key.get(r.get("id"))
        if not k:
            continue
        raters.add(r.get("rater", "anon"))
        out["n"] += 1
        for q, field in (("preservation", "q1"), ("suitability", "q2")):
            side = r.get(field)
            if side not in ("left", "right"):
                continue
            winner = k[side]
            loser = k["right" if side == "left" else "left"]
            d = out[q]
            d.setdefault(winner, {"win": 0, "loss": 0})["win"] += 1
            d.setdefault(loser, {"win": 0, "loss": 0})["loss"] += 1
    out["n_raters"] = len(raters)
    for q in ("preservation", "suitability"):
        for m, c in out[q].items():
            n = c["win"] + c["loss"]
            c["rate"] = c["win"] / n if n else float("nan")
            c["n"] = n
            # Wilson 95% interval, so a 3-rater pilot cannot look decisive
            if n:
                p, z = c["rate"], 1.96
                den = 1 + z * z / n
                cen = (p + z * z / (2 * n)) / den
                half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
                c["ci95"] = [max(0.0, cen - half), min(1.0, cen + half)]
    return out


_HTML = """<meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;
      background:#f6f6f4;color:#1c1c1c}
 header{padding:14px 20px;background:#fff;border-bottom:1px solid #e2e2de;
        position:sticky;top:0;z-index:5}
 h1{font-size:16px;margin:0 0 4px}
 .sub{font-size:13px;color:#585851}
 main{max-width:1180px;margin:0 auto;padding:18px}
 .ctx{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
 .card{background:#fff;border:1px solid #e2e2de;border-radius:10px;padding:10px}
 .card h3{margin:0 0 6px;font-size:12px;letter-spacing:.04em;
          text-transform:uppercase;color:#6b6b63}
 .cmp{display:grid;grid-template-columns:1fr 1fr;gap:14px}
 img{width:100%;display:block;border-radius:6px}
 .q{background:#fff;border:1px solid #e2e2de;border-radius:10px;padding:12px;
    margin-top:14px}
 .q p{margin:0 0 8px;font-size:14px}
 button{font:inherit;padding:8px 14px;margin-right:8px;border-radius:8px;
        border:1px solid #cfcfc9;background:#fafaf8;cursor:pointer}
 button.sel{background:#1c1c1c;color:#fff;border-color:#1c1c1c}
 .bar{height:5px;background:#e2e2de;border-radius:3px;overflow:hidden;margin-top:8px}
 .bar > div{height:100%;background:#1c1c1c;width:0}
 #done{padding:20px;font-size:14px}
 textarea{width:100%;height:150px;font-family:ui-monospace,monospace;font-size:12px}
</style>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">Trial <span id="i">1</span> of __N__ &mdash; the two questions
  are independent; a result may win one and lose the other.</div>
  <div class="bar"><div id="p"></div></div>
</header>
<main>
 <div id="trial">
  <div class="ctx">
    <div class="card"><h3>Reference room (the design to carry over)</h3>
      <img id="ref"></div>
    <div class="card"><h3>Target room (empty)</h3><img id="empty"></div>
  </div>
  <div class="cmp">
    <div class="card"><h3>Result A</h3><img id="left"></div>
    <div class="card"><h3>Result B</h3><img id="right"></div>
  </div>
  <div class="q">
    <p><b>Q1.</b> Which result better preserves the reference room's furniture
       arrangement and design style?</p>
    <button data-q="q1" data-v="left">A</button>
    <button data-q="q1" data-v="right">B</button>
    <button data-q="q1" data-v="tie">No difference</button>
  </div>
  <div class="q">
    <p><b>Q2.</b> Which result better suits the target room &mdash; would be
       more usable to actually live in?</p>
    <button data-q="q2" data-v="left">A</button>
    <button data-q="q2" data-v="right">B</button>
    <button data-q="q2" data-v="tie">No difference</button>
  </div>
  <div class="q"><button id="next">Next trial</button>
    <span class="sub" id="warn"></span></div>
 </div>
 <div id="done" style="display:none">
   <p>Done. Copy the JSON below (or use the download button) and send it back.</p>
   <p><button id="dl">Download responses.json</button></p>
   <textarea id="out"></textarea>
 </div>
</main>
<script>
const DATA = __DATA__;
let i = 0, ans = {}, out = [], t0 = Date.now();
const rater = "r" + Math.random().toString(36).slice(2, 8);
function show(){
  const d = DATA[i];
  for (const k of ["ref","empty","left","right"])
    document.getElementById(k).src = "data:image/png;base64," + d[k];
  document.getElementById("i").textContent = i + 1;
  document.getElementById("p").style.width = (100*i/DATA.length) + "%";
  ans = {}; t0 = Date.now();
  document.querySelectorAll("button[data-q]").forEach(b=>b.classList.remove("sel"));
  document.getElementById("warn").textContent = "";
}
document.querySelectorAll("button[data-q]").forEach(b=>{
  b.onclick = () => {
    ans[b.dataset.q] = b.dataset.v;
    document.querySelectorAll(`button[data-q="${b.dataset.q}"]`)
      .forEach(x=>x.classList.remove("sel"));
    b.classList.add("sel");
  };
});
document.getElementById("next").onclick = () => {
  if (!ans.q1 || !ans.q2){
    document.getElementById("warn").textContent = "Please answer both questions.";
    return;
  }
  out.push({id: DATA[i].id, rater: rater, q1: ans.q1, q2: ans.q2,
            ms: Date.now() - t0});
  i++;
  if (i >= DATA.length){
    document.getElementById("trial").style.display = "none";
    document.getElementById("done").style.display = "block";
    document.getElementById("p").style.width = "100%";
    document.getElementById("out").value = JSON.stringify(out, null, 1);
  } else show();
};
document.getElementById("dl").onclick = () => {
  const b = new Blob([JSON.stringify(out, null, 1)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "responses.json"; a.click();
};
show();
</script>
"""
