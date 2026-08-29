#!/usr/bin/env python
"""Evaluate the Graph-Refinement Transformer on the retargeting task.

One forward pass, no post-processing. Scored with project4's own metrics so the
numbers sit beside its tables, plus PhyScene's physical suite. The bar is set in
DESIGN.md: feasibility WITHOUT a projection should approach project4's
*post*-projection level, with no relational regression.
"""
import os, sys, argparse, copy
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import numpy as np, torch
from scipy import stats as sstats

from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import (uniform_scale, aspect_deform, corner_cut,
                                _anchor_openings, _replace_openings)
from reroom.core.scene import Room
from reroom.eval.metrics import evaluate
from reroom.eval.physcene import physcene_metrics
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.target import build_design_intent
from reroom.generative.tokens import build_tokens, collate, from_frame
from reroom.generative.refiner import GraphRefinementTransformer
from reroom.core.scene import Scene

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/crt_walk/crt_best.pt")
ap.add_argument("--seeds", default="6,8,25,2,10,14,1,3,5,7,9,11")
ap.add_argument("--mode", default="uniform", choices=["uniform", "aniso", "hard"])
a = ap.parse_args()

SIZES = {"uniform": [0.75, 1.0, 1.35],
         "aniso":   [(1.5, 0.75), (0.75, 1.5), (1.7, 0.85), (0.85, 1.7)]}


def mkroom(r, s):
    if s == "L":
        p = corner_cut(r.polygon, 0, 0.5, 0.5, 0.0)
    elif isinstance(s, (tuple, list)):
        p = aspect_deform(r.polygon, float(s[0]), float(s[1]))
    else:
        p = uniform_scale(r.polygon, s)
    an = _anchor_openings(r)
    return Room(polygon=p, height=r.height,
                openings=_replace_openings(p, an, len(r.polygon)), room_type=r.room_type)


def run(model, graph, room, el, dev):
    """One forward pass; write the poses back into a Scene."""
    intent = build_design_intent(graph, room, elasticity=el)
    src = graph.scene
    intent.source = Scene(scene_id=src.scene_id, room=src.room,
                          objects=[o.copy() for o in src.objects], source=src.source)
    item = build_tokens(intent, room, None)
    batch = collate([item], device=dev)
    with torch.no_grad():
        x = model(batch)[0].cpu().numpy()
    fr = item.meta["frame_tgt"]
    out = Scene(scene_id=f"{src.scene_id}__grt", room=room.copy(),
                objects=[o.copy() for o in intent.source.objects], source="grt")
    for i, o in enumerate(out.objects):
        if i >= x.shape[0]:
            break
        p, ang = from_frame(x[i], fr)
        o.xy = np.asarray(p, dtype=float); o.yaw = float(ang)
    return out


el = load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
scenes = [s for s in iter_scenes("data/processed", limit=None, min_objects=6)
          if s.room.room_type in ("bedroom", "living_room")]
_, _, test = split_scenes(scenes)
seeds = [int(v) for v in a.seeds.split(",")]
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
cfg = ck.get("cfg", {})
model = GraphRefinementTransformer(cfg.get("d_model", 384), cfg.get("n_blocks", 6),
                                   cfg.get("heads", 8)).to(dev)
model.load_state_dict(ck.get("ema", ck["model"])); model.eval()
print(f"loaded {a.ckpt} (epoch {ck.get('epoch')})", flush=True)

sizes = SIZES.get(a.mode, ["L"]) if a.mode != "hard" else ["L"]
K = ("S_rel", "S_rel_kept", "S_motif", "R_col", "ps_Col_obj", "ps_R_out",
     "ps_R_walkable", "ps_R_reach")
rows = {k: [] for k in K}
for sd in seeds:
    try:
        src = test[sd]; g = build_motifs(build_scene_graph(src))
        for s in sizes:
            room = mkroom(src.room, s)
            out = run(model, g, room, el, dev)
            m = evaluate(g, out); pm = physcene_metrics(out)
            rows["S_rel"].append(m["S_rel"]); rows["S_rel_kept"].append(m.get("S_rel_kept", m["S_rel"]))
            rows["S_motif"].append(m.get("S_motif", np.nan)); rows["R_col"].append(100 * m["R_col"])
            for k in ("ps_Col_obj", "ps_R_out", "ps_R_walkable", "ps_R_reach"):
                rows[k].append(pm[k])
    except Exception as e:
        print("skip", sd, repr(e)[:70], flush=True)

def mn(v):
    v = np.array(v, dtype=float); v = v[~np.isnan(v)]
    return (v.mean(), v.std()) if len(v) else (float("nan"),) * 2

n = len(rows["S_rel"])
print(f"\nGRT ({a.mode}), N={n} cells — ONE forward pass, no post-processing")
for k in K:
    m, s = mn(rows[k])
    print(f"  {k:<14} {m:8.3f} ± {s:.3f}")
print("DONE_GRT_EVAL")
