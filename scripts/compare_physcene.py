#!/usr/bin/env python
"""ReRoom vs PhyScene on the same rooms, scored by the same evaluator.

This is the head-to-head the earlier "yardstick" section could not be.  Three
things are held fixed:

* **the rooms** -- the 3D-FRONT test-split floor plans PhyScene generated into;
* **the object vocabulary** -- ReRoom's reference scenes are rebuilt from the
  same cached ``boxes.npz`` PhyScene trains and evaluates on, so neither side
  sees objects the other cannot;
* **the evaluator** -- ``reroom.eval.physcene`` scores both.

What differs is the only thing that should: PhyScene generates a layout for the
floor plan from a learned prior, while ReRoom transfers the reference room's
design into it.  The ground-truth row is the reference design in its own room,
which is the ceiling for anything that tries to preserve it.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from reroom.core.categories import canonical_category
from reroom.core.scene import ObjectInstance, Room, Scene
from reroom.data.asset_bank import AssetBank
from reroom.eval.physcene import physcene_metrics
from reroom.geom.polygon import normalize_polygon
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.retarget.baselines import run_baseline
from reroom.retarget.optimizer import RetargetConfig, retarget
from scripts.physcene_to_reroom import floor_polygon


def scene_from_cache(box_p: str, classes: list, room_type: str) -> Scene | None:
    """The ground-truth room, in PhyScene's own frame and vocabulary."""
    d = np.load(box_p, allow_pickle=True)
    poly, _ = floor_polygon(box_p)
    if poly is None or len(poly) < 3:
        return None
    room = Room(polygon=normalize_polygon(np.asarray(poly, dtype=float)),
                height=2.6, room_type=room_type)
    objs = []
    for j in range(len(d["translations"])):
        ci = int(np.argmax(d["class_labels"][j]))
        name = classes[ci] if ci < len(classes) else "misc"
        if name in ("empty", "start", "end"):
            continue
        t = np.asarray(d["translations"][j], dtype=float)
        s = np.abs(np.asarray(d["sizes"][j], dtype=float)) * 2.0
        ang = np.asarray(d["angles"][j], dtype=float)
        yaw = float(ang[0]) if ang.size == 1 else float(math.atan2(ang[1], ang[0]))
        objs.append(ObjectInstance(
            oid=f"gt_{j}", category=canonical_category(name, ""),
            raw_category=name,
            position=np.array([t[0], t[2], max(0.0, t[1] - s[1] / 2)]),
            yaw=yaw,
            size=np.array([max(s[0], 0.05), max(s[2], 0.05), max(s[1], 0.05)]),
            jid=str(d["jids"][j]) if "jids" in d.files else None,
            meta={"source": "3dfront"}))
    if len(objs) < 3:
        return None
    return Scene(scene_id=os.path.basename(os.path.dirname(box_p)), room=room,
                 objects=objs, source="3dfront")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physcene-scenes", default="outputs/physcene_scenes")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--bank", default="outputs/priors/assets_future.pkl")
    ap.add_argument("--room-type", default="living_room")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/compare_physcene.json")
    a = ap.parse_args()

    stats = json.load(open(os.path.join(a.cache, "dataset_stats.txt")))
    classes = stats["class_labels"]
    bank = AssetBank.load(a.bank) if os.path.exists(a.bank) else None

    by_suffix = {}
    for p in glob.glob(os.path.join(a.cache, "*", "boxes.npz")):
        by_suffix.setdefault(
            os.path.basename(os.path.dirname(p)).split("_")[-1], p)

    ps_files = sorted(glob.glob(os.path.join(a.physcene_scenes, "*.json")))
    if a.limit:
        ps_files = ps_files[:a.limit]

    # A reference of the *same* room is an information advantage PhyScene does
    # not have, so a second ReRoom arm takes its reference from a different
    # room of the same type -- which is also the actual use case: you liked
    # someone else's living room, not a photograph of your own.
    pool = sorted(by_suffix.values())
    rng = np.random.default_rng(0)

    rows = []
    for k, pf in enumerate(ps_files):
        ps = Scene.load(pf)
        room_key = ps.meta.get("room")
        box_p = by_suffix.get(room_key)
        if box_p is None:
            continue
        gt = scene_from_cache(box_p, classes, a.room_type)
        if gt is None:
            continue

        def add(name, sc):
            m = physcene_metrics(sc)
            m.update({"method": name, "room": room_key})
            rows.append(m)

        add("PhyScene", ps)
        add("3D-FRONT reference", gt)
        try:
            g = build_motifs(build_scene_graph(gt))
            add("ReRoom", retarget(g, gt.room.copy(), bank=bank,
                                   cfg=RetargetConfig(restarts=16)).scene)
            add("ReRoom (no reference)",
                run_baseline("target_only", g, gt.room.copy(),
                             cfg=RetargetConfig()))
            # foreign reference: a different room of the same type
            for _ in range(8):
                other_p = pool[int(rng.integers(0, len(pool)))]
                if other_p != box_p:
                    break
            other = scene_from_cache(other_p, classes, a.room_type)
            if other is not None:
                go = build_motifs(build_scene_graph(other))
                add("ReRoom (foreign reference)",
                    retarget(go, gt.room.copy(), bank=bank,
                             cfg=RetargetConfig(restarts=16)).scene)
        except Exception as exc:
            print(f"  {room_key}: {type(exc).__name__}: {exc}", flush=True)
        if k % 20 == 0:
            print(f"  {k}/{len(ps_files)}  rows={len(rows)}", flush=True)

    with open(a.out, "w") as fh:
        json.dump(rows, fh)

    keys = ["ps_Col_obj", "ps_Col_scene", "ps_R_out", "ps_R_walkable",
            "ps_R_reach", "ps_n_objects"]
    lab = ["Col_obj↓", "Col_scene↓", "R_out↓", "R_walk↑", "R_reach↑", "objects"]
    order = ["3D-FRONT reference", "PhyScene", "ReRoom",
             "ReRoom (foreign reference)", "ReRoom (no reference)"]
    print(f"\n{'method':24s}" + "".join(f"{x:>12s}" for x in lab) + f"{'n':>7s}")
    for name in order:
        sub = [r for r in rows if r["method"] == name]
        if not sub:
            continue
        v = [float(np.nanmean([r[k] for r in sub])) for k in keys]
        print(f"{name:24s}" + "".join(f"{x:12.3f}" for x in v)
              + f"{len(sub):7d}")
    print("\nSame rooms, same object vocabulary, one evaluator.")
    print("->", a.out)


if __name__ == "__main__":
    main()
