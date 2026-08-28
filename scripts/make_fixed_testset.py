#!/usr/bin/env python
"""Freeze a fixed test set for quantitative tracking of the retargeting model.

Two case families, all drawn from the held-out TEST split so nothing here is
ever trained on:

  * cross  -- real (S_ref, S_tgt) pairs (same type, Jaccard>0.6, orientation-
    filtered).  S_tgt's real human layout is the ground truth, so we get a true
    position error in cm against a designer layout.  This is the headline
    benchmark.
  * three-sizes -- a reference uniformly scaled to 0.75x / 1.0x / 1.35x, the
    deployment scale sweep.  No human GT, but measures physical plausibility
    (wall float, snap, collisions, OOB) as the room grows / shrinks.

Only stable scene_ids (+ scalar scales) are stored, so the set is reproducible
regardless of RNG or code changes downstream.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import defaultdict
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.polygon import as_polygon
from reroom.generative.xscene import (build_pair_index_filtered, make_cross_pair_filtered,
    jaccard, catset)

CORPUS = "/home/gino/data/reroom/processed"
OUT = "outputs/fixed_testset.json"
N_CROSS_PER_TYPE = 6          # 12 cross pairs total
N_FWD_PER_TYPE = 2            # 4 scenes x 3 sizes = 12 forward sub-cases
SCALES = [0.75, 1.0, 1.35]


def main():
    scenes = [s for s in iter_scenes(CORPUS, min_objects=6)
              if s.room.room_type in ("bedroom", "living_room")]
    _, _, te = split_scenes(scenes)
    by_id = {s.scene_id: s for s in te}
    idx = build_pair_index_filtered(te, thresh=0.6, max_deg=30.0,
                                    max_partners=16, seed=0)
    areas = {i: as_polygon(te[i].room).area for i in range(len(te))}

    # ---- cross pairs: pick diverse (Jaccard, area-ratio) per room type -------
    cross = []
    for rt in ("living_room", "bedroom"):
        cand = []
        for a in sorted(idx.keys(), key=lambda i: te[i].scene_id):  # stable order
            if te[a].room.room_type != rt:
                continue
            for b in idx[a]:
                trip = make_cross_pair_filtered(te[a], te[b])
                if trip is None:
                    continue
                _, troom, gt = trip
                n_gt = sum(1 for o in gt.objects if o.keep)
                if n_gt < 4:
                    continue
                r = areas[b] / max(areas[a], 1e-6)
                jac = jaccard(catset(te[a]), catset(te[b]))
                cand.append((a, b, r, jac, n_gt))
                break                                   # one partner per ref
        # bucket by area-ratio (shrink / same / expand) and spread the picks
        buckets = defaultdict(list)
        for c in cand:
            key = "shrink" if c[2] < 0.9 else ("expand" if c[2] > 1.1 else "same")
            buckets[key].append(c)
        picks = []
        order = ["shrink", "same", "expand"]
        bi = 0
        while len(picks) < N_CROSS_PER_TYPE and any(buckets[k] for k in order):
            k = order[bi % 3]; bi += 1
            if buckets[k]:
                picks.append(buckets[k].pop(0))
        for a, b, r, jac, n_gt in picks[:N_CROSS_PER_TYPE]:
            cross.append({"type": "cross", "room_type": rt,
                          "ref_id": te[a].scene_id, "tgt_id": te[b].scene_id,
                          "area_ratio": round(float(r), 3),
                          "jaccard": round(float(jac), 3), "n_gt": int(n_gt)})

    # ---- three-sizes: pick mid-sized scenes with clear motifs ----------------
    fwd = []
    for rt in ("living_room", "bedroom"):
        cand = [s for s in te if s.room.room_type == rt
                and 12.0 <= as_polygon(s.room).area <= 28.0
                and sum(1 for o in s.objects if o.keep) >= 6]
        cand.sort(key=lambda s: s.scene_id)
        step = max(1, len(cand) // N_FWD_PER_TYPE)
        for k in range(N_FWD_PER_TYPE):
            s = cand[k * step]
            fwd.append({"type": "three_sizes", "room_type": rt,
                        "ref_id": s.scene_id, "scales": SCALES})

    payload = {"corpus": CORPUS, "split": "test",
               "n_cross": len(cross), "n_three_sizes_scenes": len(fwd),
               "scales": SCALES, "cases": cross + fwd}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
    # sanity: every id resolvable
    for c in payload["cases"]:
        assert c["ref_id"] in by_id, c["ref_id"]
        if c["type"] == "cross":
            assert c["tgt_id"] in by_id, c["tgt_id"]
    print(f"wrote {OUT}: {len(cross)} cross pairs + {len(fwd)} three-sizes scenes")
    for c in cross:
        print(f"  cross {c['room_type']:12s} ratio={c['area_ratio']:.2f} "
              f"jac={c['jaccard']:.2f} n_gt={c['n_gt']}")
    for c in fwd:
        print(f"  3sz   {c['room_type']:12s} {c['ref_id'][:24]}")


if __name__ == "__main__":
    main()
