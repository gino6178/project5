#!/usr/bin/env python
"""Fit relation elasticity on a real scene corpus (plan sections 4 and 19).

Both estimators of ``reroom.intent.elasticity`` are fitted and reported:

* ``StatElasticity``   -- ``d log d / d log gamma`` per (category pair, relation)
  bucket, a closed-form log-log regression;
* ``NeuralElasticity`` -- ``f_psi`` of eq. (45), trained by distance regression
  and read out by autograd sensitivity.

The headline check is whether the fitted values reproduce the plan's
qualitative claim: chair-to-table rigid, sofa-to-TV elastic.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from reroom.core.scene import scene_from_dict
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.intent.elasticity import (NeuralElasticity, PriorElasticity,
                                      StatElasticity, collect_elasticity_samples)

PROBES = [
    ("dining_table", "dining_chair", "near"),
    ("dining_table", "dining_chair", "surrounds"),
    ("double_bed", "nightstand", "near"),
    ("desk", "office_chair", "near"),
    ("sofa", "coffee_table", "facing"),
    ("sofa", "tv_stand", "face_to_face"),
    ("double_bed", "wardrobe", "face_to_face"),
    ("sofa", "armchair", "facing"),
    ("double_bed", "wardrobe", "facing"),
    ("tv_stand", "tv", "support"),
]


def _work(chunk):
    scenes = [scene_from_dict(d) for d in chunk]
    return collect_elasticity_samples(scenes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="outputs/elasticity")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--room-types", default="")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--anchor-weight", type=float, default=2.0)
    ap.add_argument("--cache", default=None,
                    help="pickle of collected samples, to skip graph building")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rts = tuple(x for x in a.room_types.split(",") if x) or None
    scenes = list(iter_scenes(a.corpus, room_types=rts, limit=a.limit or None,
                              min_objects=4))
    train, val, test = split_scenes(scenes)
    print(f"{len(scenes)} scenes -> train {len(train)} val {len(val)} test {len(test)}",
          flush=True)

    import pickle
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, "rb") as fh:
            samples = pickle.load(fh)
        print(f"loaded {len(samples)} cached samples", flush=True)
        _fit_and_report(a, samples)
        return

    dicts = [s.to_dict() for s in train]
    chunks = [dicts[i:i + 64] for i in range(0, len(dicts), 64)]
    samples = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, part in enumerate(ex.map(_work, chunks)):
            samples.extend(part)
            if k % 20 == 0:
                print(f"  graphs {k * 64}/{len(dicts)}  samples={len(samples)}",
                      flush=True)
    print(f"{len(samples)} relation samples", flush=True)
    if a.cache:
        with open(a.cache, "wb") as fh:
            pickle.dump(samples, fh)
    _fit_and_report(a, samples)


def _fit_and_report(a, samples):
    stat = StatElasticity().fit(samples)
    stat.save(os.path.join(a.out, "stat.json"))

    neural = NeuralElasticity(device=a.device)
    neural.fit(samples, epochs=a.epochs, verbose=True, anchor=stat,
               anchor_weight=a.anchor_weight)
    neural.save(os.path.join(a.out, "neural.pt"))

    prior = PriorElasticity()
    from reroom.intent.elasticity import RelationContext
    rows = []
    for ci, cj, kind in PROBES:
        ctx = RelationContext(cat_i=ci, cat_j=cj, kind=kind, d_ref=1.5,
                              gamma_src_abs=4.0)
        rows.append({"pair": f"{ci}--{cj}", "kind": kind,
                     "prior": prior.alpha(ctx), "stat": stat.alpha(ctx),
                     "neural": float(neural.alpha(ctx))})
    print(f"\n{'pair':38s}{'kind':14s}{'prior':>8s}{'stat':>8s}{'neural':>8s}")
    for r in rows:
        print(f"{r['pair']:38s}{r['kind']:14s}{r['prior']:8.3f}"
              f"{r['stat']:8.3f}{r['neural']:8.3f}")

    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump({"n_samples": len(samples), "probes": rows,
                   "kind_alpha": {k: list(v) for k, v in stat.kind_alpha.items()},
                   "top_pairs": stat.report(40)}, fh, indent=1)
    print("\nper-kind fitted alpha:")
    for k, (al, n, r2) in sorted(stat.kind_alpha.items()):
        print(f"  {k:16s} alpha={al:.3f}  n={n:6d}  r2={r2:.3f}")


if __name__ == "__main__":
    main()
