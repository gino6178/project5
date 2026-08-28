#!/usr/bin/env python
"""Extract per-model bounding boxes from an extracted 3D-FUTURE-model tree.

3D-FRONT stores only a jid and a scale, so an object's real size has to come
from the asset mesh.  Only the ``v`` lines of ``raw_model.obj`` are needed, so
this reads them directly instead of loading full meshes -- ~16 k models in a
couple of minutes on all cores.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def model_bbox(args):
    mid, path = args
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    try:
        with open(path, "rb") as fh:
            for line in fh:
                if not line.startswith(b"v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue
                lo = np.minimum(lo, v)
                hi = np.maximum(hi, v)
    except OSError:
        return mid, None
    if not np.all(np.isfinite(lo)):
        return mid, None
    return mid, np.concatenate([lo, hi]).tolist()


def main(roots: list[str], out_path: str) -> None:
    jobs = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for mid in os.listdir(root):
            obj = os.path.join(root, mid, "raw_model.obj")
            if os.path.exists(obj):
                jobs.append((mid, obj))
    print(f"{len(jobs)} models", flush=True)
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            existing = json.load(fh)
        jobs = [j for j in jobs if j[0] not in existing]
        print(f"{len(jobs)} new", flush=True)
    out = dict(existing)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for k, (mid, bb) in enumerate(ex.map(model_bbox, jobs, chunksize=16)):
            if bb is not None:
                out[mid] = bb
            if k % 2000 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {len(out)} bboxes -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
