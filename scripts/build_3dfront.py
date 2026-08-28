#!/usr/bin/env python
"""Parse the whole 3D-FRONT release into ReRoom scenes, in parallel.

Writes one gzipped JSON-lines shard per worker plus an index, so the corpus can
be streamed without holding 16 k scenes in memory.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from reroom.data.threed_front import load_bboxes, parse_scene_file

_BB: dict = {}
_CAT: dict = {}


def _init(bbox_path: str, cat_path: str | None):
    global _BB, _CAT
    _BB = load_bboxes(bbox_path)
    _CAT = json.load(open(cat_path)) if cat_path and os.path.exists(cat_path) else {}


def _work(path: str):
    try:
        scenes = parse_scene_file(path, _BB, _CAT)
    except Exception as exc:                       # keep one bad house local
        return path, [], f"{type(exc).__name__}: {exc}"
    return path, [s.to_dict() for s in scenes], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", required=True, help="dir of 3D-FRONT *.json")
    ap.add_argument("--bboxes", required=True)
    ap.add_argument("--categories", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.front, "*.json")))
    if a.limit:
        files = files[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(files)} houses -> {a.out}", flush=True)

    n_scenes = 0
    types = Counter()
    errors = []
    shard_i = 0
    fh = gzip.open(os.path.join(a.out, f"scenes_{shard_i:03d}.jsonl.gz"), "wt")
    written = 0
    with ProcessPoolExecutor(a.workers, initializer=_init,
                             initargs=(a.bboxes, a.categories)) as ex:
        for k, (path, scenes, err) in enumerate(
                ex.map(_work, files, chunksize=4)):
            if err:
                errors.append((os.path.basename(path), err))
                continue
            for s in scenes:
                fh.write(json.dumps(s) + "\n")
                types[s["room"]["room_type"]] += 1
                n_scenes += 1
                written += 1
            if written >= 2000:
                fh.close()
                shard_i += 1
                fh = gzip.open(os.path.join(a.out, f"scenes_{shard_i:03d}.jsonl.gz"), "wt")
                written = 0
            if k % 250 == 0:
                print(f"  {k}/{len(files)}  scenes={n_scenes}", flush=True)
    fh.close()
    meta = {"n_houses": len(files), "n_scenes": n_scenes,
            "room_types": dict(types), "n_errors": len(errors),
            "errors": errors[:20], "shards": shard_i + 1}
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1)[:800])


if __name__ == "__main__":
    main()
