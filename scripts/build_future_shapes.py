#!/usr/bin/env python
"""Compute ``f^geo`` (eq. 10) for every 3D-FUTURE asset.

The box in ``future_bboxes.json`` says how big a model is; this says what shape
fills that box, so retrieval can prefer a candidate that actually looks like
the reference object rather than merely measuring the same.  Only the ``v`` and
``f`` lines of ``raw_model.obj`` are needed, which is fast enough to sweep the
whole ~16 k-model tree on all cores.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reroom.perception.geometry import SHAPE_DIM, descriptor_from_mesh  # noqa: E402


def _read_obj(path: str):
    verts, faces = [], []
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b"v "):
                p = line.split()
                if len(p) >= 4:
                    try:
                        verts.append((float(p[1]), float(p[2]), float(p[3])))
                    except ValueError:
                        pass
            elif line.startswith(b"f "):
                p = line.split()[1:]
                if len(p) < 3:
                    continue
                try:
                    v = [int(t.split(b"/")[0]) - 1 for t in p]
                except ValueError:
                    continue
                for k in range(1, len(v) - 1):    # fan-triangulate n-gons
                    faces.append((v[0], v[k], v[k + 1]))
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def shape_of(args):
    mid, path = args
    try:
        v, f = _read_obj(path)
        if len(v) == 0:
            return mid, None
        if len(f):
            f = f[(f >= 0).all(1) & (f < len(v)).all(1)]
        return mid, descriptor_from_mesh(v, f if len(f) else None, seed=0)
    except Exception:
        return mid, None


def main(roots: list[str], out_path: str) -> None:
    jobs = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for mid in sorted(os.listdir(root)):
            obj = os.path.join(root, mid, "raw_model.obj")
            if os.path.exists(obj):
                jobs.append((mid, obj))
    print(f"{len(jobs)} models", flush=True)
    out: dict[str, np.ndarray] = {}
    if os.path.exists(out_path):
        with np.load(out_path) as z:
            out = {k: z[k] for k in z.files}
        jobs = [j for j in jobs if j[0] not in out]
        print(f"{len(jobs)} new", flush=True)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for n, (mid, d) in enumerate(ex.map(shape_of, jobs, chunksize=8)):
            if d is not None and d.shape[0] == SHAPE_DIM:
                out[mid] = d.astype(np.float32)
            if n % 1000 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    np.savez_compressed(out_path, **out)
    print(f"-> {out_path}  ({len(out)} descriptors, dim {SHAPE_DIM})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: build_future_shapes.py OUT.npz ROOT [ROOT...]")
    main(sys.argv[2:], sys.argv[1])
