"""``f^geo``: the per-node geometry feature of eq. (10).

The node built in Stage I carries a semantic embedding, a box, an appearance
feature and a geometry feature.  The box already says how *large* an object is;
what it cannot say is what *shape* fills that box -- a wardrobe and a bookcase
of identical extents differ entirely in where their mass sits, and an L-shaped
sofa and a three-seater differ in footprint occupancy but not in bounding box.

``f^geo`` is therefore a canonical occupancy descriptor: points are sampled on
the surface, mapped into the object's own frame (yaw undone, centred, divided
by the half-extents so the box becomes the unit cube) and histogrammed on a
coarse grid.  It is scale-free and rotation-canonical by construction, so it
composes with -- rather than duplicates -- the size term ``D_s`` and the
appearance term ``D_f`` of eq. (30).

Its consumer is retrieval: with a real mesh for the reference object (MIDI
reconstructs one; 3D-FUTURE ships one) the bank can prefer the candidate that
is the same *shape*, not merely the same size and style.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["SHAPE_GRID", "SHAPE_DIM", "shape_descriptor", "shape_distance",
           "descriptor_from_mesh"]

SHAPE_GRID = (3, 3, 4)                     # x, y, z cells of the unit box
SHAPE_DIM = SHAPE_GRID[0] * SHAPE_GRID[1] * SHAPE_GRID[2] + 3


def _sample_surface(v: np.ndarray, f: np.ndarray | None,
                    n: int, rng: np.random.Generator) -> np.ndarray:
    """Area-weighted surface samples, so tessellation density does not vote."""
    if f is None or len(f) == 0:
        return v
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tot = float(area.sum())
    if not np.isfinite(tot) or tot <= 0:
        return v
    pick = rng.choice(len(f), size=n, p=area / tot)
    u1 = rng.random((n, 1))
    u2 = rng.random((n, 1))
    root = np.sqrt(u1)
    return ((1 - root) * a[pick] + root * (1 - u2) * b[pick]
            + root * u2 * c[pick])


def shape_descriptor(points: np.ndarray, yaw: float = 0.0,
                     centre: np.ndarray | None = None,
                     size: np.ndarray | None = None) -> np.ndarray:
    """Canonical occupancy descriptor of a point cloud, eq. (10) ``f^geo``.

    ``yaw``/``centre``/``size`` describe the box the points live in; when they
    are omitted the object's own axis-aligned bounds are used, which is the
    right thing for an asset mesh that is already in its canonical frame.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(p) == 0:
        return np.zeros(SHAPE_DIM, dtype=np.float32)
    if abs(yaw) > 1e-9:
        c, s = math.cos(-yaw), math.sin(-yaw)
        if centre is not None:
            p = p - np.asarray(centre, dtype=np.float64)
        p = np.stack([c * p[:, 0] - s * p[:, 1],
                      s * p[:, 0] + c * p[:, 1], p[:, 2]], axis=1)
    elif centre is not None:
        p = p - np.asarray(centre, dtype=np.float64)

    lo, hi = p.min(0), p.max(0)
    if size is None:
        ext = np.maximum(hi - lo, 1e-6)
        q = (p - lo) / ext
    else:
        half = np.maximum(np.asarray(size, dtype=np.float64) * 0.5, 1e-6)
        q = (p / half + 1.0) * 0.5
        ext = np.maximum(hi - lo, 1e-6)
    q = np.clip(q, 0.0, 1.0 - 1e-9)

    nx, ny, nz = SHAPE_GRID
    idx = ((q[:, 2] * nz).astype(np.int64) * nx * ny
           + (q[:, 1] * ny).astype(np.int64) * nx
           + (q[:, 0] * nx).astype(np.int64))
    hist = np.bincount(idx, minlength=nx * ny * nz).astype(np.float64)
    hist /= max(hist.sum(), 1.0)

    # three shape scalars the histogram alone reads only weakly
    order = np.sort(ext)[::-1]
    elong = float(order[0] / max(order[1], 1e-6))            # how stretched
    flat = float(order[2] / max(order[0], 1e-6))             # how slab-like
    lower = float(hist[:nx * ny * (nz // 2)].sum())          # top- or bottom-heavy
    extra = np.array([math.log(min(elong, 12.0)), flat, lower])
    return np.concatenate([hist, extra]).astype(np.float32)


def descriptor_from_mesh(vertices, faces=None, n: int = 4096,
                         seed: int = 0, **kw) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = None if faces is None else np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return shape_descriptor(_sample_surface(v, f, n, rng), **kw)


def shape_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """``D_g`` in [0, 1]: half the L1 distance on the histogram part, plus a
    small scalar term.  Returns 0 when either side has no descriptor, so a
    missing ``f^geo`` never penalises a candidate."""
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape or a.shape[0] != SHAPE_DIM:
        return 0.0
    n = SHAPE_DIM - 3
    hist = 0.5 * float(np.abs(a[:n] - b[:n]).sum())          # total variation
    sc = float(np.abs(a[n:] - b[n:]).mean())
    return float(np.clip(0.8 * hist + 0.2 * min(sc, 1.0), 0.0, 1.0))
