"""PhyScene's physical-plausibility metrics, reimplemented (plan ref. [11]).

The plan's bibliography names PhyScene as one of the floor-plan-conditioned
scene synthesis systems this work sits beside, and section 16.1 asks for a
"target-only generation" baseline drawn from that family.  PhyScene reports its
physical-plausibility numbers on 3D-FRONT with metric definitions that overlap
ReRoom's -- but they are *not* the same quantities:

    ``R_out``       PhyScene: the fraction of *objects* with any footprint
                    pixel outside the floor plan.
                    ReRoom  : the fraction of furniture *area* outside it.
    ``Col_obj``     PhyScene: the fraction of objects whose 3D box overlaps
                    any other object at all.
                    ReRoom  : the fraction of furniture area in overlap.

A binary per-object rate and an area fraction differ by an order of magnitude
on the same scene, so putting ReRoom's own numbers next to PhyScene's published
table would be meaningless.  This module computes PhyScene's definitions --
following their released `utils/overlap.py` and `scripts/eval/walkable_metric.py`
rather than the paper's prose -- so ReRoom can at least be read on their
yardstick.  What that still is *not* is a head-to-head: PhyScene generates its
object set from a learned model while ReRoom transfers a reference's, their
floor plans are 3D-FRONT rooms as-is while ReRoom's are deliberately deformed,
and the splits differ.  Those caveats belong with any number this produces.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon

from ..core.scene import Scene
from ..geom.polygon import as_polygon, object_polygon

__all__ = ["physcene_metrics", "PHYSCENE_IMAGE_SIZE", "PHYSCENE_ROBOT_WIDTH"]

PHYSCENE_IMAGE_SIZE = 256          # their raster resolution
PHYSCENE_ROBOT_WIDTH = 0.3         # metres, their `robot_real_width`
PHYSCENE_HEIGHT_CUTOFF = 1.5       # objects above this do not block the floor


def _raster(scene: Scene, image_size: int = PHYSCENE_IMAGE_SIZE):
    """Floor mask and per-object footprint masks on their 256x256 grid."""
    poly = as_polygon(scene.room)
    cx, cy = poly.centroid.x, poly.centroid.y
    ring = np.asarray(poly.exterior.coords) - np.array([cx, cy])
    scale = float(np.abs(ring).max()) + 0.2

    def to_px(pts):
        p = (np.asarray(pts) - np.array([cx, cy])) / scale * image_size / 2
        return p + image_size / 2

    yy, xx = np.mgrid[0:image_size, 0:image_size]
    px = np.column_stack([(xx.ravel() - image_size / 2) * scale
                          / (image_size / 2) + cx,
                          (yy.ravel() - image_size / 2) * scale
                          / (image_size / 2) + cy])

    from matplotlib.path import Path
    floor = Path(np.asarray(poly.exterior.coords)).contains_points(px)
    for ring_i in poly.interiors:
        floor &= ~Path(np.asarray(ring_i.coords)).contains_points(px)
    floor = floor.reshape(image_size, image_size)

    masks = []
    for o in scene.objects:
        if not o.keep:
            continue
        fp = np.asarray(object_polygon(o).exterior.coords)
        m = Path(to_px(fp)).contains_points(
            np.column_stack([xx.ravel(), yy.ravel()])).reshape(image_size,
                                                               image_size)
        masks.append((o, m))
    return floor, masks, scale


def _iou3d(a, b) -> float:
    """Oriented 3D box IoU, the quantity PhyScene's `cal_iou_3d` returns."""
    pa, pb = object_polygon(a), object_polygon(b)
    inter2d = pa.intersection(pb).area
    if inter2d <= 0:
        return 0.0
    lo = max(a.z, b.z)
    hi = min(a.top, b.top)
    if hi <= lo:
        return 0.0
    inter = inter2d * (hi - lo)
    va = pa.area * max(a.top - a.z, 1e-6)
    vb = pb.area * max(b.top - b.z, 1e-6)
    return float(inter / max(va + vb - inter, 1e-9))


def physcene_metrics(scene: Scene, robot_width: float = PHYSCENE_ROBOT_WIDTH,
                     image_size: int = PHYSCENE_IMAGE_SIZE) -> dict:
    """``Col_obj``, ``R_out``, ``R_walkable`` and ``R_reach`` as PhyScene defines
    them, plus the per-scene collision flag their ``Col_scene`` averages."""
    kept = [o for o in scene.objects if o.keep]
    n = len(kept)
    if n == 0:
        return {"ps_Col_obj": float("nan"), "ps_Col_scene": float("nan"),
                "ps_R_out": float("nan"), "ps_R_walkable": float("nan"),
                "ps_R_reach": float("nan"), "ps_n_objects": 0}

    # ---- Col_obj: an object counts once if it overlaps anything at all ----
    hit = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            if _iou3d(kept[i], kept[j]) > 0:
                hit[i] = hit[j] = True
    col_obj = float(sum(hit) / n)

    floor, masks, scale = _raster(scene, image_size)
    rw = max(int(robot_width / scale * image_size / 2), 1)

    # ---- R_out: any footprint pixel outside the floor mask ----
    out = sum(1 for _, m in masks if (m & ~floor).any())
    r_out = float(out / n)

    # ---- R_walkable: largest connected free component / all free floor ----
    # Their raster does two things this has to match or the number drifts: the
    # floor is eroded by the robot width, and each box is *stroked* with a
    # contour of that same width before being filled, so an object blocks its
    # own footprint plus half a robot on every side.  A plain 3x3 dilation
    # instead reported 0.98 where their script reported 0.83 on the same run.
    st = np.ones((max(rw, 1), max(rw, 1)), dtype=bool)
    free = ndimage.binary_erosion(floor, structure=st)
    stroke = np.ones((max(rw, 1), max(rw, 1)), dtype=bool)
    blockers = np.zeros_like(free)
    for o, m in masks:
        if o.z < PHYSCENE_HEIGHT_CUTOFF:
            blockers |= ndimage.binary_dilation(m, structure=stroke)
    free = free & ~blockers
    lab, k = ndimage.label(free)
    tot = int(free.sum())
    if tot == 0 or k == 0:
        r_walk, biggest = 0.0, np.zeros_like(free)
        big = 0
    else:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        big = int(counts.argmax())
        biggest = lab == big
        r_walk = float(counts[big] / tot)

    # ---- R_reach: an object is reachable if its dilated mask touches it ----
    if not biggest.any():
        r_reach = 0.0
    else:
        near = ndimage.binary_dilation(biggest, structure=np.ones((3, 3),
                                                                  dtype=bool),
                                       iterations=max(rw, 1))
        reach = sum(1 for _, m in masks if (m & near).any())
        r_reach = float(reach / n)

    return {"ps_Col_obj": col_obj, "ps_Col_scene": float(any(hit)),
            "ps_R_out": r_out, "ps_R_walkable": r_walk,
            "ps_R_reach": r_reach, "ps_n_objects": n}
