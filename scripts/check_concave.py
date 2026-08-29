#!/usr/bin/env python
"""Test the diagnosis before building on it.

The first claim tested here -- that containment via the nearest boundary sample's
inward halfplane is wrong at reflex corners -- was REFUTED by this script: the
halfplane reading matched shapely exactly on every probe, in the hall, beside the
reflex corner and throughout the cut-out. Signed distance to the nearest boundary
point is correct for any simple polygon; concavity does not break it, and the
only error is the 0.375 m sample spacing.

The script is kept because the question it settles is still live: whichever
predicate is used for containment has to be graded against shapely rather than
against itself, and the floor-node test has to clear the same bar before it can
replace anything.
"""
import os, sys
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import numpy as np
from shapely.geometry import Point

from reroom.core.scene import Room
from reroom.geom.polygon import as_polygon
from reroom.generative.tokens import boundary_samples, room_frame
from reroom.generative.floorgraph import floor_nodes

# L-shaped room: 6x6 with the top-right 3x3 quadrant removed.
L = np.array([[0., 0.], [6., 0.], [6., 3.], [3., 3.], [3., 6.], [0., 6.]])
room = Room(polygon=L, height=2.8, openings=[], room_type="living_room")
poly = as_polygon(room)
fr = room_frame(room)

bnd = boundary_samples(room, 64)
bp, bn = bnd[:, :2] * np.array([fr.half1, fr.half2]), bnd[:, 4:6]
# boundary_samples stores frame coords; rebuild metric points from the polygon
ring = poly.exterior
Lp = ring.length
bp = np.array([[ring.interpolate((k + .5) / 64 * Lp).x,
                ring.interpolate((k + .5) / 64 * Lp).y] for k in range(64)])
nb = []
for k in range(64):
    p = ring.interpolate((k + .5) / 64 * Lp); q = ring.interpolate(((k + .5) / 64 + 1e-3) * Lp)
    t = np.array([q.x - p.x, q.y - p.y]); t /= max(np.linalg.norm(t), 1e-9)
    n = np.array([-t[1], t[0]])
    if not poly.contains(Point(p.x + n[0] * 1e-3, p.y + n[1] * 1e-3)):
        n = -n
    nb.append(n)
nb = np.array(nb)

feat, adj, cover_r, fpts = floor_nodes(room, fr, m=48, robot=0.3)


def truth(p):
    pt = Point(p)
    return 0.0 if poly.contains(pt) else float(poly.exterior.distance(pt))


def halfplane(p):
    k = int(((p[None] - bp) ** 2).sum(-1).argmin())
    return float(max(-np.dot(p - bp[k], nb[k]), 0.0))


def floor(p):
    d = float(np.sqrt(((p[None] - fpts) ** 2).sum(-1)).min())
    return float(max(d - cover_r, 0.0))


probes = [
    ("deep inside (hall)",          np.array([1.5, 1.5])),
    ("inside, near reflex corner",  np.array([2.7, 2.7])),
    ("inside the lower arm",        np.array([5.0, 1.5])),
    ("CUT-OUT, just past corner",   np.array([3.6, 3.6])),
    ("CUT-OUT, middle",             np.array([4.5, 4.5])),
    ("CUT-OUT, deep",               np.array([5.5, 5.5])),
    ("outside past the left wall",  np.array([-1.0, 3.0])),
]
print(f"floor nodes: 48, covering radius {cover_r:.3f} m, "
      f"edges {int(adj.sum() / 2)}")
print(f"\n  {'probe':<30}{'truth':>9}{'halfplane':>12}{'floor-node':>12}")
for name, p in probes:
    t, h, f = truth(p), halfplane(p), floor(p)
    flag = "  <-- MISSES IT" if t > 0.3 and h < 0.1 else ""
    print(f"  {name:<30}{t:>9.3f}{h:>12.3f}{f:>12.3f}{flag}")
