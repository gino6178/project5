"""Free space as nodes of the design graph.

The object graph says how furniture relates to furniture. It says nothing about
where a robot can stand, and nothing about the SHAPE of the room beyond a ring of
boundary samples -- which is why containment was being approximated by the
inward halfplane of the nearest boundary sample. That approximation is fine at a
convex corner and WRONG at a reflex one: near a concave corner the interior is
locally the union of two halfplanes, not their intersection, so the nearest
sample's normal calls part of the outside "inside" and part of the inside
"outside". It is exactly where run 6 kept losing -- the corridor (purely convex)
improved to 0.184 while the L and T shapes did not move.

Sampling free space as nodes removes the approximation rather than patching it.
Nodes exist only where a robot can actually stand, so:

* **containment becomes exact** -- an object corner is inside iff it is covered
  by the floor sampling, which holds at reflex corners, convex corners and
  everything else, and whose gradient pulls a stray corner AROUND a concave
  corner instead of through the wall;
* **connectivity becomes topological** -- edges join nodes that can see each
  other inside the polygon, so the throat of an L-shaped room is a real
  bottleneck in the graph. Flood filling this graph replaces the rasterised
  version, which produced two bugs of its own (a dilated kernel stepping over
  walls, and a soft gate attenuating with distance) and needed cells finer than
  the robot half-width to be faithful.

The construction is geometric, not semantic: even coverage of the interior and
visibility edges. Which node is a doorway and which is a dead end is left to the
transformer to learn from the node features, exactly as the pair spacing is.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point

from ..core.scene import Room
from ..geom.polygon import as_polygon, erode

__all__ = ["N_FLOOR", "FLOOR_DIM", "floor_nodes"]

N_FLOOR = 48           # nodes per room
FLOOR_DIM = 6          # see floor_nodes


def _even_sample(region, m: int, step: float) -> np.ndarray:
    """Farthest-point subsample of a dense interior grid.

    Rejection sampling clusters and leaves gaps, which for a corridor or the
    throat of an L means the bottleneck may get no node at all. Farthest-point
    selection over a grid gives even coverage at any room shape.
    """
    minx, miny, maxx, maxy = region.bounds
    gx = np.arange(minx + step * 0.5, maxx, step)
    gy = np.arange(miny + step * 0.5, maxy, step)
    if gx.size == 0: gx = np.array([(minx + maxx) * 0.5])
    if gy.size == 0: gy = np.array([(miny + maxy) * 0.5])
    cand = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    keep = np.array([region.contains(Point(p[0], p[1])) for p in cand], dtype=bool)
    cand = cand[keep]
    if len(cand) == 0:
        return np.asarray(region.representative_point().coords[0])[None].repeat(m, 0)
    if len(cand) <= m:
        pad = cand[np.zeros(m - len(cand), dtype=int)] if m > len(cand) else cand[:0]
        return np.concatenate([cand, pad], 0)
    sel = [int(np.argmin(((cand - cand.mean(0)) ** 2).sum(-1)))]   # start at the centre
    d = ((cand - cand[sel[0]]) ** 2).sum(-1)
    for _ in range(m - 1):
        k = int(np.argmax(d))
        sel.append(k)
        d = np.minimum(d, ((cand - cand[k]) ** 2).sum(-1))
    return cand[np.asarray(sel)]


def floor_nodes(room: Room, frame, m: int = N_FLOOR, robot: float = 0.3):
    """Returns ``(feat, adj, cover_r, pts)``.

    ``feat`` (m, FLOOR_DIM): normalised (u, v), metric (u*h1, v*h2), clearance to
    the wall in metres, and the node's degree fraction -- a local free-width
    proxy that lets the model tell a throat from an open floor without being told
    which is which.

    ``adj`` (m, m): 1 where the straight segment between two nodes stays inside
    the room and they are within linking range. This is what makes the throat of
    an L-shaped room a bottleneck in the graph rather than a short euclidean hop.

    ``cover_r``: the covering radius, i.e. how far any interior point can be from
    the nearest node. Containment is charged only beyond this, so the sampling
    density never masquerades as a violation.

    ``pts`` (m, 2): the same nodes in world metres. Returned explicitly because
    rebuilding them from the normalised pair needs the frame axes, and doing that
    by hand is an easy place to drop the rotation -- which is exactly the bug
    that made the first version of check_concave.py report nonsense.
    """
    poly = as_polygon(room)
    stand = erode(poly, robot * 0.5)
    if stand.is_empty:
        stand = poly
    area = max(float(stand.area), 1e-6)
    step = max(np.sqrt(area / max(m, 1)) * 0.5, 0.05)
    pts = _even_sample(stand, m, step)

    # covering radius: worst interior point-to-node distance over the same grid
    minx, miny, maxx, maxy = stand.bounds
    gx = np.arange(minx, maxx + step, step); gy = np.arange(miny, maxy + step, step)
    grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    inside = np.array([stand.contains(Point(p[0], p[1])) for p in grid], dtype=bool)
    if inside.any():
        dd = np.sqrt(((grid[inside][:, None, :] - pts[None]) ** 2).sum(-1)).min(-1)
        cover_r = float(dd.max())
    else:
        cover_r = float(step)

    # visibility adjacency
    d = np.sqrt(((pts[:, None, :] - pts[None]) ** 2).sum(-1))
    link = max(cover_r * 2.5, step * 2.5)
    adj = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            if d[i, j] <= link and LineString([pts[i], pts[j]]).within(poly):
                adj[i, j] = adj[j, i] = 1.0

    clear = np.array([poly.exterior.distance(Point(p[0], p[1])) for p in pts],
                     dtype=np.float32)
    deg = adj.sum(1) / max(float(adj.sum(1).max()), 1.0)

    rel = pts - frame.centre
    u = (rel @ frame.axis1) / max(float(frame.half1), 1e-6)
    v = (rel @ frame.axis2) / max(float(frame.half2), 1e-6)
    feat = np.stack([u, v, u * frame.half1, v * frame.half2, clear, deg],
                    axis=-1).astype(np.float32)
    return feat, adj, np.float32(cover_r), pts.astype(np.float32)
