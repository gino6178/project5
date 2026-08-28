"""Free-space analysis: navigability, connectivity, door blockage, clearance.

Section 15.1 of the plan asks for minimum clearance violation, door/window
blockage, free-space connectivity and a reachable-area ratio.  All four are
computed from one rasterised occupancy grid so they stay mutually consistent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from ..core.categories import prior
from ..core.scene import ObjectInstance, Opening, Room, Scene
from .polygon import as_polygon, object_polygon

__all__ = [
    "FreeSpace", "build_freespace", "AGENT_RADIUS", "door_seeds",
    "clearance_polygon", "clearance_violation", "door_blockage",
]

AGENT_RADIUS = 0.25          # half-shoulder width of a person, metres
STAND_HEIGHT = 0.55          # objects lower than this can be stepped over? no --
                             # but they do not block *visual* passage; we keep
                             # them blocking, and only exempt ceiling mounts.
CEILING_MOUNT_MIN_Z = 1.9    # pendant lamps etc. do not block the floor


def door_seeds(room: Room, depth: float = 0.45) -> list[np.ndarray]:
    """A point just inside the room for every door.

    An opening's stored normal has an arbitrary sign, so the side that is
    actually inside the room has to be chosen explicitly -- seeding the flood
    fill outside the polygon silently reports a near-zero reachable area.
    """
    poly = as_polygon(room)
    out = []
    for op in room.openings:
        if op.kind != "door":
            continue
        c = op.centre
        for sgn in (1.0, -1.0):
            p = c + op.normal * depth * sgn
            if poly.contains(Point(float(p[0]), float(p[1]))):
                out.append(p)
                break
    return out


def _blocks_floor(o: ObjectInstance) -> bool:
    if not o.keep:
        return False
    if o.z >= CEILING_MOUNT_MIN_Z:
        return False
    if o.category in ("rug", "wall_art"):
        return False
    return True


@dataclass
class FreeSpace:
    """A rasterised view of what part of the floor a person can stand on."""

    origin: np.ndarray          # (2,) world coords of cell (0, 0) centre
    res: float                  # metres per cell
    inside: np.ndarray          # bool (H, W) -- inside the room polygon
    free: np.ndarray            # bool (H, W) -- inside and not occupied
    walkable: np.ndarray        # bool (H, W) -- free after agent erosion

    @property
    def cell_area(self) -> float:
        return self.res * self.res

    def world_to_ij(self, p) -> tuple[int, int]:
        d = (np.asarray(p, dtype=float)[:2] - self.origin) / self.res
        return int(round(d[1])), int(round(d[0]))

    def components(self) -> tuple[np.ndarray, int]:
        lab, n = ndimage.label(self.walkable, structure=np.ones((3, 3)))
        return lab, int(n)

    def largest_component_ratio(self) -> float:
        lab, n = self.components()
        if n == 0:
            return 0.0
        sizes = ndimage.sum(self.walkable, lab, index=range(1, n + 1))
        return float(sizes.max() / max(self.walkable.sum(), 1))

    def n_components(self) -> int:
        return self.components()[1]

    def free_ratio(self) -> float:
        tot = int(self.inside.sum())
        return float(self.free.sum() / tot) if tot else 0.0

    def walkable_ratio(self) -> float:
        tot = int(self.inside.sum())
        return float(self.walkable.sum() / tot) if tot else 0.0

    def reachable_ratio(self, seeds: list[np.ndarray] | None = None) -> float:
        """Fraction of walkable floor reachable from the doors.

        With no door, the largest connected component is used as the seed --
        that is the standard "is the room one usable space?" question.
        """
        lab, n = self.components()
        if n == 0:
            return 0.0
        total = float(self.walkable.sum())
        if not seeds:
            sizes = ndimage.sum(self.walkable, lab, index=range(1, n + 1))
            return float(sizes.max() / total)
        hit = set()
        H, W = lab.shape
        for s in seeds:
            i, j = self.world_to_ij(s)
            found = False
            for rad in range(0, 12):
                for di in range(-rad, rad + 1):
                    for dj in range(-rad, rad + 1):
                        ii, jj = i + di, j + dj
                        if 0 <= ii < H and 0 <= jj < W and lab[ii, jj] > 0:
                            hit.add(int(lab[ii, jj]))
                            found = True
                if found:
                    break
        if not hit:
            return 0.0
        mask = np.isin(lab, list(hit))
        return float(mask.sum() / total)


def build_freespace(scene: Scene, res: float = 0.05,
                    agent_radius: float = AGENT_RADIUS) -> FreeSpace:
    room_poly = as_polygon(scene.room)
    minx, miny, maxx, maxy = room_poly.bounds
    pad = 2 * res
    minx, miny = minx - pad, miny - pad
    W = max(int(math.ceil((maxx - minx + pad) / res)), 4)
    H = max(int(math.ceil((maxy - miny + pad) / res)), 4)
    origin = np.array([minx + res / 2, miny + res / 2])
    xs = origin[0] + np.arange(W) * res
    ys = origin[1] + np.arange(H) * res
    gx, gy = np.meshgrid(xs, ys)

    inside = _rasterize(room_poly, gx, gy)
    occ = np.zeros_like(inside)
    blockers = [o for o in scene.objects if _blocks_floor(o)]
    if blockers:
        occ = _rasterize(unary_union([object_polygon(o) for o in blockers]), gx, gy)
    free = inside & (~occ)
    r_cells = int(round(agent_radius / res))
    if r_cells > 0:
        st = _disk(r_cells)
        walkable = ndimage.binary_erosion(free, structure=st, border_value=0)
    else:
        walkable = free.copy()
    return FreeSpace(origin=origin, res=res, inside=inside, free=free,
                     walkable=walkable)


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _rasterize(poly, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Point-in-polygon on a grid, via matplotlib's fast path."""
    from matplotlib.path import Path
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    mask = np.zeros(pts.shape[0], dtype=bool)
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for g in geoms:
        if g.is_empty:
            continue
        ext = Path(np.asarray(g.exterior.coords))
        m = ext.contains_points(pts)
        for ring in g.interiors:
            m &= ~Path(np.asarray(ring.coords)).contains_points(pts)
        mask |= m
    return mask.reshape(gx.shape)


# --------------------------------------------------------------------------
# clearance
# --------------------------------------------------------------------------
def clearance_polygon(o: ObjectInstance) -> Polygon:
    """The strip of floor an object needs in front of it to be usable."""
    p = prior(o.category)
    if p.front_clear <= 1e-6:
        return Polygon()
    hx, hy = o.half
    f = o.forward
    r = o.right
    depth = p.front_clear
    c = o.xy + f * hy
    a = c - r * hx
    b = c + r * hx
    return Polygon([a, b, b + f * depth, a + f * depth])


def clearance_violation(scene: Scene, res: float = 0.05) -> dict:
    """How much required clearance is stolen by other objects or by walls.

    Returns the total violated area, the violated *fraction* of demanded
    clearance, and the worst per-object fraction.
    """
    room_poly = as_polygon(scene.room)
    objs = scene.kept()
    total_demand = 0.0
    total_viol = 0.0
    worst = 0.0
    per_object = {}
    for i, o in enumerate(objs):
        cp = clearance_polygon(o)
        if cp.is_empty or cp.area <= 1e-9:
            continue
        demand = cp.area
        blocked = cp.difference(room_poly)          # clearance through a wall
        others = [object_polygon(p) for j, p in enumerate(objs)
                  if j != i and _blocks_floor(p) and p.top > o.z + 0.05]
        if others:
            blocked = unary_union([blocked, cp.intersection(unary_union(others))])
        v = float(blocked.area)
        total_demand += demand
        total_viol += v
        frac = v / demand
        per_object[o.oid] = frac
        worst = max(worst, frac)
    return {
        "clearance_violation_area": total_viol,
        "clearance_violation_ratio": total_viol / total_demand if total_demand > 1e-9 else 0.0,
        "clearance_worst_object": worst,
        "per_object": per_object,
    }


# --------------------------------------------------------------------------
# openings
# --------------------------------------------------------------------------
def door_swing(op: Opening, depth: float = 0.9) -> tuple[Polygon, Polygon]:
    """The floor area a door needs to open into the room."""
    u = op.p1 - op.p0
    L = np.linalg.norm(u)
    if L < 1e-6:
        return Polygon()
    u = u / L
    n = op.normal
    a, b = op.p0, op.p1
    cands = [Polygon([a, b, b + n * depth, a + n * depth]),
             Polygon([a, b, b - n * depth, a - n * depth])]
    return cands[0], cands[1]


def door_blockage(scene: Scene) -> dict:
    """Fraction of every door's swing area covered by furniture."""
    room_poly = as_polygon(scene.room)
    objs = [o for o in scene.kept() if _blocks_floor(o)]
    union = unary_union([object_polygon(o) for o in objs]) if objs else Polygon()
    doors = [o for o in scene.room.openings if o.kind == "door"]
    windows = [o for o in scene.room.openings if o.kind == "window"]
    if not doors and not windows:
        return {"door_blockage": 0.0, "window_blockage": 0.0, "n_openings": 0}

    def _blocked(ops, depth):
        tot = viol = 0.0
        for op in ops:
            insides = [g.intersection(room_poly) for g in door_swing(op, depth)]
            g = max(insides, key=lambda p: p.area)
            if g.area <= 1e-9:
                continue
            tot += g.area
            if not union.is_empty:
                viol += float(g.intersection(union).area)
        return viol / tot if tot > 1e-9 else 0.0

    return {
        "door_blockage": _blocked(doors, 0.9),
        "window_blockage": _blocked(windows, 0.35),
        "n_openings": len(scene.room.openings),
    }
