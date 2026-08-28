"""Walkability mechanisms (PhyScene-inspired), inserted into the existing
Summarise -> Guidance/Polish -> Ranking pipeline without changing the
architecture.

Three pieces, each usable independently:

  * capacity gate (Summarise)      -- ``walkable_ratio`` + ``capacity_prune``:
    a small room physically cannot hold the reference's full furniture set AND
    stay walkable, so when free-floor ratio drops below a threshold we demote
    dining sets (drop the wall-side chairs) and shed low-degree secondary items
    before the flow ever sees them.
  * door boxes + affordance (Polish)-- ``door_boxes`` marks a 1.0x0.9 m no-go
    zone just inside every door; ``walkable_push`` repels furniture out of it
    and out of each other's human-activity buffers (sofa front, chair back, TV
    front).
  * navigation score (Ranking)     -- ``nav_penalty``: an A* check from the
    door to each anchor over a 10 cm grid; layouts that are collision-free but
    seal a corridor (< 50 cm) are penalised so a walkable candidate wins.
"""
from __future__ import annotations
import heapq
import math
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import translate

from ..geom.polygon import as_polygon, object_polygon

ROBOT_W = 0.30           # metres, PhyScene robot_real_width
DOOR_DEPTH = 0.90        # no-go depth into the room
DOOR_HALF = 0.50         # no-go half-width along the door
CORRIDOR_MIN = 0.50      # metres, minimum passable corridor

# directional human-activity buffer: (front_m, back_m, side_m)
AFFORDANCE = {
    "sofa": (0.35, 0.0, 0.0), "l_sofa": (0.35, 0.0, 0.0),
    "loveseat": (0.35, 0.0, 0.0), "armchair": (0.30, 0.0, 0.0),
    "lounge_chair": (0.30, 0.0, 0.0),
    "dining_chair": (0.0, 0.40, 0.0), "office_chair": (0.0, 0.40, 0.0),
    "tv_stand": (0.80, 0.0, 0.0), "tv": (0.80, 0.0, 0.0),
    "bed": (0.35, 0.0, 0.0), "double_bed": (0.35, 0.0, 0.0),
    "single_bed": (0.35, 0.0, 0.0), "desk": (0.45, 0.0, 0.0),
}


def door_boxes(room, depth: float = DOOR_DEPTH, half: float = DOOR_HALF):
    """One no-go rectangle just inside each door, as shapely polygons."""
    poly = as_polygon(room)
    centroid = np.array(poly.centroid.coords[0])
    boxes = []
    for op in room.openings:
        if op.kind != "door":
            continue
        m = op.centre
        t = op.p1 - op.p0
        L = np.linalg.norm(t) + 1e-9
        t = t / L
        n = op.normal
        if np.dot(centroid - m, n) < 0:         # make the normal point inward
            n = -n
        hw = max(half, L * 0.5)
        c = [m - t * hw, m + t * hw,
             m + t * hw + n * depth, m - t * hw + n * depth]
        boxes.append(Polygon(c))
    return boxes


def affordance_zone(obj):
    """A shapely polygon = object footprint extended by its directional human-
    activity buffer (front along +forward, back along -forward)."""
    fp = object_polygon(obj)
    front, back, side = AFFORDANCE.get(obj.category, (0.0, 0.0, 0.0))
    if front == back == side == 0.0:
        return fp
    f = obj.forward
    zone = fp
    if front > 0:
        zone = zone.union(translate(fp, xoff=f[0] * front, yoff=f[1] * front)).convex_hull
    if back > 0:
        zone = zone.union(translate(fp, xoff=-f[0] * back, yoff=-f[1] * back)).convex_hull
    return zone


def walkable_ratio(scene) -> float:
    """(room area - furniture footprint area) / room area, on kept objects."""
    poly = as_polygon(scene.room)
    if poly.area <= 1e-6:
        return 1.0
    occ = 0.0
    for o in scene.objects:
        if not o.keep or o.z >= 1.4:
            continue
        occ += object_polygon(o).area
    return max(0.0, 1.0 - occ / poly.area)


def walkable_push(scene, iters: int = 6):
    """Repel furniture out of door boxes and out of each other's affordance
    buffers (in place).  A light iterative geometric solver run after polish."""
    poly = as_polygon(scene.room)
    boxes = door_boxes(scene.room)
    objs = [o for o in scene.objects if o.keep and o.z < 1.4]
    for _ in range(iters):
        moved = False
        # 1. door boxes: push object deeper into the room, away from the door
        for o in objs:
            fp = object_polygon(o)
            for b in boxes:
                if fp.intersects(b):
                    inter = fp.intersection(b)
                    if inter.area < 1e-4:
                        continue
                    # push direction: from door-box centre toward room centroid
                    bc = np.array(b.centroid.coords[0])
                    rc = np.array(poly.centroid.coords[0])
                    d = rc - bc
                    nd = np.linalg.norm(d)
                    if nd < 1e-6:
                        continue
                    o.xy = o.xy + (d / nd) * 0.12
                    moved = True
        # 2. affordance: if A's body intrudes B's buffer, push apart
        for i in range(len(objs)):
            zi = affordance_zone(objs[i])
            fi = object_polygon(objs[i])
            for j in range(len(objs)):
                if i == j:
                    continue
                fj = object_polygon(objs[j])
                if not fj.intersects(zi):
                    continue
                # only push if bodies aren't a legitimate nestable pair
                from ..generative.guidance import NESTABLE_PAIRS
                if frozenset({objs[i].category, objs[j].category}) in NESTABLE_PAIRS:
                    continue
                ci = objs[i].xy; cj = objs[j].xy
                d = cj - ci
                nd = np.linalg.norm(d)
                if nd < 1e-6:
                    continue
                objs[j].xy = objs[j].xy + (d / nd) * 0.06
                moved = True
        # keep everything inside
        for o in objs:
            if not poly.contains(Point(*o.xy)):
                ex = poly.exterior
                p_on = ex.interpolate(ex.project(Point(*o.xy)))
                c = np.array(poly.centroid.coords[0])
                to_c = c - np.array([p_on.x, p_on.y])
                n = np.linalg.norm(to_c)
                o.xy = (np.array([p_on.x, p_on.y]) + to_c / n * 0.05
                        if n > 1e-6 else np.array([p_on.x, p_on.y]))
        if not moved:
            break
    return scene


def _grid_obstacles(scene, cell: float = 0.10):
    poly = as_polygon(scene.room)
    minx, miny, maxx, maxy = poly.bounds
    W = max(int((maxx - minx) / cell) + 1, 1)
    H = max(int((maxy - miny) / cell) + 1, 1)
    free = np.zeros((H, W), dtype=bool)
    # erode room by robot radius: a cell is floor if a robot disk fits
    r = ROBOT_W / 2
    for j in range(H):
        for i in range(W):
            x = minx + (i + 0.5) * cell
            y = miny + (j + 0.5) * cell
            free[j, i] = poly.contains(Point(x, y).buffer(r * 0.6))
    for o in scene.objects:
        if not o.keep or o.z >= 1.4:
            continue
        fp = object_polygon(o).buffer(r)
        bx0, by0, bx1, by1 = fp.bounds
        i0 = max(int((bx0 - minx) / cell), 0); i1 = min(int((bx1 - minx) / cell) + 1, W)
        j0 = max(int((by0 - miny) / cell), 0); j1 = min(int((by1 - miny) / cell) + 1, H)
        for j in range(j0, j1):
            for i in range(i0, i1):
                x = minx + (i + 0.5) * cell; y = miny + (j + 0.5) * cell
                if fp.contains(Point(x, y)):
                    free[j, i] = False
    return free, (minx, miny, cell)


def _astar(free, a, b):
    H, W = free.shape
    if not (0 <= a[0] < H and 0 <= a[1] < W and 0 <= b[0] < H and 0 <= b[1] < W):
        return False
    if not free[a] or not free[b]:
        return False
    openq = [(0, a)]
    seen = {a: 0}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == b:
            return True
        cj, ci = cur
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nj, ni = cj + dj, ci + di
            if 0 <= nj < H and 0 <= ni < W and free[nj, ni]:
                g = seen[cur] + (1.4 if dj and di else 1.0)
                if (nj, ni) not in seen or g < seen[(nj, ni)]:
                    seen[(nj, ni)] = g
                    h = abs(nj - b[0]) + abs(ni - b[1])
                    heapq.heappush(openq, (g + h, (nj, ni)))
    return False


def nav_penalty(scene, cell: float = 0.10):
    """A* from the door(s) to every anchor over a robot-eroded grid.  Returns a
    penalty in [0, large]: 0 if all anchors reachable, big if a corridor is
    sealed."""
    doors = [op for op in scene.room.openings if op.kind == "door"]
    anchors = [o for o in scene.objects
               if o.keep and o.z < 1.4 and o.category in (
                   "sofa", "l_sofa", "bed", "double_bed", "single_bed",
                   "dining_table", "desk", "tv_stand")]
    if not doors or not anchors:
        return 0.0
    free, (minx, miny, c) = _grid_obstacles(scene, cell)
    H, W = free.shape

    def to_cell(xy):
        i = int((xy[0] - minx) / c); j = int((xy[1] - miny) / c)
        j = min(max(j, 0), H - 1); i = min(max(i, 0), W - 1)
        # snap to nearest free cell within a small window
        if free[j, i]:
            return (j, i)
        for rad in range(1, 6):
            for dj in range(-rad, rad + 1):
                for di in range(-rad, rad + 1):
                    nj, ni = j + dj, i + di
                    if 0 <= nj < H and 0 <= ni < W and free[nj, ni]:
                        return (nj, ni)
        return None
    # door seed: a point just inside the door
    dcell = None
    for op in doors:
        m = op.centre; n = op.normal
        rc = np.array(as_polygon(scene.room).centroid.coords[0])
        if np.dot(rc - m, n) < 0:
            n = -n
        seed = m + n * 0.4
        dcell = to_cell(seed)
        if dcell:
            break
    if dcell is None:
        return 0.0
    unreachable = 0
    for a in anchors:
        # target: a point in front of the anchor (its affordance side)
        tgt = a.xy + a.forward * (float(a.half[1]) + 0.3)
        tcell = to_cell(tgt) or to_cell(a.xy)
        if tcell is None or not _astar(free, dcell, tcell):
            unreachable += 1
    return float(unreachable) * 100.0


CORE_KEEP = frozenset({
    "sofa", "l_sofa", "loveseat", "tv_stand", "tv", "bed", "double_bed",
    "single_bed", "kids_bed", "dining_table", "wardrobe", "desk", "coffee_table",
})
SECONDARY_DROP = ("decoration", "plant", "mirror", "shoe_cabinet", "side_table",
                  "console_table", "wine_cabinet", "stool", "floor_lamp",
                  "table_lamp", "shelf", "bench")


def _ratio_in(scene, target_room):
    """Free-floor ratio of the scene's kept furniture measured against the
    TARGET room area (the furniture is still at reference size at this stage)."""
    poly = as_polygon(target_room)
    if poly.area <= 1e-6:
        return 1.0
    occ = sum(object_polygon(o).area for o in scene.objects
              if o.keep and o.z < 1.4)
    return max(0.0, 1.0 - occ / poly.area)


def capacity_prune(scene, target_room, min_walkable: float = 0.55,
                   keep_chairs: int = 2):
    """Summarise-stage capacity gate: if the TARGET room is over-full (free-floor
    ratio below ``min_walkable``), demote dining sets to ``keep_chairs`` and shed
    secondary items (keeping core anchors) until it breathes.  Marks dropped
    objects ``keep=False`` in place; returns the number dropped."""
    if _ratio_in(scene, target_room) >= min_walkable:
        return 0
    dropped = 0
    poly = as_polygon(scene.room)

    # 1. dining set demote: keep the chairs closest to the table's long front,
    #    drop the wall-side extras.
    tables = [o for o in scene.objects if o.keep and o.category == "dining_table"]
    for tbl in tables:
        chairs = [o for o in scene.objects
                  if o.keep and o.category == "dining_chair"]
        chairs.sort(key=lambda c: float(np.linalg.norm(c.xy - tbl.xy)))
        # keep the nearest keep_chairs, drop the rest that are wall-side
        for c in chairs[keep_chairs:]:
            # wall-side = footprint near the room boundary
            if poly.exterior.distance(Point(*c.xy)) < 0.6:
                c.keep = False; dropped += 1
        if _ratio_in(scene, target_room) >= min_walkable:
            return dropped

    # 2. shed secondary items by category priority, then by importance (smallest
    #    footprint / lowest anchor first), until walkable or exhausted.
    for cat in SECONDARY_DROP:
        items = [o for o in scene.objects
                 if o.keep and o.category == cat and o.z < 1.4]
        for o in items:
            o.keep = False; dropped += 1
            if walkable_ratio(scene) >= min_walkable:
                return dropped
    return dropped
