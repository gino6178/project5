"""Procedural 3D-FRONT-like scene generator.

Serves three purposes:

* a dependency-free fallback so the whole pipeline is runnable without the
  3D-FRONT download,
* a *controlled* testbed where the ground-truth motif structure is known, so
  motif-preservation metrics can be validated against a known answer,
* unit-test fixtures.

Rooms are built motif-first: a template lists functional groups with intra-group
relative placements, groups are anchored to walls or to the room centre, and
decorative clutter is sprinkled last.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.categories import prior
from ..core.scene import ObjectInstance, Opening, Room, Scene
from ..geom.polygon import as_polygon, object_polygon, polygon_from_extent

__all__ = ["generate_scene", "generate_dataset", "TEMPLATES"]


@dataclass
class Slot:
    """One object inside a motif, positioned in the motif's local frame."""

    category: str
    size: tuple[float, float, float]
    dx: float = 0.0            # local right offset (metres)
    dy: float = 0.0            # local forward offset
    dyaw: float = 0.0          # local yaw offset (radians)
    z: float = 0.0
    optional: float = 0.0      # probability of being skipped
    size_jitter: float = 0.08
    on: str | None = None      # rest on top of another slot's object


@dataclass
class MotifTemplate:
    name: str
    slots: list[Slot]
    anchor: str = "wall"       # 'wall' | 'centre' | 'free' | 'facing'
    facing: str | None = None  # motif name this one must face
    gap: float = 2.6           # preferred distance when anchor == 'facing'
    optional: float = 0.0


TEMPLATES: dict[str, list[MotifTemplate]] = {
    "bedroom": [
        MotifTemplate("sleeping", anchor="wall", slots=[
            Slot("double_bed", (1.9, 2.1, 0.5)),
            Slot("nightstand", (0.45, 0.42, 0.52), dx=-1.25, dy=0.75, optional=0.15),
            Slot("nightstand", (0.45, 0.42, 0.52), dx=+1.25, dy=0.75, optional=0.15),
            Slot("table_lamp", (0.25, 0.25, 0.45), dx=-1.25, dy=0.75, z=0.52,
                 optional=0.45, on="nightstand"),
        ]),
        MotifTemplate("storage", anchor="wall", slots=[
            Slot("wardrobe", (1.8, 0.62, 2.2)),
        ]),
        MotifTemplate("work", anchor="wall", optional=0.45, slots=[
            Slot("desk", (1.25, 0.6, 0.75)),
            Slot("office_chair", (0.55, 0.58, 0.95), dy=0.72, dyaw=math.pi),
        ]),
        MotifTemplate("dressing", anchor="wall", optional=0.7, slots=[
            Slot("dressing_table", (1.0, 0.45, 0.78)),
            Slot("stool", (0.42, 0.42, 0.45), dy=0.6, dyaw=math.pi),
        ]),
    ],
    "living_room": [
        MotifTemplate("conversation", anchor="wall", slots=[
            Slot("sofa", (2.25, 0.92, 0.82)),
            Slot("coffee_table", (1.15, 0.6, 0.42), dy=1.25),
            Slot("side_table", (0.42, 0.42, 0.55), dx=-1.5, dy=0.1, optional=0.5),
            Slot("floor_lamp", (0.35, 0.35, 1.6), dx=1.5, dy=0.15, optional=0.5),
        ]),
        MotifTemplate("media", anchor="facing", facing="conversation", gap=3.0, slots=[
            Slot("tv_stand", (1.7, 0.42, 0.48)),
            Slot("tv", (1.25, 0.08, 0.72), z=0.48, on="tv_stand"),
        ]),
        MotifTemplate("lounge", anchor="free", optional=0.4, slots=[
            Slot("armchair", (0.78, 0.8, 0.9)),
        ]),
        MotifTemplate("greenery", anchor="wall", optional=0.45, slots=[
            Slot("plant", (0.55, 0.55, 1.35)),
        ]),
    ],
    "dining_room": [
        MotifTemplate("dining", anchor="centre", slots=[
            Slot("dining_table", (1.6, 0.95, 0.76)),
            Slot("dining_chair", (0.48, 0.5, 0.92), dx=-0.55, dy=0.82, dyaw=math.pi),
            Slot("dining_chair", (0.48, 0.5, 0.92), dx=+0.55, dy=0.82, dyaw=math.pi),
            Slot("dining_chair", (0.48, 0.5, 0.92), dx=-0.55, dy=-0.82),
            Slot("dining_chair", (0.48, 0.5, 0.92), dx=+0.55, dy=-0.82),
            Slot("dining_chair", (0.48, 0.5, 0.92), dx=-1.15, dy=0.0,
                 dyaw=math.pi / 2, optional=0.4),
            Slot("dining_chair", (0.48, 0.5, 0.92), dx=+1.15, dy=0.0,
                 dyaw=-math.pi / 2, optional=0.4),
        ]),
        MotifTemplate("buffet", anchor="wall", optional=0.25, slots=[
            Slot("sideboard", (1.5, 0.45, 0.85)),
        ]),
        MotifTemplate("cabinet", anchor="wall", optional=0.6, slots=[
            Slot("wine_cabinet", (0.9, 0.42, 1.9)),
        ]),
    ],
    "library": [
        MotifTemplate("study", anchor="wall", slots=[
            Slot("desk", (1.4, 0.65, 0.76)),
            Slot("office_chair", (0.56, 0.58, 0.98), dy=0.75, dyaw=math.pi),
        ]),
        MotifTemplate("shelving", anchor="wall", slots=[
            Slot("bookcase", (1.6, 0.38, 2.0)),
        ]),
        MotifTemplate("reading", anchor="free", optional=0.35, slots=[
            Slot("armchair", (0.8, 0.82, 0.95)),
            Slot("floor_lamp", (0.33, 0.33, 1.55), dx=0.7, dy=0.1),
        ]),
    ],
}

_DECOR = [("plant", (0.5, 0.5, 1.1)), ("decoration", (0.28, 0.28, 0.35)),
          ("mirror", (0.7, 0.08, 1.1))]

_ROOM_SIZE = {
    "bedroom": ((3.2, 4.6), (3.4, 5.2)),
    "living_room": ((3.8, 6.4), (4.0, 6.8)),
    "dining_room": ((3.2, 5.2), (3.4, 5.4)),
    "library": ((3.0, 4.4), (3.2, 4.8)),
}


def _wall_frames(room: Room) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """(midpoint, inward normal, tangent, length) for every wall."""
    out = []
    for a, b in room.walls():
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        t = d / L
        n = np.array([-t[1], t[0]])
        out.append((a, n, t, L))
    return out


def _place_motif(tmpl: MotifTemplate, room: Room, rng, occupied: list,
                 anchors: dict, idx: int) -> list[ObjectInstance]:
    """Choose a pose for the motif frame, then instantiate its slots."""
    room_poly = as_polygon(room)
    best = None
    for _try in range(140):
        if tmpl.anchor == "wall":
            a, n, t, L = _wall_frames(room)[int(rng.integers(0, len(_wall_frames(room))))]
            s = float(rng.uniform(0.2, 0.8))
            depth = max(sl.size[1] for sl in tmpl.slots) * 0.5
            origin = a + t * (s * L) + n * (depth + 0.03)
            yaw = math.atan2(n[1], n[0]) - math.pi / 2
        elif tmpl.anchor == "facing" and tmpl.facing in anchors:
            oxy, oyaw = anchors[tmpl.facing]
            fwd = np.array([-math.sin(oyaw), math.cos(oyaw)])
            origin = oxy + fwd * tmpl.gap
            yaw = oyaw + math.pi
        elif tmpl.anchor == "centre":
            c = room.centroid
            origin = c + rng.normal(scale=0.25, size=2)
            yaw = float(rng.uniform(0, 2 * math.pi)) if rng.random() < 0.3 else \
                float(rng.choice([0.0, math.pi / 2]))
        else:  # free
            b = room.bbox
            origin = rng.uniform(b[0] + 0.6, b[1] - 0.6)
            yaw = float(rng.uniform(0, 2 * math.pi))

        objs = _instantiate(tmpl, origin, yaw, rng, idx)
        if not objs:
            continue
        ok = all(object_polygon(o).difference(room_poly).area < 1e-3
                 for o in objs if not prior(o.category).on_support)
        if not ok:
            continue
        clash = False
        for o in objs:
            fp = object_polygon(o, 0.02)
            for p in occupied:
                if o.z >= p.z + p.height - 1e-3 or p.z >= o.z + o.height - 1e-3:
                    continue
                if fp.intersection(object_polygon(p)).area > 1e-3:
                    clash = True
                    break
            if clash:
                break
        if clash:
            continue
        best = (objs, origin, yaw)
        break
    if best is None:
        return []
    objs, origin, yaw = best
    anchors[tmpl.name] = (np.asarray(origin, dtype=float), yaw)
    return objs


def _instantiate(tmpl: MotifTemplate, origin, yaw: float, rng, idx: int
                 ) -> list[ObjectInstance]:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    out = []
    for k, sl in enumerate(tmpl.slots):
        if sl.optional > 0 and rng.random() < sl.optional:
            continue
        jit = 1.0 + rng.uniform(-sl.size_jitter, sl.size_jitter, size=3)
        size = np.asarray(sl.size, dtype=float) * jit
        local = np.array([sl.dx, sl.dy])
        xy = np.asarray(origin, dtype=float) + rot @ local
        out.append(ObjectInstance(
            oid=f"m{idx}_{tmpl.name}_{k}_{sl.category}",
            category=sl.category,
            position=np.array([xy[0], xy[1], sl.z]),
            yaw=yaw + sl.dyaw,
            size=size,
            meta={"motif_gt": f"{idx}_{tmpl.name}", "slot": k},
        ))
    return out


def generate_scene(room_type: str = "bedroom", seed: int = 0,
                   polygon: np.ndarray | None = None,
                   n_decor: int | None = None) -> Scene:
    rng = np.random.default_rng(seed)
    if polygon is None:
        (wlo, whi), (dlo, dhi) = _ROOM_SIZE.get(room_type, ((3.5, 5.0), (3.5, 5.0)))
        polygon = polygon_from_extent(float(rng.uniform(wlo, whi)),
                                      float(rng.uniform(dlo, dhi)))
    room = Room(polygon=np.asarray(polygon, dtype=float),
                height=float(rng.uniform(2.6, 3.0)), room_type=room_type)

    # a door on the longest wall, a window on the opposite one
    frames = _wall_frames(room)
    order = np.argsort([-f[3] for f in frames])
    a, n, t, L = frames[order[0]]
    dw = min(0.9, 0.4 * L)
    dc = a + t * (L * float(rng.uniform(0.25, 0.75)))
    room.openings.append(Opening("door", dc - t * dw / 2, dc + t * dw / 2, 0.0, 2.05))
    if len(frames) > 1:
        a2, n2, t2, L2 = frames[order[min(2, len(order) - 1)]]
        ww = min(1.5, 0.5 * L2)
        wc = a2 + t2 * (L2 * 0.5)
        room.openings.append(Opening("window", wc - t2 * ww / 2, wc + t2 * ww / 2, 0.9, 2.2))

    objs: list[ObjectInstance] = []
    anchors: dict = {}
    templates = TEMPLATES.get(room_type, TEMPLATES["bedroom"])
    for i, tmpl in enumerate(templates):
        if tmpl.optional > 0 and rng.random() < tmpl.optional:
            continue
        objs.extend(_place_motif(tmpl, room, rng, objs, anchors, i))

    # a rug under the primary motif
    if anchors and rng.random() < 0.6:
        key = next(iter(anchors))
        oxy, oyaw = anchors[key]
        fwd = np.array([-math.sin(oyaw), math.cos(oyaw)])
        objs.append(ObjectInstance(
            oid=f"rug_{key}", category="rug",
            position=np.array([*(oxy + fwd * 0.9), 0.0]), yaw=oyaw,
            size=np.array([2.0, 1.5, 0.02]), meta={"motif_gt": f"decor"}))

    n_decor = n_decor if n_decor is not None else int(rng.integers(0, 4))
    room_poly = as_polygon(room)
    for k in range(n_decor):
        cat, size = _DECOR[int(rng.integers(0, len(_DECOR)))]
        for _try in range(40):
            b = room.bbox
            xy = rng.uniform(b[0] + 0.4, b[1] - 0.4)
            o = ObjectInstance(f"decor_{k}", cat, np.array([xy[0], xy[1], 0.0]),
                               float(rng.uniform(0, 2 * math.pi)),
                               np.asarray(size), meta={"motif_gt": "decor"})
            fp = object_polygon(o, 0.03)
            if fp.difference(room_poly).area > 1e-3:
                continue
            if any(fp.intersection(object_polygon(p)).area > 1e-3
                   for p in objs if p.z < o.top and o.z < p.top):
                continue
            objs.append(o)
            break

    if rng.random() < 0.5:
        c = room.centroid
        objs.append(ObjectInstance(
            "ceiling_lamp", "pendant_lamp",
            np.array([c[0], c[1], room.height - 0.5]), 0.0,
            np.array([0.4, 0.4, 0.4]), meta={"motif_gt": "decor"}))

    return Scene(scene_id=f"proc_{room_type}_{seed}", room=room, objects=objs,
                 source="procedural", meta={"seed": seed})


def generate_dataset(n: int = 200, room_types=("bedroom", "living_room", "dining_room"),
                     seed0: int = 0) -> list[Scene]:
    out = []
    for i in range(n):
        rt = room_types[i % len(room_types)]
        s = generate_scene(rt, seed=seed0 + i)
        if len(s.objects) >= 3:
            out.append(s)
    return out
