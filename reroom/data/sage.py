"""SAGE-10k loader (plan section 3.2).

SAGE-10k is 10 000 agentically generated scenes over ~50 room types, rich in
clutter and small objects.  The plan positions it deliberately: it is used for
*appearance, object diversity and open-vocabulary augmentation*, **not** as
irregular-room ground truth.  Its public schema stores walls as line segments,
which could express complex geometry, but the released layouts are dominated by
axis-aligned rectangles generated from a width/length pair -- so relying on it
for room shape would be a mistake.  ``room_is_rectangular`` makes that check
explicit rather than leaving it as an assumption.

The layout JSON is already z-up with the floor at ``z = 0``, so no axis
conversion is needed; rotations are degrees about the up axis.
"""
from __future__ import annotations

import glob
import json
import math
import os
import zipfile
from collections import Counter

import numpy as np

from ..core.categories import canonical_category, canonical_room_type
from ..core.scene import ObjectInstance, Opening, Room, Scene
from ..geom.polygon import normalize_polygon, signed_area

__all__ = ["parse_layout", "iter_sage_scenes", "room_is_rectangular",
           "sage_style_text"]


def _pt(d) -> np.ndarray:
    return np.array([float(d["x"]), float(d["y"])])


def _walls_to_polygon(walls: list[dict], tol: float = 1e-3) -> np.ndarray | None:
    """Chain wall segments into a closed ring."""
    segs = [(_pt(w["start_point"]), _pt(w["end_point"])) for w in walls]
    if len(segs) < 3:
        return None
    used = [False] * len(segs)
    ring = [segs[0][0], segs[0][1]]
    used[0] = True
    for _ in range(len(segs) * 2):
        tail = ring[-1]
        nxt = None
        for k, (a, b) in enumerate(segs):
            if used[k]:
                continue
            if np.linalg.norm(a - tail) < tol:
                nxt, used[k] = b, True
                break
            if np.linalg.norm(b - tail) < tol:
                nxt, used[k] = a, True
                break
        if nxt is None:
            break
        if np.linalg.norm(nxt - ring[0]) < tol:
            break
        ring.append(nxt)
    if len(ring) < 3:
        return None
    poly = normalize_polygon(np.asarray(ring))
    return poly if len(poly) >= 3 else None


def room_is_rectangular(room: Room, tol: float = 0.02) -> bool:
    """Is this room simply the axis-aligned box of its own extent?"""
    if len(room.polygon) != 4:
        return False
    ext = room.extent
    return abs(room.area - float(ext[0] * ext[1])) <= tol * max(room.area, 1e-6)


def _openings(room_dict: dict, walls: list[dict]) -> list[Opening]:
    lut = {w["id"]: w for w in walls}
    out = []
    for kind, key in (("door", "doors"), ("window", "windows")):
        for o in room_dict.get(key, ()) or ():
            w = lut.get(o.get("wall_id"))
            if w is None:
                continue
            a, b = _pt(w["start_point"]), _pt(w["end_point"])
            d = b - a
            L = float(np.linalg.norm(d))
            if L < 1e-6:
                continue
            u = d / L
            t = float(np.clip(o.get("position_on_wall", 0.5), 0.0, 1.0))
            half = min(float(o.get("width", 0.9)) / 2, 0.45 * L)
            c = a + u * (t * L)
            z0 = float(o.get("sill_height", 0.0 if kind == "door" else 0.9))
            out.append(Opening(kind, c - u * half, c + u * half,
                               z0, z0 + float(o.get("height", 2.0))))
    return out


def parse_layout(path_or_dict, min_objects: int = 4,
                 max_objects: int = 40) -> list[Scene]:
    """Parse one SAGE-10k ``layout_*.json`` into per-room ReRoom scenes."""
    if isinstance(path_or_dict, dict):
        d = path_or_dict
        src_name = d.get("id", "sage")
    else:
        with open(path_or_dict) as fh:
            d = json.load(fh)
        src_name = os.path.splitext(os.path.basename(str(path_or_dict)))[0]

    out: list[Scene] = []
    for r in d.get("rooms", ()):
        walls = r.get("walls", [])
        poly = _walls_to_polygon(walls)
        if poly is None:
            dim = r.get("dimensions", {})
            w, l = float(dim.get("width", 0)), float(dim.get("length", 0))
            if w <= 0 or l <= 0:
                continue
            poly = np.array([[0, 0], [w, 0], [w, l], [0, l]], dtype=float)
        height = float(r.get("ceiling_height",
                             r.get("dimensions", {}).get("height", 2.7)))
        room = Room(polygon=poly, height=height,
                    openings=_openings(r, walls),
                    room_type=canonical_room_type(r.get("room_type")))

        objs = []
        support_of = {}
        for o in r.get("objects", ()):
            dim = o.get("dimensions") or {}
            size = np.array([float(dim.get("width", 0.4)),
                             float(dim.get("length", 0.4)),
                             float(dim.get("height", 0.4))])
            if size.min() <= 1e-3 or size.max() > 8.0:
                continue
            p = o.get("position") or {}
            pos = np.array([float(p.get("x", 0.0)), float(p.get("y", 0.0)),
                            float(p.get("z", 0.0))])
            rot = (o.get("rotation") or {}).get("z", 0.0)
            desc = o.get("description") or ""
            raw = str(o.get("type", "")).rsplit("_", 1)[0]
            cat = canonical_category(f"{raw} {desc[:90]}", raw)
            inst = ObjectInstance(
                oid=o.get("id", f"sage_{len(objs)}"), category=cat,
                position=pos, yaw=math.radians(float(rot)), size=size,
                jid=o.get("source_id"), raw_category=raw,
                meta={"description": desc, "place_id": o.get("place_id"),
                      "mass": o.get("mass"), "source": o.get("source")})
            objs.append(inst)
            pid = o.get("place_id")
            if pid and pid != "floor":
                support_of[inst.oid] = pid

        if not (min_objects <= len(objs)):
            continue
        if len(objs) > max_objects:
            objs.sort(key=lambda o: -o.footprint_area)
            objs = objs[:max_objects]

        sc = Scene(scene_id=f"{src_name}__{r.get('id', 'room')}", room=room,
                   objects=objs, source="SAGE-10k",
                   meta={"layout_id": d.get("id"),
                         "raw_room_type": r.get("room_type"),
                         "building_style": d.get("building_style"),
                         "description": d.get("description"),
                         "support_of": support_of,
                         "rectangular": None})
        sc.meta["rectangular"] = room_is_rectangular(room)
        out.append(sc)
    return out


def iter_sage_scenes(root: str, limit: int | None = None, **kw):
    """Iterate scenes from a directory of extracted layouts or scene zips."""
    n = 0
    paths = sorted(glob.glob(os.path.join(root, "**", "layout_*.json"),
                             recursive=True))
    for p in paths:
        for s in parse_layout(p, **kw):
            yield s
            n += 1
            if limit and n >= limit:
                return
    for z in sorted(glob.glob(os.path.join(root, "*.zip"))):
        try:
            with zipfile.ZipFile(z) as zf:
                for name in zf.namelist():
                    if not (name.endswith(".json") and "layout_" in name):
                        continue
                    with zf.open(name) as fh:
                        d = json.load(fh)
                    for s in parse_layout(d, **kw):
                        yield s
                        n += 1
                        if limit and n >= limit:
                            return
        except zipfile.BadZipFile:
            continue


def sage_style_text(scene: Scene) -> str:
    """A short style prompt for ``z_style``, from SAGE's own descriptions."""
    bits = [scene.meta.get("building_style") or "",
            (scene.meta.get("description") or "")[:200]]
    return " ".join(b for b in bits if b).strip()
