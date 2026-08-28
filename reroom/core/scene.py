"""Core scene representation for ReRoom.

Conventions
-----------
* Floor plane is ``xy``; ``z`` is up.  3D-FRONT is y-up, the parser converts.
* An object's footprint is an *oriented* rectangle: centre ``(x, y)``, yaw
  ``theta`` (radians, CCW from +x), and full extents ``(sx, sy)`` measured in the
  object's own frame *before* rotation.
* ``size`` is the full 3D extent ``(sx, sy, sz)`` of the asset after scaling.
* ``z`` is the height of the *bottom* of the object above the floor, so objects
  resting on the floor have ``z == 0`` and objects on a support have ``z > 0``.

Everything downstream (relations, energies, metrics) reads only this module, so
3D-FRONT, SAGE-10k and the procedural generator can all feed the same pipeline.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "ObjectInstance",
    "Opening",
    "Room",
    "Scene",
    "scene_from_dict",
    "scene_to_dict",
]


def _json_meta(meta: dict) -> dict:
    """``meta`` is free-form, and arrays get put in it -- ``f^geo`` for one.

    Serialisation is where that bites, so arrays are converted here rather than
    forbidden at every call site.
    """
    out = {}
    for k, v in meta.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _arr(v, n, dtype=float) -> np.ndarray:
    a = np.asarray(v, dtype=dtype).reshape(-1)
    if a.size != n:
        raise ValueError(f"expected {n} components, got {a.size}")
    return a


@dataclass
class ObjectInstance:
    """A single editable furniture instance."""

    oid: str
    category: str                      # canonical ReRoom category
    position: np.ndarray               # (3,) centre of footprint, z = bottom
    yaw: float                         # radians, CCW about +z
    size: np.ndarray                   # (3,) full extents (sx, sy, sz)
    jid: str | None = None             # source asset id (3D-FUTURE model id)
    raw_category: str | None = None    # dataset-native category string
    style: str | None = None
    color: tuple[float, float, float] | None = None
    keep: bool = True                  # k_i in eq. (17)
    locked: bool = False               # part of C_t: the user pinned this one,
                                       # so the solver may not move or drop it
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = _arr(self.position, 3)
        self.size = np.abs(_arr(self.size, 3))
        self.yaw = float(self.yaw)

    # -- convenience views -------------------------------------------------
    @property
    def xy(self) -> np.ndarray:
        return self.position[:2].copy()

    @xy.setter
    def xy(self, v) -> None:
        self.position[:2] = _arr(v, 2)

    @property
    def z(self) -> float:
        return float(self.position[2])

    @property
    def half(self) -> np.ndarray:
        """Footprint half extents in the object frame."""
        return self.size[:2] * 0.5

    @property
    def height(self) -> float:
        return float(self.size[2])

    @property
    def top(self) -> float:
        return self.z + self.height

    @property
    def footprint_area(self) -> float:
        return float(self.size[0] * self.size[1])

    @property
    def forward(self) -> np.ndarray:
        """Unit vector the object faces.

        Local +y is 'front' (the direction a sofa seat or a TV screen looks),
        matching the 3D-FRONT convention after the y-up -> z-up conversion.
        """
        return np.array([-math.sin(self.yaw), math.cos(self.yaw)])

    @property
    def right(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw)])

    def corners(self) -> np.ndarray:
        """(4, 2) CCW footprint corners in world coordinates."""
        hx, hy = self.half
        local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + self.xy

    def bounds3d(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned world bounds of the (rotated) box."""
        c = self.corners()
        lo = np.array([c[:, 0].min(), c[:, 1].min(), self.z])
        hi = np.array([c[:, 0].max(), c[:, 1].max(), self.top])
        return lo, hi

    def copy(self) -> "ObjectInstance":
        return ObjectInstance(
            oid=self.oid,
            category=self.category,
            position=self.position.copy(),
            yaw=self.yaw,
            size=self.size.copy(),
            jid=self.jid,
            raw_category=self.raw_category,
            style=self.style,
            color=self.color,
            keep=self.keep,
            locked=self.locked,
            meta=dict(self.meta),
        )


@dataclass
class Opening:
    """A door or window, stored as a segment on the floor polygon boundary."""

    kind: str                 # 'door' | 'window'
    p0: np.ndarray            # (2,)
    p1: np.ndarray            # (2,)
    z0: float = 0.0
    z1: float = 2.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.p0 = _arr(self.p0, 2)
        self.p1 = _arr(self.p1, 2)

    @property
    def centre(self) -> np.ndarray:
        return (self.p0 + self.p1) * 0.5

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def normal(self) -> np.ndarray:
        d = self.p1 - self.p0
        n = np.array([-d[1], d[0]])
        ln = np.linalg.norm(n)
        return n / ln if ln > 1e-9 else np.array([1.0, 0.0])

    def copy(self) -> "Opening":
        return Opening(self.kind, self.p0.copy(), self.p1.copy(), self.z0, self.z1, dict(self.meta))


@dataclass
class Room:
    """Target/source room: a simple (possibly concave) polygon plus openings.

    ``keepout`` carries the rest of the plan's optional constraints ``C_t``:
    floor regions that must stay clear -- the swing of a door the user opens
    every day, the strip in front of a window they do not want blocked, a
    walkway.  Each entry is a polygon in room coordinates.
    """

    polygon: np.ndarray               # (M, 2) CCW, no repeated last vertex
    height: float = 2.8
    openings: list[Opening] = field(default_factory=list)
    room_type: str = "unknown"
    keepout: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        poly = np.asarray(self.polygon, dtype=float).reshape(-1, 2)
        if len(poly) >= 2 and np.allclose(poly[0], poly[-1]):
            poly = poly[:-1]
        if _signed_area(poly) < 0:
            poly = poly[::-1]
        self.polygon = poly

    @property
    def area(self) -> float:
        return abs(_signed_area(self.polygon))

    @property
    def centroid(self) -> np.ndarray:
        return self.polygon.mean(axis=0)

    @property
    def bbox(self) -> np.ndarray:
        return np.array([self.polygon.min(axis=0), self.polygon.max(axis=0)])

    @property
    def extent(self) -> np.ndarray:
        b = self.bbox
        return b[1] - b[0]

    def walls(self) -> list[tuple[np.ndarray, np.ndarray]]:
        p = self.polygon
        return [(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]

    def keepout_polygons(self):
        """The keep-out regions as shapely polygons, clipped to the room."""
        from shapely.geometry import Polygon as _P
        out = []
        room = _P(self.polygon)
        for k in self.keepout:
            try:
                g = _P(np.asarray(k, dtype=float).reshape(-1, 2))
                if not g.is_valid:
                    g = g.buffer(0)
                g = g.intersection(room)
                if not g.is_empty and g.area > 1e-6:
                    out.append(g)
            except Exception:
                continue
        return out

    def copy(self) -> "Room":
        return Room(self.polygon.copy(), self.height,
                    [o.copy() for o in self.openings], self.room_type,
                    [np.asarray(k, dtype=float).copy() for k in self.keepout])


@dataclass
class Scene:
    """An editable 3D scene: ``S = {(c_i, a_i, p_i, R_i, s_i)}``."""

    scene_id: str
    room: Room
    objects: list[ObjectInstance] = field(default_factory=list)
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    # -- basics ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self) -> Iterable[ObjectInstance]:
        return iter(self.objects)

    @property
    def room_type(self) -> str:
        return self.room.room_type

    def kept(self) -> list[ObjectInstance]:
        return [o for o in self.objects if o.keep]

    def by_id(self, oid: str) -> ObjectInstance | None:
        for o in self.objects:
            if o.oid == oid:
                return o
        return None

    def index_of(self, oid: str) -> int:
        for i, o in enumerate(self.objects):
            if o.oid == oid:
                return i
        raise KeyError(oid)

    def categories(self) -> list[str]:
        return [o.category for o in self.objects]

    def occupied_area(self, kept_only: bool = True) -> float:
        objs = self.kept() if kept_only else self.objects
        return float(sum(o.footprint_area for o in objs))

    def density(self, kept_only: bool = True) -> float:
        """rho(S) of eq. (28)."""
        a = self.room.area
        return self.occupied_area(kept_only) / a if a > 1e-9 else 0.0

    def recentre(self) -> "Scene":
        """Translate so the room polygon's bbox centre sits at the origin."""
        b = self.room.bbox
        c = (b[0] + b[1]) * 0.5
        self.room.polygon = self.room.polygon - c
        for op in self.room.openings:
            op.p0 -= c
            op.p1 -= c
        for o in self.objects:
            o.position[:2] -= c
        return self

    def copy(self) -> "Scene":
        return Scene(
            scene_id=self.scene_id,
            room=self.room.copy(),
            objects=[o.copy() for o in self.objects],
            source=self.source,
            meta=dict(self.meta),
        )

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return scene_to_dict(self)

    @staticmethod
    def from_dict(d: dict) -> "Scene":
        return scene_from_dict(d)

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)

    @staticmethod
    def load(path) -> "Scene":
        with open(path) as fh:
            return scene_from_dict(json.load(fh))


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def scene_to_dict(s: Scene) -> dict:
    return {
        "scene_id": s.scene_id,
        "source": s.source,
        "meta": s.meta,
        "room": {
            "polygon": s.room.polygon.tolist(),
            "height": s.room.height,
            "room_type": s.room.room_type,
            "keepout": [np.asarray(k, dtype=float).tolist() for k in s.room.keepout],
            "openings": [
                {"kind": o.kind, "p0": o.p0.tolist(), "p1": o.p1.tolist(),
                 "z0": o.z0, "z1": o.z1, "meta": o.meta}
                for o in s.room.openings
            ],
        },
        "objects": [
            {"oid": o.oid, "category": o.category, "position": o.position.tolist(),
             "yaw": o.yaw, "size": o.size.tolist(), "jid": o.jid,
             "raw_category": o.raw_category, "style": o.style,
             "color": list(o.color) if o.color else None,
             "keep": o.keep, "locked": o.locked, "meta": _json_meta(o.meta)}
            for o in s.objects
        ],
    }


def scene_from_dict(d: dict) -> Scene:
    r = d["room"]
    room = Room(
        polygon=np.asarray(r["polygon"], dtype=float),
        height=float(r.get("height", 2.8)),
        openings=[Opening(o["kind"], o["p0"], o["p1"], o.get("z0", 0.0), o.get("z1", 2.0),
                          o.get("meta", {})) for o in r.get("openings", [])],
        room_type=r.get("room_type", "unknown"),
        keepout=[np.asarray(k, dtype=float) for k in r.get("keepout", [])],
    )
    objs = [
        ObjectInstance(
            oid=o["oid"], category=o["category"], position=o["position"], yaw=o["yaw"],
            size=o["size"], jid=o.get("jid"), raw_category=o.get("raw_category"),
            style=o.get("style"), color=tuple(o["color"]) if o.get("color") else None,
            keep=o.get("keep", True), locked=o.get("locked", False),
            meta=o.get("meta", {}),
        )
        for o in d.get("objects", [])
    ]
    return Scene(scene_id=d["scene_id"], room=room, objects=objs,
                 source=d.get("source", "unknown"), meta=d.get("meta", {}))
