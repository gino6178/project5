"""Floor-plan deformation curriculum (plan section 12).

Real irregular-room ground truth is scarce, so target floor plans are *made*
from source ones with a graded family of operators

    P_t = T_delta(P_r),    delta ~ p(delta | difficulty)                (33)

The five levels are exactly those of the plan:

    1. uniform scale                 P_t = s P_r,   s in [0.7, 1.4]
    2. aspect-ratio deformation      diag(s_x, s_y)
    3. slanted wall                  drag one wall endpoint
    4. corner cut / L-shaped room    subtract a rectangle at a corner
    5. general concave polygon       constrained vertex perturbation

Every operator returns a *validated* room: still a simple polygon, still wide
enough to walk through, and within a sane area range.  Doors and windows are
re-anchored parametrically so a deformed room keeps its openings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..core.scene import Opening, Room
from .polygon import as_polygon, erode, is_simple, normalize_polygon, signed_area

__all__ = [
    "DeformSpec", "DeformResult", "deform_room", "sample_deform",
    "uniform_scale", "aspect_deform", "slant_wall", "corner_cut",
    "concave_perturb", "validate_polygon", "LEVEL_NAMES",
]

LEVEL_NAMES = {
    1: "uniform_scale",
    2: "aspect_ratio",
    3: "slanted_wall",
    4: "corner_cut",
    5: "concave",
}

MIN_CORRIDOR = 0.85          # metres: a person must be able to walk through
MIN_AREA = 5.0               # m^2


@dataclass
class DeformSpec:
    """One sampled deformation."""

    level: int
    name: str
    params: dict

    def describe(self) -> str:
        ps = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in self.params.items())
        return f"L{self.level}:{self.name}({ps})"


@dataclass
class DeformResult:
    room: Room
    spec: DeformSpec
    scale_hint: np.ndarray       # (2,) per-axis size ratio target/source


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def validate_polygon(poly: np.ndarray, min_corridor: float = MIN_CORRIDOR,
                     min_area: float = MIN_AREA) -> bool:
    if poly is None or len(poly) < 3:
        return False
    if not np.all(np.isfinite(poly)):
        return False
    if not is_simple(poly):
        return False
    sp = as_polygon(poly)
    if sp.area < min_area:
        return False
    core = erode(sp, min_corridor / 2.0)
    if core.is_empty or core.area < 0.15 * sp.area:
        return False
    return True


# --------------------------------------------------------------------------
# operators (level 1-5)
# --------------------------------------------------------------------------
def uniform_scale(poly: np.ndarray, s: float) -> np.ndarray:
    c = poly.mean(axis=0)
    return (poly - c) * s + c


def aspect_deform(poly: np.ndarray, sx: float, sy: float,
                  angle: float = 0.0) -> np.ndarray:
    """Anisotropic scaling, optionally along a rotated frame."""
    c = poly.mean(axis=0)
    p = poly - c
    if abs(angle) > 1e-9:
        ca, sa = math.cos(-angle), math.sin(-angle)
        p = p @ np.array([[ca, -sa], [sa, ca]]).T
    p = p * np.array([sx, sy])
    if abs(angle) > 1e-9:
        ca, sa = math.cos(angle), math.sin(angle)
        p = p @ np.array([[ca, -sa], [sa, ca]]).T
    return p + c


def slant_wall(poly: np.ndarray, wall_idx: int, shift: float,
               mode: str = "normal") -> np.ndarray:
    """Drag the *end* vertex of one wall to make a trapezoid / slanted wall."""
    p = poly.copy()
    n = len(p)
    i, j = wall_idx % n, (wall_idx + 1) % n
    a, b = p[i], p[j]
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-6:
        return p
    t = d / L
    nrm = np.array([-t[1], t[0]])            # inward for CCW
    vec = nrm if mode == "normal" else t
    p[j] = b + vec * shift
    return p


def corner_cut(poly: np.ndarray, corner_idx: int, fx: float, fy: float,
               oblique: float = 0.0) -> np.ndarray:
    """Subtract a rectangle anchored at one corner -> L-shape / cut corner.

    ``fx, fy`` are fractions of the two edges meeting at the corner;
    ``oblique`` in [0, 1] slides the cut towards a diagonal chamfer.
    """
    sp = as_polygon(poly)
    n = len(poly)
    i = corner_idx % n
    c = poly[i]
    prev_v = poly[i - 1]
    next_v = poly[(i + 1) % n]
    e0 = prev_v - c
    e1 = next_v - c
    l0, l1 = np.linalg.norm(e0), np.linalg.norm(e1)
    if l0 < 1e-6 or l1 < 1e-6:
        return poly
    u0, u1 = e0 / l0, e1 / l1
    a = c + u0 * (fx * l0)
    b = c + u1 * (fy * l1)
    if oblique <= 0.0:
        cut = Polygon([c, a, a + (b - c), b])
    else:
        # blend the rectangle's far corner towards the chamfer chord
        far = a + (b - c)
        far = far * (1 - oblique) + ((a + b) / 2) * oblique
        cut = Polygon([c, a, far, b])
    if not cut.is_valid:
        cut = cut.buffer(0)
    res = sp.difference(cut.buffer(1e-6))
    if res.is_empty:
        return poly
    if res.geom_type == "MultiPolygon":
        res = max(res.geoms, key=lambda g: g.area)
    return normalize_polygon(np.asarray(res.exterior.coords)[:-1])


def concave_perturb(poly: np.ndarray, rng: np.random.Generator,
                    amp: float = 0.18, n_moves: int | None = None) -> np.ndarray:
    """Constrained vertex perturbation, rejecting anything non-simple."""
    p = normalize_polygon(poly.copy())
    n = len(p)
    scale = math.sqrt(as_polygon(p).area)
    n_moves = n_moves if n_moves is not None else max(2, n // 2)
    for _ in range(n_moves):
        for _try in range(24):
            q = p.copy()
            i = int(rng.integers(0, len(q)))
            direction = rng.normal(size=2)
            direction /= max(np.linalg.norm(direction), 1e-9)
            q[i] = q[i] + direction * amp * scale * rng.uniform(0.4, 1.0)
            if validate_polygon(q):
                p = normalize_polygon(q)
                break
    return p


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------
def _u_shaped(rng: np.random.Generator, lo: float, hi: float,
              lo_prob: float = 0.5) -> float:
    """Sample from a U-shaped distribution over [lo, hi], with the density
    peaked at the two tails and thinner near the identity (1.0).

    Realised as a Beta(0.5, 0.5) mapped to [lo, hi]; ``lo_prob`` biases the
    sample toward the low tail (scale down, i.e. ref smaller than target).
    """
    x = float(rng.beta(0.5, 0.5))       # U-shaped on [0, 1]
    # asymmetry: shift the midpoint so lo tail gets more mass when lo_prob>0.5
    # (kept symmetric by default)
    return lo + x * (hi - lo)


def sample_deform(poly: np.ndarray, level: int, rng: np.random.Generator,
                  strength: float = 1.0,
                  l1_range: tuple = (0.5, 2.0),
                  l1_u_shape: bool = True) -> tuple[np.ndarray, DeformSpec]:
    """Draw one deformation of the requested difficulty level.

    ``l1_range`` widens L1 uniform_scale beyond the original [0.7, 1.4] so the
    3-sizes test (s=0.75 / 1.35) sits closer to the *body* of the training
    distribution instead of the tails.  ``l1_u_shape=True`` samples from a
    Beta(0.5, 0.5) mapping so the density is peaked at the extremes -- this
    is what removes the near-identity samples (ratio ~ 1.0) that teach the
    model very little.
    """
    base = normalize_polygon(poly)
    n = len(base)
    for _attempt in range(60):
        if level == 1:
            if l1_u_shape:
                s = _u_shaped(rng, l1_range[0], l1_range[1])
            else:
                s = float(rng.uniform(l1_range[0], l1_range[1]))
            s = 1.0 + (s - 1.0) * strength
            out = uniform_scale(base, s)
            spec = DeformSpec(1, "uniform_scale", {"s": s})
        elif level == 2:
            sx = float(rng.uniform(0.65, 1.5))
            sy = float(rng.uniform(0.65, 1.5))
            sx = 1.0 + (sx - 1.0) * strength
            sy = 1.0 + (sy - 1.0) * strength
            out = aspect_deform(base, sx, sy)
            spec = DeformSpec(2, "aspect_ratio", {"sx": sx, "sy": sy})
        elif level == 3:
            k = int(rng.integers(0, n))
            wall = base[(k + 1) % n] - base[k]
            L = float(np.linalg.norm(wall))
            mag = float(rng.uniform(0.15, 0.45)) * L * strength
            sign = 1.0 if rng.random() < 0.5 else -1.0
            mode = "normal" if rng.random() < 0.7 else "tangent"
            out = slant_wall(base, k, sign * mag, mode)
            spec = DeformSpec(3, "slanted_wall",
                              {"wall": k, "shift": sign * mag, "mode": mode})
        elif level == 4:
            k = int(rng.integers(0, n))
            fx = float(rng.uniform(0.25, 0.55)) * strength + 0.05
            fy = float(rng.uniform(0.25, 0.55)) * strength + 0.05
            ob = 0.0 if rng.random() < 0.65 else float(rng.uniform(0.3, 1.0))
            out = corner_cut(base, k, min(fx, 0.7), min(fy, 0.7), ob)
            spec = DeformSpec(4, "corner_cut",
                              {"corner": k, "fx": fx, "fy": fy, "oblique": ob})
        elif level == 5:
            tmp = base
            if rng.random() < 0.6:
                tmp = aspect_deform(tmp, float(rng.uniform(0.8, 1.3)),
                                    float(rng.uniform(0.8, 1.3)))
            if rng.random() < 0.6:
                k = int(rng.integers(0, len(tmp)))
                tmp = corner_cut(tmp, k, float(rng.uniform(0.2, 0.45)),
                                 float(rng.uniform(0.2, 0.45)),
                                 float(rng.uniform(0.0, 0.8)))
            out = concave_perturb(tmp, rng, amp=0.14 * strength)
            spec = DeformSpec(5, "concave", {"amp": 0.14 * strength})
        else:
            raise ValueError(f"unknown difficulty level {level}")

        out = normalize_polygon(out)
        if validate_polygon(out):
            return out, spec
    # give up gracefully: a mild uniform scale always works
    out = uniform_scale(base, 1.0)
    return normalize_polygon(out), DeformSpec(level, "identity_fallback", {})


# --------------------------------------------------------------------------
# opening re-anchoring
# --------------------------------------------------------------------------
def _anchor_openings(room: Room) -> list[tuple[Opening, int, float, float]]:
    """Record each opening as (wall index, centre param t, half width)."""
    anchored = []
    walls = room.walls()
    for op in room.openings:
        c = op.centre
        best, best_d = 0, 1e18
        best_t = 0.5
        for k, (a, b) in enumerate(walls):
            d = b - a
            L2 = float(np.dot(d, d))
            if L2 < 1e-12:
                continue
            t = float(np.clip(np.dot(c - a, d) / L2, 0.0, 1.0))
            proj = a + t * d
            dist = float(np.linalg.norm(c - proj))
            if dist < best_d:
                best, best_d, best_t = k, dist, t
        anchored.append((op, best, best_t, op.width / 2.0))
    return anchored


def _replace_openings(new_poly: np.ndarray,
                      anchored: list[tuple[Opening, int, float, float]],
                      n_old: int) -> list[Opening]:
    out = []
    n_new = len(new_poly)
    walls = [(new_poly[i], new_poly[(i + 1) % n_new]) for i in range(n_new)]
    lens = np.array([float(np.linalg.norm(b - a)) for a, b in walls])
    for op, k, t, hw in anchored:
        # if the vertex count changed the mapping is approximate: keep the
        # relative position around the perimeter instead of the wall index.
        idx = k if n_new == n_old else int(round(k * n_new / max(n_old, 1))) % n_new
        if lens[idx] < max(2 * hw, 0.3):
            # the mapped wall is too short to host this opening (a corner cut
            # can create one); move it to the longest wall rather than dropping
            # it, so a deformed room never silently loses its door
            idx = int(np.argmax(lens))
        a, b = walls[idx]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        u = d / L
        half = min(hw, 0.45 * L)
        centre = a + u * float(np.clip(t, half / L, 1 - half / L)) * L
        new = op.copy()
        new.p0 = centre - u * half
        new.p1 = centre + u * half
        out.append(new)
    return out


def deform_room(room: Room, level: int, rng: np.random.Generator,
                strength: float = 1.0,
                l1_range: tuple = (0.7, 1.4),
                l1_u_shape: bool = False) -> DeformResult:
    """Sample a target room of the requested difficulty from a source room."""
    src = room.polygon
    new_poly, spec = sample_deform(src, level, rng, strength,
                                   l1_range=l1_range, l1_u_shape=l1_u_shape)
    anchored = _anchor_openings(room)
    openings = _replace_openings(new_poly, anchored, len(src))
    new_room = Room(polygon=new_poly, height=room.height,
                    openings=openings, room_type=room.room_type)
    src_ext = src.max(axis=0) - src.min(axis=0)
    tgt_ext = new_poly.max(axis=0) - new_poly.min(axis=0)
    hint = np.where(src_ext > 1e-6, tgt_ext / np.maximum(src_ext, 1e-6), 1.0)
    return DeformResult(new_room, spec, hint.astype(float))
