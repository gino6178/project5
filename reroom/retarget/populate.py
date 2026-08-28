"""Filling a larger target room (plan section 10).

Pulling the furniture apart is the wrong answer.  Core motifs keep their
reference geometry and the extra floor is filled with *complementary* objects
drawn from the reference style and from corpus statistics, until

    rho(S_t) ~= rho(S_r)                                              (29)

while navigation clearance and visual balance are preserved by the optimizer
that runs afterwards.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..core.categories import DECOR_CATEGORIES, prior
from ..core.scene import ObjectInstance, Scene
from ..data.asset_bank import AssetBank
from ..geom.polygon import as_polygon, erode, sample_interior
from .summarize import MAX_DENSITY, area_budget
from .target import DesignIntent

__all__ = ["CooccurrenceModel", "PopulationPlan", "plan_population"]


@dataclass
class CooccurrenceModel:
    """Which categories go with which room type, and in what numbers."""

    counts: dict[str, Counter] = field(default_factory=dict)
    sizes: dict[str, np.ndarray] = field(default_factory=dict)
    n_scenes: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def fit(scenes: list[Scene]) -> "CooccurrenceModel":
        counts: dict[str, Counter] = defaultdict(Counter)
        sizes: dict[str, list] = defaultdict(list)
        n: Counter = Counter()
        for s in scenes:
            rt = s.room.room_type
            n[rt] += 1
            for o in s.objects:
                counts[rt][o.category] += 1
                sizes[o.category].append(o.size)
        return CooccurrenceModel(
            counts={k: v for k, v in counts.items()},
            sizes={k: np.stack(v).mean(0) for k, v in sizes.items()},
            n_scenes=dict(n))

    def candidates(self, room_type: str, present: Counter,
                   top: int = 14) -> list[tuple[str, float]]:
        """Categories worth adding, scored by expected-count deficit."""
        c = self.counts.get(room_type)
        if not c:
            c = Counter()
            for v in self.counts.values():
                c.update(v)
        ns = max(self.n_scenes.get(room_type, 1), 1)
        out = []
        for cat, k in c.items():
            expect = k / ns
            have = present.get(cat, 0)
            deficit = expect - have
            if deficit <= 0.05:
                continue
            out.append((cat, float(deficit)))
        out.sort(key=lambda t: -t[1])
        return out[:top]

    def mean_size(self, category: str) -> np.ndarray:
        return self.sizes.get(category, np.array([0.5, 0.5, 0.7]))


@dataclass
class PopulationPlan:
    additions: list[ObjectInstance] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    target_area: float = 0.0
    current_area: float = 0.0


def _wall_pose(room, rng, half_depth: float):
    """A pose with the object's back against a randomly chosen wall."""
    walls = room.walls()
    if not walls:
        return None
    lens = np.array([float(np.linalg.norm(b - a)) for a, b in walls])
    if lens.sum() <= 1e-6:
        return None
    k = int(rng.choice(len(walls), p=lens / lens.sum()))
    a, b = walls[k]
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return None
    t = d / L
    n = np.array([-t[1], t[0]])
    s = float(rng.uniform(0.12, 0.88))
    xy = a + t * (s * L) + n * (half_depth + 0.04)
    return xy, math.atan2(n[1], n[0]) - math.pi / 2


def plan_population(intent: DesignIntent, target_room, keep: np.ndarray,
                    cooc: CooccurrenceModel | None = None,
                    bank: AssetBank | None = None,
                    rng: np.random.Generator | None = None,
                    max_add: int = 10, fill_tol: float = 0.88
                    ) -> PopulationPlan:
    """Choose complementary objects to bring ``rho(S_t)`` back to ``rho(S_r)``."""
    rng = rng or np.random.default_rng(0)
    scene = intent.source
    objs = scene.objects
    budget = area_budget(intent, target_room)
    cur = float(sum(objs[i].footprint_area for i in range(len(objs)) if keep[i]))
    plan = PopulationPlan(target_area=budget, current_area=cur)
    if cur >= fill_tol * budget or intent.area_ratio < 1.1:
        return plan

    present = Counter(objs[i].category for i in range(len(objs)) if keep[i])
    room_type = scene.room.room_type
    cands = cooc.candidates(room_type, present) if cooc else []
    if not cands:
        # style-preserving default: repeat the reference's own furnishings
        cands = [(c, 1.0) for c in present
                 if prior(c).droppable >= 0.4 and c != "rug"]
    if not cands:
        return plan

    poly = as_polygon(target_room)
    free = erode(poly, 0.35)
    if free.is_empty:
        return plan

    # Prefer complementary objects that live against a wall.  A larger room
    # filled with free-standing clutter in the middle scores well on density
    # and badly on navigability -- measured directly in experiment 1, where
    # naive population cost ~0.29 of the reachable-area ratio.
    weights = np.array([w * (0.35 + 1.25 * prior(c).wall) for c, w in cands],
                       dtype=float)
    weights = weights / weights.sum()
    added = 0
    guard = 0
    while cur < fill_tol * budget and added < max_add and guard < 6 * max_add:
        guard += 1
        cat = cands[int(rng.choice(len(cands), p=weights))][0]
        size = (cooc.mean_size(cat) if cooc else np.array([0.5, 0.5, 0.7])).copy()
        if bank is not None and bank.has(cat):
            hit = bank.retrieve(cat, size, topk=3, rng=rng)
            if hit:
                size = hit[int(rng.integers(0, len(hit)))][0].size.copy()
        a = float(size[0] * size[1])
        if cur + a > budget * 1.05:
            if a > budget * 0.25:
                continue
        pose = _wall_pose(target_room, rng, float(size[1]) / 2) \
            if prior(cat).wall > 0.5 else None
        if pose is None:
            p = sample_interior(free, 1, rng)[0]
            yaw = float(rng.uniform(0, 2 * math.pi))
        else:
            p, yaw = pose
        o = ObjectInstance(
            oid=f"add_{added}_{cat}", category=cat,
            position=np.array([p[0], p[1], 0.0]), yaw=yaw, size=size,
            meta={"added": True, "reason": "density_match",
                  "wall_seeded": pose is not None})
        plan.additions.append(o)
        plan.log.append(f"add {cat} ({a:.2f} m^2) to reach rho={intent.target_density:.2f}")
        cur += a
        added += 1
    plan.current_area = cur
    return plan
