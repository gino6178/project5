"""Stage I: reference scene understanding (plan section 6).

    v_i = (e^sem_i, b_i, f^app_i, f^geo_i, q_i)                        (10)

The plan is explicit that parser error and retargeting error must not be
conflated: experiments run first on the 3D-FRONT ground-truth graph, and only
then on an image-derived one, so the gap

    I_r -> G^_r -> S_t     vs.     G^GT_r -> S_t                    (38), (39)

isolates perception from retargeting.  This module defines the interface all
source parsers implement and provides two of them: an oracle over the
ground-truth scene, and a *calibrated noisy* oracle that injects controlled
detection, category, pose and size error so the sensitivity of the retargeting
stage to perception quality can be measured without waiting on a real parser.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.categories import PRIORS, canonical_category, prior
from ..core.scene import ObjectInstance, Room, Scene

__all__ = ["SourceNode", "ParsedScene", "SourceParser", "OracleParser",
           "NoisyOracleParser", "PerceptionNoise"]


@dataclass
class SourceNode:
    """One parsed object node, eq. (10)."""

    sem: str                              # e^sem: canonical category
    box: np.ndarray                       # b_i: (centre xyz, size xyz, yaw)
    app: np.ndarray | None = None         # f^app
    geo: np.ndarray | None = None         # f^geo
    conf: float = 1.0                     # q_i
    oid: str | None = None
    jid: str | None = None


@dataclass
class ParsedScene:
    """What a source parser returns: a scene plus per-node confidence."""

    scene: Scene
    nodes: list[SourceNode] = field(default_factory=list)
    room_confidence: float = 1.0
    meta: dict = field(default_factory=dict)


class SourceParser:
    """``I_r -> G^_r``.  Implementations: oracle, noisy oracle, MIDI."""

    name = "base"

    def parse(self, source) -> ParsedScene:      # pragma: no cover
        raise NotImplementedError


class OracleParser(SourceParser):
    """Ground-truth 3D-FRONT graph -- the setting of experiment one."""

    name = "oracle"

    def __init__(self, bank=None):
        # with the 3D-FUTURE bank in hand the oracle can also hand back f^geo,
        # which is what a perfect parser would report: the shape of the actual
        # asset, not just its box
        self.bank = bank

    def parse(self, source: Scene) -> ParsedScene:
        shape_of = getattr(self.bank, "shape_of", None) if self.bank else None
        nodes = [SourceNode(sem=o.category,
                            box=np.concatenate([o.position, o.size, [o.yaw]]),
                            geo=(shape_of(o.jid) if shape_of else None),
                            conf=1.0, oid=o.oid, jid=o.jid)
                 for o in source.objects]
        return ParsedScene(scene=source.copy(), nodes=nodes,
                           meta={"parser": self.name})


@dataclass
class PerceptionNoise:
    """Calibrated error budget for the noisy oracle.

    Defaults are set to a plausible single-image parser operating point: a few
    per cent of objects missed or hallucinated, ~10 % of categories confused
    within their functional group, ~8 cm of translation error, ~6 degrees of
    yaw error and ~8 % of size error.
    """

    miss_rate: float = 0.10
    hallucination_rate: float = 0.04
    category_error: float = 0.10
    translation_std: float = 0.08
    yaw_std_deg: float = 6.0
    size_log_std: float = 0.08
    yaw_flip_rate: float = 0.04            # front/back confusion
    room_scale_log_std: float = 0.04       # metric-scale ambiguity
    seed: int = 0


# categories a parser plausibly confuses with one another
_CONFUSION = [
    ("double_bed", "single_bed", "kids_bed"),
    ("sofa", "l_sofa", "loveseat"),
    ("armchair", "lounge_chair", "office_chair", "dining_chair"),
    ("cabinet", "sideboard", "drawer_chest", "shoe_cabinet", "wine_cabinet"),
    ("bookcase", "shelf", "cabinet"),
    ("coffee_table", "side_table", "console_table"),
    ("dining_table", "desk", "dressing_table"),
    ("floor_lamp", "plant", "decoration"),
    ("table_lamp", "decoration"),
    ("tv_stand", "sideboard"),
]


class NoisyOracleParser(SourceParser):
    """The ground-truth graph, degraded by a controlled error budget.

    This is what makes experiment three interpretable *before* a real
    image parser is wired in: sweeping the noise level traces how retargeting
    quality degrades with perception quality, and where the real parser sits on
    that curve is then a single measurement.
    """

    name = "noisy_oracle"

    def __init__(self, noise: PerceptionNoise | None = None):
        self.noise = noise or PerceptionNoise()

    def parse(self, source: Scene) -> ParsedScene:
        nz = self.noise
        rng = np.random.default_rng(nz.seed)
        out = source.copy()
        out.scene_id = f"{source.scene_id}__noisy"
        out.source = "noisy_oracle"

        # metric-scale ambiguity: a single image fixes shape better than size
        s = float(np.exp(rng.normal(0, nz.room_scale_log_std)))
        out.room.polygon = out.room.polygon * s
        for op in out.room.openings:
            op.p0 *= s
            op.p1 *= s

        kept = []
        nodes = []
        for o in out.objects:
            if rng.random() < nz.miss_rate and prior(o.category).anchor < 0.85:
                continue
            o.position[:2] = o.position[:2] * s + rng.normal(0, nz.translation_std, 2)
            o.position[2] *= s
            o.yaw += math.radians(rng.normal(0, nz.yaw_std_deg))
            if rng.random() < nz.yaw_flip_rate:
                o.yaw += math.pi
            o.size = o.size * s * np.exp(rng.normal(0, nz.size_log_std, 3))
            conf = 1.0
            if rng.random() < nz.category_error:
                for group in _CONFUSION:
                    if o.category in group:
                        alt = [c for c in group if c != o.category]
                        o.category = alt[int(rng.integers(0, len(alt)))]
                        conf = 0.55
                        break
            kept.append(o)
            nodes.append(SourceNode(sem=o.category,
                                    box=np.concatenate([o.position, o.size, [o.yaw]]),
                                    geo=o.meta.get("shape"),
                                    conf=conf, oid=o.oid, jid=o.jid))

        # hallucinations: plausible small objects in plausible places
        n_hall = int(rng.poisson(nz.hallucination_rate * max(len(kept), 1)))
        for k in range(n_hall):
            if not kept:
                break
            proto = kept[int(rng.integers(0, len(kept)))]
            cat = ["decoration", "plant", "side_table", "stool"][
                int(rng.integers(0, 4))]
            o = ObjectInstance(
                oid=f"hall_{k}", category=cat,
                position=proto.position + np.array([*rng.normal(0, 0.6, 2), 0.0]),
                yaw=float(rng.uniform(0, 2 * math.pi)),
                size=np.array([0.4, 0.4, 0.5]) * np.exp(rng.normal(0, 0.2, 3)),
                meta={"hallucinated": True})
            kept.append(o)
            nodes.append(SourceNode(sem=cat,
                                    box=np.concatenate([o.position, o.size, [o.yaw]]),
                                    conf=0.35, oid=o.oid))
        out.objects = kept
        return ParsedScene(scene=out, nodes=nodes,
                           room_confidence=float(np.exp(-nz.room_scale_log_std)),
                           meta={"parser": self.name, "scale_error": s,
                                 "n_missed": len(source.objects) - len(kept) + n_hall,
                                 "n_hallucinated": n_hall})
