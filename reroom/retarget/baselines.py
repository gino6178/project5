"""Baselines for section 16.1.

The comparison the paper hinges on is

    Normalized Coordinate Scaling   vs.   Relation-Aware Retargeting        (44)

so both are implemented against the identical scene representation, the
identical target rooms and the identical metrics.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.categories import prior
from ..core.scene import ObjectInstance, Room, Scene
from ..geom.polygon import (as_polygon, erode, min_rotated_rect_params,
                            sample_interior)
from ..intent.relations import SceneGraph
from .energy import EnergyWeights
from .optimizer import RetargetConfig, _mrr_frame, retarget

__all__ = ["direct_scaling", "affine_fit", "target_only", "reference_rigid",
           "BASELINES", "run_baseline"]


def _blank(src: Scene, target_room: Room, tag: str) -> Scene:
    return Scene(scene_id=f"{src.scene_id}__{tag}", room=target_room.copy(),
                 objects=[o.copy() for o in src.objects], source=tag,
                 meta={"source_scene": src.scene_id, "method": tag})


def direct_scaling(src: Scene, target_room: Room,
                   scale_objects: bool = False) -> Scene:
    """Normalise source coordinates by the source bbox, paste into the target.

    The naive answer, and the one the plan argues against: it preserves nothing
    that matters (relative distances all stretch by the same factor) and it has
    no notion of the target polygon beyond its bounding box.
    """
    out = _blank(src, target_room, "direct_scaling")
    sb = src.room.bbox
    tb = target_room.bbox
    ss = np.maximum(sb[1] - sb[0], 1e-6)
    ts = tb[1] - tb[0]
    k = ts / ss
    for o in out.objects:
        u = (o.xy - sb[0]) / ss
        o.xy = tb[0] + u * ts
        if scale_objects:
            o.size[0] *= float(k[0])
            o.size[1] *= float(k[1])
    return out


def affine_fit(src: Scene, target_room: Room) -> Scene:
    """Best-fit similarity between the two rooms' minimum rotated rectangles.

    Stronger than direct scaling: it follows the room's orientation, so an
    L-shaped or rotated target does not shear the layout.  Still a pure
    coordinate map -- no selection, no substitution, no feasibility.
    """
    out = _blank(src, target_room, "affine_fit")
    sf = _mrr_frame(as_polygon(src.room))
    tf = _mrr_frame(as_polygon(target_room))
    cs, a1s, a2s, hls, hss, angs = sf
    ct, a1t, a2t, hlt, hst, angt = tf
    dang = angt - angs
    for o in out.objects:
        d = o.xy - cs
        u = np.array([float(np.dot(d, a1s)) / max(hls, 1e-6),
                      float(np.dot(d, a2s)) / max(hss, 1e-6)])
        o.xy = ct + u[0] * hlt * a1t + u[1] * hst * a2t
        o.yaw = o.yaw + dang
    return out


def reference_rigid(src: Scene, target_room: Room) -> Scene:
    """Copy the reference layout unchanged, only re-centred.

    The 'preserve everything, adapt nothing' extreme -- the upper bound on
    relation preservation and (usually) the worst on feasibility.
    """
    out = _blank(src, target_room, "reference_rigid")
    cs = as_polygon(src.room).centroid
    ct = as_polygon(target_room).centroid
    d = np.array([ct.x - cs.x, ct.y - cs.y])
    for o in out.objects:
        o.xy = o.xy + d
    return out


def target_only(src: Scene, graph: SceneGraph, target_room: Room,
                cfg: RetargetConfig | None = None, **kw) -> Scene:
    """Floor-plan-conditioned synthesis that ignores the reference layout.

    Implemented by running the same solver with ``lambda_rel = 0``: the object
    set and the functional priors survive, the reference *arrangement* does
    not.  This isolates 'what does looking at the reference actually buy?'.
    """
    cfg = cfg or RetargetConfig()
    w = EnergyWeights(**{f: getattr(cfg.weights, f)
                         for f in cfg.weights.__dataclass_fields__})
    w.rel = 0.0
    c2 = RetargetConfig(**{f: getattr(cfg, f) for f in cfg.__dataclass_fields__})
    c2.weights = w
    c2.use_motif_init = False
    res = retarget(graph, target_room, cfg=c2, **kw)
    res.scene.source = "target_only"
    res.scene.meta["method"] = "target_only"
    return res.scene


def source_reference(src: Scene, target_room: Room) -> Scene:
    """The reference design in its *own* room -- not a retargeting method.

    Included in every table as the natural reference point: it shows what the
    legality metrics score on professionally designed rooms, so a reader can
    tell how much of the residual clearance or blockage is the method's fault
    and how much is simply what 3D-FRONT looks like.
    """
    out = _blank(src, src.room, "source_reference")
    return out


BASELINES = ("source_reference", "direct_scaling", "affine_fit",
             "reference_rigid", "target_only", "reroom")


def run_baseline(name: str, graph: SceneGraph, target_room: Room,
                 cfg: RetargetConfig | None = None, **kw) -> Scene:
    src = graph.scene
    if name == "source_reference":
        return source_reference(src, target_room)
    if name == "direct_scaling":
        return direct_scaling(src, target_room)
    if name == "direct_scaling_resize":
        return direct_scaling(src, target_room, scale_objects=True)
    if name == "affine_fit":
        return affine_fit(src, target_room)
    if name == "reference_rigid":
        return reference_rigid(src, target_room)
    if name == "target_only":
        return target_only(src, graph, target_room, cfg, **kw)
    if name == "reroom":
        return retarget(graph, target_room, cfg=cfg, **kw).scene
    raise ValueError(f"unknown baseline {name}")
