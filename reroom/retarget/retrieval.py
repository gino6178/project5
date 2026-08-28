"""Style-aware furniture retrieval (plan section 11), eq. (30)."""
from __future__ import annotations

import numpy as np

from ..core.scene import ObjectInstance, Scene
from ..data.asset_bank import AssetBank
from ..geom.polygon import as_polygon, largest_inscribed_circle
from .target import DesignIntent

__all__ = ["substitute_assets", "retrieval_cost"]


def retrieval_cost(bank: AssetBank, o: ObjectInstance, req_size: np.ndarray,
                   ref_emb: np.ndarray | None, lambda_f: float, lambda_s: float):
    hits = bank.retrieve(o.category, req_size, ref_emb, lambda_f, lambda_s,
                         topk=1, ref_shape=o.meta.get("shape"))
    return hits[0] if hits else None


def substitute_assets(scene: Scene, intent: DesignIntent, bank: AssetBank,
                      lambda_f: float = 1.0, lambda_s: float = 1.0,
                      lambda_g: float = 0.6,
                      shrink_only: bool = False, max_rel_change: float = 0.45,
                      rng: np.random.Generator | None = None) -> Scene:
    """Replace objects whose reference size no longer suits the target room.

    The *requested* size is the reference size scaled by how much slack the
    target room has; retrieval then finds the closest-looking real asset at
    that size, instead of non-physically rescaling the reference one.
    """
    rng = rng or np.random.default_rng(0)
    if len(bank) == 0:
        return scene
    poly = as_polygon(scene.room)
    _, inr = largest_inscribed_circle(poly, tol=0.05)
    # Section 10 is explicit that a bigger room is filled by *adding* furniture,
    # not by stretching what is there -- and upsizing turns out to violate that
    # in the same way spreading does.  Asking for larger assets when the room
    # grew cost 0.041 of legality and pushed clearance violation from 0.058 to
    # 0.098 in rooms above 1.3x, because bigger furniture eats exactly the
    # circulation space the extra floor was supposed to provide.  Substitution
    # may therefore fetch a better-fitting *smaller* asset when the room
    # shrinks; growth is population's job.
    src_scale = float(np.clip(np.sqrt(max(intent.area_ratio, 1e-6)), 0.6, 1.0))

    for o in scene.objects:
        if not o.keep or o.meta.get("added"):
            continue
        req = o.size.copy()
        # only the footprint follows the room; height is a product property
        req[:2] = req[:2] * src_scale
        if shrink_only:
            req[:2] = np.minimum(req[:2], o.size[:2])
        lo = o.size[:2] * (1.0 - max_rel_change)
        hi = o.size[:2] * (1.0 + max_rel_change)
        req[:2] = np.clip(req[:2], lo, hi)
        need = float(np.abs(np.log(np.maximum(req[:2], 1e-3))
                            - np.log(np.maximum(o.size[:2], 1e-3))).mean())
        if need < 0.05 or not bank.has(o.category):
            continue
        hit = bank.retrieve(o.category, req, o.meta.get("ref_embedding"),
                            lambda_f, lambda_s, topk=1,
                            exclude={o.jid} if o.jid else None,
                            max_size=np.array([2.2 * inr, 2.2 * inr]),
                            min_size=lo,
                            ref_shape=(o.meta.get("shape")
                                       if o.meta.get("shape") is not None
                                       else bank.shape_of(o.jid)),
                            lambda_g=lambda_g)
        if not hit:
            continue
        asset, cost = hit[0]
        # hard floor: never accept an asset whose footprint collapses below the
        # allowed shrink (``lo`` = reference * (1 - max_rel_change)).  The soft
        # ``min_size`` penalty ranks these last; this rejects the case where the
        # whole category is degenerate and a tiny asset still wins.
        if float(np.prod(asset.size[:2])) < 0.9 * float(np.prod(lo)):
            continue
        gain = float(np.abs(np.log(np.maximum(o.size[:2], 1e-3))
                            - np.log(np.maximum(req[:2], 1e-3))).mean())
        new_gap = float(np.abs(np.log(np.maximum(asset.size[:2], 1e-3))
                               - np.log(np.maximum(req[:2], 1e-3))).mean())
        if new_gap >= gain * 0.9:
            continue                     # substitution would not actually help
        o.meta["substituted_from"] = o.jid
        o.meta["style_cost"] = float(cost)
        o.jid = asset.aid
        o.size = asset.size.copy()
        if asset.style:
            o.style = asset.style
    return scene
