"""Furniture asset bank for style-aware retrieval (plan section 11).

    a*_i = argmin_{a_j in A_ci} [ lf Df(f^r_i, f^3D_j) + ls Ds(s^req_i, s_j) ]  (30)

The point of retrieval rather than free rescaling: when the reference sofa is
too big for the target room, ReRoom does not squash it to 70 % of its size --
it fetches a stylistically similar sofa that is genuinely smaller.

Two backends implement the same interface:

``FutureBank``      real 3D-FUTURE assets with CLIP embeddings of their renders.
``StatisticalBank`` category size statistics harvested from a scene corpus,
                    used when 3D-FUTURE is not on disk; retrieval then returns
                    a size-matched pseudo-asset so the pipeline still runs.
"""
from __future__ import annotations

import json
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..core.categories import canonical_category
from ..core.scene import Scene

__all__ = ["Asset", "AssetBank", "StatisticalBank", "FutureBank",
           "load_bank", "merge_banks"]


@dataclass
class Asset:
    aid: str
    category: str
    size: np.ndarray                    # (3,) full extents, metres
    raw_category: str | None = None
    style: str | None = None
    theme: str | None = None
    material: str | None = None
    image: str | None = None            # path to a render/product photo
    embedding: np.ndarray | None = None  # f^3D_j
    shape: np.ndarray | None = None      # f^geo_j, eq. (10)
    source: str = "unknown"

    def size_distance(self, req: np.ndarray) -> float:
        """``Ds`` of eq. (30): scale-free size mismatch, in log space."""
        a = np.maximum(self.size, 1e-3)
        b = np.maximum(np.asarray(req, dtype=float), 1e-3)
        return float(np.abs(np.log(a) - np.log(b)).mean())


class AssetBank:
    """Category-indexed asset store with the retrieval objective of eq. (30)."""

    def __init__(self, assets: list[Asset]):
        self.assets = assets
        self.by_category: dict[str, list[int]] = defaultdict(list)
        for k, a in enumerate(assets):
            self.by_category[a.category].append(k)
        self._emb_cache: dict[str, np.ndarray] = {}
        self._shape_lut: dict[str, np.ndarray] | None = None
        # descriptors for models that are *not* retrievable candidates -- a
        # reference object may well be one of the implausible-mesh assets that
        # `_drop_implausible` removed, and it still needs its own f^geo
        self.extra_shapes: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.assets)

    def categories(self) -> list[str]:
        return sorted(self.by_category.keys())

    def shape_of(self, aid: str | None) -> np.ndarray | None:
        """``f^geo`` of an asset by id -- how a reference object that came from
        this same catalogue gets its descriptor without re-reading its mesh."""
        if aid is None:
            return None
        if self._shape_lut is None:
            self._shape_lut = dict(self.extra_shapes)
            self._shape_lut.update({a.aid: a.shape for a in self.assets
                                    if a.shape is not None})
        return self._shape_lut.get(aid)

    def has(self, category: str) -> bool:
        return len(self.by_category.get(category, ())) > 0

    # -- statistics ------------------------------------------------------
    def size_stats(self, category: str) -> dict:
        idx = self.by_category.get(category, [])
        if not idx:
            return {"mean": np.array([0.6, 0.6, 0.7]), "std": np.array([0.1, 0.1, 0.1]),
                    "min": np.array([0.3, 0.3, 0.3]), "max": np.array([1.0, 1.0, 1.0]),
                    "n": 0}
        s = np.stack([self.assets[i].size for i in idx])
        return {"mean": s.mean(0), "std": s.std(0) + 1e-3,
                "min": np.percentile(s, 5, axis=0), "max": np.percentile(s, 95, axis=0),
                "n": len(idx)}

    def _category_embeddings(self, category: str) -> np.ndarray | None:
        """Unit-norm embeddings for a category, with a NaN row where an asset
        has none.

        This used to disable the whole category the moment a *single* asset was
        missing an embedding, which quietly turned eq. (30) into pure size
        matching for any catalogue that is not embedded end to end.  Missing
        rows are marked instead, and ``retrieve`` gives them the category's
        mean appearance distance so they are neither favoured nor punished.
        """
        if category in self._emb_cache:
            return self._emb_cache[category]
        idx = self.by_category.get(category, [])
        embs = [self.assets[i].embedding for i in idx]
        if not idx or all(e is None for e in embs):
            self._emb_cache[category] = None
            return None
        dim = next(len(e) for e in embs if e is not None)
        m = np.stack([np.full(dim, np.nan, dtype=np.float32) if e is None
                      else np.asarray(e, dtype=np.float32) for e in embs])
        m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        self._emb_cache[category] = m
        return m

    # -- retrieval -------------------------------------------------------
    def retrieve(self, category: str, req_size: np.ndarray,
                 ref_embedding: np.ndarray | None = None,
                 lambda_f: float = 1.0, lambda_s: float = 1.0,
                 topk: int = 1, exclude: set[str] | None = None,
                 max_size: np.ndarray | None = None,
                 min_size: np.ndarray | None = None,
                 rng: np.random.Generator | None = None,
                 ref_shape: np.ndarray | None = None,
                 lambda_g: float = 0.6
                 ) -> list[tuple[Asset, float]]:
        """Eq. (30).  Returns ``[(asset, cost), ...]`` sorted by cost."""
        idx = list(self.by_category.get(category, []))
        if not idx:
            return []
        if exclude:
            idx = [i for i in idx if self.assets[i].aid not in exclude] or idx
        sizes = np.stack([self.assets[i].size for i in idx])
        req = np.maximum(np.asarray(req_size, dtype=float), 1e-3)
        ds = np.abs(np.log(np.maximum(sizes, 1e-3)) - np.log(req)).mean(1)
        if max_size is not None:
            over = np.maximum(sizes[:, :2] - np.asarray(max_size)[:2], 0.0).sum(1)
            ds = ds + 4.0 * over
        if min_size is not None:
            # symmetric to ``max_size``: reject a degenerate too-small asset (a
            # mislabelled 0.4 m double bed) so shrink-to-fit stays physical
            under = np.maximum(np.asarray(min_size)[:2] - sizes[:, :2], 0.0).sum(1)
            ds = ds + 4.0 * under
        df = np.zeros_like(ds)
        if ref_embedding is not None:
            embs = self._category_embeddings(category)
            if embs is not None:
                pos = {a: k for k, a in enumerate(self.by_category[category])}
                sel = np.array([pos[i] for i in idx])
                e = np.asarray(ref_embedding, dtype=np.float32)
                e = e / max(float(np.linalg.norm(e)), 1e-9)
                df = 1.0 - embs[sel] @ e
                bad = ~np.isfinite(df)
                if bad.any():
                    df[bad] = float(np.mean(df[~bad])) if (~bad).any() else 0.0
        dg = np.zeros_like(ds)
        if ref_shape is not None and lambda_g > 0:
            # f^geo: same category, same size, same style -- but is it the same
            # *shape*?  Candidates without a descriptor score 0 and are neither
            # rewarded nor punished for it.
            from ..perception.geometry import shape_distance
            dg = np.array([shape_distance(ref_shape, self.assets[i].shape)
                           for i in idx])
        cost = lambda_f * df + lambda_s * ds + lambda_g * dg
        order = np.argsort(cost)[:max(topk, 1)]
        return [(self.assets[idx[o]], float(cost[o])) for o in order]

    # -- io --------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"assets": self.assets, "extra_shapes": self.extra_shapes},
                        fh)

    @staticmethod
    def load(path: str) -> "AssetBank":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        if isinstance(d, list):                 # banks written before f^geo
            return AssetBank(d)
        b = AssetBank(d["assets"])
        b.extra_shapes = d.get("extra_shapes", {})
        return b


class StatisticalBank(AssetBank):
    """Pseudo-assets sampled from per-category size statistics of a corpus."""

    @staticmethod
    def from_scenes(scenes: list[Scene], per_category: int = 48,
                    seed: int = 0, source: str = "statistical",
                    style_from=None) -> "StatisticalBank":
        rng = np.random.default_rng(seed)
        buckets: dict[str, list[np.ndarray]] = defaultdict(list)
        styles: dict[str, str] = {}
        for s in scenes:
            st = style_from(s) if style_from else None
            for o in s.objects:
                buckets[o.category].append(o.size.copy())
                if st and o.category not in styles:
                    styles[o.category] = st[:80]
        assets: list[Asset] = []
        for cat, sizes in buckets.items():
            arr = np.stack(sizes)
            if len(arr) >= per_category:
                sel = rng.choice(len(arr), size=per_category, replace=False)
                chosen = arr[sel]
            else:
                # bootstrap with log-normal jitter so the bank spans the range
                reps = int(math.ceil(per_category / len(arr)))
                chosen = np.concatenate([arr] * reps)[:per_category]
                chosen = chosen * np.exp(rng.normal(0, 0.10, size=chosen.shape))
            for k, sz in enumerate(chosen):
                assets.append(Asset(aid=f"{source}_{cat}_{k}", category=cat,
                                    size=np.asarray(sz, dtype=float),
                                    style=styles.get(cat),
                                    source=source))
        b = StatisticalBank(assets)
        return b


class FutureBank(AssetBank):
    """3D-FUTURE assets: real meshes, categories, style tags and product images."""

    @staticmethod
    def from_dir(root, model_info: str | None = None,
                 bbox_cache: str | None = None) -> "FutureBank":
        """Build from one or more extracted ``3D-FUTURE-model`` directories.

        The release is split across four archives but every one of them ships
        the *complete* ``model_info.json``, so the categories come from any of
        them while meshes and product images have to be looked up across all
        the roots that were actually extracted.
        """
        roots = [root] if isinstance(root, (str, os.PathLike)) else list(root)
        model_info = model_info or next(
            (os.path.join(r, "model_info.json") for r in roots
             if os.path.exists(os.path.join(r, "model_info.json"))),
            os.path.join(roots[0], "model_info.json"))
        with open(model_info) as fh:
            info = json.load(fh)
        bboxes: dict[str, list[float]] = {}
        if bbox_cache and os.path.exists(bbox_cache):
            with open(bbox_cache) as fh:
                bboxes = json.load(fh)
        assets = []
        for rec in info:
            mid = rec.get("model_id")
            if mid is None:
                continue
            raw = rec.get("category") or ""
            sup = rec.get("super-category") or rec.get("super_category") or ""
            cat = canonical_category(raw, sup)
            bb = bboxes.get(mid)
            if bb is None:
                continue
            bb = np.asarray(bb, dtype=float)
            # the cache stores [min xyz, max xyz] in the model's own frame;
            # the bank wants full extents, and 3D-FUTURE is y-up while ReRoom
            # is z-up, so (x, y, z)_model -> (x, z, y)_reroom
            ext = np.abs(bb[3:] - bb[:3])
            size = np.array([ext[0], ext[2], ext[1]])
            img = next((os.path.join(r, mid, "image.jpg") for r in roots
                        if os.path.exists(os.path.join(r, mid, "image.jpg"))),
                       None)
            assets.append(Asset(
                aid=mid, category=cat, size=np.asarray(size, dtype=float),
                raw_category=raw, style=rec.get("style"), theme=rec.get("theme"),
                material=rec.get("material"),
                image=img,
                source="3D-FUTURE"))
        assets = _drop_implausible(assets)
        return FutureBank(assets)

    def attach_embeddings(self, emb_path: str) -> "FutureBank":
        d = np.load(emb_path, allow_pickle=True)
        ids = list(d["ids"])
        mat = d["emb"]
        lut = {a: k for k, a in enumerate(ids)}
        for a in self.assets:
            k = lut.get(a.aid)
            if k is not None:
                a.embedding = mat[k]
        self._emb_cache.clear()
        return self

    def attach_shapes(self, shape_path: str) -> "FutureBank":
        """Attach ``f^geo`` (eq. 10) computed by ``build_future_shapes.py``."""
        with np.load(shape_path) as z:
            have = set(z.files)
            for a in self.assets:
                if a.aid in have:
                    a.shape = z[a.aid].astype(np.float32)
            mine = {a.aid for a in self.assets}
            self.extra_shapes = {k: z[k].astype(np.float32)
                                 for k in have if k not in mine}
        self._shape_lut = None
        return self


def _drop_implausible(assets: list[Asset], abs_max: float = 5.0,
                      rel_max: float = 4.0) -> list[Asset]:
    """Discard assets whose mesh bounds are not a plausible piece of furniture.

    A handful of 3D-FUTURE meshes carry stray far-away vertices, which turns a
    dining chair into an 8-metre box.  They are rare (<1 % of models) but they
    poison per-category size statistics and would let retrieval "solve" a
    too-large object by proposing something absurd, so they are filtered
    against both an absolute cap and their own category's median.
    """
    from collections import defaultdict
    by_cat: dict[str, list[Asset]] = defaultdict(list)
    for a in assets:
        by_cat[a.category].append(a)
    out = []
    for cat, group in by_cat.items():
        med = np.median(np.stack([a.size for a in group]), axis=0)
        med = np.maximum(med, 0.05)
        for a in group:
            if float(a.size.max()) > abs_max:
                continue
            if float((a.size / med).max()) > rel_max:
                continue
            if float(a.size.min()) < 0.02:
                continue
            out.append(a)
    return out


def merge_banks(base: AssetBank, extra: AssetBank,
                only_new_categories: bool = False,
                min_existing: int = 12) -> AssetBank:
    """Fold a second bank into the first (plan section 17).

    SAGE-10k is the intended caller: the plan assigns it *object diversity and
    open-vocabulary augmentation*, so what it usefully contributes is the
    categories 3D-FUTURE either lacks entirely or covers with a handful of
    models.  Categories the base bank already covers well are left alone, since
    a real mesh with a real product image beats a size sampled from statistics.
    """
    keep = []
    for a in extra.assets:
        n = len(base.by_category.get(a.category, ()))
        if n == 0 or (not only_new_categories and n < min_existing):
            keep.append(a)
    out = AssetBank(base.assets + keep)
    out.extra_shapes = dict(base.extra_shapes)
    out.extra_shapes.update(extra.extra_shapes)
    return out


def load_bank(path: str | None, scenes: list[Scene] | None = None) -> AssetBank:
    if path and os.path.exists(path):
        return AssetBank.load(path)
    if scenes:
        return StatisticalBank.from_scenes(scenes)
    return AssetBank([])
