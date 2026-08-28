"""Appearance / style similarity -- the auxiliary metric of section 15.2.

The plan is explicit that this score "cannot replace relation/motif
evaluation, because a global CLIP-like similarity is easily dominated by colour
or by the largest object".  It is therefore computed and reported, but never
folded into the headline score, and two variants are given:

``global``          CLIP similarity between renders of the reference and the
                    retargeted scene from matched canonical views;
``object_matched``  mean CLIP similarity between each reference object's asset
                    image and the asset actually placed for it -- the number
                    that actually reflects style-aware retrieval (eq. 30), and
                    the only one that is meaningful when the two rooms have
                    different shapes and therefore different camera framings.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

from ..core.scene import Scene
from ..data.asset_bank import AssetBank

__all__ = ["ClipEncoder", "appearance_similarity", "object_matched_similarity"]


class ClipEncoder:
    """Lazy CLIP image encoder; a no-op if open_clip is unavailable."""

    def __init__(self, model: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k", device: str = "cpu"):
        self.ok = False
        try:
            import open_clip
            import torch
            self.torch = torch
            self.device = torch.device(device)
            self.model, _, self.pre = open_clip.create_model_and_transforms(
                model, pretrained=pretrained)
            self.model = self.model.to(self.device).eval()
            self.tokenizer = open_clip.get_tokenizer(model)
            self.ok = True
        except Exception as exc:                     # pragma: no cover
            self.error = str(exc)

    def encode_paths(self, paths: list[str], batch: int = 32) -> np.ndarray | None:
        if not self.ok or not paths:
            return None
        from PIL import Image
        torch = self.torch
        out = []
        for k in range(0, len(paths), batch):
            imgs = []
            for p in paths[k:k + batch]:
                try:
                    imgs.append(self.pre(Image.open(p).convert("RGB")))
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
            x = torch.stack(imgs).to(self.device)
            with torch.no_grad():
                f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            out.append(f.cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)


    def encode_images(self, images, batch: int = 32) -> np.ndarray | None:
        """Same as ``encode_paths`` but for PIL images already in memory."""
        if not self.ok or not images:
            return None
        torch = self.torch
        out = []
        for k in range(0, len(images), batch):
            x = torch.stack([self.pre(im.convert("RGB"))
                             for im in images[k:k + batch]]).to(self.device)
            with torch.no_grad():
                f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            out.append(f.cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)

    def encode_text(self, prompts: list[str]) -> np.ndarray | None:
        """Text side of the same embedding space -- what makes CLIP usable as
        the *vision-language* model of section 20, not just an image encoder."""
        if not self.ok or not prompts:
            return None
        torch = self.torch
        with torch.no_grad():
            t = self.tokenizer(prompts).to(self.device)
            f = self.model.encode_text(t)
        f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return f.cpu().numpy().astype(np.float32)


def _render_views(scene: Scene, tmpdir: str, tag: str, views) -> list[str]:
    from ..render.scene3d import render_scene3d
    paths = []
    for k, v in enumerate(views):
        p = os.path.join(tmpdir, f"{tag}_{k}.png")
        render_scene3d(scene, p, view=v, figsize=3.0, dpi=110)
        paths.append(p)
    return paths


def appearance_similarity(reference: Scene, target: Scene,
                          encoder: ClipEncoder | None = None,
                          views=None) -> dict:
    """Global CLIP similarity between matched canonical renders."""
    from ..render.scene3d import CANONICAL_VIEWS
    views = views or CANONICAL_VIEWS
    if encoder is None or not encoder.ok:
        return {"appearance_global": float("nan")}
    with tempfile.TemporaryDirectory() as td:
        a = encoder.encode_paths(_render_views(reference, td, "ref", views))
        b = encoder.encode_paths(_render_views(target, td, "tgt", views))
    if a is None or b is None:
        return {"appearance_global": float("nan")}
    per_view = float(np.mean(np.sum(a * b, axis=1)))
    pooled_a = a.mean(0)
    pooled_b = b.mean(0)
    pooled = float(pooled_a @ pooled_b /
                   max(np.linalg.norm(pooled_a) * np.linalg.norm(pooled_b), 1e-9))
    return {"appearance_global": per_view, "appearance_pooled": pooled}


def object_matched_similarity(reference: Scene, target: Scene,
                              bank: AssetBank | None = None,
                              encoder: ClipEncoder | None = None) -> dict:
    """Mean CLIP similarity between each reference asset and its stand-in.

    Objects that were kept unchanged score 1 by construction; the number is
    informative exactly to the extent that substitution happened, so the count
    of substituted objects is reported alongside it.
    """
    tmap = {o.oid: o for o in target.objects if o.keep}
    pairs = []
    same = 0
    for o in reference.objects:
        t = tmap.get(o.oid)
        if t is None:
            continue
        if t.jid == o.jid or t.jid is None or o.jid is None:
            same += 1
            continue
        pairs.append((o.jid, t.jid))
    if not pairs:
        return {"appearance_object": 1.0 if same else float("nan"),
                "n_appearance_pairs": 0}
    if bank is None:
        return {"appearance_object": float("nan"), "n_appearance_pairs": len(pairs)}
    lut = {a.aid: a for a in bank.assets}
    embs: dict[str, np.ndarray] = {}
    need = []
    for aid in {x for p in pairs for x in p}:
        a = lut.get(aid)
        if a is None:
            continue
        if a.embedding is not None:
            embs[aid] = a.embedding
        elif a.image and os.path.exists(a.image):
            need.append(aid)
    if need and encoder is not None and encoder.ok:
        # only assets whose embedding was not precomputed need the encoder
        mat = encoder.encode_paths([lut[a].image for a in need])
        if mat is not None:
            for aid, e in zip(need, mat):
                embs[aid] = e
    sims = []
    for x, y in pairs:
        ex, ey = embs.get(x), embs.get(y)
        if ex is None or ey is None:
            continue
        sims.append(float(ex @ ey /
                          max(np.linalg.norm(ex) * np.linalg.norm(ey), 1e-9)))
    if not sims:
        return {"appearance_object": float("nan"), "n_appearance_pairs": len(pairs)}
    # unchanged objects contribute a perfect match
    total = (sum(sims) + same) / (len(sims) + same)
    return {"appearance_object": float(total), "n_appearance_pairs": len(sims)}
