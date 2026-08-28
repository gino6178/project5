"""Vision-language extraction of *semantic* relations (plan section 20).

The plan puts a hard bound on this stage.  Its risk table lists "LLM/VLM
relation extraction unstable" and prescribes the mitigation directly:

    use deterministic geometry + category rules; the VLM only supplies
    semantic relations

So this module never overrides geometry.  ``build_scene_graph`` decides where
things are -- distances, facing, alignment, support -- and what is added here is
only the class of relation a metric tape cannot settle: whether two pieces read
as *belonging together*.  Two nightstands either side of a bed are a matching
pair; a nightstand and a random side table at the same distance are not, and no
amount of geometry separates those two cases.

The backend is CLIP, which is already in the pipeline for eq. (30) and is a
vision-language model in the only sense that matters here: it scores a rendered
crop against natural-language hypotheses.  A crop containing both objects is
compared against one phrase per relation kind plus an explicit *unrelated*
null hypothesis, and a relation is emitted only when it beats the null by a
margin.  ``vlm_agreement`` then measures how often that judgement matches the
geometric one on the relations both can see, which is the number the plan's
risk entry is really asking for.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ..core.scene import Scene
from .relations import Relation, SceneGraph, relation_features

__all__ = ["SEMANTIC_PROMPTS", "VLMRelation", "extract_semantic_relations",
           "augment_with_vlm", "vlm_agreement", "object_boxes_from_seg"]


# One phrase per hypothesis.  ``null`` is not a relation -- it is the escape
# hatch that keeps the extractor from labelling every pair it is shown.
SEMANTIC_PROMPTS: dict[str, str] = {
    "grouped_with": "a photo of a {a} and a {b} that belong to the same "
                    "furniture set, arranged as one group",
    "symmetric": "a photo of a matching pair, a {a} and a {b}, placed "
                 "symmetrically on either side",
    "facing": "a photo of a {a} turned to face a {b}",
    "null": "a photo of a {a} and a {b} that have nothing to do with each "
            "other, in unrelated parts of a room",
}

# only these are ever written into the graph; `facing` and the rest stay with
# geometry, which measures them directly and far more reliably
SEMANTIC_KINDS = ("grouped_with", "symmetric")


@dataclass
class VLMRelation:
    i: int
    j: int
    kind: str
    score: float                       # softmax probability of the winner
    margin: float                      # winner minus the null hypothesis
    meta: dict = field(default_factory=dict)


def object_boxes_from_seg(seg_path: str, label_to_oid: dict
                          ) -> dict[str, tuple[int, int, int, int]]:
    """Per-object pixel boxes from a rendered instance mask."""
    from PIL import Image

    m = np.asarray(Image.open(seg_path))
    if m.ndim == 3:
        m = m[..., 0]
    out = {}
    for lab, oid in label_to_oid.items():
        ys, xs = np.nonzero(m == int(lab))
        if len(xs) < 16:
            continue
        out[oid] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return out


def _crop(img, box, pad: float = 0.12):
    x0, y0, x1, y1 = box
    w, h = img.size
    px, py = (x1 - x0) * pad + 4, (y1 - y0) * pad + 4
    return img.crop((max(0, x0 - px), max(0, y0 - py),
                     min(w, x1 + px), min(h, y1 + py)))


def extract_semantic_relations(scene: Scene, rgb_path: str, seg_path: str,
                               label_to_oid: dict, encoder=None,
                               max_dist: float = 3.5,
                               min_margin: float = 0.04,
                               max_pairs: int = 120) -> list[VLMRelation]:
    """Score every plausible pair against the prompt set above."""
    from PIL import Image

    if encoder is None:
        from ..eval.appearance import ClipEncoder
        encoder = ClipEncoder()
    if not getattr(encoder, "ok", False):
        return []
    if not (os.path.exists(rgb_path) and os.path.exists(seg_path)):
        return []

    boxes = object_boxes_from_seg(seg_path, label_to_oid)
    if len(boxes) < 2:
        return []
    img = Image.open(rgb_path).convert("RGB")
    idx = {o.oid: k for k, o in enumerate(scene.objects)}

    pairs, crops = [], []
    oids = [o for o in boxes if o in idx]
    for a in range(len(oids)):
        for b in range(a + 1, len(oids)):
            oa, ob = oids[a], oids[b]
            ia, ib = idx[oa], idx[ob]
            if float(np.linalg.norm(scene.objects[ia].xy
                                    - scene.objects[ib].xy)) > max_dist:
                continue
            ba, bb = boxes[oa], boxes[ob]
            union = (min(ba[0], bb[0]), min(ba[1], bb[1]),
                     max(ba[2], bb[2]), max(ba[3], bb[3]))
            if (union[2] - union[0]) < 24 or (union[3] - union[1]) < 24:
                continue
            pairs.append((ia, ib))
            crops.append(_crop(img, union))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        return []

    feats = encoder.encode_images(crops)
    if feats is None:
        return []

    kinds = list(SEMANTIC_PROMPTS)
    # the prompts depend only on the category pair, and a room has far fewer
    # distinct category pairs than object pairs -- encoding text per object pair
    # was the whole runtime
    want = sorted({(scene.objects[ia].category, scene.objects[ib].category)
                   for ia, ib in pairs})
    flat = [SEMANTIC_PROMPTS[k].format(a=a.replace("_", " "),
                                       b=b.replace("_", " "))
            for a, b in want for k in kinds]
    tmat = encoder.encode_text(flat)
    if tmat is None:
        return []
    tmat = tmat.reshape(len(want), len(kinds), -1)
    tpos = {cp: k for k, cp in enumerate(want)}
    raw = np.zeros((len(pairs), len(kinds)), dtype=np.float32)
    for n, (ia, ib) in enumerate(pairs):
        cp = (scene.objects[ia].category, scene.objects[ib].category)
        raw[n] = feats[n] @ tmat[tpos[cp]].T

    # Prompt calibration.  Raw CLIP similarities carry a large per-phrase bias
    # -- one wording simply sits higher than another regardless of the picture,
    # and uncalibrated the extractor answered "symmetric" for every pair it was
    # shown.  Subtracting each prompt's own mean over the scene's pairs turns
    # the decision into "which hypothesis does *this* crop favour relative to
    # the others", which is what was wanted in the first place.
    sims = raw - raw.mean(axis=0, keepdims=True)

    out: list[VLMRelation] = []
    null = kinds.index("null")
    for n, (ia, ib) in enumerate(pairs):
        row = sims[n]
        p = np.exp((row - row.max()) * 100.0)        # CLIP's own logit scale
        p /= p.sum()
        w = int(np.argmax(p))
        kind = kinds[w]
        margin = float(p[w] - p[null])
        if kind == "null" or kind not in SEMANTIC_KINDS or margin < min_margin:
            continue
        # a category rule, which the plan explicitly keeps on the deterministic
        # side: a *matching pair* is two of the same thing.  Without it CLIP
        # cheerfully calls a wardrobe and a desk a symmetric pair.
        if kind == "symmetric" and (scene.objects[ia].category
                                    != scene.objects[ib].category):
            continue
        ca = scene.objects[ia].category.replace("_", " ")
        cb = scene.objects[ib].category.replace("_", " ")
        out.append(VLMRelation(ia, ib, kind, float(p[w]), margin,
                               {"category_a": ca, "category_b": cb}))
    return out


def augment_with_vlm(graph: SceneGraph, vlm_rels: list[VLMRelation],
                     weight_scale: float = 0.6) -> SceneGraph:
    """Add VLM relations that geometry did not already find.

    Never replaces or reweights an existing edge: if the deterministic rules
    already produced this pair and kind, theirs is the one that stands.  The
    added edges carry ``meta['source'] = 'vlm'`` and a weight scaled by the
    model's own confidence, so a shaky judgement moves the layout less.
    """
    have = {(r.i, r.j, r.kind) for r in graph.relations}
    have |= {(r.j, r.i, r.kind) for r in graph.relations}
    objs = graph.scene.objects
    added = 0
    for v in vlm_rels:
        if (v.i, v.j, v.kind) in have:
            continue
        graph.relations.append(Relation(
            v.i, v.j, v.kind, float(weight_scale * v.score),
            relation_features(objs[v.i], objs[v.j]),
            {"source": "vlm", "vlm_score": v.score, "vlm_margin": v.margin,
             **v.meta}))
        have.add((v.i, v.j, v.kind))
        added += 1
    graph.meta = {**getattr(graph, "meta", {}), "vlm_added": added}
    return graph


def vlm_agreement(graph: SceneGraph, vlm_rels: list[VLMRelation]) -> dict:
    """How often the VLM and the deterministic rules agree, per pair.

    Geometry is the reference here, not the truth: the point of the number is
    the plan's risk entry -- if the two disagree wildly, the VLM cannot be
    trusted with anything beyond the supplementary role it is given.
    """
    geo = {}
    for r in graph.relations:
        if r.kind in SEMANTIC_KINDS:
            geo[(min(r.i, r.j), max(r.i, r.j))] = r.kind
    vl = {(min(v.i, v.j), max(v.i, v.j)): v.kind for v in vlm_rels}
    both = set(geo) & set(vl)
    agree = sum(1 for k in both if geo[k] == vl[k])
    return {"n_geometric": len(geo), "n_vlm": len(vl),
            "n_overlap": len(both),
            "agreement": float(agree / len(both)) if both else float("nan"),
            "precision": float(len(both) / len(vl)) if vl else float("nan"),
            "recall": float(len(both) / len(geo)) if geo else float("nan")}
