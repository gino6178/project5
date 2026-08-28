"""MIDI adapter: single image -> compositional 3D instances (plan section 6).

MIDI (Multi-Instance Diffusion, CVPR 2025) produces multiple 3D instances from
one RGB image while preserving their compositional relations, which is exactly
the source-parser role ReRoom needs.  Running it requires its own checkpoint and
environment, so this module is an *adapter*, not a reimplementation: it converts
whatever MIDI (or any comparable parser) wrote to disk into a ReRoom scene, and
reports clearly when the artefacts are absent instead of silently degrading.

Expected on-disk format (one JSON per reference image)::

    {"room": {"polygon": [[x, y], ...], "height": 2.8},
     "objects": [{"category": "sofa", "position": [x, y, z], "yaw": 0.0,
                  "size": [sx, sy, sz], "score": 0.9,
                  "embedding": [...]}, ...]}

``scripts/prepare_midi_inputs.py`` writes reference crops in the layout MIDI
expects; ``from_midi_dir`` reads the results back.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

from ..core.categories import canonical_category
from ..core.scene import ObjectInstance, Room, Scene
from ..geom.polygon import normalize_polygon, polygon_from_extent
from .base import ParsedScene, SourceNode, SourceParser

__all__ = ["MIDIAdapter", "GenReconAdapter", "from_midi_json"]


def from_midi_json(path: str, scene_id: str | None = None,
                   min_score: float = 0.25) -> ParsedScene:
    with open(path) as fh:
        d = json.load(fh)
    r = d.get("room", {})
    poly = r.get("polygon")
    if poly:
        polygon = normalize_polygon(np.asarray(poly, dtype=float))
    else:
        # no floor estimate: fall back to the instances' own footprint bounds
        pts = []
        for o in d.get("objects", []):
            p = np.asarray(o["position"], dtype=float)[:2]
            s = np.asarray(o["size"], dtype=float)[:2]
            pts.extend([p - s, p + s])
        pts = np.asarray(pts) if pts else np.array([[-2.0, -2.0], [2.0, 2.0]])
        lo, hi = pts.min(0) - 0.4, pts.max(0) + 0.4
        polygon = normalize_polygon(np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                                              [hi[0], hi[1]], [lo[0], hi[1]]]))
    room = Room(polygon=polygon, height=float(r.get("height", 2.8)),
                room_type=r.get("room_type", "other"))
    objs, nodes = [], []
    for k, o in enumerate(d.get("objects", [])):
        score = float(o.get("score", 1.0))
        if score < min_score:
            continue
        cat = canonical_category(o.get("category"), o.get("super_category"))
        inst = ObjectInstance(
            oid=o.get("id", f"midi_{k}"), category=cat,
            position=np.asarray(o["position"], dtype=float),
            yaw=float(o.get("yaw", 0.0)),
            size=np.asarray(o["size"], dtype=float),
            raw_category=o.get("category"),
            meta={"score": score,
                  "ref_embedding": (np.asarray(o["embedding"], dtype=np.float32)
                                    if o.get("embedding") is not None else None),
                  "shape": (np.asarray(o["shape"], dtype=np.float32)
                            if o.get("shape") is not None else None)})
        objs.append(inst)
        emb = inst.meta.get("ref_embedding")
        nodes.append(SourceNode(
            sem=cat, box=np.concatenate([inst.position, inst.size, [inst.yaw]]),
            app=emb, geo=inst.meta.get("shape"), conf=score, oid=inst.oid))
    scene = Scene(scene_id=scene_id or os.path.splitext(os.path.basename(path))[0],
                  room=room, objects=objs, source="midi",
                  meta={"parser": "midi", "src_json": path})
    return ParsedScene(scene=scene, nodes=nodes, meta={"parser": "midi"})


class MIDIAdapter(SourceParser):
    """Read MIDI outputs produced outside this environment."""

    name = "midi"

    def __init__(self, root: str, min_score: float = 0.25):
        self.root = root
        self.min_score = min_score

    def available(self) -> bool:
        return bool(glob.glob(os.path.join(self.root, "*.json")))

    def keys(self) -> list[str]:
        return sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(self.root, "*.json")))

    def parse(self, source) -> ParsedScene:
        key = source if isinstance(source, str) else getattr(source, "scene_id", None)
        path = os.path.join(self.root, f"{key}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no MIDI output for '{key}' in {self.root}. Run MIDI on the "
                f"reference images first (see scripts/prepare_midi_inputs.py); "
                f"ReRoom will not silently substitute a different parser.")
        return from_midi_json(path, scene_id=key, min_score=self.min_score)


class GenReconAdapter(MIDIAdapter):
    """The multi-view parser of plan section 3.3.

    ``scripts/genrecon_to_reroom.py`` writes the same JSON schema, so reading
    it needs nothing new -- but it is a *separate* parser with a separate error
    profile, and experiment three compares them, so it gets its own name rather
    than being quietly folded into the MIDI arm.
    """

    name = "genrecon"

    def parse(self, source) -> ParsedScene:
        key = source if isinstance(source, str) else getattr(source, "scene_id", None)
        path = os.path.join(self.root, f"{key}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no GenRecon output for '{key}' in {self.root}. Run "
                f"scripts/run_genrecon.sh and then scripts/genrecon_to_reroom.py.")
        out = from_midi_json(path, scene_id=key, min_score=self.min_score)
        out.scene.source = "genrecon"
        out.scene.meta["parser"] = out.meta["parser"] = self.name
        return out
