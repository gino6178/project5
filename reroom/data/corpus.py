"""Streaming access to a parsed scene corpus."""
from __future__ import annotations

import glob
import gzip
import json
import os
import random
from typing import Iterator

from ..core.scene import Scene, scene_from_dict

__all__ = ["iter_scenes", "load_scenes", "corpus_index", "split_scenes"]


def corpus_index(root: str) -> dict:
    p = os.path.join(root, "index.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def iter_scenes(root: str, room_types=None, limit: int | None = None,
                min_objects: int = 0) -> Iterator[Scene]:
    n = 0
    for shard in sorted(glob.glob(os.path.join(root, "scenes_*.jsonl.gz"))):
        with gzip.open(shard, "rt") as fh:
            for line in fh:
                d = json.loads(line)
                if room_types and d["room"]["room_type"] not in room_types:
                    continue
                if len(d.get("objects", ())) < min_objects:
                    continue
                yield scene_from_dict(d)
                n += 1
                if limit and n >= limit:
                    return


def load_scenes(root: str, room_types=None, limit: int | None = None,
                min_objects: int = 0) -> list[Scene]:
    return list(iter_scenes(root, room_types, limit, min_objects))


def split_scenes(scenes: list[Scene], val_frac: float = 0.1,
                 test_frac: float = 0.1, seed: int = 0):
    """House-disjoint split so the same apartment never spans two splits."""
    houses = sorted({s.meta.get("house", s.scene_id) for s in scenes})
    rng = random.Random(seed)
    rng.shuffle(houses)
    n = len(houses)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = set(houses[:n_val])
    test = set(houses[n_val:n_val + n_test])
    tr, va, te = [], [], []
    for s in scenes:
        h = s.meta.get("house", s.scene_id)
        (va if h in val else te if h in test else tr).append(s)
    return tr, va, te
