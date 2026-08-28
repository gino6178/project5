"""Object importance for scene summarization (plan section 9).

    zeta_i = b1 zeta^sem + b2 zeta^centrality + b3 zeta^visual + b4 zeta^motif  (26)

Core furniture (bed, sofa, dining table) scores high; decoration, repeated
seating and auxiliary storage score low.  The scores feed two decisions:

* which *motifs* to keep when the target room shrinks, ``k_{m_k}`` of eq. (27);
* inside a kept motif, which repeats to prune structurally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.categories import prior
from ..core.scene import Scene
from .motifs import Motif
from .relations import SceneGraph

__all__ = ["ImportanceWeights", "object_importance", "motif_importance",
           "removal_order"]


@dataclass(frozen=True)
class ImportanceWeights:
    semantic: float = 0.40
    centrality: float = 0.20
    visual: float = 0.20
    motif: float = 0.20


DEFAULT_WEIGHTS = ImportanceWeights()


def _pagerank(adj: np.ndarray, damping: float = 0.85, iters: int = 60) -> np.ndarray:
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0)
    deg = adj.sum(1, keepdims=True)
    deg[deg < 1e-9] = 1.0
    P = adj / deg
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r = (1 - damping) / n + damping * (P.T @ r)
    return r


def object_importance(graph: SceneGraph,
                      weights: ImportanceWeights = DEFAULT_WEIGHTS) -> np.ndarray:
    """``zeta_i`` for every object, normalised to [0, 1]."""
    scene = graph.scene
    objs = scene.objects
    n = len(objs)
    if n == 0:
        return np.zeros(0)

    # semantic: how strongly the category defines the room's function
    sem = np.array([prior(o.category).anchor for o in objs])

    # centrality: PageRank over the weighted design-intent graph
    adj = graph.adjacency()
    cen = _pagerank(adj)
    cen = cen / max(cen.max(), 1e-9)

    # visual: footprint area, height and proximity to the room's centre
    areas = np.array([o.footprint_area for o in objs])
    hs = np.array([o.height for o in objs])
    c = scene.room.centroid
    diag = max(float(np.linalg.norm(scene.room.extent)), 1e-6)
    dist = np.array([float(np.linalg.norm(o.xy - c)) for o in objs]) / diag
    vis = (0.55 * areas / max(areas.max(), 1e-9)
           + 0.25 * hs / max(hs.max(), 1e-9)
           + 0.20 * (1.0 - np.clip(dist, 0, 1)))

    # motif: heads matter more than members, and a big motif's head more still
    mot = np.zeros(n)
    for m in graph.motifs:
        share = m.weight
        for i in m.members:
            role = m.roles.get(i, "member")
            base = 1.0 if role == "head" else (0.45 if role == "member" else 0.30)
            mot[i] = max(mot[i], share * base)
    if mot.max() > 1e-9:
        mot = mot / mot.max()

    w = weights
    z = (w.semantic * sem + w.centrality * cen + w.visual * vis + w.motif * mot)
    z = z / max(z.max(), 1e-9)
    return z


def motif_importance(graph: SceneGraph, zeta: np.ndarray | None = None
                     ) -> dict[str, float]:
    """Aggregate importance of each motif -- what eq. (27) selects on."""
    if zeta is None:
        zeta = object_importance(graph)
    out: dict[str, float] = {}
    for m in graph.motifs:
        if not m.members:
            out[m.mid] = 0.0
            continue
        head = zeta[m.head]
        rest = [zeta[i] for i in m.members if i != m.head]
        agg = 0.72 * head + 0.28 * (float(np.mean(rest)) if rest else head)
        out[m.mid] = float(agg * (0.55 + 0.45 * m.weight))
    return out


def removal_order(graph: SceneGraph, zeta: np.ndarray | None = None
                  ) -> list[int]:
    """Objects sorted by how happily they can be deleted (most first).

    Combines low importance with the category's ``droppable`` prior, so a
    decorative vase goes before a second armchair, which goes before a bed.
    """
    scene = graph.scene
    if zeta is None:
        zeta = object_importance(graph)
    score = []
    for i, o in enumerate(scene.objects):
        p = prior(o.category)
        score.append((0.65 * (1.0 - zeta[i]) + 0.35 * p.droppable, i))
    score.sort(key=lambda t: -t[0])
    return [i for _, i in score]
