"""Hierarchical scene motifs (plan section 7).

    M_r = {m_1, ..., m_K}                                             (13)
    m_living = {sofa, coffee table, TV cabinet}                       (14)

A motif is a *functional* group: the unit that must survive retargeting
together.  Deleting a dining table while keeping six chairs is geometrically
legal and semantically absurd, so summarization operates on motifs, and inside
a motif it does *structured* pruning (four chairs -> two) rather than arbitrary
object dropout.

Three levels are produced:

    L0  support groups   -- a lamp on a nightstand moves with the nightstand
    L1  functional motifs -- sleeping / conversation / dining / work ...
    L2  motif-to-motif relations -- 'conversation faces media'
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..core.categories import SEATING_CATEGORIES, prior
from ..core.scene import ObjectInstance, Scene
from .relations import (Relation, SceneGraph, local_offset, pair_gap,
                        relation_features)

__all__ = [
    "Motif", "MotifRelation", "MOTIF_SCHEMAS", "build_motifs",
    "structured_prune", "motif_of", "MotifSchema", "strip_motifs",
]


@dataclass(frozen=True)
class MotifSchema:
    """A named functional group: one head category plus optional members."""

    name: str
    heads: tuple[str, ...]
    members: tuple[str, ...]
    radius: float = 1.6              # metres from the head's footprint
    weight: float = 1.0              # omega_k in eq. (43)
    rigidity: float = 0.7            # 0 = free to rearrange, 1 = move as one
    min_members: dict = field(default_factory=dict, compare=False)


MOTIF_SCHEMAS: tuple[MotifSchema, ...] = (
    MotifSchema("sleeping", ("double_bed", "single_bed", "kids_bed", "bunk_bed"),
                ("nightstand", "table_lamp", "rug", "bench", "stool"),
                radius=1.3, weight=1.0, rigidity=0.9,
                min_members={"nightstand": 1}),
    MotifSchema("dining", ("dining_table",),
                ("dining_chair", "lounge_chair", "armchair", "bench",
                 "pendant_lamp", "rug", "stool", "barstool"),
                radius=1.5, weight=1.0, rigidity=0.95,
                min_members={"dining_chair": 2}),
    MotifSchema("conversation", ("sofa", "l_sofa", "loveseat"),
                ("coffee_table", "armchair", "lounge_chair", "side_table",
                 "floor_lamp", "rug", "stool"),
                radius=2.4, weight=1.0, rigidity=0.55,
                min_members={"armchair": 0, "coffee_table": 1}),
    MotifSchema("media", ("tv_stand",), ("tv", "decoration", "cabinet"),
                radius=1.0, weight=0.9, rigidity=0.95),
    MotifSchema("work", ("desk",), ("office_chair", "lounge_chair", "shelf",
                                    "table_lamp", "decoration"),
                radius=1.2, weight=0.8, rigidity=0.9),
    MotifSchema("dressing", ("dressing_table",),
                ("stool", "mirror", "table_lamp"), radius=1.1,
                weight=0.6, rigidity=0.9),
    MotifSchema("reading", ("armchair", "lounge_chair"),
                ("floor_lamp", "side_table", "rug", "bookcase"),
                radius=1.2, weight=0.5, rigidity=0.6),
    MotifSchema("storage", ("wardrobe", "bookcase", "cabinet", "sideboard",
                            "drawer_chest", "shelf", "wine_cabinet",
                            "shoe_cabinet"),
                ("decoration", "plant", "mirror"),
                radius=0.9, weight=0.7, rigidity=0.85),
    MotifSchema("hearth", ("fireplace",), ("armchair", "rug", "decoration"),
                radius=1.6, weight=0.7, rigidity=0.7),
    MotifSchema("music", ("piano",), ("stool", "bench"), radius=1.0,
                weight=0.6, rigidity=0.95),
)

_HEAD_TO_SCHEMA: dict[str, list[MotifSchema]] = defaultdict(list)
for _s in MOTIF_SCHEMAS:
    for _h in _s.heads:
        _HEAD_TO_SCHEMA[_h].append(_s)


@dataclass
class Motif:
    """One functional group of objects."""

    mid: str
    name: str
    head: int                          # index into scene.objects
    members: list[int]                 # includes the head, head first
    schema: MotifSchema | None = None
    weight: float = 1.0                # omega_k
    rigidity: float = 0.7
    roles: dict[int, str] = field(default_factory=dict)
    support_of: dict[int, int] = field(default_factory=dict)   # member -> base
    keep: bool = True                  # k_{m_k} of eq. (27)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.members)

    def category_groups(self, scene: Scene) -> dict[str, list[int]]:
        g: dict[str, list[int]] = defaultdict(list)
        for i in self.members:
            g[scene.objects[i].category].append(i)
        return dict(g)

    def centroid(self, scene: Scene) -> np.ndarray:
        w = np.array([scene.objects[i].footprint_area for i in self.members])
        p = np.array([scene.objects[i].xy for i in self.members])
        return (p * w[:, None]).sum(0) / max(w.sum(), 1e-9) if len(p) else np.zeros(2)

    def frame(self, scene: Scene) -> tuple[np.ndarray, float]:
        """Motif frame: the head's pose."""
        h = scene.objects[self.head]
        return h.xy.copy(), h.yaw

    def footprint_area(self, scene: Scene) -> float:
        return float(sum(scene.objects[i].footprint_area for i in self.members))

    def min_keep(self, category: str) -> int:
        if self.schema and category in self.schema.min_members:
            return int(self.schema.min_members[category])
        return 1


@dataclass
class MotifRelation:
    """L2: how two motifs relate (e.g. conversation faces media)."""

    a: str
    b: str
    kind: str
    weight: float
    dist: float
    dir_local: np.ndarray              # direction of b in a's frame


def strip_motifs(graph: SceneGraph) -> SceneGraph:
    """A copy of ``graph`` with the whole motif layer removed (section 16.2).

    The plan's central structural claim is that selection and placement have to
    happen at motif level, not object level.  Testing it needs the *flat*
    system: no groups, no motif-to-motif links, no ``grouped_with`` edges, and
    no per-motif rigidity -- only the pairwise relations survive.  Evaluation
    still uses the intact reference graph, so ``S_motif`` continues to measure
    the functional groups the flat system was never told about.
    """
    import copy

    out = copy.copy(graph)
    # the scene must be copied too: a shallow copy shares the object list, and
    # clearing the motif tags below would then silently damage the intact graph
    # that every other arm of the comparison is still using
    out.scene = graph.scene.copy()
    out.motifs = []
    out.relations = [copy.copy(r) for r in graph.relations
                     if r.kind != "grouped_with"
                     and not r.meta.get("motif_link")]
    for r in out.relations:
        r.meta = dict(r.meta)
        for k in ("motif_i", "motif_j", "same_motif", "rigidity"):
            r.meta.pop(k, None)
    for o in out.scene.objects:
        if "motif" in o.meta:
            o.meta = {k: v for k, v in o.meta.items() if k != "motif"}
    return out


# --------------------------------------------------------------------------
def build_motifs(graph: SceneGraph, attach_leftovers: bool = True) -> SceneGraph:
    """Cluster objects into motifs and annotate the graph in place."""
    scene = graph.scene
    objs = scene.objects
    n = len(objs)
    assigned: dict[int, str] = {}
    motifs: list[Motif] = []

    # ---- L0: support chains (a lamp on a nightstand follows its base) ----
    support_base: dict[int, int] = {}
    for r in graph.relations:
        if r.kind == "support":
            support_base[r.j] = r.i

    def _root(i: int) -> int:
        seen = set()
        while i in support_base and i not in seen:
            seen.add(i)
            i = support_base[i]
        return i

    # ---- L1: schema-driven greedy grouping, strongest anchors first ----
    order = sorted(range(n), key=lambda i: (-prior(objs[i].category).anchor,
                                            -objs[i].footprint_area))
    for i in order:
        if i in assigned:
            continue
        cat = objs[i].category
        schemas = _HEAD_TO_SCHEMA.get(cat)
        if not schemas:
            continue
        schema = schemas[0]
        head = objs[i]
        members = [i]
        roles = {i: "head"}
        for j in order:
            if j == i or j in assigned:
                continue
            b = objs[j]
            if _root(j) in members and _root(j) != j:
                members.append(j)
                roles[j] = "support"
                continue
            if b.category not in schema.members:
                continue
            gap = pair_gap(head, b)
            if gap > schema.radius:
                continue
            if prior(b.category).anchor >= 0.9 and b.category in _HEAD_TO_SCHEMA:
                continue                       # do not swallow another anchor
            members.append(j)
            roles[j] = "member"
        mid = f"m{len(motifs)}_{schema.name}"
        for j in members:
            assigned[j] = mid
        motifs.append(Motif(mid=mid, name=schema.name, head=i, members=members,
                            schema=schema, weight=schema.weight,
                            rigidity=schema.rigidity, roles=roles,
                            support_of={j: support_base[j] for j in members
                                        if j in support_base}))

    # ---- leftovers: proximity clusters, then singletons ----
    if attach_leftovers:
        left = [i for i in range(n) if i not in assigned]
        # pull supported objects into their base's motif
        for i in list(left):
            base = _root(i)
            if base != i and base in assigned:
                mid = assigned[base]
                m = next(mm for mm in motifs if mm.mid == mid)
                m.members.append(i)
                m.roles[i] = "support"
                m.support_of[i] = support_base[i]
                assigned[i] = mid
                left.remove(i)
        clusters = _proximity_clusters([objs[i] for i in left], left, thresh=0.55)
        for cl in clusters:
            head = max(cl, key=lambda i: prior(objs[i].category).anchor * 10
                       + objs[i].footprint_area)
            name = "cluster" if len(cl) > 1 else objs[head].category
            mid = f"m{len(motifs)}_{name}"
            for j in cl:
                assigned[j] = mid
            w = float(np.mean([prior(objs[j].category).anchor for j in cl])) * 0.8 + 0.1
            motifs.append(Motif(mid=mid, name=name, head=head, members=list(cl),
                                schema=None, weight=w, rigidity=0.4,
                                roles={j: ("head" if j == head else "member") for j in cl}))

    # head first in the member list, and stable ordering afterwards
    for m in motifs:
        rest = sorted(set(m.members) - {m.head})
        m.members = [m.head] + rest

    graph.motifs = motifs
    _add_grouped_with(graph)
    _add_motif_links(graph)
    graph.scene.meta["motif_relations"] = [
        {"a": r.a, "b": r.b, "kind": r.kind, "weight": r.weight,
         "dist": r.dist, "dir_local": r.dir_local.tolist()}
        for r in motif_relations(graph)
    ]
    return graph


def _proximity_clusters(objs: list[ObjectInstance], idx: list[int],
                        thresh: float = 0.55) -> list[list[int]]:
    """Single-link clustering on footprint gap."""
    n = len(objs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        for b in range(a + 1, n):
            if pair_gap(objs[a], objs[b]) <= thresh:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    groups: dict[int, list[int]] = defaultdict(list)
    for a in range(n):
        groups[find(a)].append(idx[a])
    return list(groups.values())


def _add_grouped_with(graph: SceneGraph) -> None:
    from .relations import RELATION_WEIGHT
    objs = graph.scene.objects
    have = {(r.i, r.j, r.kind) for r in graph.relations}
    for m in graph.motifs:
        for a_pos in range(len(m.members)):
            for b_pos in range(a_pos + 1, len(m.members)):
                i, j = sorted((m.members[a_pos], m.members[b_pos]))
                if (i, j, "grouped_with") in have:
                    continue
                w = RELATION_WEIGHT["grouped_with"] * (0.5 + m.rigidity)
                graph.relations.append(
                    Relation(i, j, "grouped_with", w,
                             relation_features(objs[i], objs[j]),
                             {"motif": m.mid}))


_LINK_KIND = {"face_to_face": "face_to_face", "faces": "facing",
              "beside": "aligned", "near": "near"}


def _add_motif_links(graph: SceneGraph, max_dist: float = 14.0) -> None:
    """Put the L2 motif-to-motif relations into ``E_rel`` (plan section 7).

    Without these there is nothing in the energy holding two *groups* apart.
    A dining area and a seating area six metres apart in the reference share no
    edge -- the pairwise builder's distance cutoff and its degree pruning both
    drop them -- so the optimiser is free to slide the groups together, and it
    does: measured on a room scaled 1.6x, the two groups collapsed from six
    metres apart to under two.  These are exactly the long relations that
    relation elasticity is supposed to govern, so without them alpha has
    nothing to act on either.
    """
    from .relations import RELATION_WEIGHT, Relation, relation_features
    objs = graph.scene.objects
    have = {(r.i, r.j, r.kind) for r in graph.relations}
    for mr in motif_relations(graph, max_dist=max_dist):
        a = next((m for m in graph.motifs if m.mid == mr.a), None)
        b = next((m for m in graph.motifs if m.mid == mr.b), None)
        if a is None or b is None:
            continue
        i, j = a.head, b.head
        if i == j:
            continue
        kind = _LINK_KIND.get(mr.kind, "near")
        key = (i, j, kind)
        if key in have or (j, i, kind) in have:
            continue
        w = RELATION_WEIGHT.get(kind, 0.8) * (0.7 + 0.6 * mr.weight)
        graph.relations.append(
            Relation(i, j, kind, w, relation_features(objs[i], objs[j]),
                     {"motif_link": True, "motif_a": mr.a, "motif_b": mr.b}))


def motif_of(graph: SceneGraph, i: int) -> Motif | None:
    for m in graph.motifs:
        if i in m.members:
            return m
    return None


def motif_relations(graph: SceneGraph, max_dist: float = 6.0) -> list[MotifRelation]:
    """L2 relations between motif frames."""
    scene = graph.scene
    out: list[MotifRelation] = []
    ms = graph.motifs
    for a in range(len(ms)):
        for b in range(a + 1, len(ms)):
            ma, mb = ms[a], ms[b]
            pa, ya = ma.frame(scene)
            pb, yb = mb.frame(scene)
            d = pb - pa
            dist = float(np.linalg.norm(d))
            if dist > max_dist or dist < 1e-6:
                continue
            right = np.array([math.cos(ya), math.sin(ya)])
            fwd = np.array([-math.sin(ya), math.cos(ya)])
            dl = np.array([float(np.dot(d, right)), float(np.dot(d, fwd))]) / dist
            ha, hb = scene.objects[ma.head], scene.objects[mb.head]
            kind = "near"
            w = 0.6
            if float(np.dot(d / dist, ha.forward)) > 0.75:
                kind, w = "faces", 1.5
                if float(np.dot(-d / dist, hb.forward)) > 0.75:
                    kind, w = "face_to_face", 2.0
            elif abs(float(np.dot(d / dist, ha.right))) > 0.9:
                kind, w = "beside", 0.8
            out.append(MotifRelation(ma.mid, mb.mid, kind, w, dist, dl))
    return out


# --------------------------------------------------------------------------
def structured_prune(motif: Motif, scene: Scene, drop_budget: int,
                     symmetric_pairs: list[tuple[int, int]] | None = None
                     ) -> list[int]:
    """Choose which members of a motif to delete, structurally.

    Repeated categories are thinned first (4 dining chairs -> 2), symmetric
    pairs are dropped as pairs so the result stays balanced, and the head is
    never dropped.  Returns the object indices to remove.
    """
    if drop_budget <= 0:
        return []
    sym = {}
    for a, b in (symmetric_pairs or []):
        sym[a] = b
        sym[b] = a
    groups = motif.category_groups(scene)
    removed: list[int] = []
    # order categories by how droppable they are
    cats = sorted(groups.keys(), key=lambda c: -prior(c).droppable)
    for cat in cats:
        if len(removed) >= drop_budget:
            break
        idxs = [i for i in groups[cat] if i != motif.head and i not in removed]
        if not idxs:
            continue
        floor = motif.min_keep(cat)
        # farthest-from-head first: the outermost repeat is the least missed
        h = scene.objects[motif.head]
        idxs.sort(key=lambda i: -float(np.linalg.norm(scene.objects[i].xy - h.xy)))
        while idxs and len(groups[cat]) - len([r for r in removed
                                               if scene.objects[r].category == cat]) > floor:
            if len(removed) >= drop_budget:
                break
            victim = idxs.pop(0)
            removed.append(victim)
            partner = sym.get(victim)
            if partner is not None and partner in idxs and partner != motif.head:
                remaining = len(groups[cat]) - len(
                    [r for r in removed if scene.objects[r].category == cat])
                if remaining - 1 >= floor:
                    idxs.remove(partner)
                    removed.append(partner)
    return removed
