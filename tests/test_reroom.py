"""Self-contained test suite. Run: python tests/test_reroom.py

Uses the procedural generator, so it needs no dataset. The interesting checks
are the invariants that are easy to break silently: that the exact energy and
its differentiable surrogate agree, that curriculum deformations always yield
valid rooms, and that retargeting a room into itself is a fixed point.
"""
from __future__ import annotations

import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from reroom.core.categories import canonical_category, prior
from reroom.core.scene import ObjectInstance, Room, Scene
from reroom.data.procedural import generate_dataset, generate_scene
from reroom.eval.metrics import evaluate
from reroom.geom.deform import LEVEL_NAMES, deform_room, validate_polygon
from reroom.geom.freespace import build_freespace, clearance_violation
from reroom.geom.polygon import (as_polygon, characteristic_scale,
                                 floor_descriptor, object_polygon,
                                 out_of_bounds_area, sat_separation)
from reroom.intent.elasticity import (PriorElasticity, RelationContext,
                                      StatElasticity, collect_elasticity_samples,
                                      desired_distance)
from reroom.intent.importance import object_importance, removal_order
from reroom.intent.motifs import build_motifs, structured_prune
from reroom.intent.relations import (build_scene_graph, relation_distance,
                                     relation_features)
from reroom.retarget.baselines import run_baseline
from reroom.retarget.energy import EnergyWeights, TorchProblem, exact_energy
from reroom.retarget.optimizer import RetargetConfig, retarget
from reroom.retarget.target import build_design_intent

_FAILED = []


def check(name):
    def deco(fn):
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            _FAILED.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    return deco


def _scene(rt="bedroom", seed=1):
    s = generate_scene(rt, seed=seed)
    return s, build_motifs(build_scene_graph(s))


print("geometry")


@check("footprint corners and area agree")
def _():
    o = ObjectInstance("o", "sofa", [1.0, 2.0, 0.0], 0.7, [2.0, 0.9, 0.8])
    assert abs(object_polygon(o).area - o.footprint_area) < 1e-6
    assert abs(np.linalg.norm(o.forward) - 1) < 1e-9
    assert abs(float(np.dot(o.forward, o.right))) < 1e-9


@check("SAT separation matches shapely for disjoint axis-aligned boxes")
def _():
    a = ObjectInstance("a", "misc", [0, 0, 0], 0.0, [1.0, 1.0, 1.0])
    b = ObjectInstance("b", "misc", [2.5, 0, 0], 0.0, [1.0, 1.0, 1.0])
    assert abs(sat_separation(a, b) - 1.5) < 1e-6
    b.xy = [0.5, 0.0]
    assert sat_separation(a, b) < 0        # overlapping


@check("out-of-bounds area is exact for a half-outside box")
def _():
    room = Room(polygon=np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float))
    o = ObjectInstance("o", "misc", [4.0, 2.0, 0.0], 0.0, [2.0, 1.0, 1.0])
    assert abs(out_of_bounds_area(o, as_polygon(room)) - 1.0) < 1e-6


@check("characteristic scale reads the room extent along a direction")
def _():
    poly = as_polygon(np.array([[0, 0], [6, 0], [6, 3], [0, 3]], float))
    assert abs(characteristic_scale(poly, np.array([1.0, 0.0])) - 6.0) < 1e-6
    assert abs(characteristic_scale(poly, np.array([0.0, 1.0])) - 3.0) < 1e-6


@check("floor descriptor separates a rectangle from an L-shape")
def _():
    rect = floor_descriptor(as_polygon(np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float)))
    ell = floor_descriptor(as_polygon(np.array(
        [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], float)))
    assert rect[7] > 0.99 and ell[7] < 0.9        # convexity
    assert rect[10] == 0 and ell[10] > 0          # reflex fraction


print("\ncurriculum")


@check("all five deformation levels produce valid rooms")
def _():
    rng = np.random.default_rng(0)
    s, _ = _scene("living_room", 3)
    for lvl in range(1, 6):
        for _ in range(12):
            r = deform_room(s.room, lvl, rng)
            assert validate_polygon(r.room.polygon), LEVEL_NAMES[lvl]
            assert r.room.area > 3.0
            assert len(r.room.openings) == len(s.room.openings)


@check("corner cut actually introduces concavity")
def _():
    rng = np.random.default_rng(4)
    s, _ = _scene("bedroom", 2)
    hits = 0
    for _ in range(24):
        r = deform_room(s.room, 4, rng)
        if floor_descriptor(as_polygon(r.room))[7] < 0.995:
            hits += 1
    assert hits >= 12, hits


print("\ndesign intent")


@check("category mapping handles 3D-FUTURE strings")
def _():
    assert canonical_category("King-size Bed", "Bed") == "double_bed"
    assert canonical_category("Three-seat / Multi-seat Sofa", "Sofa") == "sofa"
    assert canonical_category("Corner/Side Table") == "side_table"
    assert canonical_category("lounge chair/cafe chair/office chair") == "office_chair"
    assert prior("double_bed").anchor > prior("decoration").anchor


@check("bed motif captures its nightstands and the dining motif its chairs")
def _():
    s, g = _scene("bedroom", 5)
    sleep = [m for m in g.motifs if m.name == "sleeping"]
    assert sleep, [m.name for m in g.motifs]
    cats = [s.objects[i].category for i in sleep[0].members]
    assert "double_bed" in cats
    s2, g2 = _scene("dining_room", 5)
    din = [m for m in g2.motifs if m.name == "dining"]
    assert din and sum(1 for i in din[0].members
                       if s2.objects[i].category == "dining_chair") >= 3


@check("structured pruning drops chairs, never the table")
def _():
    s, g = _scene("dining_room", 5)
    din = [m for m in g.motifs if m.name == "dining"][0]
    victims = structured_prune(din, s, 2)
    assert victims
    assert all(s.objects[i].category != "dining_table" for i in victims)
    assert din.head not in victims


@check("importance ranks core furniture above decoration")
def _():
    s, g = _scene("living_room", 5)
    z = object_importance(g)
    cats = {s.objects[i].category: z[i] for i in range(len(s.objects))}
    if "sofa" in cats and "plant" in cats:
        assert cats["sofa"] > cats["plant"]
    order = [s.objects[i].category for i in removal_order(g, z)]
    assert order.index("sofa") > 0 if "sofa" in order else True


@check("elasticity: eq.(9) endpoints behave")
def _():
    assert abs(desired_distance(2.0, 0.0, 1.5) - 2.0) < 1e-9     # rigid
    assert abs(desired_distance(2.0, 1.0, 1.5) - 3.0) < 1e-9     # fully elastic
    p = PriorElasticity()
    rigid = p.alpha(RelationContext("dining_table", "dining_chair", "near",
                                    d_ref=0.5))
    elastic = p.alpha(RelationContext("sofa", "tv_stand", "face_to_face",
                                      d_ref=3.0))
    assert rigid < 0.2 < elastic


@check("statistical elasticity recovers a planted slope")
def _():
    rng = np.random.default_rng(0)
    ctxs = []
    for _ in range(600):
        g = float(rng.uniform(2.5, 7.0))
        ctxs.append(RelationContext("sofa", "tv_stand", "facing",
                                    d_ref=0.4 * g * float(np.exp(rng.normal(0, .05))),
                                    gamma_src_abs=g))
        ctxs.append(RelationContext("dining_table", "dining_chair", "near",
                                    d_ref=0.5 * float(np.exp(rng.normal(0, .05))),
                                    gamma_src_abs=g))
    m = StatElasticity(min_samples=50).fit(ctxs)
    a_el = m.pair_alpha["sofa|tv_stand|facing"][0]
    a_ri = m.pair_alpha["dining_chair|dining_table|near"][0]
    assert a_el > 0.8, a_el
    assert a_ri < 0.2, a_ri


@check("relation distance separates direction, orientation and scale")
def _():
    a = ObjectInstance("a", "sofa", [0, 0, 0], 0.0, [2.0, 1.0, 0.8])
    b = ObjectInstance("b", "tv_stand", [0, 3.0, 0], math.pi, [1.5, 0.5, 0.5])
    f1 = relation_features(a, b)
    b2 = b.copy()
    b2.xy = [0, 3.0]
    assert relation_distance(f1, relation_features(a, b2)) < 1e-6
    b3 = b.copy()
    b3.xy = [3.0, 0.0]                  # same distance, direction rotated 90
    d90 = relation_distance(f1, relation_features(a, b3))
    b4 = b.copy()
    b4.xy = [0.0, -3.0]                 # direction reversed
    d180 = relation_distance(f1, relation_features(a, b4))
    assert 0.05 < d90 < d180, (d90, d180)
    b5 = b.copy()
    b5.xy = [0.0, 6.0]                  # same direction, twice the distance
    assert relation_distance(f1, relation_features(a, b5)) > 0.05
    # a uniform rescale changes only the distance term, never direction,
    # orientation or contact -- so it is bounded by that term's weight share
    a2, b6 = a.copy(), b.copy()
    a2.xy, b6.xy = a.xy * 2, b.xy * 2
    a2.size, b6.size = a.size * 2, b.size * 2
    d_scale = relation_distance(relation_features(a, b),
                                relation_features(a2, b6))
    assert 0 < d_scale < 1.0 / 3.5 + 1e-6, d_scale


print("\nenergies and optimisation")


@check("exact energy is zero on the reference scene itself")
def _():
    s, g = _scene("bedroom", 2)
    e = exact_energy(s, build_design_intent(g, s.room))
    assert e["E_rel"] < 1e-9, e["E_rel"]
    assert e["E_edit"] < 1e-9


@check("surrogate and exact agree on which of two layouts is better")
def _():
    import torch
    torch.set_num_threads(1)
    s, g = _scene("living_room", 4)
    intent = build_design_intent(g, s.room)
    good = s.copy()
    bad = s.copy()
    for o in bad.objects:                       # shove everything into a corner
        o.xy = o.xy * 0.25 + s.room.bbox[0] * 0.75
    prob = TorchProblem(good, intent, EnergyWeights(), device="cpu")
    xy = np.stack([[o.xy for o in good.objects], [o.xy for o in bad.objects]])
    yaw = np.stack([[o.yaw for o in good.objects], [o.yaw for o in bad.objects]])
    with torch.no_grad():
        sur = prob.energy(torch.tensor(xy, dtype=torch.float32),
                          torch.tensor(yaw, dtype=torch.float32)).numpy()
    ex_good = exact_energy(good, intent)["E"]
    ex_bad = exact_energy(bad, intent)["E"]
    assert sur[0] < sur[1], sur
    assert ex_good < ex_bad, (ex_good, ex_bad)


@check("retargeting a room into itself is a fixed point")
def _():
    s, g = _scene("bedroom", 2)
    r = retarget(g, s.room.copy(), cfg=RetargetConfig(restarts=8, device="cpu"))
    assert r.energy["E_rel"] < 0.05, r.energy["E_rel"]
    assert len(r.scene.objects) >= len(s.objects) - 1


@check("retargeting into hard geometry stays inside and collision free")
def _():
    rng = np.random.default_rng(11)
    s, g = _scene("living_room", 6)
    for lvl in (2, 4, 5):
        room = deform_room(s.room, lvl, rng).room
        r = retarget(g, room, cfg=RetargetConfig(restarts=12, device="cpu"))
        m = evaluate(g, r.scene)
        assert m["R_OOB"] < 0.02, (lvl, m["R_OOB"])
        assert m["R_col"] < 0.02, (lvl, m["R_col"])


@check("relation-aware beats direct scaling on legality")
def _():
    rng = np.random.default_rng(2)
    ours, theirs = [], []
    for k in range(3):
        s, g = _scene(("bedroom", "living_room", "dining_room")[k], 20 + k)
        room = deform_room(s.room, 4, rng).room
        ours.append(evaluate(g, retarget(
            g, room, cfg=RetargetConfig(restarts=12, device="cpu")).scene))
        theirs.append(evaluate(g, run_baseline("direct_scaling", g, room)))
    a = float(np.mean([m["legality"] for m in ours]))
    b = float(np.mean([m["legality"] for m in theirs]))
    assert a > b + 0.05, (a, b)


print("\nmetrics and io")


@check("metrics are exact on a hand-built scene")
def _():
    room = Room(polygon=np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float))
    objs = [ObjectInstance("a", "misc", [1, 1, 0], 0.0, [1.0, 1.0, 1.0]),
            ObjectInstance("b", "misc", [4.0, 3.0, 0], 0.0, [2.0, 1.0, 1.0])]
    s = Scene("t", room, objs)
    g = build_motifs(build_scene_graph(s))
    m = evaluate(g, s)
    assert abs(m["R_OOB"] - 1.0 / 3.0) < 1e-3, m["R_OOB"]
    assert m["R_col"] < 1e-9
    assert abs(m["S_rel"] - 1.0) < 1e-6


@check("free space finds a fragmented room")
def _():
    room = Room(polygon=np.array([[0, 0], [6, 0], [6, 3], [0, 3]], float))
    wall = ObjectInstance("w", "wardrobe", [3, 1.5, 0], 0.0, [0.4, 3.0, 2.0])
    s = Scene("t", room, [wall])
    fs = build_freespace(s)
    assert fs.n_components() == 2, fs.n_components()
    assert fs.largest_component_ratio() < 0.75


@check("scene serialisation round-trips")
def _():
    s, _ = _scene("dining_room", 9)
    d = s.to_dict()
    t = Scene.from_dict(d)
    assert len(t.objects) == len(s.objects)
    assert np.allclose(t.room.polygon, s.room.polygon)
    assert np.allclose(t.objects[0].size, s.objects[0].size)
    assert abs(t.density() - s.density()) < 1e-9


@check("elasticity sample collection runs on a corpus")
def _():
    ds = generate_dataset(24)
    smp = collect_elasticity_samples(ds)
    assert len(smp) > 100
    assert all(x.d_ref > 0 for x in smp)


if __name__ == "__main__":
    print()
    if _FAILED:
        print(f"{len(_FAILED)} FAILED: {_FAILED}")
        sys.exit(1)
    print("all tests passed")


print("\nC_t constraints (section 1)")


@check("a pinned object keeps its pose and survives summarization")
def _():
    rng = np.random.default_rng(3)
    s, g = _scene("bedroom", 4)
    bed = next(o for o in s.objects if o.category == "double_bed")
    bed.locked = True
    pose = (bed.xy.copy(), bed.yaw)
    room = deform_room(s.room, 2, rng).room
    r = retarget(g, room, cfg=RetargetConfig(restarts=10, device="cpu"))
    out = r.scene.by_id(bed.oid)
    assert out is not None, "a pinned object was deleted"
    assert float(np.linalg.norm(out.xy - pose[0])) < 1e-6, out.xy - pose[0]
    assert abs(out.yaw - pose[1]) < 1e-6


@check("keep-out regions are respected")
def _():
    rng = np.random.default_rng(5)
    s, g = _scene("living_room", 8)
    room = deform_room(s.room, 1, rng).room
    b = room.bbox
    # forbid a quarter of the room
    lo = b[0]
    mid = (b[0] + b[1]) / 2
    zone = np.array([[lo[0], lo[1]], [mid[0], lo[1]], [mid[0], mid[1]],
                     [lo[0], mid[1]]])
    free_room = room.copy()
    room.keepout = [zone]
    a = retarget(g, free_room, cfg=RetargetConfig(restarts=12, device="cpu")).scene
    b2 = retarget(g, room, cfg=RetargetConfig(restarts=12, device="cpu")).scene
    from shapely.geometry import Polygon as _P
    z = _P(zone)
    inv = lambda sc: sum(object_polygon(o).intersection(z).area
                         for o in sc.objects if o.keep and o.z < 1.9)
    assert inv(b2) < max(inv(a) * 0.5, 0.05), (inv(a), inv(b2))


@check("continuous size stays inside its band and only shrinks when it must")
def _():
    rng = np.random.default_rng(11)
    s, g = _scene("bedroom", 6)
    base = {o.oid: o.size.copy() for o in s.objects}
    ident = retarget(g, s.room, cfg=RetargetConfig(restarts=8, device="cpu")).scene
    for o in ident.objects:                       # identity target: no trimming
        assert abs(o.size[0] / base[o.oid][0] - 1.0) < 1e-3, o.category
    tight = retarget(g, deform_room(s.room, 1, rng).room,
                     cfg=RetargetConfig(restarts=8, device="cpu")).scene
    for o in tight.objects:
        f = o.size[0] / base[o.oid][0]
        assert 0.92 - 1e-3 <= f <= 1.08 + 1e-3, (o.category, f)


print("\nnew stages (sections 6, 16.2, 17)")


@check("f^geo separates shapes that share a bounding box")
def _():
    from reroom.perception.geometry import (SHAPE_DIM, descriptor_from_mesh,
                                            shape_descriptor, shape_distance)
    rng = np.random.default_rng(0)
    box = rng.random((4000, 3))
    # same extents, mass in the top half only
    top = rng.random((4000, 3)) * np.array([1, 1, 0.5]) + np.array([0, 0, 0.5])
    top = np.vstack([top, [[0, 0, 0], [1, 1, 0]]])
    a, b = shape_descriptor(box), shape_descriptor(top)
    assert a.shape == (SHAPE_DIM,)
    assert shape_distance(a, shape_descriptor(rng.random((4000, 3)))) < 0.06
    assert shape_distance(a, b) > 0.25, shape_distance(a, b)
    assert shape_distance(a, None) == 0.0
    # a mesh path with faces gives the same kind of answer
    v = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    assert descriptor_from_mesh(v, f).shape == (SHAPE_DIM,)


@check("stripping motifs leaves the original graph intact")
def _():
    from reroom.intent.motifs import strip_motifs
    s, g = _scene("bedroom", 6)
    n_rel, n_mot = len(g.relations), len(g.motifs)
    tags = [o.meta.get("motif") for o in g.scene.objects]
    f = strip_motifs(g)
    assert not f.motifs and len(f.relations) <= n_rel
    assert not any("grouped_with" == r.kind for r in f.relations)
    assert len(g.relations) == n_rel and len(g.motifs) == n_mot
    assert [o.meta.get("motif") for o in g.scene.objects] == tags
    assert all(o.meta.get("motif") is None for o in f.scene.objects)


@check("the appearance term survives a partially embedded catalogue")
def _():
    from reroom.data.asset_bank import Asset, AssetBank
    rng = np.random.default_rng(1)
    e = rng.normal(size=8).astype(np.float32)
    assets = [Asset(aid="a", category="sofa", size=np.array([2.0, .9, .8]),
                    embedding=e),
              Asset(aid="b", category="sofa", size=np.array([2.0, .9, .8]),
                    embedding=-e),
              Asset(aid="c", category="sofa", size=np.array([2.0, .9, .8]))]
    bank = AssetBank(assets)
    hit = bank.retrieve("sofa", np.array([2.0, .9, .8]), e,
                        lambda_f=1.0, lambda_s=1.0, topk=3)
    assert hit[0][0].aid == "a", [h[0].aid for h in hit]
    assert hit[-1][0].aid == "b", [h[0].aid for h in hit]


@check("merging a second bank only fills gaps")
def _():
    from reroom.data.asset_bank import Asset, AssetBank, merge_banks
    base = AssetBank([Asset(aid=f"f{k}", category="sofa",
                            size=np.array([2.0, .9, .8])) for k in range(20)])
    extra = AssetBank(
        [Asset(aid="s0", category="sofa", size=np.array([2.0, .9, .8])),
         Asset(aid="s1", category="piano", size=np.array([1.4, .6, 1.2]))])
    m = merge_banks(base, extra)
    assert len(m.by_category["sofa"]) == 20, "well-covered category was padded"
    assert len(m.by_category["piano"]) == 1, "new category was not added"
