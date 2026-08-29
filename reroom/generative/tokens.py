"""Permutation-aware object tokens for the generative proposal (section 13).

The scene state ``X`` is a *set* of object tokens; the model must be
equivariant to their order, so there are no positional encodings and all
structure enters through per-token features and through relation-derived
attention biases.

Per token
    the flowed variables (u, v, cos, sin)  -- pose in the target room's
    minimum-rotated-rectangle frame, hence translation/rotation/scale free;
    plus fixed conditioning: category, log size, motif role, importance,
    wall affinity, required clearance, and the object's pose in the *reference*
    room's frame.

Per scene
    ``g(P_r)``, ``g(P_t)``, the target boundary sampled as points+normals, the
    room type, and ``z_style``.

Per edge
    relation type, weight, the fitted elasticity ``alpha``, the room-scale ratio
    ``gamma`` and the elasticity-adjusted target relation ``phi~``, which become
    additive attention biases.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shapely.geometry import LineString

from ..core.categories import PRIORS, ROOM_TYPES, prior
from ..core.scene import Room, Scene
from .floorgraph import FLOOR_DIM, N_FLOOR, floor_nodes
from ..geom.polygon import (as_polygon, floor_descriptor,
                            min_rotated_rect_params, object_polygon)
from ..intent.relations import RELATION_TYPES
from ..retarget.target import DesignIntent

__all__ = ["CATS", "N_CAT", "MOTIF_NAMES", "N_MOTIF", "N_REL", "STATE_DIM",
           "TOKEN_COND_DIM", "GLOBAL_DIM", "EDGE_DIM", "N_BOUNDARY",
           "RoomFrame", "room_frame", "to_frame", "from_frame",
           "build_tokens", "TokenBatch", "collate"]

CATS = tuple(sorted(PRIORS.keys()))
N_CAT = len(CATS)
_CAT_IX = {c: i for i, c in enumerate(CATS)}
MOTIF_NAMES = ("none", "sleeping", "dining", "conversation", "media", "work",
               "dressing", "reading", "storage", "hearth", "music", "cluster")
N_MOTIF = len(MOTIF_NAMES)
_MOTIF_IX = {m: i for i, m in enumerate(MOTIF_NAMES)}
N_REL = len(RELATION_TYPES)
_REL_IX = {r: i for i, r in enumerate(RELATION_TYPES)}

STATE_DIM = 4                 # u, v, cos yaw, sin yaw
TOKEN_COND_DIM = 15           # numeric part; categorical ids are separate
WALL_AFFINITY_TAU = 0.35      # metres: length-scale of the reference-wall signal


def ref_wall_affinity(obj, ref_room: Room) -> float:
    """How strongly this object hugged a wall *in the reference room*.

    The plan's principle is to preserve what the reference did, not to impose a
    category default.  ``prior(cat).wall`` already says "sofas tend to sit
    against walls", but it cannot tell a dining table pushed into a corner from
    one floated in the middle of the room -- both share the category prior.
    This reads the actual geometry: the object's footprint-to-nearest-wall gap
    in the reference, passed through a decaying kernel, so a table touching a
    wall scores ~1 and one a metre away scores ~0.06.  The flow then learns to
    reward wall-hugging in the target *only for the objects that were against a
    wall to begin with*.  Overhead objects (pendant lamps) do not participate.
    """
    if obj.z >= 1.4:
        return 0.0
    walls = ref_room.walls()
    if not walls:
        return 0.0
    fp = object_polygon(obj)
    d = min(fp.distance(LineString([a, b])) for a, b in walls)
    return float(math.exp(-d / WALL_AFFINITY_TAU))
N_BOUNDARY = 32
GLOBAL_DIM = 12 + 12 + 4      # g(P_r), g(P_t), area/aspect summary
EDGE_DIM = N_REL + 9


@dataclass
class RoomFrame:
    centre: np.ndarray
    axis1: np.ndarray
    axis2: np.ndarray
    half1: float
    half2: float
    angle: float


def room_frame(room: Room) -> RoomFrame:
    poly = as_polygon(room)
    long_, short_, ang = min_rotated_rect_params(poly)
    c = np.asarray(poly.centroid.coords[0])
    a1 = np.array([math.cos(ang), math.sin(ang)])
    a2 = np.array([-a1[1], a1[0]])
    return RoomFrame(c, a1, a2, max(long_ / 2, 1e-3), max(short_ / 2, 1e-3), ang)


def to_frame(xy: np.ndarray, yaw: float, fr: RoomFrame) -> np.ndarray:
    d = np.asarray(xy, dtype=float) - fr.centre
    u = float(np.dot(d, fr.axis1)) / fr.half1
    v = float(np.dot(d, fr.axis2)) / fr.half2
    th = yaw - fr.angle
    return np.array([u, v, math.cos(th), math.sin(th)], dtype=np.float32)


def from_frame(state: np.ndarray, fr: RoomFrame) -> tuple[np.ndarray, float]:
    u, v, c, s = [float(x) for x in state[:4]]
    xy = fr.centre + u * fr.half1 * fr.axis1 + v * fr.half2 * fr.axis2
    yaw = math.atan2(s, c) + fr.angle
    return xy, yaw


def boundary_samples(room: Room, n: int = N_BOUNDARY) -> np.ndarray:
    """(n, 6): boundary points in *both* MRR-normalised and metric coords.

    The extra two channels carry ``(u*h1, v*h2)``, i.e. the boundary point's
    position in the frame basis in metres.  Scale-invariant shape lives in the
    first two channels; metric size lives in the middle two.  A wall-hugging
    model that sees only the normalised (u, v) cannot tell a 3 m wall from a
    6 m wall -- adding the metric pair fixes exactly this, without giving up
    the shape prior the normalised pair provides.
    """
    fr = room_frame(room)
    poly = as_polygon(room)
    ring = poly.exterior
    L = ring.length
    out = np.zeros((n, 6), dtype=np.float32)
    for k in range(n):
        p = ring.interpolate((k + 0.5) / n * L)
        q = ring.interpolate(((k + 0.5) / n + 1e-3) * L)
        t = np.array([q.x - p.x, q.y - p.y])
        nl = np.linalg.norm(t)
        t = t / nl if nl > 1e-9 else np.array([1.0, 0.0])
        nrm = np.array([-t[1], t[0]])
        d = np.array([p.x, p.y]) - fr.centre
        u = float(np.dot(d, fr.axis1)) / fr.half1
        v = float(np.dot(d, fr.axis2)) / fr.half2
        out[k] = [u, v,                                    # normalised shape
                  u * fr.half1, v * fr.half2,              # metric position (m)
                  float(np.dot(nrm, fr.axis1)),
                  float(np.dot(nrm, fr.axis2))]            # inward normals
    return out


@dataclass
class TokenBatch:
    """One training/inference example, in numpy."""

    state: np.ndarray            # (N, 4) target state (training only)
    cat: np.ndarray              # (N,) int
    motif: np.ndarray            # (N,) int
    cond: np.ndarray             # (N, TOKEN_COND_DIM)
    edge_index: np.ndarray       # (2, E)
    edge_feat: np.ndarray        # (E, EDGE_DIM)
    glob: np.ndarray             # (GLOBAL_DIM,)
    boundary: np.ndarray         # (N_BOUNDARY, 4)
    floor: np.ndarray            # (N_FLOOR, FLOOR_DIM) free-space nodes
    floor_adj: np.ndarray        # (N_FLOOR, N_FLOOR) visibility edges
    floor_pts: np.ndarray        # (N_FLOOR, 2) same nodes, world metres
    floor_r: float               # covering radius, metres
    room_type: int
    mask: np.ndarray             # (N,) bool -- valid tokens
    mgrp: np.ndarray             # (N,) int -- motif group id, -1 for none
    parent: np.ndarray           # (N,) int -- parent (head) token index, -1 none
    meta: dict


def _floor_fields(target_room: Room, fr_tgt) -> dict:
    """Free-space nodes for the target room.

    Objects stranded in a region the target room does not have need a large,
    deliberate relocation, not a local nudge: measured on the run 6 checkpoint,
    the objects still outside an L or T room sit a median 0.47 m / 0.31 m past
    the wall and up to 1.9 m, while the corridor's offenders -- a pure squeeze --
    sit 0.09 m out. Boundary samples tell the model where the walls are; nothing
    told it where there is floor to move TO. These nodes are that.
    """
    feat, adj, cover_r, _world = floor_nodes(target_room, fr_tgt)
    # columns 2:4 are the nodes in frame-basis metres (u*h1, v*h2), the same
    # convention object centres and boundary samples use. The world-metre
    # positions are for geometry checks only and must not reach the model.
    return {"floor": feat, "floor_adj": adj, "floor_pts": feat[:, 2:4].copy(),
            "floor_r": float(cover_r)}


def build_tokens(intent: DesignIntent, target_room: Room,
                 target_scene: Scene | None = None) -> TokenBatch:
    """Assemble the conditioning (and, for training, the target state)."""
    src = intent.source
    objs = src.objects
    n = len(objs)
    fr_src = room_frame(src.room)
    fr_tgt = room_frame(target_room)

    cat = np.array([_CAT_IX.get(o.category, _CAT_IX["misc"]) for o in objs],
                   dtype=np.int64)
    motif_of_idx = {}
    for m in intent.motifs:
        for i in m.members:
            motif_of_idx[i] = m
    motif = np.array([_MOTIF_IX.get(
        motif_of_idx[i].name if i in motif_of_idx else "none", 0) for i in range(n)],
        dtype=np.int64)
    # Per-token motif group id for L_rel (Step 2).  -1 means singleton / not in
    # any motif.  Groups are indexed by their position in intent.motifs.
    _grp_of = {}
    for _gi, _m in enumerate(intent.motifs):
        for _i in _m.members: _grp_of[_i] = _gi
    # Per-token PARENT index (ReRoom 2.0 Step 2): a non-head motif member points
    # at its motif head; heads and non-motif objects get -1.  Children can then
    # be predicted as an offset relative to the parent (scale-invariant).
    _parent_of = {}
    for _m in intent.motifs:
        for _i in _m.members:
            if _i != _m.head:
                _parent_of[_i] = _m.head
    mgrp = np.array([_grp_of.get(i, -1) for i in range(n)], dtype=np.int64)
    parent = np.array([_parent_of.get(i, -1) for i in range(n)], dtype=np.int64)

    cond = np.zeros((n, TOKEN_COND_DIM), dtype=np.float32)
    for i, o in enumerate(objs):
        p = prior(o.category)
        m = motif_of_idx.get(i)
        ref_state = to_frame(o.xy, o.yaw, fr_src)
        cond[i] = [
            math.log(max(o.size[0], 1e-2)), math.log(max(o.size[1], 1e-2)),
            math.log(max(o.size[2], 1e-2)),
            float(intent.zeta[i]) if i < len(intent.zeta) else 0.3,
            p.wall, p.front_clear, p.anchor, p.droppable,
            1.0 if (m is not None and m.head == i) else 0.0,
            (m.rigidity if m is not None else 0.4),
            ref_state[0], ref_state[1], ref_state[2], ref_state[3],
            # per-instance reference-wall signal: reward wall-hugging in the
            # target only when the reference actually hugged a wall
            ref_wall_affinity(o, src.room),
        ]

    ei = np.zeros((2, len(intent.relations)), dtype=np.int64)
    ef = np.zeros((len(intent.relations), EDGE_DIM), dtype=np.float32)
    for k, r in enumerate(intent.relations):
        ei[:, k] = (r.i, r.j)
        onehot = np.zeros(N_REL, dtype=np.float32)
        onehot[_REL_IX.get(r.kind, 0)] = 1.0
        des = r.phi_des.astype(np.float32).copy()
        des[0] /= fr_tgt.half1
        des[1] /= fr_tgt.half2
        des[4] /= max(fr_tgt.half1, 1e-3)
        des[5] /= max(fr_tgt.half1, 1e-3)
        ef[k] = np.concatenate([onehot, [r.weight, r.alpha, r.gamma], des])

    g_src = floor_descriptor(as_polygon(src.room))
    g_tgt = floor_descriptor(as_polygon(target_room))
    glob = np.concatenate([
        g_src, g_tgt,
        [math.log(max(intent.area_ratio, 1e-3)),
         float(intent.scale_hint[0]), float(intent.scale_hint[1]),
         float(intent.target_density)]]).astype(np.float32)

    # present1[i] = 1 iff source object i survives into the GT target scene.
    # D1 (joint discrete-continuous flow): this is the *existence* target the
    # mask-flow learns.  ``mask`` stays "all real source tokens are valid
    # candidates" so a to-be-dropped object still participates in attention and
    # the model can arrange survivors around the hole it leaves.  For non-drop
    # data present1 is all-ones, so mask_flow-off training is byte-identical.
    present1 = np.ones(n, dtype=np.float32)
    if target_scene is not None:
        lut = {o.oid: o for o in target_scene.objects}
        state = np.zeros((n, STATE_DIM), dtype=np.float32)
        mask = np.ones(n, dtype=bool)
        for i, o in enumerate(objs):
            t = lut.get(o.oid)
            if t is None:                       # dropped in GT -> existence 0
                present1[i] = 0.0
                state[i] = to_frame(o.xy, o.yaw, fr_src)   # prior pose, unused
                continue
            state[i] = to_frame(t.xy, t.yaw, fr_tgt)
    else:
        state = np.zeros((n, STATE_DIM), dtype=np.float32)
        mask = np.ones(n, dtype=bool)

    rt = ROOM_TYPES.index(target_room.room_type) \
        if target_room.room_type in ROOM_TYPES else len(ROOM_TYPES) - 1
    return TokenBatch(state=state, cat=cat, motif=motif, cond=cond,
                      edge_index=ei, edge_feat=ef, glob=glob,
                      **_floor_fields(target_room, fr_tgt),
                      boundary=boundary_samples(target_room),
                      room_type=rt, mask=mask, mgrp=mgrp, parent=parent,
                      meta={"frame_tgt": fr_tgt, "frame_src": fr_src,
                            "present1": present1,
                            "oids": [o.oid for o in objs]})


def collate(items: list[TokenBatch], device=None):
    """Pad a list of TokenBatch into dense tensors."""
    import torch
    B = len(items)
    N = max(len(b.cat) for b in items)
    E = max(b.edge_feat.shape[0] for b in items) if items else 0
    E = max(E, 1)

    def z(*shape, dtype=torch.float32):
        return torch.zeros(*shape, dtype=dtype, device=device)

    state = z(B, N, STATE_DIM)
    cond = z(B, N, TOKEN_COND_DIM)
    cat = z(B, N, dtype=torch.long)
    motif = z(B, N, dtype=torch.long)
    mask = z(B, N, dtype=torch.bool)
    mgrp = torch.full((B, N), -1, dtype=torch.long, device=device)
    parent = torch.full((B, N), -1, dtype=torch.long, device=device)
    edge_feat = z(B, E, EDGE_DIM)
    edge_index = z(B, 2, E, dtype=torch.long)
    edge_mask = z(B, E, dtype=torch.bool)
    glob = z(B, GLOBAL_DIM)
    boundary = z(B, N_BOUNDARY, 6)
    floor = z(B, N_FLOOR, FLOOR_DIM)
    floor_adj = z(B, N_FLOOR, N_FLOOR)
    floor_pts = z(B, N_FLOOR, 2)
    floor_r = z(B)
    rt = z(B, dtype=torch.long)
    frame_h = z(B, 2)                       # metric half-sizes (h1, h2)
    present1 = z(B, N)                       # D1 existence target (0 in padding)
    for b, it in enumerate(items):
        n = len(it.cat)
        state[b, :n] = torch.as_tensor(it.state)
        _p1 = it.meta.get("present1")
        present1[b, :n] = torch.as_tensor(_p1) if _p1 is not None else 1.0
        cond[b, :n] = torch.as_tensor(it.cond)
        cat[b, :n] = torch.as_tensor(it.cat)
        motif[b, :n] = torch.as_tensor(it.motif)
        mask[b, :n] = torch.as_tensor(it.mask)
        mgrp[b, :n] = torch.as_tensor(it.mgrp)
        parent[b, :n] = torch.as_tensor(it.parent)
        e = it.edge_feat.shape[0]
        if e:
            edge_feat[b, :e] = torch.as_tensor(it.edge_feat)
            edge_index[b, :, :e] = torch.as_tensor(it.edge_index)
            edge_mask[b, :e] = True
        glob[b] = torch.as_tensor(it.glob)
        boundary[b] = torch.as_tensor(it.boundary)
        floor[b] = torch.as_tensor(it.floor)
        floor_adj[b] = torch.as_tensor(it.floor_adj)
        floor_pts[b] = torch.as_tensor(it.floor_pts)
        floor_r[b] = it.floor_r
        rt[b] = it.room_type
        fr = it.meta.get("frame_tgt")
        if fr is not None:
            frame_h[b, 0] = float(fr.half1); frame_h[b, 1] = float(fr.half2)
    return {"state": state, "cond": cond, "cat": cat, "motif": motif,
            "mask": mask, "edge_feat": edge_feat, "edge_index": edge_index,
            "edge_mask": edge_mask, "glob": glob, "boundary": boundary,
            "floor": floor, "floor_adj": floor_adj, "floor_pts": floor_pts,
            "floor_r": floor_r,
            "room_type": rt, "frame_h": frame_h, "mgrp": mgrp,
            "parent": parent, "present1": present1}
