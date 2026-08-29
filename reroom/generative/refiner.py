"""Graph-Refinement Transformer (GRT) — project5's end-to-end model.

One deterministic forward pass from the affine transplant of the reference to the
final layout. No noise, no ODE, no candidate sampling, no post-processing.

The design graph is the backbone, in three places:

* **it defines correctness** — a learned per-pair spacing (``PairGapPredictor``)
  replaces any hand-written notion of collision, so "a dining chair belongs
  tucked under its table" is learned from real layouts rather than declared in a
  rule table;
* **it routes information** — edge-conditioned attention biases the logits and
  carries messages (``GraphAttention``);
* **it is re-read every block** — violations of the learned spacing, of the
  boundary, and of walkability are recomputed at each refinement step and
  consumed as token features, so the network learns the correction instead of
  descending a fixed external energy.

project4 measured why this shape is right: its informative prior did most of the
work (removing it cost 0.054 S_rel and doubled collision), while its fixed
projection had only a local view and left a 2.83% collision residual on the
non-convex rooms it could not route around.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .graphcore import (GraphAttention, PairGapPredictor, dense_edges,
                        graph_violation)
from .floorgraph import FLOOR_DIM
from .tokens import CATS, TOKEN_COND_DIM

__all__ = ["GraphRefinementTransformer", "violation_features"]

STATE_DIM = 4          # (u, v, cos yaw, sin yaw)
N_VIOL = 11            # see violation_features


def violation_features(x, batch, predictor, use_walk: bool = True):
    """Per-object violation signals, differentiable in ``x`` (B, N, N_VIOL):

        0  violation of the LEARNED pair spacing, summed over partners (metres)
        1  signed inward distance to the nearest boundary minus circumradius
           (negative => poking through a wall)
        2,3 inward normal of that boundary sample
        4  distance to the nearest boundary (wall affinity)
        5  offset drift from the motif parent versus the reference (metres)
        6  blocked walkway near this object: free floor the rasterised flood fill
           cannot reach (zero when walkability is switched off)
        7  metres the worst oriented-box corner pokes outside the room
        8  this object's own soft reachability, the differentiable analogue of
           the per-object quantity PhyScene's R_reach counts
        9,10 offset in metres to the nearest UNOCCUPIED free-space node -- where
           there is floor to move to. Objects stranded in a region the target
           room does not have need a 0.3-1.9 m relocation (measured on L and T
           rooms), and boundary samples say where the walls are, never where
           there is space.

    These are *features*, not a loss: the blocks consume them and learn the
    correction.
    """
    mask = batch["mask"].float()
    fh = batch["frame_h"]
    p = x[..., :2] * fh[:, None, :]
    w = torch.exp(batch["cond"][..., 0]); d = torch.exp(batch["cond"][..., 1])
    r = 0.5 * torch.sqrt(w * w + d * d)

    # ---- learned spacing (replaces any hand-written collision rule) ----
    v_pair, _ = graph_violation(x, batch, predictor)

    # ---- boundary ----
    bp = batch["boundary"][..., :2] * fh[:, None, :]
    bn = batch["boundary"][..., 4:6]
    d2 = ((p[:, :, None, :] - bp[:, None, :, :]) ** 2).sum(-1)
    kmin = d2.argmin(-1)
    near_p = torch.gather(bp, 1, kmin[..., None].expand(-1, -1, 2))
    near_n = torch.gather(bn, 1, kmin[..., None].expand(-1, -1, 2))
    clearance = ((p - near_p) * near_n).sum(-1) - r
    wall_dist = d2.amin(-1).clamp(min=1e-12).sqrt()

    # ---- drift from the motif parent ----
    par = batch["parent"]
    pidx = par.clamp(min=0)
    ref = batch["cond"][..., 10:12] * fh[:, None, :]
    cur_off = p - torch.gather(p, 1, pidx[..., None].expand(-1, -1, 2))
    ref_off = ref - torch.gather(ref, 1, pidx[..., None].expand(-1, -1, 2))
    strain = (cur_off - ref_off).norm(dim=-1) * (par >= 0).float()

    # ---- blocked walkway (rasterised, differentiable) ----
    from .walkable import boundary_outside
    outside = boundary_outside(x, batch)

    if use_walk:
        from .walkable import object_reachability, walkability
        _, reach, freem = walkability(x, batch, G=32)
        obj_reach = object_reachability(x, batch, G=32)[0]
        blocked = (freem - reach).clamp(min=0.0)
        G = blocked.shape[-1]
        gxy = (x[..., :2].clamp(-1, 1) + 1.0) * 0.5 * (G - 1)
        gi = gxy.round().long().clamp(0, G - 1)
        near_blocked = torch.gather(blocked.reshape(blocked.shape[0], -1), 1,
                                    gi[..., 1] * G + gi[..., 0])
    else:
        near_blocked = torch.zeros_like(wall_dist)
        obj_reach = torch.ones_like(wall_dist)

    # ---- where is there free floor to move to? ----
    fp = batch["floor_pts"]                                     # (B,M,2) metres
    fw = torch.exp(batch["cond"][..., 0]).amax(dim=1)            # (B,) coarse size
    d_of = ((p[:, :, None, :] - fp[:, None, :, :]) ** 2).sum(-1).clamp(min=1e-12).sqrt()
    # a node is taken if some object already sits on it
    occ_f = torch.sigmoid((r[:, :, None] - d_of) * 4.0) * mask[..., None]
    free_f = (1.0 - occ_f.amax(dim=1)).clamp(0.0, 1.0)           # (B,M)
    cost = d_of + (1.0 - free_f)[:, None, :] * fw[:, None, None]
    k_free = cost.argmin(-1)                                    # detached choice
    tgt = torch.gather(fp, 1, k_free[..., None].expand(-1, -1, 2))
    to_free = tgt - p                                           # (B,N,2) metres

    v = torch.stack([v_pair, clearance, near_n[..., 0], near_n[..., 1],
                     wall_dist, strain, near_blocked, outside, obj_reach,
                     to_free[..., 0], to_free[..., 1]], dim=-1)
    return v * mask[..., None]


class RefinementBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.graph_attn = GraphAttention(d, heads)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))

    def forward(self, h, ctx, dense_edge, key_pad):
        h = h + self.graph_attn(self.n1(h), dense_edge, key_pad)
        c, _ = self.cross_attn(self.n2(h), ctx, ctx, need_weights=False)
        h = h + c
        return h + self.ff(self.n3(h))


class GraphRefinementTransformer(nn.Module):
    """L residual refinement blocks over the design graph; one pass, no
    post-processing."""

    def __init__(self, d_model: int = 384, n_blocks: int = 6, heads: int = 8,
                 n_cat: int | None = None, use_walk: bool = True):
        super().__init__()
        n_cat = n_cat or len(CATS)
        self.d, self.L, self.heads, self.use_walk = d_model, n_blocks, heads, use_walk
        self.gap = PairGapPredictor()
        self.cat_emb = nn.Embedding(n_cat, 64)
        self.in_proj = nn.Linear(STATE_DIM + TOKEN_COND_DIM + 64 + N_VIOL, d_model)
        self.step_emb = nn.Embedding(n_blocks, d_model)
        self.bnd_proj = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, d_model))
        # Free space is context the objects attend over, alongside the walls, and
        # its own edges are propagated first so a node's embedding carries the
        # room's topology -- the notch of an L is reachable only through its
        # throat, which euclidean proximity to a wall sample cannot express.
        self.flr_proj = nn.Sequential(nn.Linear(FLOOR_DIM, 128), nn.SiLU(),
                                      nn.Linear(128, d_model))
        self.flr_mp = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(2)])
        self.ctx_kind = nn.Embedding(2, d_model)      # wall vs floor
        self.blocks = nn.ModuleList([RefinementBlock(d_model, heads) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
                                  nn.SiLU(), nn.Linear(d_model // 2, STATE_DIM))
        # near-identity at init, but NOT exactly zero: a zero final weight makes
        # d(out)/d(h) = 0 and starves every upstream layer of gradient.
        nn.init.normal_(self.head[-1].weight, std=1e-3); nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch, return_trace: bool = False):
        cond, cat, mask = batch["cond"], batch["cat"], batch["mask"]
        B, N = cat.shape
        x = cond[..., 10:14].clone()               # affine transplant of the reference
        de, em = dense_edges(batch, N)
        dense_edge = torch.cat([de, em[..., None]], dim=-1)
        bnd = self.bnd_proj(batch["boundary"]) + self.ctx_kind.weight[0]
        f = self.flr_proj(batch["floor"])
        adj = batch["floor_adj"]
        deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
        for mp in self.flr_mp:                        # propagate over visibility
            f = f + torch.tanh(mp(torch.bmm(adj, f) / deg))
        ctx = torch.cat([bnd, f + self.ctx_kind.weight[1]], dim=1)
        key_pad = ~mask
        ce = self.cat_emb(cat)
        trace = []
        for l, blk in enumerate(self.blocks):
            v = violation_features(x, batch, self.gap, use_walk=self.use_walk)
            h = self.in_proj(torch.cat([x, cond, ce, v], dim=-1))
            h = h + self.step_emb.weight[l][None, None, :]
            h = blk(h, ctx, dense_edge, key_pad)
            x = x + self.head(h) * mask[..., None].float()
            if return_trace:
                trace.append(x)
        return (x, trace) if return_trace else x
