"""Constraint-Refinement Transformer (CRT) — project5's end-to-end model.

Replaces project4's DiT + 50-step ODE + external projection with L residual
refinement blocks that run once, deterministically, from the affine transplant of
the source layout. See DESIGN.md for the motivation; briefly:

* the task has a natural initialisation (the reference warped into the target),
  so generating from noise wastes compute — project4 measured that removing the
  informative prior costs 0.054 S_rel and doubles collision;
* constraint satisfaction happens *inside* the network: every block recomputes
  differentiable violation features and learns the correction, instead of
  descending a fixed external energy. A local gradient cannot route around a
  concave notch (project4's projection left a 2.83% residual there); attention
  has the global view it needs;
* the design graph enters as an attention bias, so relation preservation is part
  of the computation rather than a loss applied afterwards.

The output of the last block is the final layout — there is no post-processing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokens import TOKEN_COND_DIM, EDGE_DIM, CATS

__all__ = ["ConstraintRefinementTransformer", "violation_features"]

STATE_DIM = 4          # (u, v, cos yaw, sin yaw)
N_VIOL = 7             # see violation_features


# --------------------------------------------------------------------------- #
# differentiable violation features — the "optimisation" signal, in-network
# --------------------------------------------------------------------------- #
def violation_features(x, batch, nest_mat):
    """Per-object geometric violation signals, differentiable in ``x``.

    These are *features*, not a loss: the refinement blocks consume them and
    learn the correction. Everything is computed in metric space (positions are
    normalised MRR coordinates; multiplying by frame_h gives metres).

    Returns (B, N, N_VIOL):
        0  max pairwise overlap with a non-nestable neighbour (metres)
        1  summed overlap  (how crowded this object is)
        2  signed inward distance to the nearest boundary sample minus the
           object's circumradius  (negative => poking through the wall)
        3,4 inward normal of that nearest boundary sample
        5  distance to the nearest boundary  (wall proximity, for wall affinity)
        6  deviation of this object's offset to its motif parent from the
           reference offset  (relation strain, metres)
    """
    mask = batch["mask"].float()                                   # (B,N)
    fh = batch["frame_h"]                                          # (B,2)
    p = x[..., :2] * fh[:, None, :]                                # metric
    w = torch.exp(batch["cond"][..., 0]); d = torch.exp(batch["cond"][..., 1])
    r = 0.5 * torch.sqrt(w * w + d * d)                            # circumradius

    # ---- pairwise overlap (non-nestable only) ----
    diff = p[:, :, None, :] - p[:, None, :, :]
    dist = (diff * diff).sum(-1).clamp(min=1e-12).sqrt()
    overlap = (r[:, :, None] + r[:, None, :] - dist).clamp(min=0.0)
    nest = nest_mat[batch["cat"][:, :, None], batch["cat"][:, None, :]]
    eye = torch.eye(p.shape[1], device=p.device, dtype=torch.bool)[None]
    gate = (~nest & ~eye).float() * (mask[:, :, None] * mask[:, None, :])
    ov = overlap * gate
    v_max = ov.amax(dim=2)
    v_sum = ov.sum(dim=2)

    # ---- boundary ----
    bp = batch["boundary"][..., :2] * fh[:, None, :]
    bn = batch["boundary"][..., 4:6]
    d2 = ((p[:, :, None, :] - bp[:, None, :, :]) ** 2).sum(-1)
    kmin = d2.argmin(-1)
    near_p = torch.gather(bp, 1, kmin[..., None].expand(-1, -1, 2))
    near_n = torch.gather(bn, 1, kmin[..., None].expand(-1, -1, 2))
    inward = ((p - near_p) * near_n).sum(-1)                       # + inside
    clearance = inward - r                                          # < 0 => sticking out
    wall_dist = d2.amin(-1).clamp(min=1e-12).sqrt()

    # ---- relation strain vs the reference offset (parent-relative) ----
    par = batch["parent"]                                           # (B,N) -1 if head
    has_par = (par >= 0)
    pidx = par.clamp(min=0)
    ref = batch["cond"][..., 10:12] * fh[:, None, :]               # reference pose, metric
    cur_off = p - torch.gather(p, 1, pidx[..., None].expand(-1, -1, 2))
    ref_off = ref - torch.gather(ref, 1, pidx[..., None].expand(-1, -1, 2))
    strain = (cur_off - ref_off).norm(dim=-1) * has_par.float()

    v = torch.stack([v_max, v_sum, clearance, near_n[..., 0], near_n[..., 1],
                     wall_dist, strain], dim=-1)
    return v * mask[..., None]


# --------------------------------------------------------------------------- #
# graph -> attention bias
# --------------------------------------------------------------------------- #
class GraphBias(nn.Module):
    """Turn the design graph into a per-head additive attention bias.

    Relation preservation becomes structural: objects joined by a rigid,
    low-elasticity edge attend to one another strongly, so the block that moves
    one of them sees the other.
    """

    def __init__(self, heads: int):
        super().__init__()
        self.heads = heads
        self.proj = nn.Sequential(nn.Linear(EDGE_DIM, 32), nn.SiLU(), nn.Linear(32, heads))

    def forward(self, batch, N: int):
        ei = batch["edge_index"]                                    # (B,2,E)
        ef = batch["edge_feat"]                                     # (B,E,EDGE_DIM)
        em = batch["edge_mask"].float()                             # (B,E)
        B, E = em.shape
        w = self.proj(ef) * em[..., None]                           # (B,E,H)
        bias = ef.new_zeros(B, self.heads, N, N)
        i, j = ei[:, 0].clamp(0, N - 1), ei[:, 1].clamp(0, N - 1)
        flat = (i * N + j)                                          # (B,E)
        src = w.permute(0, 2, 1)                                    # (B,H,E)
        bias = bias.view(B, self.heads, N * N)
        bias.scatter_add_(2, flat[:, None, :].expand(-1, self.heads, -1), src)
        bias = bias.view(B, self.heads, N, N)
        return bias + bias.transpose(2, 3)                          # relations are symmetric


# --------------------------------------------------------------------------- #
# refinement block
# --------------------------------------------------------------------------- #
class RefinementBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.heads = heads

    def forward(self, h, bnd, gbias, key_pad):
        B, N, _ = h.shape
        # fold padding into the additive bias (mixing a float attn_mask with a
        # bool key_padding_mask is deprecated and semantically muddled)
        am = gbias + key_pad[:, None, None, :].float() * (-1e9)
        am = am.reshape(B * self.heads, N, N)
        a, _ = self.self_attn(self.n1(h), self.n1(h), self.n1(h),
                              attn_mask=am, need_weights=False)
        h = h + a
        c, _ = self.cross_attn(self.n2(h), bnd, bnd, need_weights=False)
        h = h + c
        return h + self.ff(self.n3(h))


class ConstraintRefinementTransformer(nn.Module):
    """L residual refinement blocks; one forward pass, no post-processing."""

    def __init__(self, d_model: int = 384, n_blocks: int = 6, heads: int = 8,
                 n_cat: int | None = None):
        super().__init__()
        n_cat = n_cat or len(CATS)
        self.d, self.L, self.heads = d_model, n_blocks, heads
        self.cat_emb = nn.Embedding(n_cat, 64)
        self.in_proj = nn.Linear(STATE_DIM + TOKEN_COND_DIM + 64 + N_VIOL, d_model)
        self.step_emb = nn.Embedding(n_blocks, d_model)             # which refinement step
        self.bnd_proj = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, d_model))
        self.gbias = GraphBias(heads)
        self.blocks = nn.ModuleList([RefinementBlock(d_model, heads) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
                                  nn.SiLU(), nn.Linear(d_model // 2, STATE_DIM))
        # Start as a near-identity refinement, but NOT exactly zero: a zero final
        # weight makes d(out)/d(h)=0, which starves every upstream layer of
        # gradient (verified in the smoke test: only 2 tensors received one).
        nn.init.normal_(self.head[-1].weight, std=1e-3); nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch, nest_mat, return_trace: bool = False):
        cond, cat = batch["cond"], batch["cat"]
        mask = batch["mask"]
        B, N = cat.shape
        # initialise at the affine transplant of the reference (cond[...,10:14])
        x = cond[..., 10:14].clone()
        bnd = self.bnd_proj(batch["boundary"])
        gb = self.gbias(batch, N)
        key_pad = ~mask
        ce = self.cat_emb(cat)
        trace = []
        for l, blk in enumerate(self.blocks):
            v = violation_features(x, batch, nest_mat)
            h = self.in_proj(torch.cat([x, cond, ce, v], dim=-1))
            h = h + self.step_emb.weight[l][None, None, :]
            h = blk(h, bnd, gb, key_pad)
            x = x + self.head(h) * mask[..., None].float()
            if return_trace:
                trace.append(x)
        if return_trace:
            return x, trace
        return x
