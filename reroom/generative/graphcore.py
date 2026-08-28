"""Graph as the backbone — and as the *learned* definition of a violation.

A blanket overlap penalty is wrong for interiors: a dining chair tucked under its
table and a nightstand against a bed are correct layouts, not collisions. Real
3D-FRONT rooms score 0.388 on a category-blind collision metric, and training
against one pulls layouts away from ground truth.

The obvious repair — a hand-written list of relation types that are allowed to
touch — just moves the modelling into a rule table. Instead the model **learns
what spacing each relation implies**:

    PairGapPredictor : (edge features, the two objects' features) -> expected gap

It is supervised by the gaps observed in real layouts, so it cannot collapse to
"everything is allowed"; and when it scores a generated layout its prediction is
detached, so the layout is pushed towards the learned spacing rather than the
prediction being bent to excuse the layout. What counts as a violation is
therefore learned from data, per instance, not declared.

Edges also condition attention and messages (a graph transformer, not a plain
transformer with a scalar bias), so the same structure that defines correctness
also routes information.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .tokens import EDGE_DIM, TOKEN_COND_DIM

__all__ = ["dense_edges", "pair_gaps", "PairGapPredictor", "GraphAttention"]


def dense_edges(batch, N: int):
    """Sparse edge list -> (B,N,N,EDGE_DIM) features and (B,N,N) presence mask.
    Relations are undirected, so the tensor is symmetrised."""
    ei, ef, em = batch["edge_index"], batch["edge_feat"], batch["edge_mask"]
    B, E, D = ef.shape
    dense = ef.new_zeros(B, N * N, D)
    m = ef.new_zeros(B, N * N)
    i = ei[:, 0].clamp(0, N - 1)
    j = ei[:, 1].clamp(0, N - 1)
    w = em.float()
    for a, b in ((i, j), (j, i)):
        idx = a * N + b
        dense.scatter_add_(1, idx[..., None].expand(-1, -1, D), ef * w[..., None])
        m.scatter_add_(1, idx, w)
    return dense.view(B, N, N, D), m.clamp(max=1.0).view(B, N, N)


def pair_gaps(x, batch):
    """(B,N,N) signed gap between footprints, in metres. Negative = overlapping."""
    fh = batch["frame_h"]
    p = x[..., :2] * fh[:, None, :]
    w = torch.exp(batch["cond"][..., 0]); d = torch.exp(batch["cond"][..., 1])
    r = 0.5 * torch.sqrt(w * w + d * d)
    diff = p[:, :, None, :] - p[:, None, :, :]
    dist = (diff * diff).sum(-1).clamp(min=1e-12).sqrt()
    return dist - (r[:, :, None] + r[:, None, :])


class PairGapPredictor(nn.Module):
    """Learns the spacing a relation implies, from data.

    Input per ordered pair: the edge features (zero when the pair has no
    relation, with the presence flag appended so "no edge" is itself a case the
    model can learn), and both objects' conditioning. Output: expected gap in
    metres, and a log-scale tolerance so the model can also express *how firmly*
    that gap is required — a dining chair is tightly pinned, a sofa-to-TV
    distance is elastic.
    """

    def __init__(self, hidden: int = 96):
        super().__init__()
        din = (EDGE_DIM + 1) + 2 * TOKEN_COND_DIM
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))

    def forward(self, batch, N: int):
        de, em = dense_edges(batch, N)
        cond = batch["cond"]
        ci = cond[:, :, None, :].expand(-1, -1, N, -1)
        cj = cond[:, None, :, :].expand(-1, N, -1, -1)
        z = torch.cat([de, em[..., None], ci, cj], dim=-1)
        out = self.net(z)
        gap = out[..., 0]
        # symmetric by construction: a pair has one spacing
        gap = 0.5 * (gap + gap.transpose(1, 2))
        logtol = out[..., 1]
        logtol = 0.5 * (logtol + logtol.transpose(1, 2))
        return gap, logtol.clamp(-3.0, 3.0), em


def graph_violation(x, batch, predictor: PairGapPredictor, detach_pred: bool = True):
    """Violation of the *learned* spacing, per object and per scene (metres).

    ``detach_pred`` stops the layout loss from being reduced by moving the
    prediction instead of the furniture; the predictor is trained by its own
    supervision against real layouts (see ``gap_supervision``).
    """
    mask = batch["mask"].float()
    B, N = mask.shape
    gap = pair_gaps(x, batch)
    pred, logtol, em = predictor(batch, N)
    if detach_pred:
        pred = pred.detach(); logtol = logtol.detach()
    tol = torch.exp(logtol)
    pair = mask[:, :, None] * mask[:, None, :]
    eye = torch.eye(N, device=x.device, dtype=torch.bool)[None]
    pair = pair * (~eye).float()
    dev = ((gap - pred).abs() - tol).clamp(min=0.0) * pair
    per_obj = dev.sum(-1)
    denom = pair.sum(dim=(1, 2)).clamp(min=1.0)
    return per_obj, dev.sum(dim=(1, 2)) / denom


def gap_supervision(batch, predictor: PairGapPredictor):
    """Train the predictor on the spacing real layouts actually use.

    The target is the gap between the same two objects in the ground-truth
    layout, so "a chair belongs against its table" is learned from the corpus
    rather than declared in a table of relation types.
    """
    mask = batch["mask"].float()
    B, N = mask.shape
    tgt = pair_gaps(batch["state"], batch)
    pred, logtol, em = predictor(batch, N)
    pair = mask[:, :, None] * mask[:, None, :]
    eye = torch.eye(N, device=tgt.device, dtype=torch.bool)[None]
    pair = pair * (~eye).float()
    # Gaussian NLL: the model must predict the spacing AND how firm it is,
    # so it cannot buy a low loss with an enormous tolerance.
    inv = torch.exp(-logtol)
    nll = ((pred - tgt).abs() * inv + logtol) * pair
    return nll.sum() / pair.sum().clamp(min=1.0)


class GraphAttention(nn.Module):
    """Edge-conditioned multi-head attention: edges bias the logits *and* carry
    messages, so a rigid relation both draws attention and transports geometry."""

    def __init__(self, d: int, heads: int, edge_dim: int = EDGE_DIM + 1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.e_bias = nn.Sequential(nn.Linear(edge_dim, 64), nn.SiLU(), nn.Linear(64, heads))
        self.e_msg = nn.Sequential(nn.Linear(edge_dim, 64), nn.SiLU(), nn.Linear(64, d))

    def forward(self, h, dense_edge, key_pad):
        B, N, _ = h.shape
        q = self.q(h).view(B, N, self.h, self.dh).transpose(1, 2)
        k = self.k(h).view(B, N, self.h, self.dh).transpose(1, 2)
        v = self.v(h).view(B, N, self.h, self.dh).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        logits = logits + self.e_bias(dense_edge).permute(0, 3, 1, 2)
        logits = logits + key_pad[:, None, None, :].float() * (-1e9)
        a = logits.softmax(-1)
        out = (a @ v).transpose(1, 2).reshape(B, N, -1)
        msg = self.e_msg(dense_edge)
        out = out + (a.mean(1)[..., None] * msg).sum(2)
        return self.o(out)
