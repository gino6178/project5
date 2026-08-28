"""Graph-conditioned conditional flow matching for layout proposal (section 13).

    p_theta(L_t | G_r, P_t, z_style)                                   (34)
    X_tau = (1 - tau) X_0 + tau X_1,   tau in [0, 1]                   (35)
    v_theta(X_tau, tau, G_r, P_t, z_style)                             (36)

The network is a permutation-equivariant transformer over object tokens.  All
scene structure enters as *bias*, never as order:

* the design-intent graph becomes an additive attention bias, one scalar per
  head per edge, computed from the relation type, its weight, its fitted
  elasticity ``alpha``, the room-scale ratio ``gamma`` and the elasticity-
  adjusted target relation ``phi~``;
* the target floor polygon enters as a set of boundary points with inward
  normals, mean-pooled and broadcast, so concave and slanted rooms are
  representable rather than being flattened to (W, D);
* the flow time ``tau`` modulates every block through adaptive layer norm.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokens import (EDGE_DIM, GLOBAL_DIM, N_BOUNDARY, N_CAT, N_MOTIF,
                     STATE_DIM, TOKEN_COND_DIM)
from ..core.categories import ROOM_TYPES

__all__ = ["FlowModel", "timestep_embedding"]


REL_SCALE = 3.0   # metres normaliser for parent-relative child offsets


def to_relative(state, parent, frame_h, scale: float = REL_SCALE):
    """ReRoom 2.0 Step 2: express motif children as a SCALE-INVARIANT offset
    from their parent (head).  state (B,N,4)=(u,v,cos,sin), parent (B,N) int
    (-1 = head/none), frame_h (B,2) metric half-sizes.  Heads pass through; a
    child becomes (metric_offset_from_parent / scale, orientation - parent's).
    Because the offset is metric, it does not shrink when the room grows -- the
    chair stays tucked to the table at 1.35x."""
    is_child = (parent >= 0)
    pidx = parent.clamp(min=0)
    pg = torch.gather(state, 1, pidx[..., None].expand(-1, -1, 4))
    pyaw = torch.atan2(pg[..., 3], pg[..., 2])
    cyaw = torch.atan2(state[..., 3], state[..., 2])
    d_metric = (state[..., :2] - pg[..., :2]) * frame_h[:, None, :]
    rel_pos = d_metric / scale
    rel_yaw = cyaw - pyaw
    child = torch.stack([rel_pos[..., 0], rel_pos[..., 1],
                         torch.cos(rel_yaw), torch.sin(rel_yaw)], dim=-1)
    return torch.where(is_child[..., None], child, state)


def to_world(state, parent, frame_h, scale: float = REL_SCALE):
    """Inverse of :func:`to_relative` (parents are heads, already in world, so a
    single pass resolves every child)."""
    is_child = (parent >= 0)
    pidx = parent.clamp(min=0)
    pg = torch.gather(state, 1, pidx[..., None].expand(-1, -1, 4))
    pyaw = torch.atan2(pg[..., 3], pg[..., 2])
    rel_yaw = torch.atan2(state[..., 3], state[..., 2])
    d_norm = (state[..., :2] * scale) / frame_h[:, None, :]
    cuv = pg[..., :2] + d_norm
    cyaw = pyaw + rel_yaw
    child = torch.stack([cuv[..., 0], cuv[..., 1],
                         torch.cos(cyaw), torch.sin(cyaw)], dim=-1)
    return torch.where(is_child[..., None], child, state)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 1000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, device=t.device, dtype=torch.float32)
                      / half)
    a = t.float()[:, None] * freqs[None] * max_period ** 0.0
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class WallCrossAttention(nn.Module):
    """Full Module 2: object <-> wall-segment cross-attention.

    The target room's boundary points become explicit wall *tokens*.  Each
    furniture token attends to every wall token, biased by the object->wall
    geometry (relative displacement, perpendicular distance, normal
    misalignment) computed from the current state.  Unlike the per-object
    nearest-wall feature, this lets the model consider *several* walls and learn
    which one to snap to (important near corners / in L-shaped rooms).  The
    output projection is zero-initialised so adding this module to a trained
    checkpoint is a behavioural no-op at start, then learns the mechanism.
    """

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.dk = d // heads
        self.wenc = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.ebias = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, heads))
        self.proj = nn.Linear(d, d)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h, x, boundary, mask, node_gate=None):
        B, N, D = h.shape
        Nb = boundary.shape[1]
        w = self.wenc(boundary)                                   # (B, Nb, d)
        q = self.q(h).view(B, N, self.h, self.dk).transpose(1, 2)          # (B,H,N,dk)
        k, v = self.kv(w).chunk(2, dim=-1)
        k = k.view(B, Nb, self.h, self.dk).transpose(1, 2)                 # (B,H,Nb,dk)
        v = v.view(B, Nb, self.h, self.dk).transpose(1, 2)
        # object->wall edge geometry from current state
        p = x[..., :2]                                            # (B, N, 2)
        bp = boundary[..., :2]                                    # (B, Nb, 2)
        bn = boundary[..., 4:6]                                   # (B, Nb, 2) normals
        dp = bp[:, None, :, :] - p[:, :, None, :]                 # (B, N, Nb, 2)
        dist = dp.norm(dim=-1, keepdim=True)                      # (B, N, Nb, 1)
        obj_dir = x[..., 2:4]                                     # (B, N, 2)
        misalign = (1.0 - (obj_dir[:, :, None, :]
                           * bn[:, None, :, :]).sum(-1).abs())[..., None]
        efeat = torch.cat([dp, dist, misalign], dim=-1)          # (B, N, Nb, 4)
        eb = self.ebias(efeat).permute(0, 3, 1, 2)               # (B, H, N, Nb)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk) + eb
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, N, D)
        add = self.proj(out) * mask[..., None].float()
        if node_gate is not None:                 # mask wall pull off children
            add = add * node_gate
        return h + add


class BiasedAttention(nn.Module):
    """Multi-head self-attention with an additive per-edge bias."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.dk = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, bias, key_mask):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.h, self.dk).transpose(1, 2)
        k = k.view(B, N, self.h, self.dk).transpose(1, 2)
        v = v.view(B, N, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        if bias is not None:
            att = att + bias
        att = att.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


def _tokenize(mod):
    """Broadcast a modulation from either (B, d) or (B, N, d) onto (B, N, d)."""
    return mod if mod.dim() == 3 else mod[:, None]


class Block(nn.Module):
    """Pre-norm transformer block with adaptive (time-conditioned) LayerNorm.

    Accepts ``c`` in either (B, d) or (B, N, d).  Per-token ``c`` lets the graph
    node stream inject its embedding at every layer via adaLN (GraphGPS-style
    per-block fusion) rather than being diluted after a single input concat.
    """

    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.att = BiasedAttention(d, heads)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(d * mlp_ratio), d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, bias, key_mask):
        s1, b1, g1, s2, b2, g2 = self.ada(c).chunk(6, dim=-1)
        s1, b1, g1 = _tokenize(s1), _tokenize(b1), _tokenize(g1)
        s2, b2, g2 = _tokenize(s2), _tokenize(b2), _tokenize(g2)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.att(h, bias, key_mask)
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class GraphNodeLayer(nn.Module):
    """Symmetric-edge message passing for the graph stream.

    Message on each edge is an MLP over (h_src, h_dst, edge_feat), aggregated
    into the destination node with a sum; the reverse direction runs the same
    MLP with sides swapped and aggregates into the source, giving an undirected
    update in one step.  A LayerNorm + residual stabilises stacking.
    """

    def __init__(self, d: int, edge_d: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * d + edge_d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.upd = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, h, edge_index, edge_feat, edge_mask, node_mask):
        B, N, d = h.shape
        E = edge_index.shape[-1]
        if E == 0:
            return h
        src = edge_index[:, 0]                                # (B, E)
        dst = edge_index[:, 1]
        # gather source / destination node features per edge
        h_src = torch.gather(h, 1, src.unsqueeze(-1).expand(-1, -1, d))
        h_dst = torch.gather(h, 1, dst.unsqueeze(-1).expand(-1, -1, d))
        em = edge_mask.float().unsqueeze(-1)                  # (B, E, 1)
        m_fwd = self.msg(torch.cat([h_src, h_dst, edge_feat], -1)) * em
        m_bwd = self.msg(torch.cat([h_dst, h_src, edge_feat], -1)) * em
        agg = torch.zeros_like(h)
        agg.scatter_add_(1, dst.unsqueeze(-1).expand(-1, -1, d), m_fwd)
        agg.scatter_add_(1, src.unsqueeze(-1).expand(-1, -1, d), m_bwd)
        # residual update, masked to real nodes
        return h + (self.upd(self.norm(agg)) * node_mask[..., None].float())


class BoundaryEncoder(nn.Module):
    """Set-transformer-style encoder over the room's boundary points.

    Each boundary point carries 6 channels: normalised (u, v) for shape,
    metric (u*h1, v*h2) for size, and (n_axis1, n_axis2) for the inward
    normal.  Points cross-attend to each other so the encoder learns wall
    layout ("this is an L-shaped room with a corner cut, wall #3 is the long
    one, 6 m from centre").  A concat of mean + max pooling produces the
    global feature that used to come from ``bound.mean(1)``.

    This replaces the plain (Linear+mean-pool) `self.bound` because that
    tore up all local shape information into one flat average and could not
    distinguish "3 m wall" from "6 m wall" at all -- exactly the failure that
    made large rooms float.
    """

    def __init__(self, in_dim: int = 6, hidden: int = 64, out_dim: int = 64,
                 depth: int = 2, heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(depth)])
        self.att = nn.ModuleList([
            nn.MultiheadAttention(hidden, heads, batch_first=True)
            for _ in range(depth)])
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden),
                          nn.Linear(hidden, hidden * 2), nn.GELU(),
                          nn.Linear(hidden * 2, hidden))
            for _ in range(depth)])
        self.pool = nn.Linear(2 * hidden, out_dim)

    def forward(self, b):
        # b: (B, Nb, 6)
        h = self.embed(b)
        for norm, att, mlp in zip(self.norms, self.att, self.mlps):
            hn = norm(h)
            a, _ = att(hn, hn, hn, need_weights=False)
            h = h + a
            h = h + mlp(h)
        # mean+max pool -> Linear
        g = torch.cat([h.mean(1), h.amax(1)], dim=-1)              # (B, 2*hidden)
        return self.pool(g)                                        # (B, out_dim)


class FlowModel(nn.Module):
    def __init__(self, d: int = 256, depth: int = 6, heads: int = 8,
                 cat_emb: int = 64, motif_emb: int = 32,
                 graph_dim: int = 128, graph_depth: int = 3,
                 geo_bias: bool = False, wall_tokens: bool = False,
                 parent_relative: bool = False, mask_flow: bool = False):
        super().__init__()
        self.d = d
        self.heads = heads
        self.parent_relative = parent_relative
        # D1: joint discrete-continuous flow.  A per-token existence logit
        # ell flows in parallel with the pose.  It enters as a zero-init input
        # projection (so warm-start from a pose-only checkpoint is a no-op) and
        # leaves through a zero-init velocity head, sharing every attention /
        # conditioning path with the pose -- one network jointly solves
        # "which objects survive" and "where they go".
        self.mask_flow = mask_flow
        if mask_flow:
            self.ell_in = nn.Linear(1, d)
            nn.init.zeros_(self.ell_in.weight); nn.init.zeros_(self.ell_in.bias)
            self.ell_head = nn.Linear(d, 1)
            nn.init.zeros_(self.ell_head.weight); nn.init.zeros_(self.ell_head.bias)
        # Full Module 2: object<->wall-segment cross-attention (wall tokens).
        self.use_wall_tokens = wall_tokens
        if wall_tokens:
            self.wall_attn = WallCrossAttention(d, heads)
        # Route B, heterogeneous node conditioning: a Type-ID embedding tells
        # the model which tokens live in the global-anchor space (0) vs the
        # parent-relative child space (1), so it never confuses the two scales.
        if parent_relative:
            self.e_type = nn.Embedding(2, d)
        # ReRoom 2.0 Module 2: LEGO-Net-style pairwise relational geometry.
        # A DENSE (all-pairs) attention bias from the *current* state (relative
        # displacement + relative orientation), plus a per-object wall feature
        # (nearest-wall distance + normal misalignment) added to the token.
        # This gives the flow the "this object is far from the wall / mis-
        # aligned" and "these two are closer/farther than they should be"
        # signals that let it snap and align from the informative prior.  Both
        # heads are zero-initialised so a warm-start from a model without them
        # is a behavioural no-op that then learns the mechanism.
        self.use_geo_bias = geo_bias
        if geo_bias:
            self.geo_bias_mlp = nn.Sequential(
                nn.Linear(5, 64), nn.GELU(), nn.Linear(64, heads))
            nn.init.zeros_(self.geo_bias_mlp[-1].weight)
            nn.init.zeros_(self.geo_bias_mlp[-1].bias)
            self.wall_proj = nn.Sequential(
                nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
            nn.init.zeros_(self.wall_proj[-1].weight)
            nn.init.zeros_(self.wall_proj[-1].bias)
        self.e_cat = nn.Embedding(N_CAT, cat_emb)
        self.e_motif = nn.Embedding(N_MOTIF, motif_emb)
        self.e_room = nn.Embedding(len(ROOM_TYPES), 32)
        self.tok = nn.Linear(STATE_DIM + TOKEN_COND_DIM + cat_emb + motif_emb, d)
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.bound = BoundaryEncoder(in_dim=6, hidden=64, out_dim=64,
                                     depth=2, heads=4)
        self.glob = nn.Sequential(
            nn.Linear(GLOBAL_DIM + 64 + 64 + 32, d), nn.SiLU(), nn.Linear(d, d))
        self.edge = nn.Sequential(nn.Linear(EDGE_DIM, 128), nn.GELU(),
                                  nn.Linear(128, heads))
        # ---- Graph node stream (parallel to box).  Learns per-object relational
        # embeddings from category/motif and edge features via message passing.
        # It does *not* see the noisy state x_t, so it is invariant to the ODE
        # tau -- a pure semantic prior on which furniture categories co-occur
        # and how, complementary to the box stream that owns geometry.  Fusion
        # weight ``g_fuse`` is zero-initialised so warm-start from a checkpoint
        # without this branch is a no-op at start; the stream turns on as the
        # model learns to use it.
        self.g_in = nn.Linear(cat_emb + motif_emb, graph_dim)
        self.g_edge = nn.Sequential(nn.Linear(EDGE_DIM, 128), nn.GELU(),
                                    nn.Linear(128, graph_dim))
        self.g_layers = nn.ModuleList([
            GraphNodeLayer(graph_dim, graph_dim) for _ in range(graph_depth)])
        self.g_fuse = nn.Linear(graph_dim, d)
        nn.init.zeros_(self.g_fuse.weight)
        nn.init.zeros_(self.g_fuse.bias)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.head = nn.Linear(d, STATE_DIM)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.out_ada[1].weight)
        nn.init.zeros_(self.out_ada[1].bias)

    def _edge_bias(self, batch, N: int):
        ef = batch["edge_feat"]                       # (B, E, EDGE_DIM)
        ei = batch["edge_index"]                      # (B, 2, E)
        em = batch["edge_mask"]                       # (B, E)
        B, E, _ = ef.shape
        w = self.edge(ef) * em[..., None]             # (B, E, heads)
        bias = ef.new_zeros(B, self.heads, N, N)
        if E == 0:
            return bias
        src = ei[:, 0].clamp(0, N - 1)
        dst = ei[:, 1].clamp(0, N - 1)
        flat = (src * N + dst)                        # (B, E)
        w = w.transpose(1, 2)                         # (B, heads, E)
        bias = bias.view(B, self.heads, N * N)
        bias.scatter_add_(2, flat[:, None].expand(-1, self.heads, -1), w)
        rev = (dst * N + src)
        bias.scatter_add_(2, rev[:, None].expand(-1, self.heads, -1), w)
        return bias.view(B, self.heads, N, N)

    def _graph_stream(self, batch, N: int):
        """Semantic per-object embedding from category/motif + edges only.

        Runs a small message-passing GNN on the intent graph.  The point is to
        learn *which categories relate to which and how* (a bed with a
        nightstand at a certain relation, a sofa with a coffee table at
        another), independent of the current noisy geometry.  This gives the
        box stream a stable relational prior no matter where in the ODE we are.
        """
        cat = batch["cat"]                                    # (B, N)
        mot = batch["motif"]                                  # (B, N)
        mask = batch["mask"]                                  # (B, N) bool
        node_in = torch.cat([self.e_cat(cat), self.e_motif(mot)], dim=-1)
        h = self.g_in(node_in) * mask[..., None].float()
        e_feat = self.g_edge(batch["edge_feat"])              # (B, E, graph_dim)
        ei = batch["edge_index"].clamp(0, N - 1)              # (B, 2, E)
        em = batch["edge_mask"]                               # (B, E) bool
        for layer in self.g_layers:
            h = layer(h, ei, e_feat, em, mask)
        return h                                              # (B, N, graph_dim)

    def forward(self, x, tau, batch):
        """``v_theta(X_tau, tau, G_r, P_t, z_style)``; ``x`` is (B, N, 4).

        Fusion pattern (GraphGPS style, per-block):
          - Box stream: state + cond + cat + motif -> tok(Linear) -> h
          - Graph stream: cat + motif + edges -> GNN -> g_h (state-independent)
          - Every DiT block's adaLN reads per-token c = global(τ) + fuse(g_h),
            so the graph signal is injected at every layer instead of being
            fused once at the input and diluted through 12 blocks.
        """
        mask = batch["mask"]
        B, N, _ = x.shape
        # ReRoom 2.0 Step 2: when children are stored as parent-relative, the
        # token keeps the (scale-invariant) relative channels, but geometry
        # features (wall distance, dense pairwise bias) need the WORLD position,
        # so decode a world copy just for those.
        if self.parent_relative and "parent" in batch:
            xg = to_world(x, batch["parent"], batch["frame_h"])
            is_child = (batch["parent"] >= 0)                 # (B, N) bool
            anchor_gate = (~is_child).float()[..., None]      # (B, N, 1)
        else:
            xg = x
            anchor_gate = None
        # box stream: current DiT input from state + cond + cat + motif
        tok = torch.cat([x, batch["cond"], self.e_cat(batch["cat"]),
                         self.e_motif(batch["motif"])], dim=-1)
        h = self.tok(tok)
        if anchor_gate is not None:
            h = h + self.e_type(is_child.long())              # heterogeneous type
        if self.mask_flow:                                     # D1 existence in
            ell_in = batch.get("ell")
            if ell_in is None:                                 # prior: assume keep
                ell_in = x.new_zeros(x.shape[0], x.shape[1], 1)
            h = h + self.ell_in(ell_in)
        if self.use_geo_bias:
            # per-object wall feature from the current state: nearest-boundary
            # distance and how far the object's facing is from that wall's
            # normal.  Added to the token so every block can drive wall-snap.
            p = xg[..., :2]                                    # (B, N, 2) world
            bp = batch["boundary"][..., :2]                   # (B, Nb, 2)
            bn = batch["boundary"][..., 4:6]                  # (B, Nb, 2) normals
            d2 = torch.cdist(p, bp)                           # (B, N, Nb)
            kmin = d2.argmin(-1)                              # (B, N)
            dist_wall = d2.gather(-1, kmin[..., None]).squeeze(-1)     # (B, N)
            n_near = torch.gather(bn, 1, kmin[..., None].expand(-1, -1, 2))
            obj_dir = xg[..., 2:4]                            # (cos, sin) world
            misalign = 1.0 - (obj_dir * n_near).sum(-1).abs()          # (B, N)
            wfeat = torch.cat([dist_wall[..., None], misalign[..., None],
                               n_near], dim=-1)               # (B, N, 4)
            wp = self.wall_proj(wfeat)
            # children depend on their parent, not the room's outer walls -- mask
            # the wall pull off them so P_t never yanks a chair off its table.
            h = h + (wp * anchor_gate if anchor_gate is not None else wp)
        if self.use_wall_tokens:
            # explicit object<->wall attention over the boundary tokens
            h = self.wall_attn(h, xg, batch["boundary"], mask, node_gate=anchor_gate)
        # graph stream: parallel semantic branch, state-independent
        g_h = self._graph_stream(batch, N)                 # (B, N, gd)
        bfeat = self.bound(batch["boundary"])                 # (B, 64)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        g = self.glob(torch.cat([batch["glob"], bfeat, pooled[:, :64],
                                 self.e_room(batch["room_type"])], dim=-1))
        c_global = g + self.time(timestep_embedding(tau, self.d))     # (B, d)
        # per-token conditioning: global + per-node graph embedding.  g_fuse
        # is zero-init, so a warm-started checkpoint starts *identical* to the
        # baseline and learns to use the graph branch gradually.
        c_tok = c_global[:, None] + self.g_fuse(g_h)                  # (B, N, d)
        bias = self._edge_bias(batch, N)
        if self.use_geo_bias:
            # dense all-pairs geometric bias from the current (world) state.
            p = xg[..., :2]
            dp = p[:, :, None, :] - p[:, None, :, :]          # (B, N, N, 2)
            dist = dp.norm(dim=-1, keepdim=True)              # (B, N, N, 1)
            co = xg[..., 2]; si = xg[..., 3]                  # (B, N)
            dco = (co[:, :, None] - co[:, None, :])[..., None]
            dsi = (si[:, :, None] - si[:, None, :])[..., None]
            geo = torch.cat([dp, dist, dco, dsi], dim=-1)     # (B, N, N, 5)
            gb = self.geo_bias_mlp(geo).permute(0, 3, 1, 2)   # (B, heads, N, N)
            bias = bias + gb
        for blk in self.blocks:
            h = blk(h, c_tok, bias, mask)
        # output adaLN uses global c (no graph modulation on the final norm --
        # the per-layer graph signal has already had 12 chances to shape h).
        s, b = self.out_ada(c_global).chunk(2, dim=-1)
        h = self.out_norm(h) * (1 + s[:, None]) + b[:, None]
        pose_v = self.head(h) * mask[..., None]
        if self.mask_flow:                                    # D1 existence out
            ell_v = self.ell_head(h) * mask[..., None]        # (B, N, 1)
            return torch.cat([pose_v, ell_v], dim=-1)         # (B, N, 5)
        return pose_v
