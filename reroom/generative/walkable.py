"""Differentiable rasterised walkability.

PhyScene computes walkability well because it rasterises: connectivity is natural
on a grid. But a hard flood fill is not differentiable, so it can only be used to
*score* a finished layout, never to shape one.

This module makes the same quantity differentiable, so it can act inside the
refinement loop (as a per-object feature) and in the loss (as a penalty on free
floor the agent cannot reach):

  1. soft occupancy   -- each object's footprint is rasterised with a smooth
                         (sigmoid) edge, so occupancy is differentiable in the
                         object's position and yaw;
  2. free space       -- room mask minus occupancy;
  3. soft reachability-- flood fill from a seed, relaxed as dilated max-pooling.
                         Doubling the dilation each round propagates 2^k cells,
                         one cell per round on a small grid (dilated jumps skip walls);
  4. blocked ratio    -- free floor that reachability never reaches. This is the
                         "no walkway here" signal: it is exactly the free area a
                         robot is cut off from.

Everything is batched and stays on the GPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["soft_occupancy", "soft_reachability", "walkability"]


def _room_grid(batch, G: int):
    """Rasterise the room polygon (from its boundary samples) to a GxG mask.

    We approximate the room by the inside-ness of the boundary samples: a cell is
    inside when it is on the inward side of its nearest boundary sample. That is
    the same predicate the violation features use, so the two agree.
    """
    fh = batch["frame_h"]                                     # (B,2) metric half-size
    bp = batch["boundary"][..., :2] * fh[:, None, :]          # (B,Nb,2) metres
    bn = batch["boundary"][..., 4:6]                          # (B,Nb,2) inward normals
    B = bp.shape[0]
    dev = bp.device
    # grid spans the room's metric extent
    lin = torch.linspace(-1.0, 1.0, G, device=dev)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    pts = torch.stack([gx, gy], -1).reshape(1, G * G, 2) * fh[:, None, :]   # (B,G*G,2)
    d2 = ((pts[:, :, None, :] - bp[:, None, :, :]) ** 2).sum(-1)            # (B,G*G,Nb)
    k = d2.argmin(-1)
    npt = torch.gather(bp, 1, k[..., None].expand(-1, -1, 2))
    nrm = torch.gather(bn, 1, k[..., None].expand(-1, -1, 2))
    inward = ((pts - npt) * nrm).sum(-1)                                    # >0 inside
    return torch.sigmoid(inward * 8.0).reshape(B, 1, G, G), pts.reshape(B, G, G, 2)


def soft_occupancy(x, batch, G: int = 64, sharp: float = 6.0):
    """(B,1,G,G) soft occupancy of the object footprints, differentiable in x."""
    fh = batch["frame_h"]
    p = x[..., :2] * fh[:, None, :]                             # (B,N,2) metric centres
    cos, sin = x[..., 2], x[..., 3]
    hw = torch.exp(batch["cond"][..., 0]) * 0.5                 # half width  (metres)
    hd = torch.exp(batch["cond"][..., 1]) * 0.5                 # half depth
    mask = batch["mask"].float()

    room, pts = _room_grid(batch, G)                            # pts (B,G,G,2) metric
    rel = pts[:, None, :, :, :] - p[:, :, None, None, :]        # (B,N,G,G,2)
    # rotate into each object's frame
    c = cos[:, :, None, None]; s = sin[:, :, None, None]
    lx = rel[..., 0] * c + rel[..., 1] * s
    ly = -rel[..., 0] * s + rel[..., 1] * c
    # smooth rectangle: inside when |lx|<hw and |ly|<hd
    ix = torch.sigmoid((hw[:, :, None, None] - lx.abs()) * sharp)
    iy = torch.sigmoid((hd[:, :, None, None] - ly.abs()) * sharp)
    occ = (ix * iy) * mask[:, :, None, None]
    occ = occ.amax(dim=1, keepdim=True)                         # union over objects
    return occ, room


def soft_reachability(free, seed, rounds: int | None = None, barrier: float = 12.0):
    """Flood fill relaxed as 3x3 max-pooling, propagating one cell per round.

    Dilated (jump-flooding) propagation was tried first and is WRONG for
    connectivity: a dilation-32 kernel samples cells 32 apart and simply steps
    over a wall four cells thick, so a room severed in two still registered as
    connected. Reachability must advance one cell at a time; the grid is kept
    small (32x32) so ~1.5*G rounds stay cheap.

    ``barrier`` sharpens the map used for propagation, because the soft footprint
    edges are otherwise semi-transparent; the measurement still uses the smooth
    free map so gradients survive.
    """
    G = free.shape[-1]
    rounds = rounds or int(1.5 * G)
    gate = torch.sigmoid((free - 0.5) * barrier)
    r = seed
    for _ in range(rounds):
        r = F.max_pool2d(r, kernel_size=3, stride=1, padding=1)
        r = torch.minimum(r, gate)
    return r


def walkability(x, batch, G: int = 32, rounds: int | None = None):
    """Returns (blocked_ratio, reach_map, free_map).

    blocked_ratio: fraction of free floor the flood fill cannot reach -- free
    space with no walkway to it. Differentiable in x, so it can be minimised.
    """
    occ, room = soft_occupancy(x, batch, G)
    free = (room * (1.0 - occ)).clamp(0.0, 1.0)                 # (B,1,G,G)
    # seed: the most free cell (a robust stand-in for the door when we have none)
    B = free.shape[0]
    flat = free.reshape(B, -1)
    idx = flat.argmax(dim=1)
    seed = torch.zeros_like(flat)
    seed.scatter_(1, idx[:, None], 1.0)
    seed = seed.reshape_as(free) * free
    reach = soft_reachability(free, seed, rounds)
    tot = free.sum(dim=(1, 2, 3)).clamp(min=1e-6)
    blocked = ((free - reach).clamp(min=0.0)).sum(dim=(1, 2, 3)) / tot
    return blocked, reach, free
