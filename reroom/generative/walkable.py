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

__all__ = ["soft_occupancy", "soft_reachability", "walkability",
           "object_reachability", "boundary_outside"]


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


def soft_occupancy(x, batch, G: int = 64, sharp: float = 6.0,
                   pad: float = 0.0, per_object: bool = False):
    """(B,1,G,G) soft occupancy of the object footprints, differentiable in x.

    ``pad`` inflates every footprint by that many metres per side. PhyScene's
    walkability strokes each box with the robot width before filling and erodes
    the floor by the same amount; inflating in metres reproduces that at any
    grid resolution, whereas pixel erosion needs cells finer than the robot
    half-width (0.15 m), i.e. G>84 for a 12.6 m room -- far past what fits in
    the refinement loop.
    """
    fh = batch["frame_h"]
    p = x[..., :2] * fh[:, None, :]                             # (B,N,2) metric centres
    cos, sin = x[..., 2], x[..., 3]
    hw = torch.exp(batch["cond"][..., 0]) * 0.5 + pad            # half width  (metres)
    hd = torch.exp(batch["cond"][..., 1]) * 0.5 + pad            # half depth
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
    occ = (ix * iy) * mask[:, :, None, None]                    # (B,N,G,G)
    if per_object:
        return occ.amax(dim=1, keepdim=True), room, occ
    return occ.amax(dim=1, keepdim=True), room                  # union over objects


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


def object_reachability(x, batch, G: int = 48, robot: float = 0.3,
                        rounds: int | None = None):
    """Per-object soft reachability in [0,1] -- the differentiable analogue of
    PhyScene's ``R_reach``.

    Their metric calls an object reachable when its footprint, dilated by the
    robot width, touches the *largest* connected free component. The blocked-area
    ratio we optimised in run 3 is a different quantity: a layout can leave very
    little unreachable floor while still walling an object off, which is what the
    L-shape and corridor numbers showed (reach 0.829 / 0.767 against PhyScene's
    0.940 / 0.870 while blocked area was already small). This targets the metric
    itself -- per object, not per unit area.
    """
    occ, room, per = soft_occupancy(x, batch, G, pad=robot * 0.5, per_object=True)
    free = (room * (1.0 - occ)).clamp(0.0, 1.0)
    B = free.shape[0]
    flat = free.reshape(B, -1)
    seed = torch.zeros_like(flat)
    seed.scatter_(1, flat.argmax(dim=1)[:, None], 1.0)
    reach = soft_reachability(free, seed.reshape_as(free) * free, rounds)
    # an object is reachable when reachable floor touches its (inflated) footprint
    ring = F.max_pool2d(per, kernel_size=3, stride=1, padding=1)   # (B,N,G,G)
    hit = (ring * reach).amax(dim=(2, 3))                          # (B,N)
    return hit, reach, free


def boundary_outside(x, batch):
    """Per-object metres poking outside the room, from the oriented box corners.

    PhyScene's ``R_out`` counts an object once if *any* footprint pixel leaves the
    floor, so the quantity to penalise is the worst corner, not the centre. The
    existing ``clearance`` feature uses the circumradius, which is conservative
    for long thin objects and, being only a feature, never entered the loss --
    hence run 3 fixed collision but left L/T out-of-floor at 0.270 / 0.252.
    """
    fh = batch["frame_h"]
    p = x[..., :2] * fh[:, None, :]
    cos, sin = x[..., 2:3], x[..., 3:4]
    hw = torch.exp(batch["cond"][..., 0:1]) * 0.5
    hd = torch.exp(batch["cond"][..., 1:2]) * 0.5
    sx = x.new_tensor([1.0, 1.0, -1.0, -1.0])
    sy = x.new_tensor([1.0, -1.0, 1.0, -1.0])
    ax = sx * hw; ay = sy * hd                                     # (B,N,4)
    cx = p[..., 0:1] + ax * cos - ay * sin
    cy = p[..., 1:2] + ax * sin + ay * cos
    corners = torch.stack([cx, cy], dim=-1)                        # (B,N,4,2)

    bp = batch["boundary"][..., :2] * fh[:, None, :]               # (B,Nb,2)
    bn = batch["boundary"][..., 4:6]
    B, N, C, _ = corners.shape
    q = corners.reshape(B, N * C, 2)
    d2 = ((q[:, :, None, :] - bp[:, None, :, :]) ** 2).sum(-1)
    k = d2.argmin(-1)
    npt = torch.gather(bp, 1, k[..., None].expand(-1, -1, 2))
    nrm = torch.gather(bn, 1, k[..., None].expand(-1, -1, 2))
    inward = ((q - npt) * nrm).sum(-1).reshape(B, N, C)
    return (-inward).clamp(min=0.0).amax(-1) * batch["mask"].float()


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
