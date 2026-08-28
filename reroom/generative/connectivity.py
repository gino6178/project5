"""D4: differentiable circulation / connectivity loss + affordance features.

PhyScene's walkability (``eval/physcene.py``) is a *hard* raster metric: erode
the floor by the robot radius, take the largest connected free component.  It is
not differentiable, so it can only score a finished layout, never shape one.

Two pieces here:

  * ``soft_connectivity_loss`` -- a differentiable surrogate.  Object footprints
    become smooth occupancy bumps on a grid; free space is ``relu(1-occ)``;
    reachability is diffused from the door cell by K differentiable max-pool
    dilations (a soft flood-fill); the loss penalises free floor the door
    cannot reach.  Gradients flow to object centres, so the flow can be trained
    to keep the room's circulation connected (component count -> 1).

  * ``affordance_channels`` -- per-boundary-sample affordance scalars (door
    swing proximity, window light-cone, visual axis) to concatenate onto the
    boundary conditioning, giving the model the functional constraints a raw
    polygon lacks.  (Feature extraction only; using it requires a retrain with
    the widened boundary channel.)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def soft_occupancy(p, half_w, half_d, fwd, grid_xy, sharp: float = 12.0):
    """Soft footprint occupancy on a grid.  p (B,N,2) metric centres; grid_xy
    (H,W,2) metric cell centres.  Returns occ (B,H,W) in [0,1)."""
    B, N, _ = p.shape
    H, Wd = grid_xy.shape[:2]
    g = grid_xy.reshape(1, 1, H * Wd, 2)                       # (1,1,HW,2)
    d = p[:, :, None, :] - g                                    # (B,N,HW,2)
    # axis-aligned half-extent approx (circumscribed): use max half-extent
    r = torch.sqrt(half_w ** 2 + half_d ** 2)[:, :, None]      # (B,N,1) metric
    dist = d.norm(dim=-1)                                       # (B,N,HW)
    bump = torch.sigmoid(sharp * (r - dist))                   # 1 inside, 0 out
    occ = bump.sum(1).clamp(max=1.0)                           # (B,HW)
    return occ.reshape(B, H, Wd)


def soft_connectivity_loss(p, half_w, half_d, fwd, room_mask, door_cell,
                           grid_xy, iters: int = 40):
    """Differentiable circulation loss.  room_mask (B,H,W) 1=inside floor;
    door_cell (B,2) int grid index of the doorway; grid_xy (H,W,2) metric.
    Penalises free floor unreachable from the door."""
    B, H, Wd = room_mask.shape
    occ = soft_occupancy(p, half_w, half_d, fwd, grid_xy)      # (B,H,W)
    free = (room_mask * (1.0 - occ)).clamp(0.0, 1.0)           # (B,H,W)
    # seed reachability at the door cell
    reach = torch.zeros_like(free)
    for b in range(B):
        dy, dx = int(door_cell[b, 0]), int(door_cell[b, 1])
        reach[b, dy, dx] = 1.0
    reach = reach * free
    # differentiable flood-fill: dilate then clamp to free
    for _ in range(iters):
        dil = F.max_pool2d(reach[:, None], kernel_size=3, stride=1, padding=1)[:, 0]
        reach = torch.minimum(free, torch.maximum(reach, dil))
    # unreachable free floor (per-cell), normalised by total free
    unreached = (free * (1.0 - reach)).sum(dim=(1, 2))
    total_free = free.sum(dim=(1, 2)).clamp(min=1.0)
    return (unreached / total_free).mean()


def make_room_grid(frame_h, res: int = 48, device="cpu"):
    """Metric grid of cell centres spanning the MRR frame [-h1,h1]x[-h2,h2].
    Returns grid_xy (H,W,2) in the SAME normalised units as the flow state
    (i.e. divide metric by frame_h) -- caller multiplies back to metric."""
    ys = torch.linspace(-1, 1, res, device=device)
    xs = torch.linspace(-1, 1, res, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1)                       # (H,W,2) normalised


# ------------------------------------------------------------------ affordance
def affordance_channels(room, boundary_pts):
    """Per-boundary-sample affordance scalars: (door_swing, light_cone,
    visual_axis).  boundary_pts (Nb,2) metric.  Returns (Nb,3) float32.

    * door_swing  -- proximity to any door opening (decaying kernel), marking
      the arc a door sweeps that must stay clear;
    * light_cone  -- proximity to any window, the daylight a layout should not
      wall off;
    * visual_axis -- alignment of the sample with the dominant door->centroid
      sightline, the circulation spine.
    """
    from ..geom.polygon import as_polygon
    poly = as_polygon(room)
    cen = np.array(poly.centroid.coords[0])
    doors = [o for o in getattr(room, "openings", []) if getattr(o, "kind", "door") == "door"]
    wins = [o for o in getattr(room, "openings", []) if getattr(o, "kind", "") == "window"]
    Nb = len(boundary_pts)
    out = np.zeros((Nb, 3), dtype=np.float32)
    def _pos(o):
        return np.array(getattr(o, "xy", getattr(o, "center", cen)))
    door_pos = [_pos(o) for o in doors] or [cen]
    for k, bp in enumerate(boundary_pts):
        dd = min(np.linalg.norm(bp - dp) for dp in door_pos)
        out[k, 0] = np.exp(-dd / 0.8)
        if wins:
            wd = min(np.linalg.norm(bp - _pos(o)) for o in wins)
            out[k, 1] = np.exp(-wd / 0.8)
        axis = cen - door_pos[0]; an = np.linalg.norm(axis) + 1e-9
        v = bp - door_pos[0]; vn = np.linalg.norm(v) + 1e-9
        out[k, 2] = float(np.clip((axis / an) @ (v / vn), 0, 1))
    return out
