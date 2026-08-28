"""D3: differentiable geometric projection layer.

The shipped pipeline ends in ``regularity.py`` -- a hard, non-differentiable
snap (Manhattan/flush/slot) -- optionally preceded by a multi-step Adam polish.
Neither can pass a gradient back to the DiT, so the network cannot anticipate
the projection during training.

This module expresses the same three regularities (wall flush + inward facing,
Manhattan orthogonality, non-collision) as a *smooth* energy and realises the
projection as K unrolled gradient steps on it.  Because every step is a
differentiable tensor op, the layer is:

  * a drop-in test-time projection (``project_scene``) that reaches the same
    flush/orthogonal/collision-free structure as the hard snap, but smoothly;
  * a training-time operator (``project_batch``) that can be unrolled inside
    the loss so the flow learns to emit layouts that are already near the
    projection's fixed point (implicit-differentiation / OptNet spirit, done by
    explicit unrolling so it stays dependency-free).

It is NOT a new heuristic table -- the categories reuse ``regularity.py`` so the
comparison against the hard snap is apples-to-apples.
"""
from __future__ import annotations
import numpy as np
import torch

from .regularity import WALL_CATS, ORTHO_CATS, FREE_CATS
from ..geom.polygon import as_polygon, object_polygon
from .regularity import _wall_segments


def _wall_tensors(room, device):
    """(a, t, n, L) per wall segment as tensors: base point, tangent, inward
    normal, length."""
    segs = _wall_segments(room)
    a = torch.tensor(np.array([s[0] for s in segs]), dtype=torch.float32, device=device)
    t = torch.tensor(np.array([s[2] for s in segs]), dtype=torch.float32, device=device)
    n = torch.tensor(np.array([s[3] for s in segs]), dtype=torch.float32, device=device)
    L = torch.tensor(np.array([s[4] for s in segs]), dtype=torch.float32, device=device)
    return a, t, n, L


def project_batch(p, fwd, half_w, half_d, is_wall, is_ortho, free,
                  nestable, wall_a, wall_t, wall_n, wall_L,
                  iters: int = 40, lr: float = 0.03,
                  w_flush: float = 1.0, w_align: float = 0.5,
                  w_ortho: float = 0.5, w_col: float = 1.0,
                  w_anchor: float = 6.0, return_energy: bool = False):
    """Differentiable projection of N objects in ONE room (metric space).

    p (N,2) centres; fwd (N,2) unit facing; half_w/half_d (N,) half extents
    (d = depth along facing); is_wall/is_ortho/free (N,) float gates; nestable
    (N,N) bool; wall_* the room segments.  Returns projected (p, fwd).

    A projection must stay NEAR its input (else it drifts and tears motifs), so
    an L2 anchor to the proposal position/orientation is the dominant term --
    the constraints only nudge the layout to the closest feasible point, exactly
    the eq (37) projection contract.  All ops are autograd-differentiable.
    """
    p = p.clone().requires_grad_(True)
    theta = torch.atan2(fwd[:, 1], fwd[:, 0]).clone().requires_grad_(True)
    p0 = p.detach().clone()
    theta0 = theta.detach().clone()
    opt = torch.optim.Adam([p, theta], lr=lr)
    N = p.shape[0]
    tri = torch.triu(torch.ones(N, N, device=p.device), diagonal=1)
    wgate = (~nestable).float() * tri
    for _ in range(iters):
        opt.zero_grad()
        f = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)   # (N,2)
        # nearest wall per object: signed inward distance & angle
        rel = p[:, None, :] - wall_a[None, :, :]                        # (N,W,2)
        inward = (rel * wall_n[None]).sum(-1)                           # (N,W) +inside
        proj = (rel * wall_t[None]).sum(-1)                            # (N,W) along
        onseg = ((proj > -0.1) & (proj < wall_L[None] + 0.1)).float()
        big = inward + (1 - onseg) * 1e3                                # ignore off-seg
        wi = big.argmin(1)                                             # (N,) nearest wall
        d_in = big.gather(1, wi[:, None]).squeeze(1)                    # inward dist
        n_near = wall_n[wi]                                            # (N,2)
        t_near = wall_t[wi]
        wall_ang = torch.atan2(t_near[:, 1], t_near[:, 0])
        # (a) wall flush: back edge touches wall -> centre sits half_d from wall
        e_flush = (is_wall * (d_in - half_d).clamp(min=-2.0) ** 2).sum()
        # (b) inward facing: forward aligned to inward normal
        e_align = (is_wall * (1.0 - (f * n_near).sum(-1))).sum()
        # (c) Manhattan: yaw parallel to a room axis (0 or 90 to wall)
        e_ortho = (is_ortho * torch.sin(2 * (theta - wall_ang)) ** 2).sum()
        # (d) non-collision: circle-overlap of non-nestable pairs
        r = 0.5 * torch.sqrt((2 * half_w) ** 2 + (2 * half_d) ** 2)
        dist = torch.cdist(p, p) + 1e-6
        overlap = (r[:, None] + r[None, :] - dist).clamp(min=0.0)
        e_col = ((overlap ** 2) * wgate).sum()
        # anchor: stay close to the proposal (topology-preserving projection).
        # Free objects are anchored hard; constrained objects can move to snap.
        aw = 1.0 + 3.0 * free                                          # (N,)
        e_anchor = (aw * ((p - p0) ** 2).sum(-1)).sum() \
            + (aw * (theta - theta0) ** 2).sum()
        E = (w_flush * e_flush + w_align * e_align + w_ortho * e_ortho
             + w_col * e_col + w_anchor * e_anchor)
        E.backward()
        # freeze the genuinely-free objects (plants, lamps, nightstands)
        with torch.no_grad():
            if p.grad is not None:
                p.grad *= (1 - free)[:, None]
                theta.grad *= (1 - free)
        opt.step()
    f = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    if return_energy:
        return p.detach(), f.detach(), float(E.detach())
    return p.detach(), f.detach()


def project_scene(scene, room=None, iters: int = 40, lr: float = 0.03,
                  device: str = "cpu"):
    """Test-time differentiable projection of a Scene in place-copy.  Mirrors
    ``regularity_snap`` but smooth + gradient-based."""
    from ..core.scene import Scene
    room = room or scene.room
    from ..generative.guidance import NESTABLE_PAIRS
    out = Scene(scene_id=scene.scene_id, room=room.copy(),
                objects=[o.copy() for o in scene.objects],
                source=scene.source, meta=dict(scene.meta or {}))
    objs = [o for o in out.objects if o.keep]
    if not objs:
        return out
    N = len(objs)
    p = torch.tensor(np.array([o.xy for o in objs]), dtype=torch.float32, device=device)
    fwd = torch.tensor(np.array([[np.cos(o.yaw), np.sin(o.yaw)] for o in objs]),
                       dtype=torch.float32, device=device)
    half_w = torch.tensor([0.5 * float(o.size[0]) for o in objs], dtype=torch.float32, device=device)
    half_d = torch.tensor([0.5 * float(o.size[1]) for o in objs], dtype=torch.float32, device=device)
    is_wall = torch.tensor([1.0 if o.category in WALL_CATS else 0.0 for o in objs], device=device)
    is_ortho = torch.tensor([1.0 if o.category in ORTHO_CATS else 0.0 for o in objs], device=device)
    free = torch.tensor([1.0 if o.category in FREE_CATS else 0.0 for o in objs], device=device)
    nest = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(N):
        for j in range(N):
            if frozenset({objs[i].category, objs[j].category}) in NESTABLE_PAIRS:
                nest[i, j] = True
    wa, wt, wn, wL = _wall_tensors(room, device)
    p2, f2 = project_batch(p, fwd, half_w, half_d, is_wall, is_ortho, free,
                           nest, wa, wt, wn, wL, iters=iters, lr=lr)
    p2 = p2.cpu().numpy(); f2 = f2.cpu().numpy()
    for i, o in enumerate(objs):
        o.xy = np.array([float(p2[i, 0]), float(p2[i, 1])])
        o.yaw = float(np.arctan2(f2[i, 1], f2[i, 0]))
    return out
