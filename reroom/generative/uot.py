"""D2: unbalanced optimal-transport coupling for OT-CFM data.

The shipped data is hand-scripted forward-deform pairs, which a reviewer can
attack as shortcut learning (the model fits the author's warp).  This module
replaces the hand-coupling with a measure-theoretic one: given a real reference
scene and a real target scene (both human-designed, DIFFERENT object counts),
it solves an *unbalanced* OT problem to find the minimum-transport coupling
between their objects.

Unbalanced (KL soft-marginals) is required because the two rooms do not have
the same number of objects -- standard Monge/Kantorovich has no solution when
mass is not conserved.  The slack in the marginals is exactly the add/drop
signal: reference objects that transport little mass are *dropped* (feed D1's
mask=0), target objects that receive little are *added*.

Self-contained log-domain Sinkhorn (no POT dependency).
"""
from __future__ import annotations
import numpy as np


def sinkhorn_uot(cost, a, b, eps: float = 0.05, tau: float = 1.0,
                 iters: int = 200):
    """Log-domain unbalanced Sinkhorn.  cost (n,m); a (n,), b (m,) positive
    marginals; eps entropic reg; tau KL marginal penalty (large -> balanced).
    Returns coupling P (n,m)."""
    n, m = cost.shape
    la, lb = np.log(a + 1e-12), np.log(b + 1e-12)
    lam = tau / (tau + eps)                        # KL-unbalanced damping in [0,1)
    logK = -cost / eps                             # (n,m)
    lu = np.zeros(n); lv = np.zeros(m)             # log scaling potentials
    for _ in range(iters):
        lu = lam * (la - _lse(logK + lv[None, :], axis=1))
        lv = lam * (lb - _lse(logK + lu[:, None], axis=0))
    P = np.exp(lu[:, None] + logK + lv[None, :])
    return P


def _lse(A, axis):
    mx = np.max(A, axis=axis, keepdims=True)
    return (mx + np.log(np.sum(np.exp(A - mx), axis=axis, keepdims=True))).squeeze(axis)


def couple_scenes(ref, tgt, eps: float = 0.3, tau: float = 3.0):
    """UOT-couple two scenes' objects.  Cost = category mismatch + normalised
    centre distance in each room's MRR frame + size mismatch.  Returns
    (P, ref_keep, tgt_keep, matches) where ref_keep[i]=transported mass of ref
    object i (low -> drop), matches = argmax target per ref."""
    from ..retarget.optimizer import _mrr_frame, _map_point
    from ..geom.polygon import as_polygon
    ro = [o for o in ref.objects if o.keep]
    to = [o for o in tgt.objects if o.keep]
    n, m = len(ro), len(to)
    if n == 0 or m == 0:
        return None
    fr = _mrr_frame(as_polygon(ref.room)); ft = _mrr_frame(as_polygon(tgt.room))
    # normalised centres in each frame (translation/scale/rot free).
    # _mrr_frame -> (centre, axis1, axis2, half_long, half_short, angle)
    def nrm(o, f):
        cen, a1, a2, h1, h2, _ = f
        rel = np.array(o.xy) - np.asarray(cen)
        u = float(rel @ np.asarray(a1)) / (h1 + 1e-6)
        v = float(rel @ np.asarray(a2)) / (h2 + 1e-6)
        return np.array([u, v])
    Rp = np.array([nrm(o, fr) for o in ro])
    Tp = np.array([nrm(o, ft) for o in to])
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dpos = np.linalg.norm(Rp[i] - Tp[j])
            dcat = 0.0 if ro[i].category == to[j].category else 2.0
            dsz = abs(np.log(max(ro[i].size[0], 1e-2)) - np.log(max(to[j].size[0], 1e-2)))
            cost[i, j] = dpos + dcat + 0.3 * dsz
    a = np.ones(n) / n; b = np.ones(m) / m
    P = sinkhorn_uot(cost, a, b, eps=eps, tau=tau)
    ref_mass = P.sum(1) * n            # ~1 if fully matched, <1 if partially dropped
    tgt_mass = P.sum(0) * m
    matches = P.argmax(1)
    return {"P": P, "ref_mass": ref_mass, "tgt_mass": tgt_mass,
            "matches": matches, "ro": ro, "to": to, "cost": cost}
