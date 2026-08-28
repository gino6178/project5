"""Unbalanced/partial entropic Gromov-Wasserstein SELECTION for capacity-forced
retargeting (the design-identity-preserving pruning mechanism).

Given a reference layout with a design graph and a target room that can only hold
K objects, decide WHICH K to keep so that the reference's relational structure is
maximally transported into the new room.  We couple two metric-measure spaces:

  * source: reference objects, structure C1 = pairwise distances, mass p_i biased
    by relational centrality (objects carrying more design-graph weight are
    harder to drop);
  * target: K "slots" (the affine-warped reference positions), structure
    C2 = pairwise distances in the target room.

An entropic Gromov-Wasserstein coupling preserves *relative* distances (design
identity), and a partial/unbalanced marginal (total target mass K/n < 1, excess
absorbed by a zero-cost dummy slot) drops the n-K least-transportable objects.
Self-contained (no POT dependency): entropic-GW outer loop of Peyré et al. with a
dummy-augmented Sinkhorn for the partial mass.
"""
from __future__ import annotations
import numpy as np


def _sinkhorn_partial(M, p, q, drop_mass, eps, n_iter=200):
    """Entropic OT with an extra zero-cost dummy target column absorbing
    ``drop_mass`` (partial transport).  Returns the n x m coupling (dummy col
    removed)."""
    n, m = M.shape
    Ma = np.concatenate([M, np.zeros((n, 1))], axis=1)          # dummy col, 0 cost
    qa = np.concatenate([q, [drop_mass]])
    K = np.exp(-Ma / eps)
    u = np.ones(n); v = np.ones(m + 1)
    for _ in range(n_iter):
        u = p / (K @ v + 1e-30)
        v = qa / (K.T @ u + 1e-30)
    T = u[:, None] * K * v[None, :]
    return T[:, :m]                                             # drop dummy


def gw_select(pos_ref, C_tgt_pos, rel_w, K, eps=0.02, out_iter=50, sink_iter=200):
    """Return indices of the K reference objects to keep.

    pos_ref  (n,2)  reference positions.
    C_tgt_pos(n,2)  the same objects' affine-warped target positions (the slots).
    rel_w    (n,)   per-object relational centrality (design-graph incident wt).
    """
    n = len(pos_ref)
    if K >= n:
        return list(range(n))
    C1 = np.linalg.norm(pos_ref[:, None, :] - pos_ref[None, :, :], axis=-1)
    C2 = np.linalg.norm(C_tgt_pos[:, None, :] - C_tgt_pos[None, :, :], axis=-1)
    C1 = C1 / (C1.max() + 1e-9); C2 = C2 / (C2.max() + 1e-9)
    # source mass: relationally-central objects carry more (harder to drop)
    p = 1.0 + rel_w / (rel_w.max() + 1e-9)
    p = p / p.sum()
    q = np.ones(n) / n * (K / n)          # total target mass K/n < 1 (partial)
    drop = 1.0 - q.sum()
    C1sq = C1 ** 2; C2sq = C2 ** 2
    T = np.outer(p, q)
    for _ in range(out_iter):
        # FULL entropic-GW square-loss cost (Peyre 2016): the row/col constant
        # terms are essential here because the partial dummy column has a fixed
        # zero cost, so real distortions must be on an absolute scale.  Entry
        # (i,j) = sum_kl (C1_ik - C2_jl)^2 T_kl >= 0, so the dummy (=0) absorbs
        # the *most-distorted* objects, i.e. we KEEP the best-preserved ones.
        Tr = T.sum(1); Tc = T.sum(0)
        Cost = (C1sq @ Tr)[:, None] - 2.0 * (C1 @ T @ C2) + (C2sq @ Tc)[None, :]
        T = _sinkhorn_partial(Cost, p, q, drop, eps, sink_iter)
    row_mass = T.sum(1)                    # mass retained on real slots (not dummy)
    return list(np.argsort(row_mass)[-K:])


def relational_keep(graph, K, must_keep=()):
    """Partial-relational-transport SELECTION: return a boolean keep-mask over
    ``graph.scene.objects`` of size K that maximises retained design-graph
    relational mass (the densest-retained-relational-subgraph objective, of
    which unbalanced GW on the design graph is the continuous relaxation).

    Solved by greedy removal -- repeatedly drop the object carrying the least
    incident relation weight among the currently-kept set -- which the offline
    probe showed preserves markedly more S_rel under pruning than the flow's
    learned mask.  ``must_keep`` (e.g. motif anchors) are never dropped.
    """
    import numpy as np
    ref = graph.scene
    n = len(ref.objects)
    if K >= n:
        return np.ones(n, dtype=bool)
    must = set(int(i) for i in must_keep)
    rels = [(r.i, r.j, float(r.weight)) for r in graph.relations]
    present = set(range(n))

    def incident(pres):
        w = {i: 0.0 for i in pres}
        for i, j, wt in rels:
            if i in pres and j in pres:
                w[i] += wt; w[j] += wt
        return w

    # greedy removal of the least relationally-central object.  (A motif-aware
    # variant that protects intact motif groups was tried and measured WORSE on
    # both S_rel and S_motif -- because the flow places all objects regardless of
    # keep, forcing whole motifs displaces relationally-valuable singletons; the
    # plain relational-coverage rule wins, so we keep it.)
    while len(present) > max(K, len(must & present)):
        w = incident(present)
        cand = [i for i in present if i not in must]
        if not cand:
            break
        present.discard(min(cand, key=lambda i: w[i]))
    mask = np.zeros(n, dtype=bool)
    for i in present:
        mask[i] = True
    return mask
