"""Relation elasticity -- the core technical point of the plan (sections 4, 19).

    alpha_r in [0, 1]                                                  (8)
    d~_ij = (1 - alpha) d^r_ij + alpha * gamma_ij * d^r_ij             (9)
    alpha_ij = f_psi(c_i, c_j, rho_ij, m(i), m(j), g(P_r), g(P_t))    (45)

    "Preserve what is semantically rigid, adapt what is spatially elastic."(46)

``gamma_ij`` is the *ratio* of the target room's characteristic scale along the
relation direction to the source room's, so ``alpha = 0`` reproduces the
reference distance exactly (chair-to-table: human scale, rigid) and
``alpha = 1`` scales it with the room (sofa-to-TV: elastic).

Three estimators are provided, in increasing order of fidelity:

``PriorElasticity``   hand-specified per relation type -- always available.
``StatElasticity``    a log-log regression of observed distance against room
                      scale, fitted on a corpus; this is a genuine elasticity
                      in the economic sense, ``d log d / d log gamma``.
``NeuralElasticity``  an MLP conditioned on the full context of eq. (45); the
                      elasticity is read out as the autograd sensitivity
                      ``d log d_hat / d log gamma``, so the model is trained
                      with ordinary distance regression and never needs
                      elasticity labels.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..core.categories import PRIORS, prior
from ..core.scene import Scene
from ..geom.polygon import as_polygon, characteristic_scale, floor_descriptor

__all__ = [
    "RelationContext", "ElasticityModel", "PriorElasticity", "StatElasticity",
    "NeuralElasticity", "desired_distance", "gamma_for", "KIND_ALPHA_PRIOR",
    "collect_elasticity_samples", "load_elasticity",
]

# Hand priors: how much each relation type may stretch with the room.
KIND_ALPHA_PRIOR = {
    "support": 0.0,
    "symmetric": 0.05,
    "grouped_with": 0.20,
    "near": 0.15,
    "centered_with": 0.30,
    "aligned": 0.45,
    "surrounds": 0.02,
    "in_front_of": 0.45,
    "behind": 0.45,
    "left_of": 0.55,
    "right_of": 0.55,
    "facing": 0.75,
    "face_to_face": 0.80,
    "against_wall": 1.00,
}

# Category pairs whose distance is set by the human body, not by the room.
_RIGID_PAIRS = {
    frozenset({"dining_table", "dining_chair"}): 0.0,
    frozenset({"dining_table", "barstool"}): 0.0,
    frozenset({"desk", "office_chair"}): 0.0,
    frozenset({"dressing_table", "stool"}): 0.0,
    frozenset({"double_bed", "nightstand"}): 0.05,
    frozenset({"single_bed", "nightstand"}): 0.05,
    frozenset({"tv_stand", "tv"}): 0.0,
    frozenset({"sofa", "coffee_table"}): 0.15,
    frozenset({"l_sofa", "coffee_table"}): 0.15,
    frozenset({"piano", "stool"}): 0.0,
}
_ELASTIC_PAIRS = {
    frozenset({"sofa", "tv_stand"}): 0.85,
    frozenset({"sofa", "tv"}): 0.85,
    frozenset({"l_sofa", "tv_stand"}): 0.85,
    frozenset({"double_bed", "wardrobe"}): 0.80,
    frozenset({"sofa", "armchair"}): 0.55,
    frozenset({"dining_table", "sideboard"}): 0.80,
}


@dataclass
class RelationContext:
    """Everything eq. (45) conditions on."""

    cat_i: str
    cat_j: str
    kind: str
    motif_i: str = "none"
    motif_j: str = "none"
    same_motif: bool = False
    rigidity: float = 0.5              # of the shared motif, if any
    g_src: np.ndarray | None = None    # floor descriptor of P_r
    g_tgt: np.ndarray | None = None    # floor descriptor of P_t
    gamma: float = 1.0                 # target/source scale along the relation
    d_ref: float = 1.0                 # reference distance in metres
    gamma_src_abs: float = 4.0         # source characteristic scale, metres


def gamma_for(direction: np.ndarray, src_poly, tgt_poly) -> float:
    """``gamma_ij``: how much the room grew along the relation's direction."""
    gs = characteristic_scale(as_polygon(src_poly), direction)
    gt = characteristic_scale(as_polygon(tgt_poly), direction)
    if gs < 1e-6:
        return 1.0
    return float(np.clip(gt / gs, 0.25, 4.0))


def desired_distance(d_ref: float, alpha: float, gamma: float) -> float:
    """Eq. (9)."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return float((1.0 - a) * d_ref + a * gamma * d_ref)


# --------------------------------------------------------------------------
class ElasticityModel:
    """Interface: context -> alpha in [0, 1]."""

    def alpha(self, ctx: RelationContext) -> float:      # pragma: no cover
        raise NotImplementedError

    def alphas(self, ctxs: list[RelationContext]) -> np.ndarray:
        return np.array([self.alpha(c) for c in ctxs], dtype=float)


class PriorElasticity(ElasticityModel):
    """Hand-specified, no fitting required.  The plan's stage-one estimator."""

    def __init__(self, rigid_weight: float = 0.65):
        self.rigid_weight = rigid_weight

    def alpha(self, ctx: RelationContext) -> float:
        a = KIND_ALPHA_PRIOR.get(ctx.kind, 0.4)
        pair = frozenset({ctx.cat_i, ctx.cat_j})
        if pair in _RIGID_PAIRS:
            a = min(a, _RIGID_PAIRS[pair])
        elif pair in _ELASTIC_PAIRS:
            a = max(a, _ELASTIC_PAIRS[pair])
        if ctx.same_motif:
            a *= (1.0 - self.rigid_weight * ctx.rigidity)
        # a relation that is already tight in absolute terms is body scale
        if ctx.d_ref < 0.9:
            a *= 0.35
        return float(np.clip(a, 0.0, 1.0))


class StatElasticity(ElasticityModel):
    """Log-log regression of distance against room scale, fitted per bucket.

    For every (category pair, relation kind) bucket we fit

        log d = b + alpha * log gamma_abs

    across a corpus of scenes whose rooms differ in size.  ``alpha`` is exactly
    the elasticity of eq. (8).  Buckets with too few samples or too little
    variation in ``gamma_abs`` back off to the relation-kind bucket and then to
    the hand prior.
    """

    def __init__(self, min_samples: int = 24, min_span: float = 0.20,
                 fallback: ElasticityModel | None = None):
        self.min_samples = min_samples
        self.min_span = min_span
        self.pair_alpha: dict[str, tuple[float, int, float]] = {}
        self.kind_alpha: dict[str, tuple[float, int, float]] = {}
        self.fallback = fallback or PriorElasticity()

    # -- fitting ---------------------------------------------------------
    @staticmethod
    def _key(cat_i: str, cat_j: str, kind: str) -> str:
        a, b = sorted((cat_i, cat_j))
        return f"{a}|{b}|{kind}"

    def fit(self, samples: list[RelationContext]) -> "StatElasticity":
        pair_buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
        kind_buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for s in samples:
            if s.d_ref <= 1e-3 or s.gamma_src_abs <= 1e-3:
                continue
            pt = (math.log(s.gamma_src_abs), math.log(s.d_ref))
            pair_buckets[self._key(s.cat_i, s.cat_j, s.kind)].append(pt)
            kind_buckets[s.kind].append(pt)
        self.pair_alpha = {k: v for k, v in
                           ((k, self._fit_bucket(v)) for k, v in pair_buckets.items())
                           if v is not None}
        self.kind_alpha = {k: v for k, v in
                           ((k, self._fit_bucket(v)) for k, v in kind_buckets.items())
                           if v is not None}
        return self

    def _fit_bucket(self, pts: list[tuple[float, float]]):
        if len(pts) < self.min_samples:
            return None
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        if float(x.max() - x.min()) < self.min_span:
            return None
        xm, ym = x.mean(), y.mean()
        var = float(((x - xm) ** 2).sum())
        if var < 1e-9:
            return None
        slope = float(((x - xm) * (y - ym)).sum() / var)
        resid = y - (ym + slope * (x - xm))
        ss_tot = float(((y - ym) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 1e-9 else 0.0
        return (float(np.clip(slope, 0.0, 1.0)), len(pts), r2)

    # -- inference -------------------------------------------------------
    def alpha(self, ctx: RelationContext) -> float:
        hit = self.pair_alpha.get(self._key(ctx.cat_i, ctx.cat_j, ctx.kind))
        if hit is None:
            hit = self.kind_alpha.get(ctx.kind)
        if hit is None:
            return self.fallback.alpha(ctx)
        a, n, r2 = hit
        # shrink towards the prior when the fit is weak
        w = float(np.clip(r2, 0.0, 1.0)) * float(np.clip(n / (n + 60.0), 0.0, 1.0))
        return float(np.clip(w * a + (1 - w) * self.fallback.alpha(ctx), 0.0, 1.0))

    # -- io --------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"pair": self.pair_alpha, "kind": self.kind_alpha,
                       "min_samples": self.min_samples}, fh)

    @staticmethod
    def load(path: str) -> "StatElasticity":
        m = StatElasticity()
        with open(path) as fh:
            d = json.load(fh)
        m.pair_alpha = {k: tuple(v) for k, v in d["pair"].items()}
        m.kind_alpha = {k: tuple(v) for k, v in d["kind"].items()}
        return m

    def report(self, top: int = 25) -> list[tuple[str, float, int, float]]:
        rows = [(k, a, n, r2) for k, (a, n, r2) in self.pair_alpha.items()]
        rows.sort(key=lambda r: -r[2])
        return rows[:top]


# --------------------------------------------------------------------------
def collect_elasticity_samples(scenes, graphs=None) -> list[RelationContext]:
    """Turn a corpus of scenes into (context, distance) elasticity samples."""
    from .relations import build_scene_graph
    from .motifs import build_motifs, motif_of

    out: list[RelationContext] = []
    for k, scene in enumerate(scenes):
        g = graphs[k] if graphs is not None else build_motifs(build_scene_graph(scene))
        poly = as_polygon(scene.room)
        gd = floor_descriptor(poly)
        objs = scene.objects
        for r in g.relations:
            if r.kind == "support":
                continue
            a, b = objs[r.i], objs[r.j]
            d = b.xy - a.xy
            if np.linalg.norm(d) < 1e-6:
                continue
            gs = characteristic_scale(poly, d)
            mi, mj = motif_of(g, r.i), motif_of(g, r.j)
            same = mi is not None and mj is not None and mi.mid == mj.mid
            out.append(RelationContext(
                cat_i=a.category, cat_j=b.category, kind=r.kind,
                motif_i=mi.name if mi else "none", motif_j=mj.name if mj else "none",
                same_motif=same,
                rigidity=(mi.rigidity if same and mi else 0.4),
                g_src=gd, g_tgt=gd, gamma=1.0,
                d_ref=float(r.dist), gamma_src_abs=float(gs)))
    return out


# --------------------------------------------------------------------------
# neural estimator
# --------------------------------------------------------------------------
_KINDS: tuple[str, ...] = tuple(KIND_ALPHA_PRIOR.keys())
_CATS: tuple[str, ...] = tuple(sorted(PRIORS.keys()))
_MOTIF_NAMES = ("none", "sleeping", "dining", "conversation", "media", "work",
                "dressing", "reading", "storage", "hearth", "music", "cluster")

_SHAPE_IDX = [6, 7, 8, 9, 10]        # aspect, convexity, rect_fill, compact, reflex


def _shape_feats(g: np.ndarray | None) -> np.ndarray:
    if g is None:
        return np.array([1.0, 1.0, 1.0, 0.78, 0.0], dtype=np.float32)
    return np.asarray(g, dtype=np.float32)[_SHAPE_IDX]


def context_features(ctx: RelationContext) -> tuple[np.ndarray, float]:
    """Static features plus the differentiable ``log gamma`` input."""
    ci = _CATS.index(ctx.cat_i) if ctx.cat_i in _CATS else _CATS.index("misc")
    cj = _CATS.index(ctx.cat_j) if ctx.cat_j in _CATS else _CATS.index("misc")
    kd = _KINDS.index(ctx.kind) if ctx.kind in _KINDS else 0
    mi = _MOTIF_NAMES.index(ctx.motif_i) if ctx.motif_i in _MOTIF_NAMES else 0
    mj = _MOTIF_NAMES.index(ctx.motif_j) if ctx.motif_j in _MOTIF_NAMES else 0
    cont = np.concatenate([
        _shape_feats(ctx.g_src), _shape_feats(ctx.g_tgt),
        np.array([float(ctx.same_motif), ctx.rigidity,
                  prior(ctx.cat_i).anchor, prior(ctx.cat_j).anchor,
                  math.log(max(prior(ctx.cat_i).front_clear, 0.02)),
                  math.log(max(prior(ctx.cat_j).front_clear, 0.02))],
                 dtype=np.float32),
    ]).astype(np.float32)
    idx = np.array([ci, cj, kd, mi, mj], dtype=np.int64)
    log_gamma = math.log(max(ctx.gamma_src_abs, 1e-3))
    return np.concatenate([idx.astype(np.float32), cont]), log_gamma


class NeuralElasticity(ElasticityModel):
    """``f_psi`` of eq. (45), as a *varying-coefficient* distance model.

    A plain regressor for ``log d`` can fit the data perfectly while having an
    arbitrary local slope in ``log gamma``, so reading the elasticity off its
    autograd gradient is unreliable -- measured on 3D-FRONT it put
    chair-to-table at 0.43, which is nonsense.  Instead the elasticity is made
    an explicit parameter of the model:

        log d_hat(ctx, gamma) = b(ctx) + alpha(ctx) * (log gamma - log gamma_bar)
        alpha(ctx) = sigmoid(a(ctx)) in [0, 1]

    This is a conditional log-log regression whose slope is shared across
    contexts by the network, so rare category pairs borrow strength from common
    ones while common pairs keep their own value.  ``alpha`` is then read
    directly, is in range by construction, and is exactly the quantity eq. (8)
    defines.  The supervision remains ordinary distance regression -- no
    elasticity labels are ever needed.
    """

    def __init__(self, hidden: int = 192, emb: int = 24, device: str = "cpu",
                 fallback: ElasticityModel | None = None, blend: float = 0.8):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.device = torch.device(device)
        self.blend = blend
        self.fallback = fallback or PriorElasticity()
        self.log_gamma_bar = math.log(4.0)
        n_cont = 5 + 5 + 6

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.e_cat = nn.Embedding(len(_CATS), emb)
                self.e_kind = nn.Embedding(len(_KINDS), emb)
                self.e_motif = nn.Embedding(len(_MOTIF_NAMES), emb // 2)
                d_in = 2 * emb + emb + 2 * (emb // 2) + n_cont
                self.trunk = nn.Sequential(
                    nn.Linear(d_in, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden), nn.SiLU())
                self.base = nn.Linear(hidden, 1)
                self.slope = nn.Linear(hidden, 1)
                nn.init.zeros_(self.slope.weight)
                nn.init.constant_(self.slope.bias, -1.0)   # start near alpha~0.27

            def features(self, idx, cont):
                ei = self.e_cat(idx[:, 0])
                ej = self.e_cat(idx[:, 1])
                ek = self.e_kind(idx[:, 2])
                mi = self.e_motif(idx[:, 3])
                mj = self.e_motif(idx[:, 4])
                # the model must be symmetric in (i, j): a relation is a pair
                pair = ei + ej
                return self.trunk(torch.cat([pair, ei * ej, ek, mi + mj,
                                             mi * mj, cont], -1))

            def alpha(self, idx, cont):
                return torch.sigmoid(self.slope(self.features(idx, cont))).squeeze(-1)

            def forward(self, idx, cont, log_gamma, log_gamma_bar):
                h = self.features(idx, cont)
                b = self.base(h).squeeze(-1)
                a = torch.sigmoid(self.slope(h)).squeeze(-1)
                return b + a * (log_gamma - log_gamma_bar), a

        self.net = Net().to(self.device)
        self.fitted = False

    # -- data -----------------------------------------------------------
    def _pack(self, ctxs: list[RelationContext]):
        torch = self.torch
        feats = [context_features(c) for c in ctxs]
        idx = torch.tensor(np.stack([f[0][:5] for f in feats]).astype(np.int64),
                           device=self.device)
        cont = torch.tensor(np.stack([f[0][5:] for f in feats]),
                            dtype=torch.float32, device=self.device)
        lg = torch.tensor(np.array([f[1] for f in feats]), dtype=torch.float32,
                          device=self.device)
        return idx, cont, lg

    def fit(self, samples: list[RelationContext], epochs: int = 120,
            batch: int = 4096, lr: float = 2e-3, verbose: bool = False,
            val_frac: float = 0.1, alpha_reg: float = 0.0,
            anchor: "StatElasticity | None" = None,
            anchor_weight: float = 1.0, anchor_n0: float = 400.0
            ) -> "NeuralElasticity":
        """Fit ``log d`` and, optionally, anchor ``alpha`` to a bucket fit.

        The closed-form per-bucket regression is unbiased but only exists where
        a bucket has enough samples; the network generalises but, left alone,
        can trade slope against intercept and land on values the data does not
        support.  Anchoring the network's ``alpha`` to the bucket estimate with
        a weight that grows with the bucket's sample count gives a hierarchical
        estimator: data-driven where data is plentiful, smoothed elsewhere.
        """
        torch = self.torch
        samples = [s for s in samples if s.d_ref > 1e-3 and s.gamma_src_abs > 1e-3]
        if len(samples) < 64:
            return self
        idx, cont, lg = self._pack(samples)
        y = torch.tensor(np.array([math.log(s.d_ref) for s in samples]),
                         dtype=torch.float32, device=self.device)
        self.log_gamma_bar = float(lg.mean())
        n = len(samples)
        perm0 = torch.randperm(n, device=self.device)
        n_val = int(n * val_frac)
        val_ix, tr_ix = perm0[:n_val], perm0[n_val:]
        if anchor is not None:
            tgt = np.zeros(n, dtype=np.float32)
            wgt = np.zeros(n, dtype=np.float32)
            for i, sm in enumerate(samples):
                hit = anchor.pair_alpha.get(
                    anchor._key(sm.cat_i, sm.cat_j, sm.kind))
                if hit is None:
                    continue
                al, nb, r2 = hit
                tgt[i] = al
                wgt[i] = nb / (nb + anchor_n0)
            a_tgt = torch.tensor(tgt, device=self.device)
            a_w = torch.tensor(wgt, device=self.device)
        else:
            a_tgt = a_w = None

        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(epochs):
            perm = tr_ix[torch.randperm(len(tr_ix), device=self.device)]
            tot = 0.0
            for k in range(0, len(perm), batch):
                sel = perm[k:k + batch]
                pred, a = self.net(idx[sel], cont[sel], lg[sel], self.log_gamma_bar)
                loss = torch.nn.functional.mse_loss(pred, y[sel])
                if a_tgt is not None:
                    loss = loss + anchor_weight * (
                        a_w[sel] * (a - a_tgt[sel]) ** 2).mean()
                if alpha_reg > 0:
                    loss = loss + alpha_reg * (a ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tot += float(loss) * len(sel)
            sched.step()
            if verbose and (ep % 20 == 0 or ep == epochs - 1):
                msg = f"  [elasticity] epoch {ep:3d}  train {tot / len(perm):.4f}"
                if n_val:
                    msg += f"  val_r2 {self.r2(idx[val_ix], cont[val_ix], lg[val_ix], y[val_ix]):.4f}"
                print(msg, flush=True)
        self.fitted = True
        if n_val:
            self.val_r2 = float(self.r2(idx[val_ix], cont[val_ix], lg[val_ix], y[val_ix]))
        return self

    def r2(self, idx, cont, lg, y) -> float:
        torch = self.torch
        with torch.no_grad():
            pred, _ = self.net(idx, cont, lg, self.log_gamma_bar)
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / max(ss_tot, 1e-9)

    # -- inference -------------------------------------------------------
    def alpha(self, ctx: RelationContext) -> float:
        return float(self.alphas([ctx])[0])

    def alphas(self, ctxs: list[RelationContext]) -> np.ndarray:
        if not self.fitted or not ctxs:
            return np.array([self.fallback.alpha(c) for c in ctxs], dtype=float)
        torch = self.torch
        idx, cont, _ = self._pack(ctxs)
        with torch.no_grad():
            a = self.net.alpha(idx, cont).cpu().numpy()
        pr = np.array([self.fallback.alpha(c) for c in ctxs], dtype=float)
        return np.clip(self.blend * a + (1.0 - self.blend) * pr, 0.0, 1.0)

    def save(self, path: str) -> None:
        self.torch.save({"state": self.net.state_dict(), "fitted": self.fitted,
                         "log_gamma_bar": self.log_gamma_bar,
                         "blend": self.blend}, path)

    def load(self, path: str) -> "NeuralElasticity":
        d = self.torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(d["state"])
        self.fitted = bool(d.get("fitted", True))
        self.log_gamma_bar = float(d.get("log_gamma_bar", math.log(4.0)))
        self.blend = float(d.get("blend", self.blend))
        return self


def load_elasticity(path: str | None, device: str = "cpu") -> ElasticityModel:
    """Load whichever estimator is available at ``path``, else the prior."""
    if path and os.path.exists(path):
        if path.endswith(".json"):
            return StatElasticity.load(path)
        m = NeuralElasticity(device=device)
        return m.load(path)
    return PriorElasticity()
