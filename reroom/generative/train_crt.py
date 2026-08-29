"""Training for the Constraint-Refinement Transformer (project5).

Direct supervision, no flow matching: the model starts at the affine transplant
and produces the final layout in one forward pass through L refinement blocks.
The objective is the one we deploy (DESIGN.md §3):

  recon        weighted L2 to the real layout (motif children upweighted)
  terminal     E_geo(x_L) -- whatever is still infeasible at the last block is
               charged to the model; this is the pressure that removes the need
               for any post-processing
  monotone     deep supervision: violation must not increase from block to block
  relation     motif rigidity against the reference offsets, so feasibility is
               not bought by pulling the design apart
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .refiner import GraphRefinementTransformer
from .train import RetargetPairs, _collate_fn
from .graphcore import gap_supervision, graph_violation
from .walkable import walkability

__all__ = ["CRTConfig", "train_crt"]


@dataclass
class CRTConfig:
    epochs: int = 120
    batch: int = 128
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-4
    d_model: int = 384
    n_blocks: int = 6
    heads: int = 8
    # loss weights
    w_recon: float = 1.0
    w_terminal: float = 1.0        # E_geo at the final block
    w_monotone: float = 0.2        # penalise violation increasing across blocks
    w_relation: float = 2.0        # motif rigidity vs the reference
    w_walk: float = 1.0            # blocked-walkway penalty: free floor the
                                   # rasterised flood fill cannot reach. This is
                                   # the axis PhyScene wins on (walkable 0.963 vs
                                   # project4's 0.900), so we optimise it directly.
    w_gap: float = 1.0             # supervision for the learned pair spacing
    child_weight: float = 10.0     # upweight motif children (as in project4)
    # data
    levels: tuple = (1, 2, 3, 4, 5)
    l1_range: tuple = (0.6, 1.7)
    use_hybrid: bool = True
    hybrid_jaccard: float = 0.6
    hybrid_max_deg: float = 30.0
    # runtime
    workers: int = 0               # remote /dev/shm is tiny and shared
    ema: float = 0.999
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda:0"
    out: str = "outputs/crt"
    log_every: int = 50


def _relation_loss(x, batch):
    """Motif rigidity: keep each child's offset to its parent at the reference
    offset (metric).  Mirrors project4's rel_loss but computed directly on the
    final layout rather than on a flow endpoint."""
    fh = batch["frame_h"]
    p = x[..., :2] * fh[:, None, :]
    ref = batch["cond"][..., 10:12] * fh[:, None, :]
    par = batch["parent"]
    has = (par >= 0).float() * batch["mask"].float()
    pidx = par.clamp(min=0)
    cur = p - torch.gather(p, 1, pidx[..., None].expand(-1, -1, 2))
    rf = ref - torch.gather(ref, 1, pidx[..., None].expand(-1, -1, 2))
    return ((cur - rf).norm(dim=-1) * has).sum() / has.sum().clamp(min=1)


def train_crt(scenes, val_scenes=None, cfg: CRTConfig | None = None, elasticity=None):
    cfg = cfg or CRTConfig()
    os.makedirs(cfg.out, exist_ok=True)
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    if cfg.use_hybrid:
        from .xscene import HybridPairs, build_pair_index_filtered
        print("[crt] building cross-scene pair index...", flush=True)
        pair_index = build_pair_index_filtered(
            scenes, thresh=cfg.hybrid_jaccard, max_deg=cfg.hybrid_max_deg,
            max_partners=16, seed=cfg.seed)
        print(f"[crt] pairs for {len(pair_index)}/{len(scenes)} scenes", flush=True)
        train_ds = HybridPairs(scenes, pair_index, forward_frac=1.0,
                               elasticity=elasticity, levels=cfg.levels,
                               l1_range=cfg.l1_range, l1_u_shape=False,
                               max_deg=cfg.hybrid_max_deg, seed=cfg.seed,
                               cache=(cfg.workers == 0))
    else:
        train_ds = RetargetPairs(scenes, cfg.levels, elasticity, seed=cfg.seed,
                                 l1_range=cfg.l1_range, cache=(cfg.workers == 0))

    dl = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,
                    num_workers=cfg.workers, collate_fn=_collate_fn,
                    drop_last=True, persistent_workers=False)
    val_dl = None
    if val_scenes:
        val_ds = RetargetPairs(val_scenes, cfg.levels, elasticity, seed=cfg.seed + 1,
                               l1_range=cfg.l1_range, cache=(cfg.workers == 0))
        val_dl = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False,
                            num_workers=0, collate_fn=_collate_fn)

    model = GraphRefinementTransformer(cfg.d_model, cfg.n_blocks, cfg.heads).to(dev)
    ema = GraphRefinementTransformer(cfg.d_model, cfg.n_blocks, cfg.heads).to(dev)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"[crt] params {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * max(len(dl), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr,
                                                total_steps=max(steps, 1), pct_start=0.06)

    best = float("inf"); step = 0; history = []
    for ep in range(cfg.epochs):
        model.train()
        agg = {k: 0.0 for k in ("recon", "term", "mono", "rel", "walk", "gap", "move")}
        cnt = 0
        for batch in dl:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            x, trace = model(batch, return_trace=True)
            # displacement from the affine transplant it starts at. The first run
            # collapsed to the identity (0.048 m) and scored exactly like the
            # affine warp, so this is logged every epoch to catch it at once.
            with torch.no_grad():
                x0 = batch["cond"][..., 10:14]
                mk = batch["mask"].float()
                move = (((x[..., :2] - x0[..., :2]).norm(dim=-1) * mk).sum()
                        / mk.sum().clamp(min=1) * batch["frame_h"].mean())

            m = batch["mask"][..., None].float()
            is_child = (batch["parent"] >= 0).float()[..., None]
            w = m * (1.0 + (cfg.child_weight - 1.0) * is_child)
            recon = ((x - batch["state"]) ** 2 * w).sum() / w.sum().clamp(min=1)

            term = graph_violation(x, batch, model.gap)[1].mean()

            mono = x.new_zeros(())
            if cfg.w_monotone > 0.0 and len(trace) > 1:
                prev = graph_violation(trace[0], batch, model.gap)[1].mean()
                for t in trace[1:]:
                    cur = graph_violation(t, batch, model.gap)[1].mean()
                    mono = mono + (cur - prev).clamp(min=0.0)   # only penalise increases
                    prev = cur

            rel = _relation_loss(x, batch)
            walk = walkability(x, batch, G=32)[0].mean() if cfg.w_walk > 0 else x.new_zeros(())
            # trains what "correct spacing" means, from the real layouts themselves
            gapsup = gap_supervision(batch, model.gap)

            loss = (cfg.w_recon * recon + cfg.w_terminal * term
                    + cfg.w_monotone * mono + cfg.w_relation * rel
                    + cfg.w_walk * walk + cfg.w_gap * gapsup)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(cfg.ema).add_(pm, alpha=1 - cfg.ema)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)

            n = int(batch["mask"].sum())
            agg["recon"] += float(recon) * n; agg["term"] += float(term) * n
            agg["mono"] += float(mono) * n; agg["rel"] += float(rel) * n
            agg["walk"] += float(walk) * n; agg["gap"] += float(gapsup) * n
            agg["move"] += float(move) * n
            cnt += n; step += 1
            if step % cfg.log_every == 0:
                print(f"  ep {ep} step {step}/{steps} " +
                      " ".join(f"{k} {v/max(cnt,1):.4f}" for k, v in agg.items()), flush=True)

        row = {"epoch": ep, **{k: v / max(cnt, 1) for k, v in agg.items()}}
        if val_dl is not None:
            ema.eval(); vr = vt = vn = 0.0
            with torch.no_grad():
                for b in val_dl:
                    b = {k: v.to(dev) for k, v in b.items()}
                    xv = ema(b)
                    mm = b["mask"][..., None].float()
                    vr += float(((xv - b["state"]) ** 2 * mm).sum() / mm.sum().clamp(min=1)) * int(b["mask"].sum())
                    vt += float(graph_violation(xv, b, ema.gap)[1].mean()) * int(b["mask"].sum())
                    vn += int(b["mask"].sum())
            row["val_recon"] = vr / max(vn, 1); row["val_energy"] = vt / max(vn, 1)
        history.append(row)
        print("[crt] epoch {}: ".format(ep) +
              "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"), flush=True)

        ck = {"model": model.state_dict(), "ema": ema.state_dict(),
              "cfg": cfg.__dict__, "history": history, "epoch": ep}
        torch.save(ck, os.path.join(cfg.out, "crt.pt"))
        score = row.get("val_recon", row["recon"]) + row.get("val_energy", row["term"])
        if score < best:
            best = score
            torch.save(ck, os.path.join(cfg.out, "crt_best.pt"))
            print(f"[crt] epoch {ep}: new best {best:.4f}, saved crt_best.pt", flush=True)
    return model
