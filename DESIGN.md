# ReRoom-E2E — design

project4 ships a **staged** system: a hand-designed discrete prune (Stage 1), a
flow trained on raw endpoints (Stage 2), and a projection \(\Pi_\theta\) bolted on
at test time. Reviewers hit the same three seams every round: the pruning is a
heuristic, \(\Pi_\theta\) is test-time-only, and the pieces are optimised for
different objectives than the one we actually deploy.

project5 collapses the pipeline into **one differentiable function trained with
one objective**.

---

## 1. What actually changes

The decisive change is *where the supervision is applied*.

```
project4 (staged)
  x0 ──flow──► x̂₁ ──────────────────────────► loss = FM(x̂₁, x₁)
                └─(test time only)─► Πθ ─► x*        Πθ never sees a gradient
                                                     that matters

project5 (end-to-end)
  x0 ──flow──► x̂₁ ──Πθ(K steps, differentiable)──► x*
       │                                             loss = FM(x*, x₁) + …
       └─existence head ℓ ──(straight-through)──► keep mask, gates Πθ's
                                                   collision/containment terms
```

The model is trained so that the layout is correct **after** projection, and so
that the objects it decides to keep are the ones that survive projection well.
Selection, placement, and feasibility stop being three objectives and become one.

**Why this is not the experiment that already failed.** project4 tried two
bolt-ons — `proj_loss` (train-through \(\Pi_\theta\)) and a Gromov–Wasserstein
relational loss — and both measured a clean null (project4 §7). Both *fine-tuned
an already-converged model* while still supervising the raw endpoint; the
projection was an extra penalty, not part of the model. Here the projection is
inside the forward pass from step 0 of training, and the reconstruction loss is
computed through it. A converged staged model has no reason to move; a model
trained this way from scratch has no other option.

## 2. Components (all already in the library, rewired)

| Piece | Source | Role in the end-to-end model |
|---|---|---|
| Hierarchical metric code | `reroom/generative/model.py` | unchanged — scale-invariant parameterization |
| Rectified flow + informative prior | `reroom/generative/train.py` | unchanged — transport from the affine transplant |
| Existence head (`mask_flow`) | `train.py` (D1) | promoted from auxiliary to the **selection mechanism**; straight-through so it gates geometry |
| \(\Pi_\theta\) unrolled projection | `retarget/diffproj.py`, `_proj_through_energy` | moved **into** the forward pass; supervision flows through it |
| Relational mass | `retarget/gwselect.py` | kept as the *inference-time* fallback and as the ablation baseline for the learned selection |

## 3. Training objective

With \(x^* = \Pi_\theta(\hat x_1, m)\) where \(m\) is the straight-through keep mask:

* **reconstruction through the projection** — flow-matching against the real
  layout, evaluated on \(x^*\); gated to \(\tau > \tau_\text{proj}\) where the
  endpoint estimate is meaningful, falling back to raw-endpoint FM below it (the
  model still needs a usable velocity field at low \(\tau\)).
* **existence** — the D1 velocity loss on the keep logit, plus a capacity term so
  the kept count matches what the target room can hold.
* **feasibility residual** — \(E_\text{geo}(x^*)\): whatever the projection could
  not fix is charged to the flow, which is the pressure that makes the output
  feasible-by-construction rather than projection-dependent.
* **relational** — the existing motif-rigidity term, unchanged, so identity
  preservation is not traded away silently.

## 4. What must be measured (and what would falsify this)

Comparisons run against project4's shipped checkpoint on the **same protocol**,
so the tables stay comparable (`REPRODUCE.md` in project4 maps table → command).

The end-to-end model earns its place only if it shows, with reference-clustered
statistics:

1. **learned selection ≥ relational-mass heuristic** on \(S_\text{rel}\) under
   capacity pressure (project4 Table 7 is the bar), and
2. **feasibility without the deployed projection** — i.e. raw output collision
   approaching the staged model's *post*-projection collision, especially on the
   non-convex boundaries where the staged model degrades (project4 Table 2/10),
   and
3. **no relational regression** — \(S_\text{rel}^\text{kept}\) within noise of the
   staged model.

If the end-to-end model only matches the staged pipeline, that is a **negative
result and gets reported as one**: it would say the staged decomposition is not a
compromise but the right factorization — which is exactly what project4 §7
argues. Either outcome is publishable; neither is worth overselling.

## 5. Status

Scaffolded from project4 at commit `40d25ba` (library + the entry points that
produce the paper tables). Datasets and checkpoints live on the GPU box; nothing
large is committed.
