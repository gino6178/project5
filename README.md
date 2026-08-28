# ReRoom-E2E

End-to-end joint training for **reference-guided 3D interior scene retargeting**:
transplant a specific reference layout into a new room boundary, preserving its
relational structure while satisfying hard geometry (wall-flush, in-bounds,
collision-free).

This is an independent line from [project4](https://github.com/gino6178/project4),
which ships the **staged** system — a hand-designed discrete prune, a flow trained
on raw endpoints, and a projection \(\Pi_\theta\) applied at test time. Here the
pipeline collapses into **one differentiable function trained with one objective**:

```
x0 ──flow──► x̂₁ ──Πθ (K unrolled steps, differentiable)──► x*
     │                                        loss = FM(x*, x₁) + residual feasibility
     └─existence head ──(straight-through)──► keep mask, gates Πθ's collision terms
```

The reconstruction loss is applied to the **projected** layout, so selection,
placement, and feasibility are optimised against the objective actually deployed —
rather than three objectives that only meet at inference time.

See **[DESIGN.md](DESIGN.md)** for the mechanism, why this is not the bolt-on
experiment that already returned a null in project4, and the criteria that would
falsify it.

## Quick start

```bash
conda create -n reroom python=3.11 -y && conda activate reroom
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements.txt
```

Build the corpus and priors, then train (scripts resolve the repo root from their
own location; override with `REROOM_ROOT`):

```bash
python scripts/build_3dfront.py        # 3D-FRONT scene JSONs -> ReRoom scenes
python scripts/build_future_bboxes.py  # asset sizes from 3D-FUTURE meshes
python scripts/build_priors.py         # corpus statistics, priors
python scripts/fit_elasticity.py       # relation elasticity alpha

python scripts/train_e2e.py            # end-to-end joint training (from scratch)
```

**First run:** check the epoch-0 row — `e2e_recon` should be the same order as
`train_loss`. If it dominates, lower `e2e_recon` in `scripts/train_e2e.py`; an
over-weighted geometric term degrades the flow-matching objective.

## Evaluation

The training protocol is otherwise identical to project4's shipped model, so
project4's evaluation scripts apply unchanged and the numbers stay comparable —
see `REPRODUCE.md` there for the paper-table → command map. The end-to-end model
has to show, with reference-clustered statistics:

1. learned selection ≥ the relational-mass heuristic under capacity pressure,
2. feasibility **without** the deployed projection, especially on non-convex
   boundaries, and
3. no relational regression.

If it merely matches the staged pipeline, that is reported as a negative result:
it would mean the staged decomposition is the right factorization, not a
compromise.

## Layout

```
reroom/            library — geometry, intent graph, retargeting, generative flow
  intent/          relations, motifs, elasticity
  retarget/        optimizer, regularity, diffproj (Pi_theta), gwselect
  generative/      tokens, model (DiT), train (+ e2e), sample, guidance, xscene
scripts/           training + evaluation entry points
baselines_legonet/ LEGO-Net cross-lineage baseline (separate py3.7 env)
tests/             unit tests
```

Datasets and checkpoints are not committed; they are built and trained on the GPU
box (remote tree `/opt/NeMo/reroom/e2e`, kept separate from project4's).
