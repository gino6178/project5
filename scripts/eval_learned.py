#!/usr/bin/env python
"""Learned baseline + paired statistics (reviewer asks #1 and #2).

Baseline = a REAL generative flow ablated to global coordinates: flow_r2m2full
(same DiT + geo-bias + wall-tokens + informative prior, but WITHOUT the
parent-relative hierarchical reparameterization). Comparing it to flow_bfresh
directly tests Prop 1 (global-coordinate flow disperses motifs) with a learned
model on the same S_rel/R_col metrics.

Also reports paired per-cell statistics (Wilcoxon signed-rank) for:
  * S_rel(hierarchical) − S_rel(global)      -> does the reparam. help?
  * S_rel(+Pi_theta) − S_rel(+hard snap)     -> does Pi_theta preserve topology?
"""
import os, sys, math, argparse
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
import numpy as np
from scipy import stats
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import uniform_scale, _anchor_openings, _replace_openings
from reroom.core.scene import Room
from reroom.eval.metrics import evaluate
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.retarget.regularity import regularity_snap, WALL_CATS
from reroom.retarget.diffproj import project_scene
from reroom.geom.polygon import as_polygon
from reroom.data.asset_bank import AssetBank
import copy

from reroom.geom.deform import aspect_deform
def scaled_room(r, s):
    # s is a float (uniform) or an (sx, sy) tuple (anisotropic aspect stretch --
    # the regime where the hierarchical parent-relative reparameterization / Prop 1
    # should matter, since a global-coord flow tears motifs apart under
    # non-uniform scaling while parent-relative topology is invariant).
    if isinstance(s, (tuple, list)):
        p = aspect_deform(r.polygon, float(s[0]), float(s[1]))
    else:
        p = uniform_scale(r.polygon, s)
    a=_anchor_openings(r)
    return Room(polygon=p,height=r.height,openings=_replace_openings(p,a,len(r.polygon)),room_type=r.room_type)

def snap_pct(scene):
    poly=as_polygon(scene.room); ring=np.asarray(poly.exterior.coords)[:-1]
    edges=[(ring[i],ring[(i+1)%len(ring)]) for i in range(len(ring))]
    ok=tot=0
    for o in scene.objects:
        if not o.keep or o.category not in WALL_CATS: continue
        tot+=1; corners=o.corners(); best=None
        for a,b in edges:
            d=b-a; L=np.linalg.norm(d)+1e-9; t=d/L; nrm=np.array([-t[1],t[0]])
            rel=corners-a; perp=np.abs(rel@nrm); pj=rel@t; ins=(pj>-0.05)&(pj<L+0.05)
            if not ins.any(): continue
            gap=perp[ins].min()
            if best is None or gap<best[0]:
                ang=math.atan2(t[1],t[0]); ax0=np.array([math.cos(o.yaw),math.sin(o.yaw)]); ax1=np.array([-math.sin(o.yaw),math.cos(o.yaw)])
                sk=min(abs(((math.atan2(ax0[1],ax0[0])-ang+math.pi/2)%math.pi)-math.pi/2),
                       abs(((math.atan2(ax1[1],ax1[0])-ang+math.pi/2)%math.pi)-math.pi/2))
                best=(gap,math.degrees(sk))
        if best is not None and best[0]<=0.12 and best[1]<=4.0: ok+=1
    return ok/max(tot,1)

def meas(g,sc):
    m=evaluate(g,sc); return (100*m["R_col"],100*snap_pct(sc),m["S_rel"],m.get("S_rel_kept",m["S_rel"]),m.get("S_motif",float('nan')))

ap=argparse.ArgumentParser()
ap.add_argument("--hier", default="outputs/flow_bfresh/flow_best.pt")
ap.add_argument("--global_", dest="glob", default="outputs/flow_r2m2full/flow_best.pt")
ap.add_argument("--seeds", default="6,8,25,2,10,14,1,3,5,7,9,11")
ap.add_argument("--sizes", default="0.75,1.0,1.35")
ap.add_argument("--aniso", action="store_true",
                help="anisotropic aspect-stretch deformations (tests Prop 1's "
                     "invariance regime) instead of uniform scaling")
a=ap.parse_args()
el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[int(x) for x in a.seeds.split(",")]
if a.aniso:
    # aspect-ratio stretches: elongate along x, then along y (area ~ preserved),
    # at two strengths.  This is where a global-coord flow should tear motifs.
    sizes=[(1.5,0.75),(0.75,1.5),(1.7,0.85),(0.85,1.7)]
    print("ANISO mode: aspect stretches",sizes,flush=True)
else:
    sizes=[float(x) for x in a.sizes.split(",")]
fh=load_flow(a.hier,device="cuda:0"); fg=load_flow(a.glob,device="cuda:0")
print("hier parent_relative:",getattr(fh,"parent_relative",None),"| global parent_relative:",getattr(fg,"parent_relative",None),flush=True)
cfg=lambda snap: RetargetConfig(restarts=16,regularity_snap=snap,device="cuda:0")
KEYS=("R_col","snap","S_rel","S_rel_kept","S_motif")
rows={n:{k:[] for k in KEYS} for n in ("global","hier","hier_snap","hier_diff")}
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for s in sizes:
        room=scaled_room(src.room,s)
        gl=generative_retarget(fg,g,room,elasticity=el,bank=bank,cfg=cfg(False),k=16,polish=False).scene
        hi=generative_retarget(fh,g,room,elasticity=el,bank=bank,cfg=cfg(False),k=16,polish=False).scene
        hs=copy.deepcopy(hi); regularity_snap(hs)
        hd=project_scene(hi,room,iters=40,lr=0.03,device="cuda:0")
        for name,sc in [("global",gl),("hier",hi),("hier_snap",hs),("hier_diff",hd)]:
            for k,v in zip(KEYS,meas(g,sc)): rows[name][k].append(v)
  except Exception as e: print("skip",sd,e,flush=True)

def ms(x): x=np.array(x); x=x[~np.isnan(x)]; return x.mean(),x.std()
n=len(rows["hier"]["S_rel"]); print(f"\nN cells = {n}")
print(f"\n{'model':<28}{'S_rel':>13}{'S_rel_kept':>13}{'S_motif':>12}{'R_col%':>12}{'snap%':>10}")
for name,label in [("global","Global-coord flow (raw)"),("hier","Hierarchical flow (raw, ours)"),
                   ("hier_snap","  + hard snap"),("hier_diff","  + diff Pi_theta")]:
    r=rows[name]; (sm,ss)=ms(r["S_rel"]); (km,ks)=ms(r["S_rel_kept"]); (mm,mms)=ms(r["S_motif"]); (cm,cs)=ms(r["R_col"]); (pm,ps)=ms(r["snap"])
    print(f"{label:<28}{sm:6.3f}±{ss:<5.3f}{km:6.3f}±{ks:<5.3f}{mm:5.3f}±{mms:<4.3f}{cm:5.2f}±{cs:<4.2f}{pm:5.0f}±{ps:<3.0f}")

def paired(a,b,lbl):
    a=np.array(a);b=np.array(b);d=a-b
    try: w,p=stats.wilcoxon(d)
    except Exception: p=float('nan')
    pos=(d>0).mean()
    print(f"  {lbl}: mean Δ={d.mean():+.3f} (±{d.std():.3f}), win-rate={pos:.0%}, Wilcoxon p={p:.4g}")
print("\nPaired tests (per-cell, n={}):".format(n))
paired(rows["hier"]["S_rel"],rows["global"]["S_rel"],"S_rel: hierarchical − global-coord")
paired(rows["hier"]["S_motif"],rows["global"]["S_motif"],"S_motif: hierarchical − global-coord")
paired(rows["hier_diff"]["S_rel"],rows["hier_snap"]["S_rel"],"S_rel: +Pi_theta − +hard snap")
paired(rows["hier_diff"]["R_col"],rows["hier_snap"]["R_col"],"R_col: +Pi_theta − +hard snap")

# REVIEWER FIX: reference-clustered (cluster-robust) test.  The 36 cells are
# 12 references × 3 correlated sizes; the per-cell Wilcoxon above treats
# pseudo-replicates as independent.  Aggregate the sizes per reference first
# (mean Δ over the 3 sizes → one value per reference), then test on the
# independent references (effective n = #references).  This is the correct
# statistic and is what we report as the headline.
nsz=len(sizes)
def clustered(a,b,lbl):
    a=np.array(a); b=np.array(b); d=a-b
    if len(d)%nsz!=0:
        print(f"  {lbl}: (cannot cluster, {len(d)} not divisible by {nsz})"); return
    dref=d.reshape(-1,nsz).mean(1)                 # per-reference mean Δ
    dref=dref[~np.isnan(dref)]
    try: w,p=stats.wilcoxon(dref)
    except Exception: p=float('nan')
    # exact sign test as a second, assumption-light check
    from scipy.stats import binomtest
    pos=int((dref>0).sum()); tot=int((dref!=0).sum())
    sp=binomtest(pos,tot,0.5).pvalue if tot>0 else float('nan')
    print(f"  {lbl}: per-ref mean Δ={dref.mean():+.4f} (±{dref.std():.4f}), "
          f"n_ref={len(dref)}, win {pos}/{tot}, Wilcoxon p={p:.4g}, sign-test p={sp:.4g}")
print(f"\nReference-clustered tests (effective n={n//nsz} references):")
clustered(rows["hier"]["S_rel"],rows["global"]["S_rel"],"S_rel: hierarchical − global-coord")
clustered(rows["hier"]["S_motif"],rows["global"]["S_motif"],"S_motif: hierarchical − global-coord")
clustered(rows["hier_diff"]["S_rel"],rows["hier_snap"]["S_rel"],"S_rel: +Pi_theta − +hard snap")

# dump raw per-cell arrays for provenance / re-analysis
import json as _json
_out={n_:{k:[float(x) for x in v] for k,v in d_.items()} for n_,d_ in rows.items()}
_out["_meta"]={"seeds":list(seeds),"sizes":list(sizes),
               "hier_ckpt":a.hier,"global_ckpt":a.glob}
with open(("outputs/eval_learned_twin_aniso.json" if a.aniso else "outputs/eval_learned_twin.json"),"w") as _f: _json.dump(_out,_f)
print("wrote", ("outputs/eval_learned_twin_aniso.json" if a.aniso else "outputs/eval_learned_twin.json"))
print("\nDONE_LEARNED")
