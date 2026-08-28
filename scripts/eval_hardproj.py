#!/usr/bin/env python
"""Projection ablation (none / hard-snap / Pi_theta) on the HARD non-convex
regimes where the base flow breaks (corner-cut / L-shape / extreme aspect,
collision ~5%).  Does Pi_theta recover feasibility WITHOUT hurting S_rel there?
Decides: existing mechanism suffices (scope Pi_theta to the failure regime) vs a
new concave-aware mechanism is needed."""
import os, sys, copy, numpy as np
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import aspect_deform, corner_cut, sample_deform, _anchor_openings, _replace_openings
from reroom.core.scene import Room
from reroom.eval.metrics import evaluate
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.retarget.regularity import regularity_snap
from reroom.retarget.diffproj import project_scene
from reroom.data.asset_bank import AssetBank
from scipy import stats

def mkroom(r, poly):
    a=_anchor_openings(r); return Room(polygon=poly,height=r.height,openings=_replace_openings(poly,a,len(r.polygon)),room_type=r.room_type)
def hardrooms(poly, sd):
    return {"corner DEEP": corner_cut(poly,0,0.6,0.6,0.0),
            "L4 corner x2": sample_deform(poly,4,np.random.default_rng(sd*7+1),strength=2.0)[0],
            "aspect EXTREME": aspect_deform(poly,2.2,0.45)}

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[6,8,25,2,10,14,1,3,5,7,9,11]
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0")
rows={n:{"R_col":[],"S_rel":[]} for n in ("raw","snap","pi")}
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for name,poly in hardrooms(src.room.polygon, sd).items():
      try:
        room=mkroom(src.room,poly)
        raw=generative_retarget(fb,g,room,elasticity=el,bank=bank,cfg=cfg,k=16,polish=False).scene
        snap=copy.deepcopy(raw); regularity_snap(snap)
        pi=project_scene(raw,room,iters=40,lr=0.03,device="cuda:0")
        for nm,sc in (("raw",raw),("snap",snap),("pi",pi)):
            m=evaluate(g,sc); rows[nm]["R_col"].append(100*m["R_col"]); rows[nm]["S_rel"].append(m["S_rel"])
      except Exception as e: print("  skip",name,repr(e)[:50],flush=True)
  except Exception as e: print("skip",sd,repr(e)[:50],flush=True)
def ms(x): x=np.array(x,float); x=x[~np.isnan(x)]; return x.mean(),x.std()
n=len(rows["raw"]["R_col"]); print(f"\nHARD non-convex regimes, N={n} cells (12 refs x 3 regimes)")
print(f"{'finish':<26}{'R_col%':>12}{'S_rel':>12}")
for nm,l in (("raw","none (base flow raw)"),("snap","+ hard snap"),("pi","+ diff Pi_theta")):
    (cm,cs)=ms(rows[nm]["R_col"]); (sm,ss)=ms(rows[nm]["S_rel"])
    print(f"{l:<26}{cm:6.2f}±{cs:<5.2f}{sm:6.3f}±{ss:<5.3f}")
def paired(a,b,k,lbl):
    d=np.array(rows[a][k])-np.array(rows[b][k])
    try:_,p=stats.wilcoxon(d)
    except:p=float('nan')
    print(f"  {lbl}: Δ={d.mean():+.3f} p={p:.3g}")
print("\nPaired:")
paired("pi","raw","R_col","Pi_theta − raw collision (want <<0)")
paired("pi","raw","S_rel","Pi_theta − raw S_rel (want >=0)")
paired("pi","snap","R_col","Pi_theta − snap collision")
paired("pi","snap","S_rel","Pi_theta − snap S_rel (want >0: keeps relations vs snap)")

def clustered(a,b,k,lbl,nsz=3):
    d=np.array(rows[a][k])-np.array(rows[b][k])
    if len(d)%nsz: print(f"  {lbl}: (n {len(d)} not /{nsz})"); return
    dr=d.reshape(-1,nsz).mean(1)
    try:_,p=stats.wilcoxon(dr)
    except:p=float('nan')
    print(f"  {lbl}: clustered n={len(dr)} Δ={dr.mean():+.3f} p={p:.3g}")
print("\nReference-clustered (n=12):")
clustered("pi","raw","R_col","Πθ − raw collision")
clustered("pi","raw","S_rel","Πθ − raw S_rel")
clustered("pi","snap","S_rel","Πθ − snap S_rel")

print("DONE_HARDPROJ")
