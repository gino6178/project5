#!/usr/bin/env python
"""Full-pipeline eval of the partial-relational-transport SELECTION module:
flow WITH vs WITHOUT cfg.relational_select, across a shrink sweep (where pruning
bites).  Reports S_rel / S_motif / R_col and paired + reference-clustered tests."""
import os, sys, numpy as np
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import uniform_scale, _anchor_openings, _replace_openings
from reroom.core.scene import Room
from reroom.eval.metrics import evaluate
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.data.asset_bank import AssetBank
from scipy import stats

def rm(r,s):
    p=uniform_scale(r.polygon,s); a=_anchor_openings(r)
    return Room(polygon=p,height=r.height,openings=_replace_openings(p,a,len(r.polygon)),room_type=r.room_type)

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[6,8,25,2,10,14,1,3,5,7,9,11]
sizes=[0.6,0.7,0.8]     # shrink regime where pruning happens
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
def cfg(rel): return RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0",relational_select=rel)
rows={n:{k:[] for k in ("S_rel","S_motif","R_col","nkeep")} for n in ("off","on")}
nsz=len(sizes)
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for s in sizes:
      room=rm(src.room,s)
      for nm,rel in (("off",False),("on",True)):
        sc=generative_retarget(fb,g,room,elasticity=el,bank=bank,cfg=cfg(rel),k=16,polish=False).scene
        m=evaluate(g,sc)
        rows[nm]["S_rel"].append(m["S_rel"]); rows[nm]["S_motif"].append(m.get("S_motif",np.nan))
        rows[nm]["R_col"].append(100*m["R_col"]); rows[nm]["nkeep"].append(sum(1 for o in sc.objects if o.keep))
  except Exception as e: print("skip",sd,repr(e),flush=True)
def ms(x): x=np.array(x,float); x=x[~np.isnan(x)]; return x.mean(),x.std()
n=len(rows["off"]["S_rel"]); print(f"\nN={n} cells ({n//nsz} refs x {nsz} shrink sizes)")
print(f"{'variant':<28}{'S_rel':>12}{'S_motif':>11}{'R_col%':>10}{'nkeep':>8}")
for nm,l in (("off","flow (Summarise mask)"),("on","flow + relational-select")):
    r=rows[nm]; print(f"{l:<28}{ms(r['S_rel'])[0]:6.3f}±{ms(r['S_rel'])[1]:<4.3f}{ms(r['S_motif'])[0]:7.3f}{ms(r['R_col'])[0]:9.2f}{ms(r['nkeep'])[0]:8.1f}")
def paired(k,lbl):
    a=np.array(rows["on"][k]); b=np.array(rows["off"][k]); d=a-b
    dr=d.reshape(-1,nsz).mean(1)
    try:_,p=stats.wilcoxon(d)
    except:p=float('nan')
    try:_,pc=stats.wilcoxon(dr)
    except:pc=float('nan')
    print(f"  {lbl}: Δ={d.mean():+.3f} percell p={p:.3g} | clustered(n={len(dr)}) Δ={dr.mean():+.3f} p={pc:.3g}")
print("\nPaired (relational-select ON − OFF):")
paired("S_rel","S_rel"); paired("S_motif","S_motif"); paired("R_col","R_col%")
print("DONE_RELSEL")
