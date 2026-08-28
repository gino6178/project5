#!/usr/bin/env python
"""Failure-mode diagnostic: where does the BASE flow (bfresh) actually break?
Sweep the deformation family from mild to extreme (uniform / aspect / slant /
corner-cut / concave, at strength 1x and 2x, plus hand-crafted extremes) and
measure S_rel, S_motif, collision, and OUT-OF-BOUNDS.  A regime where fidelity
collapses or OOB/collision spikes is an honest target for a new mechanism."""
import os, sys, numpy as np
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import (uniform_scale, aspect_deform, slant_wall, corner_cut,
                                 sample_deform, _anchor_openings, _replace_openings)
from reroom.core.scene import Room
from reroom.eval.metrics import evaluate
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.data.asset_bank import AssetBank
from reroom.geom.polygon import as_polygon
from shapely.geometry import Point

def mkroom(r, poly):
    a=_anchor_openings(r); return Room(polygon=poly,height=r.height,openings=_replace_openings(poly,a,len(r.polygon)),room_type=r.room_type)

def oob_frac(scene):
    poly=as_polygon(scene.room); ks=[o for o in scene.objects if o.keep]
    if not ks: return 0.0
    return 100.0*sum(0 if poly.contains(Point(*o.xy)) else 1 for o in ks)/len(ks)

# named regimes: room -> deformed poly
def regimes(poly, rng):
    R={}
    R["L1 uniform .8"]=uniform_scale(poly,0.8)
    for lv in (2,3,4,5):
        R[f"L{lv} strength1"]=sample_deform(poly,lv,np.random.default_rng(rng),strength=1.0)[0]
        R[f"L{lv} strength2"]=sample_deform(poly,lv,np.random.default_rng(rng+1),strength=2.0)[0]
    R["aspect EXTREME 2.2x.45"]=aspect_deform(poly,2.2,0.45)
    R["corner DEEP .6x.6"]=corner_cut(poly,0,0.6,0.6,0.0)
    return R

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[6,8,25,2,10,14,1,3]   # 8 refs for speed
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0")
agg={}
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for name,poly in regimes(src.room.polygon, sd*7).items():
      try:
        room=mkroom(src.room, poly)
        sc=generative_retarget(fb,g,room,elasticity=el,bank=bank,cfg=cfg,k=16,polish=False).scene
        m=evaluate(g,sc)
        agg.setdefault(name,{"S_rel":[],"S_motif":[],"R_col":[],"OOB":[]})
        agg[name]["S_rel"].append(m["S_rel"]); agg[name]["S_motif"].append(m.get("S_motif",np.nan))
        agg[name]["R_col"].append(100*m["R_col"]); agg[name]["OOB"].append(oob_frac(sc))
      except Exception as e: print("  regime skip",name,repr(e)[:60],flush=True)
  except Exception as e: print("skip",sd,repr(e)[:60],flush=True)
def mn(x): x=np.array(x,float); x=x[~np.isnan(x)]; return x.mean() if len(x) else float('nan')
order=["L1 uniform .8","L2 strength1","L2 strength2","L3 strength1","L3 strength2",
       "L4 strength1","L4 strength2","L5 strength1","L5 strength2","aspect EXTREME 2.2x.45","corner DEEP .6x.6"]
print(f"\n{'regime':<24}{'S_rel':>8}{'S_motif':>9}{'R_col%':>8}{'OOB%':>7}{'n':>4}")
for name in order:
    if name in agg:
        a=agg[name]; print(f"{name:<24}{mn(a['S_rel']):8.3f}{mn(a['S_motif']):9.3f}{mn(a['R_col']):8.2f}{mn(a['OOB']):7.1f}{len(a['S_rel']):4d}")
print("DONE_FAILURE")
