#!/usr/bin/env python
"""Challenging-floorplan stress test: per-type quantitative breakdown on named
non-trivial boundaries (L-shape, T-shape, trapezoid, narrow 1:4 corridor),
comparing the base flow's raw output vs + Πθ. Reports S_rel, collision, OOB."""
import os, sys, numpy as np
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import aspect_deform, corner_cut, slant_wall, _anchor_openings, _replace_openings
from reroom.core.scene import Room
from reroom.eval.metrics import evaluate
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.retarget.diffproj import project_scene
from reroom.data.asset_bank import AssetBank
from reroom.geom.polygon import as_polygon
from shapely.geometry import Point

def mkroom(r,poly):
    a=_anchor_openings(r); return Room(polygon=poly,height=r.height,openings=_replace_openings(poly,a,len(r.polygon)),room_type=r.room_type)
def oob(scene):
    poly=as_polygon(scene.room); ks=[o for o in scene.objects if o.keep]
    return 100.0*sum(0 if poly.contains(Point(*o.xy)) else 1 for o in ks)/max(len(ks),1)

def floorplans(poly):
    """named challenging boundaries from the reference room polygon."""
    F={}
    F["L-shape"]=corner_cut(poly,0,0.5,0.5,0.0)
    F["T-shape"]=corner_cut(corner_cut(poly,0,0.33,0.45,0.0),2,0.33,0.45,0.0)  # two adjacent cuts
    F["trapezoid"]=slant_wall(poly,1,0.55,"normal")
    F["corridor 1:4"]=aspect_deform(poly,2.0,0.5)
    return F

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[6,8,25,2,10,14,1,3,5,7,9,11]
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0",relational_select=True)
agg={}
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for name,poly in floorplans(src.room.polygon).items():
      try:
        room=mkroom(src.room,poly)
        raw=generative_retarget(fb,g,room,elasticity=el,bank=bank,cfg=cfg,k=16,polish=False).scene
        pi=project_scene(raw,room,iters=40,lr=0.03,device="cuda:0")
        mr=evaluate(g,raw); mp=evaluate(g,pi)
        a=agg.setdefault(name,{'S_rel':[],'Rcol_raw':[],'Rcol_pi':[],'OOB':[]})
        a['S_rel'].append(mp['S_rel']); a['Rcol_raw'].append(100*mr['R_col'])
        a['Rcol_pi'].append(100*mp['R_col']); a['OOB'].append(oob(pi))
      except Exception as e: print('  skip',name,repr(e)[:40],flush=True)
  except Exception as e: print('skip',sd,repr(e)[:40],flush=True)
def mn(x): x=np.array(x,float); x=x[~np.isnan(x)]; return x.mean() if len(x) else float('nan')
print(f"\nChallenging floorplans (n={len(seeds)} refs), flow + Πθ")
print(f"{'floorplan':<16}{'S_rel':>9}{'Rcol raw%':>11}{'Rcol +Πθ%':>11}{'OOB%':>8}")
for name in ["L-shape","T-shape","trapezoid","corridor 1:4"]:
    if name in agg:
        a=agg[name]; print(f"{name:<16}{mn(a['S_rel']):>9.3f}{mn(a['Rcol_raw']):>11.2f}{mn(a['Rcol_pi']):>11.2f}{mn(a['OOB']):>8.1f}")
print("DONE_FLOORPLANS")
