#!/usr/bin/env python
"""Consolidated ablation matrix: isolate each component's marginal contribution
under a single protocol (uniform 0.75/1.0/1.35). Toggles prior, guidance, and
projection on the shipped hierarchical flow; the flat-vs-hier axis is the matched
twin (Table 4). Reports S_rel and collision."""
import os, sys, copy, numpy as np
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
from reroom.retarget.regularity import regularity_snap
from reroom.retarget.diffproj import project_scene
from reroom.data.asset_bank import AssetBank
from scipy import stats

def mkroom(r,s):
    p=uniform_scale(r.polygon,s); a=_anchor_openings(r)
    return Room(polygon=p,height=r.height,openings=_replace_openings(p,a,len(r.polygon)),room_type=r.room_type)

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[6,8,25,2,10,14,1,3,5,7,9,11]; sizes=[0.75,1.0,1.35]
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
def cfg(): return RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0")

# ablation configs: (label, prior_x0, guidance, projection)
CONF=[("full (informative+pullback+Πθ)",True,"default","pi"),
      ("− Πθ (raw flow)",              True,"default","none"),
      ("Πθ→hard snap",                 True,"default","snap"),
      ("− manifold-pullback guidance", True,None,     "pi"),
      ("− informative prior (gaussian)",False,"default","pi")]
rows={c[0]:{'S_rel':[],'R_col':[]} for c in CONF}
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for s in sizes:
      room=mkroom(src.room,s)
      for lab,prior,guid,proj in CONF:
        fb._prior_x0=bool(prior)      # toggle informative prior at inference
        try:
          sc=generative_retarget(fb,g,room,elasticity=el,bank=bank,cfg=cfg(),k=16,polish=False,guidance=guid).scene
          if proj=="snap": sc=copy.deepcopy(sc); regularity_snap(sc)
          elif proj=="pi": sc=project_scene(sc,room,iters=40,lr=0.03,device="cuda:0")
          m=evaluate(g,sc); rows[lab]['S_rel'].append(m['S_rel']); rows[lab]['R_col'].append(100*m['R_col'])
        except Exception as e: print('  skip',lab,repr(e)[:40],flush=True)
    fb._prior_x0=True
  except Exception as e: print('skip',sd,repr(e)[:40],flush=True)
def mn(x): x=np.array(x,float); x=x[~np.isnan(x)]; return x.mean() if len(x) else float('nan')
full=rows[CONF[0][0]]
print(f"\nAblation matrix (36 layouts; toggles on the shipped hierarchical flow)")
print(f"{'configuration':<34}{'S_rel':>9}{'R_col%':>9}{'ΔS_rel vs full':>16}")
for lab,_,_,_ in CONF:
    r=rows[lab]; d=mn(r['S_rel'])-mn(full['S_rel'])
    print(f"{lab:<34}{mn(r['S_rel']):>9.3f}{mn(r['R_col']):>9.2f}{('' if lab==CONF[0][0] else f'{d:+.3f}'):>16}")
print("DONE_ABLATION")
