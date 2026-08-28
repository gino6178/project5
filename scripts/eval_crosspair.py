#!/usr/bin/env python
"""Real cross-pairing OOD: retarget a reference design into a DIFFERENT real
scene's room boundary (not a synthetic deformation of its own room), addressing
the circularity concern. Pairs are matched by category Jaccard>0.6 + anchor-yaw
filter + spatial Hungarian. Score S_rel/collision/OOB against the reference graph."""
import os, sys, numpy as np
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.eval.metrics import evaluate
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.generative.xscene import build_pair_index_filtered, anchor_orientation_ok
from reroom.retarget.baselines import run_baseline
from reroom.data.asset_bank import AssetBank
from reroom.geom.polygon import as_polygon
from shapely.geometry import Point
from scipy import stats

def oob(scene):
    poly=as_polygon(scene.room); ks=[o for o in scene.objects if o.keep]
    return 100.0*sum(0 if poly.contains(Point(*o.xy)) else 1 for o in ks)/max(len(ks),1)

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0",relational_select=True)
# real cross-pairs among the held-out TEST scenes (OOD: boundary from a different scene)
idx=build_pair_index_filtered(test, thresh=0.6, max_deg=30.0, max_partners=8, seed=0)
res={'ours':{'S_rel':[],'R_col':[],'OOB':[]},'affine':{'S_rel':[],'R_col':[],'OOB':[]}}
rng=np.random.default_rng(0); n_pairs=0
for ri,partners in idx.items():
  if n_pairs>=40: break
  try:
    ref=test[ri]; g=build_motifs(build_scene_graph(ref))
    tj=partners[int(rng.integers(0,len(partners)))]
    tgt=test[tj]
    if not anchor_orientation_ok(ref,tgt,30.0): continue
    room=tgt.room.copy()   # real OOD boundary from a different scene
    o1=generative_retarget(fb,g,room,elasticity=el,bank=bank,cfg=cfg,k=16,polish=False).scene
    m1=evaluate(g,o1); res['ours']['S_rel'].append(m1['S_rel']); res['ours']['R_col'].append(100*m1['R_col']); res['ours']['OOB'].append(oob(o1))
    o2=run_baseline("affine_fit",g,room,cfg=cfg)
    m2=evaluate(g,o2); res['affine']['S_rel'].append(m2['S_rel']); res['affine']['R_col'].append(100*m2['R_col']); res['affine']['OOB'].append(oob(o2))
    n_pairs+=1
  except Exception as e: print('skip',ri,repr(e)[:50],flush=True)
def mn(x): x=np.array(x,float); x=x[~np.isnan(x)]; return (x.mean(),x.std())
print(f"\nReal cross-pairing OOD (retarget into a DIFFERENT real scene's boundary), n={n_pairs} pairs")
print(f"{'method':<22}{'S_rel':>12}{'R_col%':>10}{'OOB%':>8}")
for k,l in (('affine','Affine warp'),('ours','ReRoom (ours)')):
    r=res[k]; print(f"{l:<22}{mn(r['S_rel'])[0]:6.3f}±{mn(r['S_rel'])[1]:<4.3f}{mn(r['R_col'])[0]:8.2f}{mn(r['OOB'])[0]:8.1f}")
if len(res['ours']['S_rel'])>3:
    d=np.array(res['ours']['R_col'])-np.array(res['affine']['R_col'])
    try:_,p=stats.wilcoxon(d)
    except:p=float('nan')
    print(f"  collision ours−affine: Δ={d.mean():+.2f}% p={p:.3g}")
print("DONE_CROSSPAIR")
