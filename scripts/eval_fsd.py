#!/usr/bin/env python
"""Fréchet Scene Distance (FSD): distributional realism of retargeted layouts.
Extract a compact layout feature per scene (category histogram + spatial &
orientation statistics), fit a Gaussian to each method's generated set and to the
real 3D-FRONT layouts, and report the Fréchet distance to the real distribution.
Lower = statistically closer to real human designs."""
import os, sys, numpy as np
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
from scipy import linalg
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import uniform_scale, aspect_deform, _anchor_openings, _replace_openings
from reroom.core.scene import Room
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.retarget.baselines import run_baseline
from reroom.generative.tokens import CATS
from reroom.geom.polygon import as_polygon
from reroom.data.asset_bank import AssetBank

CIX={c:i for i,c in enumerate(CATS)}
def feat(scene):
    objs=[o for o in scene.objects if o.keep]
    if len(objs)<2: return None
    p=np.array([o.xy for o in objs]); n=len(objs)
    h=np.zeros(len(CATS))
    for o in objs: h[CIX.get(o.category,CIX.get('misc',0))]+=1
    h=h/max(n,1)
    D=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1); iu=np.triu_indices(n,1)
    pd=D[iu]
    area=as_polygon(scene.room).area
    occ=sum(float(o.size[0])*float(o.size[1]) for o in objs)/max(area,1e-3)
    yaw=np.array([o.yaw for o in objs]); axa=np.mean(np.minimum(np.abs(np.sin(2*yaw)),np.abs(np.cos(2*yaw)))<0.15)
    return np.concatenate([[n/15.0, pd.mean()/5.0, pd.std()/5.0, min(occ,2.0), axa], h])

def frechet(A,B):
    A=np.array(A); B=np.array(B)
    mu1,mu2=A.mean(0),B.mean(0); s1,s2=np.cov(A,rowvar=False),np.cov(B,rowvar=False)
    diff=mu1-mu2; cov,_=linalg.sqrtm(s1@s2,disp=False)
    if np.iscomplexobj(cov): cov=cov.real
    return float(diff@diff+np.trace(s1+s2-2*cov))

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
tr,_,test=split_scenes(scenes)
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0",relational_select=True)
def mkroom(r,s):
    p=aspect_deform(r.polygon,float(s[0]),float(s[1])) if isinstance(s,(tuple,list)) else uniform_scale(r.polygon,s)
    a=_anchor_openings(r); return Room(polygon=p,height=r.height,openings=_replace_openings(p,a,len(r.polygon)),room_type=r.room_type)
# real reference distribution = many real 3D-FRONT layouts (train split, held out from test)
real=[feat(s) for s in tr[:400]]; real=[f for f in real if f is not None]
seeds=list(range(60))
sizes=[0.75,1.35,(1.5,0.75),(0.75,1.5)]
GEN={'ReRoom (ours)':[],'Affine warp':[]}
for sd in seeds:
  if sd>=len(test): break
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for s in sizes:
      room=mkroom(src.room,s)
      try:
        r1=generative_retarget(fb,g,room,elasticity=el,bank=bank,cfg=cfg,k=16,polish=False).scene
        f1=feat(r1); GEN['ReRoom (ours)'].append(f1) if f1 is not None else None
        r2=run_baseline("affine_fit",g,room,cfg=cfg); f2=feat(r2); GEN['Affine warp'].append(f2) if f2 is not None else None
      except Exception as e: print('  skip',repr(e)[:40],flush=True)
  except Exception as e: print('skip',sd,repr(e)[:40],flush=True)
print(f"\nFréchet Scene Distance to real 3D-FRONT layouts (n_real={len(real)}) — lower is more realistic")
# real-vs-real lower bound (split the real set)
half=len(real)//2
print(f"{'distribution':<22}{'n':>6}{'FSD↓':>10}")
print(f"{'real vs real (floor)':<22}{half:>6}{frechet(real[:half],real[half:]):>10.3f}")
for k,v in GEN.items():
    v=[x for x in v if x is not None]
    if len(v)>3: print(f"{k:<22}{len(v):>6}{frechet(v,real):>10.3f}")
print("DONE_FSD")
