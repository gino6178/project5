#!/usr/bin/env python
"""Graph-perturbation sensitivity (robustness to imperfect intent graphs).
Retarget with a NOISY intent graph (edges dropped / weights perturbed / motifs
dropped / elasticity α perturbed) but score against the TRUE reference graph.
A graceful degradation curve shows the method does not require a perfect,
hand-authored graph."""
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
from reroom.data.asset_bank import AssetBank

def mkroom(r,s):
    p=uniform_scale(r.polygon,s); a=_anchor_openings(r)
    return Room(polygon=p,height=r.height,openings=_replace_openings(p,a,len(r.polygon)),room_type=r.room_type)

def perturb(g, mode, p, rng):
    """return a noisy COPY of the graph."""
    gg=copy.deepcopy(g)
    if mode=='edge' and gg.relations:
        keep=[r for r in gg.relations if rng.random()>p]
        gg.relations=keep if keep else gg.relations[:1]
    elif mode=='weight':
        for r in gg.relations:
            r.weight=float(max(0.05, r.weight*(1.0+rng.normal(0,2*p))))
    elif mode=='motif' and gg.motifs:
        gg.motifs=[m for m in gg.motifs if rng.random()>p]
    return gg

class NoisyElastic:
    """wrap an elasticity model, adding relative noise to its alpha outputs."""
    def __init__(self, base, sigma, rng): self.base=base; self.sigma=sigma; self.rng=rng
    def alphas(self, ctxs):
        a=np.asarray(self.base.alphas(ctxs),dtype=float)
        return np.clip(a*(1.0+self.rng.normal(0,self.sigma,size=a.shape)),0.0,1.0)
    def __getattr__(self,k): return getattr(self.base,k)

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[6,8,25,2,10,14,1,3,5,7,9,11]
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
cfg=lambda: RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0",relational_select=True)
sizes=[1.0,0.7]   # normal + shrink (shrink exercises graph-driven pruning)
levels=[0.0,0.1,0.2,0.3]
modes=['edge','weight','motif','alpha']
res={(mo,lv):{'S_rel':[],'R_col':[]} for mo in modes for lv in levels}
for sd in seeds:
  try:
    src=test[sd]; g_true=build_motifs(build_scene_graph(src))
    for s in sizes:
      room=mkroom(src.room,s)
      for mo in modes:
        for lv in levels:
          rng=np.random.default_rng(sd*1000+int(lv*100)+hash(mo)%97)
          if mo=='alpha':
            g_use=g_true; el_use=NoisyElastic(el,lv*1.5,rng) if lv>0 else el
          else:
            g_use=perturb(g_true,mo,lv,rng) if lv>0 else g_true; el_use=el
          try:
            sc=generative_retarget(fb,g_use,room,elasticity=el_use,bank=bank,cfg=cfg(),k=16,polish=False).scene
            m=evaluate(g_true,sc)   # score against TRUE graph
            res[(mo,lv)]['S_rel'].append(m['S_rel']); res[(mo,lv)]['R_col'].append(100*m['R_col'])
          except Exception as e: print('  cell skip',mo,lv,repr(e)[:40],flush=True)
  except Exception as e: print('skip',sd,repr(e)[:50],flush=True)
def mn(x): x=np.array(x,float); x=x[~np.isnan(x)]; return x.mean() if len(x) else float('nan')
print(f"\nGraph-perturbation robustness (score vs TRUE graph; n={len(seeds)} refs x {len(sizes)} sizes)")
print(f"{'noise level →':<16}"+''.join(f'{lv:>10.0%}' for lv in levels))
for mo in modes:
    print(f'--- {mo} ---')
    print('  S_rel'.ljust(16)+''.join('%10.3f'%mn(res[(mo,lv)]['S_rel']) for lv in levels))
    print('  R_col%'.ljust(16)+''.join('%10.2f'%mn(res[(mo,lv)]['R_col']) for lv in levels))
    d=mn(res[(mo,0.3)]['S_rel'])-mn(res[(mo,0.0)]['S_rel'])
    print(f"    S_rel drop @30% noise: {d:+.3f}")
print("DONE_GRAPHNOISE")
