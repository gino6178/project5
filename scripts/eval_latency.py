#!/usr/bin/env python
"""Inference-latency breakdown: per-layout wall-clock of each finishing strategy,
sharing the same base flow sample, on a single L40. Supports the efficiency claim
(one forward pass + one projection)."""
import os, sys, copy, time, numpy as np
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
import torch
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import uniform_scale, _anchor_openings, _replace_openings
from reroom.core.scene import Room
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.retarget.regularity import regularity_snap
from reroom.retarget.diffproj import project_scene
from reroom.retarget.baselines import run_baseline

def mkroom(r,s):
    p=uniform_scale(r.polygon,s); a=_anchor_openings(r)
    return Room(polygon=p,height=r.height,openings=_replace_openings(p,a,len(r.polygon)),room_type=r.room_type)
def clk():
    torch.cuda.synchronize() if torch.cuda.is_available() else None; return time.perf_counter()

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[6,8,25,2,10,14,1,3,5,7,9,11]
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0")
T={k:[] for k in ('affine','flow_raw','snap','pi','polish','heavy')}
# warmup
src=test[6]; g=build_motifs(build_scene_graph(src)); room=mkroom(src.room,1.0)
_=generative_retarget(fb,g,room,elasticity=el,cfg=cfg,k=16,polish=False)
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src)); room=mkroom(src.room,1.0)
    t=clk(); base=run_baseline("affine_fit",g,room,cfg=cfg); T['affine'].append((clk()-t)*1000)
    t=clk(); r=generative_retarget(fb,g,room,elasticity=el,cfg=cfg,k=16,polish=False).scene; T['flow_raw'].append((clk()-t)*1000)
    t=clk(); s2=copy.deepcopy(r); regularity_snap(s2); T['snap'].append((clk()-t)*1000)
    t=clk(); _=project_scene(r,room,iters=40,lr=0.03,device="cuda:0"); T['pi'].append((clk()-t)*1000)
    t=clk(); _=generative_retarget(fb,g,room,elasticity=el,cfg=cfg,k=16,polish=True).scene; T['polish'].append((clk()-t)*1000)
    t=clk(); _=generative_retarget(fb,g,room,elasticity=el,cfg=cfg,k=16,polish=False,project=True).scene; T['heavy'].append((clk()-t)*1000)
  except Exception as e: print('skip',sd,repr(e)[:50],flush=True)
def ms(x): x=np.array(x); return x.mean(),x.std()
print(f"\nInference latency (ms/layout, single L40, k=16, n={len(T['flow_raw'])})")
print(f"{'method':<34}{'ms':>10}{'±std':>8}")
lab=[('affine','Affine warp (baseline)'),('flow_raw','ReRoom flow (sample only)'),
     ('snap','  + hard snap'),('pi','  + differentiable Πθ (ours)'),
     ('polish','  + light polish'),
     ('heavy','  + heavy multi-step projection (project=True)')]
for k,l in lab:
    m,s=ms(T[k]); print(f"{l:<34}{m:>10.1f}{s:>8.1f}")
fr=ms(T['flow_raw'])[0]; pi=ms(T['pi'])[0]; po=ms(T['polish'])[0]
print(f"\nReRoom (flow + Πθ) = {fr+pi:.0f} ms vs full multi-step retarget {po:.0f} ms  → {po/(fr+pi):.1f}× faster")
print("DONE_LATENCY")
