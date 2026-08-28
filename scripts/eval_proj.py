#!/usr/bin/env python
"""D3 eval: on the same raw flow_bfresh outputs, compare projection variants
  none | hard-snap (shipped) | diff-proj (D3) | polish
on R_col, wall-snap%, S_rel.  Establishes whether the differentiable projection
matches the hard snap (so it can replace it and be trained through)."""
import os, sys, math, argparse
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
import numpy as np
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
from reroom.geom.polygon import as_polygon, object_polygon
from reroom.data.asset_bank import AssetBank

def scaled_room(room, s):
    poly = uniform_scale(room.polygon, s)
    a = _anchor_openings(room)
    return Room(polygon=poly, height=room.height,
                openings=_replace_openings(poly, a, len(room.polygon)),
                room_type=room.room_type)

def snap_pct(scene):
    """fraction of wall-affinity objects flush (<=12cm) AND square (<=4deg)."""
    poly = as_polygon(scene.room); ring = np.asarray(poly.exterior.coords)[:-1]
    edges = [(ring[i], ring[(i+1)%len(ring)]) for i in range(len(ring))]
    ok=tot=0
    for o in scene.objects:
        if not o.keep or o.category not in WALL_CATS: continue
        tot+=1; corners=o.corners(); best=None
        for a,b in edges:
            d=b-a; L=np.linalg.norm(d)+1e-9; t=d/L; nrm=np.array([-t[1],t[0]])
            rel=corners-a; perp=np.abs(rel@nrm); pj=rel@t
            ins=(pj>-0.05)&(pj<L+0.05)
            if not ins.any(): continue
            gap=perp[ins].min()
            if best is None or gap<best[0]:
                ang=math.atan2(t[1],t[0])
                ax0=np.array([math.cos(o.yaw),math.sin(o.yaw)])
                ax1=np.array([-math.sin(o.yaw),math.cos(o.yaw)])
                sk=min(abs(((math.atan2(ax0[1],ax0[0])-ang+math.pi/2)%math.pi)-math.pi/2),
                       abs(((math.atan2(ax1[1],ax1[0])-ang+math.pi/2)%math.pi)-math.pi/2))
                best=(gap,math.degrees(sk))
        if best is not None and best[0]<=0.12 and best[1]<=4.0: ok+=1
    return ok/max(tot,1), tot

def measure(g, scene):
    m=evaluate(g,scene); sp,_=snap_pct(scene)
    return 100*m["R_col"], 100*sp, m["S_rel"]

ap=argparse.ArgumentParser()
ap.add_argument("--flow", default="outputs/flow_bfresh/flow_best.pt")
ap.add_argument("--seeds", default="6,8,25,2,10,14,1,3,5,7,9,11")
ap.add_argument("--sizes", default="0.75,1.0,1.35")
a=ap.parse_args()

flow=load_flow(a.flow, device="cuda:0")
el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6)
        if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[int(x) for x in a.seeds.split(",")]; sizes=[float(x) for x in a.sizes.split(",")]

agg={k:[[],[],[]] for k in ("raw","snap","diff","polish")}
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for s in sizes:
        room=scaled_room(src.room,s)
        # raw = no polish, no snap
        raw=generative_retarget(flow,g,room,elasticity=el,bank=bank,
              cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0"),k=16,polish=False).scene
        for i,v in enumerate(measure(g,raw)): agg["raw"][i].append(v)
        # hard snap (shipped) operates in place on a copy
        import copy
        sc=copy.deepcopy(raw); regularity_snap(sc)
        for i,v in enumerate(measure(g,sc)): agg["snap"][i].append(v)
        # D3 differentiable projection
        dp=project_scene(raw, room, iters=40, lr=0.03, device="cuda:0")
        for i,v in enumerate(measure(g,dp)): agg["diff"][i].append(v)
        # polish (25-step)
        pol=generative_retarget(flow,g,room,elasticity=el,bank=bank,
              cfg=RetargetConfig(restarts=16,regularity_snap=True,device="cuda:0"),k=16,polish=True).scene
        for i,v in enumerate(measure(g,pol)): agg["polish"][i].append(v)
  except Exception as _e:
    print("skip seed", sd, _e)

def ms(x): 
    import numpy as _n; a=_n.array(x); return a.mean(), a.std()
print(f"\nN cells = {len(agg['raw'][0])}")
print(f"{'variant':<26}{'R_col%':>14}{'snap%':>12}{'S_rel':>14}")
for k in ("raw","snap","diff","polish"):
    (rm,rs),(pm,ps),(sm,ss)=[ms(x) for x in agg[k]]
    print(f"{k:<26}{rm:6.2f}±{rs:<5.2f}{pm:6.0f}±{ps:<4.0f}{sm:6.3f}±{ss:<5.3f}")

from scipy import stats as _st
import numpy as _np
def _clus(am,bm,idx,lbl,nsz=3):
    d=_np.array(agg[am][idx])-_np.array(agg[bm][idx])
    dr=d.reshape(-1,nsz).mean(1)
    try:_,p=_st.wilcoxon(dr)
    except:p=float('nan')
    print(f"  {lbl}: clustered n={len(dr)} Δ={dr.mean():+.3f} p={p:.3g}")
print("\nReference-clustered (n=12):")
_clus("diff","raw",2,"Πθ − raw S_rel")
_clus("diff","raw",0,"Πθ − raw R_col")
_clus("diff","snap",2,"Πθ − snap S_rel")

print("\nDONE_EVAL_PROJ")
