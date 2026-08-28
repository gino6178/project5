#!/usr/bin/env python
"""Fair, runnable baseline for the retargeting task: AFFINE / global warp — the
exact method the paper critiques (transplant the reference by MRR-affine into
the target boundary, no flow). Compared against Ours (flow + differentiable
projection) on the SAME seeds/sizes as Table 1's projection ablation, so the
baseline row is directly comparable."""
import os, sys, math, time, argparse
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
from reroom.generative.train import warp_scene            # MRR-affine transplant
from reroom.retarget.regularity import WALL_CATS
from reroom.retarget.diffproj import project_scene
from reroom.geom.polygon import as_polygon
from reroom.data.asset_bank import AssetBank

def scaled_room(r, s):
    p = uniform_scale(r.polygon, s); a = _anchor_openings(r)
    return Room(polygon=p, height=r.height, openings=_replace_openings(p, a, len(r.polygon)), room_type=r.room_type)

def snap_pct(scene):
    poly = as_polygon(scene.room); ring = np.asarray(poly.exterior.coords)[:-1]
    edges = [(ring[i], ring[(i+1)%len(ring)]) for i in range(len(ring))]
    ok=tot=0
    for o in scene.objects:
        if not o.keep or o.category not in WALL_CATS: continue
        tot+=1; corners=o.corners(); best=None
        for a,b in edges:
            d=b-a; L=np.linalg.norm(d)+1e-9; t=d/L; nrm=np.array([-t[1],t[0]])
            rel=corners-a; perp=np.abs(rel@nrm); pj=rel@t; ins=(pj>-0.05)&(pj<L+0.05)
            if not ins.any(): continue
            gap=perp[ins].min()
            if best is None or gap<best[0]:
                ang=math.atan2(t[1],t[0]); ax0=np.array([math.cos(o.yaw),math.sin(o.yaw)]); ax1=np.array([-math.sin(o.yaw),math.cos(o.yaw)])
                sk=min(abs(((math.atan2(ax0[1],ax0[0])-ang+math.pi/2)%math.pi)-math.pi/2),
                       abs(((math.atan2(ax1[1],ax1[0])-ang+math.pi/2)%math.pi)-math.pi/2))
                best=(gap,math.degrees(sk))
        if best is not None and best[0]<=0.12 and best[1]<=4.0: ok+=1
    return ok/max(tot,1)

def measure(g, sc):
    m=evaluate(g,sc); return 100*m["R_col"], 100*snap_pct(sc), m["S_rel"], m.get("S_rel_kept", m["S_rel"])

ap=argparse.ArgumentParser()
ap.add_argument("--flow", default="outputs/flow_bfresh/flow_best.pt")
ap.add_argument("--seeds", default="6,8,25,2,10,14,1,3,5,7,9,11")
ap.add_argument("--sizes", default="0.75,1.0,1.35")
a,_=ap.parse_known_args()
flow=load_flow(a.flow, device="cuda:0")
el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[int(x) for x in a.seeds.split(",")]; sizes=[float(x) for x in a.sizes.split(",")]

agg={"affine":[[],[],[],[],[]], "ours":[[],[],[],[],[]]}
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for s in sizes:
        room=scaled_room(src.room, s)
        # --- affine / global warp baseline ---
        t0=time.time(); base=warp_scene(src, room, clip_inside=True); tb=1000*(time.time()-t0)
        for i,v in enumerate(measure(g,base)): agg["affine"][i].append(v)
        agg["affine"][4].append(tb)
        # --- ours: flow + differentiable projection ---
        t0=time.time()
        raw=generative_retarget(flow,g,room,elasticity=el,bank=bank,
              cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0"),k=16,polish=False).scene
        out=project_scene(raw, room, iters=40, lr=0.03, device="cuda:0")
        to=1000*(time.time()-t0)
        for i,v in enumerate(measure(g,out)): agg["ours"][i].append(v)
        agg["ours"][4].append(to)
  except Exception as _e:
    print("skip",sd,_e)

print(f"\n{'method':<28}{'R_col%':>8}{'snap%':>8}{'S_rel':>8}{'Srel_kept':>10}{'latency':>10}")
for k,label in [("affine","Affine / global warp"),("ours","Ours: flow + diff Πθ")]:
    r=agg[k]
    a=[np.array(x) for x in r]
    print(f"{label:<28}{a[0].mean():5.2f}±{a[0].std():<4.2f}{a[1].mean():5.0f}±{a[1].std():<3.0f}{a[2].mean():5.3f}±{a[2].std():<4.3f}{a[3].mean():5.3f}±{a[3].std():<4.3f}{a[4].mean():6.0f}ms")

from scipy import stats as _st
import numpy as _np
def _clus(idx,lbl,nsz=3):
    d=_np.array(agg["ours"][idx])-_np.array(agg["affine"][idx])
    dr=d.reshape(-1,nsz).mean(1)
    try:_,p=_st.wilcoxon(dr)
    except:p=float('nan')
    print(f"  {lbl}: clustered n={len(dr)} Δ(ours−affine)={dr.mean():+.3f} p={p:.3g}")
print("\nReference-clustered (n=12), ours vs affine:")
_clus(3,"S_rel_kept"); _clus(2,"S_rel"); _clus(0,"R_col%")

print("\nDONE_BASELINE")
