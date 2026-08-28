#!/usr/bin/env python
"""Measurement validity: does the affine warp actually break design identity where
the motivation claims (metric intra-motif distances under non-uniform scaling)?
S_rel's offset features are scale-normalized, so affine trivially preserves them —
but the METRIC (meters) intra-motif spacing and the group-level S_motif are what
identity really means. Under anisotropic stretch, compare affine vs ReRoom on
(a) S_motif and (b) mean metric intra-motif distance drift (meters)."""
import os, sys, itertools, numpy as np
_ROOT = os.environ.get("REROOM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)   # repo root; override with REROOM_ROOT
from reroom.data.corpus import iter_scenes, split_scenes
from reroom.geom.deform import aspect_deform, _anchor_openings, _replace_openings
from reroom.core.scene import Room
from reroom.eval.metrics import evaluate
from reroom.intent.motifs import build_motifs
from reroom.intent.relations import build_scene_graph
from reroom.intent.elasticity import load_elasticity
from reroom.retarget.optimizer import RetargetConfig
from reroom.generative.sample import load_flow, generative_retarget
from reroom.retarget.baselines import run_baseline
from reroom.data.asset_bank import AssetBank
from scipy import stats

def mkroom(r,s):
    p=aspect_deform(r.polygon,float(s[0]),float(s[1])); a=_anchor_openings(r)
    return Room(polygon=p,height=r.height,openings=_replace_openings(p,a,len(r.polygon)),room_type=r.room_type)

def motif_drift(src, out, motifs):
    """mean |Δ metric distance| (meters) over intra-motif member pairs present in out."""
    om={o.oid:o for o in out.objects if o.keep}
    ds=[]
    for m in motifs:
        mem=list(m.members)
        for a,b in itertools.combinations(mem,2):
            oa,ob=src.objects[a],src.objects[b]
            if oa.oid in om and ob.oid in om:
                d_ref=float(np.linalg.norm(np.array(oa.xy)-np.array(ob.xy)))
                d_out=float(np.linalg.norm(np.array(om[oa.oid].xy)-np.array(om[ob.oid].xy)))
                ds.append(abs(d_out-d_ref))
    return float(np.mean(ds)) if ds else np.nan

el=load_elasticity("outputs/elasticity/neural.pt") if os.path.exists("outputs/elasticity/neural.pt") else None
bank=AssetBank.load("outputs/priors/assets_future.pkl") if os.path.exists("outputs/priors/assets_future.pkl") else None
scenes=[s for s in iter_scenes("data/processed",limit=None,min_objects=6) if s.room.room_type in ("bedroom","living_room")]
_,_,test=split_scenes(scenes)
seeds=[6,8,25,2,10,14,1,3,5,7,9,11]
sizes=[(1.5,0.75),(0.75,1.5),(1.7,0.85),(0.85,1.7)]   # anisotropic (affine should break)
fb=load_flow("outputs/flow_bfresh/flow_best.pt",device="cuda:0")
cfg=RetargetConfig(restarts=16,regularity_snap=False,device="cuda:0",relational_select=True)
res={'affine':{'S_motif':[],'drift':[],'S_rel_kept':[]},'ours':{'S_motif':[],'drift':[],'S_rel_kept':[]}}
nsz=len(sizes)
for sd in seeds:
  try:
    src=test[sd]; g=build_motifs(build_scene_graph(src))
    for s in sizes:
      room=mkroom(src.room,s)
      aff=run_baseline("affine_fit",g,room,cfg=cfg)
      our=generative_retarget(fb,g,room,elasticity=el,bank=bank,cfg=cfg,k=16,polish=False).scene
      for nm,sc in (('affine',aff),('ours',our)):
        m=evaluate(g,sc)
        res[nm]['S_motif'].append(m.get('S_motif',np.nan))
        res[nm]['S_rel_kept'].append(m.get('S_rel_kept',m['S_rel']))
        res[nm]['drift'].append(motif_drift(src,sc,g.motifs))
  except Exception as e: print('skip',sd,repr(e)[:50],flush=True)
def mn(x): x=np.array(x,float); x=x[~np.isnan(x)]; return (x.mean(),x.std())
print(f"\nMeasurement validity under ANISOTROPIC stretch (n={len(seeds)} refs x {nsz})")
print(f"{'method':<16}{'S_motif↑':>11}{'metric drift(m)↓':>18}{'S_rel_kept↑':>13}")
for k,l in (('affine','Affine warp'),('ours','ReRoom (ours)')):
    r=res[k]; print(f"{l:<16}{mn(r['S_motif'])[0]:7.3f}{mn(r['drift'])[0]:18.3f}{mn(r['S_rel_kept'])[0]:13.3f}")
def clus(key,invert=False):
    a=np.array(res['ours'][key],float); b=np.array(res['affine'][key],float)
    msk=~(np.isnan(a)|np.isnan(b)); a,b=a[msk],b[msk]; d=a-b
    dr=d[:len(d)//nsz*nsz].reshape(-1,nsz).mean(1)
    try:_,p=stats.wilcoxon(dr)
    except:p=float('nan')
    print(f"  {key}: ours−affine Δ={dr.mean():+.3f} clustered n={len(dr)} p={p:.3g}")
print("Reference-clustered (ours − affine):")
clus('S_motif'); clus('drift'); clus('S_rel_kept')
print("DONE_MOTIFVALID")
