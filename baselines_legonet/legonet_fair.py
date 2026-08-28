#!/usr/bin/env python
"""Fair LEGO-Net-as-retargeter baseline, built ENTIRELY on LEGO-Net's own loader
(correct fpbpn/padding/normalization). For each clean 3D-FRONT livingroom, the
retarget input is the affine transplant into a scaled room (scale positions AND
the floorplan boundary by s, in LEGO-Net's normalized frame); LEGO-Net then
denoises. We report scale-normalized relational retention (vs the reference
clean layout) and collision, for the affine init vs LEGO-Net's output."""
import os,sys,numpy as np,torch
sys.path.insert(0,os.getcwd()); os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
from scipy import stats
from data.TDFront import TDFDataset
from model.transformer import TransformerWrapper
DEV="cuda"
CFG=dict(pos_dim=2,ang_dim=2,siz_dim=2,cla_dim=22,maxnfpoc=51,nfpbpn=250,invsha_d=0,use_invariant_shape=False,
 ang_initial_d=128,siz_initial_unit=None,cla_initial_unit=[128,128],invsha_initial_unit=[128,128],
 all_initial_unit=[512,512],final_lin_unit=[256,4],use_two_branch=0,pe_numfreq=32,pe_end=128,
 use_floorplan=True,floorplan_encoder_type='pointnet_simple')
m=TransformerWrapper(**CFG).to(DEV)
sd=torch.load("weights/50000.pt",map_location=DEV)['model_state_dict']
sd={k.replace('module.','',1) if k.startswith('module.') else k:v for k,v in sd.items()}
m.load_state_dict(sd); m.eval()
tdf=TDFDataset("livingroom",use_augment=False)
N=16
out=tdf.gen_3dfront(N, data_partition='test', use_emd=False, abs_pos=True, abs_ang=True,
                    use_floorplan=True, noise_level_stddev=0.0, angle_noise_level_stddev=0.0)
inp,_,pad,_,fpoc,nfpc,fpmask,fpbpn = out
inp=np.asarray(inp); pad=np.asarray(pad); fpbpn=np.asarray(fpbpn)
pd,ad=2,2
def denoise(x,pm,f):
    x=torch.tensor(x,dtype=torch.float32,device=DEV)[None]; pm=torch.tensor(pm,dtype=torch.bool,device=DEV)[None]
    f=torch.tensor(f,dtype=torch.float32,device=DEV)[None]
    for it in range(1500):
        with torch.no_grad(): p=m(x,pm,DEV,fpoc=None,nfpc=None,fpmask=None,fpbpn=f)
        a=p[:,:,pd:pd+ad];p[:,:,pd:pd+ad]=a/(a.norm(dim=-1,keepdim=True)+1e-8)
        disp=(p[:,:,:pd]-x[:,:,:pd]).norm().item()
        nx=x.clone();nx[:,:,:pd+ad]=p[:,:,:pd+ad];x=nx
        if disp<0.01 and it>3: break
    return x[0].detach().cpu().numpy()
def relret(ref,o):
    def nz(p):p=p-p.mean(0);return p/(np.sqrt((p**2).sum(1).mean())+1e-8)
    a=nz(ref);b=nz(o);n=len(a);e=[]
    for i in range(n):
        for j in range(i+1,n):
            oa=a[j]-a[i];ob=b[j]-b[i];e.append(min(1.,np.linalg.norm(ob-oa)/(np.linalg.norm(oa)+1e-8)))
    return 1-np.mean(e) if e else 1.0
def collide(pos_m,half):
    n=len(pos_m);c=t=0
    for i in range(n):
        for j in range(i+1,n):
            t+=1;dx=abs(pos_m[i,0]-pos_m[j,0]);dz=abs(pos_m[i,1]-pos_m[j,1])
            if (half[i,0]+half[j,0]-dx)>0.02 and (half[i,1]+half[j,1]-dz)>0.02:c+=1
    return 100*c/max(t,1)
rows={'aff':{'ret':[],'col':[]},'lego':{'ret':[],'col':[]}}
per={0.75:{'ret':[],'col':[]},1.0:{'ret':[],'col':[]},1.35:{'ret':[],'col':[]}}
for s_i in range(N):
    nobj=int((~pad[s_i].astype(bool)).sum())
    if nobj<3: continue
    clean=inp[s_i].copy()  # [21,28] normalized
    ref_pos=clean[:nobj,:pd].copy()  # reference (normalized)
    # half-extents in metres: siz stored = size*2/6-1  => size(half,m) = (siz+1)*6/2 /2 = (siz+1)*1.5
    half=(clean[:nobj,pd+ad:pd+ad+2]+1)*1.5
    for s in (0.75,1.0,1.35):
        x=clean.copy(); x[:nobj,:pd]*=s          # affine transplant into scaled room
        f=fpbpn[s_i].copy(); f[:, :2]*=s          # scale floorplan boundary points
        aff_pos=x[:nobj,:pd].copy()
        den=denoise(x,pad[s_i],f); den_pos=den[:nobj,:pd]
        # metres for collision (positions *6)
        rows['aff']['ret'].append(relret(ref_pos,aff_pos)); rows['aff']['col'].append(collide(aff_pos*6,half))
        rows['lego']['ret'].append(relret(ref_pos,den_pos)); rows['lego']['col'].append(collide(den_pos*6,half))
        per[s]['ret'].append(relret(ref_pos,den_pos)); per[s]['col'].append(collide(den_pos*6,half))
def ms(a):a=np.array(a);return a.mean(),a.std()
print(f"\nN cells = {len(rows['aff']['ret'])}")
print(f"{'variant':<24}{'RelRet':>14}{'Collision%':>14}")
for k,l in [('aff','Affine warp (init)'),('lego','LEGO-Net denoise')]:
    (rm,rs)=ms(rows[k]['ret']);(cm,cs)=ms(rows[k]['col']); print(f"{l:<24}{rm:6.3f}±{rs:<5.3f}{cm:7.2f}±{cs:<5.2f}")
d=np.array(rows['lego']['ret'])-np.array(rows['aff']['ret'])
try:_,p=stats.wilcoxon(d)
except:p=float('nan')
print(f"  paired RelRet (LEGO−affine): Δ={d.mean():+.3f} (win {100*(d>0).mean():.0f}%) Wilcoxon p={p:.4g}")
print("per-scale RelRet(LEGO): "+" ".join(f"s={s}:{np.mean(per[s]['ret']):.3f}" for s in (0.75,1.0,1.35)))
print("DONE_FAIR")
