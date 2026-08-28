#!/usr/bin/env python
"""Validation: use LEGO-Net's OWN loader (correct fpbpn/padding) with ZERO noise,
then denoise. A clean scene should be a near fixed-point of a working denoiser.
This isolates whether my earlier 'wrecks clean scene' was a conversion bug (my
fpbpn) or LEGO-Net's real behavior."""
import os,sys,numpy as np,torch
sys.path.insert(0,os.getcwd())
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
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
# their loader, ZERO noise -> input == clean
out=tdf.gen_3dfront(8, data_partition='test', use_emd=False, abs_pos=True, abs_ang=True,
                    use_floorplan=True, noise_level_stddev=0.0, angle_noise_level_stddev=0.0)
inp,labels,pad,_,fpoc,nfpc,fpmask,fpbpn = out
print("shapes: inp",inp.shape,"pad",None if pad is None else pad.shape,"fpbpn",None if fpbpn is None else fpbpn.shape)
inp=torch.tensor(inp,dtype=torch.float32,device=DEV); pad=torch.tensor(pad,dtype=torch.bool,device=DEV)
fpb=torch.tensor(fpbpn,dtype=torch.float32,device=DEV) if fpbpn is not None else None
pd,ad=2,2
def rr(a,b):
    a=a-a.mean(0);a=a/ (np.sqrt((a**2).sum(1).mean())+1e-8)
    b=b-b.mean(0);b=b/ (np.sqrt((b**2).sum(1).mean())+1e-8)
    n=len(a);e=[]
    for i in range(n):
        for j in range(i+1,n):
            oa=a[j]-a[i];ob=b[j]-b[i];e.append(min(1.,np.linalg.norm(ob-oa)/(np.linalg.norm(oa)+1e-8)))
    return 1-np.mean(e) if e else 1.0
for s_i in range(inp.shape[0]):
    nobj=int((~pad[s_i]).sum().item())
    x=inp[s_i:s_i+1].clone(); f=fpb[s_i:s_i+1] if fpb is not None else None; pm=pad[s_i:s_i+1]
    clean=x[0,:nobj,:pd].detach().cpu().numpy().copy()
    for it in range(1500):
        with torch.no_grad(): p=m(x,pm,DEV,fpoc=None,nfpc=None,fpmask=None,fpbpn=f)
        a=p[:,:,pd:pd+ad];p[:,:,pd:pd+ad]=a/(a.norm(dim=-1,keepdim=True)+1e-8)
        disp=(p[:,:,:pd]-x[:,:,:pd]).norm().item()
        nx=x.clone();nx[:,:,:pd+ad]=p[:,:,:pd+ad];x=nx
        if disp<0.01 and it>3: break
    out_pos=x[0,:nobj,:pd].detach().cpu().numpy()
    moved=np.mean(np.linalg.norm(out_pos-clean,axis=1))
    print(f"scene {s_i}: nobj={nobj} RelRet(clean,denoised)={rr(clean,out_pos):.3f} mean_moved={moved:.3f} (norm units, *6=m)")
print("DONE_VALIDATE")
