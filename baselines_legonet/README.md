# LEGO-Net cross-lineage baseline (Table 4)

LEGO-Net (Wei et al., CVPR 2023) repurposed as a retargeter. Runs locally in a
py3.7/torch-1.12 conda env (`legonet`); the torch exec-stack ELF flag must be
cleared for this hardened kernel (see clear_execstack in session notes).

Result (48 layouts = 16 livingrooms x 3 sizes, via LEGO-Net's own loader):
  Affine warp (init):  RelRet 1.000, collision 1.28%
  LEGO-Net denoise:    RelRet 0.703, collision 2.86%   (paired Δ=-0.30, p=1.6e-9)
Finding: LEGO-Net regularizes any layout toward its tidy ideal rather than
binding the specific reference. Validated as genuine (clean-scene fixed-point
test gives RelRet~0.74, matching).

Scripts: legonet_fair.py (fair eval), legonet_validate.py (clean-scene sanity).
Repo cloned at /home/gino/project/baselines/LEGO-Net; weights via gdown.
