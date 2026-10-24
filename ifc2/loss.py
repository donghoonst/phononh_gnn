"""Train-only scaling; physical targets are audited, never silently rewritten."""
import torch
from .physics import decompose,constrain_pairs,compact_from_pairs,dynamical_matrix

SLICES=(slice(0,1),slice(1,4),slice(4,9))


def train_scales(samples):
    coeff=torch.cat([decompose(s["target"]) for s in samples])
    return torch.stack([coeff[:,sl].square().mean().sqrt().clamp_min(1e-4) for sl in SLICES])


def fc_loss(pred,target,scales):
    delta=decompose(pred-target)
    return torch.stack([(delta[:,sl]/scales[k]).square().mean() for k,sl in enumerate(SLICES)]).mean()


def reciprocal_loss(pred,sample,q):
    # Both sides occupy the same cutoff + reciprocity + ASR hypothesis space.
    target=constrain_pairs(sample["target"],sample)
    pd=dynamical_matrix(compact_from_pairs(pred,sample),sample,q)
    td=dynamical_matrix(compact_from_pairs(target,sample),sample,q)
    scale=td.abs().square().mean().clamp_min(1e-10)
    dloss=(pd-td).abs().square().mean()/scale
    if max(float((pd-pd.mH).abs().max().detach()),float((td-td.mH).abs().max()))>1e-7:
        raise ValueError("Reciprocal loss received non-Hermitian D")
    eigloss=(torch.linalg.eigvalsh(pd)-torch.linalg.eigvalsh(td)).square().mean()/(td.shape[-1]*scale)
    return dloss,eigloss
