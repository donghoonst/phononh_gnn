"""Synthetic harmonic fixtures, explicitly NOT DFT data or a materials benchmark."""
from pathlib import Path
import numpy as np
import torch
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_CONSTANTS
from .data import from_phonopy
from .physics import scatter,constrain_pairs,compact_from_pairs


def fixture(seed=0,supercell=(2,2,2)):
    rng=np.random.default_rng(seed)
    cell=np.array([[4.7,0.0,0.0],[0.3,4.6,0.0],[0.2,0.1,4.5]])*(1+seed*0.012)
    frac=np.array([[0.12,0.18,0.23],[0.53,0.35,0.63],[0.74,0.79,0.41]])+rng.normal(0,0.012,(3,3))
    atoms=PhonopyAtoms(symbols=["Si","C","O"],cell=cell,scaled_positions=frac)
    ph=Phonopy(atoms,supercell_matrix=np.diag(supercell),primitive_matrix=np.eye(3),is_symmetry=False)
    return ph


def attach_synthetic_fc(ph):
    s=from_phonopy(ph,4.5,5.5)
    v=s["pair_vectors"]; r=v.norm(dim=-1); u=v/r[:,None]
    raw=-torch.exp(-r[:,None,None]/2)*(torch.eye(3,dtype=v.dtype)+0.6*u[:,:,None]*u[:,None,:])
    # Small noncentral skew contribution; projection retains only allowed cycle flow.
    axis=torch.linalg.cross(s["cell"][0],s["cell"][1]);axis=axis/axis.norm()
    x,y,z=axis.unbind();zero=x*0
    skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
    i,j=s["pair_index"][:,s["image_owner"]]
    raw=raw+0.03*torch.sin((i-j).to(v))[:,None,None]*torch.exp(-r[:,None,None]/3)*skew
    owner=s["image_owner"];counts=torch.bincount(owner).to(v)
    pairs=constrain_pairs(scatter(raw,owner,len(counts))/counts[:,None,None],s)
    ph.force_constants=compact_from_pairs(pairs,s).numpy()
    return ph


def make_dataset(root,count=6):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for seed in range(count):
        path=root/f"synthetic_{seed:03d}";path.mkdir(exist_ok=True)
        ph=attach_synthetic_fc(fixture(seed))
        ph.save(filename=str(path/"phonopy.yaml"))
        write_FORCE_CONSTANTS(ph.force_constants,filename=str(path/"FORCE_CONSTANTS"),p2s_map=ph.primitive.p2s_map)
    (root/"PROVENANCE.txt").write_text("SYNTHETIC harmonic fixtures. No DFT calculations. Do not report as materials accuracy.\n")
