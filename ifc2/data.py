"""Geometry-only periodic graphs and explicit Phonopy compact-label mapping."""
from pathlib import Path
import hashlib
import numpy as np
import torch
import phonopy
from ase import Atoms
from ase.neighborlist import neighbor_list


def shortest_vectors(primitive):
    vectors, multi = primitive.get_smallest_vectors()
    ns, n = len(primitive.s2p_map), len(primitive)
    # Never guess transposition from shape: ns == n is ambiguous.
    if multi.shape == (ns, n, 2) and vectors.ndim == 2:
        return [[vectors[int(multi[s,i,1]):int(multi[s,i,1]+multi[s,i,0])]
                 for s in range(ns)] for i in range(n)]
    if multi.shape == (ns, n) and vectors.ndim == 4:
        return [[vectors[s,i,:int(multi[s,i])] for s in range(ns)] for i in range(n)]
    raise ValueError(f"Unsupported Phonopy shortest-vector layout: {vectors.shape}, {multi.shape}")


def graph(z, positions, cell, cutoff, pbc=True):
    atoms = Atoms(numbers=np.asarray(z), positions=np.asarray(positions), cell=np.asarray(cell), pbc=pbc)
    i, j, shift, vec = neighbor_list("ijSD", atoms, cutoff, self_interaction=False)
    keep = (np.linalg.norm(vec, axis=1)>1e-8) & (np.linalg.norm(vec,axis=1)<cutoff)
    return (torch.tensor(np.stack([i[keep],j[keep]]),dtype=torch.long),
            torch.tensor(vec[keep],dtype=torch.float64), torch.tensor(shift[keep],dtype=torch.long))


def from_phonopy(ph, conv_cutoff=4.5, pair_cutoff=5.5, material_id="structure"):
    p = ph.primitive
    n, ns = len(p), len(ph.supercell)
    p2s = np.asarray(p.p2s_map)
    representatives = {int(s):i for i,s in enumerate(p2s)}
    s2p = np.array([representatives[int(s)] for s in p.s2p_map])
    vectors = shortest_vectors(p)
    frac, cell = np.asarray(p.scaled_positions), np.asarray(p.cell)
    pi, pj, ps, image, owner, shifts = [], [], [], [], [], []
    dm_vec, dm_owner, dm_count = [], [], []
    for i in range(n):
        for s in range(ns):
            vf = np.asarray(vectors[i][s])
            if not len(vf): raise ValueError("Empty shortest-vector set")
            dm_vec.extend(vf); dm_owner.extend([i*ns+s]*len(vf)); dm_count.append(len(vf))
            r = vf @ cell
            dist = np.linalg.norm(r,axis=1)
            if not np.allclose(dist,dist[0],atol=1e-6): raise ValueError("Unequal shortest distances")
            if s == p2s[i] and dist[0]<1e-8: continue
            if dist[0]>=pair_cutoff: continue
            j = int(s2p[s]); shift = vf - (frac[j]-frac[i])
            if not np.allclose(shift,np.rint(shift),atol=1e-6): raise ValueError("Noninteger primitive translation")
            owner.extend([len(pi)]*len(vf)); image.extend(r); shifts.extend(np.rint(shift).astype(int))
            pi.append(i); pj.append(j); ps.append(s)
    if not pi: raise ValueError("No target pairs inside cutoff")
    owner = np.asarray(owner)
    shifts = np.asarray(shifts)
    keys = {}
    for k,(i,j) in enumerate(zip(pi,pj)):
        key = (i,j,tuple(sorted(map(tuple,shifts[owner==k]))))
        if key in keys: raise ValueError("Duplicate compact image group")
        keys[key]=k
    reverse=[]
    for k,(i,j) in enumerate(zip(pi,pj)):
        key=(j,i,tuple(sorted(map(tuple,-shifts[owner==k]))))
        if key not in keys: raise ValueError("Target graph is not closed under reversal")
        reverse.append(keys[key])
    image_keys={(pi[k],pj[k],tuple(shifts[t])):t for t,k in enumerate(owner)}
    image_reverse=[image_keys[(pj[k],pi[k],tuple(-shifts[t]))] for t,k in enumerate(owner)]
    lap = np.zeros((n,n))
    for i,j in zip(pi,pj): lap[i,i]+=1; lap[i,j]-=1
    ci,cv,cs=graph(p.numbers,p.positions,cell,conv_cutoff)
    f=lambda x:torch.tensor(np.asarray(x),dtype=torch.float64)
    l=lambda x:torch.tensor(np.asarray(x),dtype=torch.long)
    sample=dict(material_id=material_id,z=l(p.numbers),positions=f(p.positions),cell=f(cell),
                fractional=f(frac),masses=f(p.masses),p2s=l(p2s),s2p=l(s2p),n_super=ns,
                pair_index=l([pi,pj]),pair_super=l(ps),pair_vectors=f(image),image_owner=l(owner),
                pair_shift=l(shifts),reverse=l(reverse),laplacian_pinv=f(np.linalg.pinv(lap,hermitian=True)),
                image_reverse=l(image_reverse),
                conv_index=ci,conv_vectors=cv,conv_shift=cs,dm_vectors=f(dm_vec),dm_owner=l(dm_owner),
                dm_count=f(dm_count),frequency_factor=float(ph.unit_conversion_factor))
    if ph.force_constants is not None:
        fc=np.asarray(ph.force_constants)
        if fc.shape[:2] == (ns,ns): fc=fc[p2s]
        if fc.shape != (n,ns,3,3): raise ValueError(f"Invalid FC shape: {fc.shape}")
        if not np.isfinite(fc).all(): raise ValueError("Nonfinite FC")
        sample["target_compact"]=f(fc)
        sample["target"]=f(fc[np.asarray(pi),np.asarray(ps)])
    return sample


def load_material(path, conv_cutoff=4.5, pair_cutoff=5.5, nac=False, labels=True):
    path=Path(path)
    kwargs=dict(phonopy_yaml=str(path/"phonopy.yaml"),is_nac=nac,is_compact_fc=True)
    if labels: kwargs["force_constants_filename"]=str(path/"FORCE_CONSTANTS")
    if nac and (path/"BORN").exists(): kwargs["born_filename"]=str(path/"BORN")
    ph=phonopy.load(**kwargs)
    if not labels: ph.force_constants=None
    return ph,from_phonopy(ph,conv_cutoff,pair_cutoff,path.name)


def material_split(ids,seed=17):
    if len(ids)!=len(set(ids)): raise ValueError("Duplicate material identifiers")
    if len(ids)<3: raise ValueError("At least three independent materials required")
    ordered=sorted(ids,key=lambda x:hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    nv=max(1,len(ids)//5); nt=max(1,len(ids)//5)
    return dict(train=ordered[nv+nt:],validation=ordered[:nv],test=ordered[nv:nv+nt])


def to_device(sample,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in sample.items()}
