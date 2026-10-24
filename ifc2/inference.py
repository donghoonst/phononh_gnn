"""Checkpoint inference and Phonopy validation, with optional supplied BORN/NAC."""
from pathlib import Path
import copy,json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phonopy.file_IO import write_FORCE_CONSTANTS
from .data import load_material,from_phonopy
from .physics import compact_from_pairs,dynamical_matrix,diagnostics
from .training import load_checkpoint


def structure_phonopy(path,supercell=(2,2,2)):
    from ase.io import read
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    atoms=read(path)
    if not atoms.pbc.all(): raise ValueError("Phonopy export requires 3D periodic cell; use a vacuum supercell for slabs")
    unit=PhonopyAtoms(numbers=atoms.numbers,masses=atoms.get_masses(),cell=atoms.cell.array,scaled_positions=atoms.get_scaled_positions())
    return Phonopy(unit,supercell_matrix=np.diag(supercell),primitive_matrix=np.eye(3),is_symmetry=False)


def predict(checkpoint,material,out,structure=False,supercell=(2,2,2)):
    model,_=load_checkpoint(checkpoint);c=model.config;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if structure:
        ph=structure_phonopy(material,supercell);s=from_phonopy(ph,c.conv_cutoff,c.pair_cutoff)
    else:ph,s=load_material(material,c.conv_cutoff,c.pair_cutoff,labels=False)
    with torch.no_grad():pairs,images=model(s,return_images=True);fc=compact_from_pairs(pairs,s)
    write_FORCE_CONSTANTS(fc.numpy(),filename=str(out/"FORCE_CONSTANTS"),p2s_map=ph.primitive.p2s_map)
    ph.force_constants=fc.numpy();ph.save(filename=str(out/"phonopy.yaml"))
    np.savez(out/"pairs.npz",pair_index=s["pair_index"].numpy(),supercell_j=s["pair_super"].numpy(),
             shortest_cartesian=s["pair_vectors"].numpy(),image_owner=s["image_owner"].numpy(),
             translation=s["pair_shift"].numpy(),compact_pair_fc=pairs.numpy(),image_fc=images.numpy())
    report=diagnostics(pairs,s);(out/"diagnostics.json").write_text(json.dumps(report,indent=2))
    return report


def compare(checkpoint,material,out,nac=False,qfile=None):
    model,_=load_checkpoint(checkpoint);c=model.config;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ph,s=load_material(material,c.conv_cutoff,c.pair_cutoff,nac=nac)
    # User path or illustrative fractional path; not claimed to be universal high symmetry.
    nodes=np.array([[0,0,0],[.5,0,0],[.5,.5,0],[0,0,0]])
    q=np.loadtxt(qfile).reshape(-1,3) if qfile else np.concatenate([np.linspace(a,b,21,endpoint=False) for a,b in zip(nodes[:-1],nodes[1:])]+[nodes[-1:]])
    with torch.no_grad():pred=model(s);fc=compact_from_pairs(pred,s)
    original=copy.deepcopy(ph);predicted=copy.deepcopy(ph)
    predicted.force_constants=fc.numpy()
    original.run_qpoints(q);predicted.run_qpoints(q)
    true=original.qpoints.frequencies;estimated=predicted.qpoints.frequencies
    raw=dynamical_matrix(s["target_compact"],s,torch.tensor(q))
    p_dm=dynamical_matrix(fc,s,torch.tensor(q))
    report={"frequency_mae_thz":float(np.abs(true-estimated).mean()),"nac":bool(nac),
            "nac_parameters_present":ph.nac_params is not None,"target_raw_hermiticity_error":float((raw-raw.mH).abs().max()),
            "predicted_raw_hermiticity_error":float((p_dm-p_dm.mH).abs().max()),
            "path":"user supplied" if qfile else "illustrative Gamma-(.5,0,0)-(.5,.5,0)-Gamma",
            "reference":"provided FORCE_CONSTANTS (provenance determined by input dataset)"}
    if nac and ph.nac_params is None: raise ValueError("NAC requested but BORN/NAC parameters missing")
    # Phonopy may Hermitianize targets; expose this rather than conceal it.
    report["target_requires_hermitian_repair"]=report["target_raw_hermiticity_error"]>1e-7
    np.savez(out/"dispersion.npz",qpoints=q,reference_thz=true,predicted_thz=estimated)
    fig,ax=plt.subplots(figsize=(8,4))
    for branch in range(true.shape[1]):
        ax.plot(true[:,branch],color="black",lw=1,label="Reference FC" if branch==0 else None)
        ax.plot(estimated[:,branch],color="tab:red",ls="--",lw=1,label="Predicted FC" if branch==0 else None)
    ax.axhline(0,color="gray",lw=.5);ax.set(xlabel="q-path sample",ylabel="Frequency (THz)")
    ax.legend();fig.tight_layout();fig.savefig(out/"dispersion.png",dpi=180);plt.close(fig)
    (out/"comparison.json").write_text(json.dumps(report,indent=2));return report
