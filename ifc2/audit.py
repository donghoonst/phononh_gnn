"""Pre-training physical audit of actual labels, including cutoff error."""
from pathlib import Path
import json
import numpy as np
import torch
from .data import load_material
from .physics import dynamical_matrix,compact_from_pairs,constrain_pairs


def audit_dataset(root,out,conv_cutoff=4.5,pair_cutoff=5.5):
    rows=[]
    for path in sorted(Path(root).rglob("phonopy.yaml")):
        if not (path.parent/"FORCE_CONSTANTS").exists():continue
        ph,s=load_material(path.parent,conv_cutoff,pair_cutoff)
        q=torch.tensor([[0.,0.,0.],[.123,-.217,.319]],dtype=torch.float64)
        fc=s["target_compact"];pairs=s["target"]
        dm=dynamical_matrix(fc,s,q)
        ph.run_qpoints(q.numpy(),with_dynamical_matrices=True)
        oracle=ph.qpoints.dynamical_matrices
        rawcut=compact_from_pairs(pairs,s)
        projected=compact_from_pairs(constrain_pairs(pairs,s),s)
        onsite=fc[torch.arange(len(fc)),s["p2s"]]
        rows.append(dict(material=str(path.parent.relative_to(root)),atoms=len(s["z"]),supercell_atoms=s["n_super"],
                         compact_groups=len(pairs),shortest_images=len(s["image_owner"]),max_multiplicity=int(s["dm_count"].max()),
                         raw_asr=float(fc.sum(1).abs().max()),raw_onsite_symmetry=float((onsite-onsite.mT).abs().max()),
                         raw_pair_reciprocity=float((pairs-pairs[s["reverse"]].mT).abs().max()),
                         raw_dm_hermiticity=float((dm-dm.mH).abs().max()),
                         phonopy_dm_max_error=float(np.max(np.abs(dm.numpy()-oracle))),
                         cutoff_dm_mae=float((dynamical_matrix(rawcut,s,q)-dm).abs().mean()),
                         constraint_dm_shift=float((dynamical_matrix(projected-rawcut,s,q)).abs().mean())))
    if not rows:raise ValueError("No phonopy.yaml + FORCE_CONSTANTS materials found")
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rows,indent=2))
    return rows
