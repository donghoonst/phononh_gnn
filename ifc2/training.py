"""Single-structure batches, material split, validation-selected checkpoint."""
from dataclasses import asdict
from pathlib import Path
import csv,json,time
import numpy as np
import torch
from e3nn import o3
from .model import ModelConfig,IFCModel
from .data import load_material,material_split,to_device
from .loss import train_scales,fc_loss,reciprocal_loss,SLICES
from .physics import decompose,constrain_pairs,compact_from_pairs,dynamical_matrix,frequencies,diagnostics


def load_dataset(root,config):
    paths=sorted(p.parent for p in Path(root).rglob("phonopy.yaml") if (p.parent/"FORCE_CONSTANTS").exists())
    samples={}
    for p in paths:
        _,s=load_material(p,config.conv_cutoff,config.pair_cutoff)
        # Relative paths allow same leaf folder names in different datasets.
        s["material_id"]=str(p.relative_to(root));samples[s["material_id"]]=s
    return samples


@torch.no_grad()
def evaluate(model,samples,scales):
    model.eval();rows=[]
    for s in samples:
        p=model(s);fc=compact_from_pairs(p,s)
        coeff=decompose(p-s["target"])
        q=p.new_tensor([[0.,0.,0.],[.13,.21,.31],[.37,-.19,.23]])
        pd=dynamical_matrix(fc,s,q);td=dynamical_matrix(s["target_compact"],s,q)
        cut=compact_from_pairs(constrain_pairs(s["target"],s),s)
        cd=dynamical_matrix(cut,s,q)
        rawcut=compact_from_pairs(s["target"],s)
        rawcd=dynamical_matrix(rawcut,s,q)
        row={"fc_mae":float((p-s["target"]).abs().mean()),"scaled_loss":float(fc_loss(p,s["target"],scales)),
             **{f"irrep_{k}_mae":float(coeff[:,sl].abs().mean()) for k,sl in enumerate(SLICES)},
             "dm_mae":float((pd-td).abs().mean()),"dm_model_mae":float((pd-cd).abs().mean()),
             "cutoff_dm_mae":float((rawcd-td).abs().mean()),
             "constraint_dm_shift":float((cd-rawcd).abs().mean()),
             "target_constraint_fc_shift":float((constrain_pairs(s["target"],s)-s["target"]).abs().mean()),
             "hermiticity_error":float((pd-pd.mH).abs().max()),
             "target_hermiticity_error":float((td-td.mH).abs().max()),**diagnostics(p,s)}
        pf=frequencies(pd,s["frequency_factor"])
        row["imaginary_modes"]=int((pf < -0.1).sum())
        if row["target_hermiticity_error"]<1e-7:
            tf=frequencies(td,s["frequency_factor"])
            row["frequency_mae_thz"]=float((pf-tf).abs().mean())
            row["target_imaginary_modes"]=int((tf < -0.1).sum())
            row["imaginary_mode_mismatch"]=float(((pf<-.1)!=(tf<-.1)).to(p).mean())
        else:
            row["frequency_mae_thz"]=None;row["target_imaginary_modes"]=None;row["imaginary_mode_mismatch"]=None
        qrot=o3.rand_matrix(dtype=p.dtype,device=p.device);rot=dict(s)
        for key in ["conv_vectors","pair_vectors","cell","positions"]:rot[key]=s[key]@qrot.T
        row["rotation_error"]=float((model(rot)-qrot@p@qrot.T).abs().max())
        rows.append(row)
    return {k:float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None for k in rows[0]}


def train(root,out,config,epochs=10,seed=17,lr=.003,device="cpu",reciprocal_epochs=0,lambda_d=.1,lambda_eigen=.01,split=None):
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.manual_seed(seed)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cpu_samples=load_dataset(root,config)
    split=split or material_split(list(cpu_samples),seed)
    flattened=[x for names in split.values() for x in names]
    if set(split)!={"train","validation","test"} or len(flattened)!=len(set(flattened)) or set(flattened)!=set(cpu_samples):
        raise ValueError("Split must partition all materials without overlap")
    if any(not split[k] for k in split): raise ValueError("Empty split")
    samples={k:to_device(v,device) for k,v in cpu_samples.items()}
    groups={k:[samples[x] for x in names] for k,names in split.items()}
    scales=train_scales(groups["train"])
    model=IFCModel(config).double().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    if device.startswith("cuda"):torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();best=float("inf");history=[]
    for epoch in range(epochs+reciprocal_epochs):
        model.train();losses=[]
        for s in groups["train"]:
            optimizer.zero_grad();pred=model(s);loss=fc_loss(pred,s["target"],scales)
            if epoch>=epochs:
                q=pred.new_tensor([[.13,.21,.31],[.37,-.19,.23]])
                dl,el=reciprocal_loss(pred,s,q);loss=loss+lambda_d*dl+lambda_eigen*el
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():val=sum(float(fc_loss(model(s),s["target"],scales)) for s in groups["validation"])/len(groups["validation"])
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=val,stage="real" if epoch<epochs else "reciprocal"))
        if val<best:
            best=val
            torch.save(dict(config=asdict(config),state_dict=model.state_dict(),scales=scales.cpu(),split=split,seed=seed,epoch=epoch),out/"best.pt")
    seconds=time.perf_counter()-started
    model,_=load_checkpoint(out/"best.pt",device)
    metrics=evaluate(model,groups["test"],scales)
    metrics.update(parameters=sum(p.numel() for p in model.parameters()),training_seconds=seconds,
                   peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
                   device=device,epochs=epochs,reciprocal_epochs=reciprocal_epochs,seed=seed,
                   data_kind="synthetic" if (Path(root)/"PROVENANCE.txt").exists() and "SYNTHETIC" in (Path(root)/"PROVENANCE.txt").read_text() else "user_supplied",
                   train_materials=len(groups["train"]),validation_materials=len(groups["validation"]),test_materials=len(groups["test"]))
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    (out/"history.json").write_text(json.dumps(history,indent=2))
    (out/"split.json").write_text(json.dumps(split,indent=2))
    (out/"config.json").write_text(json.dumps(asdict(config),indent=2))
    return metrics


def load_checkpoint(path,device="cpu"):
    torch.set_default_dtype(torch.float64)
    ckpt=torch.load(path,map_location=device,weights_only=True)
    m=IFCModel(ModelConfig(**ckpt["config"])).double().to(device)
    m.load_state_dict(ckpt["state_dict"]);m.eval()
    return m,ckpt


def ablations(root,out,epochs=5,width=3,seed=17):
    rows=[];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    specs={"A":(1,False,False,False),"B":(2,False,False,False),"C":(1,True,False,False),
           "D":(2,True,False,False),"E":(2,True,True,False),"F":(2,True,True,True)}
    for name,(corr,attention,fusion,residual) in specs.items():
        c=ModelConfig(width=width,correlation=corr,attention=attention,enhanced_fusion=fusion,physics_residual=residual,independent_heads=True)
        metrics=train(root,out/name,c,epochs=epochs,seed=seed)
        rows.append(dict(model=name,**metrics));print(name,json.dumps(metrics),flush=True)
    # Separate comparison, so parameter sharing does not confound D -> E.
    c=ModelConfig(width=width,correlation=2,attention=True,enhanced_fusion=True,independent_heads=False)
    rows.append(dict(model="E_shared",**train(root,out/"E_shared",c,epochs=epochs,seed=seed)))
    c=ModelConfig(width=width,correlation=2,attention=True,enhanced_fusion=True,independent_heads=True,aggregation="attention_only")
    rows.append(dict(model="E_attention_only",**train(root,out/"E_attention_only",c,epochs=epochs,seed=seed)))
    (out/"results.json").write_text(json.dumps(rows,indent=2))
    with (out/"results.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows
