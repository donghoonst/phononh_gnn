import json
import numpy as np
import torch
from ifc2.synthetic import make_dataset
from ifc2.model import ModelConfig
from ifc2.training import train,load_checkpoint
from ifc2.data import load_material
from ifc2.inference import predict,compare


def test_train_checkpoint_structure_only_prediction(tmp_path):
    root=tmp_path/"data";make_dataset(root,3)
    out=tmp_path/"train"
    metrics=train(root,out,ModelConfig(width=2,layers=1),epochs=2,reciprocal_epochs=1)
    assert metrics["data_kind"]=="synthetic" and np.isfinite(metrics["fc_mae"])
    first,_=load_checkpoint(out/"best.pt");second,_=load_checkpoint(out/"best.pt")
    material=root/"synthetic_000";_,s=load_material(material)
    torch.testing.assert_close(first(s),second(s),atol=0,rtol=0)
    report=compare(out/"best.pt",material,tmp_path/"plot")
    assert not report["target_requires_hermitian_repair"]
    assert (tmp_path/"plot"/"dispersion.png").stat().st_size>1000
    (material/"FORCE_CONSTANTS").unlink()
    predict(out/"best.pt",material,tmp_path/"predicted")
    _,s2=load_material(tmp_path/"predicted")
    assert s2["target_compact"].sum(1).abs().max()<1e-10


def test_nac_born_io(tmp_path):
    from phonopy.file_IO import write_BORN
    from ifc2.synthetic import fixture,attach_synthetic_fc
    from phonopy.file_IO import write_FORCE_CONSTANTS
    ph=attach_synthetic_fc(fixture())
    ph.save(filename=str(tmp_path/"phonopy.yaml"))
    write_FORCE_CONSTANTS(ph.force_constants,filename=str(tmp_path/"FORCE_CONSTANTS"),p2s_map=ph.primitive.p2s_map)
    born=np.array([np.eye(3),-np.eye(3),np.zeros((3,3))])
    write_BORN(ph.primitive,born,np.eye(3)*5,filename=str(tmp_path/"BORN"))
    loaded,_=load_material(tmp_path,nac=True)
    assert loaded.nac_params is not None
    loaded.run_qpoints([[.001,0.,0.]])
    assert np.isfinite(loaded.qpoints.frequencies).all()


def test_data_audit_measures_cutoff_error(tmp_path):
    from ifc2.audit import audit_dataset
    make_dataset(tmp_path/"data",1)
    full=audit_dataset(tmp_path/"data",tmp_path/"full.json")[0]
    cut=audit_dataset(tmp_path/"data",tmp_path/"cut.json",pair_cutoff=3.0)[0]
    assert full["phonopy_dm_max_error"]<1e-12
    assert full["cutoff_dm_mae"]<1e-12
    assert cut["cutoff_dm_mae"]>1e-3
