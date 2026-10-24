import copy
import numpy as np
import pytest
import torch
from e3nn import o3
from ifc2.physics import decompose,reconstruct,constrain_pairs,compact_from_pairs,dynamical_matrix,diagnostics
from ifc2.data import from_phonopy,graph,load_material,material_split
from ifc2.model import IFCModel,ModelConfig,Interaction,geometry
from ifc2.synthetic import fixture,attach_synthetic_fc,make_dataset


@pytest.fixture(scope="session",autouse=True)
def numeric_defaults():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)


@pytest.fixture
def sample():
    return from_phonopy(attach_synthetic_fc(fixture()))


def test_decomposition():
    x=torch.randn(12,3,3);c=decompose(x)
    torch.testing.assert_close(reconstruct(c),x,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(c.square().sum(-1),x.square().sum((-1,-2)))
    assert decompose((x+x.mT)/2)[:,1:4].abs().max()<1e-12


def test_constraints_preserve_nonzero_skew_and_gradients(sample):
    raw=torch.randn_like(sample["target"],requires_grad=True)
    pairs=constrain_pairs(raw,sample)
    assert max(diagnostics(pairs,sample).values())<1e-12
    assert decompose(pairs)[:,1:4].abs().max()>1e-3
    torch.testing.assert_close(constrain_pairs(pairs,sample),pairs)
    pairs.square().sum().backward();assert torch.isfinite(raw.grad).all()


@pytest.mark.parametrize("supercell",[(1,1,1),(2,2,2)])
def test_raw_dm_matches_phonopy(supercell):
    ph=attach_synthetic_fc(fixture(supercell=supercell));s=from_phonopy(ph)
    q=torch.tensor([[0.,0.,0.],[0.123,-0.217,0.319],[-0.34,0.15,0.23]])
    dm=dynamical_matrix(s["target_compact"],s,q)
    assert (dm-dm.mH).abs().max()<1e-12
    assert dm.imag.abs().max()>1e-5  # Discriminates phase sign, unlike centrosymmetric one-atom fixtures.
    ph.run_qpoints(q.numpy(),with_dynamical_matrices=True)
    np.testing.assert_allclose(dm.numpy(),ph.qpoints.dynamical_matrices,atol=2e-12,rtol=2e-12)
    fc=s["target_compact"].clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda x:dynamical_matrix(x,s,q),fc,fast_mode=True)


def test_onsite_counterexample_is_detected(sample):
    raw=torch.randn_like(sample["target"])
    reciprocal=(raw+raw[sample["reverse"]].mT)/2
    fc=compact_from_pairs(reciprocal,sample)
    assert fc.sum(1).abs().max()<1e-12
    dm=dynamical_matrix(fc,sample,torch.tensor([[.17,.23,.31]]))
    assert (dm-dm.mH).abs().max()>1e-3


@pytest.mark.parametrize("inversion",[False,True])
def test_model_o3_equivariance(sample,inversion):
    torch.manual_seed(9);m=IFCModel(ModelConfig(width=3,physics_residual=True)).double()
    q=o3.rand_matrix();q=-q if inversion else q
    rotated=dict(sample)
    for key in ["conv_vectors","pair_vectors","positions","cell"]:rotated[key]=sample[key]@q.T
    with torch.no_grad(): a=m(sample);b=m(rotated)
    torch.testing.assert_close(b,q@a@q.T,atol=2e-9,rtol=2e-9)


def test_periodic_translation_and_permutation(sample):
    s=sample;torch.manual_seed(21);m=IFCModel(ModelConfig(width=3)).double()
    translated=dict(s)
    translation=torch.tensor([[2.,-1.,0.],[-2.,3.,1.],[1.,0.,-3.]])@s["cell"]
    ci,cv,cs=graph(s["z"],s["positions"]+translation+0.371,s["cell"],4.5)
    translated.update(conv_index=ci,conv_vectors=cv,conv_shift=cs)
    with torch.no_grad(): torch.testing.assert_close(m(s),m(translated),atol=1e-10,rtol=1e-10)
    perm=torch.tensor([2,0,1]);inv=torch.argsort(perm);p=dict(s)
    p["z"]=s["z"][perm];p["pair_index"]=inv[s["pair_index"]];p["conv_index"]=inv[s["conv_index"]]
    p["laplacian_pinv"]=s["laplacian_pinv"][perm][:,perm]
    with torch.no_grad():torch.testing.assert_close(m(s),m(p),atol=1e-10,rtol=1e-10)


def test_cutoff_excludes_attention_edges(sample):
    torch.manual_seed(1);m=IFCModel(ModelConfig(width=3)).double();s=dict(sample)
    s["conv_index"]=torch.cat([s["conv_index"],torch.tensor([[0],[1]])],1)
    s["conv_vectors"]=torch.cat([s["conv_vectors"],torch.tensor([[9.,0.,0.]])])
    with torch.no_grad():torch.testing.assert_close(m(sample),m(s),atol=1e-12,rtol=1e-12)


def test_compact_io_and_all_shortest_images(tmp_path):
    make_dataset(tmp_path,3);ph,s=load_material(tmp_path/"synthetic_000")
    assert s["dm_count"].max()>1
    assert len(s["pair_vectors"])>s["pair_index"].shape[1]
    torch.testing.assert_close(compact_from_pairs(s["target"],s),s["target_compact"],atol=1e-12,rtol=1e-12)
    ph2,s2=load_material(tmp_path/"synthetic_000",labels=False)
    assert "target" not in s2
    for key in ["pair_index","pair_vectors","conv_vectors"]:torch.testing.assert_close(s[key],s2[key])
    split=material_split(["a","b","c","d","e","f"])
    assert not set(split["train"]) & set(split["test"])


@pytest.mark.parametrize("independent",[False,True])
def test_cpu_forward_backward(sample,independent):
    m=IFCModel(ModelConfig(width=2,independent_heads=independent,correlation=3)).double()
    optimizer=torch.optim.Adam(m.parameters(),lr=1e-3)
    output=m(sample);loss=(output-sample["target"]).square().mean();loss.backward()
    grads=[p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads)>10 and all(torch.isfinite(x).all() for x in grads)
    assert sum(x.abs().sum() for x in grads)>0
    optimizer.step();assert torch.isfinite(m(sample)).all()


def test_image_outputs_sum_and_reciprocate(sample):
    from ifc2.physics import scatter
    m=IFCModel(ModelConfig(width=2,enhanced_fusion=True)).double()
    pairs,images=m(sample,return_images=True)
    torch.testing.assert_close(scatter(images,sample["image_owner"],len(pairs)),pairs)
    torch.testing.assert_close(images,images[sample["image_reverse"]].mT)


def test_conventional_primitive_mapping_and_full_fc(tmp_path):
    from ase.build import bulk
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.file_IO import write_FORCE_CONSTANTS
    from phonopy.harmonic.force_constants import compact_fc_to_full_fc
    a=bulk("NaCl","rocksalt",a=5.6,cubic=True)
    ph=Phonopy(PhonopyAtoms(numbers=a.numbers,cell=a.cell.array,scaled_positions=a.get_scaled_positions()),
               supercell_matrix=np.diag([2,1,1]),primitive_matrix="auto")
    ph=attach_synthetic_fc(ph);s=from_phonopy(ph)
    assert len(ph.primitive)<len(ph.unitcell)
    full=compact_fc_to_full_fc(ph.primitive,ph.force_constants)
    ph.save(filename=str(tmp_path/"phonopy.yaml"));write_FORCE_CONSTANTS(full,filename=str(tmp_path/"FORCE_CONSTANTS"))
    _,loaded=load_material(tmp_path)
    torch.testing.assert_close(loaded["target_compact"],s["target_compact"],atol=1e-12,rtol=1e-12)
    q=torch.tensor([[.13,.22,-.31]])
    dm=dynamical_matrix(s["target_compact"],s,q)
    ph.run_qpoints(q.numpy(),with_dynamical_matrices=True)
    np.testing.assert_allclose(dm.numpy(),ph.qpoints.dynamical_matrices,atol=2e-12,rtol=2e-12)
