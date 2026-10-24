"""Cartesian/irrep transforms and differentiable physical constraints."""
from functools import lru_cache
import torch
from e3nn.io import CartesianTensor


@lru_cache(None)
def _basis():
    # e3nn constructs its CG constants in the default dtype.
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        return CartesianTensor("ij").to_cartesian(torch.eye(9))
    finally:
        torch.set_default_dtype(old)


def decompose(fc):
    """[...,3,3] -> [...,9], orthonormal 0e (1), 1e (3), 2e (5)."""
    return torch.einsum("...ij,kij->...k", fc, _basis().to(fc))


def reconstruct(coeff):
    return torch.einsum("...k,kij->...ij", coeff, _basis().to(coeff))


def scatter(values, index, size):
    return values.new_zeros((size,) + values.shape[1:]).index_add(0, index, values)


def constrain_pairs(raw, sample):
    """[P,3,3] -> reciprocal pairs with divergence-free antisymmetric part.

    After transpose reciprocity, project skew parts with the graph Laplacian.
    This preserves nonzero cycle-space 1e channels, then allows symmetric ASR onsite.
    Geometry-only pseudoinverse is precomputed by the data loader.
    """
    i, j = sample["pair_index"]
    x = (raw + raw[sample["reverse"]].transpose(-1, -2)) / 2
    skew = (x - x.transpose(-1, -2)) / 2
    divergence = scatter(skew, i, len(sample["z"]))
    potential = (sample["laplacian_pinv"].to(raw) @ divergence.reshape(len(sample["z"]), 9)).reshape(-1, 3, 3)
    return x - (potential[i] - potential[j])


def compact_from_pairs(pairs, sample):
    n, ns = len(sample["z"]), sample["n_super"]
    i = sample["pair_index"][0]
    ids = i * ns + sample["pair_super"]
    fc = scatter(pairs, ids, n * ns).reshape(n, ns, 3, 3)
    onsite = -fc.sum(1)
    onsite_ids = torch.arange(n, device=pairs.device) * ns + sample["p2s"]
    return (fc.reshape(n * ns, 3, 3) + scatter(onsite, onsite_ids, n * ns)).reshape(n, ns, 3, 3)


def dynamical_matrix(fc, sample, q):
    """Raw complex D[Q,3N,3N], eV/Angstrom^2/amu. NO Hermitian repair.

    Phonopy atomic-position gauge: exp(+2pi i q dot shortest_fractional).
    Compact FC labels multiply the mean phase across ALL shortest images.
    """
    n, ns = fc.shape[:2]
    owner = sample["dm_owner"]
    angles = 2 * torch.pi * q @ sample["dm_vectors"].T
    phase_images = torch.exp(1j * angles).T
    phase = scatter(phase_images, owner, n * ns).T / sample["dm_count"][None, :]
    i = torch.arange(n, device=fc.device).repeat_interleave(ns)
    j = sample["s2p"].repeat(n)
    mass = torch.sqrt(sample["masses"][i] * sample["masses"][j])
    values = phase[:, :, None, None] * fc.reshape(1, n * ns, 3, 3) / mass[None, :, None, None]
    blocks = values.new_zeros((len(q), n * n, 3, 3)).index_add(1, i * n + j, values)
    return blocks.reshape(-1, n, n, 3, 3).permute(0, 1, 3, 2, 4).reshape(-1, 3*n, 3*n)


def frequencies(dm, factor):
    error = (dm - dm.mH).abs().max()
    if error > 1e-7 * max(1.0, float(dm.abs().max().detach())):
        raise ValueError(f"Non-Hermitian dynamical matrix: {float(error.detach())}")
    eig = torch.linalg.eigvalsh(dm)
    return eig.sign() * eig.abs().sqrt() * factor


def diagnostics(pairs, sample):
    fc = compact_from_pairs(pairs, sample)
    onsite = fc[torch.arange(len(fc), device=fc.device), sample["p2s"]]
    return {"reciprocity_error": float((pairs-pairs[sample["reverse"]].transpose(-1,-2)).abs().max().detach()),
            "asr_error": float(fc.sum(1).abs().max().detach()),
            "onsite_symmetry_error": float((onsite-onsite.transpose(-1,-2)).abs().max().detach())}
