"""Print the real model shapes and check a small O(3) tensor-product example.

Run from an installed checkout:
    python scripts/explain_tensor_products.py
    python scripts/explain_tensor_products.py --out /tmp/model_walkthrough.json

This uses a fresh, untrained model and a synthetic geometry. It measures no
materials prediction accuracy and writes no training checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

import torch
import yaml
from e3nn import o3

from ifc2.data import from_phonopy
from ifc2.model import IFCModel, ModelConfig, geometry
from ifc2.physics import compact_from_pairs, dynamical_matrix
from ifc2.synthetic import fixture


def describe():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    root = Path(__file__).resolve().parents[1]
    config = ModelConfig(**yaml.safe_load((root / "configs/small.yaml").read_text()))
    model = IFCModel(config).double().eval()
    sample = from_phonopy(fixture(), config.conv_cutoff, config.pair_cutoff)

    # A rank-2 Cartesian tensor consists of a trace, a skew part, and a
    # symmetric-traceless part. This is a concrete 1 + 3 + 5 decomposition.
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    outer = torch.outer(a, b)
    trace_part = torch.trace(outer) * torch.eye(3) / 3
    skew = (outer - outer.T) / 2
    symmetric_traceless = (outer + outer.T) / 2 - trace_part
    torch.testing.assert_close(trace_part + skew + symmetric_traceless, outer)

    # The 1e output of 1o x 1o transforms as an axial vector, not a polar one.
    # FullTensorProduct is a simple teaching example, not the learned FCTP
    # used in the model. Its Clebsch--Gordan normalization need not give the
    # literal Cartesian dot/cross-product coefficients.
    tp = o3.FullTensorProduct("1o", "1o").double()
    pair_coefficients = tp(a, b)
    errors = {}
    rotation = o3.rand_matrix()
    for name, transform in [("rotation", rotation), ("inversion_and_rotation", -rotation)]:
        actual = tp(transform @ a, transform @ b)
        expected = tp.irreps_out.D_from_matrix(transform) @ pair_coefficients
        error = float((actual - expected).abs().max())
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        errors[name] = error

    # Hooks report the exact tensors passing through real modules.
    traces = []
    watched = {
        "embedding", "embed", "pair_init", "pair_init.radial",
        "layers.0.message", "layers.0.message.radial", "layers.0.score",
        "layers.0.att_mix", "layers.0.products.0", "layers.0.gate.gate",
        "layers.0", "pair_updates.0", "pair_context", "pair_products.0",
        "pair_gates.0.gate", "layers.1", "pair_updates.1",
        "pair_products.1", "readout",
    }

    def hook_for(name):
        def record(module, inputs, output):
            traces.append({
                "module": name,
                "inputs": [list(x.shape) for x in inputs if isinstance(x, torch.Tensor)],
                "output": list(output.shape),
            })
        return record

    handles = [
        module.register_forward_hook(hook_for(name))
        for name, module in model.backbone.named_modules() if name in watched
    ]
    try:
        with torch.no_grad():
            pairs, images = model(sample, return_images=True)
            compact = compact_from_pairs(pairs, sample)
            dm = dynamical_matrix(compact, sample, torch.tensor([[0.13, 0.21, 0.31]]))
    finally:
        for handle in handles:
            handle.remove()

    radial, harmonics, envelope = geometry(
        sample["conv_vectors"], config.conv_cutoff, config.num_rbf, config.lmax
    )
    learned_tp = {}
    for name, module in model.backbone.named_modules():
        if isinstance(module, o3.FullyConnectedTensorProduct):
            learned_tp[name] = {
                "in1": str(module.irreps_in1), "in2": str(module.irreps_in2),
                "out": str(module.irreps_out), "weight_numel": module.weight_numel,
                "shared_weights": module.shared_weights,
                "internal_weights": module.internal_weights,
            }

    return {
        "purpose": "Untrained model shape trace and a toy equivariance check; not a DFT benchmark",
        "torch_version": str(torch.__version__),
        "e3nn_version": version("e3nn"),
        "seed": 17,
        "fixture_seed": 0,
        "config": asdict(config),
        "hidden_irreps": str(model.backbone.ir),
        "hidden_dimension": model.backbone.ir.dim,
        "parameters": sum(p.numel() for p in model.parameters()),
        "geometry_shapes": {
            "z": list(sample["z"].shape),
            "conv_index": list(sample["conv_index"].shape),
            "pair_index": list(sample["pair_index"].shape),
            "pair_vectors": list(sample["pair_vectors"].shape),
            "radial": list(radial.shape), "spherical_harmonics": list(harmonics.shape),
            "envelope": list(envelope.shape),
        },
        "output_shapes": {
            "pairs": list(pairs.shape), "images": list(images.shape),
            "compact": list(compact.shape), "dynamical_matrix": list(dm.shape),
        },
        "toy_tensor_product": {
            "a": a.tolist(), "b": b.tolist(), "outer": outer.tolist(),
            "trace_part": trace_part.tolist(), "skew": skew.tolist(),
            "symmetric_traceless": symmetric_traceless.tolist(),
            "e3nn_output_irreps": str(tp.irreps_out),
            "e3nn_output_shape": list(pair_coefficients.shape),
            "equivariance_max_errors": errors,
        },
        "module_trace": traces,
        "learned_tensor_products": learned_tp,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    report = json.dumps(describe(), indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
