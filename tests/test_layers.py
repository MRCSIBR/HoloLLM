import pytest
import torch
from src.layers.fourier import HolographicFourierInterference
from src.layers.hrr import HolographicReducedRepresentation
from src.utils.audit import compute_state_dict_hash

def test_fourier_layer_forward_backward() -> None:
    batch_size = 8
    features = 32
    modes = 12

    layer = HolographicFourierInterference(features=features, modes=modes, use_bias=True)
    x = torch.randn(batch_size, features, requires_grad=True)
    out = layer(x)

    assert out.shape == (batch_size, features)
    loss = out.sum()
    loss.backward()

    assert layer.spectral_filter.grad is not None
    assert not torch.isnan(layer.spectral_filter.grad).any()
    assert x.grad is not None

def test_hrr_commutativity_and_invariants() -> None:
    dim = 64
    hrr = HolographicReducedRepresentation(dim=dim)
    a = torch.randn(4, dim)
    b = torch.randn(4, dim)

    ab = hrr.bind(a, b)
    ba = hrr.bind(b, a)
    diff = torch.max(torch.abs(ab - ba)).item()
    assert diff < 1e-6, f"Violación de conmutatividad. Error: {diff}"

def test_audit_hash_sensitivity() -> None:
    features = 16
    layer = HolographicFourierInterference(features=features)
    hash_orig = compute_state_dict_hash(layer.state_dict())

    with torch.no_grad():
        layer.spectral_filter[1] += 1e-5

    hash_mod = compute_state_dict_hash(layer.state_dict())
    assert len(hash_orig) == 64
    assert hash_orig != hash_mod
