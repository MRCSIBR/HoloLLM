#!/usr/bin/env bash
set -e

echo "=== Creando estructura de directorios ==="
mkdir -p src/utils src/layers src/models tests checkpoints

echo "=== Creando pyproject.toml ==="
cat << 'FILE_EOF' > pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "holographic_nn"
version = "0.1.0"
description = "Holographic Neural Networks & HRR in pure PyTorch"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1.0",
    "numpy>=1.24.0,<2.0.0"
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
FILE_EOF

echo "=== Creando requirements.txt ==="
cat << 'FILE_EOF' > requirements.txt
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.1.0
numpy>=1.24.0,<2.0.0
typing-extensions>=4.8.0
pytest>=7.4.0
FILE_EOF

echo "=== Creando paquetes Python (__init__.py) ==="
touch src/__init__.py
touch src/utils/__init__.py
touch src/layers/__init__.py
touch src/models/__init__.py
touch tests/__init__.py

echo "=== Creando src/utils/audit.py ==="
cat << 'FILE_EOF' > src/utils/audit.py
import hashlib
import inspect
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn

def compute_tensor_hash(tensor: torch.Tensor) -> str:
    tensor_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()

def compute_state_dict_hash(state_dict: Dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        hasher.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()

def compute_object_code_hash(obj: Any) -> str:
    try:
        source_code = inspect.getsource(obj)
        return hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    except (TypeError, OSError):
        return "SourceNotAvailable"

def generate_audit_metadata(
    model: nn.Module,
    hyperparams: Dict[str, Any],
    seed: int
) -> Dict[str, Any]:
    return {
        "model_class": model.__class__.__name__,
        "model_code_hash": compute_object_code_hash(model.__class__),
        "weights_hash": compute_state_dict_hash(model.state_dict()),
        "seed": seed,
        "hyperparameters": hyperparams,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }
FILE_EOF

echo "=== Creando src/utils/seed.py ==="
cat << 'FILE_EOF' > src/utils/seed.py
import random
import numpy as np
import torch

def enforce_reproducibility(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
FILE_EOF

echo "=== Creando src/layers/fourier.py ==="
cat << 'FILE_EOF' > src/layers/fourier.py
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.fft as fft

class HolographicFourierInterference(nn.Module):
    def __init__(
        self,
        features: int,
        modes: Optional[int] = None,
        use_bias: bool = True
    ) -> None:
        super().__init__()
        self.features = features
        self.max_modes = features // 2 + 1
        self.modes = min(modes or self.max_modes, self.max_modes)

        scale = 1.0 / math.sqrt(self.modes)
        real_part = torch.randn(self.modes) * scale
        imag_part = torch.randn(self.modes) * scale
        imag_part[0] = 0.0

        self.spectral_filter = nn.Parameter(torch.complex(real_part, imag_part))

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, f"Se esperaba tensor 2D [batch, features], recibido: {x.shape}"
        assert x.shape[-1] == self.features, (
            f"Dimensión de entrada ({x.shape[-1]}) != configurada ({self.features})"
        )

        batch_size = x.shape[0]
        x_implicate = fft.rfft(x, n=self.features, dim=-1)

        filt = self.spectral_filter.clone()
        filt[0] = torch.complex(filt[0].real, torch.tensor(0.0, device=filt.device))

        modulated_modes = x_implicate[:, :self.modes] * filt

        if self.modes < self.max_modes:
            zeros = torch.zeros(
                batch_size,
                self.max_modes - self.modes,
                dtype=torch.cfloat,
                device=x.device
            )
            x_implicate_mod = torch.cat([modulated_modes, zeros], dim=-1)
        else:
            x_implicate_mod = modulated_modes

        x_explicate = fft.irfft(x_implicate_mod, n=self.features, dim=-1)

        if self.bias is not None:
            x_explicate = x_explicate + self.bias

        return x_explicate

    def get_filter_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        amplitude = torch.abs(self.spectral_filter)
        phase = torch.angle(self.spectral_filter)
        return amplitude, phase
FILE_EOF

echo "=== Creando src/layers/hrr.py ==="
cat << 'FILE_EOF' > src/layers/hrr.py
import torch
import torch.nn as nn
import torch.fft as fft

class HolographicReducedRepresentation(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._validate_operands(a, b)
        a_fft = fft.fft(a, n=self.dim, dim=-1)
        b_fft = fft.fft(b, n=self.dim, dim=-1)
        bound_fft = a_fft * b_fft
        return fft.ifft(bound_fft, n=self.dim, dim=-1).real

    def unbind(self, trace: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        self._validate_operands(trace, key)
        trace_fft = fft.fft(trace, n=self.dim, dim=-1)
        key_fft = fft.fft(key, n=self.dim, dim=-1)
        unbound_fft = trace_fft * torch.conj(key_fft)
        return fft.ifft(unbound_fft, n=self.dim, dim=-1).real

    def normalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norms + eps)

    def _validate_operands(self, a: torch.Tensor, b: torch.Tensor) -> None:
        assert a.shape[-1] == self.dim and b.shape[-1] == self.dim, (
            f"Operandos deben coincidir con dim HRR {self.dim}. Recibidos: {a.shape}, {b.shape}"
        )
        assert a.shape == b.shape, f"Los shapes deben ser idénticos. Recibidos: {a.shape} vs {b.shape}"
FILE_EOF

echo "=== Creando src/models/holographic_mlp.py ==="
cat << 'FILE_EOF' > src/models/holographic_mlp.py
from typing import Dict
import torch
import torch.nn as nn
from src.layers.fourier import HolographicFourierInterference
from src.utils.audit import compute_object_code_hash, compute_state_dict_hash

class HolographicClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        modes: int = 32
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.modes = modes

        self.holo1 = HolographicFourierInterference(features=input_dim, modes=modes)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()

        self.holo2 = HolographicFourierInterference(features=hidden_dim, modes=modes)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, f"Shape de entrada incorrecto: {x.shape}"
        h1 = self.holo1(x)
        h1 = self.act(self.proj(h1))
        h2 = self.holo2(h1)
        h2 = self.norm(h2)
        return self.head(h2)

    def audit_hashes(self) -> Dict[str, str]:
        return {
            "class_code_hash": compute_object_code_hash(self.__class__),
            "state_dict_hash": compute_state_dict_hash(self.state_dict())
        }
FILE_EOF

echo "=== Creando tests/test_layers.py ==="
cat << 'FILE_EOF' > tests/test_layers.py
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
FILE_EOF

echo "=== Creando train.py ==="
cat << 'FILE_EOF' > train.py
import os
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.holographic_mlp import HolographicClassifier
from src.layers.hrr import HolographicReducedRepresentation
from src.utils.seed import enforce_reproducibility
from src.utils.audit import generate_audit_metadata

def generate_harmonic_dataset(
    num_samples: int = 1024,
    dim: int = 64,
    num_classes: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(0, 1, dim)
    X = torch.zeros(num_samples, dim)
    y = torch.randint(0, num_classes, (num_samples,))

    for i in range(num_samples):
        cls = y[i].item()
        base_freq = (cls + 1) * 2.5
        signal = torch.sin(2 * np.pi * base_freq * t) + 0.4 * torch.cos(4 * np.pi * base_freq * t)
        noise = torch.randn(dim) * 0.08
        X[i] = signal + noise

    return X, y

def run_hrr_algebra_check(batch_eval: int = 32, dim: int = 128) -> None:
    print("\n--- [Audit] Verificando Álgebra Holográfica Reducida (Plate HRR) ---")
    hrr = HolographicReducedRepresentation(dim=dim)
    v_keys = hrr.normalize(torch.randn(batch_eval, dim))
    v_values = hrr.normalize(torch.randn(batch_eval, dim))

    memory_traces = hrr.bind(v_keys, v_values)
    retrieved_values = hrr.unbind(memory_traces, v_keys)

    cos_sims = torch.cosine_similarity(v_values, retrieved_values, dim=-1)
    mean_sim = cos_sims.mean().item()
    std_sim = cos_sims.std().item()

    print(f"Similitud Coseno Media: {mean_sim:.4f} (± {std_sim:.4f}) | Teórico: ~0.707")
    assert mean_sim > 0.60, f"Fidelidad asociativa baja: {mean_sim}"
    print("✔ HRR: Operaciones circulares validadas.")

def main() -> None:
    SEED = 42
    enforce_reproducibility(SEED)
    run_hrr_algebra_check()

    HYPERPARAMS: Dict[str, Any] = {
        "input_dim": 64,
        "hidden_dim": 64,
        "num_classes": 4,
        "modes": 24,
        "learning_rate": 3e-3,
        "batch_size": 32,
        "epochs": 12
    }

    print("\n--- [Audit] Inicializando Red Holográfica ---")
    model = HolographicClassifier(
        input_dim=HYPERPARAMS["input_dim"],
        hidden_dim=HYPERPARAMS["hidden_dim"],
        num_classes=HYPERPARAMS["num_classes"],
        modes=HYPERPARAMS["modes"]
    )

    initial_audit = model.audit_hashes()
    print(f"Model Code SHA-256:  {initial_audit['class_code_hash']}")
    print(f"Initial Weights SHA: {initial_audit['state_dict_hash']}")

    X, y = generate_harmonic_dataset(
        num_samples=1024,
        dim=HYPERPARAMS["input_dim"],
        num_classes=HYPERPARAMS["num_classes"]
    )
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=HYPERPARAMS["batch_size"], shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=HYPERPARAMS["learning_rate"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("\n--- [Train] Optimizando espectro de interferencia ---")
    for epoch in range(1, HYPERPARAMS["epochs"] + 1):
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        if epoch % 3 == 0 or epoch == HYPERPARAMS["epochs"]:
            acc = correct / total
            print(f"Epoch [{epoch:02d}/{HYPERPARAMS['epochs']:02d}] | Loss: {total_loss/total:.4f} | Acc: {acc * 100:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/holographic_mlp_v1.pt"
    audit_metadata = generate_audit_metadata(model=model, hyperparams=HYPERPARAMS, seed=SEED)
    torch.save({"state_dict": model.state_dict(), "audit": audit_metadata}, checkpoint_path)
    print(f"\n✔ Checkpoint guardado en: {checkpoint_path}")
    print(f"Final Weights SHA-256: {audit_metadata['weights_hash']}")

if __name__ == "__main__":
    main()
FILE_EOF

echo "=== Estructura del proyecto configurada con éxito ==="
