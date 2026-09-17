r"""
MÓDULO: src/layers/phase_disentangler.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM)
DESCRIPCIÓN: Regularizador de Ortogonalidad Espectral con escala invariante.
"""

import math
from typing import Tuple, Dict, Any
import hashlib
import inspect
import torch
import torch.nn as nn


class HolographicPhaseDisentangler(nn.Module):
    r"""
    Penaliza la coherencia mutua de fases en el toro unitario |F(K)| = 1.
    La pérdida está normalizada analíticamente en [0, 1].
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        k_complex: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert torch.is_complex(k_complex), f"k_complex debe ser complejo. Recibido: {k_complex.dtype}"
        B, H, S, D_f = k_complex.shape
        if S <= 1:
            zero = torch.tensor(0.0, device=k_complex.device, dtype=torch.float32)
            return zero, {"phase_coherence_max": 0.0, "gram_deviation": 0.0}

        # 1. Proyección al Toro Unitario: z / (|z| + eps)
        mag = torch.clamp(k_complex.abs(), min=self.eps)
        unitary_keys = k_complex / mag

        # 2. Matriz de Gram Espectral (B*H, S, S)
        flat_keys = unitary_keys.reshape(B * H, S, D_f)
        gram_complex = torch.bmm(flat_keys, flat_keys.transpose(1, 2).conj()) / float(D_f)
        gram_real = gram_complex.real

        # 3. Penalización normalizada de crosstalk
        identity = torch.eye(S, device=k_complex.device, dtype=gram_real.dtype).unsqueeze(0)
        off_diagonal = gram_real - identity

        # Factor de normalización para que la norma Frobenius sea invariante a S
        norm_factor = math.sqrt(S * (S - 1)) if S > 1 else 1.0
        crosstalk_loss = (torch.norm(off_diagonal, p="fro", dim=(-2, -1)) / norm_factor).mean()

        with torch.no_grad():
            diag_mask = torch.eye(S, device=k_complex.device, dtype=torch.bool).unsqueeze(0)
            off_diag_values = gram_real.masked_select(~diag_mask)
            max_coherence = off_diag_values.abs().max().item() if off_diag_values.numel() > 0 else 0.0

        return crosstalk_loss, {
            "phase_coherence_max": float(max_coherence),
            "gram_deviation": float(crosstalk_loss.item())
        }

    @staticmethod
    def get_source_hash() -> str:
        source = inspect.getsource(HolographicPhaseDisentangler)
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
