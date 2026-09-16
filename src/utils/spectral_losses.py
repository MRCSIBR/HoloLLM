"""
Módulo: src/utils/spectral_losses.py
Fundamento Teórico:
- Dennis Gabor (1948): Resolución del Problema de Fase en Óptica Holográfica.
- Karl Pribram (1991): Sintonización coherente de desfases en microcircuitos dendríticos.
- David Bohm: Consistencia de fase en la función de onda del Orden Implicado.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.fft as fft

from src.utils.audit import compute_object_code_hash


class HolographicPhaseConsistencyLoss(nn.Module):
    """
    Función de pérdida espectral combinada (MSE Espacial + Amplitud + Coherencia de Fase).
    Elimina las aristas de alta frecuencia y el rizado residual en difusión holográfica.
    """

    def __init__(
        self,
        alpha_mag: float = 0.5,
        beta_phase: float = 1.0,
        eps: float = 1e-7
    ) -> None:
        super().__init__()
        self.alpha_mag = alpha_mag
        self.beta_phase = beta_phase
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Pérdida física directa (Orden Explicado)
        loss_mse = self.mse(pred, target)

        # 2. Proyección al espectro complejo (Orden Implicado)
        pred_fft = fft.rfft(pred, dim=-1)
        target_fft = fft.rfft(target, dim=-1)

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Pérdida de Amplitud (L1)
        loss_mag = torch.mean(torch.abs(pred_mag - target_mag))

        # 3. Pérdida de Alineación de Fase Continua (Distancia Coseno de Fase)
        # Re(z1 * conj(z2)) / (|z1| * |z2| + eps) = cos(phi_1 - phi_2)
        inner_product = torch.real(pred_fft * torch.conj(target_fft))
        norm_product = (pred_mag * target_mag) + self.eps
        cos_phase_diff = inner_product / norm_product

        loss_phase = torch.mean(1.0 - cos_phase_diff)

        # Pérdida total ponderada
        total_loss = loss_mse + (self.alpha_mag * loss_mag) + (self.beta_phase * loss_phase)

        metrics = {
            "loss_mse": loss_mse.item(),
            "loss_mag": loss_mag.item(),
            "loss_phase": loss_phase.item(),
            "total_loss": total_loss.item()
        }

        return total_loss, metrics

    def audit_hashes(self) -> Dict[str, str]:
        return {
            "class_code_hash": compute_object_code_hash(self.__class__)
        }
