"""
Módulo: src/layers/adscft.py
Fundamento Teórico:
- Juan Maldacena (1997): The Large N Limit of Superconformal Field Theories and Supergravity.
- Edward Witten (1998): Anti-de Sitter Space and Holography (Boundary-to-Bulk Propagators).
- Vitaly Vanchurin (2020): The World as a Neural Network (Spacetime emergence from learning).
"""

import math
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.fft as fft

from src.utils.audit import compute_object_code_hash, compute_state_dict_hash


class AdSCFTBulkBoundaryLayer(nn.Module):
    """
    Capa de Dualidad Holográfica AdS/CFT.
    
    Toma una secuencia de entrada (frontera z=0) y genera una dimensión radial extra emergente z,
    simulando la propagación en un espacio Anti-de Sitter curvo con decaimiento conforme de Witten.
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        bulk_depth_slices: int = 8,
        z_max: float = 2.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bulk_slices = bulk_depth_slices
        self.z_max = z_max

        # Coordenadas discretas de la dimensión radial z (profundidad en el Bulk AdS)
        # z va desde la frontera UV (cerca de 0) hacia el interior IR (z_max)
        z_coords = torch.linspace(0.1, z_max, bulk_depth_slices)
        self.register_buffer("z_coords", z_coords)

        # Factor de curvatura AdS: g_xx ~ 1 / z^2 (métrica de Poincaré)
        ads_metric_weights = 1.0 / (z_coords ** 2)
        ads_metric_weights = ads_metric_weights / ads_metric_weights.sum()
        self.register_buffer("metric_weights", ads_metric_weights.view(1, bulk_depth_slices, 1, 1))

        # Parámetro conforme entrenable (dimensión conforme del operador dual)
        self.conformal_scale = nn.Parameter(torch.ones(dim))

        # Dinámica no lineal en el interior del Bulk (interacción de campos en volumen)
        self.bulk_interaction = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # Proyección de frontera
        self.out_proj = nn.Linear(dim, dim)

    def boundary_to_bulk(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagador de Witten: Proyecta la frontera 1D [B, T, D]
        hacia el volumen AdS 2D continuo [B, Bulk_Z, T, D].
        """
        B, T, D = x.shape
        # FFT a lo largo de la dimensión espacial/temporal de la frontera
        x_fft = fft.rfft(x, n=T, dim=1)  # [B, T//2 + 1, D]
        num_modes = x_fft.shape[1]

        # Vector de frecuencias de onda |k|
        k_vals = torch.arange(num_modes, device=x.device).float().view(1, 1, num_modes, 1)

        # Propagador conforme: exp(- |k| * z * scale)
        z_expanded = self.z_coords.view(1, self.bulk_slices, 1, 1)
        scale_expanded = torch.clamp(self.conformal_scale.view(1, 1, 1, D), min=0.1)

        damping = torch.exp(- k_vals * z_expanded * (scale_expanded * 0.1))  # [1, Z, modes, D]

        # Aplicar propagador a cada modo
        x_fft_expanded = x_fft.unsqueeze(1)  # [B, 1, modes, D]
        bulk_fft = x_fft_expanded * damping   # [B, Z, modes, D]

        # Retornar al espacio físico en cada rebanada radial z del bulk
        bulk_field = fft.irfft(bulk_fft, n=T, dim=2)  # [B, Z, T, D]
        return bulk_field

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Señal en la frontera [Batch, SeqLen, Dim]
        Returns:
            out_boundary: Señal reconstituida en la frontera tras resonar en el Bulk [Batch, SeqLen, Dim]
            bulk_field: El campo gravitacional completo en el volumen [Batch, Bulk_Z, SeqLen, Dim]
        """
        assert x.dim() == 3, f"Se esperaba tensor [B, T, D], recibido: {x.shape}"
        assert x.shape[-1] == self.dim, f"Dimensión {x.shape[-1]} != {self.dim}"

        # 1. Propagación Holográfica Frontera -> Bulk (Witten)
        bulk_field = self.boundary_to_bulk(x)  # [B, Z, T, D]

        # 2. Dinámica de interacción gravitacional en el Bulk curvo
        # El campo interactúa ponderado por la métrica de Poincaré (1 / z^2)
        curved_field = self.bulk_interaction(bulk_field) * self.metric_weights

        # 3. Proyección Holográfica Bulk -> Frontera (Integración de volumen a superficie)
        # La frontera recolecta la acción total integrada sobre la coordenada radial z
        boundary_integral = torch.sum(curved_field, dim=1)  # [B, T, D]

        out = x + self.out_proj(boundary_integral)
        return out, bulk_field

    def audit_hashes(self) -> Dict[str, str]:
        return {
            "class_code_hash": compute_object_code_hash(self.__class__),
            "state_dict_hash": compute_state_dict_hash(self.state_dict())
        }
