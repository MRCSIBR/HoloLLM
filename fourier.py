"""
Módulo: src/layers/fourier.py
Fundamento Teórico:
- Karl Pribram (1971, 1991): Holonomic Brain Theory. Modulación en el dominio frecuencial.
- David Bohm (1980): Orden Implicado. Conservación del flujo y simetría espectral.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.fft as fft


class HolographicFourierInterference(nn.Module):
    """
    Capa Holográfica basada en modulación de fase y amplitud en el dominio de Fourier.
    Garantiza simetría física en el modo DC (frecuencia cero) para señales reales.
    """

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

        # Inicialización de escala unitaria espectral
        scale = 1.0 / math.sqrt(self.modes)
        real_part = torch.randn(self.modes) * scale
        imag_part = torch.randn(self.modes) * scale
        # Modo DC (k=0) debe ser puramente real en señales reales
        imag_part[0] = 0.0

        self.spectral_filter = nn.Parameter(torch.complex(real_part, imag_part))

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: [batch_size, features] (espacio real / explícito)
        """
        assert x.dim() == 2, f"Se esperaba tensor 2D [batch, features], recibido: {x.shape}"
        assert x.shape[-1] == self.features, (
            f"Dimensión de entrada ({x.shape[-1]}) != configurada ({self.features})"
        )

        batch_size = x.shape[0]

        # 1. Proyección al Orden Implicado (Espectro RFFT)
        x_implicate = fft.rfft(x, n=self.features, dim=-1)

        # 2. Interferencia Espectral
        # Aseguramos que el componente DC del filtro permanezca puramente real
        filt = self.spectral_filter.clone()
        filt[0] = torch.complex(filt[0].real, torch.tensor(0.0, device=filt.device))

        modulated_modes = x_implicate[:, :self.modes] * filt

        # 3. Relleno de altas frecuencias no moduladas si modes < max_modes
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

        # 4. Proyección de vuelta al Orden Explicado (Espacio Físico)
        x_explicate = fft.irfft(x_implicate_mod, n=self.features, dim=-1)

        if self.bias is not None:
            x_explicate = x_explicate + self.bias

        return x_explicate

    def get_filter_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extrae la descomposición en amplitud y fase de la placa de interferencia."""
        amplitude = torch.abs(self.spectral_filter)
        phase = torch.angle(self.spectral_filter)
        return amplitude, phase