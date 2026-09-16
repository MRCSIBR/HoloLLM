import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.fft as fft

class HolographicFourierInterference(nn.Module):
    """
    Capa Holográfica basada en modulación de fase y amplitud en el dominio de Fourier.
    Garantiza simetría física en el modo DC (frecuencia cero) sin operaciones in-place.
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

        # 1. Proyección al Orden Implicado (RFFT)
        x_implicate = fft.rfft(x, n=self.features, dim=-1)

        # 2. Ensamblado funcional out-of-place:
        # Modo DC (k=0) puramente real para señales físicas, modos restantes complejos
        dc_mode = torch.complex(
            self.spectral_filter[0].real,
            torch.zeros((), device=self.spectral_filter.device)
        ).unsqueeze(0)

        if self.modes > 1:
            effective_filter = torch.cat([dc_mode, self.spectral_filter[1:]], dim=0)
        else:
            effective_filter = dc_mode

        # 3. Interferencia espectral
        modulated_modes = x_implicate[:, :self.modes] * effective_filter

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

        # 4. Proyección de vuelta al Orden Explicado
        x_explicate = fft.irfft(x_implicate_mod, n=self.features, dim=-1)

        if self.bias is not None:
            x_explicate = x_explicate + self.bias

        return x_explicate

    def get_filter_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        amplitude = torch.abs(self.spectral_filter)
        phase = torch.angle(self.spectral_filter)
        return amplitude, phase
