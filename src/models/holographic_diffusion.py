"""
Módulo: src/models/holographic_diffusion.py
Fundamento Teórico:
- David Bohm: Enfundado del Orden Explicado en el vacío estocástico (Forward)
  y despliegue mediante interferencia de fase coherente (Reverse).
- Karl Pribram: Reconstrucción holonómica a partir de espectros de Fourier.
- Vitaly Vanchurin: Dinámica de difusión como relajación termodinámica de una red neuronal.
"""

import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.fft as fft

from src.utils.audit import compute_object_code_hash, compute_state_dict_hash


class SinusoidalTimeEmbedding(nn.Module):
    """Proyecta el paso de difusión temporal t en un vector armónico continuo."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class SpectralDenoisingBlock(nn.Module):
    """
    Bloque de Interferencia Espectral condicionado por tiempo.
    Modula la amplitud y fase de las ondas de ruido en el Orden Implicado.
    """
    def __init__(self, signal_len: int, time_dim: int, modes: Optional[int] = None) -> None:
        super().__init__()
        self.signal_len = signal_len
        self.max_modes = signal_len // 2 + 1
        self.modes = min(modes or self.max_modes, self.max_modes)

        # Placa holográfica base compleja
        scale = 1.0 / math.sqrt(self.modes)
        real_p = torch.randn(self.modes) * scale
        imag_p = torch.randn(self.modes) * scale
        imag_p[0] = 0.0  # Conservación de simetría en DC
        self.base_spectral_filter = nn.Parameter(torch.complex(real_p, imag_p))

        # Modulación del haz de referencia guiado por el tiempo t
        self.time_mod_real = nn.Sequential(
            nn.Linear(time_dim, self.modes),
            nn.SiLU(),
            nn.Linear(self.modes, self.modes)
        )
        self.time_mod_imag = nn.Sequential(
            nn.Linear(time_dim, self.modes),
            nn.SiLU(),
            nn.Linear(self.modes, self.modes)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        # 1. Proyección al espectro complejo
        x_fft = fft.rfft(x, n=self.signal_len, dim=-1)

        # 2. Modulación del filtro por el tiempo de difusión
        mod_r = self.time_mod_real(t_emb)
        mod_i = self.time_mod_imag(t_emb)
        time_filter = torch.complex(mod_r, mod_i)

        # Interferencia constructiva/destructiva
        effective_filter = self.base_spectral_filter.unsqueeze(0) * (1.0 + time_filter)

        # Conservación física de DC (la frecuencia cero debe ser real)
        dc_real = effective_filter[:, 0].real
        effective_filter = torch.cat([
            torch.complex(dc_real, torch.zeros_like(dc_real)).unsqueeze(1),
            effective_filter[:, 1:]
        ], dim=1)

        modulated_modes = x_fft[:, :self.modes] * effective_filter

        if self.modes < self.max_modes:
            zeros = torch.zeros(B, self.max_modes - self.modes, dtype=torch.cfloat, device=x.device)
            x_fft_full = torch.cat([modulated_modes, zeros], dim=-1)
        else:
            x_fft_full = modulated_modes

        # 3. Retorno al espacio físico (Orden Explicado)
        return fft.irfft(x_fft_full, n=self.signal_len, dim=-1)


class HolographicDiffusion(nn.Module):
    """
    Modelo de Difusión Holográfico Completo (DDPM en Orden Implicado).
    Aprende a des-enfundar ondas físicas coherentes a partir de ruido blanco cuántico.
    """
    def __init__(
        self,
        signal_len: int = 64,
        time_dim: int = 64,
        modes: int = 32,
        timesteps: int = 150,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ) -> None:
        super().__init__()
        self.signal_len = signal_len
        self.timesteps = timesteps

        # Programación de ruido (Variance Preserving Schedule)
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Red de Inferencia Holográfica (Denoising Score Model)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.spectral_block1 = SpectralDenoisingBlock(signal_len, time_dim, modes)
        self.spectral_block2 = SpectralDenoisingBlock(signal_len, time_dim, modes)
        self.spatial_proj = nn.Sequential(
            nn.Linear(signal_len, signal_len),
            nn.GELU(),
            nn.Linear(signal_len, signal_len)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predice el ruido epsilon añadido en el paso t."""
        t_emb = self.time_embed(t)
        # Paso espectral 1
        h = x + self.spectral_block1(x, t_emb)
        # Paso espectral 2
        h = h + self.spectral_block2(h, t_emb)
        # Refinamiento y predicción de ruido
        pred_noise = self.spatial_proj(h)
        return pred_noise

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Proceso Forward: enfunda la señal física en ruido estocástico."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, num_samples: int = 4) -> torch.Tensor:
        """
        Proceso Reverse: despliega señales físicas puras a partir de ruido blanco.
        """
        self.eval()
        device = self.betas.device
        # Comienza en el caos cuántico del Orden Implicado
        x = torch.randn(num_samples, self.signal_len, device=device)

        for t_step in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), t_step, device=device, dtype=torch.long)
            pred_noise = self.forward(x, t)

            alpha = self.alphas[t_step]
            alpha_hat = self.alphas_cumprod[t_step]
            beta = self.betas[t_step]

            if t_step > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Ecuación de Langevin / Muestreo DDPM
            mean = (1.0 / math.sqrt(alpha)) * (x - (beta / math.sqrt(1.0 - alpha_hat)) * pred_noise)
            sigma = math.sqrt(beta)
            x = mean + sigma * noise

        return x

    def audit_hashes(self) -> Dict[str, str]:
        return {
            "class_code_hash": compute_object_code_hash(self.__class__),
            "state_dict_hash": compute_state_dict_hash(self.state_dict())
        }
