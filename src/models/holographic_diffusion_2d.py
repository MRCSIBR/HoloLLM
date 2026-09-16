"""
Módulo: src/models/holographic_diffusion_2d.py
Fundamento Teórico:
- Karl Pribram & Dennis Gabor: Holografía óptica bidimensional y campos receptivos de Fourier 2D.
- David Bohm: Enfundado 2D de campos espaciales en el vacío cuántico.
- Vitaly Vanchurin: Emergencia de filamentos cósmicos por relajación de redes espectrales.
"""

import math
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.fft as fft

from src.utils.audit import compute_object_code_hash, compute_state_dict_hash
from src.models.holographic_diffusion import SinusoidalTimeEmbedding


class SpectralBlock2D(nn.Module):
    """
    Filtro de Interferencia Holográfica 2D.
    Modula la amplitud y fase del espectro espacial bidimensional (kx, ky).
    """

    def __init__(self, img_size: int, time_dim: int) -> None:
        super().__init__()
        self.H = img_size
        self.W = img_size
        self.W_freq = img_size // 2 + 1

        # Placa holográfica compleja base 2D [1, 1, H, W_freq]
        scale = 1.0 / math.sqrt(self.H * self.W_freq)
        real_w = torch.randn(1, 1, self.H, self.W_freq) * scale
        imag_w = torch.randn(1, 1, self.H, self.W_freq) * scale
        imag_w[0, 0, 0, 0] = 0.0  # Conservación de simetría en DC global (0, 0)
        self.spectral_filter = nn.Parameter(torch.complex(real_w, imag_w))

        # Modulación temporal del haz de referencia (Time conditioning)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.H * self.W_freq * 2)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Proyección al espectro 2D del Orden Implicado
        x_fft = fft.rfft2(x, s=(H, W), dim=(-2, -1))

        # 2. Modulación del filtro por el haz temporal
        t_mod = self.time_proj(t_emb).view(B, 1, self.H, self.W_freq, 2)
        mod_complex = torch.complex(t_mod[..., 0], t_mod[..., 1])

        eff_filter = self.spectral_filter * (1.0 + mod_complex)

        # Modulación de ondas 2D
        out_fft = x_fft * eff_filter

        # 3. Retorno al espacio físico 2D (Orden Explicado)
        return fft.irfft2(out_fft, s=(H, W), dim=(-2, -1))


class HoloDiffusion2D(nn.Module):
    """
    Modelo de Difusión Holográfico Bidimensional.
    Genera imágenes a partir de ruido cuántico mediante interferometría espectral 2D pura.
    """

    def __init__(
        self,
        img_size: int = 32,
        time_dim: int = 64,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.block1 = SpectralBlock2D(img_size, time_dim)
        self.block2 = SpectralBlock2D(img_size, time_dim)

        # Refinamiento local en espacio físico
        self.conv_refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = x + self.block1(x, t_emb)
        h = h + self.block2(h, t_emb)
        noise_pred = self.conv_refine(h)
        return noise_pred

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, num_samples: int = 4) -> torch.Tensor:
        self.eval()
        device = self.betas.device
        x = torch.randn(num_samples, 1, self.img_size, self.img_size, device=device)

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

            mean = (1.0 / math.sqrt(alpha)) * (x - (beta / math.sqrt(1.0 - alpha_hat)) * pred_noise)
            x = mean + math.sqrt(beta) * noise

        return x

    def audit_hashes(self) -> Dict[str, str]:
        return {
            "class_code_hash": compute_object_code_hash(self.__class__),
            "state_dict_hash": compute_state_dict_hash(self.state_dict())
        }
