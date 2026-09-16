"""
Script: train_diffusion_2d_hq.py
Propósito: Entrenamiento de Alta Definición para HoloDiffusion2D incorporando:
1. Espacio latente multicanal para invarianza rotacional.
2. Pérdida de Coherencia de Fase 2D (Gabor / Pribram).
3. 60 épocas de optimización armónica.
"""

import os
import math
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.fft as fft
import torch.optim as optim

from src.models.holographic_diffusion import SinusoidalTimeEmbedding
from src.utils.seed import enforce_reproducibility
from src.utils.audit import generate_audit_metadata
from train_diffusion_2d import generate_cosmic_web_dataset


class MultiChannelSpectralBlock2D(nn.Module):
    """Filtro holográfico multicanal: cada canal aprende una orientación angular distinta."""
    def __init__(self, channels: int, img_size: int, time_dim: int) -> None:
        super().__init__()
        self.C = channels
        self.H = img_size
        self.W_freq = img_size // 2 + 1

        scale = 1.0 / math.sqrt(channels * img_size * self.W_freq)
        self.filter = nn.Parameter(torch.randn(1, channels, img_size, self.W_freq, dtype=torch.cfloat) * scale)

        self.time_mod = nn.Sequential(
            nn.Linear(time_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_fft = fft.rfft2(x, s=(H, W), dim=(-2, -1))

        t_m = self.time_mod(t_emb).view(B, C, 1, 1, 2)
        time_complex = torch.complex(t_m[..., 0], t_m[..., 1])

        eff_filter = self.filter * (1.0 + time_complex)
        out_fft = x_fft * eff_filter
        return fft.irfft2(out_fft, s=(H, W), dim=(-2, -1))


class HoloDiffusion2D_HQ(nn.Module):
    def __init__(self, img_size: int = 32, hidden_channels: int = 16, time_dim: int = 64, timesteps: int = 100) -> None:
        super().__init__()
        self.img_size = img_size
        self.timesteps = timesteps

        betas = torch.linspace(1e-4, 0.02, timesteps)
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

        # Proyección a espacio latente multiespectral
        self.in_conv = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1)
        self.spec_block1 = MultiChannelSpectralBlock2D(hidden_channels, img_size, time_dim)
        self.spec_block2 = MultiChannelSpectralBlock2D(hidden_channels, img_size, time_dim)
        self.act = nn.GELU()
        self.out_conv = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.act(self.in_conv(x))
        h = h + self.spec_block1(h, t_emb)
        h = self.act(h)
        h = h + self.spec_block2(h, t_emb)
        return self.out_conv(h)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        self.eval()
        device = self.betas.device
        x = torch.randn(num_samples, 1, self.img_size, self.img_size, device=device)

        for t_step in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), t_step, device=device, dtype=torch.long)
            pred_noise = self.forward(x, t)

            alpha = self.alphas[t_step]
            alpha_hat = self.alphas_cumprod[t_step]
            beta = self.betas[t_step]

            noise = torch.randn_like(x) if t_step > 0 else torch.zeros_like(x)
            mean = (1.0 / math.sqrt(alpha)) * (x - (beta / math.sqrt(1.0 - alpha_hat)) * pred_noise)
            x = mean + math.sqrt(beta) * noise

        return x


def loss_phase_2d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pérdida combinada MSE + Fase Espectral 2D."""
    mse = nn.functional.mse_loss(pred, target)
    p_fft = fft.rfft2(pred, dim=(-2, -1))
    t_fft = fft.rfft2(target, dim=(-2, -1))

    inner = torch.real(p_fft * torch.conj(t_fft))
    norm = (torch.abs(p_fft) * torch.abs(t_fft)) + 1e-7
    cos_phase = inner / norm
    phase_loss = torch.mean(1.0 - cos_phase)

    return mse + 0.5 * phase_loss


def main() -> None:
    SEED = 42
    enforce_reproducibility(SEED)

    CONFIG = {
        "img_size": 32,
        "hidden_channels": 16,
        "time_dim": 64,
        "timesteps": 80,
        "batch_size": 32,
        "lr": 3e-3,
        "epochs": 55,
        "samples": 512
    }

    print("\n=======================================================")
    print("  ENTRENANDO HOLODIFFUSION 2D HQ (ALTA DEFINICIÓN)    ")
    print("=======================================================")

    model = HoloDiffusion2D_HQ(
        img_size=CONFIG["img_size"],
        hidden_channels=CONFIG["hidden_channels"],
        time_dim=CONFIG["time_dim"],
        timesteps=CONFIG["timesteps"]
    )

    dataset = generate_cosmic_web_dataset(num_samples=CONFIG["samples"], size=CONFIG["img_size"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)

    model.train()
    for epoch in range(1, CONFIG["epochs"] + 1):
        tot_loss = 0.0
        for x_0 in dataloader:
            optimizer.zero_grad()
            B = x_0.shape[0]
            t = torch.randint(0, CONFIG["timesteps"], (B,), device=x_0.device)
            x_t, true_noise = model.q_sample(x_0, t)
            pred = model(x_t, t)
            loss = loss_phase_2d(pred, true_noise)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * B

        if epoch % 10 == 0 or epoch == CONFIG["epochs"]:
            print(f"Época [{epoch:02d}/{CONFIG['epochs']:02d}] | Loss Combinada (MSE+Fase 2D): {tot_loss/CONFIG['samples']:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/holodiff_2d_hq.pt")
    print("✔ Checkpoint HQ guardado.")


if __name__ == "__main__":
    main()
