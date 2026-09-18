"""
Script: train_diffusion_2d.py
Propósito: Entrenar HoloDiffusion2D sobre un manifold de filamentos cósmicos y patrones de interferencia.
"""

import os
import math
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.holographic_diffusion_2d import HoloDiffusion2D
from src.utils.seed import enforce_reproducibility
from src.utils.audit import generate_audit_metadata


def generate_cosmic_web_dataset(num_samples: int = 512, size: int = 32) -> torch.Tensor:
    """
    Genera patrones 2D que simulan filamentos cósmicos e interferencias de ondas:
    Superposición de ondas sinusoidales 2D moduladas por envolventes gaussianas elípticas.
    """
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    images = []
    for _ in range(num_samples):
        # 2 filamentos que se cruzan en ángulos variables
        theta1 = torch.empty(1).uniform_(0, math.pi).item()
        theta2 = theta1 + torch.empty(1).uniform_(0.8, 1.8).item()
        freq = torch.empty(1).uniform_(2.0, 4.5).item()

        # Coordenadas rotadas
        u1 = xx * math.cos(theta1) + yy * math.sin(theta1)
        u2 = xx * math.cos(theta2) + yy * math.sin(theta2)

        filament1 = torch.exp(-12.0 * (u1 ** 2)) * torch.cos(2 * math.pi * freq * xx)
        filament2 = torch.exp(-12.0 * (u2 ** 2)) * torch.cos(2 * math.pi * freq * yy)

        img = filament1 + 0.8 * filament2
        img = img / (torch.std(img) + 1e-6)
        images.append(img.unsqueeze(0))

    return torch.stack(images)


def main() -> None:
    SEED = 42
    enforce_reproducibility(SEED)

    CONFIG: Dict[str, Any] = {
        "img_size": 32,
        "time_dim": 64,
        "timesteps": 80,
        "batch_size": 32,
        "learning_rate": 4e-3,
        "epochs": 20,
        "num_samples": 512
    }

    print("\n=======================================================")
    print("  INICIALIZANDO HOLODIFFUSION 2D (RED CÓSMICA / PRIBRAM)")
    print("=======================================================")

    model = HoloDiffusion2D(
        img_size=CONFIG["img_size"],
        time_dim=CONFIG["time_dim"],
        timesteps=CONFIG["timesteps"]
    )

    audit = model.audit_hashes()
    print(f"HoloDiff2D Code SHA-256:   {audit['class_code_hash']}")
    print(f"Initial Weights SHA-256:   {audit['state_dict_hash']}")

    dataset = generate_cosmic_web_dataset(num_samples=CONFIG["num_samples"], size=CONFIG["img_size"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("\n--- [Train] Entrenando interferometría 2D en el Orden Implicado ---")
    model.train()
    for epoch in range(1, CONFIG["epochs"] + 1):
        tot_loss = 0.0
        for x_0 in dataloader:
            optimizer.zero_grad()
            B = x_0.shape[0]
            t = torch.randint(0, CONFIG["timesteps"], (B,), device=x_0.device)
            x_t, true_noise = model.q_sample(x_0, t)
            pred_noise = model(x_t, t)

            loss = criterion(pred_noise, true_noise)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * B

        avg_loss = tot_loss / CONFIG["num_samples"]
        if epoch % 5 == 0 or epoch == CONFIG["epochs"]:
            print(f"Época [{epoch:02d}/{CONFIG['epochs']:02d}] | 2D Denoising MSE: {avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/holodiff_2d_v1.pt"
    audit_meta = generate_audit_metadata(model=model, hyperparams=CONFIG, seed=SEED)
    torch.save({"state_dict": model.state_dict(), "audit": audit_meta}, ckpt_path)
    print(f"\n✔ Checkpoint 2D guardado exitosamente en: {ckpt_path}")


if __name__ == "__main__":
    main()
