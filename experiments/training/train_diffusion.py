"""
Script: train_diffusion.py
Propósito: Entrenamiento del primer modelo de Difusión Holográfica (HoloDiffusion)
en PyTorch puro sobre paquetes de ondas físicas armónicas.
"""

import os
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.holographic_diffusion import HolographicDiffusion
from src.utils.seed import enforce_reproducibility
from src.utils.audit import generate_audit_metadata


def generate_wavepacket_dataset(num_samples: int = 1024, length: int = 64) -> torch.Tensor:
    """
    Genera un manifold físico de paquetes de onda (Bohmian wavepackets):
    Combinación armónica coherente con modulación de envolvente gaussiana.
    """
    t = torch.linspace(-1, 1, length)
    signals = []
    for _ in range(num_samples):
        # Frecuencias fundamentales y envolvente
        f1 = torch.empty(1).uniform_(2.0, 5.0).item()
        f2 = torch.empty(1).uniform_(8.0, 12.0).item()
        envelope = torch.exp(-4.0 * (t ** 2))
        carrier = torch.sin(2 * math.pi * f1 * t) + 0.5 * torch.cos(2 * math.pi * f2 * t)
        signal = envelope * carrier
        # Normalizar a varianza unitaria
        signal = signal / (torch.std(signal) + 1e-6)
        signals.append(signal)
    return torch.stack(signals)


import math

def main() -> None:
    SEED = 42
    enforce_reproducibility(SEED)

    CONFIG: Dict[str, Any] = {
        "signal_len": 64,
        "time_dim": 64,
        "modes": 24,
        "timesteps": 120,
        "batch_size": 32,
        "learning_rate": 3e-3,
        "epochs": 25,
        "num_samples": 1024
    }

    print("\n=======================================================")
    print("  INICIALIZANDO HOLOGRAPHIC DIFFUSION MODEL (HoloDiff) ")
    print("=======================================================")
    model = HolographicDiffusion(
        signal_len=CONFIG["signal_len"],
        time_dim=CONFIG["time_dim"],
        modes=CONFIG["modes"],
        timesteps=CONFIG["timesteps"]
    )

    audit = model.audit_hashes()
    print(f"HoloDiff Code SHA-256:    {audit['class_code_hash']}")
    print(f"Initial Weights SHA-256:  {audit['state_dict_hash']}")

    # Dataset de entrenamiento
    dataset = generate_wavepacket_dataset(num_samples=CONFIG["num_samples"], length=CONFIG["signal_len"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("\n--- [Train] Entrenando el des-enfundado holográfico de ruido ---")
    model.train()
    for epoch in range(1, CONFIG["epochs"] + 1):
        total_loss = 0.0
        for x_0 in dataloader:
            optimizer.zero_grad()
            B = x_0.shape[0]
            # Muestrear timesteps aleatorios uniformemente
            t = torch.randint(0, CONFIG["timesteps"], (B,), device=x_0.device)
            # Aplicar difusión forward
            x_t, true_noise = model.q_sample(x_0, t)
            # Predecir ruido con el modelo espectral
            pred_noise = model(x_t, t)

            loss = criterion(pred_noise, true_noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * B

        avg_loss = total_loss / CONFIG["num_samples"]
        if epoch % 5 == 0 or epoch == CONFIG["epochs"]:
            print(f"Época [{epoch:02d}/{CONFIG['epochs']:02d}] | MSE Denoising Loss: {avg_loss:.4f}")

    # Guardar Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/holodiff_v1.pt"
    audit_meta = generate_audit_metadata(model=model, hyperparams=CONFIG, seed=SEED)
    torch.save({"state_dict": model.state_dict(), "audit": audit_meta}, ckpt_path)
    print(f"\n✔ Checkpoint HoloDiff guardado en: {ckpt_path}")
    print(f"Final Weights SHA-256: {audit_meta['weights_hash']}")

    # Generación de Muestras desde el Caos Puro
    print("\n=======================================================")
    print("  GENERANDO ONDAS DESDE EL VACÍO CUÁNTICO (REVERSE)    ")
    print("=======================================================")
    generated_waves = model.sample(num_samples=2)
    print(f"✔ 2 ondas sintéticas desplegadas con éxito. Shape: {generated_waves.shape}")

if __name__ == "__main__":
    main()
