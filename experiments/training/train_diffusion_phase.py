"""
Script: train_diffusion_phase.py
Propósito: Re-entrenamiento de HoloDiff incorporando la pérdida de consistencia
de fase de Gabor/Pribram para eliminar asperezas residuales.
"""

import os
import math
from typing import Dict, Any
import torch
import torch.optim as optim

from src.models.holographic_diffusion import HolographicDiffusion
from src.utils.spectral_losses import HolographicPhaseConsistencyLoss
from src.utils.seed import enforce_reproducibility
from src.utils.audit import generate_audit_metadata
from train_diffusion import generate_wavepacket_dataset


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
        "num_samples": 1024,
        "alpha_mag": 0.4,
        "beta_phase": 0.8
    }

    print("\n=======================================================")
    print("  ENTRENAMIENTO HOLODIFF CON COHERENCIA DE FASE (GABOR) ")
    print("=======================================================")

    model = HolographicDiffusion(
        signal_len=CONFIG["signal_len"],
        time_dim=CONFIG["time_dim"],
        modes=CONFIG["modes"],
        timesteps=CONFIG["timesteps"]
    )

    criterion = HolographicPhaseConsistencyLoss(
        alpha_mag=CONFIG["alpha_mag"],
        beta_phase=CONFIG["beta_phase"]
    )

    dataset = generate_wavepacket_dataset(num_samples=CONFIG["num_samples"], length=CONFIG["signal_len"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)

    model.train()
    print("Optimizando simultáneamente: MSE espacial + Amplitud + Desfase angular...")
    for epoch in range(1, CONFIG["epochs"] + 1):
        tot_loss, tot_mse, tot_phase = 0.0, 0.0, 0.0
        for x_0 in dataloader:
            optimizer.zero_grad()
            B = x_0.shape[0]
            t = torch.randint(0, CONFIG["timesteps"], (B,), device=x_0.device)
            x_t, true_noise = model.q_sample(x_0, t)
            pred_noise = model(x_t, t)

            loss, metrics = criterion(pred_noise, true_noise)
            loss.backward()
            optimizer.step()

            tot_loss += metrics["total_loss"] * B
            tot_mse += metrics["loss_mse"] * B
            tot_phase += metrics["loss_phase"] * B

        N = CONFIG["num_samples"]
        if epoch % 5 == 0 or epoch == CONFIG["epochs"]:
            print(f"Época [{epoch:02d}/{CONFIG['epochs']:02d}] | Total: {tot_loss/N:.4f} | MSE: {tot_mse/N:.4f} | Fase Residual: {tot_phase/N:.4f}")

    # Guardar Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/holodiff_phase_v1.pt"
    audit_meta = generate_audit_metadata(model=model, hyperparams=CONFIG, seed=SEED)
    torch.save({"state_dict": model.state_dict(), "audit": audit_meta}, ckpt_path)
    print(f"\n✔ Checkpoint con sintonización de fase guardado en: {ckpt_path}")


if __name__ == "__main__":
    main()
