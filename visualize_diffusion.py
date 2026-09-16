"""
Script: visualize_diffusion.py
Propósito: Graficar la trayectoria del des-enfundado del Orden Implicado:
desde el ruido gaussiano blanco (t=120) hasta el paquete de ondas final (t=0).
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.holographic_diffusion import HolographicDiffusion

def main() -> None:
    ckpt = torch.load("checkpoints/holodiff_v1.pt", map_location="cpu", weights_only=False)
    hp = ckpt["audit"]["hyperparameters"]

    model = HolographicDiffusion(
        signal_len=hp["signal_len"],
        time_dim=hp["time_dim"],
        modes=hp["modes"],
        timesteps=hp["timesteps"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    torch.manual_seed(42)
    # Muestrear guardando estados intermedios en pasos clave
    timesteps = hp["timesteps"]
    x = torch.randn(1, hp["signal_len"])
    saved_steps = [timesteps - 1, int(timesteps * 0.66), int(timesteps * 0.33), 0]
    trajectories = {}

    with torch.no_grad():
        for t_step in reversed(range(timesteps)):
            t = torch.tensor([t_step], dtype=torch.long)
            pred_noise = model(x, t)

            alpha = model.alphas[t_step]
            alpha_hat = model.alphas_cumprod[t_step]
            beta = model.betas[t_step]

            if t_step > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            mean = (1.0 / math.sqrt(alpha)) * (x - (beta / math.sqrt(1.0 - alpha_hat)) * pred_noise)
            x = mean + math.sqrt(beta) * noise

            if t_step in saved_steps:
                trajectories[t_step] = x[0].clone().numpy()

    # Graficar la trayectoria de Bohm
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.patch.set_facecolor('#0f141c')

    titles = [
        f"1. t = {timesteps-1} (Orden Implicado / Ruido Gaussiano Puro)",
        f"2. t = {int(timesteps*0.66)} (Aparición de frentes de onda de baja frecuencia)",
        f"3. t = {int(timesteps*0.33)} (Sintonización de fase y armónicos superiores)",
        "4. t = 0 (Orden Explicado / Paquete de Onda Coherente Generado)"
    ]
    colors = ['#e53e3e', '#dd6b20', '#3182ce', '#38a169']

    for i, step_val in enumerate(saved_steps):
        ax = axes[i]
        ax.set_facecolor('#161d27')
        ax.tick_params(colors='#c0c7d0')
        ax.yaxis.label.set_color('#c0c7d0')
        ax.set_title(titles[i], color='#e2e8f0', fontsize=11, pad=8)
        for s in ax.spines.values():
            s.set_color('#2d3748')

        ax.plot(trajectories[step_val], color=colors[i], lw=1.8)
        ax.set_ylabel("Amplitud")

    axes[-1].set_xlabel("Coordenada Espacial (x)", color='#c0c7d0')
    plt.tight_layout()

    out_file = "checkpoints/diffusion_unfolding.png"
    plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor())
    print(f"\n✔ Gráfico de Difusión Holográfica exportado a: {out_file}")

if __name__ == "__main__":
    main()
