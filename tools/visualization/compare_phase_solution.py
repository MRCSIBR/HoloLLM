"""
Script: compare_phase_solution.py
Propósito: Comparar empíricamente la calidad de las ondas generadas
con y sin la corrección del problema de fase de Dennis Gabor.
"""

import math
import torch
import matplotlib.pyplot as plt
from src.models.holographic_diffusion import HolographicDiffusion

def load_and_sample(ckpt_path: str, seed: int = 42) -> torch.Tensor:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["audit"]["hyperparameters"]
    model = HolographicDiffusion(
        signal_len=hp["signal_len"],
        time_dim=hp["time_dim"],
        modes=hp["modes"],
        timesteps=hp["timesteps"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    torch.manual_seed(seed)
    with torch.no_grad():
        sample = model.sample(num_samples=1)
    return sample[0].numpy()

def main() -> None:
    # Cargar muestra del modelo antiguo (solo MSE)
    wave_standard = load_and_sample("checkpoints/holodiff_v1.pt")
    # Cargar muestra del modelo nuevo (con coherencia de fase)
    wave_phase = load_and_sample("checkpoints/holodiff_phase_v1.pt")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.patch.set_facecolor('#0f141c')

    for ax in axes:
        ax.set_facecolor('#161d27')
        ax.tick_params(colors='#c0c7d0')
        ax.yaxis.label.set_color('#c0c7d0')
        for s in ax.spines.values():
            s.set_color('#2d3748')

    # Gráfico 1: Modelo Antiguo (Solo MSE)
    axes[0].plot(wave_standard, color='#f687b3', lw=1.8, label="Solo MSE (Problema de Fase)")
    axes[0].set_title("1. Modelo Original: Con rizado de alta frecuencia y desfase residual", color='#e2e8f0', fontsize=11)
    axes[0].set_ylabel("Amplitud")
    axes[0].legend(facecolor='#161d27', edgecolor='#2d3748', labelcolor='white')

    # Gráfico 2: Modelo Nuevo (Fase Sintonizada)
    axes[1].plot(wave_phase, color='#48bb78', lw=2.0, label="Con Pérdida de Coherencia de Fase (Gabor / Pribram)")
    axes[1].set_title("2. Modelo con Coherencia de Fase: Onda armónica pura, suave y continua", color='#e2e8f0', fontsize=11)
    axes[1].set_ylabel("Amplitud")
    axes[1].set_xlabel("Coordenada Espacial (x)", color='#c0c7d0')
    axes[1].legend(facecolor='#161d27', edgecolor='#2d3748', labelcolor='white')

    plt.tight_layout()
    out_file = "checkpoints/phase_comparison.png"
    plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor())
    print(f"\n✔ Gráfico comparativo de resolución de fase exportado a: {out_file}")

if __name__ == "__main__":
    main()
