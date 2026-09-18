"""
Script: visualize_adscft.py
Propósito: Visualizar la geometría del espacio curvo Anti-de Sitter (AdS)
y la emergencia de la dimensión radial z a partir de una secuencia 1D.
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.layers.adscft import AdSCFTBulkBoundaryLayer


def main() -> None:
    torch.manual_seed(42)

    B = 1
    T = 64
    D = 32
    Z_SLICES = 32
    Z_MAX = 3.0

    print("\n=======================================================")
    print("  SIMULANDO CORRESPONDENCIA AdS/CFT (MALDACENA)       ")
    print("=======================================================")

    # Instanciamos la capa AdS/CFT con alta resolución radial
    adscft = AdSCFTBulkBoundaryLayer(
        dim=D,
        seq_len=T,
        bulk_depth_slices=Z_SLICES,
        z_max=Z_MAX
    )
    adscft.eval()

    # Creamos una señal en la frontera (z=0) con dos impulsos localizados y ruido
    t = torch.linspace(0, 1, T)
    boundary_signal = torch.zeros(B, T, D)
    # Impulso 1 en x=16, Impulso 2 en x=48
    impulse1 = torch.exp(-80.0 * (t - 0.25) ** 2)
    impulse2 = torch.exp(-80.0 * (t - 0.75) ** 2)
    carrier = impulse1 + 0.7 * impulse2 + 0.15 * torch.randn(T)

    for d in range(D):
        boundary_signal[0, :, d] = carrier * math.cos(d * 0.2)

    with torch.no_grad():
        out_boundary, bulk_field = adscft(boundary_signal)

    # bulk_field: [1, Z, T, D]
    # Calculamos la energía total del campo en cada punto (x, z) colapsando la dimensión de canales
    bulk_energy = torch.norm(bulk_field[0], p=2, dim=-1).numpy()  # [Z, T]

    # --- Graficar el Espacio-Tiempo AdS ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [1, 2.2]})
    fig.patch.set_facecolor('#0f141c')

    for ax in axes:
        ax.set_facecolor('#161d27')
        ax.tick_params(colors='#c0c7d0')
        ax.xaxis.label.set_color('#c0c7d0')
        ax.yaxis.label.set_color('#c0c7d0')
        for s in ax.spines.values():
            s.set_color('#2d3748')

    # Gráfico 1: La Frontera 1D (CFT en z = 0)
    axes[0].plot(carrier.numpy(), color='#63b3ed', lw=2.0, label="Entrada en Frontera x(t) [z = 0]")
    axes[0].plot(out_boundary[0, :, 0].numpy(), color='#48bb78', lw=1.6, linestyle='--', label="Salida Holográfica Re-proyectada")
    axes[0].set_title("1. Señal Física en la Frontera (z = 0): Impulsos UV con Ruido Local", color='#e2e8f0', fontsize=11)
    axes[0].set_ylabel("Amplitud")
    axes[0].legend(facecolor='#161d27', edgecolor='#2d3748', labelcolor='white')

    # Gráfico 2: El Bulk de Maldacena 2D (Emergencia del Espacio Curvo Anti-de Sitter)
    x_coords = np.arange(T)
    z_coords = adscft.z_coords.numpy()

    im = axes[1].imshow(
        bulk_energy,
        extent=[0, T - 1, Z_MAX, 0.1],  # z=0.1 arriba (frontera), z=Z_MAX abajo (interior profundo)
        aspect='auto',
        cmap='magma'
    )
    axes[1].set_title("2. Espacio-Tiempo Curvo de Maldacena: Propagación en el Bulk AdS [x, z]", color='#e2e8f0', fontsize=11)
    axes[1].set_xlabel("Coordenada Espacial de la Frontera (x)")
    axes[1].set_ylabel("Profundidad Radial en el Bulk AdS (z) [UV -> IR]")

    cbar = fig.colorbar(im, ax=axes[1], pad=0.02)
    cbar.ax.yaxis.set_tick_params(color='#c0c7d0')
    cbar.outline.set_edgecolor('#2d3748')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#c0c7d0')
    cbar.set_label("Densidad de Energía del Campo ||Phi(x, z)||", color='#c0c7d0')

    plt.tight_layout()
    out_png = "checkpoints/adscft_spacetime.png"
    plt.savefig(out_png, dpi=180, facecolor=fig.get_facecolor())
    print(f"\n✔ Mapa del Espacio-Tiempo AdS/CFT exportado exitosamente a: {out_png}")


if __name__ == "__main__":
    main()
