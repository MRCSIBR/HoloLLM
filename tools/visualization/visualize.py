"""
Script: visualize.py
Propósito: Inspección visual de la modulación espectral (Pribram/Bohm)
y recuperación asociativa distribuida (Plate HRR).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.holographic_mlp import HolographicClassifier
from src.layers.hrr import HolographicReducedRepresentation
from train import generate_harmonic_dataset

def main() -> None:
    # 1. Cargar modelo y checkpoint auditado
    checkpoint_path = "checkpoints/holographic_mlp_v1.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    audit = checkpoint["audit"]
    hp = audit["hyperparameters"]

    print(f"Cargando checkpoint con SHA de pesos: {audit['weights_hash'][:16]}...")
    model = HolographicClassifier(
        input_dim=hp["input_dim"],
        hidden_dim=hp["hidden_dim"],
        num_classes=hp["num_classes"],
        modes=hp["modes"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 2. Extraer parámetros del "Holograma" (Capa 1)
    amplitude, phase = model.holo1.get_filter_stats()
    modes = np.arange(len(amplitude))

    # 3. Test de recuperación HRR
    dim = 128
    hrr = HolographicReducedRepresentation(dim=dim)
    torch.manual_seed(42)
    key = hrr.normalize(torch.randn(1, dim))
    value = hrr.normalize(torch.randn(1, dim))
    trace = hrr.bind(key, value)
    recovered = hrr.unbind(trace, key)

    # 4. Señal de prueba antes y después de interferencia
    X, _ = generate_harmonic_dataset(num_samples=1, dim=hp["input_dim"], num_classes=hp["num_classes"])
    with torch.no_grad():
        x_in = X[0:1]
        x_holo = model.holo1(x_in)

    # --- Generación de Gráficos ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    fig.patch.set_facecolor('#0f141c')

    for ax in axes:
        ax.set_facecolor('#161d27')
        ax.tick_params(colors='#c0c7d0')
        ax.xaxis.label.set_color('#c0c7d0')
        ax.yaxis.label.set_color('#c0c7d0')
        ax.title.set_color('#e2e8f0')
        for spine in ax.spines.values():
            spine.set_color('#2d3748')

    # Gráfico 1: Placa Holográfica (Amplitud y Fase por Modo)
    ax1_twin = axes[0].twinx()
    ax1_twin.tick_params(colors='#f6ad55')
    ax1_twin.yaxis.label.set_color('#f6ad55')
    
    line1 = axes[0].stem(modes, amplitude.detach().numpy(), linefmt='c-', markerfmt='co', basefmt='k-', label='Amplitud |W|')
    line2 = ax1_twin.plot(modes, phase.detach().numpy(), color='#f6ad55', linestyle='--', marker='x', label='Fase arg(W)')
    axes[0].set_title("1. Placa Holográfica Aprendida (Karl Pribram) - Amplitud y Desfase por Modo", fontsize=11, pad=10)
    axes[0].set_xlabel("Modo de Frecuencia (k)")
    axes[0].set_ylabel("Amplitud Espectral", color='cyan')
    ax1_twin.set_ylabel("Ángulo de Fase (rad)", color='#f6ad55')

    # Gráfico 2: Orden Explicado vs Filtrado Holográfico
    axes[1].plot(x_in[0].numpy(), label="Señal Original (Entrada Física)", color='#63b3ed', lw=2)
    axes[1].plot(x_holo[0].numpy(), label="Interferencia Espectral (Salida Holo1)", color='#68d391', lw=1.8, linestyle='-.')
    axes[1].set_title("2. Dinámica Bohmiana: Entrada vs Señal Reorganizada por Interferencia", fontsize=11, pad=10)
    axes[1].set_xlabel("Índice Espacial / Temporal")
    axes[1].set_ylabel("Valor de Activación")
    axes[1].legend(facecolor='#161d27', edgecolor='#2d3748', labelcolor='white')

    # Gráfico 3: Álgebra de Tony Plate (HRR)
    t_dim = np.arange(dim)
    axes[2].plot(t_dim, value[0].numpy(), label="Vector Original V (Explicado)", color='#cbd5e0', lw=1.5)
    axes[2].plot(t_dim, recovered[0].numpy(), label="Vector Recuperado V̂ tras Unbind (Plate)", color='#f687b3', lw=1.5, linestyle='--')
    axes[2].set_title("3. Memoria Asociativa HRR: Recuperación desde el Orden Implicado V̂ = (K ⊛ V) ⊚ K", fontsize=11, pad=10)
    axes[2].set_xlabel("Dimensión Vectorial")
    axes[2].set_ylabel("Amplitud")
    axes[2].legend(facecolor='#161d27', edgecolor='#2d3748', labelcolor='white')

    plt.tight_layout()
    output_png = "checkpoints/holographic_inspection.png"
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor())
    print(f"\n✔ Gráfico científico exportado con éxito a: {output_png}")

if __name__ == "__main__":
    main()
