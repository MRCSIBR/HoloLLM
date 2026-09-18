"""
Script: visualize_diffusion_2d.py
Propósito: Visualizar el despliegue generativo de imágenes 2D (Red Cósmica)
y el holograma espectral de Fourier correspondiente.
"""

import math
import torch
import matplotlib.pyplot as plt
from src.models.holographic_diffusion_2d import HoloDiffusion2D
from train_diffusion_2d import generate_cosmic_web_dataset

def main() -> None:
    ckpt = torch.load("checkpoints/holodiff_2d_v1.pt", map_location="cpu", weights_only=False)
    hp = ckpt["audit"]["hyperparameters"]

    model = HoloDiffusion2D(
        img_size=hp["img_size"],
        time_dim=hp["time_dim"],
        timesteps=hp["timesteps"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    torch.manual_seed(1337)
    timesteps = hp["timesteps"]
    x = torch.randn(1, 1, hp["img_size"], hp["img_size"])
    saved_steps = [timesteps - 1, int(timesteps * 0.6), int(timesteps * 0.3), 0]
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
                trajectories[t_step] = x[0, 0].clone().numpy()

    # Muestra real del dataset para comparación
    real_sample = generate_cosmic_web_dataset(num_samples=1, size=hp["img_size"])[0, 0].numpy()

    # --- Mosaico Gráfico ---
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    fig.patch.set_facecolor('#0f141c')

    titles = [
        "Real (Dataset)",
        f"t = {timesteps-1} (Caos)",
        f"t = {int(timesteps*0.6)}",
        f"t = {int(timesteps*0.3)}",
        "t = 0 (Generado)"
    ]
    images = [
        real_sample,
        trajectories[saved_steps[0]],
        trajectories[saved_steps[1]],
        trajectories[saved_steps[2]],
        trajectories[saved_steps[3]]
    ]

    for i, ax in enumerate(axes):
        ax.set_facecolor('#161d27')
        im = ax.imshow(images[i], cmap='inferno')
        ax.set_title(titles[i], color='#e2e8f0', fontsize=11, pad=8)
        ax.axis('off')

    plt.tight_layout()
    out_file = "checkpoints/diffusion_2d_cosmic.png"
    plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor())
    print(f"\n✔ Mosaico de Difusión 2D exportado a: {out_file}")

if __name__ == "__main__":
    main()
