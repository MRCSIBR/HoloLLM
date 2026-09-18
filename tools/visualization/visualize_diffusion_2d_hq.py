import torch
import matplotlib.pyplot as plt
from train_diffusion_2d_hq import HoloDiffusion2D_HQ
from train_diffusion_2d import generate_cosmic_web_dataset

def main():
    model = HoloDiffusion2D_HQ(img_size=32, hidden_channels=16, time_dim=64, timesteps=80)
    model.load_state_dict(torch.load("checkpoints/holodiff_2d_hq.pt", map_location="cpu"))
    model.eval()

    torch.manual_seed(1337)
    sample_gen = model.sample(num_samples=1)[0, 0].numpy()
    sample_real = generate_cosmic_web_dataset(num_samples=1, size=32)[0, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.patch.set_facecolor('#0f141c')

    for ax, img, title in zip(axes, [sample_real, sample_gen], ["Real (Filamentos Cósmicos)", "Generado HQ (Fase 2D + Multicanal)"]):
        ax.set_facecolor('#161d27')
        ax.imshow(img, cmap='inferno')
        ax.set_title(title, color='#e2e8f0', fontsize=11, pad=8)
        ax.axis('off')

    plt.tight_layout()
    out = "checkpoints/diffusion_2d_hq_comparison.png"
    plt.savefig(out, dpi=180, facecolor=fig.get_facecolor())
    print(f"\n✔ Comparativa de Alta Definición guardada en: {out}")

if __name__ == "__main__":
    main()
