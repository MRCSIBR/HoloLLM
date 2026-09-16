"""
Script: train_ads_llm.py
Propósito: Entrenar HoloAdS-LLM y exportar el mapa gravitacional AdS del bulk
mientras razona sobre principios holográficos y de física fundamental.
"""

import os
import math
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.models.holo_ads_llm import HoloAdSLLM
from src.utils.seed import enforce_reproducibility
from src.utils.audit import generate_audit_metadata
from train_lm import CORPUS, CharTokenizer


def main() -> None:
    SEED = 42
    enforce_reproducibility(SEED)

    tokenizer = CharTokenizer(CORPUS)
    data = tokenizer.encode(CORPUS)
    vocab_size = tokenizer.vocab_size

    DIM = 128
    DEPTH = 2
    NUM_HEADS = 4
    SEQ_LEN = 64
    BULK_SLICES = 24
    Z_MAX = 2.5
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 3e-3

    print("\n=======================================================")
    print("  INICIALIZANDO HoloAdS-LLM (MALDACENA BULK-BOUNDARY)  ")
    print("=======================================================")
    print(f"Vocabulario: {vocab_size} tokens | Dimensión latente: {DIM}")
    print(f"Rebanadas radiales en el Bulk AdS (z): {BULK_SLICES} (z_max = {Z_MAX})")

    model = HoloAdSLLM(
        vocab_size=vocab_size,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        max_seq_len=SEQ_LEN,
        bulk_slices=BULK_SLICES,
        z_max=Z_MAX
    )

    audit = model.audit_hashes()
    print(f"HoloAdS-LLM Code SHA-256: {audit['class_code_hash']}")
    print(f"Initial Weights SHA-256:  {audit['state_dict_hash']}")

    # Preparar lotes de entrenamiento de longitud fija
    X_list, Y_list = [], []
    for i in range(0, len(data) - SEQ_LEN - 1, 2):
        X_list.append(data[i : i + SEQ_LEN])
        Y_list.append(data[i + 1 : i + SEQ_LEN + 1])

    X = torch.stack(X_list)
    Y = torch.stack(Y_list)

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\n--- [Train] Optimizando el lenguaje a través del Bulk Gravitacional ---")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        tot_loss = 0.0
        for bx, by in dataloader:
            optimizer.zero_grad()
            logits, _, _ = model(bx)
            loss = criterion(logits.view(-1, vocab_size), by.view(-1))
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * bx.size(0)

        ep_loss = tot_loss / len(X)
        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Época [{epoch:02d}/{EPOCHS:02d}] | AdS-Language Loss: {ep_loss:.4f}")

    # Guardar Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/holo_ads_llm_v1.pt"
    audit_meta = generate_audit_metadata(model=model, hyperparams={"dim": DIM, "z_slices": BULK_SLICES}, seed=SEED)
    torch.save({"state_dict": model.state_dict(), "audit": audit_meta}, ckpt_path)
    print(f"\n✔ Checkpoint HoloAdS-LLM guardado en: {ckpt_path}")

    # Demostración y Mapeo del Espaciotiempo Interno
    print("\n=======================================================")
    print("  MAPEO DEL BULK AdS INTERNO DURANTE LA GENERACIÓN     ")
    print("=======================================================")

    test_prompt = "El universo no esta "
    prompt_tensor = tokenizer.encode(test_prompt).unsqueeze(0)

    gen_tokens, bulk_field = model.generate_and_inspect_bulk(
        prompt_ids=prompt_tensor,
        max_new_tokens=40,
        temperature=0.3
    )
    gen_text = tokenizer.decode(gen_tokens)

    print(f"\nPrompt:    '{test_prompt}'")
    print(f"Generado:  '{gen_text}'")

    # Graficar la densidad de energía del campo AdS durante la lectura del texto
    # bulk_field: [Z, T, D] -> calcular norma L2 sobre los canales latentes D
    bulk_energy = torch.norm(bulk_field, p=2, dim=-1).numpy()  # [Z, T]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('#0f141c')
    ax.set_facecolor('#161d27')

    im = ax.imshow(
        bulk_energy,
        extent=[0, bulk_energy.shape[1] - 1, Z_MAX, 0.1],
        aspect='auto',
        cmap='plasma'
    )
    ax.set_title(f"Espaciotiempo AdS Curvo Emergente del Prompt: '{test_prompt}'", color='#e2e8f0', fontsize=12, pad=10)
    ax.set_xlabel("Índice de Caracter / Token en la Frontera (t)", color='#c0c7d0')
    ax.set_ylabel("Profundidad Radial en el Bulk AdS (z) [UV -> IR]", color='#c0c7d0')
    ax.tick_params(colors='#c0c7d0')

    for s in ax.spines.values():
        s.set_color('#2d3748')

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color='#c0c7d0')
    cbar.outline.set_edgecolor('#2d3748')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#c0c7d0')
    cbar.set_label("Densidad Gravitacional del Concepto ||Phi(t, z)||", color='#c0c7d0')

    plt.tight_layout()
    out_map = "checkpoints/llm_bulk_spacetime.png"
    plt.savefig(out_map, dpi=180, facecolor=fig.get_facecolor())
    print(f"\n✔ Mapa del Espaciotiempo Interno del LLM exportado a: {out_map}")


if __name__ == "__main__":
    main()
