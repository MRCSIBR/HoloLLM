import os
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.holographic_mlp import HolographicClassifier
from src.layers.hrr import HolographicReducedRepresentation
from src.utils.seed import enforce_reproducibility
from src.utils.audit import generate_audit_metadata

def generate_harmonic_dataset(
    num_samples: int = 1024,
    dim: int = 64,
    num_classes: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(0, 1, dim)
    X = torch.zeros(num_samples, dim)
    y = torch.randint(0, num_classes, (num_samples,))

    for i in range(num_samples):
        cls = y[i].item()
        base_freq = (cls + 1) * 2.5
        signal = torch.sin(2 * np.pi * base_freq * t) + 0.4 * torch.cos(4 * np.pi * base_freq * t)
        noise = torch.randn(dim) * 0.08
        X[i] = signal + noise

    return X, y

def run_hrr_algebra_check(batch_eval: int = 32, dim: int = 128) -> None:
    print("\n--- [Audit] Verificando Álgebra Holográfica Reducida (Plate HRR) ---")
    hrr = HolographicReducedRepresentation(dim=dim)
    v_keys = hrr.normalize(torch.randn(batch_eval, dim))
    v_values = hrr.normalize(torch.randn(batch_eval, dim))

    memory_traces = hrr.bind(v_keys, v_values)
    retrieved_values = hrr.unbind(memory_traces, v_keys)

    cos_sims = torch.cosine_similarity(v_values, retrieved_values, dim=-1)
    mean_sim = cos_sims.mean().item()
    std_sim = cos_sims.std().item()

    print(f"Similitud Coseno Media: {mean_sim:.4f} (± {std_sim:.4f}) | Teórico: ~0.707")
    assert mean_sim > 0.60, f"Fidelidad asociativa baja: {mean_sim}"
    print("✔ HRR: Operaciones circulares validadas.")

def main() -> None:
    SEED = 42
    enforce_reproducibility(SEED)
    run_hrr_algebra_check()

    HYPERPARAMS: Dict[str, Any] = {
        "input_dim": 64,
        "hidden_dim": 64,
        "num_classes": 4,
        "modes": 24,
        "learning_rate": 3e-3,
        "batch_size": 32,
        "epochs": 12
    }

    print("\n--- [Audit] Inicializando Red Holográfica ---")
    model = HolographicClassifier(
        input_dim=HYPERPARAMS["input_dim"],
        hidden_dim=HYPERPARAMS["hidden_dim"],
        num_classes=HYPERPARAMS["num_classes"],
        modes=HYPERPARAMS["modes"]
    )

    initial_audit = model.audit_hashes()
    print(f"Model Code SHA-256:  {initial_audit['class_code_hash']}")
    print(f"Initial Weights SHA: {initial_audit['state_dict_hash']}")

    X, y = generate_harmonic_dataset(
        num_samples=1024,
        dim=HYPERPARAMS["input_dim"],
        num_classes=HYPERPARAMS["num_classes"]
    )
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=HYPERPARAMS["batch_size"], shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=HYPERPARAMS["learning_rate"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("\n--- [Train] Optimizando espectro de interferencia ---")
    for epoch in range(1, HYPERPARAMS["epochs"] + 1):
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        if epoch % 3 == 0 or epoch == HYPERPARAMS["epochs"]:
            acc = correct / total
            print(f"Epoch [{epoch:02d}/{HYPERPARAMS['epochs']:02d}] | Loss: {total_loss/total:.4f} | Acc: {acc * 100:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/holographic_mlp_v1.pt"
    audit_metadata = generate_audit_metadata(model=model, hyperparams=HYPERPARAMS, seed=SEED)
    torch.save({"state_dict": model.state_dict(), "audit": audit_metadata}, checkpoint_path)
    print(f"\n✔ Checkpoint guardado en: {checkpoint_path}")
    print(f"Final Weights SHA-256: {audit_metadata['weights_hash']}")

if __name__ == "__main__":
    main()
