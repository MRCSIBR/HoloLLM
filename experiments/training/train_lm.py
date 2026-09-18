"""
Script: train_lm.py
Propósito: Entrenamiento del primer Holographic Language Model (HoloLM)
a nivel de caracteres con registro criptográfico y generación generativa.
"""

import os
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.holographic_lm import HolographicLM
from src.utils.seed import enforce_reproducibility
from src.utils.audit import generate_audit_metadata


# Texto de entrenamiento: fragmento alegórico sobre la naturaleza holográfica
CORPUS = """
El universo no esta compuesto por partes aisladas, sino por un orden implicado donde
cada region del espacio contiene la totalidad de la informacion en forma de interferencia.
En el cerebro holografico de Pribram, la memoria no se ubica en un punto fijo, sino que
se distribuye como ondas que resuenan con el cosmos en continuo aprendizaje.
Toda frontera proyecta su volumen interior de acuerdo con el principio holografico.
"""

class CharTokenizer:
    """Tokenizer determinista a nivel de caracteres."""
    def __init__(self, text: str) -> None:
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)

    def decode(self, indices: List[int]) -> str:
        return "".join([self.idx2char[idx] for idx in indices])


def main() -> None:
    SEED = 42
    enforce_reproducibility(SEED)

    tokenizer = CharTokenizer(CORPUS)
    data = tokenizer.encode(CORPUS)

    HYPERPARAMS: Dict[str, Any] = {
        "vocab_size": tokenizer.vocab_size,
        "dim": 96,
        "depth": 2,
        "modes": 24,
        "decay": 0.96,
        "seq_len": 48,
        "batch_size": 16,
        "learning_rate": 4e-3,
        "epochs": 40
    }

    print("\n=======================================================")
    print("  INICIALIZANDO HOLOGRAPHIC LANGUAGE MODEL (HoloLM)    ")
    print("=======================================================")
    print(f"Vocabulario: {tokenizer.vocab_size} caracteres únicos.")
    print(f"Dimensión latente (D): {HYPERPARAMS['dim']} | Modos Fourier: {HYPERPARAMS['modes']}")

    model = HolographicLM(
        vocab_size=HYPERPARAMS["vocab_size"],
        dim=HYPERPARAMS["dim"],
        depth=HYPERPARAMS["depth"],
        modes=HYPERPARAMS["modes"],
        decay=HYPERPARAMS["decay"]
    )

    initial_audit = model.audit_hashes()
    print(f"HoloLM Code SHA-256:    {initial_audit['class_code_hash']}")
    print(f"Initial Weights SHA-256: {initial_audit['state_dict_hash']}")

    # Preparar lotes de entrenamiento (secuencias de longitud fija)
    X_list, Y_list = [], []
    for i in range(0, len(data) - HYPERPARAMS["seq_len"] - 1, 2):
        X_list.append(data[i : i + HYPERPARAMS["seq_len"]])
        Y_list.append(data[i + 1 : i + HYPERPARAMS["seq_len"] + 1])

    X = torch.stack(X_list)
    Y = torch.stack(Y_list)

    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=HYPERPARAMS["batch_size"], shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=HYPERPARAMS["learning_rate"], weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\n--- [Train] Optimizando HoloLM sobre el corpus en el Orden Implicado ---")
    model.train()
    for epoch in range(1, HYPERPARAMS["epochs"] + 1):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            # Reshape para CrossEntropyLoss: [B*T, Vocab] vs [B*T]
            loss = criterion(logits.view(-1, HYPERPARAMS["vocab_size"]), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0 or epoch == HYPERPARAMS["epochs"]:
            print(f"Epoch [{epoch:02d}/{HYPERPARAMS['epochs']:02d}] | Cross-Entropy Loss: {avg_loss:.4f}")

    # Guardar Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/hololm_v1.pt"
    audit_meta = generate_audit_metadata(model=model, hyperparams=HYPERPARAMS, seed=SEED)
    torch.save({"state_dict": model.state_dict(), "audit": audit_meta}, ckpt_path)
    print(f"\n✔ Checkpoint HoloLM guardado en: {ckpt_path}")
    print(f"Final Weights SHA-256: {audit_meta['weights_hash']}")

    # Prueba de Generación Autoregresiva
    print("\n=======================================================")
    print("  GENERACIÓN HOLOGRÁFICA AUTOREGRESIVA (PROMPT TEST)   ")
    print("=======================================================")
    prompt_text = "El universo "
    prompt_ids = tokenizer.encode(prompt_text).unsqueeze(0)

    generated_ids = model.generate(prompt_ids, max_new_tokens=80, temperature=0.7)
    decoded_output = tokenizer.decode(generated_ids[0].tolist())

    print(f"Prompt:    '{prompt_text}'")
    print(f"Generado:  '{decoded_output}'")

if __name__ == "__main__":
    main()
