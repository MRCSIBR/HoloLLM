"""
Script: train_causal_code.py
Propósito: Entrenar HoloCausalLM (~10.5M parámetros) con Loss Masking
sobre las 500 funciones del corpus de CodeAlpaca.
"""

import os
import json
import time
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken

from src.models.holo_causal_lm import HoloCausalLM
from src.utils.seed import enforce_reproducibility
from src.utils.audit import generate_audit_metadata


CORPUS_PATH = "data/python_code_corpus.json"


def prepare_causal_dataset(
    enc: any,
    max_samples: int = 400,
    max_len: int = 96
) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)[:max_samples]

    all_inputs, all_labels = [], []

    for item in raw:
        prompt_str = f"Task: {item['prompt'].strip()}\nCode:\n"
        code_str = item["completion"].strip() + "<|endoftext|>"

        p_ids = enc.encode(prompt_str, allowed_special={"<|endoftext|>"})
        c_ids = enc.encode(code_str, allowed_special={"<|endoftext|>"})

        seq = p_ids + c_ids
        if len(seq) > max_len:
            seq = seq[:max_len]
            # Asegurar delimitador de fin si fue truncado
            seq[-1] = 50256

        # Loss Masking: los tokens del prompt no se penalizan (-100)
        labels = [-100] * min(len(p_ids), len(seq)) + seq[len(p_ids):]

        all_inputs.append(torch.tensor(seq, dtype=torch.long))
        all_labels.append(torch.tensor(labels, dtype=torch.long))

    # Padding homogéneo
    m_len = max(len(s) for s in all_inputs)
    pad_inputs = torch.full((len(all_inputs), m_len), 50256, dtype=torch.long)
    pad_labels = torch.full((len(all_labels), m_len), -100, dtype=torch.long)

    for i in range(len(all_inputs)):
        pad_inputs[i, :len(all_inputs[i])] = all_inputs[i]
        pad_labels[i, :len(all_labels[i])] = all_labels[i]

    return pad_inputs, pad_labels


import typing
from typing import Any

def main() -> None:
    SEED = 42
    enforce_reproducibility(SEED)

    enc = tiktoken.get_encoding("gpt2")

    DIM = 192
    NUM_HEADS = 4
    DEPTH = 2
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 2.5e-3
    SAMPLES = 450

    print("\n=======================================================")
    print("  ENTRENANDO HOLO-CAUSAL LM (DECODER-ONLY 10.5M)       ")
    print("=======================================================")

    model = HoloCausalLM(
        vocab_size=50257,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        max_seq_len=256
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"Total de Parámetros: {params:,} (~{params/1e6:.1f}M)")

    inputs, labels = prepare_causal_dataset(enc, max_samples=SAMPLES, max_len=96)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"\n--- [Train] Optimizando sobre {SAMPLES} funciones de código con Loss Masking ---")
    start = time.time()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        tot_loss = 0.0
        for b_x, b_y in dataloader:
            optimizer.zero_grad()
            logits, _ = model(b_x)
            # Desplazamiento causal: logits[:, :-1] predicen b_y[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = b_y[:, 1:].contiguous()

            loss = criterion(shift_logits.view(-1, 50257), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * b_x.size(0)

        ep_loss = tot_loss / SAMPLES
        if epoch % 4 == 0 or epoch == EPOCHS:
            print(f"Época [{epoch:02d}/{EPOCHS:02d}] | Code Loss: {ep_loss:.4f} | Tiempo: {time.time()-start:.1f}s")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/holo_causal_code_v1.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✔ Checkpoint guardado en: {ckpt_path}")

    # Demostración en Vivo
    print("\n=======================================================")
    print("  DEMOSTRACIÓN: GENERACIÓN DE CÓDIGO PYTHON REAL       ")
    print("=======================================================")

    test_prompts = [
        "Task: Write a function to check if a number is even.\nCode:\n",
        "Task: Create a python function to add two numbers.\nCode:\n"
    ]

    for p in test_prompts:
        prompt_ids = enc.encode(p)
        gen_tokens = model.generate(
            prompt_tokens=prompt_ids,
            max_new_tokens=40,
            temperature=0.4,
            top_k=25
        )
        # Decodificar solo la porción generada
        full_text = enc.decode(gen_tokens)
        code_part = full_text[len(p):].strip()

        print(f"\n[Prompt]:\n{p.strip()}")
        print(f"[Código Generado por HoloCausalLM]:\n{code_part}")
        print("-" * 55)


if __name__ == "__main__":
    main()
