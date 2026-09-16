"""
Script: train_15min_chat.py
Propósito: Entrenamiento de 15 minutos en NVIDIA A100 de un modelo híbrido
Conversacional (Inglés) + Python Puro con atención holográfica O(1).
"""

import os
import json
import time
import urllib.request
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken

from src.models.holo_causal_lm import HoloCausalLM
from src.utils.seed import enforce_reproducibility

ALPACA_URL = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/master/alpaca_data_cleaned.json"
CODE_FILE = "data/code_alpaca_full.json"


def prepare_hybrid_dataset(enc, max_total_samples: int = 10000, max_len: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    print("--- [1/2] Preparando dataset híbrido: Conversación + Python Puro ---", flush=True)

    # 1. Cargar Código y filtrar estrictamente a Python (sin JS/C++)
    with open(CODE_FILE, "r", encoding="utf-8") as f:
        code_raw = json.load(f)

    python_pairs = []
    for item in code_raw:
        code = item.get("output", "").strip()
        inst = item.get("instruction", "").strip()
        # Filtro estricto: Python sintáctico sin llaves ni variables JS
        if "def " in code and ":" in code and "{" not in code and "var " not in code and "function " not in code:
            if 20 < len(code) < 300:
                python_pairs.append({
                    "prompt": f"User: Write a python function for the following task:\n{inst}\n\nAssistant:\n```python\n",
                    "response": f"{code}\n```<|endoftext|>"
                })
        if len(python_pairs) >= max_total_samples // 2:
            break

    print(f"✔ Muestras de Python Puro filtradas: {len(python_pairs)}", flush=True)

    # 2. Descargar Diálogo y Razonamiento General en Inglés (Alpaca Cleaned)
    alpaca_file = "data/alpaca_dialogue.json"
    if not os.path.exists(alpaca_file):
        print("Descargando corpus de diálogo instructivo en inglés...", flush=True)
        req = urllib.request.Request(ALPACA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            alpaca_raw = json.loads(resp.read().decode("utf-8"))
        with open(alpaca_file, "w", encoding="utf-8") as f:
            json.dump(alpaca_raw, f)
    else:
        with open(alpaca_file, "r", encoding="utf-8") as f:
            alpaca_raw = json.load(f)

    chat_pairs = []
    for item in alpaca_raw:
        inst = item.get("instruction", "").strip()
        out = item.get("output", "").strip()
        inp = item.get("input", "").strip()

        if 15 < len(inst) < 150 and 20 < len(out) < 250:
            if inp:
                prompt_text = f"User: {inst}\nContext: {inp}\n\nAssistant:\n"
            else:
                prompt_text = f"User: {inst}\n\nAssistant:\n"

            chat_pairs.append({
                "prompt": prompt_text,
                "response": f"{out}<|endoftext|>"
            })

        if len(chat_pairs) >= max_total_samples // 2:
            break

    print(f"✔ Muestras de Diálogo y Razonamiento: {len(chat_pairs)}", flush=True)

    # Mezclar 50% / 50%
    combined = []
    for i in range(max(len(python_pairs), len(chat_pairs))):
        if i < len(chat_pairs):
            combined.append(chat_pairs[i])
        if i < len(python_pairs):
            combined.append(python_pairs[i])

    print(f"--- [2/2] Tokenizando {len(combined)} pares híbridos ---", flush=True)
    all_inputs, all_labels = [], []

    for item in combined:
        p_ids = enc.encode(item["prompt"], allowed_special={"<|endoftext|>"})
        r_ids = enc.encode(item["response"], allowed_special={"<|endoftext|>"})

        seq = p_ids + r_ids
        if len(seq) > max_len:
            seq = seq[:max_len]
            seq[-1] = 50256

        # Loss masking: el prompt no se penaliza (-100)
        labels = [-100] * min(len(p_ids), len(seq)) + seq[len(p_ids):]

        all_inputs.append(torch.tensor(seq, dtype=torch.long))
        all_labels.append(torch.tensor(labels, dtype=torch.long))

    m_len = max(len(s) for s in all_inputs)
    pad_inputs = torch.full((len(all_inputs), m_len), 50256, dtype=torch.long)
    pad_labels = torch.full((len(all_labels), m_len), -100, dtype=torch.long)

    for i in range(len(all_inputs)):
        pad_inputs[i, :len(all_inputs[i])] = all_inputs[i]
        pad_labels[i, :len(all_labels[i])] = all_labels[i]

    return pad_inputs, pad_labels


def main():
    enforce_reproducibility(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    enc = tiktoken.get_encoding("gpt2")

    # Hiperparámetros calculados para 12-15 minutos exactos en A100
    DIM = 384
    HEADS = 8
    DEPTH = 6
    MAX_LEN = 128
    BATCH_SIZE = 64
    EPOCHS = 18
    LR = 1.2e-3
    SAMPLES = 10000

    print("\n=======================================================", flush=True)
    print("  ENTRENAMIENTO HÍBRIDO A100 (DIÁLOGO + PYTHON PURO)   ", flush=True)
    print("=======================================================", flush=True)

    model = HoloCausalLM(
        vocab_size=50257,
        dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        max_seq_len=MAX_LEN
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {total_params:,} (~{total_params/1e6:.1f}M) | Capas: {DEPTH}", flush=True)

    inputs, labels = prepare_hybrid_dataset(enc, max_total_samples=SAMPLES, max_len=MAX_LEN)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"\n--- Lanzando entrenamiento de {EPOCHS} épocas sobre {len(inputs)} pares ---", flush=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss = 0.0
        ep_start = time.time()

        for b_x, b_y in dataloader:
            b_x = b_x.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(b_x)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = b_y[:, 1:].contiguous()
                loss = criterion(shift_logits.view(-1, 50257), shift_labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item() * b_x.size(0)

        scheduler.step()
        avg_loss = tot_loss / len(inputs)
        ep_time = time.time() - ep_start
        total_time = time.time() - start_time
        tok_sec = (len(inputs) * MAX_LEN) / max(ep_time, 1e-4)

        print(f"Época [{epoch:02d}/{EPOCHS:02d}] | Loss: {avg_loss:.4f} | Tiempo: {ep_time:.1f}s | Acumulado: {total_time/60:.1f} min | ~{tok_sec:.0f} tok/s", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/holo_chat_30m.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✔ Checkpoint Híbrido guardado con éxito en: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
