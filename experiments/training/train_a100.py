"""
Script: train_a100.py
Entrenamiento de Alta Velocidad en NVIDIA A100 con atención holográfica paralelizada.
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

DATA_URL = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"
DATA_FILE = "data/code_alpaca_full.json"


def download_full_dataset() -> List[dict]:
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print("Descargando corpus completo de CodeAlpaca (20,000 pares)...", flush=True)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"✔ Dataset guardado en {DATA_FILE}", flush=True)
    else:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data


def prepare_dataset(enc, max_samples: int = 5000, max_len: int = 128):
    data = download_full_dataset()
    print(f"Tokenizando {max_samples} muestras para la A100...", flush=True)

    all_inputs, all_labels = [], []
    valid_count = 0

    for item in data:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        code = item.get("output", "").strip()

        if not code or len(code) < 15:
            continue

        if inp:
            prompt = f"### Task:\n{inst}\nContext:\n{inp}\n\n### Code:\n"
        else:
            prompt = f"### Task:\n{inst}\n\n### Code:\n"

        code_str = code + "<|endoftext|>"

        p_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
        c_ids = enc.encode(code_str, allowed_special={"<|endoftext|>"})

        seq = p_ids + c_ids
        if len(seq) > max_len:
            seq = seq[:max_len]
            seq[-1] = 50256

        labels = [-100] * min(len(p_ids), len(seq)) + seq[len(p_ids):]

        all_inputs.append(torch.tensor(seq, dtype=torch.long))
        all_labels.append(torch.tensor(labels, dtype=torch.long))
        valid_count += 1

        if valid_count >= max_samples:
            break

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
    print(f"\n=======================================================", flush=True)
    print(f"  HOLO-CODE LM EN NVIDIA A100 (PARALELO + BFLOAT16)    ", flush=True)
    print(f"=======================================================", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    enc = tiktoken.get_encoding("gpt2")

    DIM = 384
    HEADS = 8
    DEPTH = 6
    MAX_LEN = 128
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 1.5e-3
    SAMPLES = 5000

    model = HoloCausalLM(
        vocab_size=50257,
        dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        max_seq_len=MAX_LEN
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {total_params:,} (~{total_params/1e6:.1f}M) | Capas: {DEPTH}", flush=True)

    inputs, labels = prepare_dataset(enc, max_samples=SAMPLES, max_len=MAX_LEN)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"\n--- [Train] Lanzando optimización paralela ({len(inputs)} funciones) ---", flush=True)
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
        total_elapsed = time.time() - start_time
        tok_sec = (len(inputs) * MAX_LEN) / max(ep_time, 1e-4)

        print(f"Época [{epoch:02d}/{EPOCHS:02d}] | Loss: {avg_loss:.4f} | Tiempo: {ep_time:.2f}s/época | ~{tok_sec:.0f} tok/s", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/holo_code_a100_35m.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✔ Checkpoint guardado exitosamente en: {ckpt_path}", flush=True)

    # Evaluación y depuración en vivo
    print("\n=======================================================", flush=True)
    print("  EVALUACIÓN EN VIVO: CÓDIGO Y DEPURACIÓN EN LA A100   ", flush=True)
    print("=======================================================", flush=True)

    test_prompts = [
        "### Task:\nWrite a python function to calculate the factorial of a number.\n\n### Code:\n",
        "### Task:\nCreate a function that reverses a string.\n\n### Code:\n",
        "### Task:\nWrite a function to check if a number is prime.\n\n### Code:\n",
        "### Task:\nFix this broken function:\ndef sum_list(lst):\n    total = 0\n    for x in lst\n        total += x\n    return totl\n\n### Code:\n"
    ]

    model.eval()
    for p in test_prompts:
        p_ids = enc.encode(p)
        gen_tokens = model.generate(
            prompt_tokens=p_ids,
            max_new_tokens=60,
            temperature=0.3,
            top_k=30
        )
        full_text = enc.decode(gen_tokens)
        code_part = full_text[len(p):].strip()
        print(f"\n[Prompt]:\n{p.strip()}", flush=True)
        print(f"[Código Generado por HoloCodeLM]:\n{code_part}", flush=True)
        print("-" * 60, flush=True)


if __name__ == "__main__":
    main()
