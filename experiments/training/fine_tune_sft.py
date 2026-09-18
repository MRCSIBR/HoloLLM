"""
Script: fine_tune_sft.py
Propósito: Supervised Fine-Tuning (SFT) sobre HoloLLM-65M pre-entrenado (451M tokens).
Convierte el modelo base en un Asistente que responde con precisión a la orden exacta.
"""

import os
import json
import time
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken

from src.models.holo_causal_lm import HoloCausalLM
from src.utils.seed import enforce_reproducibility

BASE_CKPT = "checkpoints/holo_65m_final.pt"
SFT_CKPT = "checkpoints/holo_65m_instruct.pt"
ALPACA_URL = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/master/alpaca_data_cleaned.json"
CODE_URL = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"


def prepare_sft_dataset(enc, max_samples: int = 4000, max_len: int = 140):
    print("Preparando dataset curado de instrucciones de código y ciencia...", flush=True)
    os.makedirs("data", exist_ok=True)
    
    # 1. Muestras de código estricto
    code_path = "data/code_alpaca_full.json"
    if not os.path.exists(code_path):
        req = urllib.request.Request(CODE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            code_data = json.loads(resp.read().decode("utf-8"))
        with open(code_path, "w") as f:
            json.dump(code_data, f)
    else:
        with open(code_path, "r") as f:
            code_data = json.load(f)

    # 2. Muestras de razonamiento/ciencia
    chat_path = "data/alpaca_dialogue.json"
    if not os.path.exists(chat_path):
        req = urllib.request.Request(ALPACA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            chat_data = json.loads(resp.read().decode("utf-8"))
        with open(chat_path, "w") as f:
            json.dump(chat_data, f)
    else:
        with open(chat_path, "r") as f:
            chat_data = json.load(f)

    all_pairs = []
    
    # Pares de Código conciso
    for item in code_data:
        inst = item.get("instruction", "").strip()
        code = item.get("output", "").strip()
        if "def " in code and 20 < len(code) < 220:
            all_pairs.append({
                "prompt": f"User: Write a python function for the following task:\n{inst}\n\nAssistant:\n```python\n",
                "response": f"{code}\n```<|endoftext|>"
            })
        if len(all_pairs) >= max_samples // 2:
            break

    # Pares de Explicación y Ciencia concisa
    for item in chat_data:
        inst = item.get("instruction", "").strip()
        out = item.get("output", "").strip()
        if any(k in inst.lower() for k in ("explain", "what is", "why", "how", "difference", "define")):
            if 25 < len(out) < 220:
                all_pairs.append({
                    "prompt": f"User: {inst}\n\nAssistant:\n",
                    "response": f"{out}<|endoftext|>"
                })
        if len(all_pairs) >= max_samples:
            break

    print(f"Tokenizando {len(all_pairs)} pares de alineación estricta...", flush=True)
    all_inputs, all_labels = [], []

    for item in all_pairs:
        p_ids = enc.encode(item["prompt"], allowed_special={"<|endoftext|>"})
        r_ids = enc.encode(item["response"], allowed_special={"<|endoftext|>"})

        seq = p_ids + r_ids
        if len(seq) > max_len:
            seq = seq[:max_len]
            seq[-1] = 50256

        # Loss masking: El modelo NO se penaliza por el prompt (-100)
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

    DIM = 512
    DEPTH = 10
    HEADS = 8
    MAX_LEN = 256 + 150
    BATCH_SIZE = 32
    EPOCHS = 8          # SFT requiere pocas épocas para no destruir las representaciones base
    LR = 3.0e-4         # Learning rate bajo para afinar sin borrar el conocimiento de 451M

    print("\n=======================================================", flush=True)
    print("  SUPERVISED FINE-TUNING (SFT) SOBRE HOLOLLM-65M       ", flush=True)
    print("=======================================================", flush=True)

    model = HoloCausalLM(
        vocab_size=50257,
        dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        max_seq_len=MAX_LEN
    ).to(device)

    print(f"Cargando pesos pre-entrenados de 451M tokens desde {BASE_CKPT}...", flush=True)
    model.load_state_dict(torch.load(BASE_CKPT, map_location=device))
    print("✔ Pesos base cargados con éxito.", flush=True)

    inputs, labels = prepare_sft_dataset(enc, max_samples=3500, max_len=140)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"\n--- Iniciando SFT de precisión (8 épocas sobre {len(inputs)} instrucciones) ---", flush=True)
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
        ep_loss = tot_loss / len(inputs)
        ep_time = time.time() - ep_start
        print(f"Época [{epoch:02d}/{EPOCHS:02d}] | SFT Loss: {ep_loss:.4f} | Tiempo: {ep_time:.1f}s/época", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), SFT_CKPT)
    total_time = time.time() - start_time
    print(f"\n✔ Modelo Alineado (Instruct) guardado con éxito en: {SFT_CKPT} (Tiempo: {total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
