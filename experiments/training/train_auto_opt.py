"""
Script: train_auto_opt.py
Propósito: Optimización científica con Train/Val split, Early Stopping
y guardado automático del mejor checkpoint (HoloChat-30M en A100).
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken

from src.models.holo_causal_lm import HoloCausalLM
from src.utils.seed import enforce_reproducibility
from train_15min_chat import prepare_hybrid_dataset


def evaluate_val_loss(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(bx)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = by[:, 1:].contiguous()
                loss = criterion(shift_logits.view(-1, 50257), shift_labels.view(-1))
            total_loss += loss.item() * bx.size(0)
            total_tokens += bx.size(0)
    return total_loss / total_tokens


def quick_sample_check(model, enc, device, prompt):
    model.eval()
    p_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    current = list(p_ids)
    with torch.no_grad():
        t_in = torch.tensor([current], device=device, dtype=torch.long)
        logits, states = model(t_in)
        for _ in range(35):
            last_logits = logits[0, -1, :].clone().float()
            for prev in set(current):
                if last_logits[prev] > 0:
                    last_logits[prev] /= 1.35
                else:
                    last_logits[prev] *= 1.35
            v, _ = torch.topk(last_logits, 20)
            last_logits[last_logits < v[-1]] = -float('Inf')
            probs = torch.softmax(last_logits / 0.35, dim=-1)
            next_t = torch.multinomial(probs, 1).item()
            if next_t == 50256:
                break
            current.append(next_t)
            cur_pos = len(current) - 1
            x_step = model.token_emb(torch.tensor([[next_t]], device=device)) + model.pos_emb(torch.tensor([[cur_pos]], device=device))
            next_states = []
            h = x_step
            for block, s in zip(model.blocks, states):
                h, next_s = block.step(h, s)
                next_states.append(next_s)
            states = next_states
            logits = model.lm_head(model.ln_f(h))
    return enc.decode(current)[len(prompt):].strip()


def main():
    enforce_reproducibility(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = tiktoken.get_encoding("gpt2")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    DIM = 384
    HEADS = 8
    DEPTH = 6
    MAX_LEN = 128
    BATCH_SIZE = 64
    MAX_EPOCHS = 50
    LR = 1.2e-3

    print("\n=======================================================", flush=True)
    print("  OPTIMIZACIÓN CIENTÍFICA HOLOCHAT (TRAIN / VAL SPLIT) ", flush=True)
    print("=======================================================", flush=True)

    inputs, labels = prepare_hybrid_dataset(enc, max_total_samples=10000, max_len=MAX_LEN)

    # 1. División Train (90%) y Validación (10%)
    total_samples = len(inputs)
    val_size = int(total_samples * 0.10)
    train_size = total_samples - val_size

    full_dataset = torch.utils.data.TensorDataset(inputs, labels)
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    print(f"Dataset total: {total_samples} | Train: {train_size} | Validación (Prueba ciega): {val_size}", flush=True)

    model = HoloCausalLM(
        vocab_size=50257,
        dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        max_seq_len=MAX_LEN
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float('inf')
    best_epoch = 0
    start_time = time.time()

    print(f"\n--- Iniciando ciclo de hasta {MAX_EPOCHS} épocas con guardado del mejor modelo ---", flush=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tot_loss = 0.0

        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model(bx)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = by[:, 1:].contiguous()
                loss = criterion(shift_logits.view(-1, 50257), shift_labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += loss.item() * bx.size(0)

        scheduler.step()
        train_loss = tot_loss / train_size
        val_loss = evaluate_val_loss(model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        # Detectar nuevo récord en validación
        is_best = val_loss < best_val_loss
        marker = "✔ MEJOR MODELO" if is_best else ""

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/holo_chat_best.pt")

        print(f"Época [{epoch:02d}/{MAX_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {marker}", flush=True)

        # Cada 10 épocas, mostrar un preview en vivo de cómo programa
        if epoch % 10 == 0:
            sample_code = quick_sample_check(
                model, enc, device,
                "User: Write a python function for the following task:\nCreate a function that returns only even numbers from a list.\n\nAssistant:\n```python\n"
            )
            print(f"   [Preview Código Época {epoch}]: {sample_code.replace(chr(10), ' ')}", flush=True)

    print("\n=======================================================", flush=True)
    print(f"  ENTRENAMIENTO FINALIZADO CON ÉXITO EN {elapsed/60:.1f} MINUTOS  ", flush=True)
    print(f"  Mejor Época: {best_epoch:02d} con Validation Loss: {best_val_loss:.4f}", flush=True)
    print(f"  Checkpoint guardado en: checkpoints/holo_chat_best.pt", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    main()
