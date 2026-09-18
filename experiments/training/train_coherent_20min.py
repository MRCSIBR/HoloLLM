"""
Script: train_coherent_20min.py
Propósito: Entrenamiento de 20 minutos en A100 sobre TinyStories (150M tokens)
con buffer contiguo ultra-rápido y generación periódica en vivo.
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken
from datasets import load_dataset

from src.models.holo_causal_lm import HoloCausalLM
from src.utils.seed import enforce_reproducibility


class FastTokenBuffer:
    """Búfer en memoria de alta velocidad para alimentar a la A100 sin latencia de Python."""
    def __init__(self, enc, dataset_stream, prefetch_target: int = 50000):
        self.enc = enc
        self.stream_iter = iter(dataset_stream)
        self.prefetch_target = prefetch_target
        self.buffer = []

    def _refill(self):
        while len(self.buffer) < self.prefetch_target:
            try:
                item = next(self.stream_iter)
            except StopIteration:
                self.stream_iter = iter(load_dataset("roneneldan/TinyStories", split="train", streaming=True))
                item = next(self.stream_iter)
            text = item.get("text", "").strip()
            if text:
                toks = self.enc.encode(text + "<|endoftext|>", allowed_special={"<|endoftext|>"})
                self.buffer.extend(toks)

    def get_batch(self, batch_size: int, seq_len: int, device: torch.device):
        needed = batch_size * (seq_len + 1)
        if len(self.buffer) < needed:
            self._refill()

        batch_tokens = self.buffer[:needed]
        self.buffer = self.buffer[needed:]

        tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device).view(batch_size, seq_len + 1)
        # .contiguous() garantiza compatibilidad total con view/reshape
        return tensor[:, :-1].contiguous(), tensor[:, 1:].contiguous()


def sample_story(model, enc, device, prompt: str, max_new: int = 75):
    model.eval()
    p_tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    current = list(p_tokens)
    with torch.no_grad():
        t_in = torch.tensor([current], device=device, dtype=torch.long)
        logits, states = model(t_in)
        for _ in range(max_new):
            last_logits = logits[0, -1, :].clone().float()
            # Penalización de repetición moderada para narrativa fluida
            for prev in set(current[-30:]):
                if last_logits[prev] > 0:
                    last_logits[prev] /= 1.20
                else:
                    last_logits[prev] *= 1.20

            # Top-K + Temperature para prosa creativa y coherente
            v, _ = torch.topk(last_logits, 35)
            last_logits[last_logits < v[-1]] = -float('Inf')
            probs = torch.softmax(last_logits / 0.7, dim=-1)
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

    return enc.decode(current)


def main():
    enforce_reproducibility(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    enc = tiktoken.get_encoding("gpt2")

    DIM = 384
    HEADS = 8
    DEPTH = 6
    SEQ_LEN = 256
    BATCH_SIZE = 64
    TARGET_MINUTES = 20.0
    LR = 1.0e-3

    print("\n=======================================================", flush=True)
    print("  PRE-ENTRENAMIENTO HOLOGRÁFICO CONTINUO (TINYSTORIES) ", flush=True)
    print("=======================================================", flush=True)
    print(f"Objetivo: 20 minutos de streaming continuo en A100 (~150M tokens)", flush=True)
    print(f"Secuencia: {SEQ_LEN} tokens | Batch: {BATCH_SIZE} (16,384 tokens/paso)", flush=True)

    print("Conectando stream de 'roneneldan/TinyStories'...", flush=True)
    dataset_stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    data_buffer = FastTokenBuffer(enc, dataset_stream, prefetch_target=60000)

    model = HoloCausalLM(
        vocab_size=50257,
        dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        max_seq_len=SEQ_LEN + 150
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros del Modelo: {total_params:,} (~{total_params/1e6:.1f}M)", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss()

    total_seconds = TARGET_MINUTES * 60.0
    start_time = time.time()
    last_print = start_time
    last_sample_time = start_time

    step = 0
    total_tokens_processed = 0

    print(f"\n--- Iniciando pre-entrenamiento ininterrumpido (0 a {TARGET_MINUTES:.0f} min) ---", flush=True)

    while True:
        elapsed = time.time() - start_time
        if elapsed >= total_seconds:
            break

        model.train()
        b_x, b_y = data_buffer.get_batch(BATCH_SIZE, SEQ_LEN, device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(b_x)
            # .reshape() y .contiguous() eliminan cualquier fallo de memoria
            loss = criterion(logits.reshape(-1, 50257), b_y.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        tokens_in_step = BATCH_SIZE * SEQ_LEN
        total_tokens_processed += tokens_in_step

        now = time.time()
        if now - last_print >= 20.0:
            last_print = now
            tok_s = total_tokens_processed / max(elapsed, 1e-4)
            pct = (elapsed / total_seconds) * 100
            print(f"[{elapsed/60:>4.1f} min / {TARGET_MINUTES:.0f} min] ({pct:>4.1f}%) | Paso {step:05d} | Loss: {loss.item():.4f} | ~{tok_s:.0f} tok/s | Tokens: {total_tokens_processed/1e6:.1f}M", flush=True)

        # Generar una historia cada 4 minutos para escuchar la evolución del modelo
        if now - last_sample_time >= 240.0:
            last_sample_time = now
            print("\n" + "─" * 60, flush=True)
            print(f"📖 [Preview en vivo a los {elapsed/60:.1f} minutos]:", flush=True)
            preview = sample_story(model, enc, device, "Once upon a time, there was a little robot named Holo who")
            print(preview.strip(), flush=True)
            print("─" * 60 + "\n", flush=True)

    total_elapsed = time.time() - start_time
    print("\n=======================================================", flush=True)
    print(f"  PRE-ENTRENAMIENTO COMPLETADO EN {total_elapsed/60:.1f} MINUTOS  ", flush=True)
    print(f"  Total de Tokens Ingeridos: {total_tokens_processed:,} (~{total_tokens_processed/1e6:.1f}M)", flush=True)
    print(f"  Velocidad Promedio: {total_tokens_processed/total_elapsed:.0f} tokens/segundo", flush=True)
    print("=======================================================", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/holo_coherent_28m.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✔ Checkpoint guardado en: {ckpt_path}", flush=True)

    # Pruebas finales de coherencia
    print("\n=======================================================", flush=True)
    print("  EVALUACIÓN DE INGLÉS COHERENTE Y DIÁLOGO             ", flush=True)
    print("=======================================================", flush=True)

    test_prompts = [
        "Once upon a time, a little girl named Lily found a magic key in the garden.",
        "One sunny day, Tom and his puppy were playing in the park when",
        "\"Look at that big tree!\" said Max. \"I wonder what is at the top.\""
    ]

    for p in test_prompts:
        story = sample_story(model, enc, device, p, max_new=80)
        print(f"\n[Prompt]: {p}\n[Continuación de HoloLLM]:\n{story}")
        print("-" * 65, flush=True)


if __name__ == "__main__":
    main()
