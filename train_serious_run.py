"""
Script: train_serious_run.py
Entrenamiento de Producción HoloLLM-65M en NVIDIA A100.
Corpus: HuggingFaceTB/smollm-corpus (Cosmopedia-v2 + Python-Edu).
Memoria O(1) con cabezas unitarias de Tony Plate (d_h = 64).
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


class DualDataStreamBuffer:
    """Intercala de forma balanceada texto educativo (Cosmopedia) y código Python (Python-Edu)."""
    def __init__(self, enc, prefetch_target: int = 80000):
        self.enc = enc
        self.prefetch_target = prefetch_target
        self.buffer = []

        print("Conectando stream 1: 'cosmopedia-v2' (razonamiento/inglés)...", flush=True)
        self.ds_text = iter(load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True))

        print("Conectando stream 2: 'python-edu' (código/algoritmos)...", flush=True)
        self.ds_code = iter(load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True))

    def _refill(self):
        while len(self.buffer) < self.prefetch_target:
            # 1. Muestra de texto general
            try:
                item_t = next(self.ds_text)
            except StopIteration:
                self.ds_text = iter(load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True))
                item_t = next(self.ds_text)
            text_str = item_t.get("text", "").strip()
            if text_str:
                self.buffer.extend(self.enc.encode(text_str + "<|endoftext|>", allowed_special={"<|endoftext|>"}))

            # 2. Muestra de código Python
            try:
                item_c = next(self.ds_code)
            except StopIteration:
                self.ds_code = iter(load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True))
                item_c = next(self.ds_code)
            code_str = item_c.get("text", "").strip()
            if code_str:
                self.buffer.extend(self.enc.encode(code_str + "<|endoftext|>", allowed_special={"<|endoftext|>"}))

    def get_batch(self, batch_size: int, seq_len: int, device: torch.device):
        needed = batch_size * (seq_len + 1)
        if len(self.buffer) < needed:
            self._refill()

        batch_tokens = self.buffer[:needed]
        self.buffer = self.buffer[needed:]

        tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device).view(batch_size, seq_len + 1)
        return tensor[:, :-1].contiguous(), tensor[:, 1:].contiguous()


def sample_evaluation(model, enc, device, prompt: str, max_new: int = 70):
    model.eval()
    p_tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    current = list(p_tokens)
    with torch.no_grad():
        t_in = torch.tensor([current], device=device, dtype=torch.long)
        logits, states = model(t_in)
        for _ in range(max_new):
            last_logits = logits[0, -1, :].clone().float()
            for prev in set(current[-30:]):
                if last_logits[prev] > 0:
                    last_logits[prev] /= 1.20
                else:
                    last_logits[prev] *= 1.20

            v, _ = torch.topk(last_logits, 35)
            last_logits[last_logits < v[-1]] = -float('Inf')
            probs = torch.softmax(last_logits / 0.65, dim=-1)
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

    # Configuración de Escala Industrial
    DIM = 512
    DEPTH = 10
    HEADS = 8           # d_h = 64 (máxima separación espectral de Plate)
    SEQ_LEN = 256
    BATCH_SIZE = 64     # 64 * 256 = 16,384 tokens por paso
    TARGET_MINUTES = 90.0  # 1.5 horas = ~750M - 800M tokens
    LR_PEAK = 1.2e-3
    WARMUP_STEPS = 400

    print("\n=======================================================", flush=True)
    print("  HOLO-LLM-65M: RUN SERIO DE PRODUCCIÓN (NVIDIA A100)  ", flush=True)
    print("=======================================================", flush=True)
    print(f"Duración programada: {TARGET_MINUTES:.0f} minutos", flush=True)
    print(f"Tokens por lote:     {BATCH_SIZE * SEQ_LEN:,} tokens", flush=True)

    stream_buffer = DualDataStreamBuffer(enc, prefetch_target=80000)

    model = HoloCausalLM(
        vocab_size=50257,
        dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        max_seq_len=SEQ_LEN + 150
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {total_params:,} (~{total_params/1e6:.1f}M) | Capas: {DEPTH} | d_h: {DIM//HEADS}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR_PEAK, weight_decay=1e-2, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss()

    total_seconds = TARGET_MINUTES * 60.0
    start_time = time.time()
    last_print = start_time
    last_eval_time = start_time
    last_save_time = start_time

    step = 0
    total_tokens = 0
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n--- [Train] Iniciando entrenamiento masivo en A100 ---", flush=True)

    while True:
        elapsed = time.time() - start_time
        if elapsed >= total_seconds:
            break

        # Warmup lineal + Cosine decay
        step += 1
        if step < WARMUP_STEPS:
            curr_lr = LR_PEAK * (step / WARMUP_STEPS)
        else:
            progress = (elapsed / total_seconds)
            curr_lr = 1e-4 + 0.5 * (LR_PEAK - 1e-4) * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr

        model.train()
        b_x, b_y = stream_buffer.get_batch(BATCH_SIZE, SEQ_LEN, device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(b_x)
            loss = criterion(logits.reshape(-1, 50257), b_y.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tokens_step = BATCH_SIZE * SEQ_LEN
        total_tokens += tokens_step

        now = time.time()

        # Log cada 25 segundos
        if now - last_print >= 25.0:
            last_print = now
            tok_s = total_tokens / max(elapsed, 1e-4)
            pct = (elapsed / total_seconds) * 100
            print(f"[{elapsed/60:>4.1f}m / {TARGET_MINUTES:.0f}m] ({pct:>4.1f}%) | Paso {step:05d} | Loss: {loss.item():.4f} | LR: {curr_lr:.2e} | ~{tok_s:.0f} tok/s | Total: {total_tokens/1e6:.1f}M toks", flush=True)

        # Evaluación en vivo cada 15 minutos
        if now - last_eval_time >= 900.0:
            last_eval_time = now
            print("\n" + "═" * 70, flush=True)
            print(f"🔍 [EVALUACIÓN EN VIVO - {elapsed/60:.1f} MINUTOS ({total_tokens/1e6:.1f}M TOKENS)]:", flush=True)
            p1 = sample_evaluation(model, enc, device, "def calculate_factorial(n):\n    \"\"\"Calculate the factorial of n.\"\"\"\n")
            print(f"\n[Prompt 1: Factorial en Python]:\n{p1.strip()}\n")
            p2 = sample_evaluation(model, enc, device, "The principle of quantum physics states that particles")
            print(f"\n[Prompt 2: Física / Inglés]:\n{p2.strip()}")
            print("═" * 70 + "\n", flush=True)

        # Checkpoint de respaldo cada 20 minutos
        if now - last_save_time >= 1200.0:
            last_save_time = now
            ckpt_path = f"checkpoints/holo_65m_step_{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Checkpoint de seguridad guardado en: {ckpt_path}", flush=True)

    # Checkpoint final
    final_path = "checkpoints/holo_65m_final.pt"
    torch.save(model.state_dict(), final_path)
    total_time = time.time() - start_time
    print("\n=======================================================", flush=True)
    print(f"  RUN FINALIZADO CON ÉXITO EN {total_time/60:.1f} MINUTOS  ", flush=True)
    print(f"  Tokens Totales Ingeridos: {total_tokens:,} (~{total_tokens/1e6:.1f}M)", flush=True)
    print(f"  Checkpoint final guardado en: {final_path}", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    main()
