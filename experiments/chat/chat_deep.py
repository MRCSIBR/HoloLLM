"""
Script: chat_deep.py
Consola interactiva en streaming para HoloLLM-DeepDistilled (66.4M)
con pesos destilados de Qwen2.5-Coder, vocabulario de 151,936 tokens y memoria O(1).
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer
from src.models.holo_causal_lm import HoloCausalLM

TEACHER_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CKPT_PATH = "checkpoints/holo_deep_distilled_75m.pt"


def stream_response(
    model: HoloCausalLM,
    tokenizer,
    user_prompt: str,
    device: torch.device,
    max_tokens: int = 70,
    temperature: float = 0.25,
    top_k: int = 25,
    repetition_penalty: float = 1.25
):
    full_prompt = f"<|im_start|>user\nWrite a python function for the following task:\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n```python\n"
    prompt_ids = tokenizer.encode(full_prompt)
    current_tokens = list(prompt_ids)

    print("\nHoloDeep > ```python\n", end="", flush=True)
    start_time = time.time()
    first_tok_time = None
    gen_count = 0
    eos_ids = {151645, 151643}  # <|im_end|>, <|endoftext|>

    with torch.no_grad():
        prompt_tensor = torch.tensor([current_tokens], device=device, dtype=torch.long)
        logits, states = model(prompt_tensor)

        for _ in range(max_tokens):
            last_logits = logits[0, -1, :].clone().float()

            for prev in set(current_tokens[-30:]):
                if last_logits[prev] > 0:
                    last_logits[prev] /= repetition_penalty
                else:
                    last_logits[prev] *= repetition_penalty

            last_logits = last_logits / max(temperature, 1e-4)

            v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
            last_logits[last_logits < v[-1]] = -float('Inf')

            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token in eos_ids:
                break

            if first_tok_time is None:
                first_tok_time = time.time() - start_time

            token_str = tokenizer.decode([next_token])
            print(token_str, end="", flush=True)

            current_tokens.append(next_token)
            gen_count += 1

            cur_pos = len(current_tokens) - 1
            x_step = model.token_emb(torch.tensor([[next_token]], device=device)) + model.pos_emb(torch.tensor([[cur_pos]], device=device))

            next_states = []
            h = x_step
            for block, s in zip(model.blocks, states):
                h, next_s = block.step(h, s)
                next_states.append(next_s)

            states = next_states
            logits = model.lm_head(model.ln_f(h))

    total_time = time.time() - start_time
    tok_s = gen_count / max(total_time, 1e-4)
    first_ms = (first_tok_time or 0.0) * 1000

    print("\n" + "─" * 65)
    print(f"📊 [Métricas]: {gen_count} tokens | {total_time:.2f}s | {tok_s:.1f} tok/s | TTFT: {first_ms:.1f}ms | KV-Cache: 0 KB (O(1))")
    print("─" * 65 + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCargando modelo destilado en: {torch.cuda.get_device_name(0)}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)

    # Configuración adaptada a deep_distill_a100.py (140 + 32 = 172)
    model = HoloCausalLM(
        vocab_size=151936,
        dim=384,
        depth=6,
        num_heads=8,
        max_seq_len=172
    ).to(device)

    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print("✔ HoloLLM-DeepDistilled listo. Escribe una tarea de programación (o 'exit'):\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break
            stream_response(model, tokenizer, user_input, device)
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
