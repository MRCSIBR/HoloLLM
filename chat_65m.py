"""
Script: chat_65m.py
Consola Interactiva de Streaming en vivo para HoloLLM-65M (451M tokens).
Memoria constante O(1), baja latencia y métricas de generación en tiempo real.
"""

import os
import sys
import time
import torch
import tiktoken
from src.models.holo_causal_lm import HoloCausalLM

CKPT_PATH = "checkpoints/holo_65m_instruct.pt"

def stream_holo_response(
    model: HoloCausalLM,
    enc: tiktoken.Encoding,
    user_prompt: str,
    device: torch.device,
    max_tokens: int = 120,
    temperature: float = 0.45,
    top_k: int = 35,
    repetition_penalty: float = 1.25
):
    # Formato pedagógico consistente con Cosmopedia y Python-Edu
    prompt_ids = enc.encode(user_prompt, allowed_special={"<|endoftext|>"})
    current_tokens = list(prompt_ids)

    print("\nHoloLLM-65M > ", end="", flush=True)
    start_time = time.time()
    first_tok_time = None
    gen_count = 0

    with torch.no_grad():
        prompt_tensor = torch.tensor([current_tokens], device=device, dtype=torch.long)
        logits, states = model(prompt_tensor)

        for step in range(max_tokens):
            last_logits = logits[0, -1, :].clone().float()

            # Inhibición de repetición moderada para no cortar ideas complejas
            for prev in set(current_tokens[-35:]):
                if last_logits[prev] > 0:
                    last_logits[prev] /= repetition_penalty
                else:
                    last_logits[prev] *= repetition_penalty

            last_logits = last_logits / max(temperature, 1e-4)

            v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
            last_logits[last_logits < v[-1]] = -float('Inf')

            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == 50256:  # <|endoftext|>
                break

            if first_tok_time is None:
                first_tok_time = time.time() - start_time

            token_str = enc.decode([next_token])
            print(token_str, end="", flush=True)

            current_tokens.append(next_token)
            gen_count += 1

            # Paso incremental O(1) con tracking de posición real
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
    enc = tiktoken.get_encoding("gpt2")

    print("\n=======================================================", flush=True)
    print("  HOLO-LLM-65M: CHAT INTERACTIVO (451M TOKENS EN A100) ", flush=True)
    print("=======================================================", flush=True)

    # Configuración de producción exacta del checkpoint final
    DIM = 512
    DEPTH = 10
    HEADS = 8
    MAX_LEN = 256 + 150

    model = HoloCausalLM(
        vocab_size=50257,
        dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        max_seq_len=MAX_LEN
    ).to(device)

    if not os.path.exists(CKPT_PATH):
        print(f"No se encontró el checkpoint en {CKPT_PATH}")
        return

    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✔ Modelo HoloLLM-65M cargado ({total_params:,} parámetros en {device}).")
    print("Escribe una instrucción de código o una pregunta científica (o 'exit'):\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            # Prompt adaptado a la distribución educativa del entrenamiento
            if any(k in user_input.lower() for k in ("def ", "code", "function", "python", "algorithm", "implement", "write a")):
                prompt = f"# Python Implementation:\n# Task: {user_input}\n"
            else:
                prompt = f"Question: {user_input}\n\nAnswer: "

            stream_holo_response(model, enc, prompt, device)

        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
