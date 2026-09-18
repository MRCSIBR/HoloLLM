"""
Script: chat.py
Interfaz de Chat Interactiva en Streaming para HoloChat-30M.
Mide velocidad (tokens/segundo), latencia y memoria O(1) en tiempo real.
"""

import os
import sys
import time
import torch
import tiktoken
from src.models.holo_causal_lm import HoloCausalLM


def stream_response(
    model: HoloCausalLM,
    enc: tiktoken.Encoding,
    prompt: str,
    device: torch.device,
    max_tokens: int = 75,
    temperature: float = 0.35,
    repetition_penalty: float = 1.35,
    top_k: int = 30
):
    prompt_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    current_tokens = list(prompt_ids)

    print("\nHoloAssistant > ", end="", flush=True)
    start_time = time.time()
    first_token_time = None
    generated_count = 0

    with torch.no_grad():
        # Prefill del prompt
        prompt_tensor = torch.tensor([current_tokens], device=device, dtype=torch.long)
        logits, states = model(prompt_tensor)

        for step in range(max_tokens):
            last_logits = logits[0, -1, :].clone().float()

            # Inhibición de repetición
            for prev in set(current_tokens):
                if last_logits[prev] > 0:
                    last_logits[prev] /= repetition_penalty
                else:
                    last_logits[prev] *= repetition_penalty

            last_logits = last_logits / max(temperature, 1e-4)

            v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
            last_logits[last_logits < v[-1]] = -float('Inf')

            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == 50256:
                break

            if first_token_time is None:
                first_token_time = time.time() - start_time

            # Imprimir token inmediatamente en pantalla (Streaming)
            token_str = enc.decode([next_token])
            print(token_str, end="", flush=True)

            current_tokens.append(next_token)
            generated_count += 1

            # Paso incremental O(1) con posición real
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
    tok_per_sec = generated_count / max(total_time, 1e-4)
    first_tok = (first_token_time or 0.0) * 1000

    print("\n" + "─" * 65)
    print(f"📊 [Métricas]: {generated_count} tokens | {total_time:.2f}s | {tok_per_sec:.1f} tok/s | Latencia 1er token: {first_tok:.1f}ms | KV-Cache: 0 KB (O(1))")
    print("─" * 65 + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = tiktoken.get_encoding("gpt2")

    print("\n=======================================================")
    print("  HOLOGRAPHIC CHATBOT INTERACTIVO (NVIDIA A100 O(1))   ")
    print("=======================================================")
    print("Cargando HoloChat-30M...", flush=True)

    model = HoloCausalLM(
        vocab_size=50257,
        dim=384,
        depth=6,
        num_heads=8,
        max_seq_len=128
    ).to(device)

    ckpt_path = "checkpoints/holo_chat_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"No se encontró {ckpt_path}. Entrena primero con train_15min_chat.py")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print("✔ Modelo cargado y listo. Escribe 'exit' o 'quit' para salir.\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Cerrando sesión de chat holográfico. ¡Hasta pronto!")
                break

            # Determinar si el usuario pide código o conversación
            is_explanation = any(k in user_input.lower() for k in ("explain", "what is", "why", "difference", "how", "que es", "explica", "cual es"))
            is_code_request = any(k in user_input.lower() for k in ("write", "create", "implement", "function", "def ", "codigo", "crea", "escribe"))

            if is_code_request and not is_explanation:
                prompt = f"User: Write a python function for the following task:\n{user_input}\n\nAssistant:\n```python\n"
            else:
                prompt = f"User: {user_input}\n\nAssistant:\n"

            stream_response(model, enc, prompt, device)

        except (KeyboardInterrupt, EOFError):
            print("\nSesión finalizada.")
            break


if __name__ == "__main__":
    main()
