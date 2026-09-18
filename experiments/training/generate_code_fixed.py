"""
Script: generate_code_fixed.py
Propósito: Inferencia corregida de HoloCausalLM cargando el checkpoint entrenado,
con tracking estricto de posición temporal y penalización de repetición.
"""

import torch
import tiktoken
from src.models.holo_causal_lm import HoloCausalLM


def generate_clean(
    model: HoloCausalLM,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.4,
    repetition_penalty: float = 1.3
) -> str:
    model.eval()
    prompt_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    current_tokens = list(prompt_ids)

    with torch.no_grad():
        # 1. Prefill del prompt completo (posiciones reales 0..T-1)
        prompt_tensor = torch.tensor([current_tokens], dtype=torch.long)
        logits, states = model(prompt_tensor)

        # 2. Generación autoregresiva O(1) con posición real
        for _ in range(max_new_tokens):
            last_logits = logits[0, -1, :].clone()

            # Aplicar Penalización de Repetición sobre tokens ya emitidos
            for prev_token in set(current_tokens):
                if last_logits[prev_token] > 0:
                    last_logits[prev_token] /= repetition_penalty
                else:
                    last_logits[prev_token] *= repetition_penalty

            # Temperatura
            last_logits = last_logits / max(temperature, 1e-4)

            # Top-K
            top_k = 25
            v, _ = torch.topk(last_logits, top_k)
            last_logits[last_logits < v[-1]] = -float('Inf')

            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == 50256:  # <|endoftext|>
                break

            current_tokens.append(next_token)

            # Siguiente paso con tracking de posición real
            cur_pos = len(current_tokens) - 1
            x_step = model.token_emb(torch.tensor([[next_token]])) + model.pos_emb(torch.tensor([[cur_pos]]))

            next_states = []
            h = x_step
            for block, s in zip(model.blocks, states):
                h, next_s = block.step(h, s)
                next_states.append(next_s)

            states = next_states
            logits = model.lm_head(model.ln_f(h))

    full_output = enc.decode(current_tokens)
    return full_output[len(prompt):].strip()


def main():
    enc = tiktoken.get_encoding("gpt2")
    model = HoloCausalLM(vocab_size=50257, dim=192, depth=2, num_heads=4, max_seq_len=256)
    model.load_state_dict(torch.load("checkpoints/holo_causal_code_v1.pt", map_location="cpu"))
    print("✔ Checkpoint de 10.5M parámetros cargado con éxito.")

    test_prompts = [
        "Task: Write a function to check if a number is even.\nCode:\n",
        "Task: Create a python function to add two numbers.\nCode:\n",
        "Task: Write a python function to return the square of a number.\nCode:\n"
    ]

    print("\n=======================================================")
    print("  GENERACIÓN DE CÓDIGO CORREGIDA (HoloCausalLM 10.5M)  ")
    print("=======================================================")

    for p in test_prompts:
        code = generate_clean(model, enc, p, max_new_tokens=35, temperature=0.35, repetition_penalty=1.35)
        print(f"\n[Prompt]:\n{p.strip()}")
        print(f"[Código Generado]:\n{code}")
        print("-" * 55)


if __name__ == "__main__":
    main()
