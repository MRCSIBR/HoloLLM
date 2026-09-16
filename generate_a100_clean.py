"""
Script: generate_a100_clean.py
Inferencia de Alta Precisión sobre HoloCodeLM-35M (A100):
1. Rompe la resonancia láser con Repetition Penalty (Inhibición de Pribram).
2. Tracking estricto de la posición temporal (reloj no congelado).
3. Muestreo guiado por temperatura y Top-K.
"""

import os
import torch
import tiktoken
from src.models.holo_causal_lm import HoloCausalLM

def generate_coherent_code(
    model: HoloCausalLM,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.35,
    top_k: int = 30,
    repetition_penalty: float = 1.35
) -> str:
    model.eval()
    device = model.token_emb.weight.device

    prompt_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    current_tokens = list(prompt_ids)

    with torch.no_grad():
        # 1. Prefill del prompt con sus posiciones reales
        prompt_tensor = torch.tensor([current_tokens], device=device, dtype=torch.long)
        logits, states = model(prompt_tensor)

        # 2. Generación token a token
        for step in range(max_new_tokens):
            last_logits = logits[0, -1, :].clone().float()

            # Inhibición Recurrente de Pribram (Repetition Penalty)
            # Penaliza los tokens ya emitidos para impedir que la cavidad holográfica resuene consigo misma
            for prev_tok in set(current_tokens):
                if last_logits[prev_tok] > 0:
                    last_logits[prev_tok] /= repetition_penalty
                else:
                    last_logits[prev_tok] *= repetition_penalty

            # Muestreo con Temperatura
            last_logits = last_logits / max(temperature, 1e-4)

            # Top-K
            v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
            last_logits[last_logits < v[-1]] = -float('Inf')

            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == 50256:  # <|endoftext|>
                break

            current_tokens.append(next_token)

            # Siguiente paso con tracking de posición real en el espaciotiempo de la secuencia
            cur_pos = len(current_tokens) - 1
            x_step = model.token_emb(torch.tensor([[next_token]], device=device)) + model.pos_emb(torch.tensor([[cur_pos]], device=device))

            next_states = []
            h = x_step
            for block, s in zip(model.blocks, states):
                h, next_s = block.step(h, s)
                next_states.append(next_s)

            states = next_states
            logits = model.lm_head(model.ln_f(h))

    full_text = enc.decode(current_tokens)
    return full_text[len(prompt):].strip()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando modelo en: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    enc = tiktoken.get_encoding("gpt2")

    # Cargar el modelo de 27.3M parámetros entrenado en la A100
    model = HoloCausalLM(
        vocab_size=50257,
        dim=384,
        depth=6,
        num_heads=8,
        max_seq_len=128
    ).to(device)

    ckpt_path = "checkpoints/holo_code_a100_35m.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"✔ Checkpoint de 27.3M parámetros cargado con éxito desde {ckpt_path}\n", flush=True)

    test_prompts = [
        "### Task:\nWrite a python function to check if a number is prime.\n\n### Code:\n",
        "### Task:\nWrite a python function to calculate the factorial of a number.\n\n### Code:\n",
        "### Task:\nCreate a function that reverses a string.\n\n### Code:\n",
        "### Task:\nFix this broken function:\ndef sum_list(lst):\n    total = 0\n    for x in lst\n        total += x\n    return totl\n\n### Code:\n"
    ]

    print("=======================================================", flush=True)
    print("  RESULTADOS DE GENERACIÓN Y DEPURACIÓN (INHIBICIÓN ON)", flush=True)
    print("=======================================================", flush=True)

    for p in test_prompts:
        code = generate_coherent_code(
            model=model,
            enc=enc,
            prompt=p,
            max_new_tokens=45,
            temperature=0.3,
            top_k=25,
            repetition_penalty=1.4
        )
        print(f"\n[Prompt]:\n{p.strip()}", flush=True)
        print(f"[Código Generado por HoloCodeLM-27M]:\n{code}", flush=True)
        print("-" * 65, flush=True)


if __name__ == "__main__":
    main()
