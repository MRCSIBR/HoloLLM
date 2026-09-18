"""
Script: chat_distilled.py
Consola interactiva en streaming para el alumno HoloLLM-66M destilado
desde Qwen2.5-Coder en la A100, con vocabulario de 151,936 tokens y memoria O(1).
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer
from src.models.holo_causal_lm import HoloCausalLM

TEACHER_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CKPT_PATH = "checkpoints/holo_distilled_student_75m.pt"

def stream_distilled_response(
    model: HoloCausalLM,
    tokenizer,
    user_prompt: str,
    device: torch.device,
    max_tokens: int = 80,
    temperature: float = 0.35,
    top_k: int = 30,
    repetition_penalty: float = 1.3
):
    # Formato ChatML nativo de Qwen
    full_prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(full_prompt)
    current_tokens = list(prompt_ids)

    print("\nHoloStudent > ", end="", flush=True)
    start_time = time.time()
    first_tok_time = None
    gen_count = 0

    # Tokens de parada de Qwen
    eos_ids = {151645, 151643}  # <|im_end|>, <|endoftext|>

    with torch.no_grad():
        prompt_tensor = torch.tensor([current_tokens], device=device, dtype=torch.long)
        logits, states = model(prompt_tensor)

        for step in range(max_tokens):
            last_logits = logits[0, -1, :].clone().float()

            # Inhibición de repetición sobre tokens recientes
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

            if next_token in eos_ids:
                break

            if first_tok_time is None:
                first_tok_time = time.time() - start_time

            token_str = tokenizer.decode([next_token])
            print(token_str, end="", flush=True)

            current_tokens.append(next_token)
            gen_count += 1

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
    tok_s = gen_count / max(total_time, 1e-4)
    first_ms = (first_tok_time or 0.0) * 1000

    print("\n" + "─" * 65)
    print(f"📊 [Métricas]: {gen_count} tokens | {total_time:.2f}s | {tok_s:.1f} tok/s | TTFT: {first_ms:.1f}ms | KV-Cache: 0 KB (O(1))")
    print("─" * 65 + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=======================================================")
    print("  HOLO-STUDENT (DESTILADO DESDE QWEN2.5-CODER EN A100) ")
    print("=======================================================")

    print(f"Cargando tokenizer de {TEACHER_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)

    # Dimensiones exactas con las que se destiló (~66.4M parámetros)
    DIM = 384
    DEPTH = 6
    HEADS = 8
    MAX_LEN = 128 + 32
    VOCAB_SIZE = 151936

    model = HoloCausalLM(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        max_seq_len=MAX_LEN
    ).to(device)

    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print(f"✔ Alumno cargado exitosamente ({sum(p.numel() for p in model.parameters()):,} parámetros).")
    print("Escribe tu instrucción de código o pregunta (o 'exit' para salir):\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break
            stream_distilled_response(model, tokenizer, user_input, device)
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
