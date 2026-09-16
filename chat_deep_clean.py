import os
import sys
import time
import torch
from transformers import AutoTokenizer
from src.models.holo_causal_lm import HoloCausalLM

TEACHER_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CKPT_PATH = "checkpoints/holo_deep_distilled_75m.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)

    model = HoloCausalLM(
        vocab_size=151936,
        dim=384,
        depth=6,
        num_heads=8,
        max_seq_len=172
    ).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    print("\n=======================================================")
    print("  HOLO-CODER 75M (ALGORITMOS LIMPIOS EN TIEMPO REAL)   ")
    print("=======================================================")
    print("✔ Modelo cargado. Escribe una tarea de código en Python (o 'exit'):\n")

    eos_ids = {151645, 151643}

    while True:
        try:
            task = input("Task > ").strip()
            if not task or task.lower() in ("exit", "quit"):
                break

            prompt = f"<|im_start|>user\nWrite a python function for the following task:\n{task}<|im_end|>\n<|im_start|>assistant\n```python\n"
            p_ids = tokenizer.encode(prompt)
            current = list(p_ids)

            print("\nHoloCoder > ```python\n", end="", flush=True)
            start_t = time.time()
            gen_count = 0

            with torch.no_grad():
                t_in = torch.tensor([current], device=device, dtype=torch.long)
                logits, states = model(t_in)

                for _ in range(60):
                    last_logits = logits[0, -1, :].clone().float()

                    for prev in set(current[-25:]):
                        if last_logits[prev] > 0:
                            last_logits[prev] /= 1.25
                        else:
                            last_logits[prev] *= 1.25

                    v, _ = torch.topk(last_logits, 20)
                    last_logits[last_logits < v[-1]] = -float('Inf')
                    probs = torch.softmax(last_logits / 0.25, dim=-1)
                    next_t = torch.multinomial(probs, 1).item()

                    if next_t in eos_ids:
                        break

                    tok_str = tokenizer.decode([next_t])
                    
                    # Corte limpio si el modelo cierra el bloque de código con ```
                    if "```" in tok_str:
                        print("\n```", flush=True)
                        break

                    print(tok_str, end="", flush=True)
                    current.append(next_t)
                    gen_count += 1

                    cur_pos = len(current) - 1
                    x_step = model.token_emb(torch.tensor([[next_t]], device=device)) + model.pos_emb(torch.tensor([[cur_pos]], device=device))

                    next_states = []
                    h = x_step
                    for block, s in zip(model.blocks, states):
                        h, next_s = block.step(h, s)
                        next_states.append(next_s)
                    states = next_states
                    logits = model.lm_head(model.ln_f(h))

            elapsed = time.time() - start_t
            print(f"\n📊 [{gen_count} tokens | {elapsed:.2f}s | {gen_count/max(elapsed, 1e-4):.1f} tok/s | KV-Cache: 0 KB (O(1))]\n")

        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
