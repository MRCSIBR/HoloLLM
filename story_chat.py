"""
Script: story_chat.py
Consola Interactiva de Narrativa y Diálogo para HoloLLM-174M.
"""

import torch
import tiktoken
from src.models.holo_causal_lm import HoloCausalLM

def generate_interactive(model, enc, device, prompt: str, max_tokens: int = 100):
    p_tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    current = list(p_tokens)
    print("\nHoloLLM > ", end="", flush=True)

    with torch.no_grad():
        t_in = torch.tensor([current], device=device, dtype=torch.long)
        logits, states = model(t_in)
        for _ in range(max_tokens):
            last_logits = logits[0, -1, :].clone().float()

            for prev in set(current[-35:]):
                if last_logits[prev] > 0:
                    last_logits[prev] /= 1.20
                else:
                    last_logits[prev] *= 1.20

            v, _ = torch.topk(last_logits, 35)
            last_logits[last_logits < v[-1]] = -float('Inf')
            probs = torch.softmax(last_logits / 0.72, dim=-1)
            next_t = torch.multinomial(probs, 1).item()

            if next_t == 50256:
                break

            current.append(next_t)
            print(enc.decode([next_t]), end="", flush=True)

            cur_pos = len(current) - 1
            x_step = model.token_emb(torch.tensor([[next_t]], device=device)) + model.pos_emb(torch.tensor([[cur_pos]], device=device))
            next_states = []
            h = x_step
            for block, s in zip(model.blocks, states):
                h, next_s = block.step(h, s)
                next_states.append(next_s)
            states = next_states
            logits = model.lm_head(model.ln_f(h))
    print("\n" + "─" * 60)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = tiktoken.get_encoding("gpt2")

    print("\n=======================================================")
    print("  CONSOLA INTERACTIVA HOLO-LLM (174M TOKENS EN A100)   ")
    print("=======================================================")

    model = HoloCausalLM(
        vocab_size=50257,
        dim=384,
        depth=6,
        num_heads=8,
        max_seq_len=406
    ).to(device)

    model.load_state_dict(torch.load("checkpoints/holo_coherent_28m.pt", map_location=device))
    model.eval()
    print("✔ Modelo cargado. Escribe cualquier frase en inglés para continuar (o 'exit'):\n")

    while True:
        try:
            user_prompt = input("Prompt > ").strip()
            if not user_prompt:
                continue
            if user_prompt.lower() in ("exit", "quit"):
                break
            generate_interactive(model, enc, device, user_prompt, max_tokens=100)
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()
