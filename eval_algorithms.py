r"""
SCRIPT: eval_algorithms.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM)
DESCRIPCIÓN: Evaluación formal con memoria recurrente O(1) nativa (block.step),
             validación sintáctica AST y medición de latencia TTFT.
"""

import os
import sys
import time
import ast
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.models.holo_causal_lm import HoloCausalLM

CONFIG = {
    "teacher_id": "Qwen/Qwen2.5-Coder-1.5B",
    "checkpoint_path": "checkpoints/holo_deep_distilled_75m.pt",
    "vocab_size": 151936,
    "dim": 384,
    "depth": 6,
    "num_heads": 8,
    "seq_len": 172,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

TEST_TASKS = [
    ("Recursión Factorial", "factorial"),
    ("Fibonacci Memo", "fibonacci with memoization"),
    ("Búsqueda Binaria", "binary search"),
    ("Lista Enlazada", "reverse linked list"),
    ("Merge Sort", "merge sort"),
    ("Recorrido BFS", "breadth first search bfs"),
    ("Kadane Subarray", "maximum subarray kadane"),
    ("Paréntesis Válidos", "valid parentheses check"),
    ("Quickselect", "quick select kth smallest"),
    ("Dijkstra", "dijkstra shortest path")
]


def check_python_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


@torch.no_grad()
def generate_recurrent_o1(model: HoloCausalLM, tokenizer: AutoTokenizer, task: str, device: str) -> tuple:
    prompt = f"<|im_start|>user\nWrite a python function for the following task:\n{task}<|im_end|>\n<|im_start|>assistant\n```python\n"
    p_ids = tokenizer.encode(prompt)
    current = list(p_ids)
    eos_ids = {151645, 151643}

    start_t = time.time()
    t_in = torch.tensor([current], device=device, dtype=torch.long)
    
    # Prefill inicial: computa logits y estados recurrentes iniciales
    out = model(t_in)
    logits, states = out if isinstance(out, tuple) else (out, None)
    
    ttft_ms = (time.time() - start_t) * 1000
    gen_count = 0
    generated_tokens = []

    for _ in range(80):
        if len(current) >= CONFIG["seq_len"]:
            break

        last_logits = logits[0, -1, :].clone().float()

        # Penalización por repetición (idéntico a chat_deep_clean)
        for prev in set(current[-25:]):
            if last_logits[prev] > 0:
                last_logits[prev] /= 1.25
            else:
                last_logits[prev] *= 1.25

        v, _ = torch.topk(last_logits, 20)
        last_logits[last_logits < v[-1]] = -float('Inf')
        probs = torch.softmax(last_logits / 0.2, dim=-1)
        next_t = torch.multinomial(probs, 1).item()

        if next_t in eos_ids:
            break

        tok_str = tokenizer.decode([next_t])
        if "```" in tok_str:
            break

        current.append(next_t)
        generated_tokens.append(next_t)
        gen_count += 1

        # Paso O(1) estricto usando block.step
        cur_pos = len(current) - 1
        x_step = model.token_emb(torch.tensor([[next_t]], device=device)) + model.pos_emb(torch.tensor([[cur_pos]], device=device))

        next_states = []
        h = x_step
        for block, s in zip(model.blocks, states):
            h, next_s = block.step(h, s)
            next_states.append(next_s)
        states = next_states
        logits = model.to_logits(model.ln_f(h)) if hasattr(model, "to_logits") else model.lm_head(model.ln_f(h))

    total_time = time.time() - start_t
    tok_s = gen_count / max(total_time, 1e-4)
    code_result = tokenizer.decode(generated_tokens)
    return code_result, ttft_ms, tok_s


def main():
    device = torch.device(CONFIG["device"])
    print("=" * 80)
    print("AUDITORÍA DE SINTAXIS AST CON MEMORIA O(1) RECURRENTE")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["teacher_id"], trust_remote_code=True)
    model = HoloCausalLM(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"],
        max_seq_len=CONFIG["seq_len"]
    ).to(device).eval()

    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)

    passes = 0
    for idx, (label, task) in enumerate(TEST_TASKS, 1):
        code, ttft, speed = generate_recurrent_o1(model, tokenizer, task, device)
        valid = check_python_syntax(code)
        if valid:
            passes += 1
        print(f"[{idx:02d}/10] {label.upper()} | AST: {'[OK]' if valid else '[FAIL]'} | {speed:.1f} tok/s | TTFT: {ttft:.1f} ms")
        print(code.strip())
        print("-" * 80)

    print(f"\nRESULTADO FINAL: {passes}/10 Algoritmos Sintácticamente Válidos.")


if __name__ == "__main__":
    main()
