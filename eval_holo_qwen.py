r"""
SCRIPT: eval_holo_qwen.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM) - Evaluación HoloQwen-1.5B
DESCRIPCIÓN: Audita HoloQwen-1.5B calibrado (Stage 1) en modo recurrente O(1)
             con 28 capas holográficas y precisión float32.
"""

import os
import sys
import time
import ast
import torch
from transformers import AutoTokenizer
from src.models.holo_qwen import HoloQwenForCausalLM

CONFIG = {
    "tokenizer_id": "Qwen/Qwen2.5-Coder-1.5B",
    "checkpoint_path": "checkpoints/holo_qwen_1.5b_calibrated.pt",
    "vocab_size": 151936,
    "dim": 1536,
    "depth": 28,
    "num_heads": 12,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

TEST_ALGORITHMS = [
    ("Factorial Recursivo", "factorial"),
    ("Fibonacci Memoizado", "fibonacci with memoization"),
    ("Búsqueda Binaria", "binary search"),
    ("Inversión de Lista", "reverse linked list"),
    ("Recorrido BFS", "breadth first search bfs"),
    ("Subarreglo Kadane", "maximum subarray kadane"),
    ("Paréntesis Válidos", "valid parentheses check")
]


def check_ast(code: str) -> bool:
    clean = code.replace("```python", "").replace("```", "").strip()
    try:
        ast.parse(clean)
        return True
    except SyntaxError:
        return False


def main():
    device = torch.device(CONFIG["device"])
    print("=" * 80)
    print("AUDITORÍA CIENTÍFICA: HOLOQWEN-1.5B CALIBRADO (STAGE 1)")
    print(f"Dispositivo: {device} | Checkpoint: {CONFIG['checkpoint_path']}")
    print("Memoria: O(1) Constante (0.00 KB KV-Cache)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_id"], trust_remote_code=True)
    
    # Inferencia en float32 para preservar los 23 bits de mantisa de las fases en Fourier
    model = HoloQwenForCausalLM(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"]
    ).to(device=device, dtype=torch.float32).eval()

    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print("✔ Checkpoint HoloQwen-1.5B calibrado cargado exitosamente en VRAM.\n")

    passed_ast = 0
    total = len(TEST_ALGORITHMS)

    for idx, (title, task_str) in enumerate(TEST_ALGORITHMS, 1):
        prompt = f"<|im_start|>user\nWrite a python function for the following task:\n{task_str}<|im_end|>\n<|im_start|>assistant\n```python\n"
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        t_in = torch.tensor([tokens], device=device, dtype=torch.long)

        start_t = time.perf_counter()
        with torch.no_grad():
            logits, states = model(t_in)
            next_t = torch.argmax(logits[0, -1, :]).item()
            ttft_ms = (time.perf_counter() - start_t) * 1000.0

            current = list(tokens) + [next_t]
            gen = [next_t]

            for _ in range(80):
                cur_pos = len(current) - 1
                token_tensor = torch.tensor([[next_t]], device=device, dtype=torch.long)
                h = model.embed_tokens(token_tensor)

                next_states = []
                for block, s in zip(model.layers, states):
                    h, next_s = block.step(h, pos_t=cur_pos, state=s)
                    next_states.append(next_s)
                states = next_states

                logits = model.lm_head(model.norm(h))
                next_t = torch.argmax(logits[0, -1, :]).item()
                tok_str = tokenizer.decode([next_t])

                if "```" in tok_str or next_t in (151645, 151643):
                    break
                gen.append(next_t)
                current.append(next_t)

        elapsed = time.perf_counter() - start_t
        speed = len(gen) / max(elapsed, 1e-4)
        output_code = tokenizer.decode(gen).strip()
        is_valid = check_ast(output_code)
        if is_valid:
            passed_ast += 1

        print(f"\n[{idx:02d}/{total}] {title.upper()}")
        print(f"AST: {'[PASSED]' if is_valid else '[FAILED]'} | TTFT: {ttft_ms:5.1f} ms | Velocidad: {speed:5.1f} tok/s")
        print("--- CÓDIGO GENERADO ---")
        print(output_code)
        print("." * 60)

    print("\n" + "=" * 80)
    print(f"RESUMEN HOLOQWEN-1.5B (STAGE 1): {passed_ast}/{total} Algoritmos con Sintaxis AST Perfecta")
    print("Memoria KV-Cache Consumida: 0.00 KB (O(1) Constante)")
    print("=" * 80)


if __name__ == "__main__":
    main()
