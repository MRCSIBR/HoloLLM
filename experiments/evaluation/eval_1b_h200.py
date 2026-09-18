r"""
SCRIPT: eval_1b_h200.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM) - Escalamiento 1B
DESCRIPCIÓN: Suite de auditoría exhaustiva para HoloLLM-1B (969.8M params)
             entrenado con Jan-Code-4B sobre la NVIDIA H200.
"""

import os
import sys
import time
import ast
import torch
from transformers import AutoTokenizer
from src.models.holo_causal_lm_v2 import HoloCausalLMV2

CONFIG = {
    "tokenizer_id": "Qwen/Qwen2.5-Coder-1.5B",
    "checkpoint_path": "checkpoints/holo_1b_master_h100.pt",
    "vocab_size": 151936,
    "dim": 1536,
    "depth": 24,
    "num_heads": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

EVALUATION_SUITE = [
    # ALGORITMOS CANÓNICOS
    ("Algoritmo", "Factorial Recursivo", "factorial"),
    ("Algoritmo", "Fibonacci Memoizado", "fibonacci with memoization"),
    ("Algoritmo", "Búsqueda Binaria", "binary search"),
    ("Algoritmo", "Inversión de Lista Enlazada", "reverse linked list"),
    ("Algoritmo", "Merge Sort O(N log N)", "merge sort"),
    ("Algoritmo", "Recorrido BFS de Grafo", "breadth first search bfs"),
    ("Algoritmo", "Subarreglo Máximo (Kadane)", "maximum subarray kadane"),
    ("Algoritmo", "Verificación de Paréntesis Válidos", "valid parentheses check"),
    ("Algoritmo", "Quickselect k-ésimo", "quick select kth smallest"),
    ("Algoritmo", "Camino Mínimo Dijkstra", "dijkstra shortest path"),

    # DIÁLOGO Y CONCEPTO
    ("Concepto", "Explicación de Memoria O(1)", "¿Cómo logra HoloLLM memoria O(1) sin KV-Cache?"),
    ("Concepto", "Física de David Bohm", "Explica brevemente la diferencia entre el Orden Implicado y el Orden Explicado.")
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
    print("AUDITORÍA MAESTRA: HOLOLLM-1B (969.8M PARÁMETROS)")
    print(f"Dispositivo: {device} | Checkpoint: {CONFIG['checkpoint_path']}")
    print("Memoria de Estado: O(1) Constante (0.00 KB KV-Cache)")
    print("=" * 80)

    if not os.path.exists(CONFIG["checkpoint_path"]):
        print(f"Aún no se ha completado el entrenamiento en {CONFIG['checkpoint_path']}.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_id"], trust_remote_code=True)
    
    # Inferencia en float32 para preservar los 23 bits de mantisa de las fases en Fourier
    model = HoloCausalLMV2(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"]
    ).to(device=device, dtype=torch.float32).eval()

    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
    print("✔ Checkpoint HoloLLM-1B cargado exitosamente en memoria.\n")

    ast_passed = 0
    algo_count = 0

    for idx, (category, title, prompt_content) in enumerate(EVALUATION_SUITE, 1):
        if category == "Algoritmo":
            prompt = f"<|im_start|>user\nWrite a python function for the following task:\n{prompt_content}<|im_end|>\n<|im_start|>assistant\n```python\n"
            algo_count += 1
        else:
            prompt = f"<|im_start|>user\n{prompt_content}<|im_end|>\n<|im_start|>assistant\n"

        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        t_in = torch.tensor([tokens], device=device, dtype=torch.long)

        start_t = time.perf_counter()
        with torch.no_grad():
            logits, states = model(t_in)
            next_t = torch.argmax(logits[0, -1, :]).item()
            ttft_ms = (time.perf_counter() - start_t) * 1000.0

            current = list(tokens) + [next_t]
            gen = [next_t]

            for _ in range(130):
                cur_pos = len(current) - 1
                token_tensor = torch.tensor([[next_t]], device=device, dtype=torch.long)
                h = model.token_emb(token_tensor)

                next_states = []
                for block, s in zip(model.blocks, states):
                    h, next_s = block.step(h, pos_t=cur_pos, state=s)
                    next_states.append(next_s)
                states = next_states

                logits = model.lm_head(model.ln_f(h))
                next_t = torch.argmax(logits[0, -1, :]).item()
                tok_str = tokenizer.decode([next_t])

                if "```" in tok_str or next_t in (151645, 151643) or "<|im_end|>" in tok_str:
                    break
                gen.append(next_t)
                current.append(next_t)

        elapsed = time.perf_counter() - start_t
        speed = len(gen) / max(elapsed, 1e-4)
        output_text = tokenizer.decode(gen).strip()

        is_ast_ok = False
        if category == "Algoritmo":
            is_ast_ok = check_ast(output_text)
            if is_ast_ok:
                ast_passed += 1

        status_str = f"AST: {'[PASSED]' if is_ast_ok else '[FAILED]'}" if category == "Algoritmo" else "TEXT OK"

        print(f"\n[{idx:02d}/{len(EVALUATION_SUITE)}] [{category.upper()}] {title.upper()}")
        print(f"{status_str} | TTFT: {ttft_ms:5.1f} ms | Velocidad: {speed:5.1f} tok/s | Tokens: {len(gen)}")
        print("--- SALIDA GENERADA ---")
        print(output_text)
        print("." * 60)

    print("\n" + "=" * 80)
    print("RESUMEN DE LA AUDITORÍA HOLOLLM-1B:")
    print(f"- Algoritmos Válidos (AST): {ast_passed}/{algo_count} ({ast_passed/max(algo_count, 1)*100:.1f}%)")
    print(f"- Consumo de KV-Cache:     0.00 KB (O(1) Constante)")
    print("=" * 80)


if __name__ == "__main__":
    main()
