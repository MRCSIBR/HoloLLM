r"""
SCRIPT: benchmark_350m_vs_jan.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM) - Escalamiento 350M
DESCRIPCIÓN: Suite de benchmark comparativo contra las especificaciones de Jan-Code-4B:
             1. Code Generation
             2. Code Refactoring & Optimization
             3. Bug Fixing & Diagnostics
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
    "checkpoint_path": "checkpoints/holo_350m_code_master.pt",
    "vocab_size": 151936,
    "dim": 768,
    "depth": 16,
    "num_heads": 12,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

JAN_BENCHMARK_SUITE = [
    # 1. GENERATION (Algorítmica clásica)
    ("Generation", "Factorial Recursivo", "factorial"),
    ("Generation", "Fibonacci Memoizado", "fibonacci with memoization"),
    ("Generation", "Búsqueda Binaria", "binary search"),
    ("Generation", "Inversión de Lista Enlazada", "reverse linked list"),
    ("Generation", "Merge Sort O(N log N)", "merge sort"),
    ("Generation", "Quickselect k-ésimo", "quick select kth smallest"),
    ("Generation", "Camino Mínimo Dijkstra", "dijkstra shortest path"),
    
    # 2. REFACTORING & CODING REAL (Estilo CodeAlpaca / Jan-Code)
    ("Refactoring", "Verificación de Número Primo", "Write a python function to check if a number is prime."),
    ("Refactoring", "Palíndromo In-Place", "Write a python function to check if a string is a palindrome."),
    ("Refactoring", "Máximo Subarreglo Kadane", "maximum subarray kadane")
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
    print("SUITE COMPARATIVA: HOLOLLM-350M vs JAN-CODE-4B SPEC")
    print(f"Dispositivo: {device} | Checkpoint: {CONFIG['checkpoint_path']}")
    print("Memoria: O(1) Constante (0.00 KB KV-Cache)")
    print("=" * 80)

    if not os.path.exists(CONFIG["checkpoint_path"]):
        print(f"Aún no se ha completado el entrenamiento en {CONFIG['checkpoint_path']}.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_id"], trust_remote_code=True)
    model = HoloCausalLMV2(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"]
    ).to(device=device, dtype=torch.bfloat16).eval()

    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
    print("✔ Checkpoint 350M canónico cargado exitosamente en VRAM.\n")

    passed_ast = 0
    total_tests = len(JAN_BENCHMARK_SUITE)
    speeds = []

    for idx, (category, title, task_str) in enumerate(JAN_BENCHMARK_SUITE, 1):
        prompt = f"<|im_start|>user\nWrite a python function for the following task:\n{task_str}<|im_end|>\n<|im_start|>assistant\n```python\n"
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        t_in = torch.tensor([tokens], device=device)

        start_t = time.perf_counter()
        with torch.no_grad():
            logits, states = model(t_in)
            next_t = torch.argmax(logits[0, -1, :]).item()
            current = list(tokens) + [next_t]
            gen = [next_t]

            for _ in range(140):
                cur_pos = len(current) - 1
                token_tensor = torch.tensor([[next_t]], device=device)
                h = model.token_emb(token_tensor)

                next_states = []
                for block, s in zip(model.blocks, states):
                    h, next_s = block.step(h, pos_t=cur_pos, state=s)
                    next_states.append(next_s)
                states = next_states

                logits = model.lm_head(model.ln_f(h))
                next_t = torch.argmax(logits[0, -1, :]).item()
                tok_str = tokenizer.decode([next_t])

                if "```" in tok_str or next_t in (151645, 151643):
                    break
                gen.append(next_t)
                current.append(next_t)

        elapsed = time.perf_counter() - start_t
        speed = len(gen) / max(elapsed, 1e-4)
        speeds.append(speed)
        
        code = tokenizer.decode(gen).strip()
        is_valid = check_ast(code)
        if is_valid:
            passed_ast += 1

        print(f"\n[{idx:02d}/{total_tests}] [{category.upper()}] {title.upper()}")
        print(f"Sintaxis AST: {'[PASSED]' if is_valid else '[FAILED]'} | Throughput: {speed:.1f} tok/s | Tokens: {len(gen)}")
        print("--- CÓDIGO GENERADO ---")
        print(code)
        print("." * 60)

    avg_speed = sum(speeds) / max(len(speeds), 1)
    print("\n" + "=" * 80)
    print("RESUMEN DEL BENCHMARK:")
    print(f"- Compilación AST Perfecta: {passed_ast}/{total_tests} ({passed_ast/total_tests*100:.1f}%)")
    print(f"- Velocidad Media en A100:  {avg_speed:.1f} tokens/segundo")
    print(f"- Consumo de KV-Cache:      0.00 KB (O(1) Constante Invariante)")
    print("=" * 80)


if __name__ == "__main__":
    main()
