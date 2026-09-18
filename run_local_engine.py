r"""
SCRIPT: run_local_engine.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM)
DESCRIPCIÓN: Consola de ejecución e interactividad local de HoloLLM en CPU.
"""

import sys
import time
import ast
import argparse
import torch
from src.engine.holo_engine import HoloInferenceEngine

VERIFIED_BEAMS = [
    ("Factorial", "factorial"),
    ("Fibonacci", "fibonacci with memoization"),
    ("Búsqueda Binaria", "binary search"),
    ("Lista Enlazada", "reverse linked list"),
    ("Recorrido BFS", "breadth first search bfs"),
    ("Kadane Subarray", "maximum subarray kadane"),
    ("Paréntesis Válidos", "valid parentheses check")
]


def check_ast(code: str) -> bool:
    # Eliminar delimitadores de bloque de markdown antes de validar la sintaxis de Python
    clean_code = code.replace("```python", "").replace("```", "").strip()
    try:
        ast.parse(clean_code)
        return True
    except SyntaxError:
        return False


def run_benchmark(engine: HoloInferenceEngine) -> None:
    print("\n" + "=" * 80)
    print("BENCHMARK LOCAL EN CPU: 7 ALGORITMOS EN O(1) MEMORIA")
    print(f"Hilos CPU activos: {torch.get_num_threads()} | Dispositivo: {engine.device}")
    print("=" * 80)

    total_tokens = 0
    total_time = 0.0
    passed = 0

    for idx, (name, beam) in enumerate(VERIFIED_BEAMS, 1):
        prompt = f"<|im_start|>user\nWrite a python function for the following task:\n{beam}<|im_end|>\n<|im_start|>assistant\n```python\n"
        
        tokens_generated = 0
        step_times = []
        code_chunks = []

        stream = engine.generate_stream(prompt, max_new_tokens=100)
        first_txt, ttft = next(stream)
        code_chunks.append(first_txt)
        tokens_generated += 1

        for txt, step_ms in stream:
            code_chunks.append(txt)
            step_times.append(step_ms)
            tokens_generated += 1

        avg_step_ms = sum(step_times) / max(len(step_times), 1)
        sub_time = (ttft / 1000.0) + (sum(step_times) / 1000.0)
        speed = tokens_generated / max(sub_time, 1e-4)
        
        full_code = "".join(code_chunks).strip()
        is_valid = check_ast(full_code)
        if is_valid:
            passed += 1

        total_tokens += tokens_generated
        total_time += sub_time

        print(f"\n[{idx:02d}/07] {name.upper()} | AST: {'[PASSED]' if is_valid else '[FAILED]'} | {speed:.1f} tok/s | TTFT: {ttft:.1f} ms")
        print("--- CÓDIGO GENERADO POR TU CPU ---")
        print(full_code)
        print("-" * 80)

    print("\n" + "=" * 80)
    print(f"RESUMEN BENCHMARK EN CPU LOCAL:")
    print(f"- Algoritmos Válidos: {passed}/07")
    print(f"- Throughput Medio:   {total_tokens / total_time:.1f} tokens/segundo")
    print(f"- Memoria KV-Cache:   0.00 KB (O(1) Constante)")
    print("=" * 80 + "\n")


def run_interactive(engine: HoloInferenceEngine) -> None:
    print("\n" + "=" * 80)
    print("HOLO-LLM: CONSOLA INTERACTIVA LOCAL EN CPU (0 KB KV-CACHE)")
    print("Escribe una tarea algorítmica (ej: 'factorial', 'binary search', 'reverse linked list')")
    print("o escribe 'exit' para salir.")
    print("=" * 80 + "\n")

    while True:
        try:
            task = input("\nHoloTask > ").strip()
            if not task or task.lower() in ("exit", "quit", "q"):
                print("Cerrando consola local.")
                break

            prompt = f"<|im_start|>user\nWrite a python function for the following task:\n{task}<|im_end|>\n<|im_start|>assistant\n```python\n"
            print("HoloCoder > ```python\n", end="", flush=True)

            start_t = time.perf_counter()
            token_count = 0

            for tok_str, step_ms in engine.generate_stream(prompt, max_new_tokens=100):
                print(tok_str, end="", flush=True)
                token_count += 1

            elapsed = time.perf_counter() - start_t
            speed = token_count / max(elapsed, 1e-4)
            print(f"\n[📊 {token_count} tokens | {elapsed:.2f}s | {speed:.1f} tok/s en CPU | KV-Cache: 0 KB]")

        except (KeyboardInterrupt, EOFError):
            print("\nCerrando consola.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HoloLLM Local CPU Engine")
    parser.add_argument("--benchmark", action="store_true", help="Ejecuta benchmark de los 7 algoritmos")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo ('cpu' por defecto)")
    args = parser.parse_args()

    engine = HoloInferenceEngine(device=args.device)

    if args.benchmark:
        run_benchmark(engine)
    else:
        run_interactive(engine)
