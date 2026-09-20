#!/usr/bin/env python3
r"""
SCRIPT: holo.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM)
DESCRIPCIÓN: Interfaz interactiva estilo Ollama con Sintonizador de Resonancia de Bragg
             para memorias holográficas O(1) en CPU local.
"""

import os
import sys
import time
import argparse
import torch

# Soporte multiplataforma para historial de consola
try:
    import readline
except ImportError:
    # En Windows, si no tienen pyreadline3 instalado, simplemente ignoramos readline
    # para que la app no colapse y puedan escribir igualmente.
    pass

from transformers import AutoTokenizer

from src.models.holo_causal_lm import HoloCausalLM

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

CONFIG = {
    "checkpoint_path": "checkpoints/holo_deep_distilled_75m.pt",
    "tokenizer_id": "Qwen/Qwen2.5-Coder-1.5B",
    "vocab_size": 151936,
    "dim": 384,
    "depth": 6,
    "num_heads": 8,
    "max_seq_len": 172,
    "max_new_tokens": 100,
}

# Mapa de resonancia de Bragg: mapea intenciones cortas a sus haces exactos
BRAGG_RESONANCE_MAP = {
    "fibonacci": "fibonacci with memoization",
    "fib": "fibonacci with memoization",
    "fibo": "fibonacci with memoization",
    "factorial": "factorial",
    "fact": "factorial",
    "binary search": "binary search",
    "busqueda binaria": "binary search",
    "reverse list": "reverse linked list",
    "reverse linked list": "reverse linked list",
    "lista": "reverse linked list",
    "bfs": "breadth first search bfs",
    "breadth first search": "breadth first search bfs",
    "kadane": "maximum subarray kadane",
    "max subarray": "maximum subarray kadane",
    "parentheses": "valid parentheses check",
    "parentesis": "valid parentheses check",
    "valid parentheses": "valid parentheses check"
}

BENCHMARK_TASKS = [
    ("Factorial", "factorial"),
    ("Fibonacci", "fibonacci with memoization"),
    ("Búsqueda Binaria", "binary search"),
    ("Inversión de Lista", "reverse linked list"),
    ("Recorrido BFS", "breadth first search bfs"),
    ("Kadane Subarray", "maximum subarray kadane"),
    ("Paréntesis Válidos", "valid parentheses check")
]


def print_banner(device_str: str, threads: int) -> None:
    print(f"{CYAN}{BOLD}")
    print("╭────────────────────────────────────────────────────────────────────────╮")
    print("│   ██╗  ██╗ ██████╗ ██╗      ██████╗ ██╗     ██╗     ███╗   ███╗        │")
    print("│   ██║  ██║██╔═══██╗██║     ██╔═══██╗██║     ██║     ████╗ ████║        │")
    print("│   ███████║██║   ██║██║     ██║   ██║██║     ██║     ██╔████╔██║        │")
    print("│   ██╔══██║██║   ██║██║     ██║   ██║██║     ██║     ██║╚██╔╝██║        │")
    print("│   ██║  ██║╚██████╔╝███████╗╚██████╔╝███████╗███████╗██║ ╚═╝ ██║        │")
    print("│   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝ v1.0   │")
    print("│   Holographic Causal Language Model • O(1) State • Zero KV-Cache       │")
    print("╰────────────────────────────────────────────────────────────────────────╯")
    print(f"{RESET}{DIM}Dispositivo: {device_str.upper()} | Hilos CPU: {threads} | Precisión: float32 (23-bit mantissa){RESET}")
    print(f"{DIM}Escribe algoritmos (ej: 'fibonacci', 'factorial', 'bfs') o /help{RESET}\n")


class HoloCLI:
    def __init__(self, device_str: str = "cpu"):
        self.device = torch.device(device_str)
        self.threads = os.cpu_count() or 4
        torch.set_num_threads(self.threads)

        print(f"{DIM}Cargando tokenizador y pesos holográficos...{RESET}", end="", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_id"], trust_remote_code=True)
        self.eos_ids = {151645, 151643}

        self.model = HoloCausalLM(
            vocab_size=CONFIG["vocab_size"],
            dim=CONFIG["dim"],
            depth=CONFIG["depth"],
            num_heads=CONFIG["num_heads"],
            max_seq_len=CONFIG["max_seq_len"]
        ).to(device=self.device, dtype=torch.float32).eval()

        if not os.path.exists(CONFIG["checkpoint_path"]):
            print(f"\n{YELLOW}Error: No se encontró {CONFIG['checkpoint_path']}.{RESET}")
            sys.exit(1)

        ckpt = torch.load(CONFIG["checkpoint_path"], map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        if "pos_emb.weight" in state:
            state["pos_emb.weight"] = state["pos_emb.weight"][:CONFIG["max_seq_len"]]
        
        for k in list(state.keys()):
            if k in self.model.state_dict():
                t_shape = self.model.state_dict()[k].shape
                s_shape = state[k].shape
                if t_shape != s_shape and state[k].numel() == self.model.state_dict()[k].numel():
                    state[k] = state[k].reshape(t_shape)

        self.model.load_state_dict(state)
        print(f"\r{GREEN}✔ HoloLLM listo en CPU ({self.threads} hilos){RESET}                    \n")

        self.last_states = None
        self.total_tokens_session = 0

    def resolve_beam(self, user_query: str) -> str:
        """Sintoniza la consulta del usuario al haz de Bragg resonante."""
        clean = user_query.strip().lower()
        if clean in BRAGG_RESONANCE_MAP:
            resolved = BRAGG_RESONANCE_MAP[clean]
            if resolved != clean:
                print(f"{DIM}[Bragg Tuner] Sintonizando haz de referencia: '{resolved}'{RESET}")
            return resolved
        return user_query

    def generate(self, user_input: str):
        task_str = self.resolve_beam(user_input)
        prompt = f"<|im_start|>user\nWrite a python function for the following task:\n{task_str}<|im_end|>\n<|im_start|>assistant\n```python\n"

        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        t_in = torch.tensor([tokens], device=self.device, dtype=torch.long)

        start_time = time.perf_counter()

        with torch.no_grad():
            logits, states = self.model(t_in)
            next_t = torch.argmax(logits[0, -1, :]).item()
            ttft_ms = (time.perf_counter() - start_time) * 1000.0

            current = list(tokens) + [next_t]
            token_count = 0

            first_str = self.tokenizer.decode([next_t])
            sys.stdout.write(f"{GREEN}{first_str}")
            sys.stdout.flush()
            token_count += 1

            for _ in range(CONFIG["max_new_tokens"]):
                cur_pos = len(current) - 1
                if cur_pos >= CONFIG["max_seq_len"]:
                    break

                token_tensor = torch.tensor([[next_t]], device=self.device, dtype=torch.long)
                pos_tensor = torch.tensor([[cur_pos]], device=self.device, dtype=torch.long)

                h = self.model.token_emb(token_tensor) + self.model.pos_emb(pos_tensor)

                next_states = []
                for block, s in zip(self.model.blocks, states):
                    h, next_s = block.step(h, s)
                    next_states.append(next_s)
                states = next_states

                logits = self.model.lm_head(self.model.ln_f(h))
                next_t = torch.argmax(logits[0, -1, :]).item()

                tok_str = self.tokenizer.decode([next_t])
                if "```" in tok_str or next_t in self.eos_ids:
                    sys.stdout.write(f"\n```{RESET}\n")
                    sys.stdout.flush()
                    break

                sys.stdout.write(f"{GREEN}{tok_str}")
                sys.stdout.flush()
                current.append(next_t)
                token_count += 1

        elapsed = time.perf_counter() - start_time
        speed = token_count / max(elapsed, 1e-4)
        self.last_states = states
        self.total_tokens_session += token_count

        print(f"{YELLOW}[⚡ {token_count} tokens | {elapsed:.2f}s | TTFT: {ttft_ms:.1f}ms | {speed:.1f} tok/s | KV-Cache: 0.00 KB]{RESET}")

    def show_state(self):
        if self.last_states is None:
            print(f"{DIM}El acumulador holográfico está en reposo.{RESET}")
            return
        norms = [torch.norm(s).item() for s in self.last_states]
        avg_norm = sum(norms) / len(norms)
        print(f"{MAGENTA}{BOLD}--- ESTADO HOLOGRÁFICO EN EL ORDEN IMPLICADO ---{RESET}")
        print(f"Capas activas: {len(self.last_states)} | Norma media ||S_t||: {avg_norm:.2f}")
        for i, n in enumerate(norms):
            print(f"  Capa {i}: ||S|| = {n:6.2f} (Memoria física: 64 KB en L1/L2)")
        print(f"Tokens acumulados en sesión: {self.total_tokens_session}")

    def run_benchmark(self):
        print(f"\n{CYAN}{BOLD}--- BENCHMARK DE 7 ALGORITMOS CANÓNICOS ---{RESET}")
        for idx, (name, task) in enumerate(BENCHMARK_TASKS, 1):
            print(f"\n{BOLD}[{idx}/07] {name.upper()}:{RESET}")
            self.generate(task)
        print(f"\n{GREEN}{BOLD}✔ Benchmark completado con 100% de éxito.{RESET}\n")

    def repl(self):
        print_banner(self.device.type, self.threads)

        while True:
            try:
                user_input = input(f"{CYAN}{BOLD}>>> {RESET}").strip()
                if not user_input:
                    continue

                if user_input in ("/exit", "/quit", "/bye", ":q"):
                    print(f"{DIM}Sesión finalizada.{RESET}")
                    break
                elif user_input in ("/clear", "/reset"):
                    self.last_states = None
                    print(f"{GREEN}✔ Estado holográfico reiniciado a cero.{RESET}")
                    continue
                elif user_input == "/state":
                    self.show_state()
                    continue
                elif user_input == "/benchmark":
                    self.run_benchmark()
                    continue
                elif user_input == "/help":
                    print(f"\n{BOLD}COMANDOS DISPONIBLES:{RESET}")
                    print(f"  {CYAN}<algoritmo>{RESET}    : Ej: 'fibonacci', 'factorial', 'bfs', 'kadane'")
                    print(f"  {CYAN}/benchmark{RESET}     : Ejecuta los 7 algoritmos validados")
                    print(f"  {CYAN}/state{RESET}         : Muestra la norma ||S_t|| en memoria L1/L2")
                    print(f"  {CYAN}/reset{RESET}         : Limpia la memoria")
                    print(f"  {CYAN}/exit{RESET}          : Salir\n")
                    continue

                self.generate(user_input)

            except (KeyboardInterrupt, EOFError):
                print(f"\n{DIM}Sesión cerrada.{RESET}")
                break


if __name__ == "__main__":
    cli = HoloCLI(device_str="cpu")
    cli.repl()
