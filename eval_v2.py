r"""
SCRIPT: eval_v2.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM) - Arquitectura v2
DESCRIPCIÓN: Auditoría integral de HoloLLM-v2 con flujo profundo de logits corregido.
"""

import os
import sys
import time
import ast
import torch
from transformers import AutoTokenizer
from src.models.holo_causal_lm_v2 import HoloCausalLMV2

CONFIG = {
    "teacher_id": "Qwen/Qwen2.5-Coder-1.5B",
    "checkpoint_path": "checkpoints/holo_v2_conversational_70m.pt",
    "vocab_size": 151936,
    "dim": 384,
    "depth": 6,
    "num_heads": 8,
    "max_seq_len": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

ALGORITHMS = [
    ("Factorial Recursivo", "factorial"),
    ("Fibonacci Memoizado", "fibonacci with memoization"),
    ("Búsqueda Binaria", "binary search"),
    ("Inversión de Lista", "reverse linked list"),
    ("Merge Sort", "merge sort"),
    ("Recorrido BFS", "breadth first search bfs"),
    ("Subarreglo Kadane", "maximum subarray kadane"),
    ("Paréntesis Válidos", "valid parentheses check"),
    ("Quickselect k-ésimo", "quick select kth smallest"),
    ("Camino Mínimo Dijkstra", "dijkstra shortest path")
]

DIALOGUE_TEST = (
    "Hola, me llamo Lucas y estoy investigando redes neuronales holográficas.",
    "¿Recuerdas cuál es mi nombre y qué estoy investigando?"
)


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
    print("AUDITORÍA CIENTÍFICA HOLOLLM-v2: FLUJO PROFUNDO Y MEMORIA MULTI-TURNO")
    print(f"Dispositivo: {device} | Checkpoint: {CONFIG['checkpoint_path']}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["teacher_id"], trust_remote_code=True)
    model = HoloCausalLMV2(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"]
    ).to(device=device, dtype=torch.bfloat16).eval()

    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
    print("✔ Checkpoint v2 cargado con éxito en VRAM.\n")

    # =================================================================
    # PARTE 1: BENCHMARK ALGORÍTMICO
    # =================================================================
    print("─" * 80)
    print("PARTE 1: BENCHMARK DE ALGORITMOS (CAPACIDAD 512 TOKENS)")
    print("─" * 80)

    passed = 0
    for idx, (title, beam) in enumerate(ALGORITHMS, 1):
        prompt = f"<|im_start|>user\nWrite a python function for the following task:\n{beam}<|im_end|>\n<|im_start|>assistant\n```python\n"
        tokens = tokenizer.encode(prompt)
        t_in = torch.tensor([tokens], device=device)

        start_t = time.perf_counter()
        with torch.no_grad():
            logits, states = model(t_in)
            next_t = torch.argmax(logits[0, -1, :]).item()
            current = list(tokens) + [next_t]
            gen = [next_t]

            for _ in range(160):
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
        code = tokenizer.decode(gen).strip()
        is_valid = check_ast(code)
        if is_valid:
            passed += 1

        print(f"\n[{idx:02d}/10] {title.upper():<24} | AST: {'[PASSED]' if is_valid else '[FAILED]'} | {speed:5.1f} tok/s")
        print(code)
        print("." * 60)

    print("\n" + "=" * 80)
    print(f"RESULTADO ALGORITMOS: {passed}/10 Sintaxis AST Perfecta")
    print("=" * 80 + "\n")

    # =================================================================
    # PARTE 2: PRUEBA DE MEMORIA CONVERSACIONAL MULTI-TURNO (CORREGIDA)
    # =================================================================
    print("─" * 80)
    print("PARTE 2: CONVERSACIÓN MULTI-TURNO VIVA (absorb_context O(1))")
    print("─" * 80)

    u1, u2 = DIALOGUE_TEST
    print(f"Usuario (Turno 1): \"{u1}\"")

    t1_prompt = f"<|im_start|>user\n{u1}<|im_end|>\n<|im_start|>assistant\n"
    t1_tokens = tokenizer.encode(t1_prompt)

    with torch.no_grad():
        # 1. Absorber Turno 1 y obtener los logits profundos reales
        logits1, states, pos = model.absorb_context(t1_tokens, start_pos=0, states=None, device=device)
        next_t = torch.argmax(logits1).item()
        r1_tokens = [next_t]

        # Generar respuesta del Turno 1
        for _ in range(70):
            cur_pos = pos
            token_tensor = torch.tensor([[next_t]], device=device)
            h = model.token_emb(token_tensor)
            
            next_states = []
            for block, s in zip(model.blocks, states):
                h, next_s = block.step(h, pos_t=cur_pos, state=s)
                next_states.append(next_s)
            states = next_states
            pos += 1

            logits = model.lm_head(model.ln_f(h))
            next_t = torch.argmax(logits[0, -1, :]).item()
            tok_str = tokenizer.decode([next_t])
            if "<|im_end|>" in tok_str or next_t in (151645, 151643):
                break
            r1_tokens.append(next_t)

        resp1_text = tokenizer.decode(r1_tokens).strip()
        print(f"HoloLLM-v2 (Respuesta 1): \"{resp1_text}\"\n")

        # 2. Turno 2: Preguntar por la memoria del turno anterior
        print(f"Usuario (Turno 2): \"{u2}\"")
        t2_prompt = f"<|im_start|>user\n{u2}<|im_end|>\n<|im_start|>assistant\n"
        t2_tokens = tokenizer.encode(t2_prompt)

        # Absorbemos el nuevo turno SOBRE EL ESTADO VIVO "states" (sin reenviar el turno 1)
        logits2, states, pos = model.absorb_context(t2_tokens, start_pos=pos, states=states, device=device)
        next_t = torch.argmax(logits2).item()
        r2_tokens = [next_t]

        # Generar respuesta del Turno 2
        for _ in range(70):
            cur_pos = pos
            token_tensor = torch.tensor([[next_t]], device=device)
            h = model.token_emb(token_tensor)

            next_states = []
            for block, s in zip(model.blocks, states):
                h, next_s = block.step(h, pos_t=cur_pos, state=s)
                next_states.append(next_s)
            states = next_states
            pos += 1

            logits = model.lm_head(model.ln_f(h))
            next_t = torch.argmax(logits[0, -1, :]).item()
            tok_str = tokenizer.decode([next_t])
            if "<|im_end|>" in tok_str or next_t in (151645, 151643):
                break
            r2_tokens.append(next_t)

        resp2_text = tokenizer.decode(r2_tokens).strip()
        print(f"HoloLLM-v2 (Respuesta 2): \"{resp2_text}\"")

    print("\n" + "=" * 80)
    print("AUDITORÍA DE MEMORIA PERSISTENTE COMPLETADA.")
    print("=" * 80)


if __name__ == "__main__":
    main()
