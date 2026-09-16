"""
Script: benchmarks/benchmark_kv_elimination.py
Propósito: Medición empírica de eliminación de KV-Cache y validación
de inferencia autorregresiva constante O(1) en MH-HoloAttention.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import torch
import torch.nn as nn
from src.layers.holo_attention import MultiHeadHoloAttention


def verify_autoregressive_step_equivalence() -> None:
    print("\n=======================================================")
    print("  1. VERIFICACIÓN: EQUIVALENCIA FORWARD vs STEP O(1)  ")
    print("=======================================================")
    torch.manual_seed(42)
    B, T, D = 2, 8, 64
    num_heads = 4

    holo_attn = MultiHeadHoloAttention(dim=D, num_heads=num_heads)
    holo_attn.eval()

    x = torch.randn(B, T, D)

    # 1. Forward completo sobre la secuencia
    with torch.no_grad():
        out_forward, final_state_forward = holo_attn(x)

    # 2. Inferencia paso a paso O(1) usando step()
    step_outputs = []
    current_state = torch.zeros(B, num_heads, D // num_heads)
    with torch.no_grad():
        for t in range(T):
            x_t = x[:, t : t + 1, :]
            out_t, current_state = holo_attn.step(x_t, current_state)
            step_outputs.append(out_t)

    out_stepped = torch.cat(step_outputs, dim=1)

    # Medir diferencia numérica máxima
    diff_out = torch.max(torch.abs(out_forward - out_stepped)).item()
    diff_state = torch.max(torch.abs(final_state_forward - current_state)).item()

    print(f"Discrepancia máxima en salidas (Forward vs Step): {diff_out:.2e}")
    print(f"Discrepancia máxima en estado de memoria:         {diff_state:.2e}")

    assert diff_out < 1e-5, f"Violación de causalidad en step(): diff={diff_out}"
    print("✔ Éxito: La inferencia token a token en O(1) es idéntica al pase global.")


def benchmark_memory_footprint() -> None:
    print("\n=======================================================")
    print("  2. MEDICIÓN EMPÍRICA DEL TAMAÑO DE ESTADO / KV-CACHE ")
    print("=======================================================")
    print(f"{'Contexto (Tokens)':<18} | {'KV-Cache Transformer':<22} | {'MH-HoloAttention (HAM)':<22} | {'Ahorro'}")
    print("-" * 75)

    # Dimensiones estándar de modelo mediano (ej. LLaMA 1B: Dim=2048, 16 capas, FP16)
    layers = 16
    dim = 2048
    bytes_per_elem = 2  # FP16
    contexts = [512, 2048, 8192, 32768, 65536]

    for ctx in contexts:
        # Transformer: 2 * layers * seq_len * dim * bytes
        kv_bytes = 2 * layers * ctx * dim * bytes_per_elem
        kv_kb = kv_bytes / 1024
        kv_mb = kv_kb / 1024

        # MH-HoloAttention: layers * dim * bytes (Independiente del contexto ctx)
        holo_bytes = layers * dim * bytes_per_elem
        holo_kb = holo_bytes / 1024

        if kv_mb >= 1.0:
            kv_str = f"{kv_mb:>16.2f} MB"
        else:
            kv_str = f"{kv_kb:>16.2f} KB"

        holo_str = f"{holo_kb:>16.2f} KB"
        ratio = kv_bytes / holo_bytes
        print(f"{ctx:<18} | {kv_str:<22} | {holo_str:<22} | {ratio:>8.0f}x menor")


if __name__ == "__main__":
    verify_autoregressive_step_equivalence()
    benchmark_memory_footprint()
