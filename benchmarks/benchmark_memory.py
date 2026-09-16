"""
Script: benchmarks/benchmark_memory.py
Propósito: Evaluar la retención asociativa de largo alcance de HAM:
1. Auto-recuperación temporal con decaimiento (Pribram / Bohm).
2. Hetero-asociación Clave -> Valor (Tony Plate HRR).
3. Comparativa de escalado de memoria frente a Transformer KV-Cache.
"""

import math
import torch
import torch.nn as nn
from src.layers.associative_memory import HolographicAssociativeMemory

def run_retrieval_experiment() -> None:
    print("\n=======================================================")
    print("  EXPERIMENTO 1: AUTO-RECUPERACIÓN TEMPORAL (PRIBRAM) ")
    print("=======================================================")

    torch.manual_seed(42)
    dim = 128
    seq_len = 30
    batch_size = 1
    decay = 0.98

    ham = HolographicAssociativeMemory(dim=dim, decay=decay, learnable_decay=False)

    # Identidad en proyecciones para aislar la física del Orden Implicado
    with torch.no_grad():
        ham.proj_q.weight.copy_(torch.eye(dim))
        ham.proj_k.weight.copy_(torch.eye(dim))
        ham.proj_v.weight.copy_(torch.eye(dim))

    # Secuencia con ruido gaussiano leve de fondo
    x = torch.randn(batch_size, seq_len, dim) * 0.05

    # Evento específico almacenado en t=3
    secret_event = torch.randn(dim)
    x[0, 3, :] = secret_event

    _, final_memory = ham(x)

    # Consultamos en t=30 con la firma del evento
    q_probe = ham._normalize(secret_event.unsqueeze(0))
    retrieved = ham._unbind(final_memory, q_probe)

    cos_sim = torch.cosine_similarity(retrieved, secret_event.unsqueeze(0), dim=-1).item()
    expected_decay = (decay ** (seq_len - 3)) * (1.0 / math.sqrt(2))

    print(f"Evento fijado en t=3 | Consultado en t={seq_len} (27 pasos de retención)")
    print(f"Similitud Coseno obtenida: {cos_sim:.4f}")
    print(f"Límite teórico con decaimiento (gamma^27 * 0.707): ~{expected_decay:.4f}")

    assert cos_sim > 0.30, "Fallo en la auto-recuperación temporal."
    print("✔ Éxito: La traza sobrevivió a 27 pasos de perturbación y decaimiento.")

def run_hetero_associative_experiment() -> None:
    print("\n=======================================================")
    print("  EXPERIMENTO 2: ASOCIACIÓN CLAVE -> VALOR (PLATE HRR) ")
    print("=======================================================")

    torch.manual_seed(1337)
    dim = 128
    half = dim // 2
    decay = 0.99

    ham = HolographicAssociativeMemory(dim=dim, decay=decay, learnable_decay=False)

    # Configuramos W_k para leer el canal de claves y W_v para leer el de valores
    W_k = torch.zeros(dim, dim)
    W_v = torch.zeros(dim, dim)
    W_k[:half, :half] = torch.eye(half)  # Lee subespacio de claves
    W_v[:half, half:] = torch.eye(half)  # Lee subespacio de valores

    with torch.no_grad():
        ham.proj_k.weight.copy_(W_k)
        ham.proj_v.weight.copy_(W_v)
        ham.proj_q.weight.copy_(W_k)  # La consulta busca en el espacio de claves

    # Par Clave-Valor independiente
    key = torch.randn(half)
    value = torch.randn(half)

    # Token compuesto [clave, valor] en t=3
    x = torch.zeros(1, 20, dim)
    x[0, 3, :half] = key
    x[0, 3, half:] = value

    _, final_memory = ham(x)

    # En t=20, presentamos SOLO la clave como consulta
    query_token = torch.zeros(1, dim)
    query_token[0, :half] = key

    # Proyección y desenfundado
    Q = ham._normalize(ham.proj_q(query_token))
    retrieved_full = ham._unbind(final_memory, Q)
    retrieved_value = retrieved_full[0, :half]

    cos_sim = torch.cosine_similarity(retrieved_value.unsqueeze(0), value.unsqueeze(0), dim=-1).item()
    print(f"Par (K, V) almacenado en t=3. En t=20 consultamos SOLO con K.")
    print(f"Similitud Coseno con el Valor V secreto: {cos_sim:.4f}")
    assert cos_sim > 0.50, "Fallo en la hetero-asociación clave-valor."
    print("✔ Éxito: La clave extrajo exitosamente el valor complementario de la memoria acumulada.")

def print_memory_scaling_comparison() -> None:
    print("\n=======================================================")
    print("  COMPARATIVA TEÓRICA DE ESCALADO DE MEMORIA           ")
    print("=======================================================")
    print(f"{'Contexto (Tokens)':<20} | {'Transformer KV-Cache':<22} | {'Holographic Memory (HAM)':<24}")
    print("-" * 72)

    dim = 4096
    layers = 32
    bytes_per_elem = 2

    contexts = [512, 2048, 8192, 32768, 131072]
    for ctx in contexts:
        t_mb = (2 * layers * ctx * dim * bytes_per_elem) / (1024 ** 2)
        h_mb = (layers * dim * bytes_per_elem) / (1024 ** 2)
        print(f"{ctx:<20} | {t_mb:>16.2f} MB | {h_mb:>18.2f} MB  (O(1) CONSTANTE)")

if __name__ == "__main__":
    run_retrieval_experiment()
    run_hetero_associative_experiment()
    print_memory_scaling_comparison()
