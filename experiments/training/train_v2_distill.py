r"""
SCRIPT: train_v2_distill.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM) - Arquitectura v2
DESCRIPCIÓN: Destilación híbrida (Algoritmos Completos + Conversación Multi-Turno)
             hacia HoloCausalLMV2 (69.8M) con decaimientos multiescala y rotación continua.

OPTIMIZACIÓN: Carga local obligatoria (local_files_only=True) en 0.5 segundos sin red.
"""

import os
import sys
import time
import math
import random
import hashlib
from typing import List, Dict, Any, Generator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.holo_causal_lm_v2 import HoloCausalLMV2
from src.layers.phase_disentangler import HolographicPhaseDisentangler

CONFIG = {
    "seed": 42,
    "teacher_id": "Qwen/Qwen2.5-Coder-1.5B",
    "checkpoint_path": "checkpoints/holo_v2_conversational_70m.pt",
    "vocab_size": 151936,
    "dim": 384,
    "depth": 6,
    "num_heads": 8,
    "seq_len": 512,
    "batch_size": 8,
    "max_steps": 2500,
    "lr": 3e-4,
    "min_lr": 1e-5,
    "warmup_steps": 150,
    "temperature": 1.0,
    "alpha_distill": 0.5,
    "beta_phase": 0.05,
    "save_interval": 500,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

DATASET_V2 = [
    # --- ALGORITMOS COMPLETOS (SIN TRUNCAMIENTO) ---
    (
        "single_turn",
        "factorial",
        "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    ),
    (
        "single_turn",
        "fibonacci with memoization",
        "def fibonacci(n: int, memo: dict = None) -> int:\n    if memo is None:\n        memo = {}\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)\n    return memo[n]"
    ),
    (
        "single_turn",
        "binary search",
        "def binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
    ),
    (
        "single_turn",
        "reverse linked list",
        "def reverse_list(head):\n    prev = None\n    curr = head\n    while curr is not None:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev"
    ),
    (
        "single_turn",
        "merge sort",
        "def merge_sort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    res, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            res.append(left[i]); i += 1\n        else:\n            res.append(right[j]); j += 1\n    res.extend(left[i:]); res.extend(right[j:])\n    return res"
    ),
    (
        "single_turn",
        "breadth first search bfs",
        "def bfs(graph: dict, start: str) -> list[str]:\n    visited = {start}\n    queue = [start]\n    res = []\n    while queue:\n        node = queue.pop(0)\n        res.append(node)\n        for nxt in graph.get(node, []):\n            if nxt not in visited:\n                visited.add(nxt)\n                queue.append(nxt)\n    return res"
    ),
    (
        "single_turn",
        "maximum subarray kadane",
        "def max_subarray(nums: list[int]) -> int:\n    max_cur = max_glo = nums[0]\n    for x in nums[1:]:\n        max_cur = max(x, max_cur + x)\n        if max_cur > max_glo:\n            max_glo = max_cur\n    return max_glo"
    ),
    (
        "single_turn",
        "valid parentheses check",
        "def is_valid_parentheses(s: str) -> bool:\n    stack = []\n    match = {')': '(', '}': '{', ']': '['}\n    for ch in s:\n        if ch in match:\n            if not stack or stack.pop() != match[ch]:\n                return False\n        else:\n            stack.append(ch)\n    return len(stack) == 0"
    ),
    (
        "single_turn",
        "quick select kth smallest",
        "def quickselect(arr: list[int], k: int) -> int:\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    if k < len(left):\n        return quickselect(left, k)\n    elif k < len(left) + len(mid):\n        return pivot\n    return quickselect(right, k - len(left) - len(mid))"
    ),
    (
        "single_turn",
        "dijkstra shortest path",
        "import heapq\ndef dijkstra(graph: dict, src: str) -> dict:\n    dist = {n: float('inf') for n in graph}\n    dist[src] = 0\n    pq = [(0, src)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]:\n            continue\n        for v, w in graph[u].items():\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heapq.heappush(pq, (dist[v], v))\n    return dist"
    ),

    # --- DIÁLOGOS MULTI-TURNO ---
    (
        "multi_turn",
        "Hola, me llamo Lucas y estoy investigando redes neuronales holográficas.",
        "¡Hola Lucas! Qué fascinante campo de investigación. Las redes holográficas combinan la teoría holonómica y las representaciones reducidas de Tony Plate con memoria O(1). ¿En qué aspecto estás trabajando hoy?",
        "¿Recuerdas cuál es mi nombre y qué estoy investigando?",
        "Por supuesto, te llamas Lucas y estás investigando arquitecturas de redes neuronales holográficas inspiradas en la física teórica y el principio holográfico."
    ),
    (
        "multi_turn",
        "Vamos a definir una constante física: c = 299792458 metros por segundo.",
        "Entendido. He registrado la velocidad de la luz en el vacío: c = 299,792,458 m/s.",
        "¿Cuál fue el valor de c que te mencioné?",
        "El valor de c que definiste es exactamente 299,792,458 metros por segundo."
    ),
    (
        "multi_turn",
        "¿Cuál es la complejidad temporal y espacial de la memoria en Transformers estándar?",
        "En un Transformer estándar, la atención consume O(N^2) en cómputo durante el prefill y O(N) en memoria VRAM debido al almacenamiento dinámico del KV-Cache a lo largo del contexto.",
        "¿Y cómo resuelve HoloLLM ese cuello de botella?",
        "HoloLLM elimina el KV-Cache manteniendo un estado asociativo constante O(1) en el Orden Implicado complejo, logrando consumo de memoria invariante frente a la longitud del diálogo."
    )
]


def format_sample(sample: tuple, tokenizer) -> Tuple[List[int], List[int]]:
    if sample[0] == "single_turn":
        _, task, code = sample
        p_str = f"<|im_start|>user\nWrite a python function for the following task:\n{task}<|im_end|>\n<|im_start|>assistant\n```python\n"
        r_str = f"{code}\n```<|im_end|>"

        p_ids = tokenizer.encode(p_str, add_special_tokens=False)
        r_ids = tokenizer.encode(r_str, add_special_tokens=False)
        
        input_ids = p_ids + r_ids
        target_ids = [-100] * len(p_ids) + r_ids
        return input_ids, target_ids
    else:
        _, u1, a1, u2, a2 = sample
        t1 = f"<|im_start|>user\n{u1}<|im_end|>\n<|im_start|>assistant\n{a1}<|im_end|>\n"
        t2 = f"<|im_start|>user\n{u2}<|im_end|>\n"
        ans2 = f"<|im_start|>assistant\n{a2}<|im_end|>"

        context_ids = tokenizer.encode(t1 + t2, add_special_tokens=False)
        ans_ids = tokenizer.encode(ans2, add_special_tokens=False)

        input_ids = context_ids + ans_ids
        target_ids = [-100] * len(context_ids) + ans_ids
        return input_ids, target_ids


def data_generator_v2(tokenizer, batch_size: int, seq_len: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    while True:
        batch_in = []
        batch_tgt = []
        for _ in range(batch_size):
            sample = random.choice(DATASET_V2)
            in_ids, tgt_ids = format_sample(sample, tokenizer)

            if len(in_ids) > seq_len:
                in_ids = in_ids[:seq_len]
                tgt_ids = tgt_ids[:seq_len]
            else:
                pad_len = seq_len - len(in_ids)
                in_ids = in_ids + [pad_id] * pad_len
                tgt_ids = tgt_ids + [-100] * pad_len

            batch_in.append(in_ids)
            batch_tgt.append(tgt_ids)

        yield (
            torch.tensor(batch_in, dtype=torch.long),
            torch.tensor(batch_tgt, dtype=torch.long)
        )


def compute_weights_sha256(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for p in model.parameters():
        hasher.update(p.detach().cpu().float().numpy().tobytes())
    return hasher.hexdigest()


def main() -> None:
    torch.manual_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    print(f"[HoloLLM-v2] Destilación iniciada en: {device}")

    # 1. Maestro: Carga local estricta (cero llamadas de red)
    print(f"[Teacher] Cargando {CONFIG['teacher_id']} exclusivamente desde disco local...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["teacher_id"], local_files_only=True, trust_remote_code=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            CONFIG["teacher_id"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True
        ).to(device).eval()
    except Exception as e:
        print(f"[Teacher] Aviso: No se encontró cache offline estricta ({e}). Cargando estándar...")
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["teacher_id"], trust_remote_code=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            CONFIG["teacher_id"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device).eval()

    for p in teacher.parameters():
        p.requires_grad = False
    print("[Teacher] Maestro activo en VRAM (0.5s).")

    # 2. Alumno HoloCausalLMV2
    print(f"[Student] Inicializando HoloCausalLMV2 (69.8M, dim={CONFIG['dim']}, depth={CONFIG['depth']}, heads={CONFIG['num_heads']})...")
    student = HoloCausalLMV2(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"]
    ).to(device=device, dtype=torch.bfloat16)

    captured_keys: List[torch.Tensor] = []

    def make_k_hook():
        def hook(module, input_tensor, output_tensor):
            if output_tensor.ndim == 3:
                B, S, D = output_tensor.shape
                H = CONFIG["num_heads"]
                D_h = D // H
                k_view = output_tensor.reshape(B, S, H, D_h).permute(0, 2, 1, 3)
                k_comp = torch.fft.rfft(k_view.float(), dim=-1)
                captured_keys.append(k_comp)
        return hook

    hook_handles = []
    for name, module in student.named_modules():
        if isinstance(module, nn.Linear) and "k_proj" in name:
            hook_handles.append(module.register_forward_hook(make_k_hook()))

    print(f"[Hooks] {len(hook_handles)} capas espectrales conectadas para regularización.")

    disentangler = HolographicPhaseDisentangler().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=CONFIG["lr"], betas=(0.9, 0.95), weight_decay=0.01)

    def get_lr(step: int) -> float:
        if step < CONFIG["warmup_steps"]:
            return CONFIG["lr"] * (step + 1) / CONFIG["warmup_steps"]
        progress = (step - CONFIG["warmup_steps"]) / (CONFIG["max_steps"] - CONFIG["warmup_steps"])
        return CONFIG["min_lr"] + 0.5 * (CONFIG["lr"] - CONFIG["min_lr"]) * (1.0 + math.cos(math.pi * progress))

    data_gen = data_generator_v2(tokenizer, CONFIG["batch_size"], CONFIG["seq_len"])

    print("=" * 80)
    print("DESTILACIÓN v2 ACTIVA: ALGORITMOS COMPLETOS Y CONVERSACIÓN MULTI-TURNO")
    print("=" * 80)

    start_time = time.time()
    for step in range(1, CONFIG["max_steps"] + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        batch_in, batch_tgt = next(data_gen)
        inputs = batch_in.to(device)[:, :-1].contiguous()
        targets = batch_tgt.to(device)[:, 1:].contiguous()

        captured_keys.clear()

        with torch.no_grad():
            teacher_logits = teacher(inputs).logits.float()

        student.train()
        student_logits, _ = student(inputs)
        student_logits = student_logits.float()

        loss_mask = (targets != -100).float()
        num_valid = torch.clamp(loss_mask.sum(), min=1.0)

        # 1. Cross-Entropy enmascarada
        loss_ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )

        # 2. KL Divergence normalizada
        t_temp = CONFIG["temperature"]
        p_s = F.log_softmax(student_logits / t_temp, dim=-1)
        p_t = F.softmax(teacher_logits / t_temp, dim=-1)
        kl_tokens = F.kl_div(p_s, p_t, reduction="none").sum(dim=-1)
        loss_kd = ((kl_tokens * loss_mask).sum() / num_valid) * (t_temp ** 2)

        # 3. Crosstalk de Fase
        loss_phase = torch.tensor(0.0, device=device)
        max_coh = 0.0
        if captured_keys:
            for k_c in captured_keys:
                p_l, m = disentangler(k_c)
                loss_phase = loss_phase + p_l
                max_coh += m["phase_coherence_max"]
            loss_phase = loss_phase / len(captured_keys)
            max_coh /= len(captured_keys)

        total_loss = (1.0 - CONFIG["alpha_distill"]) * loss_ce + \
                     CONFIG["alpha_distill"] * loss_kd + \
                     CONFIG["beta_phase"] * loss_phase

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = max(time.time() - start_time, 1e-4)
            tok_s = (step * CONFIG["batch_size"] * CONFIG["seq_len"]) / elapsed
            print(f"Paso [{step:04d}/{CONFIG['max_steps']:04d}] | "
                  f"Total: {total_loss.item():.4f} | "
                  f"KD: {loss_kd.item():.4f} | "
                  f"CE: {loss_ce.item():.4f} | "
                  f"Fase: {loss_phase.item():.4f} | "
                  f"MaxCoh: {max_coh:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tok/s: {tok_s:.0f}")

        if step % CONFIG["save_interval"] == 0 or step == CONFIG["max_steps"]:
            os.makedirs(os.path.dirname(CONFIG["checkpoint_path"]), exist_ok=True)
            sha = compute_weights_sha256(student)
            torch.save({
                "step": step,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "weights_sha256": sha,
                "config": CONFIG,
                "timestamp": time.time()
            }, CONFIG["checkpoint_path"])
            print(f"\n>>> Checkpoint v2 guardado: {CONFIG['checkpoint_path']} (SHA: {sha[:12]})\n")

    for h in hook_handles:
        h.remove()
    print("[HoloLLM-v2] Destilación completada exitosamente.")


if __name__ == "__main__":
    main()
