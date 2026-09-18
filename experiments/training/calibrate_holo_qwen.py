r"""
SCRIPT: calibrate_holo_qwen.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM) - Calibración de Transmutación
DESCRIPCIÓN: Calibra HoloQwen-1.5B con Gradient Accumulation (B=2, Accum=4) para operar
             en menos de 18 GB de VRAM en la A100, congelando el 83% pre-entrenado.
"""

import os
import sys
import time
import math
import random
import json
import hashlib
from typing import List, Dict, Any, Generator, Tuple

# Prevenir fragmentación de memoria en PyTorch CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.holo_qwen import HoloQwenForCausalLM
from src.layers.phase_disentangler import HolographicPhaseDisentangler

CONFIG = {
    "seed": 42,
    "teacher_id": "Qwen/Qwen2.5-Coder-1.5B",
    "init_checkpoint": "checkpoints/holo_qwen_1.5b_transmuted.pt",
    "output_checkpoint": "checkpoints/holo_qwen_1.5b_calibrated.pt",
    "vocab_size": 151936,
    "dim": 1536,
    "depth": 28,
    "num_heads": 12,
    "seq_len": 512,
    "micro_batch_size": 2,     # Micro-batch ligero: reduce memoria de logits a 620 MB
    "accum_steps": 4,          # 2 x 4 = Batch efectivo de 8 (4096 tokens por paso)
    "max_steps": 1500,
    "lr": 3e-4,
    "min_lr": 1e-5,
    "warmup_steps": 100,
    "temperature": 1.0,
    "alpha_distill": 0.5,
    "beta_phase": 0.03,
    "save_interval": 300,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

CANONICAL_GEMS = [
    ("factorial", "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
    ("fibonacci with memoization", "def fibonacci(n: int, memo: dict = None) -> int:\n    if memo is None:\n        memo = {}\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)\n    return memo[n]"),
    ("binary search", "def binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"),
    ("reverse linked list", "def reverse_list(head):\n    prev = None\n    curr = head\n    while curr is not None:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev"),
    ("merge sort", "def merge_sort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    res, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            res.append(left[i]); i += 1\n        else:\n            res.append(right[j]); j += 1\n    res.extend(left[i:]); res.extend(right[j:])\n    return res"),
    ("breadth first search bfs", "def bfs(graph: dict, start: str) -> list[str]:\n    visited = {start}\n    queue = [start]\n    res = []\n    while queue:\n        node = queue.pop(0)\n        res.append(node)\n        for nxt in graph.get(node, []):\n            if nxt not in visited:\n                visited.add(nxt)\n                queue.append(nxt)\n    return res"),
    ("maximum subarray kadane", "def max_subarray(nums: list[int]) -> int:\n    max_cur = max_glo = nums[0]\n    for x in nums[1:]:\n        max_cur = max(x, max_cur + x)\n        if max_cur > max_glo:\n            max_glo = max_cur\n    return max_glo"),
    ("valid parentheses check", "def is_valid_parentheses(s: str) -> bool:\n    stack = []\n    match = {')': '(', '}': '{', ']': '['}\n    for ch in s:\n        if ch in match:\n            if not stack or stack.pop() != match[ch]:\n                return False\n        else:\n            stack.append(ch)\n    return len(stack) == 0"),
    ("quick select kth smallest", "def quickselect(arr: list[int], k: int) -> int:\n    pivot = arr[len(arr) // 2]\n    left = [x for x if x < pivot]\n    mid = [x for x if x == pivot]\n    right = [x for x if x > pivot]\n    if k < len(left):\n        return quickselect(left, k)\n    elif k < len(left) + len(mid):\n        return pivot\n    return quickselect(right, k - len(left) - len(mid))"),
    ("dijkstra shortest path", "import heapq\ndef dijkstra(graph: dict, src: str) -> dict:\n    dist = {n: float('inf') for n in graph}\n    dist[src] = 0\n    pq = [(0, src)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]:\n            continue\n        for v, w in graph[u].items():\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heapq.heappush(pq, (dist[v], v))\n    return dist")
]


def load_hybrid_dataset() -> List[Tuple[str, str]]:
    data_path = "data/code_alpaca_full.json"
    dataset = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            inst = item.get("instruction", "").strip()
            code = item.get("output", "").strip()
            if "def " in code and 30 < len(code) < 400:
                dataset.append((inst, code))
        print(f"[Dataset] Muestras de CodeAlpaca cargadas: {len(dataset)}")

    for _ in range(80):
        dataset.extend(CANONICAL_GEMS)

    random.shuffle(dataset)
    print(f"[Dataset] Total muestras en corpus híbrido: {len(dataset)}")
    return dataset


def streaming_generator(
    samples: List[Tuple[str, str]],
    tokenizer,
    batch_size: int,
    seq_len: int
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    num_samples = len(samples)
    idx = 0

    while True:
        batch_inputs = []
        batch_targets = []

        for _ in range(batch_size):
            inst, code = samples[idx % num_samples]
            idx += 1

            p_str = f"<|im_start|>user\nWrite a python function for the following task:\n{inst}<|im_end|>\n<|im_start|>assistant\n```python\n"
            r_str = f"{code}\n```<|im_end|>"

            p_ids = tokenizer.encode(p_str, add_special_tokens=False)
            r_ids = tokenizer.encode(r_str, add_special_tokens=False)

            total_ids = p_ids + r_ids
            target_ids = [-100] * len(p_ids) + r_ids

            if len(total_ids) > seq_len:
                total_ids = total_ids[:seq_len]
                target_ids = target_ids[:seq_len]
            else:
                pad_len = seq_len - len(total_ids)
                total_ids = total_ids + [pad_id] * pad_len
                target_ids = target_ids + [-100] * pad_len

            batch_inputs.append(total_ids)
            batch_targets.append(target_ids)

        yield (
            torch.tensor(batch_inputs, dtype=torch.long),
            torch.tensor(batch_targets, dtype=torch.long)
        )


def compute_weights_sha256(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for p in model.parameters():
        hasher.update(p.detach().cpu().float().numpy().tobytes())
    return hasher.hexdigest()


def main() -> None:
    torch.manual_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    print("=" * 80)
    print(f"HOLOQWEN-1.5B: CALIBRACIÓN HOLOGRÁFICA OPTIMIZADA (VRAM < 18 GB)")
    print(f"Micro-Batch: {CONFIG['micro_batch_size']} | Acumulación: {CONFIG['accum_steps']} (Batch Efectivo: 8)")
    print("=" * 80)

    # 1. Cargar Maestro Qwen original (Solo inferencia)
    print(f"[Teacher] Cargando {CONFIG['teacher_id']} desde cache local...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["teacher_id"], trust_remote_code=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            CONFIG["teacher_id"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True
        ).to(device).eval()
    except Exception:
        teacher = AutoModelForCausalLM.from_pretrained(
            CONFIG["teacher_id"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device).eval()

    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[Teacher] Maestro Qwen activo en VRAM (3.2 GB).")

    # 2. Cargar Alumno Transmutado HoloQwen-1.5B
    print(f"[Student] Cargando HoloQwen-1.5B transmutado desde {CONFIG['init_checkpoint']}...")
    student = HoloQwenForCausalLM(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"]
    ).to(device=device, dtype=torch.bfloat16)

    ckpt = torch.load(CONFIG["init_checkpoint"], map_location=device)
    student.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print(f"✔ Pesos transmutados inyectados con éxito.")

    # Congelamiento: solo entrenamos la atención holográfica (16.9%)
    trainable_params = 0
    frozen_params = 0
    for name, param in student.named_parameters():
        if "attn" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()

    print(f"\n[Arquitectura de Gradientes]")
    print(f"  ❄ Parámetros Congelados (Embeddings, RMSNorm, MLPs): {frozen_params / 1e6:.1f} M ({frozen_params/(trainable_params+frozen_params)*100:.1f}%)")
    print(f"  🔥 Parámetros a Calibrar (Atención Holográfica O(1)): {trainable_params / 1e6:.1f} M ({trainable_params/(trainable_params+frozen_params)*100:.1f}%)\n")

    # 3. Hooks Espectrales
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

    hook_handles = [
        m.register_forward_hook(make_k_hook())
        for n, m in student.named_modules()
        if isinstance(m, nn.Linear) and "k_proj" in n
    ]

    dataset = load_hybrid_dataset()
    data_gen = streaming_generator(dataset, tokenizer, CONFIG["micro_batch_size"], CONFIG["seq_len"])

    disentangler = HolographicPhaseDisentangler().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=CONFIG["lr"],
        betas=(0.9, 0.98),
        weight_decay=0.01
    )

    def get_lr(step: int) -> float:
        if step < CONFIG["warmup_steps"]:
            return CONFIG["lr"] * (step + 1) / CONFIG["warmup_steps"]
        progress = (step - CONFIG["warmup_steps"]) / (CONFIG["max_steps"] - CONFIG["warmup_steps"])
        return CONFIG["min_lr"] + 0.5 * (CONFIG["lr"] - CONFIG["min_lr"]) * (1.0 + math.cos(math.pi * progress))

    print("=" * 80)
    print("INICIO DE CALIBRACIÓN HOLOGRÁFICA (1500 PASOS CON GRADIENT ACCUMULATION)")
    print("=" * 80)

    start_time = time.time()
    effective_tokens_per_step = CONFIG["micro_batch_size"] * CONFIG["accum_steps"] * CONFIG["seq_len"]

    for step in range(1, CONFIG["max_steps"] + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        
        step_loss = 0.0
        step_kd = 0.0
        step_ce = 0.0
        step_phase = 0.0
        step_max_coh = 0.0

        for micro in range(CONFIG["accum_steps"]):
            batch_in, batch_tgt = next(data_gen)
            inputs = batch_in.to(device)[:, :-1].contiguous()
            targets = batch_tgt.to(device)[:, 1:].contiguous()

            captured_keys.clear()

            with torch.inference_mode():
                teacher_logits = teacher(inputs).logits

            student.train()
            student_logits, _ = student(inputs)

            loss_mask = (targets != -100).float()
            num_valid = torch.clamp(loss_mask.sum(), min=1.0)

            # 1. Cross-Entropy en float32
            loss_ce = F.cross_entropy(
                student_logits.float().view(-1, student_logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

            # 2. Divergencia KL en float32 con tensores ligeros (B=2)
            t_temp = CONFIG["temperature"]
            p_s = F.log_softmax(student_logits.float() / t_temp, dim=-1)
            p_t = F.softmax(teacher_logits.float() / t_temp, dim=-1)
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

            # Escalar pérdida por los pasos de acumulación
            scaled_loss = total_loss / CONFIG["accum_steps"]
            scaled_loss.backward()

            step_loss += total_loss.item() / CONFIG["accum_steps"]
            step_kd += loss_kd.item() / CONFIG["accum_steps"]
            step_ce += loss_ce.item() / CONFIG["accum_steps"]
            step_phase += loss_phase.item() / CONFIG["accum_steps"]
            step_max_coh += max_coh / CONFIG["accum_steps"]

        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1:
            elapsed = max(time.time() - start_time, 1e-4)
            tok_s = (step * effective_tokens_per_step) / elapsed
            print(f"Paso [{step:04d}/{CONFIG['max_steps']:04d}] | "
                  f"Total: {step_loss:.4f} | "
                  f"KD: {step_kd:.4f} | "
                  f"CE: {step_ce:.4f} | "
                  f"Fase: {step_phase:.4f} | "
                  f"MaxCoh: {step_max_coh:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tok/s: {tok_s:.0f}", flush=True)

        if step % CONFIG["save_interval"] == 0 or step == CONFIG["max_steps"]:
            os.makedirs(os.path.dirname(CONFIG["output_checkpoint"]), exist_ok=True)
            sha = compute_weights_sha256(student)
            torch.save({
                "step": step,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "weights_sha256": sha,
                "config": CONFIG,
                "timestamp": time.time()
            }, CONFIG["output_checkpoint"])
            print(f"\n>>> Checkpoint Calibrado guardado: {CONFIG['output_checkpoint']} (SHA: {sha[:12]})\n", flush=True)

    for h in hook_handles:
        h.remove()
    print("[HoloQwen-1.5B] Calibración Stage 1 completada exitosamente.", flush=True)


if __name__ == "__main__":
    main()
