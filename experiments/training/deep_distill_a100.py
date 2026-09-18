r"""
SCRIPT: deep_distill_a100.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM)
DESCRIPCIÓN: Destilación enmascarada ChatML hacia HoloCausalLM (66.4M) con normalización
             por token y supresión de crosstalk espectral en el toro unitario.
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

from src.models.holo_causal_lm import HoloCausalLM
from src.layers.phase_disentangler import HolographicPhaseDisentangler

CONFIG = {
    "seed": 42,
    "teacher_id": "Qwen/Qwen2.5-Coder-1.5B",
    "checkpoint_path": "checkpoints/holo_deep_distilled_75m.pt",
    "vocab_size": 151936,
    "dim": 384,
    "depth": 6,
    "num_heads": 8,
    "seq_len": 172,
    "batch_size": 16,
    "max_steps": 2500,
    "lr": 2e-4,
    "min_lr": 1e-5,
    "warmup_steps": 100,
    "temperature": 1.0,
    "alpha_distill": 0.5,
    "beta_phase": 0.05,
    "save_interval": 500,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

ALGO_DATASET = [
    (
        "factorial",
        "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    ),
    (
        "fibonacci with memoization",
        "def fibonacci(n: int, memo: dict = None) -> int:\n    if memo is None:\n        memo = {}\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)\n    return memo[n]"
    ),
    (
        "binary search",
        "def binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
    ),
    (
        "reverse linked list",
        "def reverse_list(head):\n    prev = None\n    curr = head\n    while curr is not None:\n        nxt = curr.next\n        curr.next = prev\n        prev = curr\n        curr = nxt\n    return prev"
    ),
    (
        "merge sort",
        "def merge_sort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    res, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            res.append(left[i]); i += 1\n        else:\n            res.append(right[j]); j += 1\n    res.extend(left[i:]); res.extend(right[j:])\n    return res"
    ),
    (
        "breadth first search bfs",
        "def bfs(graph: dict, start: str) -> list[str]:\n    visited = {start}\n    queue = [start]\n    res = []\n    while queue:\n        node = queue.pop(0)\n        res.append(node)\n        for nxt in graph.get(node, []):\n            if nxt not in visited:\n                visited.add(nxt)\n                queue.append(nxt)\n    return res"
    ),
    (
        "maximum subarray kadane",
        "def max_subarray(nums: list[int]) -> int:\n    max_cur = max_glo = nums[0]\n    for x in nums[1:]:\n        max_cur = max(x, max_cur + x)\n        if max_cur > max_glo:\n            max_glo = max_cur\n    return max_glo"
    ),
    (
        "valid parentheses check",
        "def is_valid_parentheses(s: str) -> bool:\n    stack = []\n    match = {')': '(', '}': '{', ']': '['}\n    for ch in s:\n        if ch in match:\n            if not stack or stack.pop() != match[ch]:\n                return False\n        else:\n            stack.append(ch)\n    return len(stack) == 0"
    ),
    (
        "quick select kth smallest",
        "def quickselect(arr: list[int], k: int) -> int:\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    if k < len(left):\n        return quickselect(left, k)\n    elif k < len(left) + len(mid):\n        return pivot\n    return quickselect(right, k - len(left) - len(mid))"
    ),
    (
        "dijkstra shortest path",
        "import heapq\ndef dijkstra(graph: dict, src: str) -> dict:\n    dist = {n: float('inf') for n in graph}\n    dist[src] = 0\n    pq = [(0, src)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]:\n            continue\n        for v, w in graph[u].items():\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heapq.heappush(pq, (dist[v], v))\n    return dist"
    )
]


def chatml_data_generator(tokenizer, batch_size: int, seq_len: int) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    while True:
        batch_inputs = []
        batch_targets = []
        for _ in range(batch_size):
            task, code = random.choice(ALGO_DATASET)
            prefix = f"<|im_start|>user\nWrite a python function for the following task:\n{task}<|im_end|>\n<|im_start|>assistant\n```python\n"
            suffix = f"{code}\n```<|im_end|>"

            p_ids = tokenizer.encode(prefix, add_special_tokens=False)
            s_ids = tokenizer.encode(suffix, add_special_tokens=False)

            total_ids = p_ids + s_ids
            target_ids = [-100] * len(p_ids) + s_ids

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
    for param in model.parameters():
        hasher.update(param.detach().cpu().float().numpy().tobytes())
    return hasher.hexdigest()


def main() -> None:
    torch.manual_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    print(f"[HoloLLM] Destilación ChatML en: {device}")

    print(f"[Teacher] Cargando {CONFIG['teacher_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["teacher_id"], trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        CONFIG["teacher_id"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[Student] Inicializando HoloCausalLM (66.4M)...")
    student = HoloCausalLM(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"],
        max_seq_len=CONFIG["seq_len"]
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
        if isinstance(module, nn.Linear) and any(tag in name.lower() for tag in ["w_k", "k_proj", "to_k", "key"]):
            hook_handles.append(module.register_forward_hook(make_k_hook()))

    disentangler = HolographicPhaseDisentangler().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=CONFIG["lr"], betas=(0.9, 0.95), weight_decay=0.01)

    def get_lr(step: int) -> float:
        if step < CONFIG["warmup_steps"]:
            return CONFIG["lr"] * (step + 1) / CONFIG["warmup_steps"]
        progress = (step - CONFIG["warmup_steps"]) / (CONFIG["max_steps"] - CONFIG["warmup_steps"])
        return CONFIG["min_lr"] + 0.5 * (CONFIG["lr"] - CONFIG["min_lr"]) * (1.0 + math.cos(math.pi * progress))

    data_gen = chatml_data_generator(tokenizer, CONFIG["batch_size"], CONFIG["seq_len"])

    print("=" * 80)
    print("INICIO DE ENTRENAMIENTO CHATML + DISENTANGLER")
    print("=" * 80)

    start_time = time.time()
    for step in range(1, CONFIG["max_steps"] + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_batch, target_batch = next(data_gen)
        inputs = input_batch.to(device)[:, :-1].contiguous()
        targets = target_batch.to(device)[:, 1:].contiguous()

        captured_keys.clear()

        with torch.no_grad():
            teacher_logits = teacher(inputs).logits.float()

        student.train()
        student_out = student(inputs)
        student_logits = (student_out[0] if isinstance(student_out, tuple) else student_out).float()

        loss_mask = (targets != -100).float()
        num_valid = torch.clamp(loss_mask.sum(), min=1.0)

        # 1. Cross-Entropy enmascarada
        loss_ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )

        # 2. KL Divergence normalizada por token de código
        t_temp = CONFIG["temperature"]
        p_s = F.log_softmax(student_logits / t_temp, dim=-1)
        p_t = F.softmax(teacher_logits / t_temp, dim=-1)
        kl_tokens = F.kl_div(p_s, p_t, reduction="none").sum(dim=-1)
        loss_kd = ((kl_tokens * loss_mask).sum() / num_valid) * (t_temp ** 2)

        # 3. Crosstalk de Fase en el toro unitario
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
            print(f"\n>>> Checkpoint guardado: {CONFIG['checkpoint_path']} (SHA: {sha[:12]})\n")

    for h in hook_handles:
        h.remove()
    print("[HoloLLM] Destilación completada con éxito.")


if __name__ == "__main__":
    main()
