r"""
SCRIPT: cloud/train_dailydialog_distill.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM) - Cloud Training Suite
DESCRIPCIÓN: Destilación de Diálogos Cotidianos en Inglés (DailyDialog)
             hacia la arquitectura HoloCausalLMV2 con memoria O(1).
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

# Asegurar acceso al paquete src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.holo_causal_lm_v2 import HoloCausalLMV2
from src.layers.phase_disentangler import HolographicPhaseDisentangler

try:
    from datasets import load_dataset
except ImportError:
    print("Instalando datasets de Hugging Face...")
    os.system("pip install datasets")
    from datasets import load_dataset

CONFIG = {
    "seed": 42,
    "teacher_id": "Qwen/Qwen2.5-Coder-1.5B",
    "checkpoint_path": "checkpoints/holo_70m_dailydialog.pt",
    "vocab_size": 151936,
    "dim": 384,
    "depth": 6,
    "num_heads": 8,
    "seq_len": 512,
    "batch_size": 8,
    "max_steps": 2000,
    "lr": 2.5e-4,
    "min_lr": 1e-5,
    "warmup_steps": 100,
    "temperature": 1.0,
    "alpha_distill": 0.6,     # 60% peso a los soft targets del maestro
    "beta_phase": 0.03,
    "save_interval": 500,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_and_format_dailydialog(tokenizer, max_samples: int = 12000) -> List[Tuple[List[int], List[int]]]:
    """Carga DailyDialog y empaqueta conversaciones multi-turno en ChatML con máscara de labels."""
    print(f"[Dataset] Descargando/Cargando DailyDialog desde Hugging Face...")
    ds = load_dataset("daily_dialog", split="train")
    
    formatted_dataset = []
    print(f"[Dataset] Procesando {min(len(ds), max_samples)} conversaciones...")

    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        dialog = item["dialog"]
        if len(dialog) < 2:
            continue

        # Formatear la conversación alternando User y Assistant
        full_text = ""
        for turn_idx, utterance in enumerate(dialog):
            role = "user" if turn_idx % 2 == 0 else "assistant"
            full_text += f"<|im_start|>{role}\n{utterance.strip()}<|im_end|>\n"

        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        if len(tokens) < 10 or len(tokens) > CONFIG["seq_len"]:
            continue

        # Máscara de etiquetas: calcular pérdida únicamente en los turnos del assistant
        labels = []
        is_assistant = False
        sub_tokens = full_text.split("<|im_start|>")
        
        for part in sub_tokens:
            if not part:
                continue
            part_str = "<|im_start|>" + part
            part_ids = tokenizer.encode(part_str, add_special_tokens=False)
            if part.startswith("assistant"):
                labels.extend(part_ids)
            else:
                labels.extend([-100] * len(part_ids))

        labels = labels[:len(tokens)]
        if len(labels) < len(tokens):
            labels.extend([-100] * (len(tokens) - len(labels)))

        formatted_dataset.append((tokens, labels))

    print(f"✔ {len(formatted_dataset)} conversaciones multi-turno procesadas con éxito.")
    return formatted_dataset


def dailydialog_generator(
    dataset: List[Tuple[List[int], List[int]]],
    batch_size: int,
    seq_len: int,
    pad_id: int
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    num_samples = len(dataset)
    idx = 0
    while True:
        batch_in = []
        batch_tgt = []
        for _ in range(batch_size):
            tokens, labels = dataset[idx % num_samples]
            idx += 1

            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
                labels = labels[:seq_len]
            else:
                pad_amount = seq_len - len(tokens)
                tokens = tokens + [pad_id] * pad_amount
                labels = labels + [-100] * pad_amount

            batch_in.append(tokens)
            batch_tgt.append(labels)

        yield (
            torch.tensor(batch_in, dtype=torch.long),
            torch.tensor(batch_tgt, dtype=torch.long)
        )


def main():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    print("=" * 80)
    print("HOLOLLM-v2: ENTRENAMIENTO CONVERSACIONAL (DAILYDIALOG DISTILLATION)")
    print(f"Dispositivo: {device} | Memoria: O(1) Constante (0.00 KB KV-Cache)")
    print("=" * 80)

    # 1. Cargar Maestro
    print(f"[Teacher] Cargando {CONFIG['teacher_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["teacher_id"], trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        CONFIG["teacher_id"],
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("✔ Maestro activo en memoria.")

    # 2. Alumno HoloCausalLMV2
    student = HoloCausalLMV2(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"]
    ).to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)

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

    # Cargar y preparar DailyDialog
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dialogue_data = load_and_format_dailydialog(tokenizer)
    data_gen = dailydialog_generator(dialogue_data, CONFIG["batch_size"], CONFIG["seq_len"], pad_id)

    disentangler = HolographicPhaseDisentangler().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=CONFIG["lr"], betas=(0.9, 0.98), weight_decay=0.01)

    print("=" * 80)
    print("INICIO DE ENTRENAMIENTO: DESTILACIÓN CONVERSACIONAL EN EL TORO UNITARIO")
    print("=" * 80)

    start_time = time.time()
    for step in range(1, CONFIG["max_steps"] + 1):
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

        # Cross-Entropy enmascarada
        loss_ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )

        # KL Divergence supervisada por el maestro
        t_temp = CONFIG["temperature"]
        p_s = F.log_softmax(student_logits / t_temp, dim=-1)
        p_t = F.softmax(teacher_logits / t_temp, dim=-1)
        kl_tokens = F.kl_div(p_s, p_t, reduction="none").sum(dim=-1)
        loss_kd = ((kl_tokens * loss_mask).sum() / num_valid) * (t_temp ** 2)

        # Regularización de fase
        loss_phase = torch.tensor(0.0, device=device)
        if captured_keys:
            for k_c in captured_keys:
                p_l, _ = disentangler(k_c)
                loss_phase = loss_phase + p_l
            loss_phase = loss_phase / len(captured_keys)

        total_loss = (1.0 - CONFIG["alpha_distill"]) * loss_ce + \
                     CONFIG["alpha_distill"] * loss_kd + \
                     CONFIG["beta_phase"] * loss_phase

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1:
            elapsed = max(time.time() - start_time, 1e-4)
            tok_s = (step * CONFIG["batch_size"] * CONFIG["seq_len"]) / elapsed
            print(f"Paso [{step:04d}/{CONFIG['max_steps']:04d}] | "
                  f"Total: {total_loss.item():.4f} | "
                  f"KD: {loss_kd.item():.4f} | "
                  f"CE: {loss_ce.item():.4f} | "
                  f"Fase: {loss_phase.item():.4f} | "
                  f"Tok/s: {tok_s:.0f}", flush=True)

        if step % CONFIG["save_interval"] == 0 or step == CONFIG["max_steps"]:
            os.makedirs(os.path.dirname(CONFIG["checkpoint_path"]), exist_ok=True)
            torch.save({
                "step": step,
                "model_state_dict": student.state_dict(),
                "config": CONFIG,
                "timestamp": time.time()
            }, CONFIG["checkpoint_path"])
            print(f"\n>>> Checkpoint conversacional guardado en: {CONFIG['checkpoint_path']}\n", flush=True)

    for h in hook_handles:
        h.remove()
    print("[HoloLLM] Entrenamiento conversacional completado con éxito.")


if __name__ == "__main__":
    main()
