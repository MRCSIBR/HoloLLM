"""
Script: distill_a100.py
Propósito: Destilación de Conocimiento (Knowledge Distillation) desde Qwen2.5-Coder
hacia el alumno Holográfico (HoloCausalLM O(1)) con vocabulario alineado a 151,936 tokens.
"""

import os
import json
import time
from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.holo_causal_lm import HoloCausalLM
from src.utils.seed import enforce_reproducibility

TEACHER_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA_FILE = "data/code_alpaca_full.json"


def prepare_distillation_dataset(tokenizer, pad_token_id: int, max_samples: int = 3500, max_len: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    print(f"Cargando y formateando {max_samples} muestras para destilación...", flush=True)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)[:max_samples]

    samples = []
    for item in raw:
        inst = item.get("instruction", "").strip()
        code = item.get("output", "").strip()
        if not code or len(code) < 15:
            continue

        text = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{code}<|im_end|>"
        tokens = tokenizer.encode(text, truncation=True, max_length=max_len)
        samples.append(torch.tensor(tokens, dtype=torch.long))

    m_len = max(len(s) for s in samples)
    padded_inputs = torch.full((len(samples), m_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(samples), m_len), -100, dtype=torch.long)

    for i, s in enumerate(samples):
        padded_inputs[i, :len(s)] = s
        padded_labels[i, :len(s)] = s

    return padded_inputs, padded_labels


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.6
) -> torch.Tensor:
    """Calcula KL Divergence sobre tokens reales no enmascarados + CrossEntropy."""
    # Máscara para ignorar tokens de padding
    mask = (labels != -100)
    
    if mask.sum() > 0:
        # Extraer solo logits de tokens activos para ahorrar memoria VRAM
        t_active = teacher_logits[mask] / temperature
        s_active = student_logits[mask] / temperature
        
        p_teacher = F.softmax(t_active, dim=-1)
        log_p_student = F.log_softmax(s_active, dim=-1)
        kl_div = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
    else:
        kl_div = torch.tensor(0.0, device=student_logits.device)

    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    return (alpha * kl_div) + ((1.0 - alpha) * ce_loss)


def main():
    enforce_reproducibility(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("\n=======================================================", flush=True)
    print("  PIPELINE DE DESTILACIÓN HOLOGRÁFICA (TEACHER -> STUDENT)", flush=True)
    print("=======================================================", flush=True)

    # 1. Cargar el Profesor
    print(f"Cargando Profesor: {TEACHER_MODEL_ID} en bfloat16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
    
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda"
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # El tamaño real de vocabulario de Qwen (151,936) para alojar todos los tokens de chat
    aligned_vocab_size = teacher.config.vocab_size
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    print(f"✔ Profesor listo. Vocabulario alineado exacto: {aligned_vocab_size:,} tokens", flush=True)

    # 2. Instanciar Alumno Holográfico (~96.6M parámetros)
    DIM = 384
    DEPTH = 6
    HEADS = 8
    MAX_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1.0e-3

    student = HoloCausalLM(
        vocab_size=aligned_vocab_size,
        dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        max_seq_len=MAX_LEN + 32
    ).to(device)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"✔ Alumno HoloLLM instanciado: {student_params:,} parámetros (~{student_params/1e6:.1f}M)", flush=True)

    # 3. Preparar Dataset con máscara de etiquetas
    inputs, labels = prepare_distillation_dataset(tokenizer, pad_id, max_samples=3500, max_len=MAX_LEN)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    optimizer = optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print(f"\n--- Iniciando ciclo de destilación en A100 ({len(inputs)} ejemplos) ---", flush=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        student.train()
        total_loss = 0.0
        ep_start = time.time()

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            shift_labels = batch_y[:, 1:].contiguous()

            # Inferencia del Profesor (sin gradientes)
            with torch.no_grad():
                t_out = teacher(batch_x)
                teacher_shift_logits = t_out.logits[:, :-1, :].contiguous()

            # Inferencia y optimización del Alumno (bfloat16)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                s_logits, _ = student(batch_x)
                student_shift_logits = s_logits[:, :-1, :].contiguous()

                loss = distillation_loss(
                    student_logits=student_shift_logits,
                    teacher_logits=teacher_shift_logits,
                    labels=shift_labels,
                    temperature=2.0,
                    alpha=0.6
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        ep_loss = total_loss / len(inputs)
        ep_time = time.time() - ep_start
        print(f"Época [{epoch:02d}/{EPOCHS:02d}] | KD Loss: {ep_loss:.4f} | Tiempo: {ep_time:.1f}s/época", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/holo_distilled_student_75m.pt"
    torch.save(student.state_dict(), ckpt_path)
    print(f"\n✔ Checkpoint del Alumno destilado guardado exitosamente en: {ckpt_path}")


if __name__ == "__main__":
    main()
