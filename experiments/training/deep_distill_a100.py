"""
Script: deep_distill_a100.py
Propósito: Destilación Profunda de Código (Knowledge Distillation) en NVIDIA A100.
Profesor: Qwen2.5-Coder-1.5B-Instruct en bfloat16.
Alumno: HoloCausalLM (~66.4M parámetros) con memoria O(1) y carga de posiciones adaptativa.
"""

import os
import json
import time
import urllib.request
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.holo_causal_lm import HoloCausalLM
from src.utils.seed import enforce_reproducibility

TEACHER_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA_URL = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"
DATA_FILE = "data/code_alpaca_full.json"


def prepare_algorithmic_dataset(tokenizer, pad_token_id: int, max_samples: int = 10000, max_len: int = 140) -> Tuple[torch.Tensor, torch.Tensor]:
    print(f"Preparando dataset de destilación algorítmica ({max_samples} muestras)...", flush=True)
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print("Descargando corpus de CodeAlpaca...", flush=True)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    else:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

    samples_inputs = []
    samples_labels = []

    for item in data:
        inst = item.get("instruction", "").strip()
        code = item.get("output", "").strip()

        # Filtrar solo funciones de Python válidas y limpias
        if "def " in code and ":" in code and "{" not in code and 20 < len(code) < 320:
            prompt = f"<|im_start|>user\nWrite a python function for the following task:\n{inst}<|im_end|>\n<|im_start|>assistant\n"
            completion = f"```python\n{code}\n```<|im_end|>"

            p_tokens = tokenizer.encode(prompt)
            c_tokens = tokenizer.encode(completion)

            seq = p_tokens + c_tokens
            if len(seq) > max_len:
                seq = seq[:max_len]

            # Loss Masking estricto: -100 en el prompt (solo destilamos el código)
            labels = [-100] * min(len(p_tokens), len(seq)) + seq[len(p_tokens):]

            samples_inputs.append(torch.tensor(seq, dtype=torch.long))
            samples_labels.append(torch.tensor(labels, dtype=torch.long))

        if len(samples_inputs) >= max_samples:
            break

    print(f"✔ Muestras de Python algorítmico curadas: {len(samples_inputs)}", flush=True)

    m_len = max(len(s) for s in samples_inputs)
    pad_inputs = torch.full((len(samples_inputs), m_len), pad_token_id, dtype=torch.long)
    pad_labels = torch.full((len(samples_labels), m_len), -100, dtype=torch.long)

    for i in range(len(samples_inputs)):
        pad_inputs[i, :len(samples_inputs[i])] = samples_inputs[i]
        pad_labels[i, :len(samples_labels[i])] = samples_labels[i]

    return pad_inputs, pad_labels


def masked_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.7
) -> torch.Tensor:
    """Calcula la divergencia KL de Hinton exclusivamente sobre los tokens de código."""
    mask = (labels != -100)

    if mask.sum() > 0:
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


def quick_eval(model, tokenizer, device, prompt: str) -> str:
    model.eval()
    p_ids = tokenizer.encode(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n```python\n")
    current = list(p_ids)
    with torch.no_grad():
        t_in = torch.tensor([current], device=device, dtype=torch.long)
        logits, states = model(t_in)
        for _ in range(45):
            last_logits = logits[0, -1, :].clone().float()
            for prev in set(current[-25:]):
                if last_logits[prev] > 0:
                    last_logits[prev] /= 1.25
                else:
                    last_logits[prev] *= 1.25
            v, _ = torch.topk(last_logits, 30)
            last_logits[last_logits < v[-1]] = -float('Inf')
            probs = torch.softmax(last_logits / 0.35, dim=-1)
            next_t = torch.multinomial(probs, 1).item()
            if next_t in (151645, 151643):  # <|im_end|>, <|endoftext|>
                break
            current.append(next_t)
            cur_pos = len(current) - 1
            x_step = model.token_emb(torch.tensor([[next_t]], device=device)) + model.pos_emb(torch.tensor([[cur_pos]], device=device))
            next_states = []
            h = x_step
            for block, s in zip(model.blocks, states):
                h, next_s = block.step(h, s)
                next_states.append(next_s)
            states = next_states
            logits = model.lm_head(model.ln_f(h))
    return tokenizer.decode(current[len(p_ids):]).strip()


def main():
    enforce_reproducibility(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("\n=======================================================", flush=True)
    print("  DESTILACIÓN PROFUNDA DE CÓDIGO (QWEN2.5 -> HOLOLLM)   ", flush=True)
    print("=======================================================", flush=True)

    # 1. Cargar el Profesor
    print(f"Cargando Profesor {TEACHER_MODEL_ID} en bfloat16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda"
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    aligned_vocab = teacher.config.vocab_size
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    print(f"✔ Profesor listo. Vocabulario: {aligned_vocab:,} tokens", flush=True)

    # 2. Configurar Alumno Holográfico (~66.4M parámetros)
    DIM = 384
    DEPTH = 6
    HEADS = 8
    MAX_LEN = 140
    BATCH_SIZE = 32
    EPOCHS = 18
    LR = 1.0e-3
    SAMPLES = 8000

    student = HoloCausalLM(
        vocab_size=aligned_vocab,
        dim=DIM,
        depth=DEPTH,
        num_heads=HEADS,
        max_seq_len=MAX_LEN + 32
    ).to(device)

    # Carga adaptativa de pesos previos
    base_ckpt = "checkpoints/holo_distilled_student_75m.pt"
    if os.path.exists(base_ckpt):
        print(f"Cargando pesos previos del alumno desde {base_ckpt}...", flush=True)
        base_dict = torch.load(base_ckpt, map_location=device)
        
        # Adaptar pos_emb si las dimensiones difieren ligeramente
        if "pos_emb.weight" in base_dict:
            old_rows = base_dict["pos_emb.weight"].shape[0]
            new_rows = student.pos_emb.weight.shape[0]
            if old_rows != new_rows:
                print(f"Adaptando pos_emb de {old_rows} a {new_rows} posiciones...", flush=True)
                min_rows = min(old_rows, new_rows)
                student.pos_emb.weight.data[:min_rows] = base_dict["pos_emb.weight"][:min_rows]
                del base_dict["pos_emb.weight"]
                student.load_state_dict(base_dict, strict=False)
            else:
                student.load_state_dict(base_dict)
        else:
            student.load_state_dict(base_dict)
            
        print("✔ Pesos previos cargados adaptativamente para destilación incremental.", flush=True)

    student_params = sum(p.numel() for p in student.parameters())
    print(f"✔ Alumno HoloLLM listo: {student_params:,} parámetros (~{student_params/1e6:.1f}M)", flush=True)

    # 3. Dataset
    inputs, labels = prepare_algorithmic_dataset(tokenizer, pad_id, max_samples=SAMPLES, max_len=MAX_LEN)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    optimizer = optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print(f"\n--- Iniciando ciclo de {EPOCHS} épocas de destilación profunda ({len(inputs)} ejemplos) ---", flush=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        student.train()
        total_loss = 0.0
        ep_start = time.time()

        temperature = 2.5 - (1.0 * (epoch / EPOCHS))

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            shift_labels = batch_y[:, 1:].contiguous()

            with torch.no_grad():
                t_out = teacher(batch_x)
                teacher_shift_logits = t_out.logits[:, :-1, :].contiguous()

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                s_logits, _ = student(batch_x)
                student_shift_logits = s_logits[:, :-1, :].contiguous()

                loss = masked_distillation_loss(
                    student_logits=student_shift_logits,
                    teacher_logits=teacher_shift_logits,
                    labels=shift_labels,
                    temperature=temperature,
                    alpha=0.7
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        scheduler.step()
        ep_loss = total_loss / len(inputs)
        ep_time = time.time() - ep_start
        print(f"Época [{epoch:02d}/{EPOCHS:02d}] | KD Loss: {ep_loss:.4f} | Temp: {temperature:.2f} | Tiempo: {ep_time:.1f}s", flush=True)

        if epoch % 6 == 0 or epoch == EPOCHS:
            print("\n" + "─" * 60, flush=True)
            print(f"🔍 [Evaluación en vivo de lógica - Época {epoch}]:", flush=True)
            res_pal = quick_eval(student, tokenizer, device, "Write a python function to check if a string is a palindrome.")
            print(f"[Palíndromo]:\n{res_pal}\n", flush=True)
            res_fact = quick_eval(student, tokenizer, device, "Write a python function to calculate the factorial of a number.")
            print(f"[Factorial]:\n{res_fact}", flush=True)
            print("─" * 60 + "\n", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    out_ckpt = "checkpoints/holo_deep_distilled_75m.pt"
    torch.save(student.state_dict(), out_ckpt)
    elapsed = time.time() - start_time
    print("\n=======================================================", flush=True)
    print(f"  DESTILACIÓN PROFUNDA FINALIZADA EN {elapsed/60:.1f} MINUTOS  ", flush=True)
    print(f"  Checkpoint guardado en: {out_ckpt}", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    main()
