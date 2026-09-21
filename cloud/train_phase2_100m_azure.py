r"""
SCRIPT: cloud/train_phase2_100m_azure.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM) - Cloud Training Suite
DESCRIPCIÓN: Entrenamiento de Fase 2 (100 Millones de tokens) en NVIDIA H100 (Azure NC40ads_H100_v5).
             Arquitectura campeona: Holo-v2 (Toro Unitario) con tolerancia a Spot y auditoría de Crespo.
"""

from __future__ import annotations

import os
import sys
import time
import math
import signal
import hashlib
from typing import List, Dict, Any, Generator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.holo_causal_lm_v2 import HoloCausalLMV2
from src.layers.phase_disentangler import HolographicPhaseDisentangler
from src.utils.spectral_diagnostics import SpectralHessianAuditor

try:
    from datasets import load_dataset
except ImportError:
    os.system("pip install datasets")
    from datasets import load_dataset

# ==============================================================================
# CONFIGURACIÓN FASE 2: 100 MILLONES DE TOKENS (AZURE H100 NVL 94GB)
# ==============================================================================
CONFIG = {
    "seed": 42,
    "teacher_id": "Qwen/Qwen2.5-Coder-1.5B",
    "checkpoint_dir": "checkpoints/phase2_100m",
    "latest_checkpoint": "checkpoints/phase2_100m/holo_v2_100m_latest.pt",
    "final_checkpoint": "checkpoints/phase2_100m/holo_v2_100m_final.pt",
    "vocab_size": 151936,
    "dim": 384,
    "depth": 6,
    "num_heads": 8,
    "seq_len": 512,
    "batch_size": 16,               # 8,192 tokens/paso
    "target_tokens": 100_000_000,   # 100 Millones de tokens
    "lr": 2.5e-4,
    "min_lr": 1.0e-5,
    "warmup_steps": 300,
    "temperature": 1.0,
    "alpha_distill": 0.6,
    "beta_phase": 0.03,             # Regularizador de Bragg en el Toro
    "audit_interval": 250,          # Diagnóstico del Hessiano cada 250 pasos
    "save_interval": 1000,          # Checkpoint periódico cada ~8M tokens
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

TOTAL_STEPS = math.ceil(CONFIG["target_tokens"] / (CONFIG["batch_size"] * CONFIG["seq_len"]))
SHUTDOWN_REQUESTED = False


def sigterm_handler(signum, frame):
    """Manejo de señales para instancias Azure Spot (guarda antes de apagarse)."""
    global SHUTDOWN_REQUESTED
    print("\n⚠️ [AZURE SPOT PREEMPTION] Señal de terminación recibida. Guardando checkpoint de emergencia...", flush=True)
    SHUTDOWN_REQUESTED = True

signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)


def load_high_density_corpus(tokenizer, max_samples: int = 60000) -> List[Tuple[List[int], List[int]]]:
    """Carga corpus de alta densidad: 60% código algorítmico + 40% diálogo estructurado."""
    print(f"[Dataset] Descargando y preparando corpus de 100M tokens...")
    formatted = []

    # 1. Código Python (Instrucción y Algoritmos)
    try:
        ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        for item in ds_code:
            prompt = f"<|im_start|>user\n{item['instruction']}\n{item.get('input', '')}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>\n"
            toks = tokenizer.encode(prompt, add_special_tokens=False)
            if 16 < len(toks) <= CONFIG["seq_len"]:
                split_idx = prompt.find("<|im_start|>assistant\n")
                prompt_len = len(tokenizer.encode(prompt[:split_idx], add_special_tokens=False))
                labels = [-100] * prompt_len + toks[prompt_len:]
                formatted.append((toks, labels[:len(toks)]))
    except Exception as e:
        print(f"⚠️ Aviso cargando código: {e}")

    # 2. Conversaciones multi-turno (DailyDialog)
    try:
        ds_diag = load_dataset("OpenRL/daily_dialog", split="train")
        for item in ds_diag:
            if len(formatted) >= max_samples:
                break
            dialog = item.get("dialog", [])
            if len(dialog) < 2:
                continue
            full_text = ""
            for turn_idx, u in enumerate(dialog):
                role = "user" if turn_idx % 2 == 0 else "assistant"
                full_text += f"<|im_start|>{role}\n{u.strip()}<|im_end|>\n"
            toks = tokenizer.encode(full_text, add_special_tokens=False)
            if 16 < len(toks) <= CONFIG["seq_len"]:
                formatted.append((toks, toks))
    except Exception as e:
        print(f"⚠️ Aviso cargando diálogo: {e}")

    print(f"✔ Corpus compilado: {len(formatted)} secuencias de entrenamiento.")
    return formatted


def data_generator(
    dataset: List[Tuple[List[int], List[int]]],
    batch_size: int,
    seq_len: int,
    pad_id: int
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    num_samples = len(dataset)
    idx = 0
    while True:
        b_in, b_tgt = [], []
        for _ in range(batch_size):
            tokens, labels = dataset[idx % num_samples]
            idx += 1
            if len(tokens) > seq_len:
                tokens, labels = tokens[:seq_len], labels[:seq_len]
            else:
                pad = seq_len - len(tokens)
                tokens = tokens + [pad_id] * pad
                labels = labels + [-100] * pad
            b_in.append(tokens)
            b_tgt.append(labels)
        yield torch.tensor(b_in, dtype=torch.long), torch.tensor(b_tgt, dtype=torch.long)


def compute_sha256(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for p in model.parameters():
        hasher.update(p.detach().cpu().float().numpy().tobytes())
    return hasher.hexdigest()


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, tokens_done: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sha = compute_sha256(model)
    torch.save({
        "step": step,
        "tokens_processed": tokens_done,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "weights_sha256": sha,
        "config": CONFIG,
        "timestamp": time.time()
    }, path)
    print(f"\n💾 Checkpoint guardado en: {path} (SHA: {sha[:12]} | Tokens: {tokens_done:,})\n", flush=True)


def main():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    print("=" * 85)
    print("HOLOLLM-v2: ENTRENAMIENTO DE PRODUCCIÓN EN AZURE H100 (FASE 2: 100M TOKENS)")
    print(f"Arquitectura Campeona: Holo-v2 (Toro Unitario) | Dispositivo: {device}")
    print(f"Pasos Totales: {TOTAL_STEPS:,} | Tokens/paso: {CONFIG['batch_size'] * CONFIG['seq_len']:,}")
    print("=" * 85)

    # 1. Cargar Maestro Qwen
    print(f"[Teacher] Inicializando {CONFIG['teacher_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["teacher_id"], trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        CONFIG["teacher_id"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2. Alumno HoloCausalLMV2 en float32
    student = HoloCausalLMV2(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"]
    ).to(device=device, dtype=torch.float32)

    captured_keys: List[torch.Tensor] = []
    def make_k_hook():
        def hook(module, inp, out):
            if out.ndim == 3:
                B, S, D = out.shape
                H = CONFIG["num_heads"]
                k_view = out.reshape(B, S, H, D // H).permute(0, 2, 1, 3)
                captured_keys.append(torch.fft.rfft(k_view.float(), dim=-1))
        return hook

    hook_handles = [
        m.register_forward_hook(make_k_hook())
        for n, m in student.named_modules()
        if isinstance(m, nn.Linear) and "k_proj" in n
    ]

    disentangler = HolographicPhaseDisentangler().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=CONFIG["lr"], betas=(0.9, 0.98), weight_decay=0.01)
    auditor = SpectralHessianAuditor(log_interval=CONFIG["audit_interval"])

    start_step = 1
    tokens_processed = 0

    # Auto-resume si existe checkpoint previo (vital para Spot)
    if os.path.exists(CONFIG["latest_checkpoint"]):
        print(f"[Auto-Resume] Reanudando desde: {CONFIG['latest_checkpoint']}...")
        ckpt = torch.load(CONFIG["latest_checkpoint"], map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        tokens_processed = ckpt.get("tokens_processed", (start_step - 1) * CONFIG["batch_size"] * CONFIG["seq_len"])
        print(f"✔ Reanudado con éxito en el paso {start_step} ({tokens_processed:,} tokens ya completados).")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dataset = load_high_density_corpus(tokenizer)
    gen = data_generator(dataset, CONFIG["batch_size"], CONFIG["seq_len"], pad_id)

    def get_lr(step: int) -> float:
        if step < CONFIG["warmup_steps"]:
            return CONFIG["lr"] * (step + 1) / CONFIG["warmup_steps"]
        progress = (step - CONFIG["warmup_steps"]) / max(TOTAL_STEPS - CONFIG["warmup_steps"], 1)
        return CONFIG["min_lr"] + 0.5 * (CONFIG["lr"] - CONFIG["min_lr"]) * (1.0 + math.cos(math.pi * progress))

    print("=" * 85)
    print("INICIO DE PRODUCCIÓN CONTINUA")
    print("=" * 85)

    start_time = time.time()
    for step in range(start_step, TOTAL_STEPS + 1):
        if SHUTDOWN_REQUESTED:
            save_checkpoint(CONFIG["latest_checkpoint"], student, optimizer, step - 1, tokens_processed)
            print("✔ Salida de emergencia completada. Estado salvado sin pérdidas.")
            sys.exit(0)

        cur_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        b_in, b_tgt = next(gen)
        inputs = b_in.to(device)[:, :-1].contiguous()
        targets = b_tgt.to(device)[:, 1:].contiguous()
        captured_keys.clear()

        with torch.no_grad():
            teacher_logits = teacher(inputs).logits.float()

        student.train()
        student_logits, _ = student(inputs)
        student_logits = student_logits.float()

        loss_mask = (targets != -100).float()
        num_valid = torch.clamp(loss_mask.sum(), min=1.0)

        # Cross Entropy
        loss_ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )

        # KL Divergence
        t_temp = CONFIG["temperature"]
        p_s = F.log_softmax(student_logits / t_temp, dim=-1)
        p_t = F.softmax(teacher_logits / t_temp, dim=-1)
        kl_tokens = F.kl_div(p_s, p_t, reduction="none").sum(dim=-1)
        loss_kd = ((kl_tokens * loss_mask).sum() / num_valid) * (t_temp ** 2)

        # Regularizador de Fase en el Toro
        loss_phase = torch.tensor(0.0, device=device)
        if captured_keys:
            for k_c in captured_keys:
                p_l, _ = disentangler(k_c)
                loss_phase += p_l
            loss_phase /= len(captured_keys)

        total_loss = (1.0 - CONFIG["alpha_distill"]) * loss_ce + \
                     CONFIG["alpha_distill"] * loss_kd + \
                     CONFIG["beta_phase"] * loss_phase

        optimizer.zero_grad()

        # Auditoría Espectral periódica de Crespo
        is_audit = (step % CONFIG["audit_interval"] == 0) or (step == TOTAL_STEPS)
        if is_audit:
            diag = auditor.inspect(student, total_loss)
            total_loss.backward()
        else:
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        tokens_processed += CONFIG["batch_size"] * CONFIG["seq_len"]

        # Telemetría cada 50 pasos
        if step % 50 == 0 or step == 1:
            el = max(time.time() - start_time, 1e-4)
            tok_s = ((step - start_step + 1) * CONFIG["batch_size"] * CONFIG["seq_len"]) / el
            print(f"Paso [{step:05d}/{TOTAL_STEPS:05d}] | "
                  f"Total: {total_loss.item():.4f} | "
                  f"KD: {loss_kd.item():.4f} | "
                  f"CE: {loss_ce.item():.4f} | "
                  f"Fase: {loss_phase.item():.4f} | "
                  f"Tokens: {tokens_processed:,} | "
                  f"Tok/s: {tok_s:.0f}", flush=True)

        # Reporte de Curvatura de Crespo
        if is_audit:
            print(f"\n🔬 [DIAGNÓSTICO ESPECTRAL DE CRESPO - Paso {step:05d}]")
            print(f"   ├─ κ (Número de Condición): {diag['crespo_kappa']:10.2f} "
                  f"-> {'🚨 CAÑÓN' if diag['is_canyon'] else '✔ Paisaje equilibrado'}")
            print(f"   ├─ ε (|λ_max| Curvatura):   {diag['crespo_epsilon_max']:10.4f}")
            print(f"   └─ δ (λ_min Punto de Silla): {diag['crespo_lambda_min']:10.4e} "
                  f"-> {'🚨 SADDLE (en descenso)' if diag['is_saddle'] else '✔ MÍNIMO LOCAL GENUINO'}\n", flush=True)

        # Guardado de checkpoints
        if step % CONFIG["save_interval"] == 0:
            save_checkpoint(CONFIG["latest_checkpoint"], student, optimizer, step, tokens_processed)

    for h in hook_handles:
        h.remove()

    save_checkpoint(CONFIG["final_checkpoint"], student, optimizer, TOTAL_STEPS, tokens_processed)
    print("=" * 85)
    print(f"✔ ENTRENAMIENTO DE FASE 2 FINALIZADO CON ÉXITO: 100 MILLONES DE TOKENS PROCESADOS.")
    print(f"✔ Checkpoint final listo para inferencia: {CONFIG['final_checkpoint']}")
    print("=" * 85)


if __name__ == "__main__":
    main()