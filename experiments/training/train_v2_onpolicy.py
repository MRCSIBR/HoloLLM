r"""
MÓDULO: train_v2_onpolicy.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM) - Arquitectura v2
DESCRIPCIÓN: Destilación On-Policy (Generalized Knowledge Distillation / Student Rollout)
             para eliminar el Exposure Bias y dotar a la memoria holográfica O(1) de estabilidad
             de Liapunov frente a perturbaciones de fase.

FUNDAMENTACIÓN TEÓRICA:
1. Tony Plate (Clean-up Memory Autoasociativo):
   La memoria asociativa circular sufre de acumulación de ruido si no se entrena al sistema
   para proyectar estados ruidosos de vuelta a la variedad ortogonal discreta.
2. David Bohm (Estabilidad de Atractores en el Orden Implicado):
   El aprendizaje On-Policy esculpe pozos de potencial profundos donde las trayectorias
   perturbadas en el Orden Explicado convergen naturalmente al camino de mínima acción.
3. Vitaly Vanchurin:
   La red neuronal aprende a lo largo de su trayectoria de evolución temporal real (path integral).
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
    "init_checkpoint": "checkpoints/holo_v2_conversational_70m.pt",
    "output_checkpoint": "checkpoints/holo_v2_onpolicy_stable_70m.pt",
    "vocab_size": 151936,
    "dim": 384,
    "depth": 6,
    "num_heads": 8,
    "max_rollout_tokens": 128,
    "batch_size": 4,          # Rollout autoregresivo por batch
    "max_steps": 1200,        # 1200 pasos on-policy bastan para fijar estabilidad
    "lr": 1.5e-5,             # Learning rate bajo para fine-tuning fino de cuencas
    "temperature": 0.7,       # Exploración suave durante el rollout
    "beta_phase": 0.03,       # Regularización de ortogonalidad
    "save_interval": 300,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Tareas para rollout On-Policy (las 10 algorítmicas + 3 conversacionales)
TASKS = [
    ("algo", "factorial"),
    ("algo", "fibonacci with memoization"),
    ("algo", "binary search"),
    ("algo", "reverse linked list"),
    ("algo", "merge sort"),
    ("algo", "breadth first search bfs"),
    ("algo", "maximum subarray kadane"),
    ("algo", "valid parentheses check"),
    ("algo", "quick select kth smallest"),
    ("algo", "dijkstra shortest path"),
    ("chat", "¿Cuál es la complejidad temporal y de memoria de HoloLLM frente a los Transformers?"),
    ("chat", "Explica la diferencia entre el Orden Implicado y el Orden Explicado de David Bohm."),
    ("chat", "¿Por qué la convolución circular en el toro unitario elimina el KV-Cache?")
]


def make_prompt(task_type: str, text: str) -> str:
    if task_type == "algo":
        return f"<|im_start|>user\nWrite a python function for the following task:\n{text}<|im_end|>\n<|im_start|>assistant\n```python\n"
    else:
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def compute_weights_sha256(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for p in model.parameters():
        hasher.update(p.detach().cpu().float().numpy().tobytes())
    return hasher.hexdigest()


def main() -> None:
    torch.manual_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    print("=" * 80)
    print(f"[HoloLLM-v2] DESTILACIÓN ON-POLICY (GKD) INICIADA EN: {device}")
    print("=" * 80)

    # 1. Maestro Qwen (Solo inferencia paralela, cero gradientes)
    print(f"[Teacher] Cargando {CONFIG['teacher_id']} (Modo Supervisor)...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["teacher_id"], trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        CONFIG["teacher_id"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2. Alumno HoloCausalLMV2
    print(f"[Student] Inicializando HoloCausalLMV2 desde {CONFIG['init_checkpoint']}...")
    student = HoloCausalLMV2(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"]
    ).to(device=device, dtype=torch.bfloat16)

    # Cargar los pesos pre-entrenados del checkpoint anterior
    assert os.path.exists(CONFIG["init_checkpoint"]), f"Falta checkpoint base: {CONFIG['init_checkpoint']}"
    ckpt = torch.load(CONFIG["init_checkpoint"], map_location=device)
    student.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
    print("✔ Pesos v2 cargados para estabilización On-Policy.\n")

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

    disentangler = HolographicPhaseDisentangler().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=CONFIG["lr"], betas=(0.9, 0.98), weight_decay=0.01)

    print("=" * 80)
    print("INICIO DEL BUCLE ON-POLICY: ROLLOUT DEL ALUMNO + CORRECCIÓN DEL MAESTRO")
    print("=" * 80)

    start_time = time.time()
    for step in range(1, CONFIG["max_steps"] + 1):
        # Seleccionar lote de prompts
        batch_prompts = [make_prompt(*random.choice(TASKS)) for _ in range(CONFIG["batch_size"])]
        
        # -------------------------------------------------------------
        # PASO A: ROLLOUT AUTORREGRESIVO DEL ALUMNO (Exploración propia)
        # -------------------------------------------------------------
        student.eval()
        rollout_sequences = []
        prompt_lens = []

        with torch.no_grad():
            for p_str in batch_prompts:
                p_tokens = tokenizer.encode(p_str, add_special_tokens=False)
                prompt_lens.append(len(p_tokens))
                
                t_in = torch.tensor([p_tokens], device=device)
                logits, states = student(t_in)
                
                # Muestreo con temperatura suave para explorar pequeñas perturbaciones
                probs = F.softmax(logits[0, -1, :] / CONFIG["temperature"], dim=-1)
                next_t = torch.multinomial(probs, num_samples=1).item()
                current = list(p_tokens) + [next_t]

                for _ in range(CONFIG["max_rollout_tokens"]):
                    cur_pos = len(current) - 1
                    tok_tensor = torch.tensor([[next_t]], device=device)
                    h = student.token_emb(tok_tensor)
                    
                    next_states = []
                    for block, s in zip(student.blocks, states):
                        h, next_s = block.step(h, pos_t=cur_pos, state=s)
                        next_states.append(next_s)
                    states = next_states

                    step_logits = student.lm_head(student.ln_f(h))
                    step_probs = F.softmax(step_logits[0, -1, :] / CONFIG["temperature"], dim=-1)
                    next_t = torch.multinomial(step_probs, num_samples=1).item()
                    
                    current.append(next_t)
                    tok_str = tokenizer.decode([next_t])
                    if "```" in tok_str or "<|im_end|>" in tok_str or next_t in (151645, 151643):
                        break

                rollout_sequences.append(current)

        # Padear las secuencias generadas por el alumno al tamaño máximo del batch
        max_batch_len = max(len(s) for s in rollout_sequences)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        padded_inputs = []
        loss_masks = []

        for seq, p_len in zip(rollout_sequences, prompt_lens):
            pad_amount = max_batch_len - len(seq)
            padded_inputs.append(seq + [pad_id] * pad_amount)
            # La máscara solo supervisa lo que el alumno generó (ignora el prompt y el padding)
            mask = [0.0] * p_len + [1.0] * (len(seq) - p_len) + [0.0] * pad_amount
            loss_masks.append(mask)

        inputs_t = torch.tensor(padded_inputs, dtype=torch.long, device=device)
        mask_t = torch.tensor(loss_masks, dtype=torch.float32, device=device)

        # -------------------------------------------------------------
        # PASO B: EVALUACIÓN DEL MAESTRO SOBRE LA TRAYECTORIA DEL ALUMNO
        # -------------------------------------------------------------
        with torch.no_grad():
            teacher_logits = teacher(inputs_t).logits.float()

        # -------------------------------------------------------------
        # PASO C: PASO HACIA ADELANTE Y PÉRDIDA CORRECTIVA EN EL ALUMNO
        # -------------------------------------------------------------
        student.train()
        captured_keys.clear()
        student_logits, _ = student(inputs_t)
        student_logits = student_logits.float()

        # Divergencia KL sobre las trayectorias propias del alumno
        p_s = F.log_softmax(student_logits, dim=-1)
        p_t = F.softmax(teacher_logits, dim=-1)
        
        # KL por token: (B, S)
        kl_tokens = F.kl_div(p_s, p_t, reduction="none").sum(dim=-1)
        
        valid_tokens = torch.clamp(mask_t.sum(), min=1.0)
        loss_onpolicy = (kl_tokens * mask_t).sum() / valid_tokens

        # Pérdida de Fase Holográfica
        loss_phase = torch.tensor(0.0, device=device)
        max_coh = 0.0
        if captured_keys:
            for k_c in captured_keys:
                p_l, m = disentangler(k_c)
                loss_phase = loss_phase + p_l
                max_coh += m["phase_coherence_max"]
            loss_phase = loss_phase / len(captured_keys)
            max_coh /= len(captured_keys)

        total_loss = loss_onpolicy + CONFIG["beta_phase"] * loss_phase

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1:
            elapsed = max(time.time() - start_time, 1e-4)
            print(f"Paso On-Policy [{step:04d}/{CONFIG['max_steps']:04d}] | "
                  f"KL_OnPolicy: {loss_onpolicy.item():.4f} | "
                  f"Fase: {loss_phase.item():.4f} | "
                  f"MaxCoh: {max_coh:.4f} | "
                  f"Tokens Válidos: {int(valid_tokens.item())}")

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
            print(f"\n>>> Checkpoint On-Policy guardado: {CONFIG['output_checkpoint']} (SHA: {sha[:12]})\n")

    for h in hook_handles:
        h.remove()
    print("[HoloLLM-v2] Entrenamiento On-Policy completado exitosamente.")


if __name__ == "__main__":
    main()
