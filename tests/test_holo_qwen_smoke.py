r"""
SCRIPT: test_holo_qwen_smoke.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM) - Validación de Transmutación
DESCRIPCIÓN: Prueba de humo rápida de HoloQwen-1.5B en CPU.
             Verifica forward pass, ausencia de NaNs, cálculo de pérdida inicial y generación O(1).
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.models.holo_qwen import HoloQwenForCausalLM

CONFIG = {
    "tokenizer_id": "Qwen/Qwen2.5-Coder-1.5B",
    "checkpoint_path": "checkpoints/holo_qwen_1.5b_transmuted.pt",
    "vocab_size": 151936,
    "dim": 1536,
    "depth": 28,
    "num_heads": 12,
    "device": "cpu"
}

SAMPLE_TARGET = "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"


def main():
    device = torch.device(CONFIG["device"])
    print("=" * 80)
    print("SMOKE TEST: VERIFICACIÓN DE TRANSMUTACIÓN EN HOLOQWEN-1.5B")
    print(f"Dispositivo: {device.type.upper()} | Dtype: torch.float32 (Alta precisión)")
    print("=" * 80)

    # 1. Cargar Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_id"], trust_remote_code=True)

    # 2. Instanciar e Inyectar Pesos Transmutados
    print(f"[1/3] Cargando HoloQwen-1.5B desde {CONFIG['checkpoint_path']} en RAM...")
    start_load = time.time()
    model = HoloQwenForCausalLM(
        vocab_size=CONFIG["vocab_size"],
        dim=CONFIG["dim"],
        depth=CONFIG["depth"],
        num_heads=CONFIG["num_heads"]
    ).to(device=device, dtype=torch.float32).eval()

    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print(f"✔ Modelo cargado y verificado en {time.time() - start_load:.1f} segundos.")

    # 3. Test de Pérdida Inicial (Comprobación de Warm-Start)
    print("\n[2/3] Calculando pérdida inicial sobre código canónico (Prueba de Warm-Start)...")
    prompt = "<|im_start|>user\nWrite a python function for the following task:\nfactorial<|im_end|>\n<|im_start|>assistant\n```python\n"
    full_text = prompt + SAMPLE_TARGET + "\n```<|im_end|>"
    
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    p_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    
    input_ids = torch.tensor([tokens[:-1]], device=device)
    targets = torch.tensor([tokens[1:]], device=device)
    
    # Máscara: solo evaluamos pérdida en el código generado
    loss_mask = torch.zeros_like(targets, dtype=torch.float32)
    loss_mask[:, p_len - 1:] = 1.0

    with torch.no_grad():
        logits, states = model(input_ids)
        assert not torch.isnan(logits).any(), "Error: NaNs detectados en logits."
        
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none"
        ).view(targets.shape)
        
        valid_ce = (ce_loss * loss_mask).sum() / loss_mask.sum()
        ppl = torch.exp(valid_ce).item()

    print(f"  ✔ Pérdida Inicial CE: {valid_ce.item():.4f}")
    print(f"  ✔ Perplejidad Inicial: {ppl:.2f}")
    print("  (En Cold Start arrancaba en 281.9; en Warm-Start arranca en orden de magnitud natural).")

    # 4. Generación Recurrente O(1) de 35 tokens en CPU
    print("\n[3/3] Probando paso recurrente O(1) (block.step) en CPU local...")
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    t_in = torch.tensor([prompt_tokens], device=device)

    start_gen = time.perf_counter()
    with torch.no_grad():
        logits, states = model(t_in)
        next_t = torch.argmax(logits[0, -1, :]).item()
        current = list(prompt_tokens) + [next_t]
        gen = [next_t]

        for _ in range(35):
            cur_pos = len(current) - 1
            token_tensor = torch.tensor([[next_t]], device=device)
            h = model.embed_tokens(token_tensor)

            next_states = []
            for block, s in zip(model.layers, states):
                h, next_s = block.step(h, pos_t=cur_pos, state=s)
                next_states.append(next_s)
            states = next_states

            logits = model.lm_head(model.norm(h))
            next_t = torch.argmax(logits[0, -1, :]).item()
            tok_str = tokenizer.decode([next_t])
            if "```" in tok_str or next_t in (151645, 151643):
                break
            gen.append(next_t)
            current.append(next_t)

    elapsed = time.perf_counter() - start_gen
    speed = len(gen) / max(elapsed, 1e-4)

    print(f"  ✔ Generación completada: {len(gen)} tokens a {speed:.1f} tok/s en CPU.")
    print("\n--- TEXTO EMITIDO POR HOLOQWEN-1.5B (WARM TRANSMUTADO) ---")
    print(tokenizer.decode(gen).strip())
    print("=" * 80)
    print("SMOKE TEST COMPLETADO EXITOSAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    main()
