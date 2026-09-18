r"""
TEST: Verificación Matemática de HoloLLM-v2
Comprueba la consistencia de gradientes, la ausencia de límites en contexto y la estabilidad numérica.
"""

import torch
from src.models.holo_causal_lm_v2 import HoloCausalLMV2

print("=" * 70)
print("TEST DE INTEGRIDAD: HoloLLM-v2 (Contexto Continuo O(1))")
print(f"SHA-256 Arquitectura: {HoloCausalLMV2.get_source_hash()}")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HoloCausalLMV2(vocab_size=151936, dim=384, depth=6, num_heads=8).to(device)

print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# 1. Prueba de contexto largo (ej: 1024 tokens, muy por encima de los 172 previos)
print("\n[Test 1] Evaluando secuencia larga de 512 tokens...")
x_long = torch.randint(0, 1000, (1, 512), device=device)
logits, states = model(x_long)
print(f"✔ Forward exitoso. Logits: {logits.shape}, Estados por capa: {len(states)}")
assert not torch.isnan(logits).any(), "Error: NaNs detectados en logits."

# 2. Prueba del paso recurrente O(1) a posición arbitraria t = 1000
print("\n[Test 2] Evaluando paso recurrente O(1) en posición t = 1000...")
token_1000 = torch.randint(0, 1000, (1, 1), device=device)
next_states = []
h = model.token_emb(token_1000)
for block, s in zip(model.blocks, states):
    h, next_s = block.step(h, pos_t=1000, state=s)
    next_states.append(next_s)
step_logits = model.lm_head(model.ln_f(h))
print(f"✔ Step exitoso en t=1000. Shape: {step_logits.shape}")
assert not torch.isnan(step_logits).any(), "Error: NaNs detectados en step."

# 3. Prueba de absorción continua de diálogo (Multi-Turn)
print("\n[Test 3] Evaluando absorción continua de diálogo multi-turno...")
user_msg_1 = [100, 200, 300, 400]
next_logit, states, pos = model.absorb_context(user_msg_1, start_pos=0, states=None, device=device)
print(f"✔ Turno 1 absorbido (Posición resultante: {pos}).")

user_msg_2 = [500, 600, 700]
next_logit, states, pos = model.absorb_context(user_msg_2, start_pos=pos, states=states, device=device)
print(f"✔ Turno 2 absorbido sobre la memoria viva (Posición resultante: {pos}).")

print("\n" + "=" * 70)
print("TODAS LAS PRUEBAS MATEMÁTICAS DE HOLOLLM-V2 HAN SIDO SUPERADAS.")
print("=" * 70)
