"""
Script: benchmarks/benchmark_transformer_vs_holo.py
Propósito: Comparativa sistemática y rigurosa entre HoloLM y un Causal Transformer estándar.
Incluye el test de ablación de Karl Pribram (resistencia a daños en los pesos).
"""

import os
import sys
from pathlib import Path

# Inyectar la raíz del proyecto en sys.path para resolución robusta de imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import copy
import math
from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.holographic_lm import HolographicLM
from src.utils.seed import enforce_reproducibility
from train_lm import CORPUS, CharTokenizer


# =====================================================================
# 1. BASELINE: Transformer Causal Estándar (Arquitectura estilo GPT)
# =====================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim=dim, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class StandardTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 96, depth: int = 2, max_seq_len: int = 512) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim=dim) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# =====================================================================
# 2. PROTOCOLO DE EVALUACIÓN Y DAÑO CEREBRAL SINTÉTICO (PRIBRAM)
# =====================================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_loss(model: nn.Module, dataloader: Any, criterion: Any, is_holo: bool = False) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for bx, by in dataloader:
            if is_holo:
                logits, _ = model(bx)
            else:
                logits = model(bx)
            loss = criterion(logits.view(-1, logits.shape[-1]), by.view(-1))
            total_loss += loss.item() * by.numel()
            total_tokens += by.numel()
    return total_loss / total_tokens


def apply_pribram_lesion(model: nn.Module, damage_ratio: float) -> nn.Module:
    """
    Simula una lesión cerebral (Lashley/Pribram):
    Poda aleatoriamente a cero una fracción de los parámetros de la red.
    """
    damaged_model = copy.deepcopy(model)
    with torch.no_grad():
        for param in damaged_model.parameters():
            if param.requires_grad:
                mask = torch.rand_like(param.real if param.is_complex() else param) > damage_ratio
                if param.is_complex():
                    param.copy_(param * mask.to(param.dtype))
                else:
                    param.copy_(param * mask.to(param.dtype))
    return damaged_model


# =====================================================================
# 3. EJECUCIÓN DEL BENCHMARK
# =====================================================================
def main() -> None:
    SEED = 42
    enforce_reproducibility(SEED)

    tokenizer = CharTokenizer(CORPUS)
    data = tokenizer.encode(CORPUS)
    vocab_size = tokenizer.vocab_size

    DIM = 96
    DEPTH = 2
    MODES = 24
    SEQ_LEN = 48
    BATCH_SIZE = 16
    EPOCHS = 35
    LR = 4e-3

    X_list, Y_list = [], []
    for i in range(0, len(data) - SEQ_LEN - 1, 2):
        X_list.append(data[i : i + SEQ_LEN])
        Y_list.append(data[i + 1 : i + SEQ_LEN + 1])
    X = torch.stack(X_list)
    Y = torch.stack(Y_list)

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, Y),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    criterion = nn.CrossEntropyLoss()

    print("\n=======================================================")
    print("  BENCHMARK: HOLOGRAPHIC LM vs STANDARD TRANSFORMER   ")
    print("=======================================================")

    holo_model = HolographicLM(vocab_size=vocab_size, dim=DIM, depth=DEPTH, modes=MODES)
    trans_model = StandardTransformerLM(vocab_size=vocab_size, dim=DIM, depth=DEPTH)

    p_holo = count_parameters(holo_model)
    p_trans = count_parameters(trans_model)

    print(f"Parámetros HoloLM:      {p_holo:,}")
    print(f"Parámetros Transformer: {p_trans:,}")
    print(f"Diferencia de pesos:    {((p_holo - p_trans)/p_trans)*100:+.2f}%")

    print("\n--- Entrenando ambos modelos bajo condiciones idénticas ---")
    opt_holo = optim.AdamW(holo_model.parameters(), lr=LR, weight_decay=1e-3)
    opt_trans = optim.AdamW(trans_model.parameters(), lr=LR, weight_decay=1e-3)

    for ep in range(1, EPOCHS + 1):
        # HoloLM
        holo_model.train()
        for bx, by in dataloader:
            opt_holo.zero_grad()
            l_holo, _ = holo_model(bx)
            loss_h = criterion(l_holo.view(-1, vocab_size), by.view(-1))
            loss_h.backward()
            opt_holo.step()

        # Transformer
        trans_model.train()
        for bx, by in dataloader:
            opt_trans.zero_grad()
            l_trans = trans_model(bx)
            loss_t = criterion(l_trans.view(-1, vocab_size), by.view(-1))
            loss_t.backward()
            opt_trans.step()

        if ep % 10 == 0 or ep == EPOCHS:
            v_loss_h = evaluate_loss(holo_model, dataloader, criterion, is_holo=True)
            v_loss_t = evaluate_loss(trans_model, dataloader, criterion, is_holo=False)
            print(f"Época [{ep:02d}/{EPOCHS:02d}] | Loss HoloLM: {v_loss_h:.4f} | Loss Transformer: {v_loss_t:.4f}")

    print("\n=======================================================")
    print("  EXPERIMENTO DE PRIBRAM: ROBUSTEZ ANTE ABLACIÓN/DAÑO  ")
    print("=======================================================")
    print("Se simula la pérdida estocástica de conexiones neuronales:")
    print(f"{'Nivel de Daño':<18} | {'Loss HoloLM':<16} | {'Loss Transformer':<18} | {'Resultado'}")
    print("-" * 72)

    damage_levels = [0.0, 0.10, 0.25, 0.40]

    for dmg in damage_levels:
        dmg_holo = apply_pribram_lesion(holo_model, dmg)
        dmg_trans = apply_pribram_lesion(trans_model, dmg)

        loss_h = evaluate_loss(dmg_holo, dataloader, criterion, is_holo=True)
        loss_t = evaluate_loss(dmg_trans, dataloader, criterion, is_holo=False)

        winner = "✔ HoloLM más robusto" if loss_h < loss_t else "✔ Transformer más robusto"
        print(f"{int(dmg*100):>3}% de pesos podados  | {loss_h:>14.4f}  | {loss_t:>16.4f}   | {winner}")

if __name__ == "__main__":
    main()
