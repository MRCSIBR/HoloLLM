r"""
MÓDULO: src/models/holo_causal_lm_v2.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM) - Arquitectura v2
DESCRIPCIÓN: Modelo Causal Holográfico con Inicialización Canónica de LLM (std=0.02).
             Garantiza que la entropía inicial coincida con el límite teórico ln(V) = 11.93.
"""

from typing import Tuple, List, Optional, Dict, Any
import hashlib
import inspect
import torch
import torch.nn as nn

from src.layers.holo_attention_v2 import MultiHeadHoloAttentionV2


class HoloBlockV2(nn.Module):
    """Bloque Causal Holográfico v2 con LayerNorm y FFN."""

    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadHoloAttentionV2(d_model=dim, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, end_state = self.attn(self.ln1(x), positions=positions)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, end_state

    def step(
        self,
        x_t: torch.Tensor,
        pos_t: int,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, next_state = self.attn.step(self.ln1(x_t), pos_t=pos_t, state=state)
        x = x_t + h
        x = x + self.mlp(self.ln2(x))
        return x, next_state


class HoloCausalLMV2(nn.Module):
    """
    Arquitectura Causal Holográfica Completa v2 (HoloLLM-350M).
    - Inicialización canónica std=0.02 (evita la colina de pérdida 280).
    - Weight Tying entre Embedding y Head.
    - Contexto ilimitado O(1).
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        dim: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        mlp_ratio: int = 4
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads

        self.token_emb = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            HoloBlockV2(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # Inicialización canónica de pesos (estándar LLaMA / Qwen)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def init_empty_states(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        freq_dim = (self.dim // self.num_heads) // 2 + 1
        return [
            torch.zeros(
                (batch_size, self.num_heads, freq_dim),
                dtype=torch.complex64,
                device=device
            )
            for _ in range(self.depth)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, S = input_ids.shape
        x = self.token_emb(input_ids)

        end_states = []
        for block in self.blocks:
            x, state = block(x, positions=positions)
            end_states.append(state)

        logits = self.lm_head(self.ln_f(x))
        return logits, end_states

    def absorb_context(
        self,
        token_ids: List[int],
        start_pos: int,
        states: Optional[List[torch.Tensor]] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        if device is None:
            device = self.token_emb.weight.device
        if states is None:
            states = self.init_empty_states(batch_size=1, device=device)

        t_in = torch.tensor([token_ids], dtype=torch.long, device=device)
        positions = torch.arange(start_pos, start_pos + len(token_ids), dtype=torch.long, device=device)

        x = self.token_emb(t_in)
        new_states = []
        for block, s in zip(self.blocks, states):
            x, final_s = block(x, positions=positions)
            decays = torch.sigmoid(block.attn.decay_logits) ** len(token_ids)
            chained_state = decays * s + final_s
            new_states.append(chained_state)

        logits = self.lm_head(self.ln_f(x))
        next_pos = start_pos + len(token_ids)
        return logits[0, -1, :], new_states, next_pos

    @staticmethod
    def get_source_hash() -> str:
        source = inspect.getsource(HoloCausalLMV2)
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
