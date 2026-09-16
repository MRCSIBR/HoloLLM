"""
Módulo: src/models/holo_causal_lm.py
Fundamento Teórico:
- David Bohm: Enfundado del contexto instructivo en el tensor M_t y despliegue causal del código.
- Karl Pribram: Recuperación asociativa continua mediante cabezas holográficas multiescala.
- Tony Plate: Proyecciones unitarias que evitan la saturación de memoria durante la generación.
"""

from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn

from src.layers.holo_attention import MultiHeadHoloAttention
from src.utils.audit import compute_object_code_hash, compute_state_dict_hash


class HoloCausalBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadHoloAttention(dim=dim, num_heads=num_heads, decay_min=0.88, decay_max=0.995)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, next_state = self.attn(self.ln1(x), state=state)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, next_state

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = self.ln1(x_t)
        h, next_state = self.attn.step(norm_x, state)
        x = x_t + h
        x = x + self.mlp(self.ln2(x))
        return x, next_state


class HoloCausalLM(nn.Module):
    """
    Modelo de Lenguaje Causal Holográfico (Arquitectura moderna estilo LLaMA/Mistral).
    Inferencia token a token O(1) sin KV-Cache.
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 192,
        depth: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 512
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList([
            HoloCausalBlock(dim=dim, num_heads=num_heads)
            for _ in range(depth)
        ])

        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        if states is None:
            states = [None] * self.depth

        next_states = []
        for block, s in zip(self.blocks, states):
            x, next_s = block(x, state=s)
            next_states.append(next_s)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, next_states

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 50,
        temperature: float = 0.5,
        top_k: int = 30,
        eos_id: int = 50256
    ) -> List[int]:
        """
        Generación autoregresiva causal:
        1. Procesa el prompt para enfundar la intención en los estados de memoria.
        2. Genera los nuevos tokens en O(1) estricto.
        """
        self.eval()
        device = self.token_emb.weight.device
        current_tokens = list(prompt_tokens)

        # 1. Pase de calentamiento (Prefill) sobre el prompt
        prompt_tensor = torch.tensor([current_tokens], device=device, dtype=torch.long)
        logits, states = self.forward(prompt_tensor)

        # 2. Generación token a token
        for _ in range(max_new_tokens):
            last_logits = logits[:, -1, :] / max(temperature, 1e-4)

            # Top-K filtering para evitar tokens raros
            if top_k > 0:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < v[:, [-1]]] = -float('Inf')

            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == eos_id:
                break

            current_tokens.append(next_token)

            # Paso incremental O(1)
            next_tensor = torch.tensor([[next_token]], device=device, dtype=torch.long)
            logits, states = self.forward(next_tensor, states=states)

        return current_tokens

    def audit_hashes(self) -> Dict[str, str]:
        return {
            "class_code_hash": compute_object_code_hash(self.__class__),
            "state_dict_hash": compute_state_dict_hash(self.state_dict())
        }
