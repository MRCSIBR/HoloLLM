"""
Módulo: src/models/holographic_lm.py
Fundamento Teórico:
- David Bohm: Enfundado del flujo temporal (tokens) en el Orden Implicado.
- Juan Maldacena (AdS/CFT): La frontera 1D (tiempo/tokens) proyectada al bulk espectral.
- Tony Plate & Karl Pribram: Memoria asociativa distribuida y modulación holonómica.
"""

from typing import Tuple, Optional, Dict, Any, List
import torch
import torch.nn as nn

from src.layers.associative_memory import HolographicAssociativeMemory
from src.layers.fourier import HolographicFourierInterference
from src.utils.audit import compute_object_code_hash, compute_state_dict_hash


class HolographicBlock(nn.Module):
    """
    Bloque fundamental del HoloLM.
    Combina mezcla temporal asociativa (HAM) con filtrado espectral de canales (HFI).
    """

    def __init__(self, dim: int, modes: int = 32, decay: float = 0.98) -> None:
        super().__init__()
        self.dim = dim

        # Sub-bloque 1: Mezcla temporal / Memoria Asociativa
        self.norm1 = nn.LayerNorm(dim)
        self.ham = HolographicAssociativeMemory(dim=dim, decay=decay, learnable_decay=True)

        # Sub-bloque 2: Mezcla espectral y no linealidad
        self.norm2 = nn.LayerNorm(dim)
        self.hfi = HolographicFourierInterference(features=dim, modes=modes, use_bias=True)
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
        # 1. Enfundado y recuperación temporal
        norm_x1 = self.norm1(x)
        ham_out, next_state = self.ham(norm_x1, initial_state=state)
        x = x + ham_out

        # 2. Filtrado holográfico en el dominio espectral de frecuencias
        norm_x2 = self.norm2(x)
        B, T, D = norm_x2.shape
        # Proyección espectral aplicada a cada vector de dimensión latente
        spectral_out = self.hfi(norm_x2.contiguous().view(-1, D)).view(B, T, D)
        ff_out = self.mlp(spectral_out)
        x = x + ff_out

        return x, next_state


class HolographicLM(nn.Module):
    """
    Modelo de Lenguaje Holográfico (HoloLM).
    Capaz de predecir el siguiente token y generar texto en O(1) de memoria por paso.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        depth: int = 2,
        modes: int = 32,
        decay: float = 0.98,
        max_seq_len: int = 512
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList([
            HolographicBlock(dim=dim, modes=modes, decay=decay)
            for _ in range(depth)
        ])

        self.ln_final = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Compartir pesos entre embedding y lm_head (Weight Tying)
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            states: Lista de tensores de memoria [batch_size, dim], uno por cada capa.
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            next_states: Lista con los estados actualizados de cada capa.
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings de frontera (Boundary)
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        if states is None:
            states = [None] * self.depth

        next_states = []
        for block, layer_state in zip(self.blocks, states):
            x, new_s = block(x, state=layer_state)
            next_states.append(new_s)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits, next_states

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8
    ) -> torch.Tensor:
        """
        Generación autoregresiva eficiente procesando paso a paso.
        """
        self.eval()
        current_ids = prompt_ids

        for _ in range(max_new_tokens):
            # Evaluamos la secuencia
            logits, _ = self.forward(current_ids)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=1)

        return current_ids

    def audit_hashes(self) -> Dict[str, str]:
        return {
            "class_code_hash": compute_object_code_hash(self.__class__),
            "state_dict_hash": compute_state_dict_hash(self.state_dict())
        }
