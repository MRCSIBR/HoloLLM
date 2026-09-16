"""
Módulo: src/models/holo_ads_llm.py
Fundamento Teórico:
- Juan Maldacena (1997): Correspondencia AdS/CFT aplicada a modelos de lenguaje.
- Edward Witten (1998): Propagador frontera-a-volumen conforme e integración gravitacional.
- Vitaly Vanchurin (2020): La red neuronal genera una dimensión espacial emergente durante el aprendizaje.
- Tony Plate & Karl Pribram: Atención holográfica causal O(1) en la frontera.
"""

from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn

from src.layers.adscft import AdSCFTBulkBoundaryLayer
from src.layers.holo_attention import MultiHeadHoloAttention
from src.utils.audit import compute_object_code_hash, compute_state_dict_hash


class HoloAdSBlock(nn.Module):
    """
    Bloque Holográfico de Maldacena.
    1. Proyecta los tokens al Bulk AdS curvo (Geometría / Flujo de Renormalización RG).
    2. Realiza mezcla temporal causal O(1) mediante MultiHeadHoloAttention.
    3. Mezcla canales mediante feed-forward no lineal.
    """
    def __init__(
        self,
        dim: int,
        seq_len: int,
        num_heads: int = 4,
        bulk_slices: int = 8,
        z_max: float = 2.0
    ) -> None:
        super().__init__()
        self.ln_ads = nn.LayerNorm(dim)
        self.bulk_layer = AdSCFTBulkBoundaryLayer(
            dim=dim,
            seq_len=seq_len,
            bulk_depth_slices=bulk_slices,
            z_max=z_max
        )

        self.ln_attn = nn.LayerNorm(dim)
        self.attn = MultiHeadHoloAttention(
            dim=dim,
            num_heads=num_heads,
            decay_min=0.88,
            decay_max=0.995
        )

        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Resonancia en el Bulk AdS
        norm_x = self.ln_ads(x)
        ads_out, bulk_field = self.bulk_layer(norm_x)
        x = x + ads_out

        # 2. Atención Causal Holográfica O(1)
        h_attn, next_state = self.attn(self.ln_attn(x), state=state)
        x = x + h_attn

        # 3. No linealidad Feed-Forward
        x = x + self.mlp(self.ln_mlp(x))

        return x, next_state, bulk_field


class HoloAdSLLM(nn.Module):
    """
    Modelo de Lenguaje AdS/CFT (HoloAdS-LLM).
    Sintetiza una dimensión espacial extra z durante la inferencia y genera texto en O(1).
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
        bulk_slices: int = 16,
        z_max: float = 2.5
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList([
            HoloAdSBlock(
                dim=dim,
                seq_len=max_seq_len,
                num_heads=num_heads,
                bulk_slices=bulk_slices,
                z_max=z_max
            )
            for _ in range(depth)
        ])

        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        if states is None:
            states = [None] * self.depth

        next_states = []
        bulk_fields = []

        for block, s in zip(self.blocks, states):
            x, next_s, bulk_f = block(x, state=s)
            next_states.append(next_s)
            bulk_fields.append(bulk_f)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, next_states, bulk_fields

    @torch.no_grad()
    def generate_and_inspect_bulk(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 0.4
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Genera la continuación y extrae el espaciotiempo curvo AdS creado por el prompt.
        """
        self.eval()
        current_ids = prompt_ids[0].tolist()

        # Prefill para obtener el campo gravitacional del Bulk
        tensor_in = torch.tensor([current_ids], device=prompt_ids.device, dtype=torch.long)
        logits, states, bulk_fields = self.forward(tensor_in)

        # Extraemos el campo gravitacional de la última capa AdS para visualización
        last_bulk = bulk_fields[-1][0].detach().cpu()  # [Z, T, D]

        for _ in range(max_new_tokens):
            last_logits = logits[:, -1, :] / max(temperature, 1e-4)
            probs = torch.softmax(last_logits, dim=-1)
            next_tok = torch.argmax(probs, dim=-1).item()

            current_ids.append(next_tok)

            next_t = torch.tensor([[next_tok]], device=prompt_ids.device, dtype=torch.long)
            logits, states, _ = self.forward(next_t, states=states)

        return current_ids, last_bulk

    def audit_hashes(self) -> Dict[str, str]:
        return {
            "class_code_hash": compute_object_code_hash(self.__class__),
            "state_dict_hash": compute_state_dict_hash(self.state_dict())
        }
