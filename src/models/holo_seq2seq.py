"""
Módulo: src/models/holo_seq2seq.py
Fundamento Teórico:
- David Bohm: Enfundado del documento fuente en la traza M_enc y despliegue causal en el decodificador.
- Karl Pribram: Recuperación asociativa por haz de referencia cruzado (Cross-Unbinding).
- Tony Plate: Álgebra unitaria en subespacios de atención multicanal.
"""

from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import torch.fft as fft

from src.layers.holo_attention import MultiHeadHoloAttention
from src.utils.audit import compute_object_code_hash, compute_state_dict_hash


class HoloCrossAttention(nn.Module):
    """
    Atención Cruzada Holográfica O(1).
    Consulta el estado enfundado del encoder M_enc mediante correlación circular directa.
    """
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _to_unitary(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = fft.fft(x, n=self.head_dim, dim=-1)
        phase = torch.angle(x_fft)
        unitary_fft = torch.complex(torch.cos(phase), torch.sin(phase))
        return fft.ifft(unitary_fft, n=self.head_dim, dim=-1).real

    def forward(self, dec_x: torch.Tensor, m_enc: torch.Tensor) -> torch.Tensor:
        B, T, D = dec_x.shape
        Q = self.q_proj(dec_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        Q = self._to_unitary(Q)
        G = torch.sigmoid(self.gate_proj(dec_x)).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Correlación circular cruzada contra el holograma del documento M_enc
        m_fft = fft.fft(m_enc.unsqueeze(2), n=self.head_dim, dim=-1)  # [B, H, 1, d_h]
        q_fft = fft.fft(Q, n=self.head_dim, dim=-1)                   # [B, H, T, d_h]
        retrieved_fft = m_fft * torch.conj(q_fft)
        retrieved = fft.ifft(retrieved_fft, n=self.head_dim, dim=-1).real

        gated = (retrieved * G).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(gated)


class HoloSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 96,
        num_heads: int = 4,
        max_seq_len: int = 256
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # Encoder
        self.enc_ln1 = nn.LayerNorm(dim)
        self.enc_attn = MultiHeadHoloAttention(dim=dim, num_heads=num_heads, decay_min=0.92, decay_max=0.999)
        self.enc_ln2 = nn.LayerNorm(dim)
        self.enc_mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

        # Decoder
        self.dec_ln1 = nn.LayerNorm(dim)
        self.dec_self_attn = MultiHeadHoloAttention(dim=dim, num_heads=num_heads, decay_min=0.85, decay_max=0.98)
        self.dec_ln2 = nn.LayerNorm(dim)
        self.dec_cross_attn = HoloCrossAttention(dim=dim, num_heads=num_heads)
        self.dec_ln3 = nn.LayerNorm(dim)
        self.dec_mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        B, T = src_ids.shape
        pos = torch.arange(0, T, device=src_ids.device).unsqueeze(0)
        x = self.token_emb(src_ids) + self.pos_emb(pos)

        h, m_enc = self.enc_attn(self.enc_ln1(x))
        x = x + h
        x = x + self.enc_mlp(self.enc_ln2(x))
        return m_enc

    def decode(
        self,
        tgt_ids: torch.Tensor,
        m_enc: torch.Tensor,
        dec_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = tgt_ids.shape
        pos = torch.arange(0, T, device=tgt_ids.device).unsqueeze(0)
        x = self.token_emb(tgt_ids) + self.pos_emb(pos)

        h_self, next_dec_state = self.dec_self_attn(self.dec_ln1(x), state=dec_state)
        x = x + h_self

        h_cross = self.dec_cross_attn(self.dec_ln2(x), m_enc)
        x = x + h_cross

        x = x + self.dec_mlp(self.dec_ln3(x))
        logits = self.lm_head(x)
        return logits, next_dec_state

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        m_enc = self.encode(src_ids)
        logits, _ = self.decode(tgt_ids, m_enc)
        return logits

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_len: int = 35
    ) -> List[int]:
        self.eval()
        m_enc = self.encode(src_ids)
        generated = [start_token_id]

        for _ in range(max_len):
            tgt_tensor = torch.tensor([generated], device=src_ids.device, dtype=torch.long)
            logits, _ = self.decode(tgt_tensor, m_enc)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            if next_token == end_token_id:
                break
            generated.append(next_token)

        return generated

    def audit_hashes(self) -> Dict[str, str]:
        return {
            "class_code_hash": compute_object_code_hash(self.__class__),
            "state_dict_hash": compute_state_dict_hash(self.state_dict())
        }
