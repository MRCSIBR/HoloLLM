"""
Módulo: src/layers/holo_attention.py
Versión Ultra-Rápida Vectorizada para GPUs NVIDIA (A100 / L40S).
Soporta entrenamiento paralelo masivo e inferencia paso a paso O(1) con broadcasting corregido.
"""

import math
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.fft as fft

from src.utils.audit import compute_object_code_hash, compute_state_dict_hash


class MultiHeadHoloAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        decay_min: float = 0.85,
        decay_max: float = 0.999
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"Dimensión {dim} no es divisible por num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        decays = torch.exp(torch.linspace(math.log(decay_min), math.log(decay_max), num_heads))
        self.register_buffer("decays", decays.view(1, num_heads, 1, 1))

    def _to_unitary(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        x_fft = fft.fft(x_fp32, n=self.head_dim, dim=-1)
        phase = torch.angle(x_fft)
        unitary_fft = torch.complex(torch.cos(phase), torch.sin(phase))
        return fft.ifft(unitary_fft, n=self.head_dim, dim=-1).real.to(orig_dtype)

    def _bind(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        orig_dtype = k.dtype
        k_fft = fft.fft(k.float(), n=self.head_dim, dim=-1)
        v_fft = fft.fft(v.float(), n=self.head_dim, dim=-1)
        return fft.ifft(k_fft * v_fft, n=self.head_dim, dim=-1).real.to(orig_dtype)

    def _unbind(self, m: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        orig_dtype = q.dtype
        m_fft = fft.fft(m.float(), n=self.head_dim, dim=-1)
        q_fft = fft.fft(q.float(), n=self.head_dim, dim=-1)
        return fft.ifft(m_fft * torch.conj(q_fft), n=self.head_dim, dim=-1).real.to(orig_dtype)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        G = torch.sigmoid(self.gate_proj(x)).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q = self._to_unitary(Q)
        K = self._to_unitary(K)

        trace = self._bind(K, V)  # [B, H, T, d_h]

        t_indices = torch.arange(T, device=x.device, dtype=torch.float32).view(1, 1, T, 1)
        log_decays = torch.log(self.decays.float())
        decay_pow = torch.exp(t_indices * log_decays)
        inv_decay_pow = torch.exp(-t_indices * log_decays)

        scaled_trace = trace.float() * inv_decay_pow
        cum_trace = torch.cumsum(scaled_trace, dim=2)

        M = (cum_trace * decay_pow).to(x.dtype)

        if state is not None:
            init_decay = (state.unsqueeze(2).float() * (decay_pow * self.decays.float())).to(x.dtype)
            M = M + init_decay

        final_state = M[:, :, -1, :]

        retrieved = self._unbind(M, Q)
        gated_out = retrieved * G

        recombined = gated_out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(recombined)

        return out, final_state

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x_t.shape
        assert T == 1

        Q = self._to_unitary(self.q_proj(x_t).view(B, self.num_heads, self.head_dim))
        K = self._to_unitary(self.k_proj(x_t).view(B, self.num_heads, self.head_dim))
        V = self.v_proj(x_t).view(B, self.num_heads, self.head_dim)
        G = torch.sigmoid(self.gate_proj(x_t)).view(B, self.num_heads, self.head_dim)

        # Broadcasting inequívoco: [1, H, 1] * [B, H, d_h] -> [B, H, d_h]
        decays_step = self.decays.view(1, self.num_heads, 1).to(dtype=x_t.dtype)
        new_state = decays_step * state + self._bind(K, V)
        retrieved = self._unbind(new_state, Q) * G

        recombined = retrieved.view(B, 1, D)
        out = self.out_proj(recombined)
        return out, new_state

    def audit_hashes(self) -> Dict[str, str]:
        return {
            "class_code_hash": compute_object_code_hash(self.__class__),
            "state_dict_hash": compute_state_dict_hash(self.state_dict())
        }
