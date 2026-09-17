r"""
MÓDULO: src/layers/holo_attention_v2.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM) - Arquitectura v2 (Aceleración Vectorizada)
DESCRIPCIÓN: Atención Causal Holográfica Multiescala con Convolución Temporal Causal por FFT
             (Bohmian Causal Filter). Elimina bucles secuenciales de Python en forward.
"""

import math
import hashlib
import inspect
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadHoloAttentionV2(nn.Module):
    """
    Capa de Atención Holográfica Multicabeza v2 vectorizada para GPUs de alta densidad.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        rope_base: float = 10000.0,
        eps: float = 1e-8
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) debe ser divisible por num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps
        self.rope_base = rope_base

        # Proyecciones lineales
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Frecuencias de rotación continua (RoPE)
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Decaimientos multiescala sigmoides acotados en (0, 1)
        lambdas_init = torch.linspace(0.75, 0.999, num_heads)
        alpha_init = torch.log(lambdas_init / (1.0 - lambdas_init))
        self.decay_logits = nn.Parameter(alpha_init.view(1, num_heads, 1))

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        angles = torch.outer(positions.float(), self.inv_freq.to(positions.device))
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(1)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(1)

        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.to(dtype=x.dtype)

    def _to_unitary(self, tensor: torch.Tensor) -> torch.Tensor:
        f = torch.fft.rfft(tensor.float(), dim=-1)
        mag = torch.clamp(f.abs(), min=self.eps)
        f_unit = f / mag
        return f_unit

    def _bind(self, k_comp: torch.Tensor, v_real: torch.Tensor) -> torch.Tensor:
        v_comp = torch.fft.rfft(v_real.float(), dim=-1)
        return v_comp * k_comp.conj()

    def _unbind(self, state_comp: torch.Tensor, q_comp: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        readout = state_comp * q_comp
        real = torch.fft.irfft(readout, n=self.head_dim, dim=-1)
        return real.to(dtype=target_dtype)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        if positions is None:
            positions = torch.arange(S, device=x.device, dtype=torch.long)

        # Proyecciones
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        G = torch.sigmoid(self.gate_proj(x)).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # RoPE continuo
        Q = self._apply_rope(Q, positions)
        K = self._apply_rope(K, positions)

        # Toro unitario
        K_unit = self._to_unitary(K)
        Q_unit = self._to_unitary(Q)

        # Enlace espectral en frecuencias: (B, H, S, Freq_Dim) en complex64
        B_comp = self._bind(K_unit, V)

        # =========================================================================
        # VECTORIZACIÓN PARALELA: Convolución Causal Temporal por FFT (O(S log S))
        # Reemplaza el bucle secuencial for t in range(S) por una sola operación GPU
        # =========================================================================
        decays = torch.sigmoid(self.decay_logits).view(1, self.num_heads, 1, 1).to(dtype=torch.float32)
        t_seq = torch.arange(S, device=x.device, dtype=torch.float32).view(1, 1, S, 1)
        log_decays = torch.log(decays.clamp(min=1e-7, max=1.0 - 1e-7))
        w = torch.exp(t_seq * log_decays) # Kernel causal (1, H, S, 1)

        # Padding a 2S para evitar convolución circular en el eje temporal
        N_fft = 2 * S
        B_pad = torch.cat([B_comp, torch.zeros_like(B_comp)], dim=-2)
        w_pad = torch.cat([w, torch.zeros_like(w)], dim=-2).to(dtype=torch.complex64)

        # Convolución 1D por FFT a lo largo de la longitud de secuencia (dim=-2)
        B_fft = torch.fft.fft(B_pad, n=N_fft, dim=-2)
        w_fft = torch.fft.fft(w_pad, n=N_fft, dim=-2)
        states_fft = B_fft * w_fft
        states_full = torch.fft.ifft(states_fft, n=N_fft, dim=-2)

        # Estados causales exactos a lo largo de la secuencia
        states_tensor = states_full[:, :, :S, :]
        cur_state = states_tensor[:, :, -1, :] # Estado acumulado final al token S-1

        # Lectura holográfica casteada limpiamente al dtype de entrada (bfloat16 o float32)
        retrieved = self._unbind(states_tensor, Q_unit, target_dtype=x.dtype) * G
        out = retrieved.permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.out_proj(out)

        return out, cur_state

    def step(
        self,
        x_t: torch.Tensor,
        pos_t: int,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x_t.shape
        assert T == 1

        Q = self.q_proj(x_t).view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(x_t).view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(x_t).view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        G = torch.sigmoid(self.gate_proj(x_t)).view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        pos_tensor = torch.tensor([pos_t], device=x_t.device, dtype=torch.long)
        Q = self._apply_rope(Q, pos_tensor)
        K = self._apply_rope(K, pos_tensor)

        K_unit = self._to_unitary(K)[:, :, 0, :]
        Q_unit = self._to_unitary(Q)[:, :, 0, :]
        V_real = V[:, :, 0, :]

        B_comp = self._bind(K_unit, V_real)
        decays = torch.sigmoid(self.decay_logits).to(dtype=torch.float32)
        new_state = decays * state + B_comp

        retrieved = self._unbind(new_state, Q_unit, target_dtype=x_t.dtype) * G[:, :, 0, :]
        recombined = retrieved.reshape(B, 1, D)
        out = self.out_proj(recombined)

        return out, new_state

    @staticmethod
    def get_source_hash() -> str:
        source = inspect.getsource(MultiHeadHoloAttentionV2)
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
