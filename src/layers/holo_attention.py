"""
Módulo: src/layers/holo_attention.py
Fundamento Teórico:
- Karl Pribram (1991): Filtros dendríticos multiescala con diferentes constantes de tiempo.
- Tony Plate (2003): Álgebra asociativa en subespacios unitarios paralelos (Multi-Head HRR).
- David Bohm: Enfundado recurrente de múltiples canales del Orden Implicado.

Propósito:
Reemplazo directo (drop-in) de la Multi-Head Attention convencional.
Elimina completamente la necesidad del KV-Cache en LLMs, logrando inferencia O(1).
"""

import math
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.fft as fft

from src.utils.audit import compute_object_code_hash, compute_state_dict_hash


class MultiHeadHoloAttention(nn.Module):
    """
    Multi-Head Holographic Attention (MH-HoloAttention).
    
    Divide la dimensión D en H cabezas de dimensión d_h = D // H.
    Cada cabeza mantiene su propio vector de memoria M_t de dimensión fija [B, d_h].
    """

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

        # Proyecciones lineales conjuntas para eficiencia de cómputo
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Decaimientos multiescala por cabeza (inspirados en Pribram y ALiBi/RetNet)
        # Cabezas tempranas atienden al contexto inmediato; cabezas tardías a largo plazo.
        decays = torch.exp(torch.linspace(math.log(decay_min), math.log(decay_max), num_heads))
        self.register_buffer("decays", decays.view(1, num_heads, 1))

    def _to_unitary(self, x: torch.Tensor) -> torch.Tensor:
        """
        Proyección de Plate: Normaliza la magnitud de cada modo espectral a 1.0 en cada cabeza.
        Forma de entrada/salida: [Batch, Heads, Tokens, head_dim]
        """
        x_fft = fft.fft(x, n=self.head_dim, dim=-1)
        phase = torch.angle(x_fft)
        unitary_fft = torch.complex(torch.cos(phase), torch.sin(phase))
        return fft.ifft(unitary_fft, n=self.head_dim, dim=-1).real

    def _bind(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Convolución circular batched por cabeza (K ⊛ V)."""
        k_fft = fft.fft(k, n=self.head_dim, dim=-1)
        v_fft = fft.fft(v, n=self.head_dim, dim=-1)
        return fft.ifft(k_fft * v_fft, n=self.head_dim, dim=-1).real

    def _unbind(self, m: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Correlación circular batched por cabeza (M ⊚ Q)."""
        m_fft = fft.fft(m, n=self.head_dim, dim=-1)
        q_fft = fft.fft(q, n=self.head_dim, dim=-1)
        return fft.ifft(m_fft * torch.conj(q_fft), n=self.head_dim, dim=-1).real

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [Batch, SeqLen, Dim]
            state: [Batch, Heads, head_dim] (opcional: estado holográfico previo O(1))
        Returns:
            out: [Batch, SeqLen, Dim]
            next_state: [Batch, Heads, head_dim]
        """
        B, T, D = x.shape

        # 1. Proyecciones y separación en cabezas [B, H, T, d_h]
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        G = torch.sigmoid(self.gate_proj(x)).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Proyección Unitaria de Plate
        Q = self._to_unitary(Q)
        K = self._to_unitary(K)

        # 2. Inicializar estado de memoria [B, H, d_h]
        if state is not None:
            assert state.shape == (B, self.num_heads, self.head_dim)
            M_t = state
        else:
            M_t = torch.zeros(B, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)

        head_outputs = []

        # 3. Recurrencia temporal causal
        for t_step in range(T):
            q_t = Q[:, :, t_step, :]  # [B, H, d_h]
            k_t = K[:, :, t_step, :]  # [B, H, d_h]
            v_t = V[:, :, t_step, :]  # [B, H, d_h]

            # Enfundado holográfico del paso actual
            trace_t = self._bind(k_t, v_t)

            # Acumulación con decaimientos multiescala por cabeza
            M_t = self.decays * M_t + trace_t

            # Recuperación asociativa por correlación
            retrieved_v = self._unbind(M_t, q_t)
            head_outputs.append(retrieved_v)

        # 4. Apilado y aplicación de la compuerta inhibitoria de Pribram
        out_stacked = torch.stack(head_outputs, dim=2)  # [B, H, T, d_h]
        gated_out = out_stacked * G                     # Modulación local

        # Recombinación espacial de cabezas [B, T, D]
        recombined = gated_out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(recombined)

        return out, M_t

    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Paso de Inferencia Autoregresiva Ultra-Rápido O(1).
        Procesa exactamente un token nuevo x_t [B, 1, Dim] sin mirar el pasado.
        """
        B, T, D = x_t.shape
        assert T == 1, "step() solo debe recibir 1 token a la vez"

        Q = self._to_unitary(self.q_proj(x_t).view(B, self.num_heads, self.head_dim))
        K = self._to_unitary(self.k_proj(x_t).view(B, self.num_heads, self.head_dim))
        V = self.v_proj(x_t).view(B, self.num_heads, self.head_dim)
        G = torch.sigmoid(self.gate_proj(x_t)).view(B, self.num_heads, self.head_dim)

        # Actualizar memoria fija O(1)
        new_state = self.decays * state + self._bind(K, V)
        retrieved = self._unbind(new_state, Q) * G

        recombined = retrieved.view(B, 1, D)
        out = self.out_proj(recombined)
        return out, new_state

    def audit_hashes(self) -> Dict[str, str]:
        return {
            "class_code_hash": compute_object_code_hash(self.__class__),
            "state_dict_hash": compute_state_dict_hash(self.state_dict())
        }
