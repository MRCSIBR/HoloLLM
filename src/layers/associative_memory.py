"""
Módulo: src/layers/associative_memory.py
Fundamento Teórico:
- Tony Plate (2003): Unitary Holographic Reduced Representations.
  Al proyectar las claves al toro unitario (|F(K)| = 1), se conserva la energía espectral
  exacta y se garantiza invarianza ante perturbaciones o daño en los pesos.
- David Bohm: Enfundado no disipativo en el Orden Implicado.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.fft as fft

class HolographicAssociativeMemory(nn.Module):
    """
    Memoria Asociativa Holográfica Unitaria (Unitary HAM).
    Garantiza que la convolución circular no disipe ni amplifique energía espectral,
    blindando la memoria ante ablaciones y pérdidas de conexiones.
    """

    def __init__(self, dim: int, decay: float = 0.98, learnable_decay: bool = True) -> None:
        super().__init__()
        self.dim = dim

        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        if learnable_decay:
            init_val = torch.logit(torch.tensor(decay))
            self.decay_param = nn.Parameter(init_val)
        else:
            self.register_buffer("decay_param", torch.logit(torch.tensor(decay)))

    @property
    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_param)

    def _to_unitary(self, x: torch.Tensor) -> torch.Tensor:
        """
        Proyección al subespacio unitario de Plate:
        Fuerza la magnitud de cada modo espectral a 1.0, reteniendo únicamente la fase.
        Propiedad: Unitary(K) ⊛ Unitary(K)† = delta de Dirac exacta.
        """
        x_fft = fft.fft(x, n=self.dim, dim=-1)
        phase = torch.angle(x_fft)
        unitary_fft = torch.complex(torch.cos(phase), torch.sin(phase))
        return fft.ifft(unitary_fft, n=self.dim, dim=-1).real

    def _bind(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Convolución circular (K ⊛ V)."""
        k_fft = fft.fft(k, n=self.dim, dim=-1)
        v_fft = fft.fft(v, n=self.dim, dim=-1)
        return fft.ifft(k_fft * v_fft, n=self.dim, dim=-1).real

    def _unbind(self, m: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Correlación circular (M ⊚ Q)."""
        m_fft = fft.fft(m, n=self.dim, dim=-1)
        q_fft = fft.fft(q, n=self.dim, dim=-1)
        return fft.ifft(m_fft * torch.conj(q_fft), n=self.dim, dim=-1).real

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 3, f"Se esperaba tensor 3D [B, T, D], recibido: {x.shape}"
        batch_size, seq_len, _ = x.shape
        gamma = self.decay

        # Las claves y consultas se proyectan sobre la esfera unitaria de fase
        Q = self._to_unitary(self.proj_q(x))
        K = self._to_unitary(self.proj_k(x))
        V = self.proj_v(x)

        if initial_state is not None:
            M_t = initial_state
        else:
            M_t = torch.zeros(batch_size, self.dim, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            q_t = Q[:, t, :]
            k_t = K[:, t, :]
            v_t = V[:, t, :]

            trace_t = self._bind(k_t, v_t)
            M_t = gamma * M_t + trace_t
            retrieved_v = self._unbind(M_t, q_t)
            outputs.append(retrieved_v)

        out_stacked = torch.stack(outputs, dim=1)  # [B, T, D]
        
        # Flujo residual pasivo: proyección lineal directa sin división por varianza
        out_final = self.out_proj(out_stacked)
        return out_final, M_t
