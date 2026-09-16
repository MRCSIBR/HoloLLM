import torch
import torch.nn as nn
import torch.fft as fft

class HolographicReducedRepresentation(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._validate_operands(a, b)
        a_fft = fft.fft(a, n=self.dim, dim=-1)
        b_fft = fft.fft(b, n=self.dim, dim=-1)
        bound_fft = a_fft * b_fft
        return fft.ifft(bound_fft, n=self.dim, dim=-1).real

    def unbind(self, trace: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        self._validate_operands(trace, key)
        trace_fft = fft.fft(trace, n=self.dim, dim=-1)
        key_fft = fft.fft(key, n=self.dim, dim=-1)
        unbound_fft = trace_fft * torch.conj(key_fft)
        return fft.ifft(unbound_fft, n=self.dim, dim=-1).real

    def normalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norms + eps)

    def _validate_operands(self, a: torch.Tensor, b: torch.Tensor) -> None:
        assert a.shape[-1] == self.dim and b.shape[-1] == self.dim, (
            f"Operandos deben coincidir con dim HRR {self.dim}. Recibidos: {a.shape}, {b.shape}"
        )
        assert a.shape == b.shape, f"Los shapes deben ser idénticos. Recibidos: {a.shape} vs {b.shape}"
