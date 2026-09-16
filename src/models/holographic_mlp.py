from typing import Dict
import torch
import torch.nn as nn
from src.layers.fourier import HolographicFourierInterference
from src.utils.audit import compute_object_code_hash, compute_state_dict_hash

class HolographicClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        modes: int = 32
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.modes = modes

        self.holo1 = HolographicFourierInterference(features=input_dim, modes=modes)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()

        self.holo2 = HolographicFourierInterference(features=hidden_dim, modes=modes)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, f"Shape de entrada incorrecto: {x.shape}"
        h1 = self.holo1(x)
        h1 = self.act(self.proj(h1))
        h2 = self.holo2(h1)
        h2 = self.norm(h2)
        return self.head(h2)

    def audit_hashes(self) -> Dict[str, str]:
        return {
            "class_code_hash": compute_object_code_hash(self.__class__),
            "state_dict_hash": compute_state_dict_hash(self.state_dict())
        }
