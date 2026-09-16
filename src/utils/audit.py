import hashlib
import inspect
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn

def compute_tensor_hash(tensor: torch.Tensor) -> str:
    tensor_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()

def compute_state_dict_hash(state_dict: Dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode("utf-8"))
        hasher.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()

def compute_object_code_hash(obj: Any) -> str:
    try:
        source_code = inspect.getsource(obj)
        return hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    except (TypeError, OSError):
        return "SourceNotAvailable"

def generate_audit_metadata(
    model: nn.Module,
    hyperparams: Dict[str, Any],
    seed: int
) -> Dict[str, Any]:
    return {
        "model_class": model.__class__.__name__,
        "model_code_hash": compute_object_code_hash(model.__class__),
        "weights_hash": compute_state_dict_hash(model.state_dict()),
        "seed": seed,
        "hyperparameters": hyperparams,
        "torch_version": str(torch.__version__),  # Forzado a str nativo de Python
        "cuda_available": bool(torch.cuda.is_available())
    }
