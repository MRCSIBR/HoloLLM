r"""
MÓDULO: src/engine/holo_engine.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM)
DESCRIPCIÓN: Motor de Inferencia Local O(1) optimizado para CPU y baja latencia.
             Mantiene el estado recurrente holográfico perpetuamente en el Orden Implicado.

FUNDAMENTACIÓN TEÓRICA:
1. David Bohm: El estado holográfico S_t se mantiene plegado en el dominio espectral.
   Solo se despliega al Orden Explicado mediante la proyección lineal de salida (lm_head).
2. Tony Plate: Recuperación asociativa instantánea mediante correlación circular unitaria.
3. Complejidad por paso: O(D) en memoria, O(D log D) en cómputo (con D_head = 48).
"""

import os
import sys
import time
import math
import hashlib
import inspect
from typing import Generator, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.models.holo_causal_lm import HoloCausalLM


class HoloInferenceEngine:
    """
    Motor de inferencia autorregresiva de alto rendimiento con estado O(1).
    Diseñado para ejecutar a máxima velocidad en CPUs sin dependencias externas pesadas.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/holo_deep_distilled_75m.pt",
        tokenizer_id: str = "Qwen/Qwen2.5-Coder-1.5B",
        device: Optional[str] = None,
        max_seq_len: int = 172,
        dim: int = 384,
        depth: int = 6,
        num_heads: int = 8,
        vocab_size: int = 151936
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Optimización de hilos en CPU
        if self.device.type == "cpu":
            num_cores = os.cpu_count() or 4
            torch.set_num_threads(num_cores)
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16

        print(f"[HoloEngine] Inicializando motor en: {self.device} (dtype: {self.dtype})")

        # 1. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        self.eos_ids = {151645, 151643}

        # 2. Arquitectura HoloCausalLM
        self.model = HoloCausalLM(
            vocab_size=vocab_size,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        ).to(device=self.device, dtype=self.dtype).eval()

        # 3. Cargar Pesos
        assert os.path.exists(checkpoint_path), (
            f"Checkpoint no encontrado en {checkpoint_path}. "
            f"Asegúrate de que el archivo holo_deep_distilled_75m.pt esté en la carpeta checkpoints/."
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

        # Ajuste de tamaño de pos_emb si difiere de max_seq_len
        if "pos_emb.weight" in state_dict:
            state_dict["pos_emb.weight"] = state_dict["pos_emb.weight"][:max_seq_len]

        # Adaptación dinámica de tensores con igual número de elementos (ej: decays 4D vs 3D)
        for k in list(state_dict.keys()):
            if k in self.model.state_dict():
                target_shape = self.model.state_dict()[k].shape
                source_shape = state_dict[k].shape
                if source_shape != target_shape:
                    if state_dict[k].numel() == self.model.state_dict()[k].numel():
                        state_dict[k] = state_dict[k].reshape(target_shape)
                    else:
                        print(f"[HoloEngine] Aviso: numel mismatch en {k} ({source_shape} vs {target_shape})")

        self.model.load_state_dict(state_dict)
        print(f"[HoloEngine] Pesos cargados y alineados exitosamente desde {checkpoint_path}")

        # Preasignación de buffers para evitar overhead de memoria
        self.cur_pos_buf = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.token_buf = torch.zeros((1, 1), dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def prefill(self, prompt: str) -> Tuple[int, list, list, float]:
        start_t = time.perf_counter()
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        t_in = torch.tensor([token_ids], device=self.device, dtype=torch.long)

        out = self.model(t_in)
        logits, states = out if isinstance(out, tuple) else (out, None)

        first_token = torch.argmax(logits[0, -1, :]).item()
        ttft_ms = (time.perf_counter() - start_t) * 1000.0

        current_tokens = list(token_ids) + [first_token]
        return first_token, states, current_tokens, ttft_ms

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        stop_on_backticks: bool = True
    ) -> Generator[Tuple[str, float], None, None]:
        first_token, states, current_tokens, ttft_ms = self.prefill(prompt)
        first_str = self.tokenizer.decode([first_token])
        yield first_str, ttft_ms

        next_token = first_token

        for _ in range(max_new_tokens):
            cur_pos = len(current_tokens) - 1
            if cur_pos >= self.model.pos_emb.weight.size(0):
                break

            step_start = time.perf_counter()

            self.token_buf[0, 0] = next_token
            self.cur_pos_buf[0, 0] = cur_pos

            x_step = self.model.token_emb(self.token_buf) + self.model.pos_emb(self.cur_pos_buf)

            next_states = []
            h = x_step
            for block, s in zip(self.model.blocks, states):
                h, next_s = block.step(h, s)
                next_states.append(next_s)
            states = next_states

            logits = self.model.lm_head(self.model.ln_f(h))
            next_token = torch.argmax(logits[0, -1, :]).item()
            step_ms = (time.perf_counter() - step_start) * 1000.0

            if next_token in self.eos_ids:
                break

            tok_str = self.tokenizer.decode([next_token])
            if stop_on_backticks and "```" in tok_str:
                yield "\n```", step_ms
                break

            current_tokens.append(next_token)
            yield tok_str, step_ms

    @staticmethod
    def get_source_hash() -> str:
        source = inspect.getsource(HoloInferenceEngine)
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
