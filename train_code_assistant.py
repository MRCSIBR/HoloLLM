"""
Script: train_code_assistant.py
Propósito: Entrenar el primer asistente de código holográfico (HoloCodeLM, ~10.4M parámetros)
usando subwords BPE (tiktoken) sobre el corpus descargado de CodeAlpaca.
"""

import os
import json
import time
from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken

from src.models.holo_seq2seq import HoloSeq2Seq
from src.utils.seed import enforce_reproducibility
from src.utils.audit import generate_audit_metadata


CORPUS_PATH = "data/python_code_corpus.json"


class BPETokenizerWrapper:
    """Wrapper sobre tiktoken BPE (GPT-2) con delimitadores de control."""
    def __init__(self) -> None:
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = 50257
        # Usamos el token de fin de texto estándar de GPT-2 (50256)
        self.bos_id = 50256
        self.eos_id = 50256
        self.pad_id = 50256

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: List[int]) -> str:
        # Filtrar tokens de control antes de decodificar
        clean_tokens = [t for t in tokens if t < 50256]
        return self.enc.decode(clean_tokens)


def load_and_preprocess_dataset(
    tokenizer: BPETokenizerWrapper,
    max_samples: int = 400,
    max_src_len: int = 48,
    max_tgt_len: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Carga y tokeniza los pares [Instrucción -> Código]."""
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)[:max_samples]

    src_list, tgt_in_list, tgt_out_list = [], [], []

    for item in data:
        prompt = "Task: " + item["prompt"].strip() + "\nCode:"
        code = item["completion"].strip()

        src_tokens = tokenizer.encode(prompt)[:max_src_len]
        code_tokens = tokenizer.encode(code)[:max_tgt_len - 1]

        # Teacher Forcing causal
        tgt_in = [tokenizer.bos_id] + code_tokens
        tgt_out = code_tokens + [tokenizer.eos_id]

        src_list.append(torch.tensor(src_tokens, dtype=torch.long))
        tgt_in_list.append(torch.tensor(tgt_in, dtype=torch.long))
        tgt_out_list.append(torch.tensor(tgt_out, dtype=torch.long))

    # Padding homogéneo
    def pad(tensors: List[torch.Tensor], pad_val: int) -> torch.Tensor:
        m_len = max(len(t) for t in tensors)
        res = torch.full((len(tensors), m_len), pad_val, dtype=torch.long)
        for i, t in enumerate(tensors):
            res[i, :len(t)] = t
        return res

    src_batch = pad(src_list, tokenizer.pad_id)
    tgt_in_batch = pad(tgt_in_list, tokenizer.pad_id)
    tgt_out_batch = pad(tgt_out_list, -100)  # -100 es ignorado automáticamente por CrossEntropyLoss

    return src_batch, tgt_in_batch, tgt_out_batch


def main() -> None:
    SEED = 42
    enforce_reproducibility(SEED)

    tokenizer = BPETokenizerWrapper()

    # Dimensiones para 10.4M parámetros (óptimo para CPU local)
    DIM = 192
    NUM_HEADS = 4
    BATCH_SIZE = 16
    EPOCHS = 18
    LR = 2e-3
    SAMPLES = 500  # Subconjunto óptimo para entrenamiento rápido en CPU (< 2 minutos)

    print("\n=======================================================")
    print("  INICIALIZANDO HOLO-CODE LM (BPE SUBWORDS + O(1))    ")
    print("=======================================================")
    print(f"Vocabulario: {tokenizer.vocab_size:,} subwords BPE.")
    print(f"Dimensión latente: {DIM} | Cabezas: {NUM_HEADS} (d_h = {DIM//NUM_HEADS})")

    model = HoloSeq2Seq(
        vocab_size=tokenizer.vocab_size,
        dim=DIM,
        num_heads=NUM_HEADS,
        max_seq_len=256
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de Parámetros: {total_params:,} (~{total_params/1e6:.1f}M)")

    src_batch, tgt_in_batch, tgt_out_batch = load_and_preprocess_dataset(
        tokenizer=tokenizer,
        max_samples=SAMPLES
    )
    dataset = torch.utils.data.TensorDataset(src_batch, tgt_in_batch, tgt_out_batch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"\n--- [Train] Optimizando HoloCodeLM sobre {SAMPLES} funciones Python ---")
    start_time = time.time()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        tot_loss = 0.0
        for b_src, b_tgt_in, b_tgt_out in dataloader:
            optimizer.zero_grad()
            logits = model(b_src, b_tgt_in)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), b_tgt_out.view(-1))
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * b_src.size(0)

        epoch_loss = tot_loss / SAMPLES
        if epoch % 3 == 0 or epoch == EPOCHS:
            elapsed = time.time() - start_time
            print(f"Época [{epoch:02d}/{EPOCHS:02d}] | Loss: {epoch_loss:.4f} | Tiempo: {elapsed:.1f}s")

    # Guardar Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/holo_code_v1.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✔ Checkpoint HoloCodeLM guardado en: {ckpt_path}")

    # Evaluación y Demostración de Código en Vivo
    print("\n=======================================================")
    print("  DEMOSTRACIÓN: GENERACIÓN DE CÓDIGO CONTEXTUAL O(1)   ")
    print("=======================================================")

    test_prompts = [
        "Task: Write a replace method for a string class which replaces the given string.\nCode:",
        "Task: Create a python function to add two numbers.\nCode:",
        "Task: Write a function to check if a number is even.\nCode:"
    ]

    for p in test_prompts:
        p_tokens = tokenizer.encode(p)
        src_tensor = torch.tensor([p_tokens], dtype=torch.long)
        gen_tokens = model.generate(
            src_ids=src_tensor,
            start_token_id=tokenizer.bos_id,
            end_token_id=tokenizer.eos_id,
            max_len=45
        )
        gen_code = tokenizer.decode(gen_tokens)
        print(f"\n[Prompt]:\n{p}")
        print(f"[Código Generado por HoloCodeLM]:\n{gen_code}")
        print("-" * 50)


if __name__ == "__main__":
    main()
