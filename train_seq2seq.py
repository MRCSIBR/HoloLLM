"""
Script: train_seq2seq.py
Propósito: Entrenar HoloSeq2Seq con Teacher Forcing causal estricto
y delimitadores atómicos (^ para inicio, $ para fin, _ para pad).
"""

import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.holo_seq2seq import HoloSeq2Seq
from src.utils.seed import enforce_reproducibility
from src.utils.audit import generate_audit_metadata


QA_PAIRS = [
    ("quien propuso el orden implicado?", "david bohm"),
    ("que teoria describe el cerebro como holograma?", "teoria holonomica de pribram"),
    ("que establece la correspondencia ads cft?", "dualidad holografica de maldacena"),
    ("quien formulo el universo como red neuronal?", "vitaly vanchurin"),
    ("que operacion usa plate para el binding?", "convolucion circular unitaria"),
    ("donde vive la frontera de baja dimension?", "en el espacio conforme cft"),
    ("donde vive el volumen interior gravitacional?", "en el bulk curvo ads"),
    ("que elimina la atencion holografica?", "el crecimiento del kv cache"),
]


class CleanSeq2SeqTokenizer:
    def __init__(self, pairs: List[Tuple[str, str]]) -> None:
        chars = set()
        for q, a in pairs:
            chars.update(list(q) + list(a))
        # Delimitadores atómicos de un solo carácter
        self.pad_char = "_"
        self.bos_char = "^"
        self.eos_char = "$"
        special = [self.pad_char, self.bos_char, self.eos_char]
        
        self.vocab = special + sorted(list(chars))
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char = {i: ch for i, ch in enumerate(self.vocab)}

        self.pad_id = self.char2idx[self.pad_char]
        self.bos_id = self.char2idx[self.bos_char]
        self.eos_id = self.char2idx[self.eos_char]

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char2idx[ch] for ch in text if ch in self.char2idx], dtype=torch.long)

    def decode(self, indices: List[int]) -> str:
        res = []
        for i in indices:
            if i == self.eos_id:
                break
            if i not in (self.pad_id, self.bos_id):
                res.append(self.idx2char.get(i, ""))
        return "".join(res)


def pad_batch(sequences: List[torch.Tensor], pad_val: int) -> torch.Tensor:
    max_len = max(len(s) for s in sequences)
    padded = torch.full((len(sequences), max_len), pad_val, dtype=torch.long)
    for i, s in enumerate(sequences):
        padded[i, :len(s)] = s
    return padded


def main() -> None:
    SEED = 42
    enforce_reproducibility(SEED)

    tokenizer = CleanSeq2SeqTokenizer(QA_PAIRS)
    vocab_size = len(tokenizer.vocab)

    print("\n=======================================================")
    print("  INICIALIZANDO HOLO-SEQ2SEQ (TEXT-TO-TEXT HOLOGRÁFICO)")
    print("=======================================================")
    print(f"Vocabulario: {vocab_size} caracteres. Delimitadores: BOS='^', EOS='$', PAD='_'")

    model = HoloSeq2Seq(vocab_size=vocab_size, dim=96, num_heads=4)
    optimizer = optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # Preparación con desplazamiento causal real:
    # tgt_in:  '^' + respuesta  (el decoder ve esto)
    # tgt_out: respuesta + '$'  (el decoder debe predecir esto)
    src_tensors = [tokenizer.encode(q) for q, _ in QA_PAIRS]
    tgt_in_tensors = [tokenizer.encode(tokenizer.bos_char + a) for _, a in QA_PAIRS]
    tgt_out_tensors = [tokenizer.encode(a + tokenizer.eos_char) for _, a in QA_PAIRS]

    src_batch = pad_batch(src_tensors, tokenizer.pad_id)
    tgt_in_batch = pad_batch(tgt_in_tensors, tokenizer.pad_id)
    tgt_out_batch = pad_batch(tgt_out_tensors, tokenizer.pad_id)

    print("\n--- [Train] Optimizando transducción causal en el Orden Implicado ---")
    model.train()
    for epoch in range(1, 45 + 1):
        optimizer.zero_grad()
        logits = model(src_batch, tgt_in_batch)
        loss = criterion(logits.view(-1, vocab_size), tgt_out_batch.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 45:
            print(f"Época [{epoch:02d}/45] | Causal Cross-Entropy Loss: {loss.item():.4f}")

    # Guardar Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/holo_seq2seq_v2.pt"
    audit_meta = generate_audit_metadata(model=model, hyperparams={"dim": 96, "heads": 4}, seed=SEED)
    torch.save({"state_dict": model.state_dict(), "audit": audit_meta}, ckpt_path)
    print(f"\n✔ Checkpoint corregido guardado en: {ckpt_path}")

    # Evaluación
    print("\n=======================================================")
    print("  EVALUACIÓN TEXT-TO-TEXT (ENCODER -> DECODER O(1))    ")
    print("=======================================================")

    test_questions = [
        "quien propuso el orden implicado?",
        "que teoria describe el cerebro como holograma?",
        "quien formulo el universo como red neuronal?",
        "que elimina la atencion holografica?"
    ]

    for q in test_questions:
        src_ids = tokenizer.encode(q).unsqueeze(0)
        generated_ids = model.generate(
            src_ids,
            start_token_id=tokenizer.bos_id,
            end_token_id=tokenizer.eos_id,
            max_len=40
        )
        ans = tokenizer.decode(generated_ids)
        print(f"\nPregunta:   '{q}'")
        print(f"Respuesta:  '{ans}'")


if __name__ == "__main__":
    main()
