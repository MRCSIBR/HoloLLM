r"""
SCRIPT: export_to_gguf.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM)
DESCRIPCIÓN: Convierte el checkpoint de PyTorch al formato GGUF.
             Versión blindada con inserción de claves a bajo nivel.
"""

import os
import sys
import torch
import gguf
from transformers import AutoTokenizer

CKPT_PATH = "checkpoints/holo_deep_distilled_75m.pt"
OUTPUT_PATH = "checkpoints/holollm-70m-f32.gguf"
TOKENIZER_ID = "Qwen/Qwen2.5-Coder-1.5B"

def main():
    print("=" * 80)
    print("📦 EMPAQUETADOR GGUF PARA HOLOLLM-70M (VULKAN READY)")
    print("=" * 80)

    if not os.path.exists(CKPT_PATH):
        print(f"Error: No se encontró el checkpoint en {CKPT_PATH}")
        sys.exit(1)

    print("[1/4] Cargando pesos de PyTorch...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    print("[2/4] Preparando archivo GGUF...")
    writer = gguf.GGUFWriter(OUTPUT_PATH, "holollm")

    writer.add_string("general.name", "HoloLLM-70M-O1")
    writer.add_string("general.description", "Holographic Causal Language Model with O(1) Constant Memory")
    
    file_type = getattr(gguf, "LlamaFileType", None)
    if file_type is not None and hasattr(file_type, "ALL_F32"):
        writer.add_file_type(file_type.ALL_F32)
    else:
        writer.add_file_type(0)

    writer.add_uint32("holollm.context_length", 172)
    writer.add_uint32("holollm.embedding_length", 384)
    writer.add_uint32("holollm.block_count", 6)
    writer.add_uint32("holollm.attention.head_count", 8)

    print("[3/4] Extrayendo vocabulario del tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    
    vocab_size = len(tokenizer)
    tokens = []
    for i in range(vocab_size):
        tok_bytes = tokenizer.convert_ids_to_tokens(i)
        if isinstance(tok_bytes, bytes):
            tokens.append(tok_bytes.decode('utf-8', errors='replace'))
        elif isinstance(tok_bytes, str):
            tokens.append(tok_bytes)
        else:
            tokens.append(f"[UNK_{i}]")

    writer.add_token_list(tokens)
    writer.add_uint32("general.vocab_size", vocab_size)
    
    # FIX: Insertar IDs especiales directamente como variables uint32
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 151643
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 151645
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643
    
    writer.add_uint32("tokenizer.ggml.bos_token_id", bos_id)
    writer.add_uint32("tokenizer.ggml.eos_token_id", eos_id)
    writer.add_uint32("tokenizer.ggml.padding_token_id", pad_id)

    print("[4/4] Transcribiendo tensores al formato GGUF (float32)...")
    for name, tensor in state_dict.items():
        if "pos_emb.weight" in name:
            tensor = tensor[:172]
        
        np_tensor = tensor.float().numpy()
        
        gguf_name = name.replace("token_emb.weight", "token_embd.weight") \
                        .replace("lm_head.weight", "output.weight") \
                        .replace("ln_f.", "output_norm.")
        
        writer.add_tensor(gguf_name, np_tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print("\n" + "=" * 80)
    print("✔ EXPORTACIÓN GGUF COMPLETADA CON ÉXITO")
    print(f"- Archivo creado: {OUTPUT_PATH}")
    print(f"- Tamaño: {size_mb:.1f} MB")
    print("=" * 80)

if __name__ == "__main__":
    main()
