"""
Script: download_corpus.py
Propósito: Descargar y formatear un corpus estructurado de código Python
y diálogo instructivo para el entrenamiento de HoloSeq2Seq.
"""

import os
import json
import urllib.request
from typing import List, Dict

CORPUS_DIR = "data"
CORPUS_FILE = os.path.join(CORPUS_DIR, "python_code_corpus.json")

# Dataset de pares: [Instrucción / Tarea] -> [Código Python Correcto]
# Descargamos una muestra curada de funciones reales de la biblioteca estándar y algoritmos
SAMPLE_DATA_URL = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"


def download_and_filter_corpus(target_samples: int = 1500) -> None:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    print(f"Descargando corpus de código desde CodeAlpaca...")

    try:
        req = urllib.request.Request(
            SAMPLE_DATA_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as response:
            raw_data = json.loads(response.read().decode("utf-8"))

        print(f"Dataset descargado. Filtrando {target_samples} muestras limpias y cortas para CPU...")
        filtered_pairs = []

        for item in raw_data:
            instruction = item.get("instruction", "").strip()
            code = item.get("output", "").strip()

            # Filtramos solo código conciso y relevante en Python (ideal para CPU local)
            if 20 < len(instruction) < 120 and 30 < len(code) < 300:
                if "def " in code or "return " in code or "print(" in code:
                    filtered_pairs.append({
                        "prompt": instruction,
                        "completion": code
                    })

            if len(filtered_pairs) >= target_samples:
                break

        with open(CORPUS_FILE, "w", encoding="utf-8") as f:
            json.dump(filtered_pairs, f, indent=2, ensure_ascii=False)

        print(f"✔ Corpus preparado con éxito en: {CORPUS_FILE}")
        print(f"Total de pares de código listos: {len(filtered_pairs)}")

        # Mostrar un ejemplo
        if filtered_pairs:
            print("\n--- Ejemplo de Par (Instrucción -> Código) ---")
            print(f"Prompt:     {filtered_pairs[0]['prompt']}")
            print(f"Completion:\n{filtered_pairs[0]['completion']}")

    except Exception as e:
        print(f"Error descargando el dataset: {e}")
        print("Creando fallback local con ejemplos sintéticos de depuración...")
        create_synthetic_fallback()


def create_synthetic_fallback() -> None:
    synthetic = [
        {"prompt": "corrige el error: def suma(a, b) retun a + b", "completion": "def suma(a, b):\n    return a + b"},
        {"prompt": "corrige el error: def resta(a, b): return a - b", "completion": "def resta(a, b):\n    return a - b"},
        {"prompt": "funcion para saber si un numero es par", "completion": "def es_par(n):\n    return n % 2 == 0"},
        {"prompt": "funcion para calcular el cuadrado", "completion": "def cuadrado(x):\n    return x ** 2"},
        {"prompt": "corrige la sintaxis: if x = 5 print(x)", "completion": "if x == 5:\n    print(x)"}
    ]
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(synthetic, f, indent=2, ensure_ascii=False)
    print(f"✔ Fallback guardado con {len(synthetic)} ejemplos.")


if __name__ == "__main__":
    download_and_filter_corpus()
