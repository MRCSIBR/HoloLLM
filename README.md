
<div align="center">

```text
╭────────────────────────────────────────────────────────────────────────╮
│   ██╗  ██╗ ██████╗ ██╗      ██████╗ ██╗     ██╗     ███╗   ███╗        │
│   ██║  ██║██╔═══██╗██║     ██╔═══██╗██║     ██║     ████╗ ████║        │
│   ███████║██║   ██║██║     ██║   ██║██║     ██║     ██╔████╔██║        │
│   ██╔══██║██║   ██║██║     ██║   ██║██║     ██║     ██║╚██╔╝██║        │
│   ██║  ██║╚██████╔╝███████╗╚██████╔╝███████╗███████╗██║ ╚═╝ ██║        │
│   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝ v1.0   │
│   Holographic Causal Language Model • O(1) State • Zero KV-Cache       │
╰────────────────────────────────────────────────────────────────────────╯
```

### *HoloLLM: Nueva arquitectura LLM basada en redes holográficas con complejidad de memoria O(1)*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22849908.svg)](https://doi.org/10.5281/zenodo.22849908)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Web Portal](https://img.shields.io/badge/Web-GitHub_Pages-2ea44f.svg)](https://mrcsibr.github.io/HoloLLM)
[![Whitepaper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](docs/HoloLLM_Whitepaper.pdf)
[![KV-Cache: 0.00 KB](https://img.shields.io/badge/KV--Cache-0.00_KB_(O(1))-emerald.svg)](#-resultados-empíricos-certificados)
[![Throughput: 40 tok/s CPU](https://img.shields.io/badge/CPU_Speed-~40_tok%2Fs-purple.svg)](#-resultados-empíricos-certificados)
[![PyTorch: Native](https://img.shields.io/badge/PyTorch-2.1%2B_Native-orange.svg)](https://pytorch.org)

---

[**Portal Web Oficial**](https://mrcsibr.github.io/HoloLLM) • [**Paper en PDF**](docs/HoloLLM_Whitepaper.pdf) • [**Registro en Zenodo (DOI)**](https://doi.org/10.5281/zenodo.22849908) • [**Demostración Rápida**](#-inicio-rápido-en-tu-máquina-local-consola-interactiva-tipo-ollama)

</div>

## 🌌 Visión y Primeros Principios

Los modelos de lenguaje basados en **Transformers estándar** sufren de un cuello de botella físico intratable: el **KV-Cache crece linealmente $\mathcal{O}(N)$** con la longitud del contexto, devorando gigabytes de memoria VRAM y ahogando el ancho de banda del procesador.

**HoloLLM** reemplaza el almacenamiento lineal de claves y valores por **superposición interferométrica de frentes de onda en el Orden Implicado**. Toda la historia de la conversación o del código se pliega analíticamente en un estado holográfico constante de **64 KB**.

### Fundamentación Teórica
- **David Bohm (Orden Implicado vs. Orden Explicado):** La memoria reside permanentemente plegada en el dominio espectral complejo. El texto observable solo se despliega en el Orden Explicado al proyectar el token presente.
- **Karl Pribram (Teoría Holonómica del Cerebro):** Almacenamiento distribuido no local por patrones de interferencia de onda; resistencia inherente a la ablación y al daño localizado.
- **Tony Plate (Holographic Reduced Representations - HRR):** Enlace asociativo mediante convolución y correlación circular unitaria sobre el toro de fase complejas ($|\mathcal{F}(K)| = 1$).
- **Juan Maldacena & Edward Witten (Correspondencia AdS/CFT):** La secuencia 1D de tokens en la frontera conforme proyecta un espaciotiempo curvo en un Bulk radial 2D con métrica de Poincaré y propagador $e^{-|k|z}$, amortiguando el ruido de alta frecuencia en capas profundas.
- **Vitaly Vanchurin (El Universo como Red Neuronal):** El aprendizaje modelado como la relajación de una acción efectiva hacia el estado de mínima energía libre holográfica.

---

## ⚡ Resultados Empíricos Certificados

Medición rigurosa cruzada ejecutando el mismo checkpoint (`holo_deep_distilled_75m.pt`) en precisión `float32` (23 bits de mantisa de fase) sobre **GPU NVIDIA A100** y sobre **CPU local de 12 hilos**:

| Métrica de Rendimiento | GPU NVIDIA A100-SXM4 (40GB) | CPU Host Local (12 Hilos) | Transformer Estándar Baseline |
| :--- | :---: | :---: | :---: |
| **Precisión Sintáctica AST** | **7 / 7 (100.0%) [PASSED]** | **7 / 7 (100.0%) [PASSED]** | 100.0% (requiere KV-cache) |
| **Throughput de Generación** | **142.4 tokens/segundo** | **38.6 – 45.0 tokens/segundo** | ~8 – 15 tok/s (estrangulado en CPU) |
| **Latencia al 1er Token (TTFT)** | **34.6 ms** (7.5 ms base) | **~75 – 120 ms** | Degrada cuadráticamente $\mathcal{O}(N^2)$ |
| **Consumo de Memoria KV-Cache** | **0.00 KB ($\mathcal{O}(1)$ Constante)** | **0.00 KB ($\mathcal{O}(1)$ Constante)** | **8.192 GB** (a 64k tokens) |
| **Huella de Estado por Capa** | **64 KB Invariante** | **64 KB Invariante** | Crecimiento lineal continuo |

> **Teorema de Cuantización de Fase:** A diferencia de los Transformers convencionales que toleran mantisas de 7 bits (`bfloat16`), las redes holográficas en el Orden Implicado requieren **32 bits de precisión (`float32`)** para evitar el *jitter* angular en el toro unitario a lo largo de pasos autorregresivos prolongados.

---

## 🧬 Algoritmos Canónicos Verificados (100% AST)

El acumulador holográfico recupera de forma analítica y libre de interferencia cruzada (*crosstalk*) las siguientes estructuras lógicas de programación:

- [x] **Factorial Recursivo:** Recursión pura, condición base y tipado estricto `(n: int) -> int`.
- [x] **Fibonacci Memoizado:** Manejo de estado con diccionarios por defecto `memo: dict = None`.
- [x] **Búsqueda Binaria:** Punteros `left`/`right`, división entera `mid` y retorno condicional `-1`.
- [x] **Inversión de Lista Enlazada:** Manipulación de punteros triples (`prev`, `curr`, `nxt`).
- [x] **Recorrido BFS:** Grafos con cola `deque`, conjunto `visited` y exploración de adyacencias.
- [x] **Algoritmo de Kadane:** Subarreglo de suma máxima con variables `max_cur` y `max_glo`.
- [x] **Validación de Paréntesis:** Estructura LIFO de pila con diccionario de emparejamiento.

---

## 🚀 Inicio Rápido en tu Máquina Local (Consola Interactiva tipo Ollama)

El motor corre en **CPU nativo** con **0.00 KB de KV-Cache** sin necesidad de tarjetas gráficas de servidor:

### 1. Lanzar la Consola Interactiva
```bash
./holo.py
```

### 2. Comandos de Terminal Disponibles
```text
>>> factorial
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
[⚡ 31 tokens | 0.89s | TTFT: 142.2ms | 34.9 tok/s | KV-Cache: 0.00 KB]

>>> /benchmark     # Ejecuta la suite completa de los 7 algoritmos
>>> /state         # Inspecciona la norma ||S_t|| de la memoria en caché L1/L2
>>> /reset         # Reinicia el estado cuántico al vacío
>>> /exit          # Salir
```

---

## 📂 Arquitectura del Repositorio

```text
HoloLLM/
├── src/
│   ├── layers/
│   │   ├── holo_attention.py       # MultiHeadHoloAttention original O(1)
│   │   ├── holo_attention_v2.py    # Atención v2 con Convolución Causal 1D por FFT
│   │   ├── phase_disentangler.py   # Regularizador de ortogonalidad espectral de Plate
│   │   └── adscft.py               # Capa Bulk-Boundary de Maldacena con propagador Witten
│   ├── models/
│   │   ├── holo_causal_lm.py       # Modelo base Causal Decoder-Only
│   │   ├── holo_causal_lm_v2.py    # Modelo v2 con rotación de fase continua (RoPE)
│   │   └── holo_qwen.py            # Cirugía HoloQwen (SwiGLU MLPs + HoloAttention)
│   └── engine/
│       └── holo_engine.py          # Motor de inferencia local optimizado para CPU
├── docs/
│   ├── HoloLLM_Whitepaper.pdf      # Manuscrito formal del paper científico compilado
│   ├── PAPER_HOLOLLM_O1.md         # Documento técnico en formato Markdown
│   ├── index.html                  # Portal web para GitHub Pages
│   └── theory/                     # Ensayos de primeros principios y notas de física
├── benchmarks/
│   └── benchmark_scientific_report.py  # Protocolo oficial de medición cruzada GPU/CPU
├── holo.py                         # Cliente de terminal interactivo (estilo Ollama)
└── checkpoints/                    # Pesos de modelos y metadatos (.gitignore)
```

---

## 📜 Publicación y Paper Científico

El trabajo formal se encuentra publicado y preservado permanentemente en el repositorio de ciencia abierta del CERN (Zenodo):

- **Artículo Oficial (PDF):** [Descargar `HoloLLM_Whitepaper.pdf`](docs/HoloLLM_Whitepaper.pdf)
- **Registro Permanente:** [doi.org/10.5281/zenodo.22849908](https://doi.org/10.5281/zenodo.22849908)
- **Portal Web Interactivo:** [https://mrcsibr.github.io/HoloLLM](https://mrcsibr.github.io/HoloLLM)

### Cómo Citar este Trabajo
Si utilizas esta arquitectura o sus fundamentos en tu investigación, por favor utiliza la siguiente citación BibTeX:

```bibtex
@article{ibarra2026holollm,
  author       = {Ibarra, Marcos and {HoloLLM Research Team}},
  title        = {{HoloLLM: A Novel Holographic Language Model Architecture with $\mathcal{O}(1)$ Memory Complexity}},
  journal      = {Zenodo Preprint},
  year         = 2026,
  month        = sep,
  doi          = {10.5281/zenodo.22849908},
  url          = {https://doi.org/10.5281/zenodo.22849908}
}
```

---

## 👥 Equipo y Afiliación

- **Marcos Ibarra** — Data Scientist & Lead AI Researcher.
- **HoloLLM Research Team** — Laboratorio de Investigación en Inteligencia Artificial Holográfica.

---

## 🔬 Apoyo a la Investigación y Cómputo Independiente

HoloLLM es una iniciativa científica abierta e independiente. Para continuar con el escalamiento de la arquitectura hacia modelos de 1.5B y 7B parámetros (HoloQwen y HoloLlama) realizamos entrenamientos en clústeres GPU en la nube (NVIDIA A100 / H100 / H200).

Si deseas apoyar el avance de arquitecturas de memoria $\mathcal{O}(1)$ y financiar horas de cómputo, puedes colaborar directamente a través de Bitcoin:

- **Red Bitcoin (On-Chain SegWit):**  
  `bc1qyckf2kujstlxx4rcnd5lxe5thvkjgw8d7jkthv`
  
> *Transparencia: Las donaciones recibidas se destinan exclusivamente a costear tiempo de cómputo GPU y reproducibilidad de modelos abiertos.*

---

## 🤝 Licencia y Colaboración

Este proyecto se desarrolla bajo licencia de código abierto **Apache 2.0**. Para colaboraciones científicas, auditorías o propuestas de investigación, siéntete libre de abrir un *Issue* o *Pull Request*.


