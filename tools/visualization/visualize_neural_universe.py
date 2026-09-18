r"""
SCRIPT: visualize_neural_universe.py
PROYECTO: HoloLLM (MRCSIBR/HoloLLM)
DESCRIPCIÓN: Renderiza la topología del "Universo Neuronal" de HoloLLM.
"""

import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURACIÓN
# ==========================================
CKPT_PATH = "checkpoints/holo_deep_distilled_75m.pt"
TOP_PERCENTILE = 98.5  # Solo mostramos el 1.5% de las conexiones más fuertes

def main():
    print("=" * 80)
    print("🔭 OBSERVATORIO HOLOLLM: MAPEANDO EL UNIVERSO NEURONAL")
    print("=" * 80)

    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint no encontrado en {CKPT_PATH}")
        return

    print("[1/4] Cargando el espaciotiempo de los pesos...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    # Auto-detectar una capa profunda válida (compatible con v1, v2 o Qwen)
    target_layer = None
    candidates = [
        "blocks.3.mlp.2.weight",          # HoloLLM 70M / 350M
        "blocks.3.mlp.down_proj.weight",  # Qwen transmutado
        "layers.3.mlp.down_proj.weight",
        "blocks.3.attn.out_proj.weight"
    ]
    for c in candidates:
        if c in state:
            target_layer = c
            break
            
    if not target_layer:
        print("Error: No se encontró una capa candidata en el state_dict. Capas disponibles:")
        print(list(state.keys())[:10])
        return

    print(f"✔ Capa detectada: {target_layer}")
    W = state[target_layer].float()
    
    print("[2/4] Calculando la matriz de correlación gravitacional...")
    adjacency = torch.abs(W @ W.T)
    adjacency.fill_diagonal_(0)
    adj_np = adjacency.numpy()

    print(f"[3/4] Filtrando el polvo cósmico (Mapeando el top {100-TOP_PERCENTILE:.1f}% de filamentos)...")
    threshold = np.percentile(adj_np, TOP_PERCENTILE)
    
    G = nx.Graph()
    num_nodes = adj_np.shape[0]
    G.add_nodes_from(range(num_nodes))

    edges = []
    weights = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_np[i, j] > threshold:
                G.add_edge(i, j, weight=adj_np[i, j])
                edges.append((i, j))
                weights.append(adj_np[i, j])

    G.remove_nodes_from(list(nx.isolates(G)))

    print(f"      -> {G.number_of_nodes()} nodos (Galaxias/Hubs)")
    print(f"      -> {G.number_of_edges()} conexiones (Filamentos)")

    print("[4/4] Simulando física de N-cuerpos (Layout dirigido por fuerzas)...")
    pos = nx.spring_layout(G, k=0.15, iterations=50, weight='weight', seed=42)

    print("      Generando visualización fotográfica...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='black')
    ax.set_facecolor('black')

    degrees = dict(G.degree(weight='weight'))
    node_sizes = [v * 50 for v in degrees.values()]
    node_colors = [v for v in degrees.values()]

    weights_norm = [w / max(weights) for w in weights]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        width=[w * 2 for w in weights_norm],
        alpha=0.4,
        edge_color=weights_norm,
        edge_cmap=plt.cm.magma,
        edge_vmin=0, edge_vmax=1,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.plasma,
        alpha=0.9,
        edgecolors='white',
        linewidths=0.5,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos,
        node_size=[s * 3 for s in node_sizes],
        node_color=node_colors,
        cmap=plt.cm.plasma,
        alpha=0.1,
        ax=ax
    )

    ax.set_title("La Red Cósmica de HoloLLM-70M\n(Topología de Correlación Semántica en el Orden Implicado)", 
                 fontsize=18, color='white', pad=20, fontname='serif')
    
    plt.axis('off')
    
    out_file = "tools/visualization/holo_neural_universe.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, facecolor='black', bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 80)
    print(f"IMAGEN RENDERIZADA: Abre el archivo '{out_file}'")
    print("=" * 80)

if __name__ == "__main__":
    main()
