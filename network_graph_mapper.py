import pandas as pd
import json
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from matplotlib import cm

# === Загрузка отчета ===
df = pd.read_parquet("/Users/Tosha/Desktop/HadithTDA/hadithtda/data/mapper_cluster_report.parquet")
df = df[df["mapper_node"] != "ALL"]

# === Построим словарь: mapper_node → set(uid) ===
node_uids = {}
for _, row in df.iterrows():
    uid_summary = json.loads(row["uid_summary"])
    uid_set = set(uid_summary.keys())
    node_uids[row["mapper_node"]] = uid_set

# === Строим граф ===
G = nx.Graph()
for node1, uids1 in node_uids.items():
    G.add_node(node1, size=len(uids1))
    for node2, uids2 in node_uids.items():
        if node1 >= node2:
            continue
        inter = uids1 & uids2
        if inter:
            G.add_edge(node1, node2, weight=len(inter))

# === Расстояния ===
dist_matrix = nx.floyd_warshall_numpy(G)
finite_max = np.nanmax(dist_matrix[dist_matrix != np.inf])
dist_matrix[dist_matrix == np.inf] = finite_max + 1

# === UMAP + нормализация (с расширением кластеров) ===
umap_model = UMAP(n_components=3, metric="precomputed", spread=5.0, min_dist=0.05, random_state=42)
pos_3d = umap_model.fit_transform(dist_matrix)
pos_3d = StandardScaler().fit_transform(pos_3d)
positions = {node: pos_3d[i] for i, node in enumerate(G.nodes)}

# === Подготовка цвета по размеру ===
sizes = np.array([G.nodes[n]["size"] for n in G.nodes()])
norm_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-8)

# Правильное преобразование в строки 'rgb(r, g, b)'
colors = [
    f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    for r, g, b, _ in cm.viridis(norm_sizes)
]

# === Визуализация ===
edge_x, edge_y, edge_z = [], [], []
for u, v in G.edges():
    x0, y0, z0 = positions[u]
    x1, y1, z1 = positions[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

node_x = [positions[n][0] for n in G.nodes()]
node_y = [positions[n][1] for n in G.nodes()]
node_z = [positions[n][2] for n in G.nodes()]
node_text = list(G.nodes())

fig = go.Figure()

# Рёбра
fig.add_trace(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='lightgray', width=1),
    hoverinfo='none'
))

# Узлы
fig.add_trace(go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers+text',
    marker=dict(
        size=[min(max(G.nodes[n]['size'] / 50, 4), 30) for n in G.nodes()],
        color=colors,
        opacity=0.85
    ),
    text=node_text,
    textposition="top center",
    hoverinfo='text'
))

# Настройки
fig.update_layout(
    title="3D Mapper Graph (UMAP, Spread=5.0, Colored by Cluster Size)",
    paper_bgcolor='white',
    plot_bgcolor='white',
    showlegend=False,
    scene=dict(
        xaxis=dict(showbackground=False, visible=False),
        yaxis=dict(showbackground=False, visible=False),
        zaxis=dict(showbackground=False, visible=False)
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)

fig.show()