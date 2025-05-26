import pandas as pd
import numpy as np
import kmapper as km
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import os
from datetime import datetime

# === 1. Пути ===
path_input = "/Users/Tosha/Desktop/HadithTDA/hadithtda/data/hadiths_with_embeddings_and_metadata_2.parquet"
output_dir = "/Users/Tosha/Desktop/HadithTDA/hadithtda/mapper_output"
os.makedirs(output_dir, exist_ok=True)

# === 2. Загрузка эмбеддингов и метаданных ===
df = pd.read_parquet(path_input)
embeddings = np.vstack(df["embedding"].to_numpy())

# === 3. Построение Mapper-графа ===
mapper = km.KeplerMapper(verbose=1)
lens = PCA(n_components=2).fit_transform(embeddings)

graph = mapper.map(
    lens,
    embeddings,
    cover=km.Cover(n_cubes=10, perc_overlap=0.5),
    clusterer=DBSCAN(eps=0.5, min_samples=5)
)

# === 4. Привязка mapper_node к строкам датафрейма ===
node_assignments = []
for node_id, indices in graph["nodes"].items():
    for idx in indices:
        node_assignments.append((idx, node_id))

df_nodes = pd.DataFrame(node_assignments, columns=["original_index", "mapper_node"])
df_nodes["original_index"] = df_nodes["original_index"].astype(int)

# === 5. Объединение с метаданными ===
df_with_index = df.reset_index(drop=False)  # сохранить index как колонку
df_merged = df_nodes.merge(df_with_index, left_on="original_index", right_on="index", how="left")
df_merged = df_merged.drop(columns=["index"])

# === 6. Сохранение итогового parquet ===
nodes_path = os.path.join(output_dir, "hadiths_with_mapper_nodes.parquet")
df_merged.to_parquet(nodes_path, index=False)
print(f"[✓] Saved mapper node assignments with metadata: {nodes_path}")

# === 7. Визуализация Mapper-графа ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
graph_path = os.path.join(output_dir, f"mapper_graph_{timestamp}.html")
mapper.visualize(graph, path_html=graph_path, title="Mapper graph of Hadith Embeddings")
print(f"[✓] Saved Mapper visualization: {graph_path}")