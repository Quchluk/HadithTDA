import pandas as pd
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import os
from datetime import datetime

path_input = "/Users/Tosha/Desktop/HadithTDA/hadithtda/data/hadiths_with_embeddings_and_metadata.parquet"
path_output = "/Users/Tosha/Desktop/HadithTDA/hadithtda/data/hadiths_clustered_by_umap.parquet"
plot_dir = "/Users/Tosha/Desktop/HadithTDA/hadithtda/plots"

df = pd.read_parquet(path_input)
embeddings = np.vstack(df["embedding"].values)

umap_model = umap.UMAP(n_components=50, metric="cosine", random_state=42)
reduced = umap_model.fit_transform(embeddings)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean")
labels = clusterer.fit_predict(reduced)

df["umap_cluster"] = labels
df.to_parquet(path_output, index=False)

# Визуализация в 2D и сохранение
umap_2d = umap.UMAP(n_components=2, metric="cosine", random_state=42).fit_transform(embeddings)
df["x"] = umap_2d[:, 0]
df["y"] = umap_2d[:, 1]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = os.path.join(plot_dir, f"umap_clusters_{timestamp}.png")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df["x"], df["y"], c=df["umap_cluster"], cmap="tab10", s=10)
plt.colorbar(scatter, label="Cluster ID")
plt.title("UMAP 2D projection of Hadith Embeddings")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()