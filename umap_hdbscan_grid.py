import pandas as pd
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import os
from datetime import datetime

path_input = "/Users/Tosha/Desktop/HadithTDA/hadithtda/data/hadiths_with_embeddings_and_metadata_2.parquet"
output_dir = "/Users/Tosha/Desktop/HadithTDA/hadithtda/cluster_search"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_parquet(path_input)
embeddings = np.vstack(df["embedding"].values)

# Параметры для перебора
n_components_list = [10, 30, 50]
n_neighbors_list = [10, 30]
min_dist_list = [0.01, 0.1]
min_cluster_sizes = [5, 20]

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

for nc in n_components_list:
    for nn in n_neighbors_list:
        for md in min_dist_list:
            for mcs in min_cluster_sizes:
                name = f"nc{nc}_nn{nn}_md{md}_mcs{mcs}"
                print(f"[→] Running: {name}")

                umap_model = umap.UMAP(n_components=nc, metric="cosine", n_neighbors=nn, min_dist=md, random_state=42)
                reduced = umap_model.fit_transform(embeddings)

                clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean")
                labels = clusterer.fit_predict(reduced)

                df_result = df.copy()
                df_result["umap_cluster"] = labels

                # Сохранение parquet
                df_result.to_parquet(os.path.join(output_dir, f"{name}_{run_id}.parquet"), index=False)

                # Визуализация
                umap_2d = umap.UMAP(n_components=2, metric="cosine", random_state=42).fit_transform(embeddings)
                df_result["x"] = umap_2d[:, 0]
                df_result["y"] = umap_2d[:, 1]

                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(df_result["x"], df_result["y"], c=labels, cmap="tab10", s=10)
                plt.colorbar(scatter, label="Cluster ID")
                plt.title(name)
                plt.xlabel("UMAP-1")
                plt.ylabel("UMAP-2")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_{run_id}.png"), dpi=300)
                plt.close()