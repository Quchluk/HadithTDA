import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt

path_input = "/Users/Tosha/Desktop/HadithTDA/hadithtda/data/hadiths_clustered_by_umap.parquet"
df = pd.read_parquet(path_input)
embeddings = np.vstack(df["embedding"].values)

umap_model = umap.UMAP(n_components=10, metric="cosine", random_state=42)
umap_2d = umap_model.fit_transform(embeddings)

df["x"] = umap_2d[:, 0]
df["y"] = umap_2d[:, 1]

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df["x"], df["y"], c=df["umap_cluster"], cmap="tab10", s=10)
plt.colorbar(scatter, label="Cluster ID")
plt.title("UMAP projection of Hadith Embeddings")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()