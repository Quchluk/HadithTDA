import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from ripser import ripser
from persim import plot_diagrams
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

# === 1. Load embedded dataframe ===
df = pd.read_parquet("/Users/Tosha/Desktop/HadithTDA/hadithtda/data/hadiths_with_embeddings_and_metadata.parquet")
embeddings = np.vstack(df["embedding"].to_numpy())

# === 2. Compute persistent homology ===
print("Computing persistent homology...")
diagrams = ripser(embeddings, maxdim=1)['dgms']

# Optional: save diagram
with open("persistence_diagram.pkl", "wb") as f:
    pickle.dump(diagrams, f)

# === 3. Extract birth/death per point (H0) ===
h0 = diagrams[0]
clusters_info = []

for idx, (birth, death) in enumerate(h0):
    clusters_info.append({
        "dim": 0,
        "cluster_id": idx,
        "birth": birth,
        "death": death,
        "lifetime": death - birth if death < np.inf else np.inf
    })

# === 4. Convert to dataframe and sort ===
clusters_df = pd.DataFrame(clusters_info)
clusters_df.sort_values(by="lifetime", ascending=False, inplace=True)
clusters_df.reset_index(drop=True, inplace=True)

# === 5. Save clusters dataframe ===
clusters_df.to_parquet("hadith_clusters_h0.parquet", index=False)
print("Saved: hadith_clusters_h0.parquet")