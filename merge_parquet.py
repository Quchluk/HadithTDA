import pandas as pd

path_embeddings = "/Users/Tosha/Desktop/HadithTDA/hadithtda/data/hadiths_with_embeddings_local.parquet"
path_metadata = "/Users/Tosha/Desktop/HadithTDA/hadithtda/data/indexed_hadiths.parquet"
path_output = "/Users/Tosha/Desktop/HadithTDA/hadithtda/data/hadiths_with_embeddings_and_metadata.parquet"

df_embeddings = pd.read_parquet(path_embeddings)
df_metadata = pd.read_parquet(path_metadata)

df_merged = pd.merge(df_metadata, df_embeddings, on="uid", how="inner")
df_merged.to_parquet(path_output, index=False)