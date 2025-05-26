import pandas as pd
df = pd.read_parquet("data/hadiths_with_embeddings.parquet")
print(len(df))  # сколько всего хадисов
print(df["embedding"].isnull().sum())  # сколько осталось без эмбеддинга