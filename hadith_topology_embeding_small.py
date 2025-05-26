# hadith_topology_step2_embed_small.py

import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# === 1. Load API key ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 2. Load full dataset ===
df = pd.read_parquet("data/indexed_hadiths_2.parquet")
df = df[df["text_ar"].notnull()].reset_index(drop=True)
df["embedding"] = None

# === 3. Try to load previous embeddings (if resuming) ===
save_path = "data/hadiths_with_embeddings_small.parquet"
if os.path.exists(save_path):
    df_prev = pd.read_parquet(save_path)
    embedded_ids = set(df_prev["hadith_id"])
    df = pd.concat([df_prev, df[~df["hadith_id"].isin(embedded_ids)]], ignore_index=True)

# === 4. Embedding function (NEW API) — smaller model ===
def embed(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# === 5. Embed missing Arabic texts with periodic saving ===
tqdm.pandas()
save_every = 50  # ← Save every N embeddings
for i, row in tqdm(df.iterrows(), total=len(df)):
    if df.at[i, "embedding"] is None:
        df.at[i, "embedding"] = embed(row["text_ar"])

    if i % save_every == 0:
        df.to_parquet(save_path, index=False)

# === 6. Final save ===
df.to_parquet(save_path, index=False)