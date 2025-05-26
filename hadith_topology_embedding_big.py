import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken

# === 1. Load API key ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 2. Tokenizer
tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")

def num_tokens(text):
    return len(tokenizer.encode(text))

def chunk_text(text, max_tokens=8000):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# === 3. Load full dataset
df_full = pd.read_parquet("data/indexed_hadiths.parquet")
df_full = df_full[df_full["text_ar"].notnull()].reset_index(drop=True)

# === 4. Try to load existing embeddings (if any)
save_path = "data/hadiths_with_embeddings.parquet"
if os.path.exists(save_path):
    df_embedded = pd.read_parquet(save_path)
    # Map existing embeddings by hadith_id
    embedded_dict = dict(zip(df_embedded["hadith_id"], df_embedded["embedding"]))
    df_full["embedding"] = df_full["hadith_id"].map(embedded_dict)
else:
    df_full["embedding"] = None

# === 5. Embedding function with full chunk support
def embed(text, retries=3):
    try:
        if num_tokens(text) <= 8000:
            for attempt in range(retries):
                try:
                    response = client.embeddings.create(input=text, model="text-embedding-3-large")
                    return response.data[0].embedding
                except Exception as e:
                    print(f"Error: {e}. Retry {attempt+1}/{retries}")
                    time.sleep(5)
            return None
        else:
            chunks = chunk_text(text, max_tokens=8000)
            embeddings = []
            for i, chunk in enumerate(chunks):
                for attempt in range(retries):
                    try:
                        response = client.embeddings.create(input=chunk, model="text-embedding-3-large")
                        embeddings.append(response.data[0].embedding)
                        break
                    except Exception as e:
                        print(f"Chunk {i+1}/{len(chunks)} error: {e}. Retry {attempt+1}/{retries}")
                        time.sleep(5)
                else:
                    return None  # Failed all retries for this chunk
            return np.mean(embeddings, axis=0).tolist()
    except Exception as e:
        print(f"Final failure: {e}")
        return None

# === 6. Embedding loop with save support
tqdm.pandas()
for idx, row in tqdm(df_full.iterrows(), total=len(df_full)):
    if row["embedding"] is None:
        emb = embed(row["text_ar"])
        if emb is not None:
            df_full.at[idx, "embedding"] = emb

    if idx % 100 == 0:
        df_full.to_parquet(save_path, index=False)

# === 7. Final save
df_full.to_parquet(save_path, index=False)