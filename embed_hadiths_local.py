import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import time


# === 1. Настройки ===
INPUT_PARQUET = "data/indexed_hadiths.parquet"
SAVE_PARQUET = "data/hadiths_with_embeddings_local.parquet"
MODEL_NAME = "intfloat/multilingual-e5-large"
TEXT_COLUMN = "text_ar"
ID_COLUMN = "uid"
CHUNK_SIZE = 1

# === 2. Загрузка всех данных и проверка UID ===
df_all = pd.read_parquet(INPUT_PARQUET)

# === 2.1. Проверка столбцов ===
required_cols = {ID_COLUMN, TEXT_COLUMN}
missing_cols = required_cols - set(df_all.columns)
if missing_cols:
    raise ValueError(f"[✘] Missing required columns in input file: {missing_cols}")

df_all = df_all[df_all[TEXT_COLUMN].notnull()].reset_index(drop=True)

#time to load a model
print(f"[⏳] Loading model: {MODEL_NAME}")
start_time = time.time()

model = SentenceTransformer(MODEL_NAME, device='cuda' if torch.cuda.is_available() else 'cpu')

elapsed = time.time() - start_time
print(f"[✓] Model loaded in {elapsed:.2f} seconds.")

# === 3. Загрузка всех данных ===
df_all = pd.read_parquet(INPUT_PARQUET)
df_all = df_all[df_all[TEXT_COLUMN].notnull()].reset_index(drop=True)

# === 4. Если уже есть результат — продолжим оттуда ===
if os.path.exists(SAVE_PARQUET):
    df_done = pd.read_parquet(SAVE_PARQUET)
    done_ids = set(df_done[ID_COLUMN])
    df_all = df_all[~df_all[ID_COLUMN].isin(done_ids)]
    print(f"[!] Continuing. Already embedded: {len(done_ids)}")
    df_all = pd.concat([df_done, df_all], ignore_index=True)

# === 5. Подготовка чанков ===
chunks = [df_all.iloc[i:i + CHUNK_SIZE] for i in range(0, len(df_all), CHUNK_SIZE)]

# === 6. Эмбеддинг и сохранение ===
results = []
for chunk in tqdm(chunks, desc="Embedding..."):
    texts = ["query: " + str(text) for text in chunk[TEXT_COLUMN].tolist()]
    try:
        embs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        df_emb = chunk[[ID_COLUMN]].copy()
        df_emb["embedding"] = [emb.tolist() for emb in embs]
        results.append(df_emb)

        # Сохраняем на каждом шаге
        pd.concat(results).to_parquet(SAVE_PARQUET, index=False)
    except Exception as e:
        print(f"[!] Error in chunk: {e}")
        continue

print("[✓] Done.")