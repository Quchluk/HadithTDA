import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json

# === 1. Ввод: нужный кластер ===
target_node = "cube10_cluster0"

# === 2. Загрузка данных ===
df_clusters = pd.read_parquet("/Users/Tosha/Desktop/HadithTDA/hadithtda/data/mapper_cluster_report_clean.parquet")
df_hadiths = pd.read_parquet("/Users/Tosha/Desktop/HadithTDA/hadithtda/data/indexed_hadiths.parquet")

uids = []
for i in range(len(df_clusters)):
    if df_clusters.loc[i, "mapper_node"] == target_node:
        uid_summary = json.loads(df_clusters.loc[i, "uid_summary"])
        uids = list(uid_summary.keys())
        break

assert len(uids) > 0, f"No UIDs found for node {target_node}"
sample_uid = uids[0]
text_row = df_hadiths[df_hadiths["uid"] == sample_uid]
assert not text_row.empty, f"UID {sample_uid} not found in hadiths"
text = text_row.iloc[0]["text_ar"]

# === 3. Загрузка модели ===
model_name = "asafaya/bert-base-arabic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# === 4. Токенизация ===
inputs = tokenizer(text, return_tensors="pt", truncation=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# === 5. Получение эмбеддингов и включение градиентов ===
embeddings = model.embeddings(input_ids)
embeddings.retain_grad()
embeddings.requires_grad_()

# === 6. Прогон через модель ===
outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] токен
score = cls_output.norm()
score.backward()

# === 7. Получение важности токенов ===
grads = embeddings.grad[0].abs().sum(dim=1)  # важность: сумма градиентов по каждому токену
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
token_scores = list(zip(tokens, grads.tolist()))

# === 8. Сортировка и вывод ===
token_scores = sorted(token_scores, key=lambda x: x[1], reverse=True)
print(f"Top-10 ключевых токенов из {target_node} (uid: {sample_uid}):\n")
for token, score in token_scores[:10]:
    print(f"{token}\t{score:.4f}")