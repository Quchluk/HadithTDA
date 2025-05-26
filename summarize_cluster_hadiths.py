import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import json

# === Настройки ===
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
target_node = "cube15_cluster0"

# === Загрузка данных ===
df_clusters = pd.read_parquet("/Users/Tosha/Desktop/HadithTDA/hadithtda/data/mapper_cluster_report_clean.parquet")
df_hadiths = pd.read_parquet("/Users/Tosha/Desktop/HadithTDA/hadithtda/data/indexed_hadiths.parquet")

# === Получаем список UID из нужного кластера ===
uid_summary = json.loads(
    df_clusters[df_clusters["mapper_node"] == target_node]["uid_summary"].values[0]
)
uids = list(uid_summary.keys())

# === Собираем английские тексты по UID ===
texts = df_hadiths[df_hadiths["uid"].isin(uids)]["text_en"].dropna().tolist()
text_combined = " ".join(texts)

# === Токенизация и обрезка до 1024 токенов ===
inputs = tokenizer(text_combined, return_tensors="pt", truncation=True, max_length=1024)

# === Генерация summary ===
with torch.no_grad():
    summary_ids = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# === Вывод ===
print(f"📍 Cluster: {target_node}")
print(f"🧠 Summary:\n{summary}")