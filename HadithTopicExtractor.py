import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from collections import Counter

# === ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ===
MODEL_NAME = "asafaya/bert-base-arabic"  # Арабский BERT
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True)
model.eval()

def extract_important_tokens(text, top_k=10):
    # Токенизация
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Получаем attention: список из [num_layers] тензоров [batch, num_heads, seq_len, seq_len]
    attentions = outputs.attentions

    # Удалим CLS и SEP, если они есть
    token_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Среднее по слоям и головам
    attn_stack = torch.stack(attentions)  # [layers, batch, heads, seq_len, seq_len]
    attn_mean = attn_stack.mean(dim=(0, 1, 2))  # [seq_len, seq_len]

    # Считаем значимость токенов: сколько внимания они получают от других
    token_importance = attn_mean.sum(dim=0).numpy()  # [seq_len]

    # Собираем токены с их важностью
    token_score_pairs = [(tok, score) for tok, score in zip(tokens, token_importance)]

    # Фильтрация служебных токенов
    filtered = [(t, s) for t, s in token_score_pairs if t not in tokenizer.all_special_tokens and not t.startswith("##")]

    # Сортировка и выбор top_k
    top_tokens = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]
    return top_tokens

# === ПРИМЕР ===
text = "قال رسول الله صلى الله عليه وسلم: إنما الأعمال بالنيات وإنما لكل امرئ ما نوى"
keywords = extract_important_tokens(text, top_k=5)
print("🔑 Важные токены:")
for token, score in keywords:
    print(f"{token:>10}: {score:.4f}")