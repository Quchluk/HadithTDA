import pandas as pd
import json

# Загружаем объединённый файл с mapper_node и метаданными
df = pd.read_parquet("/Users/Tosha/Desktop/HadithTDA/hadithtda/mapper_output/hadiths_with_mapper_nodes.parquet")

# Метаданные без текстов
text_cols = {"text_ar", "text_en", "narrator_en", "embedding", "embedding.list"}
metadata_columns = [col for col in df.columns if col not in text_cols and col not in ("original_index", "mapper_node")]

# === Группировка по mapper_node ===
report = []
grouped = df.groupby("mapper_node")

for node_id, group in grouped:
    row = {"mapper_node": node_id, "total_hadiths": len(group)}
    for col in metadata_columns:
        value_counts = group[col].value_counts().to_dict()
        value_counts_str_keys = {str(k): v for k, v in value_counts.items()}
        row[f"{col}_summary"] = json.dumps(value_counts_str_keys, ensure_ascii=False)
    report.append(row)

# === Общая строка ===
row_all = {"mapper_node": "ALL", "total_hadiths": len(df)}
for col in metadata_columns:
    value_counts = df[col].value_counts().to_dict()
    value_counts_str_keys = {str(k): v for k, v in value_counts.items()}
    row_all[f"{col}_summary"] = json.dumps(value_counts_str_keys, ensure_ascii=False)
report.append(row_all)

# === Сохранение ===
report_df = pd.DataFrame(report)
report_df.to_parquet("/Users/Tosha/Desktop/HadithTDA/hadithtda/data/mapper_cluster_report_clean.parquet", index=False)
report_df.to_json("/Users/Tosha/Desktop/HadithTDA/hadithtda/data/mapper_cluster_report_clean.json", orient="records", indent=2, force_ascii=False)