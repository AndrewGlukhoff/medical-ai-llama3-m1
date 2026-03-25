# первоначальный вариант (ошибочный) - окончательный вариант см/ analysis notebook 
from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)
DATASET_ID = "blinoff/medical_qa_ru_data"

dataset = load_dataset(DATASET_ID, split='train', trust_remote_code=True)

# Обновленный список на основе твоего Pandas output
target_categories = [
    'Терапия', 'Гинекология', 'Неврология', 'Гастроэнтеролог', 
    'Педиатрия', 'Кардиология', 'Эндокринология'
]

final_data = []

for item in dataset:
    if item['categ'] in target_categories:
        question = item['desc'].strip()
        
        # Сначала делим по точке с запятой, получаем список
        ans_parts = item['ans'].split(';')
        # Берем только первый элемент списка и убираем пробелы - ИЗ-ЗА этого получили медиану 11 !!!
        answer = ans_parts[0].strip() if ans_parts else ""
        
        if len(question) > 30 and len(answer) > 30:
            entry = {
                "text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
            }
            final_data.append(entry)
                
        if len(final_data) >= 5000:
            break

# Запись файлов
with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for entry in final_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Валидация (возьмем последние 100 строк из выборки)
with open("data/valid.jsonl", "w", encoding="utf-8") as f:
    for entry in final_data[-100:]:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Успех! Создано {len(final_data)} строк в data/train.jsonl")
