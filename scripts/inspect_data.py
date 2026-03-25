from datasets import load_dataset
import pandas as pd

print("📥 Загрузка датасета в Pandas...")
# Загружаем только первые 1000 строк для быстрого анализа
dataset = load_dataset("blinoff/medical_qa_ru_data", split='train', trust_remote_code=True)
df = pd.DataFrame(dataset.select(range(5000))) 

# 1. Смотрим на колонки и типы данных
print("\n--- Структура данных ---")
print(df.info())

# 2. Смотрим на самые популярные категории (categ)
print("\n--- Топ-15 категорий (categ) ---")
print(df['categ'].value_counts().head(15))

# 3. Смотрим на примеры тем (theme)
print("\n--- Примеры тем (theme) ---")
print(df['theme'].head(10))

# 4. Выведем одну полную строку, чтобы прочитать вопрос и ответ
print("\n--- Пример полной записи ---")
sample = df.iloc[0]
print(f"ВОПРОС: {sample['desc'][:200]}...")
print(f"ОТВЕТ (первый сегмент): {sample['ans'].split(';')[0]}...")

# Сохраним первые 100 строк в CSV, чтобы ты мог открыть его в Excel/Numbers
df.head(100).to_csv("preview_med.csv", index=False, encoding='utf-8-sig')
print("\n✅ Файл 'preview_med.csv' создан. Открой его, чтобы посмотреть глазами.")
