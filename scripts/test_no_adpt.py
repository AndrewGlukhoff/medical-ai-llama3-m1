from mlx_lm import load, generate

# Загружаем модель как она есть до обучения (т.е. без адаптеров)
model, tokenizer = load(
    "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
)

# Формируем промпт в формате Llama-3
prompt = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "У меня болит в области сердца и колет под лопаткой. Что это может быть?"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

# Оставляем только базовые параметры, которые 100% поддерживаются
# Если модель будет "бредить", попробуй добавить temp=0.7 (но только если не вылетит ошибка)
response = generate(
    model, 
    tokenizer, 
    prompt=prompt, 
    max_tokens=200
)

# Трюк: обрезаем всё, что идет после первого тега окончания или первого восклицательного знака
# Это уберет бесконечные "!", которые мы видели в терминале
clean_answer = response.split("<|eot_id|>")[0].split("!")[0].strip()

print("\n--- ОТВЕТ МОДЕЛИ БЕЗ АДАПТЕРОВ ---")
print(clean_answer)
print("------------------------------")
