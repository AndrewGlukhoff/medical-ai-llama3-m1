from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# Загружаем модель и твои обученные адаптеры
model, tokenizer = load(
    "mlx-community/Meta-Llama-3-8B-Instruct-4bit", 
    adapter_path="./adapters"
)

question = "Доктор, сильно болит живот и тошнит, что это может быть?" #"У меня болит в области сердца и колет под лопаткой. Что это может быть?"
# Формируем промпт в формате Llama-3
prompt = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    f"{question}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

# задать temp теперь нужен sampler
sampler = make_sampler(
    temp=0.8, 
    top_p=0.5, # 1 default
)
# punishes model for repeating the same tokens (аналог presence_penalty), в MLX нет отдельного frequency_penalty
logits_processors = make_logits_processors(repetition_penalty=2.0)

response = generate(
    model, 
    tokenizer, 
    prompt=prompt, 
    max_tokens=400,
    sampler=sampler,
    logits_processors=logits_processors,
)

clean_answer = response.split("<|eot_id|>")[0].strip()


print(f"Q: {question}")
print("\n--- ОТВЕТ ЭЛИТНОГО ВРАЧА (чекпойнт 400) ---")
print(clean_answer)
print("------------------------------")
