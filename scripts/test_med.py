from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))
from config import format_prompt, MODEL_PATH, GEN_SETTINGS

# Загружаем модель и твои обученные адаптеры
model, tokenizer = load(
    "mlx-community/Meta-Llama-3-8B-Instruct-4bit", 
    adapter_path="./adapters"
)

question = "Доктор, сильно болит живот и тошнит, что это может быть?" #"У меня болит в области сердца и колет под лопаткой. Что это может быть?"
# Формируем промпт в формате Llama-3
prompt = format_prompt(question)

# задать temp теперь нужен sampler
sampler = make_sampler(
    temp=GEN_SETTINGS.get("temp"), 
    top_p=GEN_SETTINGS.get("top_p", 1.0), # 1 default
)
# punishes model for repeating the same tokens (аналог presence_penalty), в MLX нет отдельного frequency_penalty
logits_processors = make_logits_processors(repetition_penalty=GEN_SETTINGS.get("repetition_penalty"))

response = generate(
    model, 
    tokenizer, 
    prompt=prompt, 
    max_tokens=GEN_SETTINGS.get("max_tokens"),
    sampler=sampler,
    logits_processors=logits_processors,
)

clean_answer = response.split("<|eot_id|>")[0].strip()


print(f"Q: {question}")
print("\n--- ОТВЕТ ЭЛИТНОГО ВРАЧА (чекпойнт 400) ---")
print(clean_answer)
print("------------------------------")
