import os
import yaml


# pathes
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "medical_llama_ELITE")
PROMPTS_PATH = os.path.join(BASE_DIR, "prompts.yaml")

def get_system_prompt():
    """Достает системную роль из вложенной структуры YAML."""
    try:
        with open(PROMPTS_PATH, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            # Достаем по цепочке: medical_expert -> system
            return data.get("medical_expert", {}).get("system", "").strip()
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return ""

def format_prompt(user_question: str) -> str:
    """Полная сборка промпта Llama-3."""
    system_text = get_system_prompt()
    
    prompt = "<|begin_of_text|>"
    # 1. Системный блок (если есть)
    if system_text:
        prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_text}<|eot_id|>"
    
    # 2. Блок пользователя
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_question}<|eot_id|>"
    
    # 3. Открываем блок ассистента
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prompt

# Параметры генерации (твои проверенные настройки "Анти-Попугай")
GEN_SETTINGS = {
    "temp": 0.4,
    "repetition_penalty": 1.8,
    "max_tokens": 150,
    "top_p": 0.5
}
