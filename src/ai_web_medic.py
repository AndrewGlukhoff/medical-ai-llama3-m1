import streamlit as st
from openai import OpenAI  # Synchronous for cleaner UI logic
import psutil
import yaml

# system prompt
with open("prompts.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

config = prompts["medical_expert"]

# 1. Page Config
st.set_page_config(page_title="TinyLlama Web", page_icon="🦙")

# 2. Connection to LM Studio (No heavy local models loaded in Python)
@st.cache_resource
def get_client():
    # Only the API client, no local neural networks in RAM
    return OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

client = get_client()

# 1. Настройка вкладки браузера
st.set_page_config(
    page_title="AI Медицинский Консультант", 
    page_icon="👨‍⚕️", 
    layout="wide"
)

# 2. Главный заголовок с иконкой доктора
st.title("👨‍⚕️ AI Медицинский Консультант")
st.subheader("Локальная модель: Llama-3 Medical (Fine-tuned)")

# 3. Добавим небольшую плашку со статусом (опционально)
st.info("⚠️ Внимание: Модель работает в справочном режиме. Для постановки диагноза обратитесь к врачу.")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

## Поле ввода (теперь на русском)
if prompt := st.chat_input("Опишите ваши симптомы..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Создаем пустой контейнер для "живого" текста
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            messages_for_llm = [
                {
                    "role": "system",
                    "content": config["system"]
                },
                {
                    "role": "user",
                    "content": config["user_template"].format(user_input=prompt)
                }
            ]

            # Запрос к LM Studio (указываем stream=True)
            response = client.chat.completions.create(
                model="local-model", 
                messages=messages_for_llm,
                temperature=0.4,
                top_p=0.5,
                presence_penalty=1.3,
                frequency_penalty=1,
                max_tokens=150,
                stream=True, # Магия живого чата
            )
            
            # Читаем поток токенов (букв)
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    # Обновляем текст на экране с эффектом печати
                    message_placeholder.markdown(full_response + "▌")
            
            # Финальный текст без курсора
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"Ошибка связи: {e}")
            full_response = "Не удалось связаться с моделью на M1."

    st.session_state.messages.append({"role": "assistant", "content": full_response})
