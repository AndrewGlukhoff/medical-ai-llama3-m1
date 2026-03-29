import streamlit as st
import psutil
from mlx_lm import load, stream_generate
from config import MODEL_PATH, GEN_SETTINGS, format_prompt
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# Page Config
st.set_page_config(
    page_title="AI Медицинский Консультант", 
    page_icon="👨‍⚕️", 
    layout="wide"
)

# Загрузка модели (кешируем, чтобы грузить первый раз)
@st.cache_resource
def load_medical_model():
    # Модель загружается напрямую в память GPU M1
    model, tokenizer = load(MODEL_PATH)
    return model, tokenizer

model, tokenizer = load_medical_model()

# interface
st.title("👨‍⚕️ AI Медицинский Консультант")
st.subheader("Локальная модель: Llama-3 Medical")

# Плашка с системными ресурсами M1
with st.sidebar:
    st.header("💻 System Monitor")
    ram = psutil.virtual_memory()
    st.metric("RAM Used", f"{ram.used / (1024**3):.1f} GB", f"{ram.percent}%")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.info("⚠️ Внимание: Модель работает в справочном режиме. Не является диагнозом.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

## Поле ввода
if user_input := st.chat_input("Опишите ваши симптомы..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Создаем пустой контейнер для "живого" текста
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            prompt = format_prompt(user_input)
            sampler = make_sampler(
                temp=GEN_SETTINGS.get("temp"), 
                top_p=GEN_SETTINGS.get("top_p", 1.0)
            )
            logits_processors = make_logits_processors(repetition_penalty=GEN_SETTINGS.get("repetition_penalty"))
            for response in stream_generate(
                model, 
                tokenizer, 
                prompt=prompt,
                sampler=sampler,
                logits_processors=logits_processors,
                max_tokens=GEN_SETTINGS.get("max_tokens")
            ):
                full_response += response.text
                # Обрезаем стоп-теги, если модель их выплевывает
                clean_text = full_response.split("<|eot_id|>")[0].strip()
                message_placeholder.markdown(clean_text + "▌")

            
            # Финальный текст 
            final_text = full_response.split("<|eot_id|>")[0].strip()
            message_placeholder.markdown(final_text)
            st.session_state.messages.append({"role": "assistant", "content": final_text})
            
            
        except Exception as e:
            st.error(f"Ошибка связи: {e}")

