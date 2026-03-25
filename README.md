## Medical AI Assistant (Llama-3 Elite v2) 🩺
DISCLAIMER: _"Этот проект создан в образовательных целях. Ответы ИИ не являются медицинской рекомендацией."_


<img src="/Users/andrewglukhoff/code/chatbot1/assets/chat-preview.png" width="666">


Локальный медицинский ассистент на базе **Llama-3-8B**, дообученный (Fine-tuned) на специализированном медицинском датасете с использованием **MLX** и **LoRA** на Apple Silicon (M1).

#### Ключевые возможности

- **Fine-tuned Engine**: Модель обучена на 10,000+ высококачественных медицинских диалогах (Elite Dataset).
- **Multi-interface**: Доступ через **Telegram-бот** (aiogram 3) и **Web-интерфейс**(Streamlit).
- **Apple Silicon Optimized**: Полная поддержка GPU-ускорения Metal через библиотеку MLX.
- **Remote Access**: Возможность работы через SSH и туннели (Pinggy/Ngrok).

#### Структура проекта

- `src/`: Исходный код бота и веб-приложения.
- `notebooks/`: Исследования данных и визуализация датасета (Pandas).
- `data/`: Обучающие выборки (Elite v2).
- `models/`: Fused модель, готовая к работе.
- `adapters/`: Чекпоинты LoRA (используется 600-я итерация).

#### Установка и запуск

1. **Окружение**:
```
conda create -n hollm_chat python=3.11
conda activate hollm_chat
pip install -r requirements.txt
```
2. **Настройка**: Создайте файл `chat_bot.env` на основе `chat_bot.env.example` и добавьте свой Telegram токен.
3. **Запуск**:
    - Telegram: `python src/bot6.py`
    - Web: `streamlit run src/ai_web_medic.py`

4. Параметры генерации (Anti-Parrot)

Для стабильной работы "Elite" модели используются:

- `repetition_penalty: 1.2`
- `temperature: 0.4`
- `top_p: 0.5`

