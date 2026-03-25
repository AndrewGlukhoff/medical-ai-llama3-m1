import os
import asyncio
import logging
import psutil
import yfinance as yf
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from openai import AsyncOpenAI
import yaml

# system prompt
with open("prompts.yaml", "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

config = prompts["medical_expert"]


# 1. Настройка окружения
logging.basicConfig(level=logging.INFO)
load_dotenv(dotenv_path="chat_bot.env")
TOKEN = os.getenv("BOT_TOKEN")

# Инициализация API клиента
client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode='Markdown'))
dp = Dispatcher()

# Функция для получения курса доллара
async def get_usd_rate():
    try:
        ticker = yf.Ticker("RUB=X")
        rate = ticker.fast_info.last_price
        return f"{rate:.2f}" if rate else "нет данных"
    except Exception as e:
        logging.error(f"Ошибка финансов: {e}")
        return "ошибка API"

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    await message.answer("👋 **AI-Бот на M1 готов!**\nПиши сообщение или используй `/status`.")

@dp.message(Command("status"))
async def status_handler(message: types.Message):
    # Системные метрики
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024 ** 3)
    
    # Финансовые метрики
    usd_rate = await get_usd_rate()
    
    status_text = (
        f"🖥 **Mac M1 Status:**\n"
        f"RAM: `{ram_used:.1f} ГБ` ({ram.percent}%)\n\n"
        f"💵 **Forex:**\n"
        f"1 USD = `{usd_rate} руб.`\n\n"
        f"⏰ {message.date.strftime('%H:%M:%S')}"
    )
    await message.answer(status_text)

@dp.message()
async def chat_handler(message: types.Message):
    try:
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
        
        messages_for_llm = [
            {
                "role": "system",
                "content": config["system"]
            },
            {
                "role": "user",
                "content": config["user_template"].format(user_input=message.text)
            }
        ]
        response = await client.chat.completions.create(
            model="local-model", # возьмется любая текущая модель по адресу localhost:1234
            messages=messages_for_llm,
            temperature=0.4, #настройки мучил до слияния
            presence_penalty=1.3, # аналог repetition_penalty
            frequency_penalty=0.5, # наказывает за повтор слов
            max_tokens=400 # даем выговориться
        )
        
        # Обрезаем лишнее (защита от мусора в конце)
        reply = response.choices[0].message.content.split("<|eot_id|>")[0].strip()
        await message.answer(reply)
        
    except Exception as e:
        await message.answer(f"❌ Ошибка: Проверь сервер")

async def main():
    print("🤖 Бот запущен с поддержкой watchdog. Можешь править код!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
