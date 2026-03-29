import os
import asyncio
import logging
import psutil
import yfinance as yf
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from config import format_prompt, MODEL_PATH, GEN_SETTINGS
from mlx_lm import load, generate

# config = prompts["medical_expert"]
model, tokenizer = load(MODEL_PATH)

# 1. Настройка окружения
logging.basicConfig(level=logging.INFO)
load_dotenv(dotenv_path="chat_bot.env")
TOKEN = os.getenv("BOT_TOKEN")

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

        prompt = format_prompt(message.text)

        # Выносим генерацию в отдельный поток, чтобы не блокировать asyncio
        loop = asyncio.get_event_loop()
        
        def run_generation():
            sampler = make_sampler(
                temp=GEN_SETTINGS.get("temp"), 
                top_p=GEN_SETTINGS.get("top_p", 1.0)
            )
             # punishes model for repeating the same tokens (аналог presence_penalty), в MLX нет отдельного frequency_penalty
            logits_processors = make_logits_processors(repetition_penalty=GEN_SETTINGS.get("repetition_penalty"))
            
            return generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=GEN_SETTINGS.get("max_tokens"),
                sampler=sampler,
                logits_processors=logits_processors,
            )

        # Запускаем и ждем результат
        response = await loop.run_in_executor(None, run_generation)

        reply = response.strip()
        # Если модель сама не закрыла тег, обрезаем
        if "<|eot_id|>" in reply:
            reply = reply.split("<|eot_id|>")[0].strip()

        await message.answer(reply)
        
    except Exception as e:
        logging.error(f"Ошибка генерации: {e}")
        await message.answer(f"❌ Ошибка: Проверь сервер")

async def main():
    print("🤖 Бот запущен (без поддержки watchdog)")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
