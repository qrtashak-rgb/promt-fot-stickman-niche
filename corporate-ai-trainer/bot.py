"""Telegram-бот корпоративного ИИ-наставника.

Запуск: python bot.py
Перед первым запуском выполните: python ingest.py
"""
import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

import config
import ingest
import llm
import vectorstore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
dp = Dispatcher()


def is_admin(user_id: int) -> bool:
    return user_id in config.ADMIN_IDS


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я корпоративный ИИ-наставник. Задай вопрос о процессах, "
        "регламентах или инструментах компании — отвечу на основе базы знаний "
        "и укажу источник.\n\n"
        "Если ответа в базе нет, я скажу об этом честно и предложу, к кому "
        "обратиться."
    )


@dp.message(Command("reload"))
async def cmd_reload(message: Message) -> None:
    if not is_admin(message.from_user.id):
        await message.answer("Эта команда доступна только администраторам.")
        return

    await message.answer("Переиндексирую базу знаний, подождите...")
    try:
        await asyncio.to_thread(ingest.main)
    except SystemExit:
        pass
    except Exception as exc:
        logger.exception("Ошибка при переиндексации")
        await message.answer(f"Не удалось переиндексировать базу: {exc}")
        return
    await message.answer("База знаний обновлена.")


@dp.message(F.text)
async def handle_question(message: Message) -> None:
    question = message.text.strip()
    if not question:
        return

    await bot.send_chat_action(message.chat.id, "typing")

    chunks = await asyncio.to_thread(vectorstore.query, question, config.TOP_K)
    relevant_chunks = [c for c in chunks if c["distance"] <= config.MAX_DISTANCE]

    if not relevant_chunks:
        await message.answer(
            "У меня нет информации по этому вопросу в базе знаний. "
            "Уточни у руководителя или HR."
        )
        return

    try:
        answer = await asyncio.to_thread(llm.answer_question, question, relevant_chunks)
    except Exception as exc:
        logger.exception("Ошибка при обращении к LLM")
        await message.answer(f"Не получилось сформировать ответ: {exc}")
        return

    sources = sorted({c["metadata"]["source"] for c in relevant_chunks})
    await message.answer(f"{answer}\n\n📄 Проверено по документам: {', '.join(sources)}")


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
