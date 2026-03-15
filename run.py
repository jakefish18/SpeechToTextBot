import asyncio
import logging

from aiogram import Bot, Dispatcher
from pyrogram import Client as PyroClient

from bot.config import BOT_TOKEN, API_ID, API_HASH
from bot import handlers

logging.basicConfig(level=logging.INFO)


async def main():
    pyro = PyroClient(
        "bot",
        api_id=API_ID,
        api_hash=API_HASH,
        bot_token=BOT_TOKEN,
        no_updates=True,
    )
    await pyro.start()
    handlers.pyro_client = pyro

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(handlers.router)

    try:
        await dp.start_polling(bot)
    finally:
        await pyro.stop()


if __name__ == "__main__":
    asyncio.run(main())
