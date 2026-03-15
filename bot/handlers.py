from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from aiogram import Router, F
from aiogram.filters import CommandStart
from aiogram.types import Message, BufferedInputFile
from pyrogram import Client as PyroClient

from bot.config import TEMP_DIR
from bot.transcriber import transcribe, SUPPORTED_EXTENSIONS

router = Router()
pyro_client: Optional[PyroClient] = None

MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB — Pyrogram (MTProto) limit

START_TEXT = (
    "Привет! Я бот для распознавания речи.\n\n"
    "Отправьте мне голосовое сообщение или аудиофайл, "
    "и я переведу речь в текст.\n\n"
    "Поддерживаемые форматы: MP3, WAV, OGG, M4A, FLAC, WEBM.\n"
    "Максимальный размер файла: 2 ГБ."
)


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(START_TEXT)


@router.message(F.voice)
async def handle_voice(message: Message):
    if _is_too_large(message.voice.file_size):
        await message.answer("Голосовое сообщение слишком большое. Максимальный размер — 2 ГБ.")
        return
    await _process_audio(message)


@router.message(F.audio)
async def handle_audio(message: Message):
    if _is_too_large(message.audio.file_size):
        await message.answer("Аудиофайл слишком большой. Максимальный размер — 2 ГБ.")
        return
    await _process_audio(message)


@router.message(F.document)
async def handle_document(message: Message):
    file_name = message.document.file_name or ""
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        await message.answer(
            f"Формат файла «{ext or 'неизвестный'}» не поддерживается.\n"
            "Отправьте аудиофайл в одном из форматов: MP3, WAV, OGG, M4A, FLAC, WEBM."
        )
        return
    if _is_too_large(message.document.file_size):
        await message.answer("Файл слишком большой. Максимальный размер — 2 ГБ.")
        return
    await _process_audio(message)


def _is_too_large(file_size: int | None) -> bool:
    return file_size is not None and file_size > MAX_FILE_SIZE


async def _process_audio(message: Message):
    processing_msg = await message.answer("Распознаю речь, подождите...")

    os.makedirs(TEMP_DIR, exist_ok=True)

    file_path = None
    try:
        pyro_msg = await pyro_client.get_messages(message.chat.id, message.message_id)
        file_path = await pyro_msg.download(file_name=f"{TEMP_DIR}/")

        text = transcribe(file_path)

        if text:
            if len(text) <= 4096:
                await message.answer(text)
            else:
                file = BufferedInputFile(text.encode("utf-8"), filename="transcription.txt")
                await message.answer_document(file, caption="Текст слишком длинный, отправляю файлом.")
        else:
            await message.answer("Не удалось распознать речь. Попробуйте отправить более чёткую запись.")
    except Exception as e:
        await message.answer(f"Произошла ошибка при обработке: {e}")
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        await processing_msg.delete()
