# SpeechToTextBot

Telegram-бот для распознавания речи с помощью OpenAI Whisper.

## Установка

```bash
pip install -r requirements.txt
```

## Настройка

Создайте файл `.env` на основе `.env.example`:

```bash
cp .env.example .env
```

Заполните `BOT_TOKEN` токеном вашего Telegram-бота.

| Переменная | Описание | По умолчанию |
|---|---|---|
| `BOT_TOKEN` | Токен Telegram-бота | — |
| `WHISPER_MODEL` | Модель Whisper (`tiny`, `base`, `small`, `medium`, `large`, `turbo`) | `turbo` |

## Запуск

```bash
python run.py
```
