import os

from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ["BOT_TOKEN"]
API_ID = os.environ["API_ID"]
API_HASH = os.environ["API_HASH"]
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "turbo")
TEMP_DIR = "temp"
