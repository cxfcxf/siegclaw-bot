import logging
import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("siegclaw")

# --- Required ---
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Webhook ---
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8643"))
WEBHOOK_CHANNEL_ID = int(os.getenv("WEBHOOK_CHANNEL_ID", "0"))

# --- Model & Discord ---
MODEL = os.getenv("MODEL", "gemini-3-flash-preview")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
MAX_DISCORD_LENGTH = 2000

# --- Browser ---
CAMOFOX_URL = os.getenv("CAMOFOX_URL", "http://localhost:9377")

# --- Context window ---
CONTEXT_MESSAGE_COUNT = int(os.getenv("CONTEXT_MESSAGE_COUNT", "30"))
CONTEXT_TIME_WINDOW_HOURS = int(os.getenv("CONTEXT_TIME_WINDOW_HOURS", "24"))
CONTEXT_ACTIVITY_THRESHOLD = int(os.getenv("CONTEXT_ACTIVITY_THRESHOLD", "20"))
CONTEXT_MAX_MESSAGES = int(os.getenv("CONTEXT_MAX_MESSAGES", "50"))

# --- Memory ---
LANCEDB_PATH = os.getenv("LANCEDB_PATH", "data/lancedb")
MEMORY_DECAY_DAYS = int(os.getenv("MEMORY_DECAY_DAYS", "90"))

# --- Search ---
FIRECRAWL_URL = os.getenv("FIRECRAWL_API_URL", "http://localhost:3002")
FIRECRAWL_MAX_RESULTS = int(os.getenv("FIRECRAWL_MAX_RESULTS", "5"))

# --- Prompts ---
with open(os.path.join(os.path.dirname(__file__), "SOUL.md")) as _f:
    SYSTEM_INSTRUCTION = _f.read().strip()

MEMORY_EXTRACTION_PROMPT = """\
Analyze the following conversation and extract important facts worth remembering \
for future conversations. Focus on:
- People's names, roles, preferences, birthdays
- Decisions made by the group
- Important events, dates, plans
- Personal details people share
- Anything that would be useful to recall later

Return ONLY a JSON array of short fact strings. If nothing is worth remembering, return [].
Example: ["John's birthday is March 5", "The team decided to use React for the frontend"]

Conversation:
{conversation}

Bot's reply:
{bot_reply}"""

# --- Client ---
genai_client = genai.Client(api_key=GOOGLE_API_KEY)
