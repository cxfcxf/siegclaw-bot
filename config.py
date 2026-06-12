import logging
import os
import sys

import openai as _openai

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
MIMO_API_KEY = os.getenv("MIMO_API_KEY")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL")

# --- Webhook ---
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8643"))
WEBHOOK_CHANNEL_ID = int(os.getenv("WEBHOOK_CHANNEL_ID", "0"))

# --- Model & Discord ---
MODEL = os.getenv("MODEL", "mimo-v2.5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
MAX_DISCORD_LENGTH = 2000
# Sampling temperature; unset → API default
_temp = os.getenv("MODEL_TEMPERATURE")
MODEL_TEMPERATURE = float(_temp) if _temp else None
MODEL_TIMEOUT = float(os.getenv("MODEL_TIMEOUT", "120"))
MODEL_MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "2"))

# --- Browser ---
CAMOFOX_URL = os.getenv("CAMOFOX_URL", "http://localhost:9377")

# --- Context window ---
CONTEXT_MESSAGE_COUNT = int(os.getenv("CONTEXT_MESSAGE_COUNT", "50"))
CONTEXT_TIME_WINDOW_HOURS = int(os.getenv("CONTEXT_TIME_WINDOW_HOURS", "24"))
CONTEXT_ACTIVITY_THRESHOLD = int(os.getenv("CONTEXT_ACTIVITY_THRESHOLD", "30"))
CONTEXT_MAX_MESSAGES = int(os.getenv("CONTEXT_MAX_MESSAGES", "150"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "16000"))

# --- Memory ---
LANCEDB_PATH = os.getenv("LANCEDB_PATH", "data/lancedb")
MEMORY_DECAY_DAYS = int(os.getenv("MEMORY_DECAY_DAYS", "90"))

# --- Search ---
FIRECRAWL_URL = os.getenv("FIRECRAWL_API_URL", "http://localhost:3002")
FIRECRAWL_MAX_RESULTS = int(os.getenv("FIRECRAWL_MAX_RESULTS", "5"))

# --- Tool loop ---
TOOL_MAX_ITERS = int(os.getenv("TOOL_MAX_ITERS", "5"))

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

# --- Clients ---
# Google genai: embeddings only
genai_client = genai.Client(api_key=GOOGLE_API_KEY)
# Mimo: chat completions (OpenAI-compatible)
mimo_client = _openai.AsyncOpenAI(
    api_key=MIMO_API_KEY,
    base_url=MIMO_BASE_URL,
    timeout=MODEL_TIMEOUT,
    max_retries=MODEL_MAX_RETRIES,
)
