import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "dummy")

# --- Webhook ---
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8643"))
WEBHOOK_CHANNEL_ID = int(os.getenv("WEBHOOK_CHANNEL_ID", "0"))

# --- Model & Discord ---
MODEL = os.getenv("MODEL", "anthropic/claude-haiku-4-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
MAX_DISCORD_LENGTH = 2000

# --- OR Proxy ---
OR_PROXY_URL = os.getenv("OR_PROXY_URL", "http://localhost:8787")

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
SYSTEM_INSTRUCTION = (
    "You are SiegClaw, a helpful AI assistant in a Discord server with multiple users. "
    "You will receive recent conversation history followed by the current question. "
    "Use the conversation history when it is relevant to the question (e.g. summaries, follow-ups, context). "
    "Do not bring up unrelated topics from the history unprompted. "
    "When referring to people, always use their name (e.g. 'siegfried said...', 'ED asked...') — "
    "never use 'you' or 'we' since there are multiple participants. "
    "Be concise. Occasionally add a brief witty remark or light commentary at the end if it fits naturally — but keep it short and don't force it. "
    "Use Discord markdown formatting when helpful. "
    "Never fabricate financial data, prices, or market numbers. "
    "You have tools available — use them. If a question requires current information, prices, news, "
    "or anything you're not certain about, call web_search or get_financial_data instead of saying you don't know. "
    "Only say you lack information if a search also fails to find it."
)

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
openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=f"{OR_PROXY_URL}/v1",
)


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def openai_generate(model: str, messages: list, **kwargs):
    return openai_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
