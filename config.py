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
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Model & Discord ---
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
MAX_DISCORD_LENGTH = 2000

# --- Context window ---
CONTEXT_MESSAGE_COUNT = int(os.getenv("CONTEXT_MESSAGE_COUNT", "30"))
CONTEXT_TIME_WINDOW_HOURS = int(os.getenv("CONTEXT_TIME_WINDOW_HOURS", "24"))
CONTEXT_ACTIVITY_THRESHOLD = int(os.getenv("CONTEXT_ACTIVITY_THRESHOLD", "20"))
CONTEXT_MAX_MESSAGES = int(os.getenv("CONTEXT_MAX_MESSAGES", "50"))

# --- Memory ---
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "data/memories.db")
MEMORY_DECAY_DAYS = int(os.getenv("MEMORY_DECAY_DAYS", "90"))

# --- Search ---
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# --- Router ---
ROUTER_ENABLED = os.getenv("ROUTER_ENABLED", "true").lower() == "true"

# --- Prompts ---
SYSTEM_INSTRUCTION = (
    "You are SiegClaw, a helpful AI assistant in a Discord server with multiple users. "
    "You will receive recent conversation history followed by the current question. "
    "Use the conversation history when it is relevant to the question (e.g. summaries, follow-ups, context). "
    "Do not bring up unrelated topics from the history unprompted. "
    "When referring to people, always use their name (e.g. 'siegfried said...', 'ED asked...') — "
    "never use 'you' or 'we' since there are multiple participants. "
    "Be concise. Do not add commentary, jokes, or tangents. "
    "Use Discord markdown formatting when helpful. "
    "Never fabricate financial data, prices, or market numbers. "
    "If real-time data is not provided in this prompt, say you don't have current data for that."
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

# --- Clients ---
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
