import asyncio
import logging

from google.genai import types

from config import MODEL, ROUTER_ENABLED, gemini_client

log = logging.getLogger("siegclaw.router")

ROUTER_PROMPT = """\
Classify the user's query into exactly one category:
- SEARCH: needs current/real-time information from the web (news, weather, recent events, factual lookups about current state of the world)
- FINANCE: asks about prices, stocks, crypto, commodities, exchange rates, or market data
- MEMORY: asks about something from past conversations, personal details, preferences, or previous decisions in this chat
- DIRECT: general conversation, opinions, creative tasks, coding help, jokes, or anything the AI can answer from training data

Also consider the recent conversation context — if the conversation has been about a topic, \
a vague follow-up likely relates to that same topic.

Reply with ONLY one word: SEARCH, FINANCE, MEMORY, or DIRECT."""


async def classify_query(query: str, recent_context: str = "") -> str:
    """Classify a user query as 'search', 'finance', 'memory', or 'direct'.

    Uses recent conversation context for multi-turn awareness.
    """
    if not ROUTER_ENABLED:
        return "both"

    try:
        content = f"User query: {query}"
        if recent_context:
            content = f"Recent conversation:\n{recent_context}\n\n{content}"

        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=MODEL,
            contents=content,
            config=types.GenerateContentConfig(
                system_instruction=ROUTER_PROMPT,
                max_output_tokens=10,
            ),
        )
        classification = response.text.strip().upper()
        if classification in ("SEARCH", "FINANCE", "MEMORY", "DIRECT"):
            log.info("Route: %s for '%s'", classification, query[:60])
            return classification.lower()
    except Exception as e:
        log.error("Classification failed: %s", e)

    return "direct"
