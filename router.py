import asyncio
import json
import logging

from google.genai import types

from config import MODEL, ROUTER_ENABLED, gemini_client

log = logging.getLogger("siegclaw.router")

ROUTER_PROMPT = """\
Classify the user's query into exactly one category:
- SEARCH: needs current/real-time information from the web (news, weather, recent events, factual lookups about current state of the world)
- FINANCE: asks about prices, stocks, crypto, commodities, exchange rates, market data, indices (Dow, S&P, Nasdaq, etc.), or any financial instrument
- MEMORY: asks about a specific fact, detail, or decision from past conversations (e.g. "what did John say about X", "do you remember my preference")
- DIRECT: general conversation, opinions, creative tasks, coding help, jokes, summarizing the current conversation, or anything the AI can answer from training data or the provided chat history

Also consider the recent conversation context — if the conversation has been about a topic, \
a vague follow-up or correction (e.g. "today is X", "that's wrong", "try again", "how about X", "what about X") likely relates to that same topic and should use the same route.

Return a JSON object with a single key "route" and one of these values: "SEARCH", "FINANCE", "MEMORY", "DIRECT".
Example: {"route": "FINANCE"}"""


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
                response_mime_type="application/json",
                max_output_tokens=600,
                thinking_config=types.ThinkingConfig(thinking_budget=512),
            ),
        )
        text = (response.text or "").strip()
        if not text:
            log.warning("Router returned empty response, defaulting to direct")
            return "direct"
        data = json.loads(text)
        classification = data.get("route", "").upper()
        log.info("Route: %s for '%s'", classification, query[:60])
        if classification in ("SEARCH", "FINANCE", "MEMORY", "DIRECT"):
            return classification.lower()
        log.warning("Router returned unexpected value: '%s', defaulting to direct", classification)
    except Exception as e:
        log.error("Classification failed: %s — defaulting to direct", e)

    return "direct"
