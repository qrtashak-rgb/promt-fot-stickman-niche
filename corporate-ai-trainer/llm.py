"""Обёртка над Anthropic Messages API."""
from anthropic import Anthropic

import config
import prompts

_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)


def answer_question(question: str, context_chunks: list[dict]) -> str:
    user_message = prompts.build_user_message(question, context_chunks)

    response = _client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=prompts.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
