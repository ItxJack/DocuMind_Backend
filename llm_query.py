"""
llm_query.py
-------------
Handles sending a user's text query to an LLM via the Groq API and
returning the answer. Used by the "/ask-llm" endpoint in main.py.

Why Groq? It's free to get an API key, and its inference is extremely fast
compared to most providers, which is great for a simple Q&A endpoint.
"""

import os
from groq import Groq

# The model name Groq currently serves for general-purpose chat.
# NOTE: llama-3.3-70b-versatile was deprecated by Groq on June 17, 2026.
# openai/gpt-oss-120b is the current recommended general-purpose replacement.
# If this model is ever deprecated too, check https://console.groq.com/docs/models
# for the latest supported model IDs.
GROQ_MODEL = "openai/gpt-oss-120b"


def ask_groq(query: str) -> str:
    """
    Sends a query to the Groq-hosted LLM and returns its text response.

    Args:
        query: The user's question/prompt.

    Returns:
        The LLM's text response.

    Raises:
        ValueError: if the GROQ_API_KEY environment variable is not set.
        RuntimeError: if the Groq API call itself fails (network issue,
                      invalid key, rate limit, etc.).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file "
            "or your deployment platform's environment variables."
        )

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": query}
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {e}")

    return response.choices[0].message.content
