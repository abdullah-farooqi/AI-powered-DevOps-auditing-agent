import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.6-flash"

load_dotenv()


def generate_text(prompt: str, model: str | None = None) -> str:
    """Generate text with the configured Gemini model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to the local .env file.")
    model_name = model or os.getenv("LLM_MODEL", DEFAULT_MODEL).strip()
    if not model_name:
        raise RuntimeError("LLM_MODEL cannot be empty.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
        if response.text is None:
            raise RuntimeError("Gemini returned no text.")
        return response.text
    finally:
        client.close()
