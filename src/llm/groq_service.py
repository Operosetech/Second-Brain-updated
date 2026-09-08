from groq import Groq

from src.config.config import settings


_client = Groq(api_key=settings.GROQ_API_KEY)


def invoke_chat(prompt: str) -> str:
    response = _client.chat.completions.create(
        model=settings.GROQ_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    return content or ""
