
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY missing from my_agent/.env"
    )

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": (
                "Search the web for the latest ISRO news. "
                "Summarize the findings and include source URLs."
            ),
        }
    ],
    tools=[{"type": "browser_search"}],
)

print(response.choices[0].message.content)