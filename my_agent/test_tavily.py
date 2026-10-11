from pathlib import Path
from dotenv import load_dotenv
from tavily import TavilyClient
import os

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise RuntimeError("TAVILY_API_KEY .env mein nahi mili")

client = TavilyClient(api_key=api_key)

response = client.search(
    query="ISRO latest news",
    max_results=3
)

for result in response.get("results", []):
    print("Title:", result.get("title"))
    print("URL:", result.get("url"))
    print()