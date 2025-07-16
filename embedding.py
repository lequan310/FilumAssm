import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


async def create_embedding(content: str):
    result = await client.aio.models.embed_content(
        model="gemini-embedding-001",
        contents=content,
    )

    return result.embeddings[0].values
