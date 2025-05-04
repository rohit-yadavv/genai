from google import genai
from dotenv import load_dotenv
from openai import OpenAI
import os

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

res = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the capital of The great and best country in the world India?",
)
print(res.text)
