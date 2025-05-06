from openai import OpenAI
from dotenv import load_dotenv
from mem0 import Memory
import os

load_dotenv()


QUADRANT_HOST = "localhost"

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "reform-william-center-vibrate-press-5829"


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY,
)


config = {
    "version": "v1.1",
    "embedder": {
        "provider": "gemini",
        "config": {"api_key": GEMINI_API_KEY, "model": "models/embedding-001"},
    },
    "llm": {
        "provider": "gemini",
        "config": {"api_key": GEMINI_API_KEY, "model": "gemini-2.0-flash"},
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QUADRANT_HOST,
            "port": 6333,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD,
        },
    },
}

mem_client = Memory.from_config(config)


SYSTEM_PROMPT = """
  You are a helpful ai assistant that resolve user query

"""


def chat(message):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    response = llm.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages,
    )

    mem_client.add(messages, user_id="rohit123")

    return response.choices[0].message.content


while True:
    message = input(">>> ")
    out = chat(message)
    print(out)
