from dotenv import load_dotenv
import os
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from indexing_chunking import topic_urls
from indexing_chunking import embedder

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
LLM = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key
)

def classify_topic(user_input):
    topics = list(topic_urls.keys())

    SYSTEM_PROMPT = f"""
        You are a smart topic classifier.

        Based on the user's query and the list of available topics below, select the **most relevant topic** that best matches the intent.

        Topics:
        {topics}

        User Query:
        "{user_input}"

        Return **only** the topic name from the list above. Do not include any explanations or extra text.
        """
    
    response = LLM.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    )

    # print("-----> ", response.choices[0].message.content)
    return response.choices[0].message.content


# classify_topic("hello world code for c++ and javacript?")

SYSTEM_PROMPT = """
    You are a helpful AI Assistant who responds based on the available context.
    Always give the answer along with the source link provided next to the content.
    If the answer is not found in the context, reply with "I don't know based on the document."

    Context:
    {context}

    source format
    Example:
    Source: link
"""

while True:
    user_input = input("👤 Ask Question: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    topic = classify_topic(user_input)
    print(f"\n>> Routed to topic: {topic}")

    retriver = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name=topic,
        embedding=embedder
    )

    relevant_chunks = retriver.similarity_search(query=user_input)
    # print(relevant_chunks)
    context = "\n\n".join([f"{doc.page_content}\n Source: {doc.metadata.get('source')}" for doc in relevant_chunks])
    formatted_prompt = SYSTEM_PROMPT.format(context=context)

    response = LLM.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    print("🤖 Answer: ", response.choices[0].message.content)