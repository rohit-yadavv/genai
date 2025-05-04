import os
import json
from pathlib import Path
from collections import defaultdict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI


# Function to load environment variables
def load_environment_variables():
    google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not google_credentials:
        raise ValueError(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set."
        )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    return gemini_api_key


# Function to initialize OpenAI client
def initialize_openai_client(api_key):
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


# Function to load and split PDF
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents=docs)


# Function to expand the query
def expand_query(client, user_query):
    augmentation_prompt = f"""Generate 3 semantically different variations of this question for better retrieval:
    "{user_query}"
    Return a Python list of 3 strings. No explanation, no formatting, no code block—just plain list."""

    query_expansion = client.chat.completions.create(
        model="gemini-2.5-flash-preview-04-17",
        messages=[{"role": "user", "content": augmentation_prompt}],
    )
    try:
        res = json.loads(query_expansion.choices[0].message.content)
        return res if isinstance(res, list) else []
    except json.JSONDecodeError:
        print("Error decoding JSON from Gemini response.")
        return []


# Function to retrieve relevant documents using Reciprocal Rank Fusion
def retrieve_relevant_docs(retriever, similar_queries, k=60, top_k=10):
    doc_scores = defaultdict(float)
    doc_text_map = {}

    for query in similar_queries:
        docs = retriever.similarity_search(query=query, k=top_k)
        for rank, doc in enumerate(docs):
            key = doc.page_content
            doc_text_map[key] = doc
            doc_scores[key] += 1 / (k + rank + 1)

    # Sort documents by RRF score
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    unique_docs = [doc_text_map[text] for text, _ in ranked_docs]

    return "\n\n".join(doc.page_content for doc in unique_docs[:top_k])


# Function to get the OpenAI response
def get_openai_response(client, context, user_query):
    response = client.chat.completions.create(
        model="gemini-2.5-flash-preview-04-17",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant knowledgeable in Node.js.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {user_query}",
            },
        ],
    )
    return response.choices[0].message.content


# Main function to run the workflow
def main(pdf_path, user_query):
    load_environment_variables()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    client = initialize_openai_client(gemini_api_key)

    # Load and split PDF
    split_docs = load_and_split_pdf(pdf_path)

    # Create embedder
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Connect to existing Qdrant vector store
    retriever = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="learning_langchain",
        embedding=embedder,
    )

    # Expand user query
    similar_queries = expand_query(client, user_query)

    # Retrieve relevant documents using RRF
    context = retrieve_relevant_docs(retriever, similar_queries)

    # Get OpenAI response
    response = get_openai_response(client, context, user_query)
    print(response)


# Run the script
if __name__ == "__main__":
    pdf_path = Path(__file__).parent.parent / "nodejs.pdf"
    user_query = input("Ask a question about Node.js: ")
    main(pdf_path, user_query)
