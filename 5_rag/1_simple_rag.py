from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os

# Constants
PDF_PATH = "nodejs.pdf"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "learning_langchain"
EMBED_MODEL = "models/embedding-001"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "E:\Downloads\gemini-key-for-learining-02807fd398a1.json"
)


def load_and_split_pdf(file_path: str):
    """
    Load PDF and split into overlapping chunks.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def index_documents():
    """
    Load, chunk, embed and index PDF content into Qdrant vector DB.
    Run once during setup.
    """
    chunks = load_and_split_pdf(PDF_PATH)
    embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedder,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    print("✅ Documents successfully indexed into Qdrant.")


def retrieve_relevant_docs(query: str):
    """
    Connect to Qdrant and retrieve all relevant documents for a query.
    """
    embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    retriever = QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embedding=embedder,
    )

    results = retriever.similarity_search(query)
    return [doc.page_content for doc in results]


def chat_with_context(query: str, context_chunks: list[str]):
    """
    Send user query and full context (all relevant chunks) to OpenAI for a response.
    """
    # Join all context chunks into a single string (as context_chunks is array so we are converting it into string)

    # Combine all chunks (array) into a single string,
    # separated by double newlines for better readability in the LLM prompt (after each chunk give two new black lines)
    context = "\n\n".join(context_chunks)

    system_prompt = f"""
You are a helpful AI assistant. Resolve user queries using the following context:
{context}
"""

    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    response = client.chat.completions.create(
        model="gemini-2.5-flash-preview-04-17",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )

    print("\n📘 Response:")
    print(response.choices[0].message.content)


def interactive_cli():
    """
    Start command-line interface for interactive Q&A.
    """
    print("🔍 Ready to search. Type your query (or 'exit' to quit):")
    while True:
        query = input("> ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        context_chunks = retrieve_relevant_docs(query)
        if not context_chunks:
            print("⚠️ No relevant documents found.")
            continue

        chat_with_context(query, context_chunks)


if __name__ == "__main__":
    # Uncomment to index documents initially
    # index_documents()

    interactive_cli()
