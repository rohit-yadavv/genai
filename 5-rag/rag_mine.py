from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore


# Data → Chunk → Embed → Store → Search → Retrieve → LLM → Output


def index_documents():
    """
    Loads a PDF, splits it into chunks, creates embeddings,
    and indexes them in a Qdrant vector store.
    Run this only once for initial indexing of documents.
    """
    file_path = Path("nodejs.pdf")

    # Load PDF using LangChain's PyPDFLoader
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    # Split documents into chunks of 1000 characters with 200 character overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Create embeddings using Google Generative AI
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Index the chunked documents into Qdrant
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url="http://localhost:6333",  # Ensure Qdrant is running locally
        collection_name="learning_langchain",  # Specify the collection name
    )

    print("✅ Documents successfully indexed into Qdrant.")


def query_documents():
    """
    Connects to the existing Qdrant collection and allows querying via CLI.
    Provides semantic search results for user queries.
    """
    # Create the embedding model (should match the model used during indexing)
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Connect to the existing Qdrant vector store
    retriever = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="learning_langchain",
        embedding=embedder,
    )

    print("🔍 Ready to search. Type your query (or 'exit' to quit):")
    while True:
        query = input("> ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        # Perform similarity search and retrieve documents
        results = retriever.similarity_search(query=query)
        for doc in results:
            print(f"\n{doc.page_content}\n{'-'*40}")


if __name__ == "__main__":
    # Uncomment one of the two lines below depending on the use case

    # insert_documents()  # ← Run this only once to insert data into Qdrant
    query_documents()  # ← Run this to start querying the inserted data
