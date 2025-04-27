from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings  # requires card billing
from langchain_qdrant import QdrantVectorStore
from pathlib import Path

# Data → Chunk → Embed → Store → Search → Retrieve → LLM → Output


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents=docs)


def create_embedding_model():
    return VertexAIEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def store_documents(documents, embedding_model):
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embedding_model,
        url="http://localhost:6333",
        collection_name="learning_langchain",
    )
    return vector_store


def search_documents(vector_store):
    while True:
        query = input("> ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        results = vector_store.similarity_search(query=query)
        for doc in results:
            print(f"\n{doc.page_content}\n{'-'*40}")


def main():
    file_path = Path(__file__).parent / "nodejs.pdf"
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    embedding_model = create_embedding_model()
    vector_store = store_documents(chunks, embedding_model)
    search_documents(vector_store)


if __name__ == "__main__":
    main()
