from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pathlib import Path

file_path = Path(__file__).parent.parent / "nodejs.pdf"

# loader
loader = PyPDFLoader(file_path)
docs = loader.load()


# chunking text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_docs = splitter.split_documents(documents=docs)


embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
# vector = embeddings.embed_query("hello, world!")
# print(vector[:5])


# Only run the below once to insert data into Qdrant
# vector_store = QdrantVectorStore.from_documents(
#     documents=splitted_docs,
#     embedding=embedder,
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
# )

retriver = QdrantVectorStore.from_existing_collection(
    documents=splitted_docs,
    embedding=embedder,
    url="http://localhost:6333",
    collection_name="learning_langchain",
)
