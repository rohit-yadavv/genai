from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os
import ast
import json

# adding google credentials for embeddings to work
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "E:\Downloads\gemini-key-for-learining-02807fd398a1.json"
)


# Initialize OpenAI client

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 1. Load and split PDF
pdf_path = Path(__file__).parent.parent / "nodejs.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
split_docs = text_splitter.split_documents(documents=docs)

# 2. Create an embedder
embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Only run the below once to insert data into Qdrant
# vector_store = QdrantVectorStore.from_documents(
#     documents=split_docs,
#     embedding=embedder,
#     url="http://localhost:6333",
#     collection_name="learning_node_js",
# )
# vector_store.add_documents(documents=split_docs)
# print("📄 PDF Ingestion Complete!\n")

# Connect to existing Qdrant vector store
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder,
)

# 3. Take user question
user_query = input("Ask a question about Node.js: ")

# 4. Query Expansion Prompt
augmentation_prompt = f"""Generate 3 semantically different variations of this question for better retrieval:
"{user_query}"
Return a json with Python list of 3 strings in that and dont wrap output in ```json give me directly .

Example: 
{{"output":["hi", "hello", "how are you"]}}
{{"output": ["good morning", "what's up", "nice to meet you"]}}
"""

# Call OpenAI to expand query
query_expansion = client.chat.completions.create(
    model="gemini-2.5-flash-preview-04-17",
    messages=[{"role": "user", "content": augmentation_prompt}],
)

# 5. Parse string output to actual Python list
res = json.loads(query_expansion.choices[0].message.content)
similar_queries = res.get("output")

# # 6. Search for relevant docs for each variation
relevant_docs = []
for query in similar_queries:
    docs = retriever.similarity_search(query=query)
    relevant_docs.extend(docs)


# Create a dictionary where each document's 'page_content' is used as the key, and the 'doc' is the value.
# This removes any duplicates, as dictionary keys must be unique.
unique_docs = list({doc.page_content: doc for doc in relevant_docs}.values())

context = "\n\n".join(doc.page_content for doc in unique_docs)


response = client.chat.completions.create(
    model="gemini-2.5-flash-preview-04-17",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant knowledgeable in Node.js.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"},
    ],
)

print(response.choices[0].message.content)