from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Load the base page
url = "https://chaidocs.vercel.app/youtube/getting-started"
loader = WebBaseLoader(web_path=url)
soup = loader.scrape()
# print(soup)

base_url = "https://chaidocs.vercel.app"
links = set()

# Grab all sidebar links
for a_tag in soup.select("details ul li a[href]"):
    href = a_tag["href"]
    # print(href)
    if href.startswith("/"):
        full_url = (
            base_url + href
        )  # e.g: https://chaidocs.vercel.app/youtube/chai-aur-html/welcome/
        links.add(full_url)

all_urls = [url] + list(links)
# print(all_urls)


# Print loaded URLs
# print("Final URL list:")
# for u in all_urls:
#     print(u)


topic_urls = defaultdict(list)

for link in all_urls:
    url_parts = link.split(
        "/"
    )  # ['https:', '', 'chaidocs.vercel.app', 'youtube', 'chai-aur-html', 'welcome']
    if "chai-aur" in url_parts[4]:
        topic = url_parts[4]  # chai-aur-html, chai-aur-git, etc.
        topic_urls[topic].append(
            link
        )  # add this (link) URL to the list of URLs under that topic in our dictionary.

# print(topic_urls)


api_key = os.getenv("OPENAI_API_KEY")
embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

for topic, url in topic_urls.items():
    # print(f"Processing Topic {topic}")
    # print(url)

    # load documents
    loader = WebBaseLoader(web_paths=url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    split_docs = text_splitter.split_documents(documents=docs)

    # print("DOCS", len(docs))
    # print("SPLIT", len(split_docs))

    # vector_store = QdrantVectorStore.from_documents(
    #     documents=[],
    #     url="http://localhost:6333",
    #     collection_name=topic,
    #     embedding=embedder
    # )

    # vector_store.add_documents(documents=split_docs)
    # print("Injection Done")
