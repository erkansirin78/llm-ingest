import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
import chromadb
from langchain_openai.embeddings import OpenAIEmbeddings
from uuid import uuid4
from langchain_core.documents import Document

load_dotenv()

chroma_db_path = os.getenv("chroma_db_path")
embeddings = OpenAIEmbeddings()



persistent_client = chromadb.PersistentClient(path=chroma_db_path)
collection = persistent_client.get_or_create_collection("example_collection")
collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

vector_store = Chroma(
    client=persistent_client,
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=chroma_db_path,
)


if __name__=='__main__':
    print("Hello")