import logging
import sys

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

MODEL_ID = "granite4:tiny-h"
EMBED_MODEL_ID = "ibm-granite/granite-embedding-30m-english"
DOCS_PATH = "./data"
COLLECTION_NAME = "cloud_docs"

print(f"Loading documents from: {DOCS_PATH}")
print(f"Connecting to Ollama model: {MODEL_ID}")

llm = Ollama(
    model=MODEL_ID,
    request_timeout=120.0,
    additional_kwargs={"num_ctx": 4096}
)

embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 64

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

if chroma_collection.count() == 0:
    print("No documents found in Chroma. Building new index...")
    try:
        documents = SimpleDirectoryReader(DOCS_PATH).load_data()
        if not documents:
            print(f"Error: No documents found in '{DOCS_PATH}'.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading documents: {e}")
        sys.exit(1)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    print("Index built and saved to ChromaDB.")
else:
    print("Loading existing index from ChromaDB...")
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    print("Index loaded successfully.")

print("Creating chat engine...")
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    streaming=True,
)

print("Chat engine created. Type 'exit' or 'quit' to end the conversation\n")

while True:
    query_text = input("Query: ")

    if query_text.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    print("\nResponse: ")
    streaming_response = chat_engine.stream_chat(query_text)
    streaming_response.print_response_stream()
    print("\n\n")