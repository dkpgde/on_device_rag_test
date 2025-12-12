import logging
import os
import pickle
import sys

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
MODEL_ID = "granite4:3b-h"
EMBED_MODEL_ID = "ibm-granite/granite-embedding-30m-english"
DOCS_PATH = "./data"
COLLECTION_NAME = "cloud_docs"
PERSIST_DIR = "./storage_nodes"
NODES_FILE = os.path.join(PERSIST_DIR, "bm25_nodes.pkl") # Explicit file for nodes

llm = Ollama(
    model=MODEL_ID, 
    request_timeout=120.0, 
    additional_kwargs={"num_ctx": 4096}
)
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID)

Settings.llm = llm
Settings.embed_model = embed_model

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

nodes = []

if os.path.exists(NODES_FILE):
    print(f"Loading cached nodes from {NODES_FILE}...")
    with open(NODES_FILE, "rb") as f:
        nodes = pickle.load(f)
else:
    print("No cached nodes found. Loading from source files...")
    documents = SimpleDirectoryReader(DOCS_PATH).load_data()
    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)
        
    print(f"Saving {len(nodes)} nodes to cache...")
    with open(NODES_FILE, "wb") as f:
        pickle.dump(nodes, f)

if chroma_collection.count() > 0:
    print("Loading existing Vector Index from Chroma...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
else:
    print("Building and populating Vector Index...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

print("Initializing Hybrid Retrievers...")
vector_retriever = index.as_retriever(similarity_top_k=5)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes, 
    similarity_top_k=5
)
fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=True,
    verbose=False
)

# chat engine itself
query_engine = RetrieverQueryEngine.from_args(fusion_retriever)
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine, 
    llm=llm,
    verbose=True
)

print("Hybrid System Ready. Type 'exit' or 'quit' to end.\n")

while True:
    try:
        query_text = input("Query: ")
        if query_text.lower() in ["exit", "quit"]:
            break
        print("\nResponse: ")
        response = chat_engine.chat(query_text)
        print(response)
        print("\n")
    except KeyboardInterrupt:
        break