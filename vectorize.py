import os
import shutil
import pickle
import chromadb
from beir.datasets.data_loader import GenericDataLoader
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_NAME = "ibm/granite-embedding:30m"
DATA_PATH = "./data"
DB_PATH = "./chroma_db_nfcorpus_granite_30m"
COLLECTION_NAME = "nfcorpus_granite"
PERSIST_DIR = "./storage_nodes_nfcorpus_granite"
NODES_FILE = os.path.join(PERSIST_DIR, "nodes.pkl")

def main():
    # 1. Clean previous DB to prevent ID conflicts
    if os.path.exists(DB_PATH):
        print(f"Deleting old DB at {DB_PATH}...")
        shutil.rmtree(DB_PATH)
    os.makedirs(DB_PATH, exist_ok=True)

    # 2. Setup Ollama Embedding
    print(f"Initializing Ollama model: {MODEL_NAME}...")
    embed_model = OllamaEmbedding(
        model_name=MODEL_NAME,
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0}
    )
    Settings.embed_model = embed_model

    # 3. Load Data
    print(f"Loading data from {DATA_PATH}...")
    corpus, _, _ = GenericDataLoader(data_folder=DATA_PATH).load(split="test")

    # 4. Create Nodes (Preserving IDs)
    print("Converting corpus to LlamaIndex Nodes...")
    nodes = []
    for doc_id, doc_data in tqdm(corpus.items()):
        text = f"Title: {doc_data.get('title', '')}\n{doc_data.get('text', '')}"
        # Critical: Explicitly set id_ to match BEIR doc_id
        node = TextNode(text=text, id_=doc_id)
        
        # Optional: Save metadata for debugging
        node.metadata = {"title": doc_data.get('title', '')}
        nodes.append(node)
    
    # Save nodes for BM25 later
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(NODES_FILE, "wb") as f:
        pickle.dump(nodes, f)

    # 5. Ingest into Chroma via LlamaIndex
    print("Ingesting into Chroma (this uses LlamaIndex's schema)...")
    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # This handles the embedding generation and correct metadata storage
    VectorStoreIndex(
        nodes, 
        storage_context=storage_context, 
        show_progress=True
    )
    
    print(f"Ingestion complete. DB at {DB_PATH}")

if __name__ == "__main__":
    main()