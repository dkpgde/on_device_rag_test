import json
import statistics
import time
import os
import pickle
import psutil
from tqdm import tqdm

import chromadb
from llama_index.core import (
    Settings, 
    StorageContext, 
    VectorStoreIndex, 
    SimpleDirectoryReader
)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

MODEL_ID = "granite4:1b-h" 
EMBED_MODEL_ID = "ibm-granite/granite-embedding-30m-english"
INPUT_FILE = "RAG.jsonl"
OUTPUT_FILE = "cloud_predictions_hybrid.json"
DB_PATH = "./chroma_db"
DOCS_PATH = "./data"
PERSIST_DIR = "./storage_nodes"
NODES_FILE = os.path.join(PERSIST_DIR, "bm25_nodes.pkl")

def load_and_filter_tasks(filepath):
    print(f"Reading {filepath}...")
    tasks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                entry = json.loads(line)
                if "cloud" in entry.get("Collection", "").lower():
                    tasks.append(entry)
            except json.JSONDecodeError:
                continue
    
    tasks.sort(key=lambda x: (x.get('conversation_id'), int(x.get('turn', 0))))
    print(f"Loaded {len(tasks)} Cloud tasks.")
    return tasks

# Setup Models
llm = Ollama(model=MODEL_ID, request_timeout=300.0, additional_kwargs={"num_ctx": 4096})
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID)
Settings.llm = llm
Settings.embed_model = embed_model

# Setup Chroma
db = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = db.get_or_create_collection("cloud_docs") 
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load Vector Index
if chroma_collection.count() > 0:
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context
    )
else:
    # Fallback if benchmark is run on empty DB
    print("Vector DB empty. Parsing documents...")
    documents = SimpleDirectoryReader(DOCS_PATH).load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Load Nodes for BM25
nodes = []
if os.path.exists(NODES_FILE):
    print(f"Loading cached nodes for BM25 from {NODES_FILE}...")
    with open(NODES_FILE, "rb") as f:
        nodes = pickle.load(f)
else:
    print("Cached nodes not found. Parsing from source...")
    documents = SimpleDirectoryReader(DOCS_PATH).load_data()
    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    # Save for future use
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(NODES_FILE, "wb") as f:
        pickle.dump(nodes, f)

# Setup Hybrid Retrievers
vector_retriever = index.as_retriever(similarity_top_k=5)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
    verbose=False
)

# Build Chat Engine
query_engine = RetrieverQueryEngine.from_args(fusion_retriever)
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine, 
    llm=llm,
    verbose=False
)

# Performance Tracking
process = psutil.Process() 
latencies = []
ram_usage = []

def run():
    tasks = load_and_filter_tasks(INPUT_FILE)
    original_count = len(tasks)
    subset_size = max(1, original_count // 10)
    tasks = tasks[:subset_size]
    print(f"Running subset: {len(tasks)} tasks (1/15th of {original_count})")
    results = []
    current_conv_id = None
    
    print("Starting Hybrid Inference...")
    
    for task in tqdm(tasks):
        conv_id = task.get('conversation_id')
        
        if conv_id != current_conv_id:
            chat_engine.reset()
            current_conv_id = conv_id
            
        input_list = task.get('input', [])
        if not input_list: continue
        query_text = input_list[-1]['text']
        
        try:
            start_time = time.time()
            
            response = chat_engine.chat(query_text)
            
            end_time = time.time()
            
            # Record Metrics
            duration = end_time - start_time
            current_mem = process.memory_info().rss / (1024 * 1024)
            
            latencies.append(duration)
            ram_usage.append(current_mem)

            # Save Output
            output_entry = task.copy()
            output_entry['model_prediction'] = response.response
            results.append(output_entry)
            
        except Exception as e:
            print(f"\nError: {e}")
            task['model_prediction'] = "Error"
            results.append(task)

    # Save Results 
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    # Print Final Stats 
    print("\n" + "="*30)
    print("BENCHMARK PERFORMANCE REPORT (HYBRID)")
    print("="*30)
    if latencies:
        print(f"Total Queries:     {len(latencies)}")
        print(f"Avg Time/Query:    {statistics.mean(latencies):.2f} seconds")
        print(f"Max Time/Query:    {max(latencies):.2f} seconds")
        print(f"Avg Script RAM:    {statistics.mean(ram_usage):.2f} MB")
        print(f"Peak Script RAM:   {max(ram_usage):.2f} MB")
    print("="*30)

if __name__ == "__main__":
    run()