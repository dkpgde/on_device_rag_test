import json
import statistics
import time

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import psutil
from tqdm import tqdm

MODEL_ID = "granite4:tiny-h" 
EMBED_MODEL_ID = "ibm-granite/granite-embedding-30m-english"
INPUT_FILE = "RAG.jsonl"
OUTPUT_FILE = "cloud_predictions.json"
DB_PATH = "./chroma_db"

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

#  Setup 
llm = Ollama(model=MODEL_ID, request_timeout=120.0, additional_kwargs={"num_ctx": 4096})
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID)
Settings.llm = llm
Settings.embed_model = embed_model

db = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = db.get_or_create_collection("cloud_docs") 
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store, 
    storage_context=StorageContext.from_defaults(vector_store=vector_store)
)

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=False)

#  Performance Tracking
process = psutil.Process() 
latencies = []
ram_usage = []

#  Main Execution 
def run():
    tasks = load_and_filter_tasks(INPUT_FILE)
    results = []
    current_conv_id = None
    
    print("Starting Inference...")
    
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

    #  Save Results 
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    #  Print Final Stats 
    print("\n" + "="*30)
    print("BENCHMARK PERFORMANCE REPORT")
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