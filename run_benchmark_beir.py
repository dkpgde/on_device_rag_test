import json
import os
import pickle
import chromadb
import pytrec_eval
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader

from llama_index.core import (
    Settings, 
    StorageContext, 
    VectorStoreIndex, 
    QueryBundle
)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank

EMBED_MODEL_ID = "ibm/granite-embedding:30m"
RERANK_MODEL_ID = "BAAI/bge-reranker-base"
MODEL_ID = "granite4:1b-h"
DATA_PATH = "./data"
DB_PATH = "./chroma_db_nfcorpus_granite_30m"
COLLECTION_NAME = "nfcorpus_granite"
PERSIST_DIR = "./storage_nodes_nfcorpus_granite"
NODES_FILE = os.path.join(PERSIST_DIR, "nodes.pkl")

# Tuning
RETRIEVAL_TOP_K = 50 
RERANK_TOP_N = 10     

def main():
    # 1. Load Data
    print(f"Loading queries and qrels from {DATA_PATH}...")
    _, queries, qrels = GenericDataLoader(data_folder=DATA_PATH).load(split="test")

    # 2. Setup Models
    print(f"Initializing models...")
    llm = Ollama(model=MODEL_ID, request_timeout=300.0)
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL_ID,
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0}
    )
    reranker = SentenceTransformerRerank(model=RERANK_MODEL_ID, top_n=RERANK_TOP_N)
    
    Settings.llm = llm
    Settings.embed_model = embed_model

    # 3. Connect to Vector Store
    if not os.path.exists(DB_PATH):
        print(f"Error: DB not found at {DB_PATH}. Run ingest_ollama.py first.")
        return

    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Load Index & Nodes
    print("Loading Index...")
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    
    if os.path.exists(NODES_FILE):
        with open(NODES_FILE, "rb") as f:
            nodes = pickle.load(f)
    else:
        print("Error: Nodes file missing. Run ingest_ollama.py first.")
        return

    # 5. Initialize Retrievers
    vector_retriever = index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=RETRIEVAL_TOP_K)

    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=RETRIEVAL_TOP_K,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=False
    )

    # 6. Run Inference
    print("Running Hybrid Retrieval + Reranking...")
    results = {}
    
    for query_id, query_text in tqdm(queries.items()):
        if query_id not in qrels: continue

        # Retrieval
        initial_nodes = fusion_retriever.retrieve(query_text)
        
        # Reranking
        query_bundle = QueryBundle(query_str=query_text)
        reranked_nodes = reranker.postprocess_nodes(initial_nodes, query_bundle)

        # pytrec_eval needs strict typing
        scores = {}
        for node in reranked_nodes:
            doc_id = str(node.node.id_)
            score = float(node.score) if node.score is not None else 0.0
            scores[doc_id] = score
            
        results[str(query_id)] = scores

    # 7. Evaluate
    print("Calculating Metrics...")
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10', 'recall_10'})
    metrics = evaluator.evaluate(results)
    
    avg_ndcg = sum(m['ndcg_cut_10'] for m in metrics.values()) / len(metrics)
    avg_recall = sum(m['recall_10'] for m in metrics.values()) / len(metrics)

    print("\n" + "="*30)
    print(f" RESULTS (Ollama Embed + BGE Rerank)")
    print("="*30)
    print(f"NDCG@10:   {avg_ndcg:.4f}")
    print(f"Recall@10: {avg_recall:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()