import json
import os

import chromadb
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

HF_MODEL = "ibm-granite/granite-embedding-30m-english"
CORPUS_PATH = "./data/cloud.jsonl"
STORAGE_PATH = "./chroma_db"
COLLECTION_NAME = "cloud_docs"
BATCH_SIZE = 64

def get_embeddings_hf(texts, model, tokenizer, device):
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(**tokenized)
        embeddings = output.last_hidden_state[:, 0]
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalized.cpu().numpy()

def main():
    hf_model, hf_tokenizer, device = None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    hf_model = AutoModel.from_pretrained(HF_MODEL).to(device)
    hf_model.eval()

    os.makedirs(STORAGE_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=STORAGE_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    print(f"Started ingestion: {CORPUS_PATH}")
    batch_ids, batch_texts, batch_metadatas = [], [], []

    try:
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing"):
                data = json.loads(line)
                text_blob = f"Title: {data['title']}\n{data['text']}"
                
                batch_ids.append(data['_id'])
                batch_texts.append(text_blob)
                batch_metadatas.append({"title": data['title'], "original_text": data['text']})

                if len(batch_ids) >= BATCH_SIZE:
                    embeddings = get_embeddings_hf(batch_texts, hf_model, hf_tokenizer, device)
                    collection.add(ids=batch_ids, embeddings=embeddings, documents=batch_texts, metadatas=batch_metadatas)
                    batch_ids, batch_texts, batch_metadatas = [], [], []

            if batch_ids:
                embeddings = get_embeddings_hf(batch_texts, hf_model, hf_tokenizer, device)
                collection.add(ids=batch_ids, embeddings=embeddings, documents=batch_texts, metadatas=batch_metadatas)

        print(f"\nFinished ingestion. {collection.count()} documents stored.")

    except FileNotFoundError:
        print(f"File not found: {CORPUS_PATH}")

if __name__ == "__main__":
    main()