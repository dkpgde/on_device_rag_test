# On-Device RAG with IBM Granite

This prototype evaluates small IBM Granite models in a local retrieval-augmented generation pipeline; the saved experiment notes show similar ROUGE scores for 1B and 7B models, with substantially lower memory use for 1B.

## Research question

How much retrieval and answer-generation quality can a small, fully local Granite stack provide under consumer-hardware memory constraints?

## Approach

The conversational experiment uses the Cloud domain of IBM's MTRAG benchmark. Documents are embedded with `ibm-granite/granite-embedding-30m-english` and stored in Chroma. The hybrid path combines dense retrieval and BM25 with reciprocal-rank fusion, retrieves five passages, and generates answers through Granite served by Ollama.

The repository also contains a retrieval-only experiment on BEIR NFCorpus. That path combines Granite 30M embeddings with BM25, retrieves 50 documents, reranks to 10 with `BAAI/bge-reranker-base`, and evaluates NDCG@10 and Recall@10.

## Results

These headline values are preserved from the original experiment notes. The ROUGE variant and per-example outputs were not retained, so they should be treated as indicative rather than as a fully auditable leaderboard.

| Experiment | Model / retriever | Data evaluated | Recorded result | Peak RAM |
|---|---|---|---:|---:|
| MTRAG generation | Granite 1B | Cloud tasks | ROUGE 22% | 2.5 GB |
| MTRAG generation | Granite 7B | Cloud tasks | ROUGE 23% | 5 GB |
| MTRAG hybrid generation | Granite 1B + dense/BM25 fusion | 10% Cloud subset | ROUGE 17% | not recorded |
| BEIR retrieval | Granite 30M + BM25 + BGE reranker | NFCorpus test split | NDCG@10 0.3219 | not recorded |

## Interpretation

The 1B model delivered nearly the same recorded answer-overlap score as the 7B model at roughly half the memory use. In the limited hybrid run, adding more retrieved context reduced the score and increased latency, which is consistent with the small generator struggling to use noisier context. The experiment does not isolate whether the drop came from retrieval quality, prompt construction, context length, or model capacity.

## Reproduction

### Conversational MTRAG run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull granite4:1b-h
```

Download the MTRAG `RAG.jsonl` tasks and Cloud corpus from IBM. Place `RAG.jsonl` in the repository root and the extracted Cloud documents in `data/`, then run:

```bash
python make_eval_file.py
python run_benchmark.py
python simple_eval.py
```

`rag.py` starts an interactive hybrid-RAG prompt and is currently configured for `granite4:3b-h`.

### BEIR NFCorpus retrieval run

Download NFCorpus in BEIR format into `data/`, pull `ibm/granite-embedding:30m` in Ollama, then build the index and run the benchmark:

```bash
ollama pull ibm/granite-embedding:30m
python vectorize.py
python run_benchmark_beir.py
```

## Limitations

- Benchmark task files, corpora, generated predictions, and stored indexes are not included.
- The exact 7B configuration and its output artifact are not preserved in the current scripts.
- The saved notes do not identify whether “ROUGE” means ROUGE-1 or ROUGE-L.
- The hybrid generation result uses only a 10% subset and is not directly comparable to a full-dataset run.
- Latency and RAM figures depend on hardware, quantization, Ollama, and model versions.
- `requirements.txt` is an environment snapshot rather than a minimal dependency specification.

## Attribution

- IBM [MTRAG benchmark](https://github.com/IBM/mt-rag-benchmark) and [paper](https://arxiv.org/abs/2501.03468)
- IBM [Granite language models](https://huggingface.co/collections/ibm-granite/granite-40-language-models) and [Granite embedding models](https://huggingface.co/collections/ibm-granite/granite-embedding-models)
- [BEIR](https://github.com/beir-cellar/beir) and its NFCorpus benchmark
- [LlamaIndex](https://github.com/run-llama/llama_index), [Chroma](https://github.com/chroma-core/chroma), and [Ollama](https://ollama.com/)
