On-device RAG using [IBM's Granite](https://huggingface.co/collections/ibm-granite/granite-40-language-models) and [IBM's Benchmark](https://github.com/IBM/mt-rag-benchmark).  
1B and 7B LLM tested, 30M Embedding Model.  
1B shows rouge score of 22% while only using 2.5GB of RAM.  
7B shows rouge score of 23% while using 5GB of RAM.  

With hybrid search:  
The system runs significantly slower. Only ran it on 10% of the test set.  
Rouge score of 17%. 1B model probably too small to handle such tasks and gets distracted.   

Retrieval w/o LLM, hybrid retrieval with reranking (BEIR nfcorpus):  
NDCG@10:   0.3219  