import json

INPUT_FULL_FILE = "RAG.jsonl"       
OUTPUT_GOLD_FILE = "cloud_gold.jsonl" 

print("Filtering gold file for Cloud domain...")

count = 0
with open(INPUT_FULL_FILE, 'r', encoding='utf-8') as fin, \
     open(OUTPUT_GOLD_FILE, 'w', encoding='utf-8') as fout:
    
    for line in fin:
        if not line.strip(): continue
        try:
            data = json.loads(line)
            if "cloud" in data.get("Collection", "").lower():
                fout.write(line)
                count += 1
        except:
            continue

print(f"Created '{OUTPUT_GOLD_FILE}' with {count} tasks.")