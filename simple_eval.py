import json
import statistics
import sys

from rouge_score import rouge_scorer

PRED_FILE = "cloud_predictions.json"
GOLD_FILE = "cloud_gold.jsonl"
SHOW_COUNT = 5

def normalize_text(text):
    if not text: return ""
    return text.lower().strip().replace("\n", " ")

def get_key(item):
    c_id = item.get('conversation_id', 'unknown')
    t_id = item.get('turn') or item.get('turn_id') or '0'
    return f"{c_id}_{t_id}"

def get_question_text(item):
    inputs = item.get('input', [])
    if isinstance(inputs, list) and inputs:
        return inputs[-1].get('text', 'Unknown Question')
    return item.get('question', 'Unknown Question')

def evaluate():
    print(f"Loading predictions from {PRED_FILE}...")
    preds = {}
    with open(PRED_FILE, 'r', encoding='utf-8') as f:
        prediction_list = json.load(f) 
    
    for data in prediction_list: 
        key = f"{data.get('conversation_id')}_{data.get('turn_id')}"
        preds[key] = data.get('model_prediction', "")

    print(f"Loading gold standards from {GOLD_FILE}...")
    golds = []
    with open(GOLD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            key = f"{data.get('conversation_id')}_{data.get('turn_id')}"
            
            if key in preds:
                target_text = ""
                targets = data.get('targets', [])
                if targets:
                    target_text = targets[0].get('text', "")
                
                golds.append({
                    "prediction": normalize_text(preds[key]),
                    "target": normalize_text(target_text)
                })

    if not golds:
        print("Error: No matching conversation_id/turn_id found between files.")
        print("Did you run the 'cloud_gold.jsonl' creation step?")
        sys.exit(1)

    print(f"Evaluating {len(golds)} samples...")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_1_scores = []
    rouge_l_scores = []
    exact_matches = 0

    for item in golds:
        p = item['prediction']
        t = item['target']

        if p == t:
            exact_matches += 1

        scores = scorer.score(t, p)
        rouge_1_scores.append(scores['rouge1'].fmeasure)
        rouge_l_scores.append(scores['rougeL'].fmeasure)

    print("\n" + "="*30)
    print(" EVALUATION RESULTS")
    print("="*30)
    print(f"Exact Match:  {exact_matches / len(golds):.2%}")
    print(f"ROUGE-1:      {statistics.mean(rouge_1_scores):.4f} (Word Overlap)")
    print(f"ROUGE-L:      {statistics.mean(rouge_l_scores):.4f} (Sentence Structure)")
    print("="*30)
    
    if statistics.mean(rouge_1_scores) < 0.1:
        print("WARNING: Scores are very low.")
        print("Check: Did the model output empty strings or error messages?")

if __name__ == "__main__":
    evaluate()
    print(f" VISUAL INSPECTION (First {SHOW_COUNT} Items) ")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    with open(PRED_FILE, 'r', encoding='utf-8') as p, \
        open(GOLD_FILE, 'r', encoding='utf-8') as g:
        
        p_lines = json.load(p) 
        
        g_lines = [json.loads(line) for line in g]
        
        gold_map = {get_key(x): x for x in g_lines}

        print(f"Loaded {len(p_lines)} predictions.\n")

        for i, pred in enumerate(p_lines[:SHOW_COUNT]):
            unique_id = get_key(pred)
            gold = gold_map.get(unique_id)
            
            if gold:
                q_text = get_question_text(pred)
                
                targets = gold.get('targets', [])
                ref_text = targets[0].get('text', "") if targets else ""
                
                pred_text = pred.get('model_prediction', "")
                
                scores = scorer.score(ref_text, pred_text)
                r1 = scores['rouge1'].fmeasure
                rl = scores['rougeL'].fmeasure
                
                print(f"ID:   {unique_id}")
                print(f"Q:    {q_text}")
                print(f"Ref:  {ref_text}")
                print(f"Pred: {pred_text}")
                print(f"SCORE -> ROUGE-1: {r1:.2f} | ROUGE-L: {rl:.2f}")
                print("-" * 60)
            else:
                print(f"Task {unique_id} not found in gold file.")
