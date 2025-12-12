import json
import statistics
import sys
import os

from rouge_score import rouge_scorer

PRED_FILE = "cloud_predictions_hybrid.json"
GOLD_FILE = "cloud_gold.jsonl"
SHOW_COUNT = 5

def normalize_text(text):
    if not text: return ""
    return text.lower().strip().replace("\n", " ")

def get_key(item):
    c_id = item.get('conversation_id', 'unknown')
    # Handle variations in turn key naming
    t_id = item.get('turn') or item.get('turn_id') or '0'
    return f"{c_id}_{t_id}"

def get_question_text(item):
    inputs = item.get('input', [])
    if isinstance(inputs, list) and inputs:
        return inputs[-1].get('text', 'Unknown Question')
    return item.get('question', 'Unknown Question')

def evaluate():
    print(f"Loading gold standards from {GOLD_FILE}...")
    gold_map = {}
    if not os.path.exists(GOLD_FILE):
        print(f"Error: {GOLD_FILE} not found.")
        sys.exit(1)

    with open(GOLD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                key = get_key(data)
                targets = data.get('targets', [])
                if targets:
                    gold_map[key] = targets[0].get('text', "")
            except json.JSONDecodeError:
                continue

    # 2. Load Predictions
    print(f"Loading predictions from {PRED_FILE}...")
    if not os.path.exists(PRED_FILE):
        print(f"Error: {PRED_FILE} not found. Run the inference script first.")
        sys.exit(1)

    with open(PRED_FILE, 'r', encoding='utf-8') as f:
        prediction_list = json.load(f)

    # 3. Evaluate Intersection
    print(f"Evaluating valid predictions within the subset...")
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_1_scores = []
    rouge_l_scores = []
    exact_matches = 0
    valid_samples = 0
    skipped_errors = 0

    for pred_item in prediction_list:
        key = get_key(pred_item)
        pred_text = pred_item.get('model_prediction', "")

        if not pred_text or pred_text == "Error":
            skipped_errors += 1
            continue
        
        # Ensure we have a ground truth for this prediction
        if key not in gold_map:
            continue

        valid_samples += 1
        target_text = gold_map[key]
        
        p = normalize_text(pred_text)
        t = normalize_text(target_text)

        if p == t:
            exact_matches += 1

        scores = scorer.score(t, p)
        rouge_1_scores.append(scores['rouge1'].fmeasure)
        rouge_l_scores.append(scores['rougeL'].fmeasure)

    # 4. Report Results
    if valid_samples == 0:
        print("\nWARNING: No valid samples found to evaluate.")
        print(f"Skipped {skipped_errors} items marked as 'Error' or empty.")
        return

    print("\n" + "="*30)
    print(f" EVALUATION RESULTS (n={valid_samples})")
    print("="*30)
    print(f"Subset Size:      {len(prediction_list)}")
    print(f"Valid Answers:    {valid_samples}")
    print(f"Skipped Errors:   {skipped_errors}")
    print("-" * 30)
    print(f"Exact Match:      {exact_matches / valid_samples:.2%}")
    print(f"ROUGE-1:          {statistics.mean(rouge_1_scores):.4f} (Word Overlap)")
    print(f"ROUGE-L:          {statistics.mean(rouge_l_scores):.4f} (Sentence Structure)")
    print("="*30)

if __name__ == "__main__":
    evaluate()
    
    # Visual Inspection (unchanged logic, just safer keys)
    print(f" VISUAL INSPECTION (First {SHOW_COUNT} Valid Items) ")
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    # Re-load for clean inspection logic or reuse variables if preferred
    with open(PRED_FILE, 'r', encoding='utf-8') as f:
        p_lines = json.load(f)
    
    gold_map_insp = {}
    with open(GOLD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            gold_map_insp[get_key(d)] = d.get('targets', [{}])[0].get('text', "")

    count = 0
    for pred in p_lines:
        if count >= SHOW_COUNT: break
        
        key = get_key(pred)
        pred_text = pred.get('model_prediction', "")
        
        if not pred_text or pred_text == "Error": continue
        if key not in gold_map_insp: continue

        ref_text = gold_map_insp[key]
        q_text = get_question_text(pred)
        
        scores = scorer.score(ref_text, pred_text)
        
        print(f"ID:   {key}")
        print(f"Q:    {q_text}")
        print(f"Ref:  {ref_text}")
        print(f"Pred: {pred_text}")
        print(f"SCORE -> R1: {scores['rouge1'].fmeasure:.2f} | RL: {scores['rougeL'].fmeasure:.2f}")
        print("-" * 60)
        count += 1