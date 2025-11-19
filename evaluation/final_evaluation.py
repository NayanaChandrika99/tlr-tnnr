"""
Comprehensive Final Evaluation Script.

Evaluates a model across multiple test sets:
- Validation Set
- CMS Holdout (if available)
- Synthetic Hard (if available)

Metrics:
- Exact Match
- Code Validity
- Family Match
- Reasoning Coverage
- Inference Time
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.code_validator import CodeValidator

def load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

class Evaluator:
    def __init__(self, model_path: str, validator_db: Path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            device_map="auto"
        )
        self.validator = CodeValidator(validator_db)
        
    def generate(self, prompt: str) -> tuple[str, float]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0, # Greedy for eval
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        end_time = time.time()
        
        output_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return output_text, (end_time - start_time)

    def extract_code(self, text: str) -> tuple[str, str]:
        import re
        # Look for Type: Code pattern
        match = re.search(r'(ICD-?10|CPT|HCPCS)\s*:?\s*([A-Z0-9.]+)', text, re.IGNORECASE)
        if match:
            t, c = match.groups()
            # Normalize type
            type_map = {"ICD-10": "ICD10", "ICD": "ICD10", "CPT": "CPT", "HCPCS": "HCPCS"}
            return type_map.get(t.upper(), t.upper()), c.strip().upper()
        return "", ""

    def evaluate_dataset(self, dataset_path: Path, name: str) -> Dict[str, Any]:
        print(f"\nEvaluating on {name} ({dataset_path})...")
        data = load_jsonl(dataset_path)
        
        results = {
            "exact_match": 0,
            "valid_code": 0,
            "family_match": 0,
            "total": 0,
            "inference_times": [],
            "reasoning_present": 0
        }
        
        predictions = []
        
        for item in tqdm(data):
            # Construct prompt
            msgs = [m for m in item["messages"] if m["role"] != "assistant"]
            prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            
            output, infer_time = self.generate(prompt)
            
            # Ground Truth
            gt_meta = item.get("_metadata", {})
            gt_code = gt_meta.get("code", "").upper()
            gt_type = gt_meta.get("code_type", "").upper()
            
            # Prediction
            pred_type, pred_code = self.extract_code(output)
            
            # Metrics
            results["total"] += 1
            results["inference_times"].append(infer_time)
            
            is_exact = (pred_code == gt_code) and (pred_type == gt_type)
            is_valid = self.validator.validate(pred_code, pred_type) if pred_code else False
            is_family = (pred_code[:3] == gt_code[:3]) if pred_code and gt_code else False
            has_reasoning = "<think>" in output or len(output.split('\n')) > 2 # Heuristic
            
            if is_exact: results["exact_match"] += 1
            if is_valid: results["valid_code"] += 1
            if is_family: results["family_match"] += 1
            if has_reasoning: results["reasoning_present"] += 1
            
            predictions.append({
                "prompt": prompt,
                "prediction": output,
                "ground_truth_code": gt_code,
                "predicted_code": pred_code,
                "exact_match": is_exact,
                "valid": is_valid
            })
            
        # Summary
        total = results["total"] or 1
        summary = {
            "dataset": name,
            "exact_match_pct": results["exact_match"] / total,
            "valid_code_pct": results["valid_code"] / total,
            "family_match_pct": results["family_match"] / total,
            "reasoning_presence_pct": results["reasoning_present"] / total,
            "avg_inference_time": np.mean(results["inference_times"]) if results["inference_times"] else 0.0
        }
        
        print(f"Results for {name}:")
        print(json.dumps(summary, indent=2))
        
        return summary, predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--validator-db", type=Path, default=Path("data/code_databases"))
    parser.add_argument("--test-sets", nargs="+", help="List of test set paths", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_eval"))
    args = parser.parse_args()
    
    # Default test sets if none provided
    test_sets = args.test_sets
    if not test_sets:
        # Try to find standard sets
        candidates = [
            Path("data/processed/foundation_val.jsonl"),
            Path("data/test/cms_holdout.jsonl")
        ]
        test_sets = [p for p in candidates if p.exists()]
    
    if not test_sets:
        print("No test sets found or provided.")
        return

    evaluator = Evaluator(args.model_path, args.validator_db)
    
    final_report = {}
    
    for ts in test_sets:
        ts_path = Path(ts)
        name = ts_path.stem
        summary, preds = evaluator.evaluate_dataset(ts_path, name)
        final_report[name] = summary
        
        # Save predictions
        save_jsonl(preds, args.output_dir / f"{name}_predictions.jsonl")
    
    # Save final report
    with open(args.output_dir / "eval_summary.json", "w") as f:
        json.dump(final_report, f, indent=2)
        
    print(f"\nEvaluation complete. Report saved to {args.output_dir}")

if __name__ == "__main__":
    main()

