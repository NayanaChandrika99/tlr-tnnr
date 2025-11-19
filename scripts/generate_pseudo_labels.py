"""
Generate Pseudo-Labels for Phase 4 Self-Amplification.

Uses the current model to generate new training examples by:
1. Creating variations of existing narratives
2. Generating narratives for specific codes
3. Validating the outputs with the CodeValidator
"""

import argparse
import json
import random
import torch
from pathlib import Path
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.code_validator import CodeValidator

def load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def generate_pseudo_labels(
    model_path: str,
    seed_data_path: Path,
    target_count: int,
    validator_db: Path,
    output_path: Path
):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    
    print("Loading seed data...")
    seeds = load_jsonl(seed_data_path)
    validator = CodeValidator(validator_db)
    
    pseudo_labels = []
    attempts = 0
    max_attempts = target_count * 3
    
    print(f"Generating {target_count} pseudo-labels...")
    
    batch_size = 8  # Adjust based on GPU VRAM
    
    while len(pseudo_labels) < target_count and attempts < max_attempts:
        attempts += batch_size
        if len(pseudo_labels) % 50 < batch_size:
             print(f"  Generated: {len(pseudo_labels)}/{target_count} (Attempts: {attempts})")

        # Prepare batch
        prompts = []
        narratives = []
        seeds_batch = []
        
        for _ in range(batch_size):
             seed = random.choice(seeds)
             # Extract narrative
             narrative = next(m["content"] for m in seed["messages"] if m["role"] == "user")
             
             messages = [
                {"role": "system", "content": "You are a medical coding expert. Use <think> tags to show your reasoning."},
                {"role": "user", "content": narrative}
             ]
             prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
             
             prompts.append(prompt)
             narratives.append(narrative)
             seeds_batch.append(seed)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.8,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Process batch
            decoded_outputs = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            for i, generated_text in enumerate(decoded_outputs):
                if len(pseudo_labels) >= target_count:
                    break
                    
                # Parse Code
                import re
                match = re.search(r'(ICD-?10|CPT|HCPCS)\s*:?\s*([A-Z0-9.]+)', generated_text, re.IGNORECASE)
                
                if not match:
                    continue
                    
                pred_type_raw, pred_code = match.groups()
                pred_code = pred_code.strip().upper()
                
                # Normalize type
                type_map = {"ICD-10": "ICD10", "ICD": "ICD10", "CPT": "CPT", "HCPCS": "HCPCS"}
                pred_type = type_map.get(pred_type_raw.upper(), pred_type_raw.upper())
                
                # Validate
                if not validator.validate(pred_code, pred_type):
                    continue
                    
                # Extract reasoning
                think_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL)
                if not think_match:
                    continue
                
                reasoning = think_match.group(1).strip()
                if len(reasoning) < 50:
                    continue
                    
                # Success
                pseudo_labels.append({
                    "messages": [
                        {"role": "system", "content": "You are a medical coding expert. Use <think> tags to show your reasoning."},
                        {"role": "user", "content": narratives[i]},
                        {"role": "assistant", "content": generated_text}
                    ],
                    "_metadata": {
                        "code_type": pred_type,
                        "code": pred_code,
                        "reasoning_present": True,
                        "source": "self_generated",
                        "original_source": seeds_batch[i].get("_metadata", {}).get("source", "unknown")
                    }
                })
            
        except Exception as e:
            print(f"Error generating batch: {e}")
            continue

    save_jsonl(pseudo_labels, output_path)
    print(f"Saved {len(pseudo_labels)} pseudo-labels to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--seed-data", type=Path, required=True, help="Path to seed jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output path")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--validator-db", type=Path, default=Path("data/code_databases"))
    
    args = parser.parse_args()
    
    generate_pseudo_labels(
        args.model_path,
        args.seed_data,
        args.count,
        args.validator_db,
        args.output
    )

if __name__ == "__main__":
    main()

