"""
Prepare Foundation Dataset for Phase 1 Training.

Combines extracted CMS examples and generated synthetic CoT examples,
validates them, and splits into training and validation sets.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

from utils.code_validator import CodeValidator

def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"Warning: {path} not found. Returning empty list.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def prepare_foundation_dataset(
    cms_path: Path,
    synthetic_path: Path,
    output_dir: Path,
    validator_db: Path
):
    print("Loading datasets...")
    cms_data = load_jsonl(cms_path)
    synthetic_data = load_jsonl(synthetic_path)
    
    print(f"Loaded {len(cms_data)} CMS examples and {len(synthetic_data)} synthetic examples.")
    
    combined = cms_data + synthetic_data
    if not combined:
        print("No data found! Please run extraction or generation scripts first.")
        return

    print("Validating dataset...")
    validator = CodeValidator(validator_db)
    validated = []
    skipped = 0
    
    seen_narratives = set()
    duplicates = 0
    
    for ex in combined:
        # Deduplication by narrative hash
        try:
            # Access narrative from user message
            narrative = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            narr_hash = hash(narrative)
            if narr_hash in seen_narratives:
                duplicates += 1
                continue
            seen_narratives.add(narr_hash)
            
            # Code validation
            meta = ex.get("_metadata", {})
            code = meta.get("code")
            code_type = meta.get("code_type")
            
            if code and code_type and validator.validate(code, code_type):
                validated.append(ex)
            else:
                skipped += 1
        except Exception as e:
            print(f"Error processing example: {e}")
            skipped += 1
            
    print(f"Validation complete. Kept: {len(validated)}, Skipped: {skipped}, Duplicates: {duplicates}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(validated)
    
    # Split
    split_idx = int(len(validated) * 0.95)
    train_data = validated[:split_idx]
    val_data = validated[split_idx:]
    
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    
    # Save
    save_jsonl(train_data, output_dir / "foundation_train.jsonl")
    save_jsonl(val_data, output_dir / "foundation_val.jsonl")
    
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cms-path", type=Path, default=Path("data/foundation/cms_examples_5k.jsonl"))
    parser.add_argument("--synthetic-path", type=Path, default=Path("data/foundation/synthetic_cot.jsonl")) # Adjusted name
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--validator-db", type=Path, default=Path("data/code_databases"))
    args = parser.parse_args()
    
    prepare_foundation_dataset(
        args.cms_path,
        args.synthetic_path,
        args.output_dir,
        args.validator_db
    )

if __name__ == "__main__":
    main()

