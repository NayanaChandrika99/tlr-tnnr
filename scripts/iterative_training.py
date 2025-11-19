"""
Orchestrate multi-iteration self-training (Phase 4).

Loop:
1. Generate Pseudo-Labels (using current model)
2. Filter & Merge with Foundation Data
3. Train New Model (RLVR)
"""

import argparse
import subprocess
import sys
import shutil
import json
import yaml
from pathlib import Path

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def run_command(cmd):
    print(f"Running: {' '.join(str(x) for x in cmd)}")
    subprocess.check_call([str(x) for x in cmd], stdout=sys.stdout, stderr=sys.stderr)

def run_iterative_training(
    foundation_model: str,
    foundation_data: Path,
    iterations: int,
    pseudo_count_schedule: list,
    config_template: Path,
    validator_db: Path
):
    current_model = foundation_model
    
    # Ensure working directories
    Path("data/pseudo").mkdir(parents=True, exist_ok=True)
    Path("data/processed_iterative").mkdir(parents=True, exist_ok=True)
    Path("outputs/iterative").mkdir(parents=True, exist_ok=True)
    
    # Load foundation data once
    foundation_dataset = load_jsonl(foundation_data)
    
    for i in range(1, iterations + 1):
        print(f"\n=== Iteration {i} ===")
        iter_output_dir = Path(f"outputs/iterative/iter_{i}")
        pseudo_data_path = Path(f"data/pseudo/iter_{i}.jsonl")
        train_data_path = Path(f"data/processed_iterative/iter_{i}_train.jsonl")
        config_path = Path(f"outputs/iterative/iter_{i}_config.yaml")
        
        target_count = pseudo_count_schedule[i-1] if i <= len(pseudo_count_schedule) else pseudo_count_schedule[-1]
        
        # 1. Generate Pseudo-Labels
        print(f"Generating {target_count} pseudo-labels...")
        run_command([
            sys.executable, "scripts/generate_pseudo_labels.py",
            "--model-path", current_model,
            "--seed-data", foundation_data, # Use foundation data as seed narratives
            "--output", pseudo_data_path,
            "--count", target_count,
            "--validator-db", validator_db
        ])
        
        # 2. Merge Data
        print("Merging datasets...")
        pseudo_data = load_jsonl(pseudo_data_path)
        merged_data = foundation_dataset + pseudo_data
        save_jsonl(merged_data, train_data_path)
        print(f"Total training examples for Iteration {i}: {len(merged_data)}")
        
        # 3. Create Config for this iteration
        with open(config_template, 'r') as f:
            config = yaml.safe_load(f)
            
        config["model"] = current_model
        config["dataset"] = str(train_data_path)
        config["output_dir"] = str(iter_output_dir)
        config["run_name"] = f"thce-iter-{i}"
        
        iter_output_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        # 4. Train (RLVR)
        print(f"Training Iteration {i} model...")
        run_command([
            sys.executable, "post_training/rlvr.py",
            "--config", config_path,
            "--validator-db", validator_db
        ])
        
        # Update current model to the newly trained one
        current_model = str(iter_output_dir / "final")
        print(f"Iteration {i} complete. New model: {current_model}")

    print("\nAll iterations complete!")
    print(f"Final model: {current_model}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation-model", type=str, required=True, help="Path to foundation SFT model")
    parser.add_argument("--foundation-data", type=Path, default=Path("data/processed/foundation_train.jsonl"))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--config-template", type=Path, default=Path("post_training/config/rlvr.yaml"))
    parser.add_argument("--validator-db", type=Path, default=Path("data/code_databases"))
    
    args = parser.parse_args()
    
    # Default schedule from plan: 30k, 50k, 80k
    schedule = [30000, 50000, 80000]
    
    run_iterative_training(
        args.foundation_model,
        args.foundation_data,
        args.iterations,
        schedule,
        args.config_template,
        args.validator_db
    )

if __name__ == "__main__":
    main()

