"""
RLVR (Reinforcement Learning with Verifiable Rewards) Training Script.

This script trains the model using PPO (Proximal Policy Optimization) with a 
rule-based reward function (CodeValidator). It optimizes the model to generate 
valid and correct medical codes without needing human preference pairs.
"""

import argparse
import re
import yaml
from typing import List, Dict, Optional
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from utils.code_validator import CodeValidator

# Reward Logic
class MedicalCodeVerifier:
    """Verifiable reward function for medical coding."""
    
    def __init__(self, validator: CodeValidator):
        self.validator = validator
    
    def __call__(self, prompts: List[str], completions: List[str], ground_truths: List[Dict]) -> List[float]:
        rewards = []
        for completion, gt_meta in zip(completions, ground_truths):
            reward = self.compute_reward(completion, gt_meta)
            rewards.append(reward)
        return rewards

    def compute_reward(self, output: str, ground_truth: Dict) -> float:
        """
        Compute verifiable reward based on code correctness.
        
        Rewards:
            1.0: Perfect match (valid and correct)
            0.5: Valid code, wrong specificity (e.g. E11 vs E11.9)
            0.3: Valid code, correct family but wrong code
            0.1: Valid code, but completely wrong
            0.0: Invalid code or no code found
        """
        # Extract code from output (looking for last occurrence or specific pattern)
        # Pattern: Code: [Type]: [Code] or just the code at the end
        # We look for the pattern defined in our prompts: "ICD10: E11.9"
        
        # Regex to find "Type: Code"
        match = re.search(r'(ICD-?10|CPT|HCPCS)\s*:?\s*([A-Z0-9.]+)', output, re.IGNORECASE)
        
        if not match:
            return 0.0
            
        pred_type_raw, pred_code = match.groups()
        pred_code = pred_code.strip().upper()
        
        # Normalize type
        type_map = {"ICD-10": "ICD10", "ICD": "ICD10", "CPT": "CPT", "HCPCS": "HCPCS"}
        pred_type = type_map.get(pred_type_raw.upper(), pred_type_raw.upper())
        
        gt_code = ground_truth['code'].upper()
        gt_type = ground_truth['code_type'] # Should be normalized already
        
        # 1. Check Type
        if pred_type != gt_type:
            return 0.0
            
        # 2. Check Validity
        if not self.validator.validate(pred_code, pred_type):
            return 0.0
            
        # 3. Exact Match
        if pred_code == gt_code:
            return 1.0
            
        # 4. Specificity (ICD10 only)
        if pred_type == "ICD10":
            pred_base = pred_code.split('.')[0]
            gt_base = gt_code.split('.')[0]
            if pred_base == gt_base:
                return 0.5
        
        # 5. Family Match (First 3 chars)
        if pred_code[:3] == gt_code[:3]:
            return 0.3
            
        # 6. Valid but wrong
        return 0.1


def main():
    parser = argparse.ArgumentParser(description="RLVR Training with PPO")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--validator-db", type=Path, default=Path("data/code_databases"))
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load Validator
    validator = CodeValidator(args.validator_db)
    verifier = MedicalCodeVerifier(validator)

    # Config
    ppo_config = PPOConfig(
        model_name=config["model"],
        learning_rate=float(config["training"]["lr"]),
        batch_size=config["training"]["batch_size"],
        mini_batch_size=config["training"]["mini_batch_size"],
        gradient_accumulation_steps=config["training"]["grad_accum"],
        log_with="wandb",
        project_kwargs={"logging_dir": config["output_dir"]},
        is_peft_model=True, # Assuming LoRA or similar if 135M? Actually 135M is small enough for full finetune, but PPO usually benefits from PEFT
    )

    # Load Model (WithValueHead for PPO)
    # We use the foundation model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config["model"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for generation

    # Dataset
    # RLVR expects prompts. We need to load the foundation train set and extract prompts.
    dataset_path = config["dataset"]
    
    def build_dataset(path):
        data = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                # Construct prompt from messages (excluding assistant response)
                # We apply chat template to system+user
                msgs = [m for m in item["messages"] if m["role"] != "assistant"]
                prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                
                data.append({
                    "query": prompt,
                    "ground_truth": item["_metadata"]
                })
        return data

    raw_dataset = build_dataset(dataset_path)
    # Convert to HF Dataset
    from datasets import Dataset
    dataset = Dataset.from_list(raw_dataset)

    # Tokenize
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])
        return sample

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")

    # Collator
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None, # PPO will create ref model copy
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    # Training Loop
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 256, # Enough for reasoning + code
    }

    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]
        
        # 1. Generate
        response_tensors = ppo_trainer.generate(
            query_tensors, 
            return_prompt=False, 
            **generation_kwargs
        )
        
        batch["response"] = tokenizer.batch_decode(response_tensors)
        
        # 2. Compute Rewards
        rewards = verifier(
            prompts=batch["query"], # Not strictly needed for verifier but good for logging
            completions=batch["response"], 
            ground_truths=batch["ground_truth"]
        )
        
        # Convert to tensors
        reward_tensors = [torch.tensor(r) for r in rewards]
        
        # 3. Step
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
        
        # Log
        ppo_trainer.log_stats(stats, batch, reward_tensors)
        
        if epoch % config["training"]["save_steps"] == 0 and epoch > 0:
            ppo_trainer.save_pretrained(f"{config['output_dir']}/checkpoint-{epoch}")

    # Save final
    ppo_trainer.save_pretrained(f"{config['output_dir']}/final")

if __name__ == "__main__":
    main()

