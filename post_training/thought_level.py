"""
Phase 3: Thought-Level Quality Refinement Training.

This script implements a weighted SFT approach where training examples are 
weighted by their reasoning quality score (derived from the Rubric).
High-quality reasoning (score ~1.0) contributes more to the loss than 
lower-quality reasoning.
"""

import argparse
import yaml
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    tokenizer.pad_token = tokenizer.eos_token

    # Load Dataset with scores
    # Expected format: JSONL with 'messages' and 'reasoning_score' (float 0-1)
    dataset = load_dataset("json", data_files=config["dataset"], split="train")
    
    # Filter out examples without scores if necessary, or default to 1.0
    def add_weight(example):
        # Default weight 1.0 if missing
        w = example.get("reasoning_score", 1.0)
        # Optional: Apply temperature or scaling to weight
        # e.g. weight = w ** 2 to emphasize high scores more
        return {"weight": float(w)}

    dataset = dataset.map(add_weight)

    # Custom Trainer to handle Weighted Loss
    class WeightedSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Extract weights
            # Note: 'weight' must be added to the batch by the collator. 
            # SFTTrainer's default collator might not pass extra columns unless we tell it to.
            # However, a simpler way in HF Trainer is to use the "weight" column if available 
            # but we need to ensure it's tensorized.
            
            # We need to pop weights before passing to model
            weights = inputs.pop("weight", None)
            
            # Standard forward pass
            outputs = model(**inputs)
            
            # If we have weights, we need to recompute loss manually
            # because standard model output loss is already averaged/reduced.
            if weights is not None:
                # We need logits and labels
                logits = outputs.get("logits")
                labels = inputs.get("labels")
                
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten tokens
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                
                # Compute raw loss per token
                token_losses = loss_fct(shift_logits, shift_labels)
                
                # Reshape back to [batch, seq_len]
                token_losses = token_losses.view(weights.shape[0], -1)
                
                # Apply weights per sample (broadcast across sequence)
                # weights shape: [batch] -> [batch, 1]
                weighted_losses = token_losses * weights.unsqueeze(1)
                
                # Mask out ignored tokens (label -100)
                mask = (shift_labels.view(weights.shape[0], -1) != -100).float()
                
                # Average loss
                loss = weighted_losses.sum() / mask.sum()
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss

    # Data Collator
    # We need a collator that includes 'weight' in the batch
    # SFTTrainer uses DataCollatorForCompletionOnlyLM or default. 
    # We wrap it to handle the extra column.
    
    # First, let's verify if we need a response template for CompletionOnly
    response_template = "<|im_start|>assistant\n"
    base_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, 
        tokenizer=tokenizer
    )
    
    def weighted_collator(features):
        # Extract weights
        weights = [f.pop("weight") for f in features]
        batch = base_collator(features)
        # Add weights back as tensor
        batch["weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch

    # Training Args
    training_args = SFTConfig(**config["trainer"])
    
    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=weighted_collator,
    )

    trainer.train()
    trainer.save_model(config["output_dir"])

if __name__ == "__main__":
    main()

