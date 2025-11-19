"""
Generate thought-level quality scores for Phase 3 training.

Uses the ReasoningScorer to grade examples and save them with scores.
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from evaluation.reasoning_rubric import ReasoningScorer
from utils.code_validator import CodeValidator

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=Path, required=True, help="JSONL file with 'messages' (CoT data)")
    parser.add_argument("--output-data", type=Path, required=True, help="Output JSONL with scores")
    parser.add_argument("--limit", type=int, default=100, help="Limit to save costs")
    args = parser.parse_args()
    
    print("Initializing scorer...")
    scorer = ReasoningScorer() # Uses env OPENAI_API_KEY
    
    data = load_jsonl(args.input_data)
    if args.limit:
        data = data[:args.limit]
    
    scored_data = []
    
    print(f"Scoring {len(data)} examples...")
    for item in tqdm(data):
        try:
            # Extract components
            messages = item["messages"]
            narrative = next(m["content"] for m in messages if m["role"] == "user")
            assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
            
            # Parse <think> and code
            import re
            think_match = re.search(r'<think>(.*?)</think>', assistant_msg, re.DOTALL)
            reasoning = think_match.group(1).strip() if think_match else ""
            
            # Code is at the end
            code_part = assistant_msg.split("</think>")[-1].strip() if "</think>" in assistant_msg else ""
            
            if not reasoning: 
                continue
                
            # Score
            score_result = scorer.score(narrative, reasoning, code_part)
            
            # Add score to item
            item["reasoning_score"] = score_result.get("total_score", 0)
            item["reasoning_feedback"] = score_result
            
            scored_data.append(item)
            
        except Exception as e:
            print(f"Skipping item due to error: {e}")
            continue
            
    # Save
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_data, "w", encoding="utf-8") as f:
        for item in scored_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(scored_data)} scored examples to {args.output_data}")

if __name__ == "__main__":
    main()

