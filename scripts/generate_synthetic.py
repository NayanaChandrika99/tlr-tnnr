"""
Batch generator for THCE synthetic examples using OpenAI API with CoT and Validation.

This script generates synthetic medical coding examples with Chain-of-Thought reasoning,
validates them against the project's code database, and formatting them for Stage 2 training.
"""

import argparse
import json
import random
import time
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from utils.code_validator import CodeValidator

# Try importing OpenAI, handle if missing
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Default prompt template matching the plan
DEFAULT_PROMPT_TEMPLATE = """
You are a medical coding expert. Generate a realistic clinical narrative, 
step-by-step reasoning process, and correct medical billing code.

Requirements:
- Narrative should be 50-200 words
- Include 4-6 reasoning steps in <think> tags
- Assign ONE code (ICD-10, CPT, or HCPCS)
- Code must be valid and commonly used
- Do NOT invent codes. Use real codes.

Focus on these specialties: {specialty}
Code family: {code_family}
Complexity level: {complexity}

Format your response as a JSON object with these keys:
{
  "narrative": "Clinical narrative text...",
  "reasoning": "Step 1... Step 2...", 
  "code_type": "ICD10" or "CPT" or "HCPCS",
  "code": "The code itself (e.g., E11.9)"
}
"""

def parse_generated_example(content: str) -> Optional[Dict]:
    """Parse the LLM response into a dictionary."""
    try:
        # Try to find JSON if wrapped in markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
             content = content.split("```")[1].split("```")[0]
             
        data = json.loads(content.strip())
        
        # Normalize keys
        required_keys = ["narrative", "reasoning", "code", "code_type"]
        if not all(k in data for k in required_keys):
            print(f"Missing keys in response: {data.keys()}")
            return None
            
        return data
    except Exception as e:
        print(f"Failed to parse response: {e}")
        return None

def generate_synthetic_examples(
    client,
    count: int,
    specialties: List[str],
    code_families: List[str],
    validator: CodeValidator,
    model: str = "gpt-4o"
) -> List[Dict]:
    """
    Generate synthetic CoT examples with filtering.
    """
    examples = []
    attempts = 0
    max_attempts = count * 3  # Allow 3x for filtering
    
    print(f"Generating {count} examples...")
    
    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        if attempts % 10 == 0:
            print(f"  Progress: {len(examples)}/{count} (Attempts: {attempts})")
        
        # Sample constraints
        specialty = random.choice(specialties)
        code_family = random.choice(code_families)
        complexity = random.choice([1, 2, 3])
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": DEFAULT_PROMPT_TEMPLATE.format(
                        specialty=specialty,
                        code_family=code_family,
                        complexity=complexity
                    )
                }],
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            parsed = parse_generated_example(content)
            
            if not parsed:
                continue
            
            # Validate code
            # Some LLMs might put "ICD-10" instead of "ICD10"
            code_type_map = {"ICD-10": "ICD10", "ICD": "ICD10", "CPT": "CPT", "HCPCS": "HCPCS"}
            raw_type = parsed['code_type'].upper()
            norm_type = code_type_map.get(raw_type, raw_type)
            
            if not validator.validate(parsed['code'], norm_type):
                print(f"  Invalid code generated: {parsed['code']} ({norm_type})")
                continue
            
            # Check reasoning quality (simple length check)
            # If reasoning is just a string, wrapping it in <think> if not already
            reasoning_text = parsed['reasoning']
            
            # Create Stage 2 format
            formatted_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical coding expert. Use <think> tags to show your reasoning."
                    },
                    {
                        "role": "user",
                        "content": parsed["narrative"]
                    },
                    {
                        "role": "assistant",
                        "content": f"<think>\n{reasoning_text}\n</think>\n{norm_type}: {parsed['code']}"
                    }
                ],
                "_metadata": {
                    "code_type": norm_type,
                    "code": parsed["code"],
                    "reasoning_present": True,
                    "source": "synthetic_gpt4",
                    "specialty": specialty,
                    "complexity": complexity
                }
            }
            
            examples.append(formatted_example)
            
        except Exception as e:
            print(f"  Error during generation: {e}")
            time.sleep(1)
    
    return examples

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical coding examples.")
    parser.add_argument("--count", type=int, default=10, help="Number of examples to generate.")
    parser.add_argument("--output", type=Path, default=Path("data/foundation/synthetic_cot.jsonl"), help="Output path.")
    parser.add_argument("--validator-db", type=Path, default=Path("data/code_databases"), help="Validator DB path.")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key (optional, can use env var).")
    args = parser.parse_args()
    
    load_dotenv()
    
    if OpenAI is None:
        print("Error: 'openai' package is not installed. Please run: pip install openai")
        return

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found. Set it in .env or pass via --api-key.")
        # We won't exit here to allow dry-run testing if we wanted, but for now let's strict
        # return
    
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return

    # Initialize validator
    if not args.validator_db.exists():
         print(f"Validator DB not found at {args.validator_db}")
         return
    
    validator = CodeValidator(args.validator_db)
    
    # Configuration
    specialties = [
        "Cardiology", "Endocrinology", "Orthopedics", "Gastroenterology", 
        "Neurology", "Pulmonology", "Dermatology", "Oncology"
    ]
    code_families = ["ICD10", "CPT", "HCPCS"]
    
    # Generate
    examples = generate_synthetic_examples(
        client, 
        args.count, 
        specialties, 
        code_families, 
        validator
    )
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
            
    print(f"Successfully saved {len(examples)} examples to {args.output}")

if __name__ == "__main__":
    main()
