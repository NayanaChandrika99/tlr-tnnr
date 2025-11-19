"""
Extract training examples from CMS Official Guidelines.

This script parses text files in data/raw/cms/ and extracts medical coding examples
following the format found in CMS guidelines (Example -> Rationale -> Code).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from utils.code_validator import CodeValidator


def extract_cms_examples(guideline_text: str) -> List[Dict]:
    """
    Parse CMS guidelines for coding examples.
    
    Expected format loosely follows:
    Example: [clinical narrative]
    Rationale: [reasoning steps]
    Code Assignment: [ICD10/CPT/HCPCS code]
    """
    examples = []
    
    # Regex to capture Example ... Rationale ... Code patterns.
    # We make it flexible to handle variations in headers.
    pattern = (
        r"(?:Example|Case Study)\s*:\s*(.+?)\s*"
        r"(?:Rationale|Reasoning|Explanation)\s*:\s*(.+?)\s*"
        r"(?:Code Assignment|Code|Assign)\s*:\s*(.+?)"
        r"(?=\n\n(?:Example|Case Study)|\Z)"
    )
    
    for match in re.finditer(pattern, guideline_text, re.DOTALL | re.IGNORECASE):
        narrative = match.group(1).strip()
        reasoning = match.group(2).strip()
        code_assignment = match.group(3).strip()
        
        # Try to extract the specific code and type
        # Looking for patterns like "ICD-10: E11.9" or just "E11.9"
        
        # Simple heuristic for code extraction from the assignment string
        # We'll rely on the Validator to confirm validity later
        code_match = re.search(r'(?:ICD-?10|CPT|HCPCS)?\s*:?\s*([A-Z0-9.]+)', code_assignment, re.IGNORECASE)
        
        if not code_match:
            continue
            
        raw_code = code_match.group(1).upper()
        
        # Determine code type if explicit, otherwise infer
        code_type = "ICD10" # Default assumption for CMS guidelines usually
        if "CPT" in code_assignment.upper():
            code_type = "CPT"
        elif "HCPCS" in code_assignment.upper():
            code_type = "HCPCS"
            
        examples.append({
            "narrative": narrative,
            "reasoning": reasoning,
            "code": raw_code,
            "code_type": code_type,
            "source": "CMS_Guidelines",
            "quality": "high"
        })
    
    return examples


def convert_to_stage2_format(examples: List[Dict], validator: Optional[CodeValidator] = None) -> List[Dict]:
    """Convert extracted examples to Stage 2 training format."""
    formatted = []
    
    for ex in examples:
        # Validate if validator provided
        if validator:
            # Try to infer type if not strictly known, or use what we parsed
            # If explicit type failed, try inference
            if not validator.validate(ex["code"], ex["code_type"]):
                # Try to infer
                inferred = validator.infer_code_type(ex["code"])
                if inferred and validator.validate(ex["code"], inferred):
                    ex["code_type"] = inferred
                else:
                    # Skip invalid codes
                    continue

        formatted.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical coding expert. Use <think> tags to show your reasoning."
                },
                {
                    "role": "user",
                    "content": ex["narrative"]
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{ex['reasoning']}\n</think>\n{ex['code_type']}: {ex['code']}"
                }
            ],
            "_metadata": {
                "code_type": ex["code_type"],
                "code": ex["code"],
                "reasoning_present": True,
                "source": ex["source"],
                "quality": ex["quality"]
            }
        })
    
    return formatted


def main():
    parser = argparse.ArgumentParser(description="Extract CMS guideline examples.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/cms"), help="Directory containing text files.")
    parser.add_argument("--output", type=Path, default=Path("data/foundation/cms_examples_5k.jsonl"), help="Output JSONL file.")
    parser.add_argument("--validator-db", type=Path, default=Path("data/code_databases"), help="Path to code validator databases.")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Input directory {args.input_dir} does not exist. Creating it.")
        args.input_dir.mkdir(parents=True, exist_ok=True)
        print(f"Please place text versions of CMS guidelines in {args.input_dir}")
        return

    # Initialize validator
    try:
        validator = CodeValidator(args.validator_db)
    except Exception as e:
        print(f"Warning: Could not initialize validator ({e}). Skipping validation.")
        validator = None

    all_examples = []
    files = list(args.input_dir.glob("*.txt"))
    
    if not files:
        print(f"No .txt files found in {args.input_dir}.")
        return

    print(f"Processing {len(files)} files...")
    for txt_file in files:
        content = txt_file.read_text(encoding="utf-8", errors="replace")
        examples = extract_cms_examples(content)
        print(f"  Found {len(examples)} examples in {txt_file.name}")
        all_examples.extend(examples)

    formatted_data = convert_to_stage2_format(all_examples, validator)
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(formatted_data)} valid examples to {args.output}")


if __name__ == "__main__":
    main()

