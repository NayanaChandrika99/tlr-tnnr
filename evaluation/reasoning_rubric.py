"""
Rubric-based reasoning quality scorer (Phase 3).

Defines the rubric and interface for scoring reasoning quality using an LLM.
"""

import json
from typing import Dict, List, Any, Optional

# Define the rubric from the plan
MEDICAL_CODING_RUBRIC = {
    "identifies_chief_complaint": {
        "weight": 10,
        "description": "Clearly identifies the main reason for the encounter",
    },
    "extracts_key_symptoms": {
        "weight": 15,
        "description": "Lists relevant clinical findings and symptoms",
    },
    "matches_code_family": {
        "weight": 25,
        "description": "Correctly identifies the ICD-10/CPT chapter or family",
    },
    "verifies_specificity": {
        "weight": 20,
        "description": "Explains why specific code is chosen over broader options",
    },
    "validates_code_format": {
        "weight": 10,
        "description": "Confirms code follows proper format rules",
    },
    "provides_clear_reasoning": {
        "weight": 20,
        "description": "Reasoning is logical, step-by-step, and understandable",
    }
}

def create_grader_prompt(narrative: str, reasoning: str, code: str) -> str:
    return f"""You are evaluating the quality of medical coding reasoning.

Clinical Narrative:
{narrative}

Student's Reasoning:
{reasoning}

Assigned Code:
{code}

Rubric (Total: 100 points):
{json.dumps(MEDICAL_CODING_RUBRIC, indent=2)}

For each rubric criterion:
1. Score from 0 to the maximum weight
2. Provide brief justification

Return ONLY a valid JSON object with this format:
{{
    "identifies_chief_complaint": {{"score": X, "justification": "..."}},
    "extracts_key_symptoms": {{"score": X, "justification": "..."}},
    ... (other criteria) ...
    "total_score": X,
    "overall_feedback": "..."
}}
"""

class ReasoningScorer:
    def __init__(self, client=None, model="gpt-4o"):
        """
        Initialize with an OpenAI-compatible client.
        If client is None, will try to import openai and init from env.
        """
        if client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except ImportError:
                print("Warning: OpenAI client not found. Scoring will fail.")
                self.client = None
        else:
            self.client = client
        self.model = model

    def score(self, narrative: str, reasoning: str, code: str) -> Dict[str, Any]:
        if not self.client:
            return {"total_score": 0, "error": "No client"}
            
        prompt = create_grader_prompt(narrative, reasoning, code)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error scoring reasoning: {e}")
            return {"total_score": 0, "error": str(e)}

