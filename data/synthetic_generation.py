"""Generate lightweight synthetic samples for THCE development and testing."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from utils.code_validator import CodeValidator
from utils.medical_chat_templates import Stage

Condition = Tuple[str, str, Sequence[str]]


ICD10_CONDITIONS: List[Condition] = [
    (
        "Patient with stable type 2 diabetes presents for medication refill.",
        "E11.9",
        (
            "- Diagnosis explicitly says type 2 diabetes",
            "- No complications mentioned, so unspecified",
            "- E11.9 fits type 2 diabetes without complications",
        ),
    ),
    (
        "Follow-up visit for essential hypertension, no complications reported.",
        "I10",
        (
            "- Hypertension is primary",
            "- No heart or kidney involvement described",
            "- I10 captures essential (primary) hypertension",
        ),
    ),
    (
        "Outpatient visit for moderate persistent asthma with acute exacerbation.",
        "J45.41",
        (
            "- Narrative specifies moderate persistent asthma",
            "- Exacerbation mentioned, so use fifth digit .41",
            "- J45.41 maps to moderate persistent asthma with (acute) exacerbation",
        ),
    ),
    (
        "Patient evaluated for chronic kidney disease stage 3 secondary to diabetes.",
        "E11.22",
        (
            "- Diabetes type 2 with kidney complications requires combination code",
            "- Stage 3 CKD triggers E11.22",
            "- Ensure hypertensive component absent in narrative",
        ),
    ),
    (
        "Clinic follow-up for malignant neoplasm of left breast, estrogen receptor positive.",
        "C50.912",
        (
            "- Laterality (left breast) and unspecified site within breast",
            "- Estrogen receptor positivity noted, optional for metadata",
            "- C50.912 covers malignant neoplasm of unspecified site of left female breast",
        ),
    ),
    (
        "Emergency visit for closed displaced fracture of shaft of right tibia, initial encounter.",
        "S82.201A",
        (
            "- Traumatic fracture with laterality and encounter status specified",
            "- ICD-10 injury codes require seventh character A for initial encounter",
            "- S82.201A captures displaced fracture of shaft of right tibia, initial",
        ),
    ),
]

CPT_CONDITIONS: List[Condition] = [
    (
        "Clinic visit lasting 22 minutes for medication management of low back pain.",
        "99213",
        (
            "- Established patient visit",
            "- Time between 20 and 29 minutes",
            "- CPT 99213 matches these characteristics",
        ),
    ),
    (
        "Laboratory request for quantitative urine culture.",
        "87086",
        (
            "- Provider orders quantitative urine culture",
            "- CPT 87086 covers urine colony count",
        ),
    ),
    (
        "CT angiography of chest with contrast to rule out pulmonary embolism.",
        "71275",
        (
            "- CTA chest with contrast corresponds to CPT 71275",
            "- Used for evaluating pulmonary vasculature",
        ),
    ),
    (
        "Follow-up transthoracic echocardiogram complete study with Doppler and color flow.",
        "93306",
        (
            "- Complete TTE with Doppler/color is CPT 93306",
            "- Narrative indicates full study rather than limited exam",
        ),
    ),
    (
        "Outpatient colonoscopy with removal of polyps using snare technique.",
        "45385",
        (
            "- Polypectomy via snare uses CPT 45385",
            "- Ensure colonoscopy was complete to cecum",
        ),
    ),
    (
        "Physical therapy session providing neuromuscular re-education for balance training.",
        "97112",
        (
            "- Neuromuscular re-education is captured by CPT 97112",
            "- Applies to balance and coordination retraining",
        ),
    ),
]

HCPCS_CONDITIONS: List[Condition] = [
    (
        "Durable equipment order for rigid walker after hip surgery.",
        "E0130",
        (
            "- Walker requested is rigid, adjustable height",
            "- HCPCS E0130 matches walker description",
        ),
    ),
    (
        "Patient requires blood glucose test strips for home monitoring.",
        "A4253",
        (
            "- Supplies are glucose test strips",
            "- HCPCS A4253 covers 50 strip packs",
        ),
    ),
    (
        "Order for pneumatic compression device sleeves to treat chronic lymphedema.",
        "E0651",
        (
            "- Pneumatic compression device segments correspond to E0651",
            "- Narrative specifies lymphedema management",
        ),
    ),
    (
        "Patient needs hand-held shower spray for adaptive bathing after spinal surgery.",
        "E0242",
        (
            "- Hand-held shower spray qualifies as bathroom safety equipment",
            "- HCPCS E0242 reimburses this durable medical equipment",
        ),
    ),
    (
        "Chemotherapy infusion pump for continuous 5-FU delivery ordered for home use.",
        "E0781",
        (
            "- Ambulatory infusion pump for chemotherapy maps to E0781",
            "- Narrative indicates continuous infusion requirement",
        ),
    ),
    (
        "Disposable insulin pump supplies refill for insulin-dependent patient.",
        "A4225",
        (
            "- Refill supplies for external insulin infusion pump use A4225",
            "- Include narrative mention of pump therapy",
        ),
    ),
]


def random_condition(code_type: str) -> Condition:
    mapping = {
        "ICD10": ICD10_CONDITIONS,
        "CPT": CPT_CONDITIONS,
        "HCPCS": HCPCS_CONDITIONS,
    }
    return random.choice(mapping[code_type])


def stage_payload(stage: Stage, code_type: str, condition: Condition) -> Dict[str, object]:
    narrative, code, reasoning = condition
    if stage is Stage.STAGE_1:
        completion = f"{code_type}: {code}"
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical coding assistant. Convert clinical narratives to billing codes.",
                },
                {"role": "user", "content": narrative},
                {"role": "assistant", "content": completion},
            ],
            "_metadata": {"code_type": code_type, "code": code, "reasoning_present": False},
        }

    if stage is Stage.STAGE_2:
        reasoning_text = "\n".join(reasoning)
        completion = f"<think>\n{reasoning_text}\n</think>\n{code_type}: {code}"
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical coding expert. Use <think> tags to show your reasoning.",
                },
                {"role": "user", "content": narrative},
                {"role": "assistant", "content": completion},
            ],
            "_metadata": {"code_type": code_type, "code": code, "reasoning_present": True},
        }

    rejected_code = "00000" if code_type == "CPT" else "Z99.9"
    stage3_template = get_template(Stage.STAGE_3)
    return {
        "prompt": [
            {"role": "system", "content": stage3_template.system_prompt},
            {"role": "user", "content": narrative},
        ],
        "chosen": (
            "<think>\n" + "\n".join(reasoning) + f"\n</think>\n{code_type}: {code}"
        ),
        "rejected": (
            "<think>\nReasoning uncertain. Selected placeholder code.\n</think>\n"
            f"{code_type}: {rejected_code}"
        ),
        "_metadata": {"code_type": code_type, "code": code, "rejected_code": rejected_code},
    }


def generate_samples(
    count: int,
    stage: Stage,
    validator: CodeValidator,
    rng: random.Random,
) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    code_types = ["ICD10", "CPT", "HCPCS"]
    for _ in range(count):
        code_type = rng.choice(code_types)
        narrative, code, reasoning = random_condition(code_type)
        if not validator.validate(code, code_type):
            raise ValueError(f"Code {code} missing from database for {code_type}")
        samples.append(stage_payload(stage, code_type, (narrative, code, reasoning)))
    return samples


def build_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic THCE samples.")
    parser.add_argument("--stage", choices=[s.value for s in Stage], required=True)
    parser.add_argument("--count", type=int, default=10, help="Number of samples to generate.")
    parser.add_argument("--output", type=Path, required=True, help="Output jsonl path.")
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=Path("data/code_databases"),
        help="Directory containing code databases.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    return parser


def main(parsed_args: argparse.Namespace | None = None) -> int:
    parser = build_args()
    args = parsed_args or parser.parse_args()

    stage = Stage.from_string(args.stage)
    rng = random.Random(args.seed)
    validator = CodeValidator(args.database_dir)
    samples = generate_samples(args.count, stage, validator, rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for record in samples:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {len(samples)} {stage.value} samples at {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
