"""Convert lightly structured medical coding samples into THCE training formats."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from utils.code_validator import CodeValidator
from utils.medical_chat_templates import ChatTemplate, Stage, apply_template, get_template


@dataclass(slots=True)
class RawRecord:
    """Representation of a single raw record before conversion."""

    narrative: str
    code_type: str
    code: str
    description: str
    reasoning: Sequence[str]
    rejected_code: Optional[str] = None
    rejected_reasoning: Sequence[str] | None = None

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "RawRecord":
        reasoning_field = payload.get("reasoning") or []
        if isinstance(reasoning_field, str):
            reasoning = [line.strip() for line in reasoning_field.split("\n") if line.strip()]
        else:
            reasoning = [str(line) for line in reasoning_field]  # type: ignore[arg-type]

        rejected_reasoning_field = payload.get("rejected_reasoning")
        if isinstance(rejected_reasoning_field, str):
            rejected_reasoning = [
                line.strip() for line in rejected_reasoning_field.split("\n") if line.strip()
            ]
        elif rejected_reasoning_field:
            rejected_reasoning = [str(line) for line in rejected_reasoning_field]  # type: ignore[arg-type]
        else:
            rejected_reasoning = None

        return cls(
            narrative=str(payload["narrative"]),
            code_type=str(payload["code_type"]),
            code=str(payload["code"]),
            description=str(payload.get("description", "")),
            reasoning=tuple(reasoning),
            rejected_code=(
                str(payload["rejected_code"])
                if payload.get("rejected_code") is not None
                else None
            ),
            rejected_reasoning=tuple(rejected_reasoning) if rejected_reasoning else None,
        )


def load_raw_records(path: Path) -> List[RawRecord]:
    records: List[RawRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                msg = f"Invalid JSON on line {index + 1}: {exc}"
                raise ValueError(msg) from exc
            records.append(RawRecord.from_dict(payload))
    return records


def stage_1_record(template: ChatTemplate, raw: RawRecord) -> Dict[str, object]:
    messages = apply_template(template, raw.narrative, [], f"{raw.code_type}: {raw.code} ({raw.description})")
    return {
        "messages": messages,
        "_metadata": {
            "code_type": raw.code_type,
            "code": raw.code,
            "description": raw.description,
            "reasoning_present": False,
        },
    }


def stage_2_record(template: ChatTemplate, raw: RawRecord) -> Dict[str, object]:
    messages = apply_template(
        template,
        raw.narrative,
        raw.reasoning,
        f"{raw.code_type}: {raw.code} ({raw.description})",
    )
    return {
        "messages": messages,
        "_metadata": {
            "code_type": raw.code_type,
            "code": raw.code,
            "description": raw.description,
            "reasoning_present": True,
        },
    }


def stage_3_record(template: ChatTemplate, raw: RawRecord) -> Dict[str, object]:
    reasoning = "\n".join(raw.reasoning)
    rejected_reasoning = "\n".join(raw.rejected_reasoning or ["Reasoning unclear."])
    chosen = (
        f"<think>\n{reasoning}\n</think>\n{raw.code_type}: {raw.code} ({raw.description})"
    )
    rejected_code = raw.rejected_code or raw.code
    rejected = (
        f"<think>\n{rejected_reasoning}\n</think>\n{raw.code_type}: {rejected_code}"
    )
    return {
        "prompt": [
            {"role": "system", "content": template.system_prompt},
            {"role": "user", "content": raw.narrative},
        ],
        "chosen": chosen,
        "rejected": rejected,
        "_metadata": {
            "code_type": raw.code_type,
            "code": raw.code,
            "rejected_code": rejected_code,
        },
    }


def convert_records(
    records: Iterable[RawRecord],
    stage: Stage,
    validator: CodeValidator,
) -> List[Dict[str, object]]:
    template = get_template(stage)
    converted: List[Dict[str, object]] = []
    for raw in records:
        if not validator.validate(raw.code, raw.code_type):
            msg = f"Unknown code {raw.code} for type {raw.code_type}."
            raise ValueError(msg)

        if stage is Stage.STAGE_1:
            converted.append(stage_1_record(template, raw))
        elif stage is Stage.STAGE_2:
            converted.append(stage_2_record(template, raw))
        elif stage is Stage.STAGE_3:
            converted.append(stage_3_record(template, raw))
    return converted


def write_jsonl(records: Iterable[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess raw THCE data into training formats.")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw jsonl file.")
    parser.add_argument("--output", type=Path, required=True, help="Destination jsonl file.")
    parser.add_argument(
        "--stage",
        choices=[stage.value for stage in Stage],
        required=True,
        help="Training stage format to produce.",
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=Path("data/code_databases"),
        help="Directory containing code databases.",
    )
    return parser


def main(args: Optional[argparse.Namespace] = None) -> int:
    parser = build_arguments()
    parsed = args or parser.parse_args()

    stage = Stage.from_string(parsed.stage)
    validator = CodeValidator(parsed.database_dir)
    records = load_raw_records(parsed.input)
    processed = convert_records(records, stage, validator)
    write_jsonl(processed, parsed.output)
    print(f"Wrote {len(processed)} records to {parsed.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
