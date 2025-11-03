"""
MIMIC adapter: transforms hospital notes and associated diagnosis/procedure codes into
the raw THCE schema consumed by `data/preprocess_medical.py`.

This module currently provides scaffolding only. Fill in the TODO sections once the
MIMIC CSV files are staged under `data/raw/mimic/`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


@dataclass
class MimicConfig:
    notes_path: Path
    diagnoses_path: Path
    procedures_path: Path | None = None


def load_mimic_notes(config: MimicConfig) -> Iterator[Dict[str, str]]:
    """Yield raw notes joined with their ICD codes. TODO: implement joins."""
    if not config.notes_path.exists():
        raise FileNotFoundError(f"Notes file missing: {config.notes_path}")
    if not config.diagnoses_path.exists():
        raise FileNotFoundError(f"Diagnoses file missing: {config.diagnoses_path}")
    # TODO: implement join logic once files are present.
    raise NotImplementedError("Implement MIMIC join logic when raw data is available.")


def convert_to_thce_records(raw_records: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    """Convert joined MIMIC rows to THCE raw record schema."""
    records: List[Dict[str, object]] = []
    for row in raw_records:
        # TODO: map MIMIC fields to THCE schema (narrative, code_type, code, description, reasoning)
        raise NotImplementedError("Map MIMIC rows to THCE raw record schema.")
    return records
