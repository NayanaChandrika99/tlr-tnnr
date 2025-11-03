"""
MTSamples adapter stub. Converts transcription records into THCE raw record schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List


def load_mtsamples(path: Path) -> Iterator[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"MTSamples file missing: {path}")
    # TODO: parse CSV/TSV content into structured dicts.
    raise NotImplementedError("Parse MTSamples transcripts into structured rows.")


def convert_to_thce_records(rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for row in rows:
        # TODO: craft narrative, assign code_type/code, and reasoning bullets.
        raise NotImplementedError("Transform MTSamples row into THCE schema.")
    return records
