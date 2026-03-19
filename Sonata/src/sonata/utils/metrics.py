from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sonata.utils.io import append_csv_row


@dataclass
class MetricsWriter:
    output_dir: Path
    csv_name: str = "metrics.csv"
    jsonl_name: str = "metrics.jsonl"
    _fieldnames: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / self.csv_name
        self.jsonl_path = self.output_dir / self.jsonl_name

    def log(self, row: dict[str, Any]) -> None:
        ordered = dict(sorted(row.items(), key=lambda item: item[0]))
        if not self._fieldnames:
            self._fieldnames = list(ordered)
        else:
            for key in ordered:
                if key not in self._fieldnames:
                    self._fieldnames.append(key)
        append_csv_row(self.csv_path, self._fieldnames, ordered)
        with self.jsonl_path.open("a") as handle:
            handle.write(json.dumps(ordered, sort_keys=True) + "\n")
