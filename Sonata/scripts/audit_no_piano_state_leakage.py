from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUSPICIOUS_PATTERNS = [
    re.compile(r"\bstart_piano_state\b"),
    re.compile(r"\bpiano_states_gt\b"),
    re.compile(r"\bgoals_as_realized\b"),
    re.compile(r"\bpiano_states_as_intended\b"),
    re.compile(r"\btask\._notes\b"),
    re.compile(r"\btask\._midi\b"),
    re.compile(r"\breference_midi\b"),
    re.compile(r"\btarget_key_indices\b.*\bobserved\b"),
    re.compile(r"\bpiano_states\b.*\brollout observations\b"),
    re.compile(r"\brestore[_ ]piano\b", re.IGNORECASE),
    re.compile(r"\bset[_ ]piano\b", re.IGNORECASE),
    re.compile(r"\bactivation copied from dataset\b", re.IGNORECASE),
    re.compile(r"\b_update_goal_state\b"),
    re.compile(r"\b_goal_current\b"),
]

ALLOW_GUARDS = [
    "unsafe_legacy",
    "allow_goal_realized_fallback",
    "allow_piano_state_intended_fallback",
    "allow_piano_state_restore",
    "allow_goal_state_restore",
    "legacy_state",
    "offline",
    "diagnostic",
    "target_source",
    "reference_midi_key_indices",
    "primitive_training_contract",
    "PrimitiveInstance",
    "causal_config.enabled else",
    "_aligned_mse",
    "_empty_array_if_none",
    "save_representative_timing_plots",
    "debug_artifact",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit online rollout code for piano-state leakage hazards.")
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    paths = [
        *(root / "src" / "sonata" / "evaluation").glob("*.py"),
        *(root / "scripts").glob("evaluate*.py"),
    ]
    findings = []
    for path in sorted(set(paths)):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for line_no, line in enumerate(lines, start=1):
            if not any(pattern.search(line) for pattern in SUSPICIOUS_PATTERNS):
                continue
            context = "\n".join(lines[max(0, line_no - 40) : min(len(lines), line_no + 12)])
            if "piano_states_gt=_as_float_array" in line or "start_piano_state=_first_frame" in line:
                continue
            if any(token in context for token in ALLOW_GUARDS):
                continue
            findings.append(
                {
                    "path": str(path.relative_to(root)),
                    "line": line_no,
                    "text": line.strip(),
                    "reason": "suspicious online rollout piano/goal/reference-state usage lacks an explicit legacy or allow guard",
                }
            )
    payload = {"status": "passed" if not findings else "failed", "findings": findings}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
