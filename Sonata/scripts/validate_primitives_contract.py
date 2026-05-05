from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the Stage 1 primitive training contract.")
    parser.add_argument("--primitive-root", required=True)
    parser.add_argument("--max-active-start-percent", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    primitive_root = Path(args.primitive_root).expanduser().resolve()
    contract_path = primitive_root / "validation" / "primitive_training_contract.json"
    if not contract_path.exists():
        raise SystemExit(f"Missing primitive training contract: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if contract.get("gmr_target_name") != "actions":
        failures.append("GMR target is not action-space.")
    if bool(contract.get("any_prior_not_action_dim", True)):
        failures.append("At least one primitive prior final dimension does not match action_dim.")
    if bool(contract.get("any_prior_uses_non_action_target", False)) or not bool(contract.get("no_piano_state_or_goal_target", True)):
        failures.append("At least one prior uses piano_state/goal targets instead of actions.")
    if bool(contract.get("wrist_key_relative_frame", False)):
        if contract.get("primitive_frame_mode") != "wrist_key_relative":
            failures.append("Wrist-relative profile does not record wrist_key_relative frame mode.")
        if not bool(contract.get("condition_features_exist", False)):
            failures.append("Conditioned wrist-relative primitives are missing condition features.")
        if bool(contract.get("missing_library_metadata_columns", [])):
            failures.append(f"Primitive library is missing metadata columns: {contract.get('missing_library_metadata_columns')}.")
        if bool(contract.get("absolute_key_position_main_clustering_driver", False)):
            failures.append("Absolute key position is still configured as a main clustering driver.")
    if bool(contract.get("causal_segment", False)):
        inactive_percent = float(contract.get("percent_segments_with_inactive_start", 0.0))
        if inactive_percent < 100.0 - float(args.max_active_start_percent):
            failures.append(f"Too many segments start with active target keys: inactive_start={inactive_percent:.2f}%.")
        if float(contract.get("percent_segments_with_activation_after_start", 0.0)) < 99.0:
            failures.append("Prepress segments do not consistently include activation after the segment start.")
        if str(contract.get("segment_alignment", "")) != "prepress_to_onset":
            failures.append("Missing prepress_to_onset segment alignment metadata.")
        if int(contract.get("num_rejected_segments", 0)) < 0 or not isinstance(contract.get("rejection_counts", {}), dict):
            failures.append("Rejection counts are missing or malformed.")
    if not bool(contract.get("pass_training_contract", False)):
        failures.append("pass_training_contract=false.")
    if failures:
        print(json.dumps({"status": "failed", "failures": failures, "contract_path": str(contract_path)}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"status": "passed", "contract_path": str(contract_path)}, indent=2))


if __name__ == "__main__":
    main()
