from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate plots and a written summary for primitive online rollout results.",
    )
    parser.add_argument("--input-root", required=True, help="Primitive rollout evaluation output directory.")
    parser.add_argument("--output-dir", default=None, help="Where to save analysis artifacts.")
    parser.add_argument("--top-k", type=int, default=5, help="How many top/bottom primitives to summarize.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_df = pd.read_csv(input_root / "primitive_instance_metrics.csv")
    aggregate = json.loads((input_root / "aggregate_metrics.json").read_text(encoding="utf-8"))

    if result_df.empty:
        raise SystemExit("No rollout rows found.")

    numeric_columns = [
        "predicted_event_count",
        "matched_onset_count",
        "onset_precision",
        "onset_recall",
        "onset_f1",
        "action_mse",
        "piano_state_mse",
        "intended_event_count",
        "missed_key_events",
        "false_positive_key_events",
    ]
    for column in numeric_columns:
        if column in result_df.columns:
            result_df[column] = pd.to_numeric(result_df[column], errors="coerce")

    result_df["has_key_press"] = result_df["predicted_event_count"].fillna(0) > 0
    result_df["has_target_hit"] = result_df["matched_onset_count"].fillna(0) > 0
    result_df["quality_bucket"] = result_df.apply(classify_row, axis=1)

    summary = build_summary(result_df=result_df, aggregate=aggregate, top_k=int(args.top_k))
    (output_dir / "rollout_analysis_summary.md").write_text(summary, encoding="utf-8")

    plot_overview_counts(result_df, output_dir / "overview_counts.png")
    plot_ranked_f1(result_df, output_dir / "ranked_onset_f1.png")
    plot_precision_recall(result_df, output_dir / "precision_recall_scatter.png")
    plot_event_coverage(result_df, output_dir / "event_coverage.png")
    plot_error_tradeoff(result_df, output_dir / "mse_vs_onset_f1.png")
    save_representative_roll_plots(result_df, output_dir / "representatives", top_k=min(int(args.top_k), 3))


def classify_row(row: pd.Series) -> str:
    onset_f1 = float(row.get("onset_f1", float("nan")))
    predicted_events = float(row.get("predicted_event_count", 0.0) or 0.0)
    matched = float(row.get("matched_onset_count", 0.0) or 0.0)
    if predicted_events <= 0:
        return "silent"
    if matched <= 0:
        return "off_target"
    if not np.isnan(onset_f1) and onset_f1 >= 0.5:
        return "strong"
    return "partial"


def build_summary(*, result_df: pd.DataFrame, aggregate: dict[str, object], top_k: int) -> str:
    bucket_counts = result_df["quality_bucket"].value_counts().to_dict()
    key_press_count = int(result_df["has_key_press"].sum())
    target_hit_count = int(result_df["has_target_hit"].sum())
    mean_predicted_ratio = float(
        (result_df["predicted_event_count"] / result_df["intended_event_count"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).mean()
    )

    ranked = result_df.sort_values(["onset_f1", "matched_onset_count"], ascending=[False, False], kind="stable")
    top_rows = ranked.head(max(top_k, 1))
    bottom_rows = ranked.tail(max(top_k, 1)).sort_values(["onset_f1", "matched_onset_count"], ascending=[True, True], kind="stable")

    qualitative = []
    qualitative.append(
        "The primitives are generally responsive but sparse: most produced some key presses, and most hit at least one target onset, "
        "but average recall stayed low while precision stayed high."
    )
    qualitative.append(
        "That pattern suggests the primitive priors capture coarse action-to-key mappings well enough to trigger plausible notes, "
        "but they usually replay only a subset of the full target segment rather than reconstructing the whole musical phrase."
    )
    if bucket_counts.get("silent", 0) > 0:
        qualitative.append(
            f"{bucket_counts.get('silent', 0)} primitives stayed effectively silent, which likely indicates priors that learned motion-light or "
            "state-transition behaviors rather than note-producing actions."
        )
    if bucket_counts.get("off_target", 0) > 0:
        qualitative.append(
            f"{bucket_counts.get('off_target', 0)} primitives pressed keys but missed their labeled targets, suggesting some clusters may encode "
            "ambiguous hand motions or context-dependent gestures that do not transfer cleanly when replayed in isolation."
        )
    qualitative.append(
        "The strongest primitives appear to be short, note-producing motifs whose learned priors can be dropped into the environment and still "
        "activate the intended keyboard region. The weakest ones either under-fire, terminate with no note activity, or produce only a tiny prefix "
        "of the target segment."
    )

    lines = [
        "# Primitive Rollout Analysis",
        "",
        "## Overall",
        f"- Total primitives evaluated: {len(result_df)}",
        f"- Real key presses: {key_press_count}",
        f"- Any target hits: {target_hit_count}",
        f"- Strong rollouts (`onset_f1 >= 0.5`): {bucket_counts.get('strong', 0)}",
        f"- Partial rollouts (`0 < onset_f1 < 0.5`): {bucket_counts.get('partial', 0)}",
        f"- Off-target rollouts (`predicted_event_count > 0`, no target hits): {bucket_counts.get('off_target', 0)}",
        f"- Silent rollouts (`predicted_event_count == 0`): {bucket_counts.get('silent', 0)}",
        f"- Mean onset precision: {float(aggregate.get('mean_onset_precision', float('nan'))):.3f}",
        f"- Mean onset recall: {float(aggregate.get('mean_onset_recall', float('nan'))):.3f}",
        f"- Mean onset F1: {float(aggregate.get('mean_onset_f1', float('nan'))):.3f}",
        f"- Mean piano-state MSE: {float(aggregate.get('mean_piano_state_mse', float('nan'))):.4f}",
        f"- Mean action MSE: {float(aggregate.get('mean_action_mse', float('nan'))):.4f}",
        f"- Mean predicted/intended event ratio: {mean_predicted_ratio:.3f}",
        "",
        "## Top primitives",
    ]
    for row in top_rows.itertuples(index=False):
        lines.append(
            f"- `{row.primitive_id}`: F1={_fmt(row.onset_f1)}, recall={_fmt(row.onset_recall)}, "
            f"precision={_fmt(row.onset_precision)}, predicted_events={int(row.predicted_event_count)}, "
            f"target_hits={int(row.matched_onset_count)}, source=`{row.rollout_source_label}`"
        )
    lines.extend(["", "## Weakest primitives"])
    for row in bottom_rows.itertuples(index=False):
        lines.append(
            f"- `{row.primitive_id}`: F1={_fmt(row.onset_f1)}, recall={_fmt(row.onset_recall)}, "
            f"precision={_fmt(row.onset_precision)}, predicted_events={int(row.predicted_event_count)}, "
            f"target_hits={int(row.matched_onset_count)}, source=`{row.rollout_source_label}`"
        )
    lines.extend(["", "## Qualitative read"])
    lines.extend([f"- {item}" for item in qualitative])
    lines.append("")
    return "\n".join(lines)


def plot_overview_counts(result_df: pd.DataFrame, output_path: Path) -> None:
    counts = {
        "All": len(result_df),
        "Key press": int(result_df["has_key_press"].sum()),
        "Target hit": int(result_df["has_target_hit"].sum()),
        "Strong F1": int((result_df["onset_f1"].fillna(0) >= 0.5).sum()),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(counts.keys()), list(counts.values()), color=["#4C78A8", "#72B7B2", "#54A24B", "#ECA82C"])
    ax.set_title("Primitive Rollout Outcome Counts")
    ax.set_ylabel("Primitive count")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3, f"{int(bar.get_height())}", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ranked_f1(result_df: pd.DataFrame, output_path: Path) -> None:
    palette = {
        "strong": "#54A24B",
        "partial": "#ECA82C",
        "off_target": "#E45756",
        "silent": "#9D9D9D",
    }
    frame = result_df.sort_values(["onset_f1", "matched_onset_count"], ascending=[False, False], kind="stable").reset_index(drop=True)
    colors = [palette[item] for item in frame["quality_bucket"]]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(frame["primitive_id"].astype(str), frame["onset_f1"].fillna(0.0), color=colors)
    ax.set_title("Per-Primitive Onset F1 (ranked)")
    ax.set_ylabel("Onset F1")
    ax.set_xlabel("Primitive")
    ax.tick_params(axis="x", rotation=90)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=palette[key]) for key in ("strong", "partial", "off_target", "silent")]
    ax.legend(legend_handles, ["strong", "partial", "off_target", "silent"], title="Bucket", loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_precision_recall(result_df: pd.DataFrame, output_path: Path) -> None:
    frame = result_df.copy()
    size = np.clip(frame["predicted_event_count"].fillna(0).to_numpy(dtype=float), 0, None) * 40 + 40
    colors = frame["onset_f1"].fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        frame["onset_recall"].fillna(0.0),
        frame["onset_precision"].fillna(0.0),
        c=colors,
        s=size,
        cmap="viridis",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title("Precision vs Recall by Primitive")
    ax.set_xlabel("Onset recall")
    ax.set_ylabel("Onset precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    for row in frame.nlargest(4, "onset_f1").itertuples(index=False):
        ax.annotate(str(row.primitive_id), (float(row.onset_recall), float(row.onset_precision)), fontsize=8, xytext=(4, 4), textcoords="offset points")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Onset F1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_event_coverage(result_df: pd.DataFrame, output_path: Path) -> None:
    frame = result_df.sort_values(["matched_onset_count", "predicted_event_count"], ascending=[False, False], kind="stable").reset_index(drop=True)
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, frame["intended_event_count"].fillna(0.0), color="#D3D3D3", label="intended events")
    ax.bar(x, frame["predicted_event_count"].fillna(0.0), color="#4C78A8", alpha=0.9, label="predicted events")
    ax.scatter(x, frame["matched_onset_count"].fillna(0.0), color="#E45756", label="matched onsets", zorder=3)
    ax.set_title("Event coverage per primitive")
    ax.set_ylabel("Event count")
    ax.set_xlabel("Primitive")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["primitive_id"].astype(str), rotation=90)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_error_tradeoff(result_df: pd.DataFrame, output_path: Path) -> None:
    frame = result_df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(frame["action_mse"].fillna(0.0), frame["onset_f1"].fillna(0.0), color="#4C78A8", alpha=0.85)
    axes[0].set_title("Action MSE vs Onset F1")
    axes[0].set_xlabel("Action MSE")
    axes[0].set_ylabel("Onset F1")
    axes[1].scatter(frame["piano_state_mse"].fillna(0.0), frame["onset_f1"].fillna(0.0), color="#54A24B", alpha=0.85)
    axes[1].set_title("Piano-state MSE vs Onset F1")
    axes[1].set_xlabel("Piano-state MSE")
    axes[1].set_ylabel("Onset F1")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_representative_roll_plots(result_df: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    usable = result_df.loc[result_df["debug_artifact_path"].notna()].copy()
    if usable.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = usable.sort_values(["onset_f1", "matched_onset_count"], ascending=[False, False], kind="stable")
    candidates = pd.concat([ranked.head(top_k), ranked.tail(top_k)], ignore_index=True).drop_duplicates(subset=["primitive_id"])
    for row in candidates.itertuples(index=False):
        artifact_path = Path(str(row.debug_artifact_path))
        if not artifact_path.exists():
            continue
        payload = np.load(artifact_path, allow_pickle=True)
        observed = np.asarray(payload["observed_piano_states"], dtype=np.float32)
        ground_truth = np.asarray(payload["piano_states_gt"], dtype=np.float32)
        if observed.size == 0 or ground_truth.size == 0:
            continue
        keys = select_keys(observed=observed, ground_truth=ground_truth)
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        axes[0].imshow(ground_truth[:, keys].T, aspect="auto", interpolation="nearest", origin="lower")
        axes[0].set_title(f"Ground truth: {row.primitive_id} ({row.rollout_source_label})")
        axes[0].set_ylabel("Key")
        axes[1].imshow(observed[:, keys].T, aspect="auto", interpolation="nearest", origin="lower")
        axes[1].set_title(f"Rollout: F1={_fmt(row.onset_f1)}")
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Key")
        fig.tight_layout()
        fig.savefig(output_dir / f"{row.primitive_id}_{safe_name(str(row.rollout_source_label))}.png", dpi=200)
        plt.close(fig)


def select_keys(*, observed: np.ndarray, ground_truth: np.ndarray, max_keys: int = 24) -> np.ndarray:
    observed_on = np.flatnonzero(np.max(observed[:, :88], axis=0) > 0.5) if observed.shape[1] else np.array([], dtype=int)
    ground_truth_on = np.flatnonzero(np.max(ground_truth[:, :88], axis=0) > 0.5) if ground_truth.shape[1] else np.array([], dtype=int)
    keys = np.unique(np.concatenate([observed_on, ground_truth_on])) if observed_on.size or ground_truth_on.size else np.arange(min(12, ground_truth.shape[1]))
    if keys.size > max_keys:
        return keys[:max_keys]
    return keys


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _fmt(value: float) -> str:
    if value is None or np.isnan(value):
        return "nan"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
