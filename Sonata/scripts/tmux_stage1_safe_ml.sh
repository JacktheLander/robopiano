#!/usr/bin/env bash
# Creates or attaches tmux session and prints the exact Slurm submit command.
# Does NOT submit automatically (submit manually after reviewing).
set -euo pipefail

SESSION="${SESSION:-sonata_stage1_safe_ml}"
SLURM_SCRIPT="${SLURM_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm_stage1_safe_ml.sh}"

tmux new-session -A -s "$SESSION" -d
echo "Attached tmux session: $SESSION"
echo ""
echo "Exact Slurm command (run on a login node when ready):"
echo "  sbatch $SLURM_SCRIPT"
echo ""
echo "Optional env overrides before sbatch:"
echo "  export SONATA_ROOT=/path/to/Sonata"
echo "  export OUT_BASE=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run3/stage1_safe_ml"
echo "  export RUN_POST_EVAL=1   # append capped online eval after Stage 1"
echo ""
echo "To attach: tmux attach -t $SESSION"
