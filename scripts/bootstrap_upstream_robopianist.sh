#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <target-parent-dir>" >&2
  exit 1
fi

target_parent="$1"
mkdir -p "$target_parent"

snapshot_dir="$target_parent/robopiano"
upstream_dir="$target_parent/robopianist-upstream"

echo "Expected deployment snapshot path: $snapshot_dir"

if [[ -d "$upstream_dir/.git" ]]; then
  echo "Upstream RoboPianist already present at $upstream_dir"
  exit 0
fi

git clone https://github.com/google-research/robopianist.git "$upstream_dir"
echo "Cloned upstream RoboPianist into $upstream_dir"
