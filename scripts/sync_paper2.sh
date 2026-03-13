#!/bin/zsh
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_paper2.sh push
  scripts/sync_paper2.sh pull
  scripts/sync_paper2.sh snapshot

Actions:
  push      copy repo paper2.tex to Desktop, backing up any Desktop copy first
  pull      copy Desktop paper2.tex into the repo, backing up the repo copy first
  snapshot  create a timestamped backup of the repo copy only
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
repo_file="$repo_root/paper2.tex"
desktop_file="/Users/vincenzobarone/Desktop/paper2.tex"
repo_backup_dir="$repo_root/manuscript_backups/paper2"
desktop_backup_dir="$repo_root/Desktop_sync_backups/paper2"
timestamp="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$repo_backup_dir" "$desktop_backup_dir"

case "$1" in
  push)
    [[ -f "$repo_file" ]] || { echo "Missing repo file: $repo_file" >&2; exit 1; }
    if [[ -f "$desktop_file" ]]; then
      cp "$desktop_file" "$desktop_backup_dir/paper2.desktop.$timestamp.tex"
    fi
    cp "$repo_file" "$desktop_file"
    echo "Pushed repo -> Desktop"
    echo "Desktop file: $desktop_file"
    ;;
  pull)
    [[ -f "$desktop_file" ]] || { echo "Missing Desktop file: $desktop_file" >&2; exit 1; }
    [[ -f "$repo_file" ]] || { echo "Missing repo file: $repo_file" >&2; exit 1; }
    cp "$repo_file" "$repo_backup_dir/paper2.repo.$timestamp.tex"
    cp "$desktop_file" "$repo_file"
    echo "Pulled Desktop -> repo"
    echo "Repo file: $repo_file"
    ;;
  snapshot)
    [[ -f "$repo_file" ]] || { echo "Missing repo file: $repo_file" >&2; exit 1; }
    cp "$repo_file" "$repo_backup_dir/paper2.repo.$timestamp.tex"
    echo "Snapshot saved: $repo_backup_dir/paper2.repo.$timestamp.tex"
    ;;
  *)
    usage
    exit 1
    ;;
esac
