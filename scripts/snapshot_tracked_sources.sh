#!/bin/zsh
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/snapshot_tracked_sources.sh [all|changed|staged]

Modes:
  all      snapshot all tracked source files
  changed  snapshot tracked source files changed in the working tree
  staged   snapshot tracked source files currently staged for commit
EOF
}

mode="${1:-changed}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
backup_root="$repo_root/source_snapshots/$(date +%Y%m%d_%H%M%S)"

is_source_file() {
  case "$1" in
    *.py|*.tex|*.md|*.bib|*.sh|*.txt) return 0 ;;
    *) return 1 ;;
  esac
}

collect_files() {
  case "$mode" in
    all)
      git -C "$repo_root" ls-files
      ;;
    changed)
      {
        git -C "$repo_root" diff --name-only
        git -C "$repo_root" diff --cached --name-only
        git -C "$repo_root" ls-files --others --exclude-standard
      } | sort -u
      ;;
    staged)
      git -C "$repo_root" diff --cached --name-only
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

copied=0
while IFS= read -r relpath; do
  [[ -n "$relpath" ]] || continue
  is_source_file "$relpath" || continue
  src="$repo_root/$relpath"
  [[ -f "$src" ]] || continue
  dest="$backup_root/$relpath"
  mkdir -p "$(dirname "$dest")"
  cp "$src" "$dest"
  copied=$((copied + 1))
done < <(collect_files)

if [[ "$copied" -eq 0 ]]; then
  rmdir "$backup_root" 2>/dev/null || true
  echo "No tracked source files to snapshot."
  exit 0
fi

echo "Snapshot saved: $backup_root ($copied files)"

