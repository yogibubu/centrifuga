#!/bin/zsh
set -euo pipefail

# Minimal compatibility shim for the repository pre-commit hook.
#
# Historical hooks expect this script to exist and to accept a mode argument
# such as `staged`. The current repository workflow no longer depends on any
# generated snapshot artifact, but commits should still succeed when the hook is
# enabled. So this helper validates the invocation shape and exits successfully
# without side effects.

mode="${1:-staged}"

case "$mode" in
  staged|tracked|all)
    exit 0
    ;;
  *)
    echo "snapshot_tracked_sources.sh: unsupported mode '$mode'" >&2
    exit 2
    ;;
esac
