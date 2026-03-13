#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/derive_watson_quartic_vanvleck.py"
RUN_DIR="$SCRIPT_DIR/.runs"
PID_FILE="$RUN_DIR/vanvleck_bg.pid"
LOG_FILE="$RUN_DIR/vanvleck_bg.log"
CMD_FILE="$RUN_DIR/vanvleck_bg.cmd"

mkdir -p "$RUN_DIR"

usage() {
  cat <<EOF
Usage:
  $0 start [args...]
  $0 status
  $0 stop
  $0 tail

Examples:
  $0 start --max-order 4 --n-modes 3 --collapsed-couplings --seed 11 --max-vib-word 3 --phi4-reduced --channel-aware
  $0 status
  $0 tail
  $0 stop
EOF
}

is_running() {
  if [ ! -f "$PID_FILE" ]; then
    return 1
  fi
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -z "$pid" ]; then
    return 1
  fi
  kill -0 "$pid" 2>/dev/null
}

start_run() {
  if is_running; then
    echo "Already running with PID $(cat "$PID_FILE")."
    exit 0
  fi

  : >"$LOG_FILE"
  printf '%s\n' "python3 $PY_SCRIPT $*" >"$CMD_FILE"

  if command -v setsid >/dev/null 2>&1; then
    setsid nohup python3 "$PY_SCRIPT" "$@" >>"$LOG_FILE" 2>&1 </dev/null &
  else
    nohup python3 "$PY_SCRIPT" "$@" >>"$LOG_FILE" 2>&1 </dev/null &
  fi

  pid=$!
  printf '%s\n' "$pid" >"$PID_FILE"
  sleep 1

  if is_running; then
    echo "Started PID $pid"
    echo "Log: $LOG_FILE"
    exit 0
  fi

  echo "Process exited immediately. Check log: $LOG_FILE" >&2
  tail -n 40 "$LOG_FILE" 2>/dev/null || true
  exit 1
}

status_run() {
  if is_running; then
    echo "Running PID $(cat "$PID_FILE")"
    echo "Log: $LOG_FILE"
    if [ -f "$CMD_FILE" ]; then
      echo "Command:"
      cat "$CMD_FILE"
    fi
    exit 0
  fi

  echo "Not running"
  if [ -f "$LOG_FILE" ]; then
    echo "Last log lines:"
    tail -n 20 "$LOG_FILE" || true
  fi
}

stop_run() {
  if ! is_running; then
    echo "Not running"
    rm -f "$PID_FILE"
    exit 0
  fi

  pid="$(cat "$PID_FILE")"
  kill "$pid" 2>/dev/null || true
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "Stopped PID $pid"
}

tail_run() {
  if [ -f "$LOG_FILE" ]; then
    tail -n 40 "$LOG_FILE"
  else
    echo "No log file: $LOG_FILE"
  fi
}

cmd="${1:-}"
case "$cmd" in
  start)
    shift
    start_run "$@"
    ;;
  status)
    status_run
    ;;
  stop)
    stop_run
    ;;
  tail)
    tail_run
    ;;
  *)
    usage
    exit 1
    ;;
esac
