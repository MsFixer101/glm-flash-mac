#!/bin/bash
# Run any command under a memory watchdog (macOS).
#
#   ./launch.sh ~/venvs/mlx-vlm-glm5/bin/python run_flash.py
#   GUARD_FREE_PCT=15 GUARD_LOG=flash_mem.log ./launch.sh ~/venvs/mlx-vlm-glm5/bin/python run_flash.py
#
# Every second it reads the system-wide free percentage from `memory_pressure` and logs
#   time, free %, and the job's memory (physical footprint of the job plus every process it started)
# to GUARD_LOG (default: launch_mem.log). If free memory drops below GUARD_FREE_PCT (default 10),
# it sends TERM to the job's whole process group, and KILL 5 seconds later if it's still there.
# At the end it prints the lowest free % it saw and the job's exit code, and exits with that code.
#
# Why: MLX's memory limit (mx.set_memory_limit) is a guideline, not a hard stop. Without the
# sparse-attention fix in patches.py, MLX went from 186 to 226 GiB in two seconds on a 300-token
# prompt, and free memory fell from 35% to under 5% in five. A watchdog polling every second, outside
# the job, is what stops a 256 GB Mac from running out of memory.
#
# The job's stdin is /dev/null (it runs in its own background process group, where reading the terminal
# would stop it). Job memory is read every 5 s (it calls `footprint`); free memory is checked every second.
# If `memory_pressure` can't be read at the start, the job isn't started; if it can't be read for 5 s in a
# row while the job runs, the job is stopped.
#
# The job's output goes to your terminal untouched. If you pipe it (./launch.sh ... | tee out.txt),
# the shell reports tee's exit code, not the job's: read the "exit" line this script prints instead.
# (Inside this script the job is never piped, so the code it prints is the job's own.)

set -u
if [ $# -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 64
fi

GUARD_FREE_PCT=${GUARD_FREE_PCT:-10}
GUARD_LOG=${GUARD_LOG:-launch_mem.log}

free_pct() {
  memory_pressure | awk -F': ' '/free percentage/ {gsub("%", "", $2); print $2}'
}

# PIDs of a process and all its descendants.
tree_pids() {
  ps -Ao pid=,ppid= | awk -v root="$1" '
    { parent[$1] = $2 }
    END {
      for (p in parent) {
        q = p
        while (q != "" && q != 0 && q != 1) {
          if (q == root) { print p; break }
          q = parent[q]
        }
      }
    }'
}

# Memory (MB) of the job and everything it started. This is the physical footprint (what Activity Monitor
# shows as Memory), not ps's RSS: MLX's GPU buffers count towards the footprint but not towards RSS, so
# ps reports ~30 MB for a process holding a 4 GiB MLX array.
tree_mem_mb() {
  local args=() p
  for p in $(tree_pids "$1"); do args+=(-p "$p"); done
  [ ${#args[@]} -eq 0 ] && { echo 0; return; }
  footprint -f bytes --noCategories "${args[@]}" 2>/dev/null |
    awk '/phys_footprint:/ { total += $2 } END { printf "%.0f", total / 1048576 }'
}

start_free=$(free_pct)
case "$start_free" in ''|*[!0-9]*) echo "launch.sh: can't read free memory from memory_pressure; not starting" >&2; exit 70;; esac
case "$GUARD_FREE_PCT" in ''|*[!0-9]*) echo "launch.sh: GUARD_FREE_PCT must be a whole number" >&2; exit 64;; esac
echo "$(date +%T) start: ${start_free}% free, guard below ${GUARD_FREE_PCT}%, cmd: $*" >> "$GUARD_LOG"

# Job control on: the job gets its own process group, so the guard can stop everything it started.
set -m
"$@" </dev/null &
JOB=$!
set +m

stop_job() {  # TERM the whole group, then KILL after 5 s
  kill -TERM -- -"$JOB" 2>/dev/null || kill -TERM "$JOB" 2>/dev/null
  for _ in 1 2 3 4 5; do
    kill -0 -- -"$JOB" 2>/dev/null || kill -0 "$JOB" 2>/dev/null || return
    sleep 1
  done
  kill -KILL -- -"$JOB" 2>/dev/null || kill -KILL "$JOB" 2>/dev/null
}
trap 'echo "$(date +%T) interrupted, stopping job $JOB" >> "$GUARD_LOG"; stop_job' INT TERM HUP

min_free=$start_free
guard_fired=0
misses=0
n=0
r=0
while kill -0 "$JOB" 2>/dev/null; do
  f=$(free_pct)
  [ $((n % 5)) -eq 0 ] && r=$(tree_mem_mb "$JOB")
  n=$((n + 1))
  echo "$(date +%T) free ${f:-?}% job ${r} MB" >> "$GUARD_LOG"
  case "$f" in ''|*[!0-9]*) f=""; misses=$((misses + 1)) ;; *) misses=0 ;; esac
  if [ "$misses" -ge 5 ] && [ "$guard_fired" -eq 0 ]; then
    guard_fired=1
    echo "$(date +%T) GUARD: can't read free memory for 5 s, stopping job $JOB" | tee -a "$GUARD_LOG" >&2
    stop_job
  fi
  if [ -n "$f" ]; then
    [ "$f" -lt "$min_free" ] && min_free=$f
    if [ "$f" -lt "$GUARD_FREE_PCT" ] && [ "$guard_fired" -eq 0 ]; then
      guard_fired=1
      echo "$(date +%T) GUARD: ${f}% free < ${GUARD_FREE_PCT}%, stopping job $JOB" | tee -a "$GUARD_LOG" >&2
      stop_job
    fi
  fi
  sleep 1
done

wait "$JOB"
rc=$?
summary="min free ${min_free}%, exit ${rc}"
[ "$guard_fired" -eq 1 ] && summary="$summary (stopped by the guard)"
echo "$(date +%T) $summary" >> "$GUARD_LOG"
echo "launch.sh: $summary (log: $GUARD_LOG)" >&2
exit "$rc"
