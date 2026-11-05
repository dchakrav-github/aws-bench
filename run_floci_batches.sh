#!/usr/bin/env bash
# Driver: run 11 batches of the 11 single-shot aws-agent-bench tasks via the
# kiro-cli session-mount method against Floci. 2-parallel, 5 trials each.
# Aborts the whole run on the FIRST sign of a kiro authentication failure.
set -uo pipefail

cd /local/home/diwakar/workspace/aws-bench/aws-bench

export AWS_PROFILE=ag-bm-adm
export AWS_BENCH_KIRO_SESSION_MOUNT=1
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1

TASKS=/home/diwakar/workspace/iac-bm/src/CloudAgentBenchmark/aws-agent-bench/tasks
SCEN=/home/diwakar/workspace/iac-bm/src/CloudAgentBenchmark/aws-agent-bench/scenarios
RUNTAG=floci-kiro-matrix-$(date +%Y%m%d-%H%M%S)
DRIVER_LOG=/tmp/${RUNTAG}.driver.log

# The 11 single-shot tasks (dataset.toml non-multistep set).
INC=( -i async-job-queue -i event-media-pipeline -i frontend-backend-cache \
      -i net-admin-probe -i private-file-ingest -i public-api \
      -i public-app-private-admin -i scheduled-reporting -i webapp-rdbms \
      -i webhook-fanout -i zonal-resilience )

N_BATCHES=11
N_CONCURRENT=2
N_ATTEMPTS=5

echo "[$(date)] RUNTAG=$RUNTAG  batches=$N_BATCHES n=$N_CONCURRENT k=$N_ATTEMPTS" | tee "$DRIVER_LOG"

# Auth-failure detector. Scans a finished job dir. Returns 0 (OK) or 1 (auth failure).
check_auth() {
  local job_dir="$1"
  # 1) Any kiro stream .err file carrying an auth-type error.
  if grep -rqiE "not authenticated|unauthorized|invalid[_ ]?grant|token (is )?expired|refresh token|403 Forbidden|AccessDeniedException: .*bedrock|failed to (login|authenticate)|no (valid )?credentials|re-?authenticate" \
        "$job_dir"/*/agent/kiro-cli-stream.err 2>/dev/null; then
    return 1
  fi
  # 2) Any trial with an empty/near-empty trajectory AND a non-zero exit in the
  #    stream err (the documented "stopped before doing work" symptom).
  local f
  for f in "$job_dir"/*/agent/kiro-cli-stream.jsonl; do
    [ -f "$f" ] || continue
    if [ "$(wc -l < "$f")" -lt 2 ]; then
      local errf="${f%.jsonl}.err"
      if [ -s "$errf" ]; then return 1; fi
    fi
  done
  return 0
}

for b in $(seq 1 "$N_BATCHES"); do
  JOB="${RUNTAG}-batch$(printf '%02d' "$b")"
  echo "[$(date)] === BATCH $b/$N_BATCHES  job=$JOB ===" | tee -a "$DRIVER_LOG"
  uv run aws-bench run \
    -p "$TASKS" --scenario-path "$SCEN" \
    "${INC[@]}" \
    -a kiro-cli -m claude-opus-5 \
    --agent-env AWS_BENCH_KIRO_SESSION_MOUNT=1 \
    -n "$N_CONCURRENT" -k "$N_ATTEMPTS" \
    --env-name aab-env --no-verify-env \
    --job-name "$JOB" \
    --yes --quiet >> "$DRIVER_LOG" 2>&1
  rc=$?
  JOB_DIR="jobs/$JOB"
  echo "[$(date)] batch $b exit=$rc" | tee -a "$DRIVER_LOG"

  # Report per-batch stats from result.json.
  if [ -f "$JOB_DIR/result.json" ]; then
    python3 - "$JOB_DIR/result.json" <<'PY' | tee -a "$DRIVER_LOG"
import json,sys
r=json.load(open(sys.argv[1]))
s=r.get("stats",{})
print(f"  trials={s.get('n_completed_trials')} errored={s.get('n_errored_trials')} "
      f"cancelled={s.get('n_cancelled_trials')} retries={s.get('n_retries')}")
for name,ev in (s.get("evals") or {}).items():
    print(f"  eval {name}: n={ev.get('n_trials')} errors={ev.get('n_errors')} "
          f"exc={ev.get('exception_stats')} metrics={ev.get('metrics')}")
PY
  fi

  # AUTH GATE: stop everything on any auth failure signal.
  if ! check_auth "$JOB_DIR"; then
    echo "[$(date)] !!! AUTH FAILURE DETECTED in $JOB_DIR — STOPPING per instruction." | tee -a "$DRIVER_LOG"
    echo "ABORTED_AUTH" > /tmp/${RUNTAG}.status
    exit 2
  fi
done

echo "[$(date)] ALL $N_BATCHES BATCHES COMPLETE — no auth failures." | tee -a "$DRIVER_LOG"
echo "DONE" > /tmp/${RUNTAG}.status
echo "RUNTAG=$RUNTAG" > /tmp/floci-latest-runtag
