#!/bin/bash
# waits for queue 1, then runs extra seeds for the two headline PRESAGE variants
cd "$(dirname "$0")/.."; export PYTHONPATH=src PRESAGE_MLFLOW=0 MLFLOW_DISABLE_AGENT_HINT=1
while ! grep -q "QUEUE DONE" reports/results/queue.log; do sleep 30; done
for s in 1 2; do
  echo "=== $(date +%H:%M:%S) START seed $s" >> reports/results/queue2.log
  python3 scripts/run_presage.py --variants full,concat --epochs 12 --seed $s > reports/results/seed$s.log 2>&1
  echo "=== $(date +%H:%M:%S) END seed $s (exit $?)" >> reports/results/queue2.log
done
echo "=== QUEUE2 DONE" >> reports/results/queue2.log
