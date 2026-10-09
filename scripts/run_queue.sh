#!/usr/bin/env bash
# Sequential experiment queue (2-core friendly). Failures do not stop later steps; see reports/results/queue.log
export PYTHONPATH=src PRESAGE_MLFLOW=0
cd "$(dirname "$0")/.."
step() { echo "=== $(date +%H:%M:%S) START $1" >> reports/results/queue.log; shift; "$@" > "reports/results/$LOGNAME_.log" 2>&1; echo "=== $(date +%H:%M:%S) END (exit $?)" >> reports/results/queue.log; }
LOGNAME_=serve_train;   step serve_train   python3 scripts/train_serving.py
LOGNAME_=explain;       step explain       python3 scripts/run_explainability.py
LOGNAME_=drift_adapt;   step drift_adapt   python3 scripts/run_drift.py --part adapt
LOGNAME_=drift_detect;  step drift_detect  python3 scripts/run_drift.py --part detect
LOGNAME_=federated;     step federated     python3 scripts/run_federated.py
LOGNAME_=real;          step real          python3 scripts/run_real_datasets.py
LOGNAME_=label_delay;   step label_delay   python3 scripts/run_drift.py --part labeldelay
LOGNAME_=ablation;      step ablation      python3 scripts/run_presage.py --variants no_graph,no_temporal,no_mule,no_multitask,no_beh,no_txn,no_benfeat --epochs 12
LOGNAME_=fusion;        step fusion        python3 scripts/run_presage.py --variants concat,attention,transformer_seq --epochs 12
LOGNAME_=graph25;       step graph25       python3 scripts/run_graph_models.py --epochs 20
echo "=== QUEUE DONE $(date +%H:%M:%S)" >> reports/results/queue.log
