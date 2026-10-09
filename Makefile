# PRESAGE-Fraud reproducible pipeline. Every target is runnable from a clean checkout after `make setup`.
PY ?= python3
export PYTHONPATH := src
export PRESAGE_MLFLOW ?= 0
# macOS: torch, xgboost, lightgbm and catboost each bundle an OpenMP runtime; loading two in one process can segfault.
export KMP_DUPLICATE_LIB_OK ?= TRUE
export OMP_NUM_THREADS ?= 1

.PHONY: setup data features test train-baselines train-graph train-presage ablation evaluate drift-experiment label-delay federated-experiment explain serve-train serve stream-demo notebooks all

setup:
	$(PY) -m pip install -r requirements.txt

data:                 ## download real datasets (PaySim, IBM AML HI-Small, Elliptic) and simulate the APP/mule world
	$(PY) scripts/download_data.py
	$(PY) scripts/make_sim.py

features:             ## leakage-safe temporal + graph features
	$(PY) scripts/build_features.py

test:
	$(PY) -X faulthandler -m pytest -v -x -p no:cacheprovider

train-baselines:      ## tabular + anomaly baselines
	$(PY) scripts/run_baselines.py

train-graph:          ## static / heterogeneous / temporal graph encoders
	$(PY) scripts/run_graph_models.py --epochs 25

train-presage:        ## PRESAGE full model + fusion variants
	$(PY) scripts/run_presage.py --variants full,concat,attention,transformer_seq --epochs 25

ablation:
	$(PY) scripts/run_presage.py --variants no_txn,no_beh,no_graph,no_mule,no_temporal,no_multitask,no_benfeat --epochs 25

evaluate:             ## master tables, statistical tests, error analysis, figures
	$(PY) scripts/final_evaluation.py

drift-experiment:
	$(PY) scripts/run_drift.py --part adapt
	$(PY) scripts/run_drift.py --part detect

label-delay:
	$(PY) scripts/run_drift.py --part labeldelay

federated-experiment:
	$(PY) scripts/run_federated.py

explain:
	$(PY) scripts/run_explainability.py

serve-train:
	$(PY) scripts/train_serving.py

serve:
	$(PY) -m presage.serving

stream-demo:
	$(PY) streaming/run_demo.py --limit 20000

notebooks:
	$(PY) scripts/build_notebooks.py

all: setup data features test train-baselines train-graph train-presage ablation drift-experiment label-delay federated-experiment explain evaluate serve-train stream-demo

adversarial:          ## feature-space robustness (evasion, poisoning, dropout)
	$(PY) scripts/run_adversarial.py

seed-variance:        ## trains missing PRESAGE seeds (about 25 min each) and compares with GBDT
	$(PY) scripts/run_seed_variance.py --train

ensemble:             ## 3-seed PRESAGE/XGBoost/CatBoost ensembles + hybrid (needs make seed-variance first); feeds Fig. 20
	$(PY) scripts/run_ensemble.py

figures:              ## publication figures (PDF + 600 dpi PNG) from the result files; run after the experiment targets
	$(PY) scripts/make_publication_figures.py
	$(PY) scripts/make_advanced_figures.py
