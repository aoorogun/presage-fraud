# PRESAGE-Fraud

Research code for early detection of Authorised Push Payment (APP) fraud and money-mule networks, with a leakage-safe feature pipeline, tabular, sequence and graph baselines, a fused multi-task model (PRESAGE), drift and label-delay studies, federated learning, explainability, a scoring API and a streaming demo.

## Read this first
- **The APP/mule data is synthetic.** No open dataset carries APP-fraud labels together with device, session and beneficiary-registration signals. All APP results come from the simulator in `src/presage/simulation/app_world.py` (`sim_v1`). They show how methods behave in that simulated world and are not evidence about any real bank.
- PaySim, IBM AML (HI-Small) and Elliptic are used as independent benchmarks for transaction fraud, anti-money-laundering and illicit-bitcoin classification. They are **not** APP data and are never reported as APP detection.
- All splits are chronological. Features at time t use only events before t (`tests/test_leakage.py` checks prefix invariance). Resampling, if used, happens after the split.
- Reported numbers are in `reports/tables/` and summarised in `reports/implementation_findings.md`. Nothing is copied from papers.

## Layout
`src/presage/` simulation, features, models, drift, federated, explainability, serving. `scripts/` experiment entry points. `streaming/` producer, consumer and demo. `notebooks/` 16 tutorial notebooks. `research/` experiment registry, dataset cards, verified literature matrix. `tests/` pytest suite.

## Quick start
```
make setup && make data && make features && make test
make train-baselines train-graph train-presage ablation evaluate
make drift-experiment label-delay federated-experiment explain adversarial
make serve-train serve         # FastAPI on :8000
make stream-demo               # in-memory transport; add --kafka host:9092 for Redpanda
```
Docker: `docker compose up --build` (compose file provided, not executed in the build environment).

## Known limitations
See the limitations section of `reports/implementation_findings.md`.

## What is not shipped
Raw datasets, processed features, simulated parquet files, prediction arrays (`*.npy`, `*.npz`) and MLflow stores are excluded from the archive for size. `make data features` and the training targets regenerate them (the simulator is seeded). Tables and figures in `reports/` are included.

## Ensembles and Fig. 20

```bash
make seed-variance    # trains PRESAGE concat seeds 42, 1, 2 if missing (about 25 min each) and compares with GBDT
make ensemble         # 3-seed ensembles + PRESAGE/XGBoost hybrid -> reports/tables/ensemble_*.csv, reports/results/ensemble_preds.npz
make figures          # Figs 1-20 -> reports/figures/publication/ (Fig. 20 is skipped with a message if make ensemble has not run)
```
