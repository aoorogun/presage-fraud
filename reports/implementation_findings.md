# PRESAGE-Fraud: implementation findings

Scope: end-to-end implementation and experiments only. No paper draft. Every number below was produced by code in this repository and is stored in `reports/tables/` or `reports/results/`. The experiment registry is `research/experiment_registry.csv` (MLflow runs in `mlflow.db`, not shipped).

## 0. Read this first

1. **All APP-fraud and mule results are on a synthetic world (`sim_v1`).** No open dataset provides APP labels together with device, session and beneficiary-registration signals. Results show how methods behave in a simulator I wrote, whose mechanisms (mule recruitment, holding times, scam typologies, drift) encode my assumptions. They are not evidence about real banks, and absolute numbers should not be expected to transfer.
2. PaySim, IBM AML and Elliptic are **not APP data**. They are reported as transaction-fraud, AML and illicit-bitcoin benchmarks only.
3. The simulator is only moderately hard. A first version was too easy (a single feature reached AUC about 0.97), was rebuilt, and the final one is still solved well by gradient boosting once beneficiary and graph features are present (PR-AUC about 0.98). Headline PR-AUCs near 0.98 mean the simulated task is nearly saturated, so differences between strong models are small and mostly within noise.
4. Single-seed results were found to be unreliable for the neural models (section 5). Where it matters I report three seeds.

## 1. Data

| Dataset | Rows | Positive rate | Use |
|---|---|---|---|
| sim_v1 (synthetic) | about 390k payments, 15,262 accounts (1,262 mules), 120 days | 0.243% (about 1:410); 950 victim payments | APP and mule experiments |
| PaySim | 6.36M | 0.129% | transaction-fraud benchmark |
| IBM AML HI-Small | 5.08M | 0.102% | laundering benchmark |
| Elliptic | 4,545 illicit, 42,019 licit, 157,205 unknown | n/a (unknowns excluded) | node classification benchmark |

Evaluation population for sim_v1: payments initiated by customers. Chronological split 60/20/20 by day: 188,381 / 62,789 / 63,283 rows with 484 / 156 / 310 frauds. Dataset cards are in `research/dataset_cards/`.

Known simulator quirks: 0.02% of payments have a slightly negative payee age (payee account opened shortly after the payment); some balances are negative. Both are retained and the API accepts them.

## 2. Leakage controls (tested)

- Every rolling feature for a payment at time t uses only events with time strictly before t (`features/windows.py`). `tests/test_leakage.py` verifies prefix invariance: features computed on a data prefix equal those computed with all data.
- Graph features come from 4-day snapshots; a snapshot at cutoff c contains only edges strictly before c.
- Confirmed-mule label features honour label delay (a mule is known only when payment time plus report delay is at most the cutoff).
- Thresholds are chosen on validation only. No SMOTE or resampling was used anywhere.
- Online feature store and batch features agree (`tests/test_online_parity.py`).

## 3. Baselines (test, simulated, PR-AUC with block-bootstrap 95% CI)

Feature staircase, XGBoost: transaction-only 0.262, plus behaviour 0.522, plus beneficiary and device (all tabular) 0.918, plus graph structure 0.948, plus label-derived graph features 0.977.
CatBoost: 0.252, 0.534, 0.951, 0.954, 0.976. Logistic regression on all features 0.502 (graph_full 0.768). LightGBM is unstable on some sets (0.335 on tabular+graph_struct, 0.943 on graph_full); I did not tune it further and report it as measured.

Unsupervised detectors trained on legitimate payments only (graph_full features): Isolation Forest 0.045, LOF 0.299, One-Class SVM 0.161, autoencoder 0.405, VAE 0.408 PR-AUC. They have high ROC-AUC (0.88 to 0.98) but poor precision at this prevalence.

The label-derived features ("neighbour is a confirmed mule") carry a large share of the last step from 0.948 to 0.977 and are only available after victims report, so that gain depends on label delay (section 7).

## 4. Graph models (20 epochs, same split, PR-AUC)

| Model | PR-AUC [95% CI] |
|---|---|
| none (tabular-only twin) | 0.874 [0.821, 0.905] |
| GCN | 0.921 |
| GraphSAGE | 0.924 |
| GAT | 0.918 |
| R-GCN | 0.928 |
| HGT | 0.959 [0.921, 0.976] |
| HAN | 0.914 |
| TGAT-style (snapshot) | 0.942 [0.904, 0.963] |
| EvolveGCN-O | 0.872 |

Heterogeneous and temporal encoders (HGT, TGAT-style) beat the tabular twin and static homogeneous GNNs. None beats XGBoost or CatBoost on graph_full features (0.977 and 0.976). Continuous-time models such as TGN were not implemented.

## 5. PRESAGE

Components: transaction MLP, sequence encoder, heterogeneous temporal graph encoder (HGT with recency-split relations and GRU memory), mule module, fusion, multi-task heads, MC-dropout uncertainty.

Three-seed comparison (`reports/tables/seed_variance.csv`, test PR-AUC, mean and standard deviation over seeds 42, 1, 2):

| Model | Test PR-AUC | Validation PR-AUC |
|---|---|---|
| PRESAGE, concat fusion | 0.979 (sd 0.002) | 0.958 (sd 0.005) |
| PRESAGE, default fusion ("full") | 0.948 (sd 0.029) | 0.941 (sd 0.020) |
| XGBoost, graph_full | 0.975 (sd 0.002) | 0.969 (sd 0.001) |
| CatBoost, graph_full | 0.975 (sd 0.003) | 0.954 (sd 0.003) |

Findings:
- PRESAGE with concat fusion is statistically tied with gradient boosting. Its 0.004 test advantage over XGBoost is the size of the seed noise, and on validation XGBoost is higher. **I do not claim PRESAGE outperforms gradient boosting.**
- The default fusion is seed-sensitive (0.919, 0.977, 0.947). The seed-42 run used in the first statistical tests happened to be its worst seed.
- Concat was chosen as the better fusion variant on validation as well as test (validation 0.957 vs 0.919 at seed 42), so the selection is not test-only, but the choice was made after seeing all variants.
- Gated and attention variants were run at seed 42 only (attention 0.943). A Transformer sequence encoder scored 0.406, almost certainly an optimisation failure under the shared hyperparameters, not evidence about Transformers. It was not tuned.
- Mule head average precision is 0.99 for most variants (seed 42, full 0.996), measured on beneficiary accounts in the simulator.

Ablations (seed 42, test PR-AUC; full 0.919): no graph 0.874, no transaction branch 0.916, no multi-task 0.906, no mule module 0.921, no temporal memory 0.936, no beneficiary features 0.928, no behaviour branch 0.960. With single-seed sd up to 0.03 for this architecture, only "no graph" (a drop of about 0.045 with mule AP falling to 0.858) is clearly outside noise. Removing behaviour **raised** PR-AUC, which I interpret as noise or redundancy with the sequence encoder, not as a finding. Component contributions are therefore not established by these runs; multi-seed ablations are needed.

## 6. Statistical tests (seed-42 PRESAGE "full" vs others)

Paired block bootstrap, DeLong, McNemar, Holm correction (`reports/tables/stat_tests_vs_presage.csv`): seed-42 default PRESAGE is significantly worse than the top 8 models (PR-AUC differences of -0.03 to -0.06, Holm-adjusted p 0.02 to 0.03). Friedman test over 24 test days with at least 2 frauds: p = 9.5e-5; PRESAGE-full has the worst mean rank (4.90) among six models, concat 2.71, XGBoost 2.92, CatBoost 3.17 (Nemenyi critical difference 1.54). These tests describe the single weak seed and should be read together with section 5. I did not rerun the tests for the seed-averaged concat model.

## 7. Concept drift and label delay

Setting: walk-forward from day 60 to 120, 7-day blocks, XGBoost base learner, 3-day label delay, simulated sudden drift at day 100 plus recurring windows. The measured drift effect on a static model is small (PR-AUC 0.967 before, 0.958 after day 98).

Adaptation strategies (mean PR-AUC): static 0.964, scheduled 14-day retrain 0.976, sliding 30-day 0.983, incremental 0.966, replay buffer 0.982, drift-triggered ADWIN or DDM 0.970. Drift-triggered strategies never retrained in this setup (0 alarms), so they equal the static model after the initial fit. Sliding windows forgot old patterns (PR-AUC on a fixed early probe 0.82 vs 0.99 to 1.0 for others).

Detectors on the miss stream (alarms at label confirmation): ADWIN, EDDM, Page-Hinkley and KSWIN raised no alarm. DDM raised 5 alarms, 2 before the drift day, first post-drift alarm at day 108.5 (delay 8.5 days). Feature-distribution monitors (PSI, JS, Wasserstein on amount, balance share, mobile share, failed adds) stayed below conventional thresholds except Wasserstein on mobile share (one alarm, day 116). Conclusion: the simulated drift is mild and the tested detectors are mostly blind to it; this is a property of my simulator and of untuned detector settings.

Label delay (mean PR-AUC, supervised / pseudo-label / weak supervision / active): 1 day 0.988 / 0.989 / 0.978 / 0.989; 3 days 0.983 / 0.985 / 0.976 / 0.985; 7 days 0.967 / 0.974 / 0.953 / 0.982; 30 days 0.914 / 0.931 / 0.554 / 0.941. Performance degrades with delay, pseudo-labelling and active labelling help modestly at long delays, weak supervision collapses at 30 days. Active learning assumes a fixed analyst budget set in the code; the numbers depend on that assumption.

## 8. Federated learning (5 simulated banks, local-view features)

Test PR-AUC: local only 0.713, FedAvg 0.811, personalised FedAvg 0.816, FedProx (mu 0.05) 0.739, FedAvg with simulated secure aggregation 0.821 (identical in effect to FedAvg by construction, no cryptography was implemented), pooled centralised on local-view features 0.821, centralised with full visibility (oracle, not realisable) 0.929. Client-level DP with noise multipliers 0.5, 1.0, 2.0 (epsilon 67, 26, 11 by the RDP accountant) collapsed PR-AUC to about 0.01 under my clipping and round settings, so DP was not usable here; I did not tune it. Cross-bank frauds are 249 of 310 test frauds, which is why local-only views do poorly. Federated graph learning was not implemented.

## 9. Explainability (PRESAGE-serving surrogate, simulated)

- Surrogate model test PR-AUC 0.967.
- Deletion-based faithfulness (remove top-5 SHAP features): mean score drop 0.846, versus 0.078 for 5 random features (n = 277).
- Greedy counterfactuals flip the decision for 45% of flagged payments, with 1.33 feature edits on average when they succeed.
- Reason coverage among 267 flagged frauds: transaction 77%, behaviour 76%, graph 68%, beneficiary 43%, device 4%.
- In the example explanations, the top reason is often the "beneficiary already reported as a mule" label feature, which exists only after a victim report. Explanations describe the model, not causal truth. A GNNExplainer demo is included but was not quantitatively evaluated.

## 10. APP-specific outcomes (simulated test period, PRESAGE seed 42 "full" with a cost-optimised policy)

- 287 of 310 scam payments (92.6%) were flagged before payment.
- Mule accounts with scams in the test period: 79; 74 flagged at or before their first test-period victim payment; 75 before report confirmation; mean warning lead time 3.0 days. "First victim payment" is the first one inside the test window, not the account's first ever.
- Expected simulated loss prevented 208,369 of 222,217 (GBP, simulated); policy cost 18,099 vs 222,217 for no action. The intervention costs and stop-probability are my assumptions in `evaluation/policy.py`; the cost numbers are illustrative.
- Intervention rate 0.83%, hold rate 0.60%. The learned thresholds T2 and T3 coincided (0.0101), so the policy effectively used three levels, not four.
- Error analysis categories overlap, so shares can sum to more than 100%. Missed scams concentrate in the post-drift period (20 of 23 false negatives), cross-bank payments (17) and recruited genuine-customer mule accounts (17). False positives are dominated by large first-time transfers to new payees (38 of 237) and new-device payments (11).
- Subgroup check by institution (not demographic fairness): recall 0.87 to 0.98, false-positive rate 0.3% to 0.4%, intervention rate 0.7% to 1.1% (bank 4 has only 15 test frauds). No demographic attributes exist in the simulation and none were invented, so no demographic fairness analysis was done.

## 11. Real benchmark datasets (not APP)

| Dataset | Feature set | Test PR-AUC [CI] |
|---|---|---|
| PaySim (subsample, chronological) | transaction only | 0.987 [0.980, 0.995] |
| | plus behaviour and beneficiary | 0.989 |
| | plus graph | 0.988 |
| IBM AML HI-Small (enriched subsample) | transaction only | 0.631 |
| | plus behaviour and beneficiary | 0.985 |
| | plus graph | 0.979 |
| Elliptic (time split, labelled nodes) | local features | 0.782 [0.591, 0.866] |
| | local plus 1-hop aggregation and degree | 0.788 |

Caveats: PaySim is restricted to TRANSFER and CASH_OUT (the only types carrying fraud) and subsampled to 35% of senders, so its test prevalence (579 positives in 15,283 rows, about 3.8%) is far above the full-dataset 0.129%; it is also easily separable from transaction fields alone, so it says little about modelling. The IBM AML test set is a class-enriched subsample (126 test rows, 77 positives), so its PR-AUC is not comparable to the natural-prevalence figures above. Graph features did not help on PaySim or IBM in this setup. Cross-dataset generalisation (train on one, test on another) was not run.

## 12. Adversarial robustness (feature space, XGBoost, graph_full; clean PR-AUC 0.977)

Mimicry moves attacker-controllable features toward the legitimate median for test frauds. Victim-coached features (amount, balance share, hour, session, failed adds): PR-AUC 0.970, 0.962, 0.944, 0.884 at strength 0.25, 0.5, 0.75, 1.0. Mule camouflage (pass-through, fan-out, holding time): 0.978, 0.977, 0.975, 0.950. Both: 0.971, 0.962, 0.936, 0.825 (recall at 1% FPR from 0.990 to 0.910). Label poisoning (frauds relabelled legitimate in train): 10% 0.979, 30% 0.949, 50% 0.883. Random feature dropout at test time: 5% 0.957, 10% 0.943, 20% 0.908, 40% 0.679.
These perturbations are not constrained to be jointly realisable and use a fixed model, so they bound sensitivity and are not attacks on a deployed system. No gradient-based or graph-structure attacks were run.

## 13. Serving, streaming, monitoring

- Online feature store: p50 20 microseconds, p99 59 microseconds per event (feature computation only). Train and serve parity test passes.
- Serving model (35 features, three XGBoost seeds): test PR-AUC 0.877, below the offline models because it uses only features computable online without the graph snapshot pipeline. Mule model 0.915. API tests (health, scoring, explanation, auth) pass.
- Streaming demo (in-memory transport, 20,000 replayed events from day 90, 15,981 scored, simulated): throughput 37 events/s, latency p50 31 ms, p95 64 ms, p99 72 ms, 127 alerts, 63 of 63 scams intervened before payment, 76 holds with 78% precision. Throughput is dominated by per-event explanation and SQLite case writes in one Python process. The stream starts at day 90, which is before the test split begins (day 96), so six days overlap the validation period and 63 frauds is a small sample. The Kafka/Redpanda code path was written but **not executed** (no broker available).
- Docker, docker-compose (API, Redpanda, MLflow, Prometheus, Grafana) and a Prometheus text endpoint are provided but **not built or run** (no Docker daemon). Evidently-based monitoring was not implemented; drift monitoring uses the PSI, JS and Wasserstein code in `drift/detectors.py`.

## 14. Tests

pytest suite covers leakage, simulation invariants, metrics, statistics, policy, drift, federated learning, graph model smoke tests, online parity and the API. 31 tests pass in the final environment (the slow graph smoke tests were included). 16 tutorial notebooks execute end to end.

## 15. Limitations and not done

- Synthetic APP data only; simulator assumptions drive every APP conclusion.
- Seeds: 3 for PRESAGE-full, PRESAGE-concat, XGBoost and CatBoost; 1 for everything else. Ablations, fusion variants other than concat, graph models and baselines are single-seed.
- Hyperparameters were not tuned systematically (no search); LightGBM, the Transformer sequence encoder and DP settings are known weak points.
- Not implemented: TGN or other continuous-time models, federated graph learning, EWC continual learning for neural models, cross-dataset generalisation, explicit imbalance-ratio sweeps (1:100, 1:500, 1:1000) and cost-ratio sweeps, Evidently reports, executed Kafka and Docker paths.
- Literature: only 7 papers are Crossref-verified in `research/literature_matrix.csv`; the rest are in `research/literature_unverified.txt` and must be verified in the paper phase. `reports/tables/benchmark_matrix.csv` lists re-implemented methods with their original references; it contains no numbers taken from papers.
- Fairness: no demographic analysis (section 10).

## 16. Reproduction

`make setup data features test train-baselines train-graph train-presage ablation evaluate drift-experiment label-delay federated-experiment explain adversarial serve-train`. Seed variance numbers come from running `scripts/run_presage.py --variants full,concat --seed N` for N in 1, 2 plus the seed-42 runs. See README for Docker and streaming commands.
