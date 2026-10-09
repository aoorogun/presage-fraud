"""Generate the 16 tutorial notebooks (nbformat) and optionally execute them: python scripts/build_notebooks.py [--execute]"""
import sys, nbformat as nbf
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; NB = ROOT / "notebooks"; NB.mkdir(exist_ok=True)
SETUP = """import sys, pathlib, json, warnings; warnings.filterwarnings('ignore')
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
from IPython.display import Image, display
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)
plt.rcParams.update({'figure.dpi': 100, 'axes.spines.top': False, 'axes.spines.right': False})
def table(name):
    p = ROOT / 'reports' / 'tables' / name
    return pd.read_csv(p) if p.exists() else None"""
def md(s): return nbf.v4.new_markdown_cell(s.strip())
def code(s): return nbf.v4.new_code_cell(s.strip())
NBS = {}

NBS["01_data_audit"] = [
md("""# 01 Data audit
**Question.** Which datasets can we legitimately use, what do they support, and what can they NOT support?

Findings up front: no open dataset carries APP labels *with* device/session/beneficiary-registration signals, so PRESAGE uses (a) three real public datasets as independent benchmarks (not APP-labelled) and (b) a clearly synthetic APP/mule world (`sim_v1`). Dataset cards live in `research/dataset_cards/`."""),
code(SETUP),
md("## Real datasets (downloaded by `scripts/download_data.py`)"),
code("""raw = ROOT / 'data' / 'raw'
print({p.name: f'{p.stat().st_size/1e6:.0f} MB' for p in raw.glob('*') if p.is_file()})
ps = pd.read_csv(raw / 'paysim.csv', usecols=['step','type','amount','isFraud'])
print('PaySim rows', len(ps), 'fraud rate %.4f%%' % (100*ps.isFraud.mean()), 'hours', ps.step.max())
print(ps.groupby('type').agg(rows=('amount','size'), fraud=('isFraud','sum'), mean_amount=('amount','mean')).round(1))"""),
code("""ibm = pd.read_csv(raw / 'HI-Small_Trans.csv', usecols=['Timestamp','Payment Format','Is Laundering','From Bank'])
print('IBM AML HI-Small rows', len(ibm), 'laundering %.4f%%' % (100*ibm['Is Laundering'].mean()), ibm.Timestamp.min(), '->', ibm.Timestamp.max())
print(ibm.groupby('Payment Format')['Is Laundering'].agg(['size','sum','mean']).round(5))
del ibm"""),
code("""cl = pd.read_csv(raw / 'elliptic_txs_classes.csv'); print(cl['class'].value_counts())
ed = pd.read_csv(raw / 'elliptic_txs_edgelist.csv'); print('Elliptic edges', len(ed))"""),
md("""## What each dataset supports
| Dataset | txn history | graph | temporal | AML/mule-like | APP labels | device/session |
|---|---|---|---|---|---|---|
| PaySim | yes | sender-recipient | 743 hourly steps | no | **no** | no |
| IBM AML HI-Small | yes | account graph, cross-bank | 18 days | **laundering networks** | **no** | no |
| Elliptic | graph of transactions | yes | 49 steps | illicit flows | **no** | no |
| sim_v1 (synthetic) | yes | yes + device/ip | 120 days | **mule networks** | **yes (synthetic)** | yes (synthetic) |

Rule followed throughout: results on PaySim/IBM/Elliptic are reported as transaction-fraud/AML benchmarks, never as APP fraud."""),
code("""x = table('real_datasets.csv'); print(x.round(3).to_string(index=False) if x is not None else 'run scripts/run_real_datasets.py')"""),
]

NBS["02_eda"] = [
md("# 02 Exploratory analysis of the simulated APP world\nGoal: see what makes scam payments differ from legitimate payments, and how the world drifts."),
code(SETUP),
code("""from presage.data.loaders import load_sim, load_features
df, acc = load_sim(); X = load_features(); cust = df.src_is_customer == 1
print(df.shape, 'victim payments:', int(df.is_app_fraud.sum()), 'prevalence 1:%d' % round(1/df[cust].is_app_fraud.mean()))
print(df.txn_type.value_counts().to_dict()); print(df[df.is_app_fraud==1].scam_type.value_counts().to_dict())"""),
code("""fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
d = df[cust]; fr = d[d.is_app_fraud==1]
daily = d.groupby('day').is_app_fraud.sum(); ax[0].bar(daily.index, daily.values); ax[0].set_title('scam payments per day'); ax[0].axvline(100, c='r', ls='--'); [ax[0].axvspan(a,b,color='orange',alpha=.15) for a,b in ((20,35),(90,105))]
ax[1].hist(np.log10(d.amount), bins=50, alpha=.6, density=True, label='legit'); ax[1].hist(np.log10(fr.amount), bins=50, alpha=.6, density=True, label='scam'); ax[1].set_title('log10 amount'); ax[1].legend()
ax[2].hist((fr.dst_age_days).clip(upper=800), bins=40, alpha=.6, density=True, label='scam beneficiary'); ax[2].hist(d.dst_age_days.clip(upper=800), bins=40, alpha=.5, density=True, label='all'); ax[2].set_title('beneficiary age (days)'); ax[2].legend(); plt.tight_layout()"""),
md("**Reading the plots.** Daily scam volume shows the recurring campaigns (orange) and the sudden-drift jump after day 100 (structuring creates more, smaller payments). Amounts and beneficiary age overlap heavily with legitimate behaviour, which is exactly the APP difficulty: no single feature is decisive."),
code("""from sklearn.metrics import roc_auc_score
cols = [c for c in X.columns if c != 'txn_id']
auc = pd.Series({c: roc_auc_score(df.is_app_fraud[cust], np.nan_to_num(X.loc[cust.values, c])) for c in cols})
print((auc - .5).abs().sort_values(ascending=False).head(12).index.map(lambda c: f'{c}: {auc[c]:.3f}').tolist())"""),
code("""print(fr.groupby('scam_type').agg(n=('amount','size'), median_amount=('amount','median'), mean_session_s=('session_s','mean'), mean_failed_adds=('failed_ben_adds','mean'), mobile_share=('channel', lambda s:(s=='mobile').mean())).round(2))"""),
]

NBS["03_app_simulation"] = [
md("""# 03 APP / mule simulation layer
Why simulate: there is no credible open APP-labelled dataset with device, session and beneficiary signals. The simulator (`src/presage/simulation/app_world.py`) is a discrete-event model. **Everything it outputs is synthetic.**

We run a tiny world here to see the mechanics; the full world (`sim_v1`) is built by `make data`."""),
code(SETUP),
code("""from presage.simulation.app_world import simulate, SimConfig
small = SimConfig(n_customers=800, n_payees=1500, days=60, n_mule_networks=20, scam_rate_per_day=2.0, recurring_windows=((10, 20),), sudden_drift_day=45)
df, acc = simulate(small); print(df.shape); df.head(3)"""),
md("""### Anatomy of one scam
A victim (genuine, authenticated) pays a tier-1 mule; the mule forwards funds to tier-2 after a random holding time, then tier-3 cashes out."""),
code("""net = df[df.is_app_fraud==1].net_id.value_counts().index[0]
flow = df[df.net_id==net].sort_values('ts')[['ts','src','dst','amount','txn_type','scam_type']]
flow['hours_since_first'] = ((flow.ts - flow.ts.min())/3600).round(1); flow.head(12)"""),
md("""### What is original vs injected
| Element | Origin |
|---|---|
| All rows, accounts, devices, IPs, sessions | synthetic |
| Real datasets (PaySim, IBM AML, Elliptic) | untouched, used separately; never blended into the simulation |
| Scam signals (new beneficiary, session length, failed adds, structuring, mule fan-in/out, fast pass-through) | injected by typology parameters |
Parameters are plausible assumptions, not calibrated to a bank."""),
code("""sig = df[df.src_is_customer==1].assign(scam=lambda d: d.is_app_fraud==1).groupby('scam').agg(median_amount=('amount','median'), mean_session=('session_s','mean'), mean_failed=('failed_ben_adds','mean'), dst_age_median=('dst_age_days','median'))
sig"""),
code("""from presage.simulation.app_world import SCAM_TYPES; print(SCAM_TYPES)"""),
]

NBS["04_feature_engineering"] = [
md("""# 04 Leakage-safe feature engineering
Rule: a feature for a payment at time *t* may use only events with time < *t*. We implement this with one primitive (`EventIndex`) and prove it with a **prefix-invariance test**: features computed on the first half of the data must equal those computed with all data."""),
code(SETUP),
code("""from presage.features.windows import EventIndex
import numpy as np
# toy: three events of one account at t=10,20,20 (values 1,2,4). A query at t=20 may only see the t=10 event.
ev = EventIndex(np.array([1,1,1]), np.array([10.,20.,20.]), dict(v=np.array([1.,2.,4.])))
print('sum before t=20:', ev.sum('v', np.array([1]), np.array([20.]), np.inf), '(expected 1.0)')"""),
code("""from presage.data.loaders import load_sim
from presage.features.temporal import build_features
df, acc = load_sim(); X, groups = build_features(df, acc)
print({k: len(v) for k, v in groups.items()}, 'total', X.shape[1]-1)
print(groups['beneficiary'][:12])"""),
md("### The leakage test (the same logic is in `tests/test_leakage.py`)"),
code("""cut = df.ts.quantile(.5); pre = df[df.ts < cut].copy(); Xp, _ = build_features(pre, acc)
a = X.iloc[:len(pre)].drop(columns='txn_id').values; b = Xp.drop(columns='txn_id').values
bad = [c for j, c in enumerate(X.columns[1:]) if not np.allclose(a[:, j], b[:, j], equal_nan=True, atol=1e-3)]
print('rows compared:', len(pre), '| features that change when the future is removed:', bad)"""),
md("""Empty list = no feature uses the future. (An earlier version of the rolling std failed this test only through floating-point cancellation; it was fixed with group-centred prefix sums and a variance floor.)

Windows: 10 min, 1 h, 6 h, 24 h, 7 d, 30 d with count, sum, mean, std, max and new-beneficiary count; plus recipient novelty, time-of-day deviation, balance share, device/IP novelty, and beneficiary-side fan-in, fan-out, pass-through and holding time."""),
code("""x = X.assign(y=df.is_app_fraud.values)[df.src_is_customer.values==1]
x.groupby('y')[['pct_balance','amt_to_hist_mean','new_beneficiary','dst_age_days','dst_fanin_new_24h','dst_pass_through_24h','session_z','new_device']].mean().T.round(3)"""),
]

NBS["05_graph_construction"] = [
md("""# 05 Temporal heterogeneous graph construction
Node types: account, device, ip. Edge types: pays, uses (device), from (ip), each with reverse edges. A snapshot at cutoff *c* contains only edges strictly before *c*; a payment in [c, c+4 days) is scored with that snapshot, so it never sees itself or the future."""),
code(SETUP),
code("""from presage.simulation.app_world import simulate, SimConfig
from presage.graph.snapshots import build_snapshots
df, acc = simulate(SimConfig(n_customers=800, n_payees=1500, days=60, n_mule_networks=20, scam_rate_per_day=2.0, recurring_windows=((10, 20),), sudden_drift_day=45))
snaps, snap_of = build_snapshots(df, acc, step_days=4.0)
print(pd.DataFrame([dict(k=s.k, day=s.cutoff/86400, pay_edges=s.pay_ei.shape[1], device_edges=s.dev_ei.shape[1], ip_edges=s.ip_ei.shape[1]) for s in snaps]).iloc[::3])"""),
md("### Strictness check\nThe edge-recency attribute is `log1p(days between cutoff and the edge's last payment)`, so it must be >= 0 for every edge."),
code("""print('min recency attr across snapshots:', min(float(s.pay_ea[:,2].min()) for s in snaps if s.pay_ea.shape[0]))"""),
md("### Graph-derived features (PageRank, clustering, components, betweenness, confirmed-mule neighbours, ...)"),
code("""from presage.features.network import build_network_features
N = build_network_features(df, acc, step_days=4.0); print(N.shape); N.describe().T[['mean','50%','max']].head(14).round(3)"""),
code("""import networkx as nx
s = snaps[10]; G = nx.Graph(); G.add_edges_from(s.pay_ei.T.tolist())
deg = np.array([d for _, d in G.degree()]); plt.hist(deg, bins=np.logspace(0, np.log10(deg.max()+1), 30)); plt.xscale('log'); plt.yscale('log'); plt.title('degree distribution, snapshot 10'); plt.xlabel('degree');"""),
md("Label-derived node features (neighbouring *confirmed* mules) are built only from reports that have already arrived by the cutoff (payment time + report delay), so label delay is respected."),
]

NBS["06_tabular_baselines"] = [
md("# 06 Tabular baselines (Model Family A) and anomaly detectors (Family B)\nChronological split: first 60% of days train, next 20% validation, last 20% test. No random split, no resampling before splitting; imbalance via class weights; thresholds chosen on validation only."),
code(SETUP),
code("""import json, time
from sklearn.metrics import average_precision_score
from presage.data.loaders import load_sim, load_features
from presage.evaluation.metrics import temporal_split
from presage.models.tabular import make_supervised
df, acc = load_sim(); X = load_features(); g = json.load(open(ROOT/'data/processed/sim_v1_feature_groups.json'))
cust = df.src_is_customer.values==1; tr0,va0,te0 = temporal_split(df.day.values); tr, te = tr0&cust, te0&cust; y = df.is_app_fraud.values
sets = {'txn_only': g['txn'], '+behaviour': g['txn']+g['behaviour'], '+beneficiary/device': g['txn']+g['behaviour']+g['beneficiary']+g['device'], '+graph': g['txn']+g['behaviour']+g['beneficiary']+g['device']+g['graph']}
res = {}
for name, cols in sets.items():
    m = make_supervised('xgboost', 50).fit(np.nan_to_num(X.loc[tr, cols].values), y[tr]); res[name] = average_precision_score(y[te], m.predict_proba(np.nan_to_num(X.loc[te, cols].values))[:,1])
pd.Series(res).plot(kind='bar', title='XGBoost PR-AUC by information added (test)', ylim=(0,1)); print(res)"""),
md("Each information source lifts detection; transaction-only learning is far from sufficient for APP. Full results for five supervised models and five anomaly detectors (with bootstrap CIs):"),
code("""b = table('baselines.csv'); b[['feature_set','model','pr_auc','pr_auc_lo','pr_auc_hi','roc_auc','recall','precision','f2','mcc','fpr','latency_ms']].round(3)"""),
md("""Anomaly detectors (Isolation Forest, LOF, One-Class SVM, autoencoder, VAE) reach only 0.05-0.41 PR-AUC: unsupervised scores flag unusual payments, but APP scams are authorised and often unremarkable individually. They are best used as a complement for unseen fraud behaviour, not a replacement.

**LightGBM note.** An early class-weighted configuration diverged on training (PR-AUC 0.004) and was removed; the final unweighted configuration is less stable than XGBoost/CatBoost on some feature sets (see table). We report it as measured."""),
]

NBS["07_sequence_models"] = [
md("# 07 Sequence models (Family C)\nFor payment *t* the input is the sender's previous 10 payments plus the current one. We compare GRU, LSTM and a Transformer encoder, alone and combined with tabular features."),
code(SETUP),
code("""import json
from presage.data.loaders import load_sim, load_features
from presage.models.sequence import step_features, build_sequences, train_seq
from presage.models.graph_train import prep_tab
from presage.evaluation.metrics import temporal_split
from sklearn.metrics import average_precision_score
df, acc = load_sim(); X = load_features(); g = json.load(open(ROOT/'data/processed/sim_v1_feature_groups.json'))
cust = df.src_is_customer.values==1; tr0,va0,te0 = temporal_split(df.day.values); tr, va, te = tr0&cust, va0&cust, te0&cust; y = df.is_app_fraud.values.astype(np.float32)
steps = step_features(df, X); mu, sd = steps[tr].mean(0), steps[tr].std(0)+1e-6
seq, mask = build_sequences(df, (steps-mu)/sd, 10); print(seq.shape, 'positions filled (current payment last):', mask.mean(0).round(2))"""),
md("**Leak check by construction:** steps before the current one come only from earlier payments of the same sender (`build_sequences` sorts by sender then time)."),
code("""import torch; torch.set_num_threads(2)
out = {}
for kind in ['gru', 'transformer']:
    m, pred = train_seq(kind, seq, mask, y, tr, va, epochs=3, verbose=False)
    out[kind] = average_precision_score(y[te], pred(np.where(te)[0]))
print('sequence-only test PR-AUC (3 epochs, demo):', out)"""),
md("Sequence-only models see behaviour but not the beneficiary's network state, so they trail the tabular+graph models. Their value is as the behaviour stream inside PRESAGE (`z_beh`). Full-length comparison is in the PRESAGE ablation (`no_beh`)."),
code("""a = table('presage_full_s42.csv'); a"""),
]

NBS["08_graph_models"] = [
md("""# 08 Static and heterogeneous graph models (Families D, E)
Encoders: GCN, GraphSAGE, GAT (static, payment edges) and R-GCN, HGT, HAN (heterogeneous: account/device/ip). Node2Vec is omitted: a static unsupervised embedding cannot embed accounts that appear after training, which is the common case in this task; GraphSAGE/GCN are the stronger static baselines.

Fair comparison: every encoder feeds the same payment head together with the same tabular features. The row `none` is the identical head without any graph, so graph contribution is measured like for like."""),
code(SETUP),
code("""import json, time
from presage.data.prepare import prepare
from presage.models.graph_train import fit_edge_model
D = prepare(); A = D['A']; df = D['df']
t0 = time.time()
mdl, pv, pt, info = fit_edge_model('gcn', D['snaps'], D['snap_of'], A['src'], A['dst'], A['tab'], A['y_app'], D['tr'], D['va'], D['te'], epochs=5, verbose=True)
from sklearn.metrics import average_precision_score
print('GCN demo (5 epochs) test PR-AUC %.3f in %.0fs' % (average_precision_score(A['y_app'][D['te']], pt[D['te']]), time.time()-t0))"""),
code("""g = table('graph_models.csv'); g[['model','pr_auc','pr_auc_lo','pr_auc_hi','recall','precision','f2','mcc','train_s','latency_ms','n_params']].round(3)"""),
md("Reading: graph encoders add information over the tabular-only twin. Compare against the XGBoost-on-engineered-graph-features rows in notebook 06: engineered graph features plus gradient boosting is a very strong baseline, and learned graph encoders do not automatically beat it."),
]

NBS["09_temporal_graph_models"] = [
md("""# 09 Temporal graph models (Family F)
Implemented because their temporal mechanism is distinct and testable on our snapshot stream:
- **TGAT-style**: attention with Time2Vec encodings of edge recency;
- **EvolveGCN-O**: GCN weights evolved by a GRU along the snapshot sequence;
- **PRESAGE graph encoder**: HGT over recency-split relations (`pays_recent`/`pays_old`) with a per-account GRU memory across snapshots.

Not implemented, on purpose: TGN/DyRep/JODIE/GraphMixer need continuous-time event batching and memory modules whose incremental value over TGAT + snapshot memory could not be separated on a 120-day synthetic stream; adding them would inflate the model count, not the insight. They are listed as next experiments."""),
code(SETUP),
code("""g = table('graph_models.csv'); g[g.model.isin(['none','gcn','sage','hgt','tgat','evolvegcn'])][['model','pr_auc','pr_auc_lo','pr_auc_hi','recall','precision','f2','train_s']].round(3)"""),
code("""from presage.models.gnn import Time2Vec
import torch
t2v = Time2Vec(8); x = torch.linspace(0, 6, 5); print(t2v(x).shape, 'time encoding of recency (log-days) -> 8 dims')"""),
md("RQ2/3 in this study: do temporal encoders beat static ones? Check the table above and the PRESAGE `no_temporal` ablation (notebook 14) before drawing a conclusion; differences between encoders are often inside the confidence intervals."),
]

NBS["10_presage_model"] = [
md("""# 10 The PRESAGE model
Streams: **z_txn** (MLP on payment/velocity/beneficiary features), **z_beh** (GRU over the sender's previous payments), **z_graph** (temporal heterogeneous encoder), **z_mule** (mule module on the beneficiary embedding). Fusion: concat, gated or attention. Heads: APP fraud (A), beneficiary mule (B), fraud-network membership (C); uncertainty (D) by MC-dropout."""),
code(SETUP),
code("""from presage.models.presage import Presage, PresageConfig
import torch
cfg = PresageConfig(); m = Presage(cfg, d_node=11, d_tab=70, d_seq=11, n_seq=11, ben_idx=list(range(10)))
print(sum(p.numel() for p in m.parameters()), 'parameters'); print(cfg)"""),
code("""from presage.data.prepare import prepare
D = prepare(); A, snaps = D['A'], D['snaps']
k = 20; idx = np.where(D['snap_of']==k)[0][:256]
mdl = Presage(cfg, snaps[0].x.shape[1], A['tab'].shape[1], A['seq'].shape[2], A['seq'].shape[1], D['ben_idx'])
out = mdl(snaps[k], torch.tensor(A['src'][idx]), torch.tensor(A['dst'][idx]), torch.tensor(A['tab'][idx]), torch.tensor(A['seq'][idx]), torch.tensor(A['mask'][idx]))
print({k_: (tuple(v.shape) if v is not None else None) for k_, v in out.items()}); print('gate weights (txn,beh,graph,mule) mean:', out['weights'].mean(0).detach().numpy().round(3), '(untrained)')"""),
md("""Training uses chronological snapshots: for each snapshot the graph is encoded once, the account memory is advanced, and payments in that window are scored and back-propagated (negatives subsampled with importance weights; positives kept). Validation PR-AUC picks the epoch. See `scripts/run_presage.py`."""),
code("""a = pd.concat([table(f) for f in ['presage_full_s42.csv','presage_concat_attention_transformer_seq_s42.csv'] if table(f) is not None]); a.round(3)"""),
]

NBS["11_concept_drift"] = [
md("""# 11 Concept drift
Four drifts are built into the world: **sudden** (day 100: structuring into smaller payments), **gradual** (shrinking scam share-of-balance), **recurring** (campaign windows days 20-35 and 90-105), **feature** (fraud moves desktop to mobile, days 50-110)."""),
code(SETUP),
code("""from presage.drift.detectors import run_detector, psi
rng = np.random.default_rng(0); stream = np.concatenate([(rng.random(1500)<.1),(rng.random(1500)<.6)]).astype(float)
for d in ['adwin','ddm','eddm','page_hinkley','kswin']: print(d, [round(a) for a in run_detector(d, stream, np.arange(3000.))][:3])"""),
md("### Detector benchmark on the real experiment (miss stream of a static model; alarms are raised when labels confirm)"),
code("""d = table('drift_detectors.csv'); d[d.weekly_values.isna()].drop(columns='weekly_values') if d is not None else 'run scripts/run_drift.py --part detect'"""),
code("""from IPython.display import Image, display
p = ROOT/'reports'/'figures'/'fig_drift_monitors.png'; display(Image(str(p))) if p.exists() else print('run make_figures.py')"""),
md("### Adaptation strategies (walk-forward, weekly blocks from day 60, 3-day label delay)"),
code("""s = table('drift_adaptation.csv'); s.round(3) if s is not None else 'run scripts/run_drift.py --part adapt'"""),
code("""p = ROOT/'reports'/'figures'/'fig_drift_curves.png'; display(Image(str(p))) if p.exists() else None"""),
md("""Metrics: mean/pre/post-drift PR-AUC, recall at 1% FPR, retrains and compute seconds, first alarm day (adaptation delay >= label delay), and `probe_pr_auc_final` (final model scored on an early-period probe; a drop relative to `static` indicates forgetting)."""),
]

NBS["12_federated_learning"] = [
md("""# 12 Privacy-preserving cross-bank learning
Five simulated banks; each sees only payments it originates or receives, so mule networks spanning banks are fragmented. Horizontal FL (same features, different customers) is simulated in one process.

**What this is not:** federated learning alone does not guarantee privacy (updates can leak). Secure aggregation here is a simulation of pairwise masking; differential privacy is client-level with only five clients, so the formal guarantee is weak and reported for the trade-off only."""),
code(SETUP),
code("""import torch
from presage.federated.fl import secure_aggregate, rdp_epsilon
ups = [torch.randn(8) for _ in range(5)]; print('masks cancel:', torch.allclose(secure_aggregate(ups), torch.stack(ups).sum(0), atol=1e-3))
print({s: round(rdp_epsilon(s, 15), 1) for s in (0.5, 1.0, 2.0)}, '<- epsilon (delta=1e-5) after 15 rounds by noise multiplier')"""),
code("""f = table('federated.csv'); f.round(3) if f is not None else 'run scripts/run_federated.py'"""),
code("""p = ROOT/'reports'/'figures'/'fig_federated.png'; display(Image(str(p))) if p.exists() else None"""),
md("""**Vertical FL and federated graph learning.** Here the parties hold *different rows of the same schema* (horizontal). Real inter-bank collaboration on graphs is closer to vertical/federated *graph* learning where cross-bank edges cannot be pooled; established approaches (e.g. federated GNN training with neighbour-feature exchange, or privacy-preserving entity linkage) are future work and not claimed here. Our `centralised_full_visibility` row is an oracle showing how much the unseen cross-bank edges are worth."""),
]

NBS["13_explainability"] = [
md("""# 13 Explainability
Investigator-facing reasons at four levels (transaction, behaviour, beneficiary, graph) from SHAP attributions plus templates; actionable counterfactuals via coherent edits (e.g. "beneficiary older than 90 days"); a deletion-based faithfulness test. Attention weights are never presented as explanations. SHAP describes the model, not the world."""),
code(SETUP),
code("""import json, xgboost as xgb
from presage.data.loaders import load_sim, load_features
from presage.evaluation.metrics import temporal_split, best_threshold
from presage.explainability.explain import Explainer
df, acc = load_sim(); X = load_features(); g = json.load(open(ROOT/'data/processed/sim_v1_feature_groups.json'))
cols = g['txn']+g['behaviour']+g['beneficiary']+g['device']+g['graph']; cust = df.src_is_customer.values==1
tr0,va0,te0 = temporal_split(df.day.values); tr, va, te = tr0&cust, va0&cust, te0&cust; y = df.is_app_fraud.values
m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.08, subsample=.8, colsample_bytree=.8, scale_pos_weight=20, tree_method='hist', n_jobs=2, verbosity=0).fit(X.loc[tr, cols].values, y[tr])
thr = best_threshold(y[va], m.predict_proba(X.loc[va, cols].values)[:,1], df.amount.values[va], 'f2')
ex = Explainer(m, cols, g, X.loc[tr & (y==0), cols].sample(3000, random_state=0), thr)
pt = m.predict_proba(X.loc[te, cols].values)[:,1]; i = np.where((pt>=thr)&(y[te]==1))[0][3]
out = ex.explain(X.loc[te, cols].iloc[i], 5); print('risk %.2f' % out['probability']); [print(' -', r['level'], '|', r['text']) for r in out['reasons']]; print(out['counterfactual']['text'])"""),
code("""print(json.dumps(json.load(open(ROOT/'reports/results/explainability_metrics.json')), indent=1) if (ROOT/'reports/results/explainability_metrics.json').exists() else 'run scripts/run_explainability.py')"""),
code("""p = ROOT/'reports/results/explanations_examples.md'; print(p.read_text()[:3500] if p.exists() else '')"""),
md("Faithfulness: removing the top-5 SHAP features lowers risk more than removing five random features (see `faithfulness`). That supports, but does not prove, that the explanations track what the model uses."),
]

NBS["14_ablation"] = [
md("# 14 Ablation study\nEach row removes one PRESAGE component (single seed, 12 epochs, same chronological split). Differences are interpreted against bootstrap CIs, not raw point estimates."),
code(SETUP),
code("""import glob
rows = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT/'reports/tables/presage_*_s42.csv'))]).drop_duplicates('variant')
rows = rows.sort_values('pr_auc', ascending=False); full = rows.loc[rows.variant=='full','pr_auc'].iloc[0]
rows['delta_vs_full'] = rows.pr_auc - full; rows[['variant','pr_auc','pr_auc_lo','pr_auc_hi','delta_vs_full','recall','precision','f2','mule_pr_auc','train_s']].round(3)"""),
code("""p = ROOT/'reports'/'figures'/'fig_ablation.png'; display(Image(str(p))) if p.exists() else None"""),
md("""Not run as separate rows and why: *without drift adaptation* and *without federated learning* are properties of the training/deployment protocol rather than the network, so they are measured in notebooks 11 (static vs adaptive strategies) and 12 (local vs federated)."""),
]

NBS["15_statistical_testing"] = [
md("# 15 Statistical testing\nPaired block bootstrap (blocks = days) for PR-AUC differences, DeLong for ROC-AUC, McNemar at a fixed alert budget, Friedman + Nemenyi across test days, Holm correction across comparisons."),
code(SETUP),
code("""t = table('stat_tests_vs_presage.csv'); t.round(4) if t is not None else 'run scripts/final_evaluation.py'"""),
code("""import json; p = ROOT/'reports/tables/friedman_nemenyi.json'; print(json.dumps(json.load(open(p)), indent=1) if p.exists() else '')"""),
md("""Interpretation guide: `delta_pr_auc` is PRESAGE minus the comparator; a CI containing 0 (or Holm-adjusted p above 0.05) means the difference is not distinguishable from noise with this test set (about 300 positives). Statistical significance is not the same as operational value; see the cost analysis in notebook 16."""),
]

NBS["16_final_evaluation"] = [
md("# 16 Final evaluation"),
code(SETUP),
code("""m = table('master_results.csv'); m.round(3).head(20) if m is not None else 'run scripts/final_evaluation.py'"""),
md("### APP-specific results (simulated money; operating policy chosen by expected cost on validation data)"),
code("""a = table('app_specific_metrics.csv'); a.T if a is not None else None"""),
md("### Error analysis (`reports/results/error_analysis.csv`)"),
code("""e = pd.read_csv(ROOT/'reports/results/error_analysis.csv') if (ROOT/'reports/results/error_analysis.csv').exists() else None; e"""),
md("### Streaming demo"),
code("""p = ROOT/'reports/results/streaming_demo.json'; print(open(p).read() if p.exists() else 'run streaming/run_demo.py')"""),
md("### Cross-dataset / real-data benchmarks"),
code("""r = table('real_datasets.csv'); r.round(3) if r is not None else None"""),
code("""p = ROOT/'reports'/'figures'/'fig_pr_curves.png'; display(Image(str(p))) if p.exists() else None"""),
md("See `reports/implementation_findings.md` for the written summary of what was and was not established."),
]

ONLY = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
if ONLY: NBS = {k: v for k, v in NBS.items() if any(k.startswith(o) for o in ONLY)}
for name, cells in NBS.items():
    nb = nbf.v4.new_notebook(); nb.cells = cells; nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, NB / f"{name}.ipynb")
print("wrote", len(NBS), "notebooks")
if "--execute" in sys.argv:
    import subprocess
    for name in NBS:
        r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=1500", str(NB / f"{name}.ipynb")], capture_output=True, text=True, cwd=NB)
        print(name, "OK" if r.returncode == 0 else "FAILED\n" + r.stderr[-1500:])
