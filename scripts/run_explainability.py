"""Phase 15: SHAP-based reasons, counterfactual explanations, faithfulness (deletion) test and a GNNExplainer demonstration."""
import json, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score
from presage.data.loaders import load_sim
from presage.evaluation.metrics import temporal_split, best_threshold
from presage.explainability.explain import Explainer, graph_evidence
from presage.utils.common import ROOT, get_logger, set_seed

log = get_logger("explain"); set_seed(42)
df, acc = load_sim(); X = pd.read_parquet(ROOT / "data/processed/sim_v1_features.parquet"); g = json.load(open(ROOT / "data/processed/sim_v1_feature_groups.json"))
cols = g["txn"] + g["behaviour"] + g["beneficiary"] + g["device"] + g["graph"]
cust = df.src_is_customer.values == 1; tr0, va0, te0 = temporal_split(df.day.values); tr, va, te = tr0 & cust, va0 & cust, te0 & cust
y = df.is_app_fraud.values
m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=20, tree_method="hist", n_jobs=2, verbosity=0, random_state=0).fit(X.loc[tr, cols].values, y[tr])
pv = m.predict_proba(X.loc[va, cols].values)[:, 1]; thr = best_threshold(y[va], pv, df.amount.values[va], "f2")
pt = m.predict_proba(X.loc[te, cols].values)[:, 1]; log.info("surrogate test PR-AUC %.3f threshold %.3f", average_precision_score(y[te], pt), thr)
bg = X.loc[tr & (y == 0), cols].sample(5000, random_state=0)
ex = Explainer(m, cols, g, bg, thr)
Xt = X.loc[te, cols]; dt = df[te]
flag_fraud = np.where((pt >= thr) & (y[te] == 1))[0]
# 1) examples
conf_mules = set(acc.account[acc.is_mule == 1])      # for the DEMO evidence view only (true mules); production uses confirmed reports
lines = ["# Example explanations (simulated data)\n", "SHAP attributions describe the model, not causal truth. Counterfactuals are model-based what-ifs.\n"]
rng = np.random.default_rng(0)
for i in rng.choice(flag_fraud, min(6, len(flag_fraud)), replace=False):
    row = Xt.iloc[i]; out = ex.explain(row, 5); r = dt.iloc[i]
    lines.append(f"## Payment {int(r.txn_id)} ({r.scam_type}) risk {out['probability']:.2f}\n")
    for q in out["reasons"]: lines.append(f"- [{q['level']}] {q['text']} (SHAP {q['shap']:+.2f})")
    lines.append(f"- Counterfactual: {out['counterfactual']['text']}\n")
# 2) counterfactual statistics over all flagged frauds (sample)
sel = rng.choice(flag_fraud, min(300, len(flag_fraud)), replace=False); cf = [ex.counterfactual(Xt.iloc[i]) for i in sel]
flip = np.mean([c["flipped"] for c in cf]); n_edits = np.mean([len(c["edits"]) for c in cf if c["flipped"]]) if any(c["flipped"] for c in cf) else np.nan
# 3) faithfulness (deletion) on flagged rows
faith = ex.faithfulness(Xt.iloc[np.where(pt >= thr)[0]], k=5, n=300)
# 4) reason coverage: share of flagged frauds with >=1 reason at each level
cov = {k: 0 for k in ["transaction", "behaviour", "beneficiary", "graph", "device"]}
for i in sel[:150]:
    e = ex.explain(Xt.iloc[i], 5)
    for lv in {q["level"] for q in e["reasons"]}: cov[lv] += 1
cov = {k: v / 150 for k, v in cov.items()}
metrics = dict(surrogate_test_pr_auc=float(average_precision_score(y[te], pt)), counterfactual_flip_rate=float(flip), mean_edits_when_flipped=float(n_edits), faithfulness=faith,
               share_of_flagged_frauds_with_reason_at_level=cov, n_flagged_frauds=int(len(flag_fraud)))
json.dump(metrics, open(ROOT / "reports/results/explainability_metrics.json", "w"), indent=1); print(json.dumps(metrics, indent=1))
# 5) symbolic graph evidence for 3 flagged frauds
lines.append("## Graph evidence (symbolic, strictly pre-payment)\n")
for i in flag_fraud[:3]:
    r = dt.iloc[i]; ev = graph_evidence(df, acc, r, conf_mules, r.ts); lines.append(f"- payment {int(r.txn_id)} -> beneficiary {int(r.dst)}: {json.dumps(ev)}")
# 6) GNNExplainer demonstration on a mule-node classifier (single snapshot; demonstration of the method, not a headline result)
try:
    import torch
    from torch_geometric.explain import Explainer as PyGExplainer, GNNExplainer
    from presage.graph.snapshots import build_snapshots
    from presage.models.gnn import make_encoder
    snaps, _ = build_snapshots(df, acc); s = snaps[22]; ei = torch.cat([s.pay_ei, s.pay_ei.flip(0)], 1)
    ym = torch.tensor(acc.is_mule.values, dtype=torch.float32); active = (s.x[:, -5] > 0)                      # in-degree feature > 0 (has received)
    class NodeNet(torch.nn.Module):
        def __init__(self): super().__init__(); self.enc = make_encoder("sage", s.x.shape[1], 32); self.out = torch.nn.Linear(32, 1)
        def forward(self, x, edge_index):
            class S_: pass
            q = S_(); q.x = x; q.pay_ei = edge_index[:, : edge_index.shape[1] // 2] if False else edge_index; return self.out(self.enc_forward(x, edge_index)).squeeze(-1)
        def enc_forward(self, x, edge_index):
            h = torch.relu(self.enc.inp(x))
            for c in self.enc.convs: h = torch.relu(c(h, edge_index)) + h
            return h
    net = NodeNet(); opt = torch.optim.Adam(net.parameters(), 5e-3); idx = torch.where(active)[0]
    for _ in range(60):
        out = net(s.x, ei); loss = torch.nn.functional.binary_cross_entropy_with_logits(out[idx], ym[idx], pos_weight=torch.tensor(10.0)); opt.zero_grad(); loss.backward(); opt.step()
    ge = PyGExplainer(model=net, algorithm=GNNExplainer(epochs=60), explanation_type="model", node_mask_type="attributes", edge_mask_type="object",
                      model_config=dict(mode="binary_classification", task_level="node", return_type="raw"))
    targets = [int(i) for i in idx[(ym[idx] == 1) & (torch.sigmoid(net(s.x, ei))[idx] > 0.5)][:2]]
    lines.append("\n## GNNExplainer demonstration (mule-node classifier on one snapshot)\n")
    for t in targets:
        e = ge(s.x, ei, index=t); em = e.edge_mask; top = torch.topk(em, 5).indices
        lines.append(f"- node {t}: top-5 influential edges (src,dst,importance) = " + "; ".join(f"({int(ei[0, j])},{int(ei[1, j])},{float(em[j]):.2f})" for j in top))
    lines.append("- Caveat: edge masks are optimisation artefacts of GNNExplainer for this model; they are evidence to inspect, not proof of causation.")
except Exception as e:
    lines.append(f"\nGNNExplainer demo skipped: {e}")
open(ROOT / "reports/results/explanations_examples.md", "w").write("\n".join(lines))
