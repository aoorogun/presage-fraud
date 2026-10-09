"""Verify candidate papers against the Crossref API and write research/literature_matrix.csv.

Only rows whose Crossref title similarity >= 0.88 are kept ('verified'). Descriptive columns (model, dataset, results...) are
left as 'to_extract' unless filled from the paper itself; NOTHING is filled from memory. Run:  python research/verify_papers.py
"""
import csv, difflib, json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

CANDIDATES = [  # (title, topic tag, relevance note)
    ("Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy", "payment fraud; concept drift; delayed labels", "Delayed-label and drift framing for fraud streams"),
    ("Learned lessons in credit card fraud detection from a practitioner perspective", "payment fraud; practice", "Practitioner lessons: temporal evaluation, alert precision"),
    ("Online payment fraud: from anomaly detection to risk management", "online banking fraud; cost-sensitive", "Cost-optimised ML for online banking payments"),
    ("Fraud Detection in Mobile Payment Systems using an XGBoost-based Framework", "mobile payment fraud; PaySim", "XGBoost + cost analysis on PaySim"),
    ("Financial Fraud Detection Based on Machine Learning: A Systematic Literature Review", "survey", "Landscape of ML fraud detection"),
    ("Considerations for Using Artificial Intelligence to Manage Authorized Push Payment (APP) Scams", "APP fraud; policy", "APP-scam framing and liability scenarios"),
    ("Machine Learning for Digital Payment Fraud Detection in Financial Institutions: A Review of Methods, Explainability, and Performance", "survey; APP", "Finds no peer-reviewed APP detection performance study"),
    ("Learning under Concept Drift: A Review", "concept drift", "Drift taxonomy for experiment design"),
    ("A survey on concept drift adaptation", "concept drift", "Adaptive learning strategies"),
    ("Learning from Time-Changing Data with Adaptive Windowing", "drift detection; ADWIN", "ADWIN detector"),
    ("Modeling Relational Data with Graph Convolutional Networks", "R-GCN", "Relational GNN baseline"),
    ("Heterogeneous Graph Transformer", "HGT", "Heterogeneous attention GNN used in PRESAGE graph encoder"),
    ("EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs", "temporal GNN", "Weight-evolving temporal GNN baseline"),
    ("Heterogeneous Graph Attention Network", "HAN", "Metapath attention baseline"),
    ("Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics", "AML; Elliptic", "Elliptic dataset origin"),
    ("Elliptic++: Elliptic Dataset Extension", "AML; graph dataset", "Extended Elliptic transaction/actor graph"),
    ("Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", "AML; synthetic data; IBM", "IBM AML synthetic dataset origin"),
    ("Federated Optimization in Heterogeneous Networks", "federated; FedProx", "FedProx baseline"),
    ("Communication-Efficient Learning of Deep Networks from Decentralized Data", "federated; FedAvg", "FedAvg baseline"),
    ("Deep Learning with Differential Privacy", "DP-SGD", "Differential privacy for training"),
    ("Overcoming catastrophic forgetting in neural networks", "continual learning; EWC", "EWC for drift adaptation"),
    ("Explainable Graph Neural Networks: GNNExplainer", "explainability", "placeholder-title-check"),
    ("GNNExplainer: Generating Explanations for Graph Neural Networks", "explainability; GNN", "Node/edge explanation for GNNs"),
    ("A Unified Approach to Interpreting Model Predictions", "explainability; SHAP", "SHAP"),
    ("Attention is not Explanation", "explainability; attention caveat", "Why attention weights are not used as explanations"),
    ("Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR", "counterfactual explanations", "Counterfactual explanation framing"),
    ("Temporal Graph Networks for Deep Learning on Dynamic Graphs", "TGN", "Temporal graph memory network"),
    ("Inductive Representation Learning on Temporal Graphs", "TGAT", "Time-encoding attention for temporal graphs"),
    ("node2vec: Scalable Feature Learning for Networks", "node2vec", "Static embedding baseline"),
    ("Graph Neural Network for Fraud Detection via Spatial-Temporal Attention", "GNN fraud; temporal", "Spatio-temporal attention for fraud graphs"),
    ("Federated Learning for Anti-Money Laundering", "federated; AML", "Federated AML framing"),
    ("Money Laundering Detection with Graph Neural Networks", "AML; GNN", "GNN AML"),
    ("Mule account detection", "money mule", "Mule account detection"),
]
OUT = Path(__file__).parent / "literature_matrix.csv"
COLS = ["paper", "authors", "year", "venue", "quartile", "DOI", "fraud_type", "dataset", "model", "graph_type", "temporal_modelling", "drift_handling",
        "explainability", "metrics", "reported_results", "limitations", "relevance_to_PRESAGE", "verification_source", "topic_tag", "crossref_title_similarity"]


def crossref(title):
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 3, "select": "DOI,title,container-title,issued,author,type"})
    req = urllib.request.Request("https://api.crossref.org/works?" + q, headers={"User-Agent": "presage-fraud/0.1 (mailto:research@example.org)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]["items"]


def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower())


rows, rejected = [], []
for title, tag, rel in CANDIDATES:
    try:
        items = crossref(title)
    except Exception as e:
        rejected.append((title, f"lookup error {e}")); continue
    best = max(items, key=lambda it: difflib.SequenceMatcher(None, norm(title), norm((it.get("title") or [""])[0])).ratio(), default=None)
    sim = difflib.SequenceMatcher(None, norm(title), norm((best.get("title") or [""])[0])).ratio() if best else 0
    if not best or sim < 0.88:
        rejected.append((title, f"no confident Crossref match (best sim {sim:.2f})")); continue
    au = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in best.get("author", [])[:6]) + (" et al." if len(best.get("author", [])) > 6 else "")
    rows.append({"paper": best["title"][0], "authors": au, "year": (best.get("issued", {}).get("date-parts") or [[""]])[0][0], "venue": (best.get("container-title") or [""])[0],
                 "quartile": "not_verified", "DOI": best["DOI"], "fraud_type": tag, "dataset": "to_extract", "model": "to_extract", "graph_type": "to_extract",
                 "temporal_modelling": "to_extract", "drift_handling": "to_extract", "explainability": "to_extract", "metrics": "to_extract", "reported_results": "to_extract",
                 "limitations": "to_extract", "relevance_to_PRESAGE": rel, "verification_source": "Crossref API (title similarity >= 0.88)", "topic_tag": tag, "crossref_title_similarity": round(sim, 3)})
    time.sleep(0.3)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
print(f"verified {len(rows)}; not verified {len(rejected)}")
for t, why in rejected: print("  NOT KEPT:", t[:80], "|", why)
(Path(__file__).parent / "literature_unverified.txt").write_text("\n".join(f"{t} | {w}" for t, w in rejected))
