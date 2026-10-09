"""Streaming simulation:  producer -> transport -> feature service + PRESAGE-serving model -> decision engine -> alert topic + case store.

Replays the test period of the simulated world in event-time order (customer payments are scored; all events update state).
Reports throughput, end-to-end latency percentiles, alert volume and pre-payment detection of scams (financial figures are SIMULATED).
Usage: python streaming/run_demo.py [--limit 20000] [--kafka localhost:9092]
"""
import argparse, json, sys, time, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(pathlib.Path(__file__).parent))
from transport import InMemoryTransport, KafkaTransport
from presage.data.loaders import load_sim
from presage.serving.app import Service, TxnEvent
from presage.utils.common import ROOT

ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=20000); ap.add_argument("--kafka", default=None); ap.add_argument("--warm-days", type=int, default=90)
a = ap.parse_args()
df, acc = load_sim(); svc = Service(str(ROOT / "models/serving"))
tp = KafkaTransport(a.kafka) if a.kafka else InMemoryTransport()
cols = ["txn_id", "ts", "customer_id", "src", "dst", "amount", "device_id", "ip_id", "session_s", "failed_ben_adds", "channel", "dst_age_days", "balance_before", "is_app_fraud", "src_is_customer"]
warm = df[df.day < a.warm_days]; live = df[df.day >= a.warm_days].head(a.limit)
# warm the stateful feature store with history (events only, no scoring)
for e in warm[["ts", "src", "dst", "amount", "device_id", "session_s", "failed_ben_adds", "channel", "dst_age_days", "balance_before"]].to_dict("records"):
    e2 = dict(e, src=svc._num(f"A{int(e['src'])}"), dst=svc._num(f"A{int(e['dst'])}"), device_id=svc._num(f"dev:D{int(e['device_id'])}")); svc.store.update(e2)
labels = {}
t_start = time.perf_counter()
for r in live[cols].itertuples(index=False):                                    # ---- producer
    msg = dict(transaction_id=f"T{r.txn_id}", timestamp=float(r.ts), customer_id=f"C{r.customer_id}", source_account=f"A{r.src}", beneficiary_account=f"A{r.dst}", amount=float(r.amount), currency="GBP",
               device_id=f"D{r.device_id}", ip=f"I{r.ip_id}", session_id=f"S{r.txn_id}", beneficiary_age_days=float(r.dst_age_days), balance_before=None if np.isnan(r.balance_before) else float(r.balance_before),
               session_seconds=float(r.session_s), failed_beneficiary_adds=int(r.failed_ben_adds), channel=r.channel, _scored=bool(r.src_is_customer))
    labels[msg["transaction_id"]] = int(r.is_app_fraud); tp.send("payments", msg)
lat, alerts, n = [], [], 0
while True:                                                                      # ---- consumer / decision engine
    m = tp.poll("payments")
    if m is None: break
    scored = m.pop("_scored"); ev = TxnEvent(**m)
    s = time.perf_counter()
    if scored:
        out = svc.score(ev)
        lat.append(time.perf_counter() - s)
        if out["recommended_action"] != "approve": tp.send("alerts", dict(out, label=labels[m["transaction_id"]]))
    else:
        svc.store.update(svc._event(ev))
    n += 1
wall = time.perf_counter() - t_start
al = []
while (x := tp.poll("alerts")) is not None: al.append(x)
y = np.array([labels[k] for k in labels]); frauds = sum(1 for v in labels.values() if v)
caught = [x for x in al if x["label"] == 1]; holds = [x for x in al if x["recommended_action"] == "hold_review"]
rep = dict(events=n, scored=len(lat), wall_s=wall, throughput_eps=n / wall, latency_ms_p50=1000 * np.percentile(lat, 50), latency_ms_p95=1000 * np.percentile(lat, 95), latency_ms_p99=1000 * np.percentile(lat, 99),
           alerts=len(al), alert_rate=len(al) / max(len(lat), 1), frauds_in_stream=int(frauds), frauds_intervened_pre_payment=len(caught), recall_any_intervention=len(caught) / max(frauds, 1),
           holds=len(holds), precision_of_hold=(sum(x["label"] for x in holds) / max(len(holds), 1)), transport="kafka" if a.kafka else "in-memory", note="payments are scored BEFORE settlement in event order; values simulated")
print(json.dumps(rep, indent=1)); json.dump(rep, open(ROOT / "reports/results/streaming_demo.json", "w"), indent=1)
