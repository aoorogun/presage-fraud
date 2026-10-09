"""FastAPI scoring service.

Endpoints: POST /score/transaction | POST /score/beneficiary | GET /risk/account/{id} | GET /explanation/{transaction_id} | GET /health | GET /metrics | GET /metrics/prometheus
Security: configuration from environment variables only (PRESAGE_MODEL_DIR, PRESAGE_API_KEY, PRESAGE_CASE_DB); inputs validated with pydantic;
identifiers are hashed before logging; no credentials or personal data are stored or logged.
"""
from __future__ import annotations
import hashlib, json, os, sqlite3, time, threading
from pathlib import Path
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator
from presage.serving.online_features import OnlineFeatureStore, SERVING_FEATURES
from presage.evaluation.policy import Policy, ACTIONS
from presage.utils.common import get_logger

log = get_logger("api")
ID = r"^[A-Za-z0-9_\-:.]{1,64}$"
CURRENCIES = {"GBP", "EUR", "USD", "NGN"}


def _h(x: str) -> str: return hashlib.sha256(x.encode()).hexdigest()[:10]


class TxnEvent(BaseModel):
    transaction_id: str = Field(pattern=ID)
    timestamp: float = Field(ge=0, description="seconds since epoch of the simulation / event clock")
    customer_id: str = Field(pattern=ID)
    source_account: str = Field(pattern=ID)
    beneficiary_account: str = Field(pattern=ID)
    amount: float = Field(gt=0, lt=1e9)
    currency: str = "GBP"
    device_id: str = Field(pattern=ID)
    ip: str = Field(pattern=ID)
    session_id: str = Field(pattern=ID)
    beneficiary_age_days: float = Field(ge=-30, lt=36500)
    balance_before: float | None = Field(default=None, ge=-1e9, lt=1e12)
    session_seconds: float = Field(default=60.0, ge=0, lt=86400)
    failed_beneficiary_adds: int = Field(default=0, ge=0, lt=1000)
    channel: str = "mobile"

    @field_validator("currency")
    @classmethod
    def _cur(cls, v):
        if v not in CURRENCIES: raise ValueError("unsupported currency")
        return v

    @field_validator("channel")
    @classmethod
    def _ch(cls, v):
        if v not in {"mobile", "desktop"}: raise ValueError("channel must be mobile|desktop")
        return v


class BenRequest(BaseModel):
    beneficiary_account: str = Field(pattern=ID)
    timestamp: float = Field(ge=0)
    beneficiary_age_days: float = Field(ge=-30, lt=36500)


class Service:
    def __init__(self, model_dir: str | None = None, case_db: str | None = None):
        import xgboost as xgb
        d = Path(model_dir or os.environ.get("PRESAGE_MODEL_DIR", "models/serving"))
        self.meta = json.load(open(d / "meta.json"))
        self.models = [self._load(xgb, p) for p in sorted(d.glob("app_s*.json"))]
        self.mule = self._load(xgb, d / "mule.json")
        pol = json.load(open(d / "policy.json")); self.policy = Policy(pol["t1"], pol["t2"], pol["t3"])
        self.store = OnlineFeatureStore(); self.lock = threading.Lock(); self.cols = SERVING_FEATURES
        self.mule_idx = [self.cols.index(c) for c in self.meta["mule_features"]]
        self.latencies: list[float] = []; self.n_scored = 0; self.score_hist: list[float] = []
        self.db = sqlite3.connect(case_db or os.environ.get("PRESAGE_CASE_DB", ":memory:"), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cases(txn_id TEXT PRIMARY KEY, ts REAL, amount REAL, p_app REAL, p_mule REAL, action TEXT, explanation TEXT)")
        self._expl = None
        ref = d / "reference_features.npy"
        self.reference = np.load(ref) if ref.exists() else None
        self.acct_state: dict[str, dict] = {}
        self._id_map: dict[str, int] = {}

    @staticmethod
    def _load(xgb, p):
        m = xgb.XGBClassifier(); m.load_model(str(p)); return m

    def _num(self, s: str) -> int:
        if s not in self._id_map: self._id_map[s] = len(self._id_map)
        return self._id_map[s]

    def _event(self, t: TxnEvent) -> dict:
        return dict(ts=t.timestamp, src=self._num(t.source_account), dst=self._num(t.beneficiary_account), amount=t.amount, device_id=self._num("dev:" + t.device_id),
                    session_s=t.session_seconds, failed_ben_adds=t.failed_beneficiary_adds, channel=t.channel, dst_age_days=t.beneficiary_age_days, balance_before=t.balance_before)

    def explainer(self):
        if self._expl is None and self.reference is not None:
            import pandas as pd
            from presage.explainability.explain import Explainer
            bg = pd.DataFrame(self.reference, columns=self.cols)
            groups = dict(txn=[c for c in self.cols if not c.startswith(("snd_", "dst_", "hist", "amt_", "secs", "new_", "prior", "device"))], behaviour=[c for c in self.cols if c.startswith(("snd_", "hist", "amt_", "secs"))],
                          beneficiary=[c for c in self.cols if c.startswith(("dst_", "new_beneficiary", "prior"))], graph=[])
            self._expl = Explainer(self.models[0], list(self.cols), groups, bg, self.policy.t1)
        return self._expl

    def score(self, t: TxnEvent) -> dict:
        t0 = time.perf_counter(); e = self._event(t)
        with self.lock:
            f = self.store.features(e); x = np.array([[f[c] for c in self.cols]], dtype=np.float32)
            self.store.update(e)
        ps = np.array([m.predict_proba(x)[0, 1] for m in self.models]); p = float(ps.mean()); unc = float(ps.std())
        pm = float(self.mule.predict_proba(x[:, self.mule_idx])[0, 1])
        action = ACTIONS[int(self.policy.decide(np.array([p]))[0])]
        level = {"approve": "low", "soft_warning": "medium", "strong_warning_confirm": "high", "hold_review": "critical"}[action]
        reasons = []
        ex = self.explainer()
        if ex is not None and action != "approve":
            import pandas as pd
            row = pd.Series(x[0], index=self.cols); out = ex.explain(row, 4); reasons = [r["text"] for r in out["reasons"]]
            expl = out
        else:
            expl = dict(reasons=[])
        res = dict(transaction_id=t.transaction_id, app_fraud_probability=round(p, 4), beneficiary_mule_probability=round(pm, 4), confidence=round(float(max(0.0, 1 - 4 * unc)), 3),
                   risk_level=level, recommended_action=action, top_reasons=reasons)
        self.db.execute("INSERT OR REPLACE INTO cases VALUES(?,?,?,?,?,?,?)", (t.transaction_id, t.timestamp, t.amount, p, pm, action, json.dumps(expl, default=float)))
        self.latencies.append(time.perf_counter() - t0); self.n_scored += 1; self.score_hist.append(p)
        self.score_hist = self.score_hist[-5000:]; self.latencies = self.latencies[-5000:]
        return res

    def score_beneficiary(self, r: BenRequest) -> dict:
        i = self._num(r.beneficiary_account)
        with self.lock:
            e = dict(ts=r.timestamp, src=-1, dst=i, amount=1.0, device_id=-1, session_s=0, failed_ben_adds=0, channel="mobile", dst_age_days=r.beneficiary_age_days, balance_before=None)
            f = self.store.features(e)
        x = np.array([[f[c] for c in self.meta["mule_features"]]], dtype=np.float32)
        return dict(beneficiary_account=r.beneficiary_account, beneficiary_mule_probability=round(float(self.mule.predict_proba(x)[0, 1]), 4),
                    evidence=dict(unique_senders=f["dst_fanin_total"], new_senders_24h=f["dst_fanin_new_24h"], pass_through_24h=round(f["dst_pass_through_24h"], 3), account_age_days=r.beneficiary_age_days))

    def metrics(self) -> dict:
        l = np.array(self.latencies) * 1000 if self.latencies else np.array([0.0]); h = np.array(self.score_hist) if self.score_hist else np.array([0.0])
        out = dict(n_scored=self.n_scored, latency_ms_p50=float(np.percentile(l, 50)), latency_ms_p99=float(np.percentile(l, 99)), mean_score=float(h.mean()),
                   alert_rate=float((h >= self.policy.t1).mean()))
        if self.reference is not None and len(h) > 200:
            from presage.drift.detectors import psi
            out["score_psi_vs_reference"] = None   # filled by monitoring job (needs reference scores)
        return out


def create_app(service: Service | None = None) -> FastAPI:
    app = FastAPI(title="PRESAGE-Fraud scoring API", version="0.1.0")
    svc = service or Service()
    key = os.environ.get("PRESAGE_API_KEY")

    def auth(x_api_key: str | None = Header(default=None)):
        if key and x_api_key != key: raise HTTPException(401, "invalid API key")

    @app.get("/health")
    def health(): return dict(status="ok", models=len(svc.models), features=len(svc.cols), simulated_data=True)

    @app.post("/score/transaction", dependencies=[Depends(auth)])
    def score_txn(t: TxnEvent):
        log.info("score txn=%s src=%s dst=%s", _h(t.transaction_id), _h(t.source_account), _h(t.beneficiary_account))
        return svc.score(t)

    @app.post("/score/beneficiary", dependencies=[Depends(auth)])
    def score_ben(r: BenRequest): return svc.score_beneficiary(r)

    @app.get("/risk/account/{account_id}", dependencies=[Depends(auth)])
    def risk(account_id: str, timestamp: float = 0.0, age_days: float = 365.0):
        import re
        if not re.match(ID, account_id): raise HTTPException(422, "bad account id")
        return svc.score_beneficiary(BenRequest(beneficiary_account=account_id, timestamp=timestamp, beneficiary_age_days=age_days))

    @app.get("/explanation/{transaction_id}", dependencies=[Depends(auth)])
    def explanation(transaction_id: str):
        import re
        if not re.match(ID, transaction_id): raise HTTPException(422, "bad id")
        r = svc.db.execute("SELECT p_app,p_mule,action,explanation FROM cases WHERE txn_id=?", (transaction_id,)).fetchone()
        if not r: raise HTTPException(404, "unknown transaction")
        return dict(transaction_id=transaction_id, app_fraud_probability=r[0], beneficiary_mule_probability=r[1], recommended_action=r[2], explanation=json.loads(r[3]),
                    caveat="SHAP attributions describe the model, not causal truth; counterfactuals are model-based what-ifs.")

    @app.get("/metrics", dependencies=[Depends(auth)])
    def metrics(): return svc.metrics()

    @app.get("/metrics/prometheus")
    def prom():
        from fastapi.responses import PlainTextResponse
        m = svc.metrics()
        txt = "".join(f"presage_{k} {v}\n" for k, v in m.items() if isinstance(v, (int, float)))
        return PlainTextResponse(txt)

    return app
