import json, os, pathlib, pytest
from fastapi.testclient import TestClient

MODEL_DIR = pathlib.Path(__file__).resolve().parents[1] / "models" / "serving"
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "policy.json").exists(), reason="run `make serve-train` first")


def _client():
    from presage.serving.app import create_app, Service
    return TestClient(create_app(Service(str(MODEL_DIR))))


EVT = dict(transaction_id="T1", timestamp=1000.0, customer_id="C1", source_account="A1", beneficiary_account="A9", amount=250.0, currency="GBP",
           device_id="D1", ip="I1", session_id="S1", beneficiary_age_days=5.0, balance_before=900.0, session_seconds=240.0, failed_beneficiary_adds=2, channel="mobile")


def test_health_and_score():
    c = _client()
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/score/transaction", json=EVT); assert r.status_code == 200
    j = r.json()
    for k in ["transaction_id", "app_fraud_probability", "beneficiary_mule_probability", "risk_level", "recommended_action", "top_reasons"]: assert k in j
    assert 0 <= j["app_fraud_probability"] <= 1


def test_input_validation_rejects_bad_payloads():
    c = _client()
    assert c.post("/score/transaction", json=dict(EVT, amount=-5)).status_code == 422
    assert c.post("/score/transaction", json=dict(EVT, source_account="A1; DROP TABLE")).status_code == 422
    assert c.post("/score/transaction", json=dict(EVT, currency="XXX")).status_code == 422
    assert c.get("/explanation/unknown").status_code == 404


def test_api_key_enforced(monkeypatch):
    monkeypatch.setenv("PRESAGE_API_KEY", "secret-test-key")
    c = _client()
    assert c.post("/score/transaction", json=EVT).status_code == 401
    assert c.post("/score/transaction", json=EVT, headers={"X-API-Key": "secret-test-key"}).status_code == 200
