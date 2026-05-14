"""
tests/test_api.py
-----------------
Comprehensive test suite for the Student Performance Prediction API.
Can be run directly (against a running API) or via pytest.

Usage:
    # Against local running API:
    python tests/test_api.py

    # Via pytest:
    pytest tests/test_api.py -v
"""

import sys
import json
import time
import requests

BASE_URL = "http://localhost:8000"

# ── Sample students ────────────────────────────────────────────────────────────

HIGH_ACHIEVER = {
    "G1": 17, "G2": 18, "failures": 0, "absences": 1,
    "studytime": 4, "age": 17, "sex": "F", "address": "U",
    "Medu": 4, "Fedu": 4, "famrel": 5, "famsup": "yes",
    "school": "GP", "schoolsup": "no", "higher": "yes",
    "internet": "yes", "goout": 1, "Dalc": 1, "Walc": 1,
    "health": 5, "romantic": "no", "freetime": 2,
    "famsize": "GT3", "Pstatus": "T", "Mjob": "teacher",
    "Fjob": "health", "reason": "reputation", "guardian": "mother",
    "paid": "yes", "activities": "yes", "nursery": "yes", "traveltime": 1
}

AT_RISK_STUDENT = {
    "G1": 4, "G2": 5, "failures": 2, "absences": 22,
    "studytime": 1, "age": 19, "sex": "M", "address": "R",
    "Medu": 0, "Fedu": 0, "famrel": 2, "famsup": "no",
    "school": "MS", "schoolsup": "no", "higher": "no",
    "internet": "no", "goout": 5, "Dalc": 4, "Walc": 5,
    "health": 2, "romantic": "yes", "freetime": 5,
    "famsize": "LE3", "Pstatus": "A", "Mjob": "other",
    "Fjob": "other", "reason": "home", "guardian": "other",
    "paid": "no", "activities": "no", "nursery": "no", "traveltime": 4
}

AVERAGE_STUDENT = {
    "G1": 11, "G2": 11, "failures": 0, "absences": 6,
    "studytime": 2, "age": 17, "sex": "M", "address": "U",
    "Medu": 2, "Fedu": 2, "famrel": 3, "famsup": "yes",
    "school": "GP", "schoolsup": "no", "higher": "yes",
    "internet": "yes", "goout": 3, "Dalc": 1, "Walc": 2,
    "health": 3, "romantic": "no", "freetime": 3,
    "famsize": "GT3", "Pstatus": "T", "Mjob": "services",
    "Fjob": "other", "reason": "course", "guardian": "mother",
    "paid": "no", "activities": "no", "nursery": "yes", "traveltime": 2
}

# ── Test functions ─────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"

def test_root():
    print("\n[TEST] GET / — API root endpoint")
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "service" in data
    assert "status" in data
    print(f"  {PASS}  status={data['status']}")


def test_health():
    print("\n[TEST] GET /health — Health check")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["model_loaded"] == True
    assert "error_rate_pct" in data
    print(f"  {PASS}  health={data['status']}, model_loaded={data['model_loaded']}")


def test_predict_high_achiever():
    print("\n[TEST] POST /predict — High achiever (expect Pass)")
    r = requests.post(f"{BASE_URL}/predict", json=HIGH_ACHIEVER)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] == "Pass", f"Expected Pass, got {data['prediction']}"
    assert data["pass_probability"] >= 0.70
    assert data["confidence"] in ("High", "Medium")
    print(f"  {PASS}  prediction={data['prediction']}, "
          f"p_pass={data['pass_probability']:.3f}, conf={data['confidence']}")


def test_predict_at_risk():
    print("\n[TEST] POST /predict — At-risk student (expect Fail)")
    r = requests.post(f"{BASE_URL}/predict", json=AT_RISK_STUDENT)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] == "Fail", f"Expected Fail, got {data['prediction']}"
    assert data["fail_probability"] >= 0.60
    print(f"  {PASS}  prediction={data['prediction']}, "
          f"p_fail={data['fail_probability']:.3f}, risk={data['risk_level']}")


def test_predict_average():
    print("\n[TEST] POST /predict — Average student")
    r = requests.post(f"{BASE_URL}/predict", json=AVERAGE_STUDENT)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in ("Pass", "Fail")
    assert 0.0 <= data["pass_probability"] <= 1.0
    assert len(data["top_risk_factors"]) > 0
    print(f"  {PASS}  prediction={data['prediction']}, p_pass={data['pass_probability']:.3f}")


def test_predict_validation_error():
    print("\n[TEST] POST /predict — Invalid input (G1=25, should fail validation)")
    invalid = {**AVERAGE_STUDENT, "G1": 25}  # Grade out of range [0, 20]
    r = requests.post(f"{BASE_URL}/predict", json=invalid)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print(f"  {PASS}  Correctly rejected invalid input (HTTP 422)")


def test_batch_predict():
    print("\n[TEST] POST /predict/batch — Batch of 3 students")
    payload = {"students": [HIGH_ACHIEVER, AT_RISK_STUDENT, AVERAGE_STUDENT]}
    r = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    print(f"  {PASS}  batch_id={data['batch_id']}, "
          f"count={data['count']}, latency={data['batch_latency_ms']:.1f}ms")
    for i, pred in enumerate(data["predictions"]):
        print(f"         Student {i+1}: {pred['prediction']} (p={pred['pass_probability']:.3f})")


def test_metrics():
    print("\n[TEST] GET /metrics — Usage metrics")
    r = requests.get(f"{BASE_URL}/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "requests" in data
    assert "predictions" in data
    assert data["predictions"]["total"] > 0
    print(f"  {PASS}  total_predictions={data['predictions']['total']}, "
          f"pass_rate={data['predictions']['pass_rate_pct']}%")


def test_logs():
    print("\n[TEST] GET /logs — Request logs")
    r = requests.get(f"{BASE_URL}/logs?n=10")
    assert r.status_code == 200
    data = r.json()
    assert "entries" in data
    assert isinstance(data["entries"], list)
    print(f"  {PASS}  Retrieved {data['count']} log entries")


def test_latency():
    print("\n[TEST] Latency — Single prediction under 500ms")
    t0 = time.time()
    r  = requests.post(f"{BASE_URL}/predict", json=AVERAGE_STUDENT)
    latency = (time.time() - t0) * 1000
    assert r.status_code == 200
    assert latency < 500, f"Latency too high: {latency:.1f}ms"
    print(f"  {PASS}  End-to-end latency: {latency:.1f}ms")


# ── Runner ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_root, test_health,
        test_predict_high_achiever, test_predict_at_risk, test_predict_average,
        test_predict_validation_error, test_batch_predict,
        test_metrics, test_logs, test_latency
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {FAIL}  {e}")
            failed += 1

    print("\n" + "=" * 45)
    print(f"  Tests: {passed + failed}  |  Passed: {passed}  |  Failed: {failed}")
    print("=" * 45)
    sys.exit(0 if failed == 0 else 1)
