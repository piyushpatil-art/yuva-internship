"""
monitoring/simulate_traffic.py
-------------------------------
Simulates a realistic traffic pattern against the running API.
Generates 50 requests with a mix of student profiles, then
prints a live monitoring summary.

Usage:
    python monitoring/simulate_traffic.py
"""

import random
import time
import json
import requests
import numpy as np

BASE_URL = "http://localhost:8000"
random.seed(42)
np.random.seed(42)


def random_student():
    g1 = random.randint(3, 19)
    g2 = max(0, min(20, g1 + random.randint(-3, 3)))
    return {
        "G1": g1, "G2": g2,
        "failures": random.choice([0, 0, 0, 1, 2]),
        "absences": random.randint(0, 25),
        "studytime": random.randint(1, 4),
        "age": random.randint(15, 21),
        "sex": random.choice(["M", "F"]),
        "address": random.choice(["U", "R"]),
        "Medu": random.randint(0, 4),
        "Fedu": random.randint(0, 4),
        "famrel": random.randint(1, 5),
        "famsup": random.choice(["yes", "no"]),
        "school": random.choice(["GP", "MS"]),
        "schoolsup": random.choice(["yes", "no"]),
        "higher": random.choice(["yes", "yes", "yes", "no"]),
        "internet": random.choice(["yes", "yes", "no"]),
        "goout": random.randint(1, 5),
        "Dalc": random.randint(1, 5),
        "Walc": random.randint(1, 5),
        "health": random.randint(1, 5),
        "romantic": random.choice(["yes", "no"]),
        "freetime": random.randint(1, 5),
        "famsize": random.choice(["GT3", "LE3"]),
        "Pstatus": random.choice(["T", "A"]),
        "Mjob": random.choice(["teacher","health","services","at_home","other"]),
        "Fjob": random.choice(["teacher","health","services","at_home","other"]),
        "reason": random.choice(["home","reputation","course","other"]),
        "guardian": random.choice(["mother","father","other"]),
        "paid": random.choice(["yes", "no"]),
        "activities": random.choice(["yes", "no"]),
        "nursery": random.choice(["yes", "no"]),
        "traveltime": random.randint(1, 4),
    }


def simulate_traffic(n=50):
    print(f"🚀 Simulating {n} API requests...\n")
    results = []
    latencies = []
    errors = 0

    for i in range(n):
        student = random_student()
        t0 = time.time()
        try:
            r = requests.post(f"{BASE_URL}/predict", json=student, timeout=5)
            latency = (time.time() - t0) * 1000
            if r.status_code == 200:
                data = r.json()
                results.append(data)
                latencies.append(latency)
                status = "✅"
            else:
                errors += 1
                status = "❌"
        except Exception:
            errors += 1
            status = "❌"
            latency = 0

        if (i + 1) % 10 == 0:
            print(f"  {status} Completed {i+1}/{n} requests...")
        time.sleep(0.05)  # 50ms between requests

    # Summary
    print("\n" + "=" * 50)
    print("  📊 TRAFFIC SIMULATION SUMMARY")
    print("=" * 50)
    total = len(results)
    if total > 0:
        passes = sum(1 for r in results if r["prediction"] == "Pass")
        fails  = total - passes
        high_conf = sum(1 for r in results if r["confidence"] == "High")
        print(f"  Total requests   : {n}")
        print(f"  Successful       : {total}")
        print(f"  Errors           : {errors}")
        print(f"  Pass predictions : {passes} ({passes/total*100:.1f}%)")
        print(f"  Fail predictions : {fails}  ({fails/total*100:.1f}%)")
        print(f"  High confidence  : {high_conf} ({high_conf/total*100:.1f}%)")
        print(f"  Avg latency      : {np.mean(latencies):.1f}ms")
        print(f"  P95 latency      : {np.percentile(latencies, 95):.1f}ms")
        print(f"  Max latency      : {np.max(latencies):.1f}ms")

    # Fetch live metrics from API
    try:
        metrics = requests.get(f"{BASE_URL}/metrics").json()
        print(f"\n  📡 Live API Metrics:")
        print(f"    Total predictions ever : {metrics['predictions']['total']}")
        print(f"    Overall pass rate      : {metrics['predictions']['pass_rate_pct']}%")
        print(f"    Avg confidence (last 100): {metrics['model_health']['avg_confidence_last_100']}")
        print(f"    Drift alert            : {metrics['model_health']['confidence_drift_alert']}")
    except Exception:
        print("\n  (Could not fetch live metrics — is the API running?)")

    print("=" * 50)


if __name__ == "__main__":
    simulate_traffic(50)
