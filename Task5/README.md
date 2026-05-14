# 🎓 Student Performance Prediction API
### Yuva Internship — Week 5: Deployment and Monitoring

---

## Overview
A production-ready FastAPI service that predicts student pass/fail outcomes
using a trained XGBoost model, complete with Docker containerisation, logging,
and live monitoring endpoints.

---

## Project Structure
```
student_performance_api/
├── app/
│   └── main.py               # FastAPI application (all endpoints)
├── model/
│   ├── train_model.py        # Training script (run once)
│   ├── xgb_model.joblib      # Trained model (generated)
│   ├── scaler.joblib         # Feature scaler (generated)
│   └── metadata.json         # Model metrics & feature list (generated)
├── tests/
│   └── test_api.py           # Full test suite (10 tests)
├── monitoring/
│   └── simulate_traffic.py   # Traffic simulation script
├── logs/
│   └── api.log               # Request logs (auto-generated)
├── Dockerfile                # Container definition
├── docker-compose.yml        # Multi-service orchestration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Quick Start

### Option A: Run with Docker (Recommended)
```bash
# 1. Build and start
docker compose up --build

# 2. Test the API
curl http://localhost:8000/health

# 3. Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"G1":14,"G2":15,"failures":0,"absences":3,"studytime":3,
       "age":17,"sex":"F","address":"U","Medu":3,"Fedu":2,
       "famrel":4,"famsup":"yes","school":"GP","schoolsup":"no",
       "higher":"yes","internet":"yes","goout":2,"Dalc":1,"Walc":2,
       "health":4,"romantic":"no","freetime":3,"famsize":"GT3",
       "Pstatus":"T","Mjob":"teacher","Fjob":"other","reason":"course",
       "guardian":"mother","paid":"no","activities":"yes",
       "nursery":"yes","traveltime":1}'
```

### Option B: Run locally (without Docker)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python model/train_model.py

# 3. Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Run tests
python tests/test_api.py

# 5. Simulate traffic
python monitoring/simulate_traffic.py
```

---

## API Endpoints

| Method | Endpoint         | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | `/`             | Service info and health summary      |
| GET    | `/health`       | Detailed health check with alerts    |
| POST   | `/predict`      | Single student prediction            |
| POST   | `/predict/batch`| Batch predictions (up to 50)         |
| GET    | `/metrics`      | Live usage & performance statistics  |
| GET    | `/logs`         | Recent request logs                  |
| GET    | `/docs`         | Interactive Swagger UI               |

---

## Sample Response
```json
{
  "request_id": "a3f9b1c2",
  "prediction": "Pass",
  "pass_probability": 0.8734,
  "fail_probability": 0.1266,
  "confidence": "High",
  "risk_level": "Low Risk",
  "top_risk_factors": ["No significant risk factors detected"],
  "model_version": "xgb_v1.0",
  "timestamp": "2025-01-15T10:32:11.423Z",
  "latency_ms": 8.4
}
```

---

## Monitoring
- **`/health`** — real-time health with drift alert
- **`/metrics`** — live pass/fail rates, latency stats
- **`/logs`** — queryable request log (last N entries)
- **`logs/api.log`** — persistent file-based log

---

*Yuva Internship — AI Trainee Programme, Week 5*
