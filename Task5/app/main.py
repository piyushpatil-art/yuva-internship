"""
app/main.py
-----------
FastAPI application that serves the student performance prediction model.

Endpoints:
  GET  /              → API info and health summary
  GET  /health        → Detailed health check
  POST /predict       → Single student prediction
  POST /predict/batch → Batch predictions (up to 50)
  GET  /metrics       → Live usage & performance metrics
  GET  /logs          → Recent request logs (last N entries)
"""

import os
import json
import time
import uuid
import logging
import traceback
from datetime import datetime, timezone
from collections import deque
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

# ── Logging setup ────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("student_api")

# ── Load model artefacts ─────────────────────────────────────────────────────
MODEL_PATH    = os.getenv("MODEL_PATH",  "model/xgb_model.joblib")
SCALER_PATH   = os.getenv("SCALER_PATH", "model/scaler.joblib")
METADATA_PATH = os.getenv("META_PATH",   "model/metadata.json")

try:
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    FEATURE_NAMES  = metadata["feature_names"]
    CONT_COLS      = metadata["continuous_cols"]
    logger.info("✅ Model artefacts loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model artefacts: {e}")
    model = scaler = metadata = None
    FEATURE_NAMES = CONT_COLS = []

# ── In-memory monitoring store ────────────────────────────────────────────────
MAX_LOG_ENTRIES = 500
request_log: deque = deque(maxlen=MAX_LOG_ENTRIES)

# Rolling counters
stats = {
    "total_requests":      0,
    "successful_requests": 0,
    "failed_requests":     0,
    "total_predictions":   0,
    "pass_predictions":    0,
    "fail_predictions":    0,
    "total_latency_ms":    0.0,
    "start_time":          datetime.now(timezone.utc).isoformat(),
}

# Confidence drift tracking (last 100 predictions)
confidence_window: deque = deque(maxlen=100)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Student Performance Prediction API",
    description=(
        "AI-powered API that predicts whether a student will pass or fail "
        "their final assessment, and estimates their predicted grade category. "
        "Built for the Yuva Internship AI Trainee Programme — Week 5."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class StudentInput(BaseModel):
    """Input schema for a single student prediction request."""
    # Core academic features
    G1:          float = Field(...,  ge=0, le=20, description="First period grade (0–20)")
    G2:          float = Field(...,  ge=0, le=20, description="Second period grade (0–20)")
    failures:    int   = Field(0,    ge=0, le=4,  description="Number of past class failures")
    absences:    int   = Field(0,    ge=0, le=93, description="Number of school absences")
    studytime:   int   = Field(2,    ge=1, le=4,  description="Weekly study time (1=<2h, 4=>10h)")
    # Demographic
    age:         int   = Field(17,   ge=15, le=22, description="Student age")
    sex:         str   = Field("M",  description="Sex: 'M' or 'F'")
    address:     str   = Field("U",  description="Address type: 'U' (urban) or 'R' (rural)")
    # Family
    Medu:        int   = Field(2,    ge=0, le=4, description="Mother's education (0–4)")
    Fedu:        int   = Field(2,    ge=0, le=4, description="Father's education (0–4)")
    famrel:      int   = Field(4,    ge=1, le=5, description="Family relationship quality (1–5)")
    famsup:      str   = Field("yes",description="Family educational support: 'yes'/'no'")
    # School
    school:      str   = Field("GP", description="School: 'GP' or 'MS'")
    schoolsup:   str   = Field("no", description="Extra educational support: 'yes'/'no'")
    higher:      str   = Field("yes",description="Wants higher education: 'yes'/'no'")
    internet:    str   = Field("yes",description="Internet access at home: 'yes'/'no'")
    # Lifestyle
    goout:       int   = Field(2,    ge=1, le=5, description="Going out with friends (1–5)")
    Dalc:        int   = Field(1,    ge=1, le=5, description="Workday alcohol consumption (1–5)")
    Walc:        int   = Field(1,    ge=1, le=5, description="Weekend alcohol consumption (1–5)")
    health:      int   = Field(3,    ge=1, le=5, description="Current health status (1–5)")
    romantic:    str   = Field("no", description="In romantic relationship: 'yes'/'no'")
    freetime:    int   = Field(3,    ge=1, le=5, description="Free time after school (1–5)")
    # Additional (used in OHE — defaults provided)
    famsize:     str   = Field("GT3",description="Family size: 'LE3' or 'GT3'")
    Pstatus:     str   = Field("T",  description="Parents' cohabitation: 'T' or 'A'")
    Mjob:        str   = Field("other", description="Mother's job")
    Fjob:        str   = Field("other", description="Father's job")
    reason:      str   = Field("course", description="Reason for choosing school")
    guardian:    str   = Field("mother", description="Guardian: mother/father/other")
    paid:        str   = Field("no", description="Extra paid classes: 'yes'/'no'")
    activities:  str   = Field("no", description="Extracurricular activities: 'yes'/'no'")
    nursery:     str   = Field("yes",description="Attended nursery school: 'yes'/'no'")
    traveltime:  int   = Field(1,    ge=1, le=4,  description="Travel time to school (1–4)")

    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v):
        if v not in ('M', 'F'):
            raise ValueError("sex must be 'M' or 'F'")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "G1": 12, "G2": 13, "failures": 0, "absences": 4,
                "studytime": 2, "age": 17, "sex": "F", "address": "U",
                "Medu": 3, "Fedu": 2, "famrel": 4, "famsup": "yes",
                "school": "GP", "schoolsup": "no", "higher": "yes",
                "internet": "yes", "goout": 2, "Dalc": 1, "Walc": 2,
                "health": 4, "romantic": "no", "freetime": 3,
                "famsize": "GT3", "Pstatus": "T", "Mjob": "services",
                "Fjob": "other", "reason": "course", "guardian": "mother",
                "paid": "no", "activities": "yes", "nursery": "yes", "traveltime": 1
            }
        }


class PredictionResponse(BaseModel):
    request_id:        str
    prediction:        str         # "Pass" or "Fail"
    pass_probability:  float
    fail_probability:  float
    confidence:        str         # "High" / "Medium" / "Low"
    risk_level:        str         # "Low Risk" / "Medium Risk" / "High Risk"
    top_risk_factors:  List[str]
    model_version:     str
    timestamp:         str
    latency_ms:        float


class BatchRequest(BaseModel):
    students: List[StudentInput] = Field(..., max_length=50)


class BatchResponse(BaseModel):
    batch_id:     str
    count:        int
    predictions:  List[PredictionResponse]
    batch_latency_ms: float


# ── Helper: build feature vector ──────────────────────────────────────────────
def build_feature_vector(student: StudentInput) -> pd.DataFrame:
    """Convert StudentInput to the full feature DataFrame expected by the model."""
    from sklearn.preprocessing import LabelEncoder

    raw = {
        'school': student.school, 'sex': student.sex, 'age': student.age,
        'address': student.address, 'famsize': student.famsize,
        'Pstatus': student.Pstatus, 'Medu': student.Medu, 'Fedu': student.Fedu,
        'Mjob': student.Mjob, 'Fjob': student.Fjob, 'reason': student.reason,
        'guardian': student.guardian, 'traveltime': student.traveltime,
        'studytime': student.studytime, 'failures': student.failures,
        'schoolsup': student.schoolsup, 'famsup': student.famsup,
        'paid': student.paid, 'activities': student.activities,
        'nursery': student.nursery, 'higher': student.higher,
        'internet': student.internet, 'romantic': student.romantic,
        'famrel': student.famrel, 'freetime': student.freetime,
        'goout': student.goout, 'Dalc': student.Dalc, 'Walc': student.Walc,
        'health': student.health, 'absences': student.absences,
        'G1': student.G1, 'G2': student.G2,
    }
    df = pd.DataFrame([raw])

    # Binary encoding
    binary_map = {
        'school':   {'GP': 1, 'MS': 0},
        'sex':      {'M': 1, 'F': 0},
        'address':  {'U': 1, 'R': 0},
        'famsize':  {'GT3': 1, 'LE3': 0},
        'Pstatus':  {'T': 1, 'A': 0},
        'schoolsup':{'yes': 1, 'no': 0},
        'famsup':   {'yes': 1, 'no': 0},
        'paid':     {'yes': 1, 'no': 0},
        'activities':{'yes': 1, 'no': 0},
        'nursery':  {'yes': 1, 'no': 0},
        'higher':   {'yes': 1, 'no': 0},
        'internet': {'yes': 1, 'no': 0},
        'romantic': {'yes': 1, 'no': 0},
    }
    for col, mapping in binary_map.items():
        df[col] = df[col].map(mapping).fillna(0).astype(int)

    # One-hot encode (must match training columns exactly)
    onehot_cols = ['Mjob', 'Fjob', 'reason', 'guardian']
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    # Feature engineering
    df['avg_grade']        = (df['G1'] + df['G2']) / 2
    df['grade_trend']      = df['G2'] - df['G1']
    df['avg_parent_edu']   = (df['Medu'] + df['Fedu']) / 2
    df['alcohol_exposure'] = df['Dalc'] + df['Walc']
    df['support_score']    = df['schoolsup'] + df['famsup']
    df['is_at_risk']       = int((df['failures'].iloc[0] > 0) and (df['absences'].iloc[0] > 6))

    # Align to training feature set
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_NAMES]

    # Scale continuous features
    df[CONT_COLS] = scaler.transform(df[CONT_COLS])
    return df


def compute_risk_factors(student: StudentInput) -> List[str]:
    """Return human-readable list of top risk factors for the student."""
    factors = []
    avg_g = (student.G1 + student.G2) / 2
    if avg_g < 10:
        factors.append(f"Low average grade ({avg_g:.1f}/20)")
    if student.failures > 0:
        factors.append(f"{student.failures} prior failure(s)")
    if student.absences > 10:
        factors.append(f"High absences ({student.absences} days)")
    if student.studytime == 1:
        factors.append("Very low study time (<2h/week)")
    if student.Dalc + student.Walc > 5:
        factors.append("Elevated alcohol exposure score")
    if student.G2 < student.G1:
        factors.append(f"Declining grade trend (G1={student.G1}, G2={student.G2})")
    if not factors:
        factors.append("No significant risk factors detected")
    return factors[:4]


# ── Middleware: request timing & logging ──────────────────────────────────────
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start   = time.time()
    req_id  = str(uuid.uuid4())[:8]
    request.state.request_id = req_id

    stats["total_requests"] += 1
    logger.info(f"[{req_id}] {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        latency  = (time.time() - start) * 1000
        stats["total_latency_ms"] += latency
        if response.status_code < 400:
            stats["successful_requests"] += 1
        else:
            stats["failed_requests"] += 1

        log_entry = {
            "request_id":  req_id,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "method":      request.method,
            "path":        request.url.path,
            "status_code": response.status_code,
            "latency_ms":  round(latency, 2),
        }
        request_log.append(log_entry)
        logger.info(f"[{req_id}] → {response.status_code} ({latency:.1f}ms)")
        return response

    except Exception as exc:
        stats["failed_requests"] += 1
        logger.error(f"[{req_id}] Unhandled exception: {exc}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """API root — returns service info and quick health summary."""
    uptime_s = (datetime.now(timezone.utc) -
                datetime.fromisoformat(stats["start_time"])).total_seconds()
    return {
        "service":     "Student Performance Prediction API",
        "version":     "1.0.0",
        "status":      "healthy" if model else "degraded",
        "uptime_seconds": round(uptime_s, 1),
        "model_loaded":   model is not None,
        "docs":        "/docs",
        "endpoints":   ["/health", "/predict", "/predict/batch", "/metrics", "/logs"],
    }


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """
    Detailed health check endpoint.
    Returns model status, memory usage, and recent error rate.
    """
    import sys
    error_rate = (
        stats["failed_requests"] / max(stats["total_requests"], 1)
    ) * 100

    avg_latency = (
        stats["total_latency_ms"] / max(stats["successful_requests"], 1)
    )

    # Confidence drift alert
    avg_confidence = (
        np.mean(list(confidence_window)) if confidence_window else None
    )
    drift_alert = (
        avg_confidence is not None and avg_confidence < 0.60
    )

    health_status = "healthy"
    if not model:
        health_status = "unhealthy"
    elif error_rate > 10:
        health_status = "degraded"
    elif drift_alert:
        health_status = "warning"

    return {
        "status":            health_status,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "model_loaded":      model is not None,
        "model_version":     "xgb_v1.0",
        "python_version":    sys.version,
        "total_requests":    stats["total_requests"],
        "error_rate_pct":    round(error_rate, 2),
        "avg_latency_ms":    round(avg_latency, 2),
        "avg_confidence":    round(avg_confidence, 4) if avg_confidence else None,
        "confidence_drift_alert": drift_alert,
        "checks": {
            "model_file":  os.path.exists(MODEL_PATH),
            "scaler_file": os.path.exists(SCALER_PATH),
            "logs_dir":    os.path.isdir("logs"),
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(student: StudentInput, request: Request):
    """
    Predict pass/fail for a single student.

    Returns:
    - prediction: 'Pass' or 'Fail'
    - pass_probability: float [0.0–1.0]
    - confidence: High / Medium / Low
    - risk_level: Low Risk / Medium Risk / High Risk
    - top_risk_factors: list of human-readable risk explanations
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    req_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    t0 = time.time()

    try:
        features = build_feature_vector(student)
        proba    = model.predict_proba(features)[0]
        pass_prob, fail_prob = float(proba[1]), float(proba[0])
        prediction = "Pass" if pass_prob >= 0.50 else "Fail"

        # Confidence band
        max_prob = max(pass_prob, fail_prob)
        if max_prob >= 0.80:
            confidence = "High"
        elif max_prob >= 0.65:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Risk level
        if pass_prob >= 0.75:
            risk_level = "Low Risk"
        elif pass_prob >= 0.50:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        # Monitoring counters
        stats["total_predictions"] += 1
        if prediction == "Pass":
            stats["pass_predictions"] += 1
        else:
            stats["fail_predictions"] += 1
        confidence_window.append(max_prob)

        latency = (time.time() - t0) * 1000
        logger.info(
            f"[{req_id}] PREDICT → {prediction} "
            f"(p_pass={pass_prob:.3f}, conf={confidence}, latency={latency:.1f}ms)"
        )

        return PredictionResponse(
            request_id       = req_id,
            prediction       = prediction,
            pass_probability = round(pass_prob, 4),
            fail_probability = round(fail_prob, 4),
            confidence       = confidence,
            risk_level       = risk_level,
            top_risk_factors = compute_risk_factors(student),
            model_version    = "xgb_v1.0",
            timestamp        = datetime.now(timezone.utc).isoformat(),
            latency_ms       = round(latency, 2),
        )

    except Exception as e:
        logger.error(f"[{req_id}] Prediction error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(batch: BatchRequest, request: Request):
    """Batch predict for up to 50 students in one request."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    t0       = time.time()
    batch_id = str(uuid.uuid4())[:8]
    results  = []

    for student in batch.students:
        try:
            features  = build_feature_vector(student)
            proba     = model.predict_proba(features)[0]
            pass_prob = float(proba[1])
            prediction = "Pass" if pass_prob >= 0.50 else "Fail"
            max_prob   = max(pass_prob, 1 - pass_prob)
            confidence = "High" if max_prob >= 0.80 else "Medium" if max_prob >= 0.65 else "Low"
            risk_level = "Low Risk" if pass_prob >= 0.75 else "Medium Risk" if pass_prob >= 0.50 else "High Risk"

            stats["total_predictions"] += 1
            confidence_window.append(max_prob)
            if prediction == "Pass": stats["pass_predictions"] += 1
            else: stats["fail_predictions"] += 1

            results.append(PredictionResponse(
                request_id       = f"{batch_id}-{len(results)}",
                prediction       = prediction,
                pass_probability = round(pass_prob, 4),
                fail_probability = round(1 - pass_prob, 4),
                confidence       = confidence,
                risk_level       = risk_level,
                top_risk_factors = compute_risk_factors(student),
                model_version    = "xgb_v1.0",
                timestamp        = datetime.now(timezone.utc).isoformat(),
                latency_ms       = 0.0,
            ))
        except Exception as e:
            logger.warning(f"[{batch_id}] Skipped student due to error: {e}")

    batch_latency = (time.time() - t0) * 1000
    logger.info(f"[{batch_id}] BATCH → {len(results)}/{len(batch.students)} predictions ({batch_latency:.1f}ms)")

    return BatchResponse(
        batch_id=batch_id, count=len(results),
        predictions=results, batch_latency_ms=round(batch_latency, 2)
    )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Live API usage and model performance statistics."""
    total    = max(stats["total_predictions"], 1)
    req_total= max(stats["total_requests"], 1)
    avg_conf = float(np.mean(list(confidence_window))) if confidence_window else None

    return {
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "uptime_since":         stats["start_time"],
        "requests": {
            "total":            stats["total_requests"],
            "successful":       stats["successful_requests"],
            "failed":           stats["failed_requests"],
            "error_rate_pct":   round(stats["failed_requests"] / req_total * 100, 2),
            "avg_latency_ms":   round(stats["total_latency_ms"] / req_total, 2),
        },
        "predictions": {
            "total":            stats["total_predictions"],
            "pass_count":       stats["pass_predictions"],
            "fail_count":       stats["fail_predictions"],
            "pass_rate_pct":    round(stats["pass_predictions"] / total * 100, 2),
            "fail_rate_pct":    round(stats["fail_predictions"] / total * 100, 2),
        },
        "model_health": {
            "avg_confidence_last_100": round(avg_conf, 4) if avg_conf else None,
            "confidence_drift_alert":  avg_conf is not None and avg_conf < 0.60,
            "model_version":           "xgb_v1.0",
        }
    }


@app.get("/logs", tags=["Monitoring"])
async def get_logs(n: int = Query(default=20, ge=1, le=500)):
    """Return the last N request log entries."""
    entries = list(request_log)[-n:]
    return {
        "count":   len(entries),
        "entries": entries
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
