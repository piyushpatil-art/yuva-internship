"""
train_model.py
--------------
Trains the XGBoost student performance model and saves it along with
the scaler and feature metadata. Run once before starting the API.

Usage:
    python model/train_model.py
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# ── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)

# ── Generate representative synthetic dataset ────────────────────────────────
def generate_dataset(n=395):
    """Generates a representative student dataset (fallback if UCI URL unavailable)."""
    np.random.seed(42)
    df = pd.DataFrame({
        'school':    np.random.choice(['GP', 'MS'], n),
        'sex':       np.random.choice(['M', 'F'], n),
        'age':       np.random.randint(15, 22, n),
        'address':   np.random.choice(['U', 'R'], n),
        'famsize':   np.random.choice(['LE3', 'GT3'], n),
        'Pstatus':   np.random.choice(['T', 'A'], n),
        'Medu':      np.random.randint(0, 5, n),
        'Fedu':      np.random.randint(0, 5, n),
        'Mjob':      np.random.choice(['teacher','health','services','at_home','other'], n),
        'Fjob':      np.random.choice(['teacher','health','services','at_home','other'], n),
        'reason':    np.random.choice(['home','reputation','course','other'], n),
        'guardian':  np.random.choice(['mother','father','other'], n),
        'traveltime':np.random.randint(1, 5, n),
        'studytime': np.random.randint(1, 4, n),
        'failures':  np.random.choice([0,1,2,3], n, p=[0.67,0.17,0.1,0.06]),
        'schoolsup': np.random.choice(['yes','no'], n),
        'famsup':    np.random.choice(['yes','no'], n),
        'paid':      np.random.choice(['yes','no'], n),
        'activities':np.random.choice(['yes','no'], n),
        'nursery':   np.random.choice(['yes','no'], n),
        'higher':    np.random.choice(['yes','no'], n, p=[0.82, 0.18]),
        'internet':  np.random.choice(['yes','no'], n, p=[0.66, 0.34]),
        'romantic':  np.random.choice(['yes','no'], n),
        'famrel':    np.random.randint(1, 6, n),
        'freetime':  np.random.randint(1, 6, n),
        'goout':     np.random.randint(1, 6, n),
        'Dalc':      np.random.randint(1, 6, n),
        'Walc':      np.random.randint(1, 6, n),
        'health':    np.random.randint(1, 6, n),
        'absences':  np.random.randint(0, 40, n),
        'G1':        np.random.randint(3, 19, n),
        'G2':        np.random.randint(3, 19, n),
        'G3':        np.random.randint(0, 20, n),
    })
    return df


def preprocess(df):
    """Apply full preprocessing pipeline."""
    binary_cols = ['school','sex','address','famsize','Pstatus','schoolsup','famsup',
                   'paid','activities','nursery','higher','internet','romantic']
    onehot_cols = ['Mjob','Fjob','reason','guardian']

    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    # Feature engineering
    df['avg_grade']        = (df['G1'] + df['G2']) / 2
    df['grade_trend']      = df['G2'] - df['G1']
    df['avg_parent_edu']   = (df['Medu'] + df['Fedu']) / 2
    df['alcohol_exposure'] = df['Dalc'] + df['Walc']
    df['support_score']    = df['schoolsup'] + df['famsup']
    df['is_at_risk']       = ((df['failures'] > 0) &
                               (df['absences'] > df['absences'].median())).astype(int)
    df['pass_fail']        = (df['G3'] >= 10).astype(int)
    return df


if __name__ == "__main__":
    print("=" * 55)
    print("  Student Performance Model — Training Script")
    print("=" * 55)

    # Load or generate data
    try:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/dsrscientist/dataset1/master/student-mat.csv",
            sep=';'
        )
        print("✅ Dataset loaded from URL.")
    except Exception:
        print("⚠️  URL unavailable. Using synthetic dataset.")
        df = generate_dataset()

    df = preprocess(df)

    TARGET = 'pass_fail'
    DROP   = ['G3', 'pass_fail']
    X = df.drop(columns=DROP)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    CONT_COLS = ['age','absences','avg_grade','grade_trend','avg_parent_edu','alcohol_exposure']
    scaler = RobustScaler()
    X_train[CONT_COLS] = scaler.fit_transform(X_train[CONT_COLS])
    X_test[CONT_COLS]  = scaler.transform(X_test[CONT_COLS])

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "n_features": int(X_train.shape[1]),
        "feature_names": list(X_train.columns),
        "continuous_cols": CONT_COLS,
    }

    print(f"\n📊 Test Accuracy : {metrics['accuracy']}")
    print(f"   Test F1-Score : {metrics['f1_score']}")
    print(f"   Test ROC-AUC  : {metrics['roc_auc']}")
    print(f"   Features      : {metrics['n_features']}")

    # Save artefacts
    os.makedirs("model", exist_ok=True)
    joblib.dump(model,  "model/xgb_model.joblib")
    joblib.dump(scaler, "model/scaler.joblib")
    with open("model/metadata.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Artefacts saved:")
    print("   model/xgb_model.joblib")
    print("   model/scaler.joblib")
    print("   model/metadata.json")
