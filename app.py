import json
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="BAC Prediction API")

# =========================
# LAZY LOAD MODEL + FEATURE LIST
# =========================
_model = None
_feature_columns = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load("model.pkl")
    return _model


def get_feature_columns():
    global _feature_columns
    if _feature_columns is None:
        with open("feature_columns.json", "r", encoding="utf-8") as f:
            _feature_columns = json.load(f)
    return _feature_columns


# =========================
# REQUEST SCHEMA
# =========================
class Sample(BaseModel):
    x: float
    y: float
    z: float


class Payload(BaseModel):
    samples: List[Sample]


# =========================
# HELPER FUNCTIONS
# =========================
def _safe_float_array(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def _zero_crossings(arr):
    arr = _safe_float_array(arr)
    if arr.size < 2:
        return 0.0

    arr = arr - np.mean(arr)
    signs = np.sign(arr)
    signs[signs == 0] = 1
    return float(np.sum(signs[:-1] * signs[1:] < 0))


def _autocorr_lag1(arr):
    arr = _safe_float_array(arr)
    if arr.size < 2:
        return 0.0

    arr = arr - np.mean(arr)
    denom = np.sum(arr ** 2)
    if denom <= 1e-12:
        return 0.0

    return float(np.sum(arr[:-1] * arr[1:]) / denom)


def _safe_percentile(arr, q, default=0.0):
    arr = _safe_float_array(arr)
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features_from_samples(samples):
    xs, ys, zs = [], [], []

    for s in samples:
        try:
            if isinstance(s, dict):
                xs.append(float(s["x"]))
                ys.append(float(s["y"]))
                zs.append(float(s["z"]))
            else:
                xs.append(float(s.x))
                ys.append(float(s.y))
                zs.append(float(s.z))
        except Exception:
            continue

    x = _safe_float_array(xs)
    y = _safe_float_array(ys)
    z = _safe_float_array(zs)

    n = int(min(len(x), len(y), len(z)))
    x, y, z = x[:n], y[:n], z[:n]

    if n < 5:
        raise ValueError("Cần ít nhất 5 mẫu hợp lệ có đủ x, y, z.")

    magnitude = np.sqrt(x**2 + y**2 + z**2)
    magnitude_dyn = magnitude - np.mean(magnitude)
    jerk = np.diff(magnitude) if n > 1 else np.array([0.0])

    features = {
        "x_energy": float(np.mean(x ** 2)),
        "y_energy": float(np.mean(y ** 2)),
        "z_energy": float(np.mean(z ** 2)),
        "magnitude_energy": float(np.mean(magnitude ** 2)),
        "x_zero_crossings": _zero_crossings(x),
        "y_zero_crossings": _zero_crossings(y),
        "z_zero_crossings": _zero_crossings(z),
        "magnitude_dyn_zero_crossings": _zero_crossings(magnitude_dyn),
        "jerk_std": float(np.std(jerk)) if jerk.size > 0 else 0.0,
        "jerk_p95": _safe_percentile(np.abs(jerk), 95, default=0.0),
        "jerk_iqr": float(
            _safe_percentile(jerk, 75, default=0.0)
            - _safe_percentile(jerk, 25, default=0.0)
        ) if jerk.size > 0 else 0.0,
        "magnitude_std": float(np.std(magnitude)),
        "magnitude_dyn_autocorr1": _autocorr_lag1(magnitude_dyn),
    }

    return features, n


# =========================
# ROOT / HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "BAC Prediction API is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# =========================
# PREDICTION API
# =========================
@app.post("/predict")
def predict_from_samples(payload: Payload):
    try:
        features, n_used = extract_features_from_samples(payload.samples)
        feature_columns = get_feature_columns()
        model = get_model()

        x_input = np.array(
            [[features.get(col, 0.0) for col in feature_columns]],
            dtype=float
        )

        pred = model.predict(x_input)[0]

        prob: Optional[float] = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(x_input)[0][1])

        return {
            "drunk": bool(int(pred)),
            "prediction": int(pred),
            "label": "intoxicated" if int(pred) == 1 else "sober",
            "probability_class_1": prob,
            "n_samples": int(n_used)
        }

    except Exception as e:
        return {
            "error": str(e)
        }
