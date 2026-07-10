import joblib
import pandas as pd
from config import MODEL_DIR

def load_model_artifacts():
    calibrated_model = joblib.load(f'{MODEL_DIR}/xgb_model.pkl')
    xgb_model_raw = joblib.load(f'{MODEL_DIR}/xgb_model_raw.pkl')
    scaler = joblib.load(f'{MODEL_DIR}/scaler.pkl')
    imputer = joblib.load(f'{MODEL_DIR}/imputer.pkl')
    feature_names = joblib.load(f'{MODEL_DIR}/feature_names.pkl')
    return calibrated_model, xgb_model_raw, scaler, imputer, feature_names

def predict_case(form_data: dict, calibrated_model, scaler, imputer, feature_names):
    row = pd.DataFrame([form_data])[feature_names]
    row_imputed = imputer.transform(row)
    row_scaled = scaler.transform(row_imputed)
    proba = calibrated_model.predict_proba(row_scaled)[0, 1]
    predicted_label = int(proba >= 0.5)
    confidence = max(proba, 1 - proba)
    return {
        'proba': float(proba),
        'predicted_label': predicted_label,
        'confidence': float(confidence),
        'row_scaled': row_scaled
    }