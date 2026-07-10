from flask import Flask, render_template, request, send_file
from modules.model_utils import load_model_artifacts, predict_case
from modules.shap_utils import get_shap_explanation
from modules.failure_audit import (
    load_failure_taxonomy, check_calibration_zone,
    check_flagged_age_band, find_similar_failures
)
from modules.groq_client import (
    generate_prediction_explanation, generate_failure_risk_warning, generate_risk_narrative
)
from config import HIGH_CONFIDENCE_THRESHOLD
import pandas as pd

app = Flask(__name__)

# Load once at startup — Flask application context keeps these in memory across requests
calibrated_model, xgb_model_raw, scaler, imputer, feature_names = load_model_artifacts()
failure_taxonomy = load_failure_taxonomy()

@app.route('/')
def index():
    stats = {
        'total_cases': 30000,
        'high_conf_error_rate': '3.10%',
        'failure_patterns': failure_taxonomy['failure_category'].nunique(),
        'business_impact': '₹41.2L (48.5% of total error cost)'
    }
    return render_template('index.html', stats=stats)

@app.route('/audit')
def audit():
    return render_template('audit.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = {f: float(request.form.get(f, 0)) for f in feature_names}
    result = predict_case(form_data, calibrated_model, scaler, imputer, feature_names)

    shap_pairs = get_shap_explanation(xgb_model_raw, result['row_scaled'], feature_names)
    explanation = generate_prediction_explanation(result['predicted_label'], result['confidence'], shap_pairs)

    calibration_warning = check_calibration_zone(result['confidence'])
    age_flag = check_flagged_age_band(form_data.get('age', 0))

    failure_warning = None
    if result['confidence'] >= HIGH_CONFIDENCE_THRESHOLD:
        similar = find_similar_failures(form_data, failure_taxonomy)
        if similar['similarity_score'].max() > 0.7:
            failure_warning = generate_failure_risk_warning(form_data, similar)

    return render_template('result.html',
        result=result, explanation=explanation,
        calibration_warning=calibration_warning, age_flag=age_flag,
        failure_warning=failure_warning, shap_pairs=shap_pairs
    )

@app.route('/risk-assessment')
def risk_assessment():
    return render_template('risk.html')

@app.route('/generate-risk', methods=['POST'])
def generate_risk():
    narrative = generate_risk_narrative(
        request.form['org_type'], request.form['use_case'], request.form['risk_appetite']
    )
    return render_template('risk-result.html', narrative=narrative)

@app.route('/audit-report')
def audit_report():
    return render_template('audit-report.html', taxonomy=failure_taxonomy.to_dict('records'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)