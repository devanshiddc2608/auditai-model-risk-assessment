import shap

def get_shap_explanation(xgb_model_raw, row_scaled, feature_names, top_n=5):
    explainer = shap.TreeExplainer(xgb_model_raw)
    shap_values = explainer(row_scaled)
    values = shap_values.values[0]
    pairs = sorted(zip(feature_names, values), key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]  # [(feature_name, shap_value), ...]