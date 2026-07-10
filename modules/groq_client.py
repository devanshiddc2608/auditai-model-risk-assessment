from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

def generate_prediction_explanation(prediction_label, confidence, shap_pairs):
    shap_text = "\n".join([f"- {name}: {value:+.3f}" for name, value in shap_pairs])
    system_prompt = (
        "You are a model risk analyst writing for a non-technical business stakeholder. "
        "Explain the prediction using ONLY the SHAP values provided below. "
        "Do not invent numbers, causes, or facts not present in this data. "
        "Keep it to 3-4 sentences, plain English, no jargon."
    )
    user_prompt = (
        f"Prediction: {'High risk of default' if prediction_label == 1 else 'Low risk of default'}\n"
        f"Model confidence: {confidence:.1%}\n"
        f"Top contributing factors (SHAP values, positive = pushes toward default risk):\n{shap_text}\n\n"
        f"Explain this decision in plain English for a business reviewer."
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def generate_failure_risk_warning(case_summary, similar_failures_df):
    failures_text = similar_failures_df[['confidence','failure_category']].to_string(index=False)
    system_prompt = (
        "You are a model risk analyst. Given a new case and a list of similar PAST cases "
        "where the model was highly confident but WRONG, write a short warning (3 sentences max) "
        "explaining the similarity and one concrete mitigation action. "
        "Only use the data provided. Do not fabricate case details."
    )
    user_prompt = (
        f"New case summary: {case_summary}\n\n"
        f"Similar past high-confidence failures:\n{failures_text}\n\n"
        f"Write the warning."
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def generate_risk_narrative(org_type, use_case, risk_appetite):
    system_prompt = (
        "You are a model risk consultant producing a structured risk narrative. "
        "Format your response with these exact section headings: "
        "Key Risks, Regulatory Considerations, Monitoring Framework, Human Oversight Requirements. "
        "Keep each section to 2-3 bullet points. Be specific to the organisation type and use case given."
    )
    user_prompt = (
        f"Organisation type: {org_type}\n"
        f"Intended model use: {use_case}\n"
        f"Risk appetite: {risk_appetite}\n\n"
        f"Generate the risk narrative."
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content