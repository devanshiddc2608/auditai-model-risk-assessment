import pandas as pd
from config import OUTPUTS_DIR, HIGH_CONFIDENCE_THRESHOLD, MISCALIBRATED_ZONE, FLAGGED_AGE_BAND

def load_failure_taxonomy():
    return pd.read_csv(f'{OUTPUTS_DIR}/failure_taxonomy.csv')

def check_calibration_zone(confidence):
    lo, hi = MISCALIBRATED_ZONE
    return lo <= confidence <= hi

def check_flagged_age_band(age):
    lo, hi = FLAGGED_AGE_BAND
    return lo <= age <= hi

def find_similar_failures(case: dict, taxonomy_df: pd.DataFrame, top_n=3):
    """Simple, fast similarity: normalized distance on key numeric features."""
    compare_cols = ['DebtRatio', 'MonthlyIncome', 'age', 'RevolvingUtilizationOfUnsecuredLines']
    df = taxonomy_df.copy()
    for col in compare_cols:
        col_range = df[col].max() - df[col].min()
        if col_range == 0:
            df[f'{col}_dist'] = 0
        else:
            df[f'{col}_dist'] = ((df[col] - case[col]).abs()) / col_range
    dist_cols = [f'{c}_dist' for c in compare_cols]
    df['similarity_score'] = 1 - df[dist_cols].mean(axis=1)
    return df.sort_values('similarity_score', ascending=False).head(top_n)