import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GROQ_MODEL = 'llama-3.1-8b-instant'

HIGH_CONFIDENCE_THRESHOLD = 0.85
COST_OPTIMAL_THRESHOLD = 0.86
COST_FALSE_POSITIVE = 5000
COST_FALSE_NEGATIVE = 800
MISCALIBRATED_ZONE = (0.70, 0.85)  # flagged from reliability diagram
FLAGGED_AGE_BAND = (25, 40)

MODEL_DIR = 'model'
OUTPUTS_DIR = 'outputs'