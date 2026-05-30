import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import requests
import time
import json
import re
from tqdm import tqdm
import getpass
# ============================================================
# CONFIG
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

GEPA_ITERATIONS = 5
GEPA_SAMPLE     = 30
# ============================================================
# OPENROUTER SETUP
# ============================================================

os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Enter OpenRouter API key: ")
# ============================================================
# FEW-SHOT EXAMPLES WITH REASONING + SELF-CORRECTION
# ============================================================

FEW_SHOT_WITH_REASONING = [
{
        "text": "Democrats are destroying this country with their socialist agenda.",
        "reasoning": "The sentence directly attacks a political party (Democrats) using emotionally charged language ('destroying') and labels their ideology negatively ('socialist agenda'). It expresses strong partisan bias designed to create political division. No racial, religious, gender, or other group is targeted.",
        "self_correction": "✅ Classification is correct. Political=1 because it attacks a political party with hostile language. No corrections needed.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
{
        "text": "Immigrants are stealing jobs from real Americans.",
        "reasoning": "This sentence promotes a harmful stereotype that immigrants take jobs from citizens using dehumanizing framing ('stealing'). It targets an ethnic/national group with hostile language. Initially I considered political=1 but self-correction shows this is primarily racial/ethnic — it attacks a group based on national origin, not a political party.",
        "self_correction": "⚠️ Corrected political from 1 to 0 because the sentence targets immigrants as an ethnic/national group, not a political party. racial_ethnic=1 is correct.",
        "political": 0, "racial_ethnic": 1, "religious": 0, "gender_sexual": 0, "other": 0
    },
        {
        "text": "Muslims are a threat to our Christian values and way of life.",
        "reasoning": "This sentence frames an entire religious group (Muslims) as a threat to another religious group (Christians). It promotes religious intolerance and division by portraying one faith as inherently dangerous to another. This is clear religious polarization.",
        "self_correction": "✅ Classification is correct. Religious=1 because it attacks Muslims and frames them as threatening. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 1, "gender_sexual": 0, "other": 0
    },
