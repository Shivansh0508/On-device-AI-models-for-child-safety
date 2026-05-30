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
