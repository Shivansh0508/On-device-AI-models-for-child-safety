import pandas as pd
import numpy as np
import requests
import os
import json
import re
import time
from tqdm import tqdm
from glob import glob
from sklearn.metrics import f1_score
import getpass
# ── API SETUP ─────────────────────────────────────────────
os.environ["OPENROUTER_API_KEY"] = getpass.getpass(
    "Enter OpenRouter API key: "
)
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL   = "google/gemma-3-27b-it"

LABEL_COLS       = [
    "political", "racial/ethnic",
    "religious", "gender/sexual", "other"
]
SAMPLES_PER_LANG = 50
SEED             = 42
