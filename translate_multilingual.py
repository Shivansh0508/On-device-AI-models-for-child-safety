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
# ── STEP 2: AUTO-DETECT YOUR PATHS ───────────────────────
# Searches your entire Drive for eng.csv to find the real path
print("🔍 Searching for your dataset files...")

found = glob(
    "/content/gdrive/My Drive/Colab Notebooks/On Device AI models for Child Safety/train/*.csv",
    recursive=True
)

# Show what we found
print(f"Found {len(found)} CSV files total:")
for f in found:
    print(f"  {f}")

# Auto-detect train folder from eng.csv location
eng_files = [f for f in found if os.path.basename(f) == "eng.csv"]
if not eng_files:
    raise FileNotFoundError(
        "❌ eng.csv not found anywhere in your Drive. "
        "Check your Drive is mounted and files are uploaded."
    )

ENG_FILE   = eng_files[0]
TRAIN_PATH = os.path.dirname(ENG_FILE)
BASE_PATH  = os.path.dirname(TRAIN_PATH)

print(f"\n✅ Auto-detected paths:")
print(f"   ENG_FILE   = {ENG_FILE}")
print(f"   TRAIN_PATH = {TRAIN_PATH}")
print(f"   BASE_PATH  = {BASE_PATH}")
