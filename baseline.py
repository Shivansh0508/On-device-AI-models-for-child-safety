import pandas as pd
import numpy as np
import requests
import time
import os
import json
import re
from tqdm import tqdm
from sklearn.metrics import f1_score
import getpass
# Load API key
os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Enter your OpenRouter API key: ")
print("API key loaded:", "OPENROUTER_API_KEY" in os.environ)

# Load dataset
df = pd.read_csv("train/eng.csv")
label_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
df[label_cols] = df[label_cols].fillna(0)
print(df.head())
print("Label distribution:")
print(df[label_cols].sum())
# Fixed 160 samples 
df_sample = df.sample(160, random_state=42)

# BASELINE: Simple Prompt 
BASELINE_PROMPT = """
Classify this sentence.
Labels: political, racial_ethnic, religious, gender_sexual, other.
Return 0 or 1 for each.
Sentence: {text}
Format: {{"political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0}}
"""
def call_model(prompt, temperature=0.0, max_retries=5):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",  "Content-Type": "application/json", "HTTP-Referer": "https://localhost", "X-Title": "SemEval-Polarization-Baseline"
    }

    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct", "messages": [{"role": "user", "content": prompt}],  "temperature": temperature
    }
