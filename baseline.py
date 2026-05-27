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
for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            if response.status_code >= 500:
                wait = 2 ** attempt
                print(f" Server error {response.status_code}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e
    raise RuntimeError(" OpenRouter failed after multiple retries")

def relaxed_parse(output):
    binary = {k: 0 for k in label_cols}
    try:
        match = re.search(r'\{.*?\}', output, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            binary["political"]     = int(parsed.get("political", 0))
            binary["racial/ethnic"] = int(parsed.get("racial_ethnic", 0))
            binary["religious"]     = int(parsed.get("religious", 0))
            binary["gender/sexual"] = int(parsed.get("gender_sexual", 0))
            binary["other"]         = int(parsed.get("other", 0))
    except Exception as e:
        print(f"Parse error: {e} | Output: {output[:100]}")
    return binary
