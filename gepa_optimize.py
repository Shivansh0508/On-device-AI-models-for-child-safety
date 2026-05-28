import os
import random
import numpy as np
import pandas as pd
import dspy
from dspy.teleprompt import MIPROv2
from sklearn.metrics import f1_score
import requests
import time
import json
import re
from tqdm import tqdm
import getpass

# CONFIG
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# OPENROUTER SETUP
os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Enter OpenRouter API key: ")

# Task model 
task_lm = dspy.LM(
    model="openrouter/google/gemma-3-27b-it",
    temperature=0.1,
    max_tokens=128,
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Reflection model 
prompt_lm = dspy.LM(
    model="openrouter/meta-llama/llama-3.3-70b-instruct",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.environ["OPENROUTER_API_KEY"],
)
dspy.configure(lm=task_lm)

# LOAD DATA: 160 samples
df = pd.read_csv("train/eng.csv")
LABELS = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
df[LABELS] = df[LABELS].fillna(0)

# SAME 160 samples 
df_sample = df.sample(160, random_state=42)

# Split for MIPRO: train (110) and val (50)
train_df = df_sample.iloc[:110]
val_df   = df_sample.iloc[110:]
print(f"Total samples : 160")
print(f"MIPRO Train   : {len(train_df)}")
print(f"MIPRO Val     : {len(val_df)}")

# DSPy DATA FORMAT
def to_examples(dataframe):
    examples = []
    for _, row in dataframe.iterrows():
        examples.append(
            dspy.Example(
                text=row["text"],
                labels={l: int(row[l]) for l in LABELS},
            ).with_inputs("text")
        )
    return examples
trainset = to_examples(train_df)
valset   = to_examples(val_df)

# BASELINE SIGNATURE — same/simple prompt
class PolarizationSignature(dspy.Signature):
    """Classify this sentence.
    Labels: political, racial_ethnic, religious, gender_sexual, other.
    Return 0 or 1 for each."""

    text          = dspy.InputField()
    political     = dspy.OutputField(desc="0 or 1")
    racial_ethnic = dspy.OutputField(desc="0 or 1")
    religious     = dspy.OutputField(desc="0 or 1")
    gender_sexual = dspy.OutputField(desc="0 or 1")
    other         = dspy.OutputField(desc="0 or 1")

class PolarizationProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PolarizationSignature)

    def forward(self, text):
        result = self.predict(text=text)
        for field in ["political", "racial_ethnic", "religious", "gender_sexual", "other"]:
            try:
                val = getattr(result, field)
                if isinstance(val, str):
                    val = val.strip()
                    if val.lower() in ['0', 'no', 'false', 'none']:
                        setattr(result, field, 0)
                    elif val.lower() in ['1', 'yes', 'true']:
                        setattr(result, field, 1)
                    else:
                        setattr(result, field, int(float(val)))
                else:
                    setattr(result, field, int(val))
            except:
                setattr(result, field, 0)
        return result
program = PolarizationProgram()

# MIPRO METRIC
def eval_metric(example, prediction, trace=None):
    try:
        label_map = {
            "political":     "political",
            "racial/ethnic": "racial_ethnic",
            "religious":     "religious",
            "gender/sexual": "gender_sexual",
            "other":         "other"
        }
        y_true, y_pred = [], []
        for csv_label, pred_field in label_map.items():
            y_true.append(int(example["labels"][csv_label]))
            try:
                val = getattr(prediction, pred_field)
                if isinstance(val, str):
                    val = val.strip()
                    if val.lower() in ['0', 'no', 'false', 'none']:
                        y_pred.append(0)
                    elif val.lower() in ['1', 'yes', 'true']:
                        y_pred.append(1)
                    else:
                        y_pred.append(int(float(val)))
                else:
                    y_pred.append(int(val))
            except:
                y_pred.append(0)
        return f1_score([y_true], [y_pred], average="macro", zero_division=0)
    except:
        return 0.0

# MIPRO OPTIMIZATION
print("\n STARTING MIPRO OPTIMIZATION ")
print("Task model      : google/gemma-3-27b-it")
print("Reflection model: meta-llama/llama-3.3-70b-instruct")

optimizer = MIPROv2(
    metric=eval_metric,
    prompt_model=prompt_lm,
    task_model=task_lm,
    auto="heavy",        # more thorough optimization
    init_temperature=0.7,
    verbose=True,
)

optimized_program = optimizer.compile(
    student=program,
    trainset=trainset,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
)

# Extract optimized prompt
OPTIMIZED_PROMPT = optimized_program.predict.signature.instructions

print("\n" + "="*60)
print(" OPTIMIZED PROMPT (AFTER MIPRO)")
print("="*60)
print(OPTIMIZED_PROMPT)
print("="*60)

# SKLEARN EVALUATION FUNCTION
label_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]

def call_gemma(prompt, temperature=0.1, max_retries=5):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "SemEval-Polarization-GEPA"
    }
    payload = {
        "model": "google/gemma-3-27b-it",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
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


























































































