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
    
def run_evaluation(prompt_template, results_file, label=""):
    """Run inference on all 160 samples and evaluate with sklearn"""

    if os.path.exists(results_file):
        print(f" Loading saved {label} results...")
        pred_df = pd.read_csv(results_file)
    else:
        print(f" Running {label} inference on 160 samples...")
        results = []

        for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
            prompt = prompt_template + f"""

Sentence: {row['text']}
Return ONLY a JSON object:
{{"political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0}}"""
            raw = call_gemma(prompt, temperature=0.1)
            pred = relaxed_parse(raw)

 results.append({
                "pred_political":     pred["political"],
                "pred_racial/ethnic": pred["racial/ethnic"],
                "pred_religious":     pred["religious"],
                "pred_gender/sexual": pred["gender/sexual"], 
                "pred_other":         pred["other"],

                "true_political":     row["political"],
                "true_racial/ethnic": row["racial/ethnic"],
                "true_religious":     row["religious"],
                "true_gender/sexual": row["gender/sexual"],
                "true_other":         row["other"],
            })
            time.sleep(0.3)

        pred_df = pd.DataFrame(results)
        pred_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")

    y_true = pred_df[["true_political", "true_racial/ethnic", "true_religious",n"true_gender/sexual", "true_other"]].values

    y_pred = pred_df[["pred_political", "pred_racial/ethnic", "pred_religious", "pred_gender/sexual", "pred_other"]].values

    exact_match = (y_true == y_pred).all(axis=1).mean()
    macro_f1    = f1_score(y_true, y_pred, average="macro", zero_division=1)

    return exact_match, macro_f1

# EVALUATE BASELINE ON 160 SAMPLES
BASELINE_PROMPT = """Classify this sentence.
Labels: political, racial_ethnic, religious, gender_sexual, other.
Return 0 or 1 for each."""

print("\n EVALUATING BASELINE PROMPT ON 160 SAMPLES ")
baseline_exact, baseline_f1 = run_evaluation(
    BASELINE_PROMPT,
    results_file="baseline_results.csv",
    label="Baseline"
)

# EVALUATE OPTIMIZED PROMPT ON 160 SAMPLES
print("\n EVALUATING GEPA OPTIMIZED PROMPT ON 160 SAMPLES ")

if os.path.exists("gepa_results.csv"):
    os.remove("gepa_results.csv")

gepa_exact, gepa_f1 = run_evaluation(
    OPTIMIZED_PROMPT,
    results_file="gepa_results.csv",
    label="GEPA Optimized"
)

# FINAL COMPARISON
print("\n" + "="*60)
print(" FINAL COMPARISON (sklearn Macro-F1 on 160 samples)")
print("="*60)
print(f"{'Metric':<30} {'Baseline':>12} {'GEPA Optimized':>15}")
print("-"*60)
print(f"{'Exact-match Accuracy':<30} {baseline_exact*100:>11.2f}% {gepa_exact*100:>14.2f}%")
print(f"{'Macro-F1 (%)':<30} {baseline_f1*100:>11.2f}% {gepa_f1*100:>14.2f}%")
print("="*60)

improvement = (gepa_f1 - baseline_f1) * 100
if improvement > 0:
    print(f" GEPA improved Macro-F1 by +{improvement:.2f}%")
else:
    print(f" No improvement: {improvement:.2f}%")














































































