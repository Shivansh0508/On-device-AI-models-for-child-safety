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

# CONFIG
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
# FEW-SHOT EXAMPLES — per label
# ============================================================

FEW_SHOT_PER_LABEL = {

    "political": [
        {"text": "Democrats are destroying this country with their socialist agenda.", "label": 1},
        {"text": "Trump is a fascist who belongs in prison.", "label": 1},
        {"text": "The rich elites control everything while the working class suffers.", "label": 1},
        {"text": "Putin is a war criminal who should be tried at the ICC.", "label": 1},
        {"text": "The government announced new infrastructure spending plans today.", "label": 0},
        {"text": "The election results will be announced tomorrow morning.", "label": 0},
        {"text": "Congress passed a new bill on healthcare funding.", "label": 0},
    ],

    "racial/ethnic": [
        {"text": "Immigrants are stealing jobs from real Americans.", "label": 1},
        {"text": "These foreign invaders are ruining our culture.", "label": 1},
        {"text": "Black lives matter and anyone who disagrees is a racist.", "label": 1},
        {"text": "Israel is committing genocide against Palestinians.", "label": 1},
        {"text": "The neighborhood has become very diverse over the past decade.", "label": 0},
        {"text": "Immigration policy was discussed at the senate hearing.", "label": 0},
        {"text": "The census recorded population changes across ethnic groups.", "label": 0},
    ],

    "religious": [
        {"text": "Muslims are a threat to our Christian values and way of life.", "label": 1},
        {"text": "The government is waging a war on Christianity.", "label": 1},
        {"text": "Jews control the media and the banks.", "label": 1},
        {"text": "Atheists are immoral people who cannot be trusted.", "label": 1},
        {"text": "All Muslims are potential terrorists and should be watched.", "label": 1},
        {"text": "The church held its annual fundraiser this weekend.", "label": 0},
        {"text": "Ramadan began this week across Muslim communities.", "label": 0},
        {"text": "The interfaith dialogue conference was held in Geneva.", "label": 0},
    ],

    "gender/sexual": [
        {"text": "Trans people are mentally ill and should not be near children.", "label": 1},
        {"text": "The radical left is pushing their LGBT agenda on our kids.", "label": 1},
        {"text": "Women should stay home and raise children instead of working.", "label": 1},
        {"text": "Gay marriage is an abomination against nature.", "label": 1},
        {"text": "Feminists are destroying traditional family values.", "label": 1},
        {"text": "The conference discussed gender equality in the workplace.", "label": 0},
        {"text": "Pride month celebrations were held across major cities.", "label": 0},
        {"text": "The study analyzed wage gaps between men and women.", "label": 0},
    ],

    "other": [
        {"text": "Boomers ruined the economy and now expect young people to be grateful.", "label": 1},
        {"text": "The rich elites control everything and the poor deserve nothing.", "label": 1},
        {"text": "Southerners are all racist and uneducated.", "label": 1},
        {"text": "Anyone who disagrees with me is stupid and brainwashed.", "label": 1},
        {"text": "Old people should just retire and stop blocking progress.", "label": 1},
        {"text": "The economy grew by 3 percent last quarter.", "label": 0},
        {"text": "Scientists published new findings on climate change.", "label": 0},
        {"text": "The sports team won their regional championship.", "label": 0},
    ],
}

# ============================================================
# BASELINE PROMPTS — enhanced for rare labels
# ============================================================

BASELINE_PROMPTS = {
    "political": "Is this sentence politically polarizing? Return 0 or 1.",

    "racial/ethnic": "Is this sentence racially or ethnically polarizing? Return 0 or 1.",

    "religious": """Does this sentence express hatred, intolerance, or strong negative bias toward a religious group?
Examples of 1: "Muslims are terrorists", "Jews control everything", "Atheists are immoral"
Examples of 0: "The church held a service", "Ramadan started this week"
Return 1 if religiously polarizing, 0 if not.""",

    "gender/sexual": """Does this sentence express hatred, discrimination or strong negative bias toward women, men, or LGBTQ+ people?
Examples of 1: "Trans people are mentally ill", "Women belong in the kitchen", "Gay marriage is wrong"
Examples of 0: "The study analyzed gender gaps", "Pride month celebrations began"
Return 1 if gender/sexual polarizing, 0 if not.""",

    "other": """Does this sentence express hatred or strong negative bias based on age, class, region, or any social group NOT covered by political/racial/religious/gender categories?
Examples of 1: "Boomers ruined everything", "The poor deserve to suffer", "Southerners are all racist"
Examples of 0: "The economy grew this quarter", "Scientists published new findings"
Return 1 if polarizing in other way, 0 if not.""",
}

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("train/eng.csv")

LABELS = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
df[LABELS] = df[LABELS].fillna(0)

# SAME 160 fixed samples as all pipelines
df_sample  = df.sample(160, random_state=42)

# Larger pool for GEPA — 500 samples for better rare label coverage
df_gepa    = df.sample(500, random_state=42)
gepa_train = df_gepa.iloc[:450]
gepa_val   = df_gepa.iloc[450:500]

print(f"GEPA Reflection Pool : {len(gepa_train)}")
print(f"GEPA Val             : {len(gepa_val)}")
print(f"Final Eval           : {len(df_sample)} samples")

# Print label distribution
print("\nLabel distribution in GEPA pool:")
for label in LABELS:
    pos = int((gepa_train[label] == 1).sum())
    print(f"  {label:<20}: {pos}/{len(gepa_train)} positive ({pos/len(gepa_train)*100:.1f}%)")

label_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]

# ============================================================
# API CALLS
# ============================================================

def call_model(prompt, model, temperature=0.1, max_retries=5):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "SemEval-Pipeline4-GEPA"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 512,
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=120   # ← increase from 60 to 120 seconds
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

 except requests.exceptions.ReadTimeout:
            # ← explicitly catch timeout and retry
            wait = 2 ** attempt
            print(f"⚠️ Timeout on attempt {attempt+1}/{max_retries}, retrying in {wait}s...")
            time.sleep(wait)

        except requests.exceptions.HTTPError as e:
            if response.status_code >= 500:
                wait = 2 ** attempt
                print(f"⚠️ Server error {response.status_code}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e

        except requests.exceptions.ConnectionError:
            wait = 2 ** attempt
            print(f"⚠️ Connection error, retrying in {wait}s...")
            time.sleep(wait)

    # If all retries failed — return empty string instead of crashing
    print(f"❌ All {max_retries} attempts failed — returning default 0")
    return "0"

# ============================================================
# BALANCED SAMPLING — fixes rare label problem
# ============================================================

def get_balanced_sample(dataframe, label, sample_size=30):
    """Ensure positive examples exist in reflection sample"""
    positives = dataframe[dataframe[label] == 1]
    negatives = dataframe[dataframe[label] == 0]

    n_pos = min(len(positives), max(8, sample_size // 3))
    n_neg = sample_size - n_pos

    print(f"  Label {label}: {len(positives)} positives available, sampling {n_pos}")

    if len(positives) == 0:
        print(f"  ⚠️ No positives for {label} — using random sample")
        return dataframe.sample(min(sample_size, len(dataframe)), random_state=SEED)

    pos_sample = positives.sample(min(n_pos, len(positives)), random_state=SEED)
    neg_sample = negatives.sample(min(n_neg, len(negatives)), random_state=SEED)
    balanced   = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=SEED)
    return balanced.reset_index(drop=True)

# ============================================================
# PARSERS
# ============================================================

def parse_binary(output):
    try:
        output = output.strip().lower()
        if output in ['0', 'no', 'false', 'none', 'not present', 'absent']:
            return 0
        if output in ['1', 'yes', 'true', 'present']:
            return 1
        match = re.search(r'\b([01])\b', output)
        if match:
            return int(match.group(1))
        json_match = re.search(r'\{.*?\}', output, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            val = list(parsed.values())[0]
            return int(bool(val))
    except:
        pass
    return 0

# ============================================================
# BUILD SINGLE-LABEL PROMPT WITH FEW-SHOT
# ============================================================

def build_single_label_prompt(instruction, text, label, use_few_shot=True):
    few_shot_block = ""
    if use_few_shot and label in FEW_SHOT_PER_LABEL:
        few_shot_block = "\n\nHere are some labeled examples:\n"
        for ex in FEW_SHOT_PER_LABEL[label]:
            few_shot_block += f"\nText: {ex['text']}\nAnswer: {ex['label']}\n"
        few_shot_block += "\nNow classify the following:\n"

    return instruction + few_shot_block + f"""

Sentence: {text}

Return ONLY 0 or 1. No explanation."""

# ============================================================
# EVALUATE SINGLE LABEL PROMPT
# ============================================================

def evaluate_single_label(instruction, dataframe, label, use_few_shot=True, desc=""):
    y_true_all, y_pred_all, texts = [], [], []

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=desc):
        prompt = build_single_label_prompt(instruction, row["text"], label, use_few_shot)
        raw    = call_llama(prompt, temperature=0.1)
        pred   = parse_binary(raw)

        y_true_all.append(int(row[label]))
        y_pred_all.append(pred)
        texts.append(row["text"])
        time.sleep(0.2)

    f1 = f1_score(y_true_all, y_pred_all, average="binary", zero_division=1)
    return f1, y_true_all, y_pred_all, texts

# ============================================================
# GEPA REFLECTION FOR SINGLE LABEL
# ============================================================

def gepa_reflect_single(current_prompt, label, failure_cases, current_score, iteration):
    label_descriptions = {
        "political":     "politically polarizing content — strong partisan bias, attacks on political groups, promotes political division",
        "racial/ethnic": "racial or ethnic polarization — prejudice, stereotypes, hatred toward racial/ethnic groups, xenophobia",
        "religious":     "religious polarization — attacks on or hatred toward religious groups or beliefs, religious intolerance",
        "gender/sexual": "gender or sexual orientation polarization — sexism, homophobia, transphobia, gender-based hatred",
        "other":         "other polarization — class-based hatred, ageism, regional hatred, general dehumanization not covered by other categories",
    }

    failure_text = ""
    fp_count, fn_count = 0, 0
    for i, (text, true, pred) in enumerate(failure_cases[:12]):
        if pred > true:
            error_type = "FALSE POSITIVE (predicted 1, true is 0 — model over-predicted)"
            fp_count += 1
        else:
            error_type = "FALSE NEGATIVE (predicted 0, true is 1 — model missed it)"
            fn_count += 1
        failure_text += f"""
Example {i+1} [{error_type}]:
  Text      : {text}
  True label: {true}
  Predicted : {pred}
"""

    reflection_prompt = f"""You are an expert NLP prompt engineer specializing in hate speech and polarization detection.

You are optimizing a BINARY classifier prompt for LLaMA 3.3 70B.

TARGET LABEL: {label.upper()}
Definition: {label_descriptions[label]}

CURRENT PROMPT (Iteration {iteration}):
===
{current_prompt}
===

CURRENT BINARY F1 SCORE: {current_score:.4f} ({current_score*100:.2f}%)
False Positives: {fp_count} (model predicted 1 when true is 0)
False Negatives: {fn_count} (model predicted 0 when true is 1)

LLaMA made the following ERRORS:
{failure_text}

YOUR TASK:
1. Analyze whether the model is over-predicting or under-predicting {label.upper()}
2. Identify the specific patterns causing errors
3. Rewrite the prompt to fix these errors
REQUIREMENTS for new prompt:
- Give CLEAR definition of what IS {label.upper()}=1
- Give CLEAR definition of what IS NOT {label.upper()}=1
- If too many False Negatives: make the definition broader, more sensitive
- If too many False Positives: make the definition stricter, more precise
- Be explicit: merely MENTIONING a topic is NOT polarizing
- The text must express BIAS, HATRED or STRONG NEGATIVE STEREOTYPES to be 1
- Do NOT include few-shot examples (added separately)
- Do NOT include "Return 0 or 1" (added separately)
- Do NOT include sentence placeholder (added separately)

Return ONLY the new improved prompt instruction. No explanation."""

    new_prompt = call_haiku(reflection_prompt, temperature=0.7)
    return new_prompt.strip()
