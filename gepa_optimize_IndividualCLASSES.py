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

# OPENROUTER SETUP
os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Enter OpenRouter API key: ")

# FEW-SHOT EXAMPLES — per label
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

# BASELINE PROMPTS — enhanced for rare labels
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

# LOAD DATA
df = pd.read_csv("train/eng.csv")

LABELS = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
df[LABELS] = df[LABELS].fillna(0)

# SAME 160 fixed samples as all pipelines
df_sample  = df.sample(160, random_state=42)

# Larger pool for GEPA; 500 samples for better rare label coverage
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

# API CALLS
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
                timeout=120 
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

 except requests.exceptions.ReadTimeout:
            # explicitly catch timeout and retry
            wait = 2 ** attempt
            print(f" Timeout on attempt {attempt+1}/{max_retries}, retrying in {wait}s...")
            time.sleep(wait)

        except requests.exceptions.HTTPError as e:
            if response.status_code >= 500:
                wait = 2 ** attempt
                print(f" Server error {response.status_code}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise e

        except requests.exceptions.ConnectionError:
            wait = 2 ** attempt
            print(f" Connection error, retrying in {wait}s...")
            time.sleep(wait)

    # If all retries failed — return empty string instead of crashing
    print(f" All {max_retries} attempts failed — returning default 0")
    return "0"

# BALANCED SAMPLING — fixes rare label problem
def get_balanced_sample(dataframe, label, sample_size=30):
    """Ensure positive examples exist in reflection sample"""
    positives = dataframe[dataframe[label] == 1]
    negatives = dataframe[dataframe[label] == 0]
    n_pos = min(len(positives), max(8, sample_size // 3))
    n_neg = sample_size - n_pos

    print(f"  Label {label}: {len(positives)} positives available, sampling {n_pos}")

    if len(positives) == 0:
        print(f"  No positives for {label} — using random sample")
        return dataframe.sample(min(sample_size, len(dataframe)), random_state=SEED)

    pos_sample = positives.sample(min(n_pos, len(positives)), random_state=SEED)
    neg_sample = negatives.sample(min(n_neg, len(negatives)), random_state=SEED)
    balanced   = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=SEED)
    return balanced.reset_index(drop=True)

# PARSERS
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

# BUILD SINGLE-LABEL PROMPT WITH FEW-SHOT
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

# ============================================================
# GEPA LOOP FOR SINGLE LABEL
# ============================================================

def run_gepa_single_label(label, initial_prompt, iterations=5, sample_size=30):
    print(f"\n{'='*60}")
    print(f" GEPA for label: {label.upper()}")
    print(f"Starting from: {initial_prompt[:80]}...")
    print(f"{'='*60}")

    # Check label distribution
    pos_count = int((gepa_train[label] == 1).sum())
    print(f"Positive examples in pool: {pos_count}/{len(gepa_train)}")

    current_prompt = initial_prompt
    best_prompt    = initial_prompt
    best_score     = 0.0
    history        = []

    # BALANCED sample — fixes 0.00% F1 for rare labels
    reflection_df = get_balanced_sample(gepa_train, label, sample_size)

    for iteration in range(1, iterations + 1):
        print(f"\n--- {label.upper()} | Iteration {iteration}/{iterations} ---")
# Step 1: LLaMA evaluates
        score, y_true_all, y_pred_all, texts = evaluate_single_label(
            current_prompt,
            reflection_df,
            label,
            use_few_shot=True,
            desc=f"{label} iter {iteration}"
        )
        print(f"Binary F1: {score:.4f} ({score*100:.2f}%)")

        if score > best_score:
            best_score  = score
            best_prompt = current_prompt
            print(f" New best for {label}: {best_score*100:.2f}%")

        history.append({"iteration": iteration, "score": score, "prompt": current_prompt})

        # Collect failures
        failure_cases = [
            (texts[i], y_true_all[i], y_pred_all[i])
            for i in range(len(texts))
            if y_true_all[i] != y_pred_all[i]
        ]
        print(f"Failures: {len(failure_cases)}/{sample_size}")

        if len(failure_cases) == 0:
            print(f"Perfect score for {label}!")
            break

        if iteration == iterations:
            break
            
            # Step 2: Claude Haiku reflects
        print(f" Claude Haiku reflecting on {label} failures...")
        new_prompt = gepa_reflect_single(
            current_prompt, label, failure_cases, score, iteration
        )
        print(f" New prompt: {new_prompt[:150]}...")
        current_prompt = new_prompt

    print(f"\n {label.upper()} History:")
    for h in history:
        marker = " ← best" if h['score'] == best_score else ""
        print(f"  Iter {h['iteration']}: {h['score']*100:.2f}%{marker}")

    return best_prompt, best_score, history

# ============================================================
# STEP 1: RUN GEPA FOR ALL 5 LABELS
# ============================================================

print("\n" + "="*60)
print(" PIPELINE 4 — GEPA INDIVIDUAL LABEL OPTIMIZATION")
print("Task model      : LLaMA 3.3 70B")
print("Reflection model: Claude Haiku")
print("Labels          : 5 individual optimizations")
print("="*60)

optimized_prompts = {}
best_scores       = {}
all_histories     = {}

for label in LABELS:
    opt_prompt, best_score, history = run_gepa_single_label(
        label=label,
        initial_prompt=BASELINE_PROMPTS[label],
        iterations=GEPA_ITERATIONS,
        sample_size=GEPA_SAMPLE,
    )
    optimized_prompts[label] = opt_prompt
    best_scores[label]       = best_score
    all_histories[label]     = history
    
# Save after each label — in case of interruption
    with open(f"p4_prompt_{label.replace('/', '_')}.txt", "w") as f:
        f.write(opt_prompt)
    print(f" Saved prompt for {label}")

print("\n" + "="*60)
print(" ALL OPTIMIZED PROMPTS")
print("="*60)
for label, prompt in optimized_prompts.items():
    print(f"\n--- {label.upper()} ---")
    print(prompt)
    
    # ============================================================
# STEP 2: FULL EVALUATION FUNCTION
# ============================================================

def run_pipeline4_evaluation(prompts_dict, results_file, use_few_shot=True):
    if os.path.exists(results_file):
        print(f"Loading saved results from {results_file}...")
        pred_df = pd.read_csv(results_file)
    else:
        print(f" Running Pipeline 4 on {len(df_sample)} samples...")
        results = []

        for idx, (_, row) in enumerate(tqdm(
            df_sample.iterrows(), total=len(df_sample), desc="Pipeline 4"
        )):
            preds = {}
            for label in LABELS:
                prompt       = build_single_label_prompt(
                    prompts_dict[label], row["text"], label, use_few_shot
                )
                raw          = call_llama(prompt, temperature=0.1)
                preds[label] = parse_binary(raw)
                time.sleep(0.15)  # small sleep between calls
                results.append({
                "pred_political":     preds["political"],
                "pred_racial/ethnic": preds["racial/ethnic"],
                "pred_religious":     preds["religious"],
                "pred_gender/sexual": preds["gender/sexual"],
                "pred_other":         preds["other"],

                "true_political":     int(row["political"]),
                "true_racial/ethnic": int(row["racial/ethnic"]),
                "true_religious":     int(row["religious"]),
                "true_gender/sexual": int(row["gender/sexual"]),
                "true_other":         int(row["other"]),
            })

            # Save every 20 samples — prevents data loss on interruption
            if (idx + 1) % 20 == 0:
                pd.DataFrame(results).to_csv(results_file + ".partial", index=False)
                print(f"  Partial save at {idx+1}/160")

        pred_df = pd.DataFrame(results)
        pred_df.to_csv(results_file, index=False)
        print(f" Saved to {results_file}")
         y_true = pred_df[
        ["true_political", "true_racial/ethnic", "true_religious",
         "true_gender/sexual", "true_other"]
    ].values
    y_pred = pred_df[
        ["pred_political", "pred_racial/ethnic", "pred_religious",
         "pred_gender/sexual", "pred_other"]
    ].values

    exact_match = (y_true == y_pred).all(axis=1).mean()
    macro_f1    = f1_score(y_true, y_pred, average="macro", zero_division=1)
    return exact_match, macro_f1

# ============================================================
# STEP 3: EVALUATE BASELINE — weak prompts, no few-shot
# ============================================================

print("\n===== EVALUATING BASELINE PROMPTS ON 160 SAMPLES =====")
baseline_exact, baseline_f1 = run_pipeline4_evaluation(
    BASELINE_PROMPTS,
    results_file="p4_baseline_results.csv",
    use_few_shot=False
)

# ============================================================
# STEP 4: EVALUATE GEPA OPTIMIZED — strong prompts + few-shot
# ============================================================

print("\n EVALUATING GEPA OPTIMIZED PROMPTS ON 160 SAMPLES ")

if os.path.exists("p4_gepa_results.csv"):
    os.remove("p4_gepa_results.csv")

gepa_exact, gepa_f1 = run_pipeline4_evaluation(
    optimized_prompts,
    results_file="p4_gepa_results.csv",
    use_few_shot=True
)

# ============================================================
# STEP 5: FINAL COMPARISON
# ============================================================

PIPELINE1_BASELINE_F1 = 0.3975

print("\n" + "="*60)
print(" FINAL COMPARISON (sklearn Macro-F1 on 160 samples)")
print("="*60)
print(f"{'Metric':<30} {'Pipeline 1':>12} {'P4 Baseline':>12} {'P4 GEPA':>12}")
print("-"*70)
print(f"{'Exact-match Accuracy':<30} {'N/A':>12} {baseline_exact*100:>11.2f}% {gepa_exact*100:>11.2f}%")
print(f"{'Macro-F1 (%)':<30} {PIPELINE1_BASELINE_F1*100:>11.2f}% {baseline_f1*100:>11.2f}% {gepa_f1*100:>11.2f}%")
print("="*70)

improvement_vs_p1     = (gepa_f1 - PIPELINE1_BASELINE_F1) * 100
improvement_vs_p4base = (gepa_f1 - baseline_f1) * 100
if improvement_vs_p1 > 0:
    print(f" GEPA improved vs Pipeline 1 by +{improvement_vs_p1:.2f}%")
else:
    print(f" No improvement vs Pipeline 1: {improvement_vs_p1:.2f}%")

if improvement_vs_p4base > 0:
    print(f" GEPA improved vs P4 baseline by +{improvement_vs_p4base:.2f}%")
    
    # ============================================================
# STEP 6: SAVE ALL RESULTS
# ============================================================

with open("pipeline4_optimized_prompts.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("PIPELINE 4 — GEPA INDIVIDUAL LABEL OPTIMIZATION\n")
    f.write("Task Model      : meta-llama/llama-3.3-70b-instruct\n")
    f.write("Reflection Model: anthropic/claude-3-haiku\n")
    f.write("GEPA Iterations : 5 per label\n")
    f.write("="*60 + "\n\n")

    for label in LABELS:
        f.write(f"{'='*40}\n")
        f.write(f"LABEL: {label.upper()}\n")
        f.write(f"{'='*40}\n")
        f.write(f"Baseline Prompt:\n{BASELINE_PROMPTS[label]}\n\n")
        f.write(f"Optimized Prompt:\n{optimized_prompts[label]}\n\n")
        f.write(f"GEPA History:\n")
        for h in all_histories[label]:
            f.write(f"  Iter {h['iteration']}: {h['score']*100:.2f}%\n")
            
    f.write(f"Best Binary F1 : {best_scores[label]*100:.2f}%\n\n")
    f.write("="*60 + "\n")
    f.write("FINAL PERFORMANCE\n")
    f.write("="*60 + "\n")
    f.write(f"Pipeline 1 Baseline : {PIPELINE1_BASELINE_F1*100:.2f}%\n")
    f.write(f"Pipeline 4 Baseline : {baseline_f1*100:.2f}%\n")
    f.write(f"Pipeline 4 GEPA     : {gepa_f1*100:.2f}%\n")
    f.write(f"Improvement vs P1   : {improvement_vs_p1:+.2f}%\n")
    f.write(f"Improvement vs P4   : {improvement_vs_p4base:+.2f}%\n")

print(" Saved to: pipeline4_optimized_prompts.txt")
