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
# ── STEP 3: LOAD NON-ENGLISH FILES ────────────────────────
all_lang_files = glob(f"{TRAIN_PATH}/*.csv")
non_eng_files  = [
    f for f in all_lang_files
    if os.path.basename(f) != "eng.csv"
]

print(f"\n✅ English file   : {ENG_FILE}")
print(f"✅ Non-English files found: {len(non_eng_files)}")
for f in non_eng_files:
    print(f"   → {os.path.basename(f)}")
# ── TRANSLATION FUNCTION ──────────────────────────────────
def translate_to_english(text, retries=3):
    prompt = f"""Translate the following sentence to English.
Return ONLY the translated English text, nothing else.

Sentence: {text}

Translation:"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "MultilingualTranslation"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256
    }
for attempt in range(retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            r.raise_for_status()
            return r.json()[
                "choices"][0]["message"]["content"].strip()

        except requests.exceptions.HTTPError as e:
            if r.status_code == 402:
                raise RuntimeError(
                    "❌ Out of credits — top up openrouter.ai"
                     )
            print(f"  ⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)

        except Exception as e:
            print(f"  ⚠️ Attempt {attempt+1} error: {e}")
            time.sleep(2 ** attempt)

    return text  # fallback — keep original
# ── STEP 4: SAMPLE + TRANSLATE ────────────────────────────
all_translated_rows = []

for lang_file in non_eng_files:
    lang_code = os.path.basename(lang_file).replace(".csv","")

    try:
        df_lang = pd.read_csv(lang_file)

        # Check columns exist
        missing = [
            c for c in LABEL_COLS
            if c not in df_lang.columns
        ]
        if missing:
            print(f"\n⚠️ Skipping {lang_code} "
                  f"— missing: {missing}")
            continue
            df_lang = df_lang.dropna(subset=["text"])
        n       = min(SAMPLES_PER_LANG, len(df_lang))
        sample  = df_lang.sample(n, random_state=SEED)

        print(f"\n🌍 [{lang_code.upper()}] "
              f"translating {n} samples...")

        for idx, (_, row) in enumerate(sample.iterrows()):
            original   = str(row["text"]).strip()
            translated = translate_to_english(original)

            all_translated_rows.append({
                "text":          translated,
                "original_text": original,
                "source_lang":   lang_code,
                "political":     int(row["political"]),
                "racial/ethnic": int(row["racial/ethnic"]),
                "religious":     int(row["religious"]),
                "gender/sexual": int(row["gender/sexual"]),
                "other":         int(row["other"]),
            })
if (idx + 1) % 10 == 0:
                print(f"   ✅ {idx+1}/{n} done")

            time.sleep(0.4)

    except RuntimeError as e:
        print(str(e))
        break

    except Exception as e:
        print(f"\n❌ Error with {lang_code}: {e}")
        continue
# ── STEP 5: SAVE + COMBINE ────────────────────────────────
df_translated  = pd.DataFrame(all_translated_rows)

# Save to same folder as eng.csv — guaranteed to exist
out_translated = os.path.join(TRAIN_PATH, "translated_samples.csv")
out_combined   = os.path.join(TRAIN_PATH, "combined_eng_multilingual.csv")
out_results    = os.path.join(BASE_PATH,  "eval_results_multilingual.csv")

df_translated.to_csv(out_translated, index=False)
print(f"\n💾 Saved translated → {out_translated}")

df_english     = pd.read_csv(ENG_FILE)
df_combined    = pd.concat(
    [
        df_english[["text"] + LABEL_COLS],
        df_translated[["text"]  + LABEL_COLS]
    ],
    ignore_index=True
)
df_combined.to_csv(out_combined, index=False)
