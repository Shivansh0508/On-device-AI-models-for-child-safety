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
