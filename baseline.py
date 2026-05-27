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
