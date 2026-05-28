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

# CONFIG# 
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
