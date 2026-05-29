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
# ============================================================
# CONFIG
# ============================================================

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
# FEW-SHOT EXAMPLES WITH REASONING + SELF-CORRECTION
# ============================================================

FEW_SHOT_WITH_REASONING = [
