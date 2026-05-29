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
