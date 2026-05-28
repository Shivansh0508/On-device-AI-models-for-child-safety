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
