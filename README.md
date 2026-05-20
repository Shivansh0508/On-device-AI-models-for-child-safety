# Multi-Label Polarisation Classification | SemEval-2026 Task 9

Detecting polarising language across five categories in social media text using prompt engineering and automated prompt optimisation with Gemma 3 27B — no model fine-tuning required.

## Results

| Method | Model | Samples | Macro-F1 | Δ |
|---|---|---|---|---|
| Zero-Shot Baseline | Gemma 3 27B | 200 | 39.5% | — |
| GEPA Optimised | Gemma 3 27B | 200 | 41.1% | +1.6 pts |
| MIPROv2 Optimised | Gemma 3 27B | 200 | 42.0% | +2.5 pts |

## Setup

```bash
pip install dspy-ai gepa datasets litellm openai scikit-learn pandas numpy requests tqdm fastapi uvicorn
```

```bash
export OPENROUTER_API_KEY="your-key-here"
```

## Dataset

Place SemEval-2026 Task 9 CSV files in the `train/` folder named by language code (`eng.csv`, `arb.csv`, `ben.csv` ...). Each file must contain: `text, political, racial/ethnic, religious, gender/sexual, other`

## Run

```bash
# Stage 1 - Baseline
python baseline.py

# Stage 2 - GEPA
python gepa_optimize.py

# Stage 3 - MIPROv2
python miprov2_optimize.py

# Stage 4 - Multilingual Translation
python translate_multilingual.py

# Stage 5 - Evaluate
python evaluate.py
```

## Deploy API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "your sentence here"}'
```

Response:
```json
{"political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0}
```

## Stack

DSPy · GEPA · MIPROv2 · Gemma 3 27B · OpenRouter · scikit-learn · FastAPI

