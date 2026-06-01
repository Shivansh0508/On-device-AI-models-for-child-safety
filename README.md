# Multi-Label Polarisation Classification | SemEval-2026 Task 9

Detecting polarising language across five categories in social media text using prompt engineering and automated prompt optimisation with Gemma 3 27B. Built as part of the **On-Device AI Models for Child Safety** project. No model fine-tuning required.

---

## What This Project Does

Social media posts often carry hostility across multiple dimensions at once , a single sentence can be politically charged, racially biased, and religiously inflammatory simultaneously. Standard toxicity detectors reduce this to a single flag and lose all categorical detail that moderation actually needs.

This system classifies each sentence across five polarisation categories simultaneously. A sentence only gets a positive label if it actively **expresses** hostility , not if it merely mentions the topic.

| Label | Positive (1) if... |
|---|---|
| Political | Actively attacks a party, leader or ideology |
| Racial/Ethnic | Dehumanises or stereotypes an ethnic group |
| Religious | Frames a faith community as dangerous or inferior |
| Gender/Sexual | Expresses hostility based on gender or sexuality |
| Other | Targets a social group not covered above |

---

## Results

| Method | Model | Samples | Macro-F1 | Δ |
|---|---|---|---|---|
| Zero-Shot Baseline | Gemma 3 27B | 200 | 39.5% |, |
| GEPA Optimised | Gemma 3 27B | 200 | 41.1% | +1.6 pts |
| MIPROv2 Optimised | Gemma 3 27B | 200 | 42.0% | +2.5 pts |

---

## Five-Stage Pipeline

```
Stage 1, Zero-Shot Baseline
        ↓
Stage 2, GEPA Prompt Optimisation
        ↓
Stage 3, MIPROv2 Prompt Optimisation
        ↓
Stage 4, Multilingual Translation Augmentation
        ↓
Stage 5, Evaluation + Confidence Intervals
```

**Stage 1, Zero-Shot Baseline**
Runs Gemma 3 27B with a minimal prompt, no definitions, no examples. Establishes the performance floor and exposes where the model fails by default.

**Stage 2, GEPA Optimisation**
Evolutionary prompt search over 100 training samples. Generates candidate prompts, evaluates each, keeps the best, mutates and repeats. Adds bootstrapped few-shot demonstrations automatically.

**Stage 3, MIPROv2 Optimisation**
Bayesian prompt optimisation within DSPy. More sample-efficient than GEPA on small training pools. Discovers improved instructions and up to 3 few-shot demonstrations.

**Stage 4, Multilingual Translation**
Samples 50 examples from each of 21 non-English language files, translates to English using Gemma 3 27B via OpenRouter, and combines with the English dataset to expand the training pool.

Languages: Arabic · Bengali · German · Hindi · Persian · Spanish · Turkish · Russian · Chinese · Polish · Swahili · Italian · Nepali · Punjabi · Burmese · Khmer · Hausa · Amharic · Odia · Telugu · Urdu

**Stage 5, Evaluation**
Full evaluation with macro-F1, per-label F1, exact-match accuracy, 95% bootstrap confidence intervals, and parse failure tracking.

---

## Project Structure

```
📦 root
├── baseline.py                                    # Stage 1
├── gepa_optimize_allCLASSES.py                    # Stage 2
├── gepa_optimize_IndividualCLASSES.py             # Stage 3
├── ChatGpt_Prompt_Examples.py                     # Stage 4
├── translate_multilingual.py                      # Stage 5
├── postman_collection.json                        # API test collection
├──Datasets
├── 📂 train/
│   ├── eng.csv                  # English data
│   ├── arb.csv                  # Arabic
│   ├── ben.csv                  # Bengali
│   └── ...                      # 19 more languages
├── 📂 dev/
│   ├──  eng.csv                  # English data
│   ├── arb.csv                  # Arabic
│   ├── ben.csv                  # Bengali
│   └── ...                      # 19 more languages
└── README.md
```

---

## Setup

**1. Clone**
```bash
git clone https://github.com/yourusername/polarisation-classification.git
cd polarisation-classification
from google.colab import drive
drive.mount("/content/gdrive")
cd '/content/gdrive/My Drive/Colab Notebooks/On Device AI models for Child Safety'
```

**2. Install dependencies**
```bash
!pip install -r requirements.txt
!pip install pandas numpy requests tqdm scikit-learn python-dotenv
!pip install -q gepa requests scikit-learn tqdm pandas numpy
```

**3. Set API key**

Get a free key from [openrouter.ai](https://openrouter.ai)
```bash
export OPENROUTER_API_KEY="your-key-here"
```

**4. Add dataset**

Place SemEval-2026 Task 9 CSV files in `train/` named by language code. Each file must contain:
```
text, political, racial/ethnic, religious, gender/sexual, other
```

---

## Run

```bash
python baseline.py              # Stage 1
python gepa_optimize.py         # Stage 2
python miprov2_optimize.py      # Stage 3
python translate_multilingual.py # Stage 4
python evaluate.py              # Stage 5
```

---

## Web Application

The model is deployed as a REST API connected to a web interface for real-time polarisation detection.

```
User inputs text
      ↓
Frontend sends POST /classify
      ↓
API prompts Gemma 3 27B via OpenRouter
      ↓
Returns five binary labels as JSON
      ↓
App displays category breakdown
```

### Run Locally
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Docs at: `http://localhost:8000/docs`

### API Usage

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "your sentence here"}'
```

Response:
```json
{
  "political": 0,
  "racial_ethnic": 0,
  "religious": 1,
  "gender_sexual": 0,
  "other": 0
}
```

### Connect to Your App

**React / React Native:**
```javascript
const res = await fetch("https://your-app.onrender.com/classify", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text })
});
const labels = await res.json();
```

**Flutter:**
```dart
final response = await http.post(
  Uri.parse("https://your-app.onrender.com/classify"),
  headers: {"Content-Type": "application/json"},
  body: jsonEncode({"text": text}),
);
final labels = jsonDecode(response.body);
```

### Deploy Free (Render)
1. Push repo to GitHub
2. Connect to [render.com](https://render.com)
3. Add `OPENROUTER_API_KEY` as environment variable
4. Deploy, live URL in minutes

---

## Testing with Postman

Import `postman_collection.json` to get pre-built requests for all five label categories, multi-label cases, true negatives, and edge cases. Includes automated tests for response format, binary values, and response time. Two environments configured, local and production.

---

## Known Limitations

- 200-sample evaluation set, confidence intervals overlap for small gains
- Same LLM used for both translation and classification, errors are not decoupled
- MIPROv2 run in light mode (~25 trials), heavier search may improve results
- No fine-tuned transformer baselines (XLM-R, mDeBERTa) for comparison
- Requires active OpenRouter API key, not suitable for fully offline deployment

---

## Stack

DSPy · GEPA · MIPROv2 · Gemma 3 27B · OpenRouter · scikit-learn · FastAPI · Google Colab

---

## Citation

```bibtex
@inproceedings{yourname2026polarisation,
  title     = {Prompt Optimisation for Multi-Label Polarisation Classification},
  author    = {Your Name},
  booktitle = {Proceedings of SemEval-2026},
  year      = {2026}
}
```

---

## License

MIT
