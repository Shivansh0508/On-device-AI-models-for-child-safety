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

# FEW-SHOT EXAMPLES WITH REASONING + SELF-CORRECTION
FEW_SHOT_WITH_REASONING = [
{
        "text": "Democrats are destroying this country with their socialist agenda.",
        "reasoning": "The sentence directly attacks a political party (Democrats) using emotionally charged language ('destroying') and labels their ideology negatively ('socialist agenda'). It expresses strong partisan bias designed to create political division. No racial, religious, gender, or other group is targeted.",
        "self_correction": " Classification is correct. Political=1 because it attacks a political party with hostile language. No corrections needed.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
{
        "text": "Immigrants are stealing jobs from real Americans.",
        "reasoning": "This sentence promotes a harmful stereotype that immigrants take jobs from citizens using dehumanizing framing ('stealing'). It targets an ethnic/national group with hostile language. Initially I considered political=1 but self-correction shows this is primarily racial/ethnic — it attacks a group based on national origin, not a political party.",
        "self_correction": " Corrected political from 1 to 0 because the sentence targets immigrants as an ethnic/national group, not a political party. racial_ethnic=1 is correct.",
        "political": 0, "racial_ethnic": 1, "religious": 0, "gender_sexual": 0, "other": 0
    },
 {
         "text": "Muslims are a threat to our Christian values and way of life.",
        "reasoning": "This sentence frames an entire religious group (Muslims) as a threat to another religious group (Christians). It promotes religious intolerance and division by portraying one faith as inherently dangerous to another. This is clear religious polarization.",
        "self_correction": " Classification is correct. Religious=1 because it attacks Muslims and frames them as threatening. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 1, "gender_sexual": 0, "other": 0
    },
{
        "text": "Trans people are mentally ill and should not be near children.",
        "reasoning": "This sentence makes a medically false and dehumanizing claim about transgender people and implies they are dangerous to children. This is severe gender/sexual polarization through harmful stereotyping that denies the humanity of a group.",
        "self_correction": " Classification is correct. gender_sexual=1 because it attacks transgender people with false medical claims. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 1, "other": 0
    },
 {
        "text": "Boomers ruined the economy and now expect young people to be grateful.",
        "reasoning": "This sentence expresses strong generational hatred toward Baby Boomers, blaming an entire age group for economic problems. This is age-based polarization that does not fit into political, racial, religious, or gender categories.",
        "self_correction": " Classification is correct. other=1 because it targets an age group (Boomers) with blame and hatred. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 1
    },
{
        "text": "The radical left is pushing their LGBT agenda on our kids in schools.",
        "reasoning": "This sentence combines political polarization ('radical left') with gender/sexual polarization (framing LGBTQ+ content as a dangerous 'agenda' targeting children). Both categories apply simultaneously. Self-correction confirms multi-label.",
        "self_correction": " Initially classified as political=1 only. Corrected to also include gender_sexual=1 because the sentence frames LGBTQ+ people as dangerous to children — this is both political AND gender/sexual polarization.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 1, "other": 0
    },
{
        "text": "These foreign invaders are being protected by the liberal government.",
        "reasoning": "This sentence uses dehumanizing language for immigrants ('foreign invaders') combined with a political attack ('liberal government'). Both racial/ethnic and political polarization are present simultaneously.",
        "self_correction": " Initially classified as racial_ethnic=1 only. Corrected to also include political=1 because the sentence attacks the liberal government. Multi-label: political=1 AND racial_ethnic=1.",
        "political": 1, "racial_ethnic": 1, "religious": 0, "gender_sexual": 0, "other": 0
    },
 {
        "text": "Jews control the media and the banks.",
        "reasoning": "This sentence promotes a well-known antisemitic conspiracy theory targeting Jewish people. It constitutes both racial/ethnic polarization (Jewish as ethnic group) and religious polarization (Jewish as religious group).",
        "self_correction": " Initially classified as racial_ethnic=1 only. Corrected to also include religious=1 because Jewish identity encompasses both ethnicity and religion. Multi-label: racial_ethnic=1 AND religious=1.",
        "political": 0, "racial_ethnic": 1, "religious": 1, "gender_sexual": 0, "other": 0
    },
 {
        "text": "The government announced new infrastructure spending plans today.",
        "reasoning": "This is a neutral factual news statement. It mentions the government but expresses no partisan bias, no attacks on any group, and no polarizing language. All categories are 0.",
        "self_correction": " Classification is correct. All zeros because the sentence is a neutral factual report with no polarizing language. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
{
        "text": "The election results will be announced tomorrow morning.",
        "reasoning": "This is a neutral factual statement about an upcoming election announcement. It mentions politics but contains no partisan bias, no attacks, and no polarizing language. Merely mentioning an election is NOT polarizing.",
        "self_correction": " Classification is correct. political=0 because merely reporting on an election announcement is neutral. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
{
        "text": "Putin is a war criminal who should be tried at the ICC.",
        "reasoning": "This sentence expresses a strong political opinion about a world leader using charged language ('war criminal'). It refers to legal accountability for an individual political figure.",
        "self_correction": " Classification is correct. political=1 due to strong charged opinion about a political figure. No racial, religious, gender, or other group targeted.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
 {
        "text": "Women should stay home and raise children instead of working.",
        "reasoning": "This sentence promotes a gender stereotype that limits women's roles to domestic duties and discourages their workforce participation. This is clear gender-based polarization through harmful stereotyping.",
        "self_correction": " Classification is correct. gender_sexual=1 because it restricts women's roles through harmful stereotyping. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 1, "other": 0
    },
{
        "text": "The government is waging a war on Christianity.",
        "reasoning": "This sentence frames government actions as a direct attack on a specific religion (Christianity). It promotes religious victimhood narrative combined with political framing.",
        "self_correction": " Initially classified as religious=1 only. Corrected to also include political=1 because the sentence attacks the government as an institution. Multi-label: political=1 AND religious=1.",
        "political": 1, "racial_ethnic": 0, "religious": 1, "gender_sexual": 0, "other": 0
    },
{
        "text": "Southerners are all racist and uneducated.",
        "reasoning": "This sentence makes a broad negative generalization about people from a geographic region (Southerners), attributing negative traits to an entire group. This is regional hatred falling under 'other'.",
        "self_correction": " Classification is correct. other=1 because it targets a regional group with sweeping negative generalizations. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 1
    },
{
        "text": "Black lives matter and anyone who disagrees is a racist.",
        "reasoning": "This sentence starts with a racial equality statement but adds a polarizing accusation against those who disagree. The framing promotes racial division by labeling all disagreement as racism.",
        "self_correction": " Classification is correct. racial_ethnic=1 because the sentence promotes racial division through accusatory framing. No corrections needed.",
        "political": 0, "racial_ethnic": 1, "religious": 0, "gender_sexual": 0, "other": 0
    },
{
        "text": "Atheists are immoral people who cannot be trusted.",
        "reasoning": "This sentence makes a broad negative moral judgment about all atheists. This constitutes religious polarization through discrimination against a non-religious group.",
        "self_correction": " Classification is correct. religious=1 because it attacks atheists as a group defined by religious identity. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 1, "gender_sexual": 0, "other": 0
    },
{
        "text": "Israel is committing genocide against Palestinians.",
        "reasoning": "This sentence makes a strong accusation about a country's actions toward an ethnic group. It combines political framing with racial/ethnic content through highly charged language.",
        "self_correction": " Initially classified as political=1 only. Corrected to also include racial_ethnic=1 because Palestinians are an ethnic group being discussed in a highly charged accusatory context. Multi-label confirmed.",
        "political": 1, "racial_ethnic": 1, "religious": 0, "gender_sexual": 0, "other": 0
    },{
        "text": "Gay marriage is an abomination against nature.",
        "reasoning": "This sentence expresses extreme hostility toward same-sex marriage using religiously charged language ('abomination'). It directly attacks LGBTQ+ relationships.",
        "self_correction": " Classification is correct. gender_sexual=1 because it attacks same-sex relationships with dehumanizing language. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 1, "other": 0
    },
 {
        "text": "The rich elites control everything while the working class suffers.",
        "reasoning": "This sentence promotes class-based conflict between rich elites and the working class. It combines political overtones with class-based polarization.",
        "self_correction": " Initially classified as political=1 only. Corrected to also include other=1 because the sentence promotes class-based division and hatred between economic groups. Multi-label confirmed.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 1
    },
 {
        "text": "Congress passed a new bill on healthcare funding.",
        "reasoning": "This is a neutral factual statement about legislative activity. It mentions government action but contains no partisan attacks, no group targeting, and no polarizing language.",
        "self_correction": " Classification is correct. All zeros because this is neutral factual reporting. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
{
        "text": "When the rich claim business bail outs it's capitalism, when the poor claim benefits it's socialism.",
        "reasoning": "Initially I considered this political=1 only, but self-correction reveals it also promotes class-based division (rich vs poor), making other=1 as well. The sentence frames economic systems as hypocritically unfair through strong contrast.",
        "self_correction": " Corrected to add other=1 because the sentence promotes class-based division between rich and poor. Multi-label: political=1 AND other=1.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 1
    },
 {
        "text": "felon musk felon trump IDF supporters",
        "reasoning": "This sentence combines political attacks (Trump, Musk labeled as felons) with reference to IDF (Israeli Defense Forces) which introduces racial/ethnic elements related to the Israeli-Palestinian conflict.",
        "self_correction": " Initially classified as political=1 only. Corrected to add racial_ethnic=1 because IDF reference introduces ethnic/national group targeting. Multi-label: political=1 AND racial_ethnic=1.",
        "political": 1, "racial_ethnic": 1, "religious": 0, "gender_sexual": 0, "other": 0
    },
{
        "text": "So Russia commits war crimes, how does that justify Ukraine also committing war crimes?",
        "reasoning": "This sentence raises a moral question about war crimes by both sides of a conflict. It does not attack an ethnic group or promote hatred — it questions moral equivalence between countries.",
        "self_correction": " Classification is correct. political=1 for geopolitical framing. racial_ethnic=0 because it discusses countries/governments not ethnic groups with hatred. No corrections needed.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
    {
        "text": "You think Putin wants to talk? Come on, wake up people.",
        "reasoning": "This sentence expresses rhetorical frustration about geopolitics and Putin's intentions. It uses charged language but addresses people broadly rather than attacking a political group.",
        "self_correction": " Classification is correct. political=1 due to charged opinion about a political figure. No group is specifically attacked with hatred. No corrections needed.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
    {
        "text": "Old people should just retire and stop blocking progress.",
        "reasoning": "This sentence expresses ageist hostility toward elderly people, blaming them for blocking societal progress. This is age-based discrimination falling under 'other'.",
        "self_correction": "Classification is correct. other=1 because it targets elderly people with hostility and blame. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 1
    },
]

# LOAD DATA
df = pd.read_csv("train/eng.csv")

LABELS = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
df[LABELS] = df[LABELS].fillna(0)

# SAME 160 fixed samples as all pipelines
df_sample  = df.sample(160, random_state=42)

# Larger pool for GEPA
df_gepa    = df.sample(500, random_state=42)
gepa_train = df_gepa.iloc[:450]
gepa_val   = df_gepa.iloc[450:500]

print(f"GEPA Pool  : {len(gepa_train)}")
print(f"GEPA Val   : {len(gepa_val)}")
print(f"Final Eval : {len(df_sample)} samples")
print(f"Few-shot   : {len(FEW_SHOT_WITH_REASONING)} examples with reasoning + self-correction")

label_cols = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]

# API CALLS
def call_model(prompt, model, temperature=0.1, max_retries=5):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "SemEval-Pipeline5-GEPA"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 1024,
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except requests.exceptions.ReadTimeout:
            wait = 2 ** attempt
            print(f" Timeout attempt {attempt+1}/{max_retries}, retrying in {wait}s...")
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
            print(f"Connection error, retrying in {wait}s...")
            time.sleep(wait)

    print(f" All retries failed — returning default")
    return '{"political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0}'


def call_llama(prompt, temperature=0.1):
    return call_model(prompt, "meta-llama/llama-3.3-70b-instruct", temperature)


def call_haiku(prompt, temperature=0.7):
    return call_model(prompt, "anthropic/claude-3-haiku", temperature)

# BASELINE PROMPT
BASELINE_PROMPT = """Classify this sentence.
Labels: political, racial_ethnic, religious, gender_sexual, other.
Return 0 or 1 for each."""

# SYSTEM PROMPT
PIPELINE5_SYSTEM_PROMPT = """You are an expert linguist and content moderation specialist with deep knowledge of polarizing, divisive, and hate-inducing language in social media text.

Your task is to analyze a sentence and classify whether it contains polarizing language across FIVE categories. A sentence can belong to MULTIPLE categories simultaneously.

STEP 1 — UNDERSTAND THE SENTENCE
Ask yourself:
→ What is the MAIN TOPIC of this sentence?
→ What is the TONE? (neutral, angry, hateful, sarcastic, opinionated?)
→ Is any GROUP, RELIGION, ETHNICITY, GENDER, or CLASS being TARGETED?
→ Does the sentence ATTACK, DEMEAN, or STEREOTYPE any group?
→ Or does it merely REPORT or MENTION a topic factually?

STEP 2 — APPLY PRECISE DEFINITIONS
POLITICAL (political=1):
 CLASSIFY AS 1 IF:
   - Sentence attacks, vilifies, or demonizes a political party, leader, or ideology
   - Sentence uses emotionally charged language to promote political division
   - Sentence frames one political group as evil, dangerous, or subhuman
   - Sentence promotes extreme partisan hostility or calls for political conflict
 CLASSIFY AS 0 IF:
   - Sentence reports on political events neutrally
   - Sentence discusses political topics without attacking any group
   - Sentence expresses mild preference without hostility

RACIAL/ETHNIC (racial_ethnic=1):
 CLASSIFY AS 1 IF:
   - Sentence promotes stereotypes, prejudice, or hatred toward a racial/ethnic group
   - Sentence uses dehumanizing language about race, ethnicity, or nationality
   - Sentence promotes white supremacy, antisemitism, Islamophobia, or xenophobia
   - Sentence frames an ethnic group as inherently criminal, dangerous, or inferior
 CLASSIFY AS 0 IF:
   - Sentence discusses racial issues factually without expressing hatred
   - Sentence mentions diversity or immigration without bias
   - Sentence advocates for racial equality without attacking any group

RELIGIOUS (religious=1):
 CLASSIFY AS 1 IF:
   - Sentence attacks, mocks, or expresses hatred toward a religious group
   - Sentence frames one religion as dangerous, evil, or inferior to another
   - Sentence promotes religious intolerance or calls for discrimination based on religion
   - Sentence uses conspiracy theories targeting a religious group
 CLASSIFY AS 0 IF:
   - Sentence reports religious news factually
   - Sentence discusses religious topics academically or respectfully
   - Sentence mentions religion without expressing bias or hatred

GENDER/SEXUAL (gender_sexual=1):
 CLASSIFY AS 1 IF:
   - Sentence expresses hatred, discrimination, or hostility toward women, men, or LGBTQ+
   - Sentence uses harmful gender stereotypes to demean or restrict a group
   - Sentence attacks transgender, gay, lesbian, or bisexual people
   - Sentence denies rights or humanity based on gender or sexuality
 CLASSIFY AS 0 IF:
   - Sentence discusses gender topics academically or in neutral context
   - Sentence reports LGBTQ+ news without expressing hostility
   - Sentence advocates for gender equality without attacking any group

OTHER (other=1):
 CLASSIFY AS 1 IF:
   - Sentence expresses hatred based on AGE (ageism)
   - Sentence promotes CLASS-BASED hatred (rich vs poor)
   - Sentence promotes REGIONAL hatred (attacking people from specific area)
   - Sentence dehumanizes a social group NOT covered by above categories
 CLASSIFY AS 0 IF:
   - Sentence discusses social or economic issues factually
   - Sentence expresses mild frustration without targeting a group
   - Sentence makes general observations without dehumanizing anyone

STEP 3 — MAKE YOUR INITIAL CLASSIFICATION
Based on Steps 1 and 2, assign 0 or 1 to each category.

STEP 4 — SELF-CORRECTION (CRITICAL STEP)
Before finalizing CHECK against these common mistakes:

 MISTAKE 1 — Over-predicting political:
   "Is this ACTUALLY attacking a political group or just mentioning politics?"
   → Neutral election/policy news → political=0

 MISTAKE 2 — Missing racial/ethnic:
   "Does this use words like 'invaders', 'stealing jobs', 'go back to your country'?"
   → These are racial/ethnic even if they seem political → racial_ethnic=1

 MISTAKE 3 — Missing religious:
   "Does this frame a religion as dangerous or evil?"
   → "Muslims are terrorists" = religious=1 even if it seems racial

 MISTAKE 4 — Missing gender/sexual:
   "Does this restrict, demean, or attack people based on gender or sexuality?"
   → "Women belong in the kitchen" = gender_sexual=1

 MISTAKE 5 — Over-predicting other:
   "Is this ACTUALLY targeting a group with hatred or just expressing frustration?"
   → General frustration without targeting = other=0

 MISTAKE 6 — Missing multi-label:
   "Could this belong to MORE THAN ONE category?"
   → "The liberal government protects foreign invaders" = political=1 AND racial_ethnic=1

After checking CORRECT any mistakes you identified and explain why.

STEP 5 — EXPLAIN YOUR FINAL DECISION
Explain in 2-3 sentences:
- WHY you assigned 1 to each positive category
- WHY you kept 0 for negative categories
- Whether you made any self-corrections and why

CRITICAL RULES:

 Merely MENTIONING a topic is NEVER polarizing
 The sentence must ACTIVELY EXPRESS bias, hatred, or strong negative stereotypes
 A sentence CAN belong to multiple categories simultaneously
 When uncertain ask: "Does this promote DIVISION or HATRED?" If yes → 1"""

# BUILD PIPELINE 5 PROMPT
def build_pipeline5_prompt(instruction, text, use_few_shot=True):
    few_shot_block = ""
    if use_few_shot:
        few_shot_block = "\n\n" + "━"*50 + "\n"
        few_shot_block += "EXAMPLES — Study the reasoning and self-correction process:\n"
        few_shot_block += "━"*50 + "\n"

        for i, ex in enumerate(FEW_SHOT_WITH_REASONING):
            few_shot_block += f"""
EXAMPLE {i+1}:
Text: "{ex['text']}"

Step 1 — Understanding: {ex['reasoning']}

Step 3 — Initial Classification:
political={ex['political']}, racial_ethnic={ex['racial_ethnic']}, religious={ex['religious']}, gender_sexual={ex['gender_sexual']}, other={ex['other']}

Step 4 — Self-Correction: {ex['self_correction']}

Step 5 — Final Explanation: {ex['reasoning']}

Final Classification: {{"political": {ex['political']}, "racial_ethnic": {ex['racial_ethnic']}, "religious": {ex['religious']}, "gender_sexual": {ex['gender_sexual']}, "other": {ex['other']}}}

{"─"*40}"""

        few_shot_block += "\n" + "━"*50
        few_shot_block += "\nNow apply ALL 5 STEPS to classify this sentence:\n"
        few_shot_block += "━"*50 + "\n"

    format_instruction = """

Your response MUST follow this EXACT format:

Step 1 — Understanding: [What is the sentence about? What is the tone? Who is targeted?]

Step 3 — Initial Classification:
political=[0/1], racial_ethnic=[0/1], religious=[0/1], gender_sexual=[0/1], other=[0/1]

Step 4 — Self-Correction:
[Check each label. Write " Correct" or " Corrected [label] from [old] to [new] because [reason]"]

Step 5 — Final Explanation:
[2-3 sentences explaining WHY each positive label was assigned and WHY negatives are 0]

Final Classification: {"political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0}"""

    return instruction + few_shot_block + f"""
Text to classify: "{text}"
{format_instruction}"""


# PARSER
def parse_with_reasoning(output):
    binary = {k: 0 for k in label_cols}
    try:
        # Primary: Final Classification keyword
        final_match = re.search(
            r'[Ff]inal\s+[Cc]lassification\s*:\s*(\{.*?\})',
            output, re.DOTALL
        )
        if final_match:
            parsed = json.loads(final_match.group(1))
            binary["political"]     = int(bool(parsed.get("political", 0)))
            binary["racial/ethnic"] = int(bool(parsed.get("racial_ethnic", 0)))
            binary["religious"]     = int(bool(parsed.get("religious", 0)))
            binary["gender/sexual"] = int(bool(parsed.get("gender_sexual", 0)))
            binary["other"]         = int(bool(parsed.get("other", 0)))
            return binary

        # Secondary: Classification keyword
        class_match = re.search(
            r'[Cc]lassification\s*:\s*(\{.*?\})',
            output, re.DOTALL
        )
        if class_match:
            parsed = json.loads(class_match.group(1))
            binary["political"]     = int(bool(parsed.get("political", 0)))
            binary["racial/ethnic"] = int(bool(parsed.get("racial_ethnic", 0)))
            binary["religious"]     = int(bool(parsed.get("religious", 0)))
            binary["gender/sexual"] = int(bool(parsed.get("gender_sexual", 0)))
            binary["other"]         = int(bool(parsed.get("other", 0)))
            return binary

        # Tertiary: last JSON in output (self-corrected)
        all_jsons = re.findall(r'\{[^{}]*\}', output, re.DOTALL)
        if all_jsons:
            parsed = json.loads(all_jsons[-1])
            binary["political"]     = int(bool(parsed.get("political", 0)))
            binary["racial/ethnic"] = int(bool(parsed.get("racial_ethnic", 0)))
            binary["religious"]     = int(bool(parsed.get("religious", 0)))
            binary["gender/sexual"] = int(bool(parsed.get("gender_sexual", 0)))
            binary["other"]         = int(bool(parsed.get("other", 0)))
            return binary

    except Exception as e:
        print(f" Parse error: {e} | Output: {output[:200]}")

    return binary

# EVALUATE PROMPT
def evaluate_prompt_p5(instruction, dataframe, use_few_shot=True, desc="Evaluating"):
    y_true_all, y_pred_all, texts = [], [], []

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=desc):
        prompt = build_pipeline5_prompt(instruction, row["text"], use_few_shot)
        raw    = call_llama(prompt, temperature=0.1)
        pred   = parse_with_reasoning(raw)

        y_true_all.append([
            int(row["political"]),
            int(row["racial/ethnic"]),
            int(row["religious"]),
            int(row["gender/sexual"]),
            int(row["other"]),
        ])
        y_pred_all.append([
            pred["political"],
            pred["racial/ethnic"],
            pred["religious"],
            pred["gender/sexual"],
            pred["other"],
        ])
        texts.append(row["text"])
        time.sleep(0.3)

    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=1)
    return macro_f1, y_true_all, y_pred_all, texts

# GEPA REFLECTION
def gepa_reflect_p5(current_prompt, failure_cases, current_score, iteration):
    failure_text = ""
    fp_count, fn_count, wrong_cat = 0, 0, 0

    for i, (text, true, pred) in enumerate(failure_cases[:12]):
        if sum(pred) > sum(true):
            error_type = "FALSE POSITIVE — model over-predicted"
            fp_count += 1
        elif sum(pred) < sum(true):
            error_type = "FALSE NEGATIVE — model under-predicted"
            fn_count += 1
        else:
            error_type = "WRONG CATEGORY — right count, wrong labels"
            wrong_cat += 1

        failure_text += f"""
Example {i+1} [{error_type}]:
  Text      : {text}
  True      : political={true[0]}, racial_ethnic={true[1]}, religious={true[2]}, gender_sexual={true[3]}, other={true[4]}
  Predicted : political={pred[0]}, racial_ethnic={pred[1]}, religious={pred[2]}, gender_sexual={pred[3]}, other={pred[4]}
"""

    reflection_prompt = f"""You are an expert NLP prompt engineer specializing in hate speech and polarization detection.

You are optimizing a Chain-of-Thought prompt with self-correction for LLaMA 3.3 70B.

CURRENT PROMPT (Iteration {iteration}):
===
{current_prompt}
===

CURRENT MACRO F1: {current_score:.4f} ({current_score*100:.2f}%)
False Positives : {fp_count} (over-predicting — model assigns 1 when true is 0)
False Negatives : {fn_count} (under-predicting — model assigns 0 when true is 1)
Wrong Categories: {wrong_cat} (right count, wrong category assigned)

FAILURE CASES:
{failure_text}

YOUR TASK:
1. Analyze the reasoning failures — is the model's Step 4 self-correction catching errors?
2. Are the definitions in Step 2 precise enough?
3. Is the model confusing categories?
4. Are the MISTAKE checks in Step 4 targeting the right error patterns?
5. Rewrite the prompt to fix these specific failures

REQUIREMENTS:
- Keep the STEP 1 / STEP 2 / STEP 3 / STEP 4 / STEP 5 structure
- Make definitions MORE precise for failing categories
- Update MISTAKE checks in Step 4 to target the actual errors you see
- Add CONFUSION AVOIDANCE rules if model is mixing categories
- If too many FP: make definitions STRICTER
- If too many FN: make definitions BROADER and more sensitive
- Keep the  YES if /  NO if format
- Do NOT include few-shot examples (added separately)
- Do NOT include JSON format (added separately)
- Do NOT include sentence placeholder (added separately)

Return ONLY the new improved prompt. No preamble."""

    new_prompt = call_haiku(reflection_prompt, temperature=0.7)
    return new_prompt.strip()

# GEPA MAIN LOOP
def run_gepa_p5(initial_prompt, iterations=5, sample_size=30):
    print("\n" + "="*60)
    print(" PIPELINE 5 — GEPA + CHAIN-OF-THOUGHT + SELF-CORRECTION")
    print(f"Task model      : LLaMA 3.3 70B")
    print(f"Reflection model: Claude Haiku")
    print(f"Iterations      : {iterations}")
    print(f"Few-shot        : {len(FEW_SHOT_WITH_REASONING)} examples with reasoning")
    print(f"Starting from   : PIPELINE5_SYSTEM_PROMPT")
    print("="*60)

    current_prompt = initial_prompt
    best_prompt    = initial_prompt
    best_score     = 0.0
    history        = []

    reflection_df  = gepa_train.sample(sample_size, random_state=SEED)

    for iteration in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f" GEPA Iteration {iteration}/{iterations}")
        print(f"{'='*60}")

        print(f" LLaMA evaluating with CoT + Self-Correction...")
        score, y_true_all, y_pred_all, texts = evaluate_prompt_p5(
            current_prompt,
            reflection_df,
            use_few_shot=True,
            desc=f"P5 iter {iteration}"
        )
        print(f"Macro F1: {score:.4f} ({score*100:.2f}%)")

        if score > best_score:
            best_score  = score
            best_prompt = current_prompt
            print(f" New best: {best_score*100:.2f}%")

        history.append({
            "iteration": iteration,
            "score": score,
            "prompt": current_prompt
        })

        failure_cases = [
            (texts[i], y_true_all[i], y_pred_all[i])
            for i in range(len(texts))
            if y_true_all[i] != y_pred_all[i]
        ]
        print(f" Failures: {len(failure_cases)}/{sample_size}")

        if len(failure_cases) == 0:
            print(" Perfect score! Stopping early.")
            break

        if iteration == iterations:
            break

        print(f" Claude Haiku reflecting on reasoning failures...")
        new_prompt = gepa_reflect_p5(current_prompt, failure_cases, score, iteration)
        print(f"New prompt (first 200 chars):\n{new_prompt[:200]}...")
        current_prompt = new_prompt

    print(f"\n GEPA Optimization History:")
    print(f"  {'Iteration':<12} {'Score':>10}")
    print(f"  {'-'*24}")
    for h in history:
        marker = " ← best" if h['score'] == best_score else ""
        print(f"  {h['iteration']:<12} {h['score']*100:>9.2f}%{marker}")
    print(f"  Best score: {best_score*100:.2f}%")

    return best_prompt, best_score, history

# RUN GEPA OPTIMIZATION
OPTIMIZED_PROMPT, gepa_best_score, gepa_history = run_gepa_p5(
    initial_prompt=PIPELINE5_SYSTEM_PROMPT,
    iterations=GEPA_ITERATIONS,
    sample_size=GEPA_SAMPLE,
)

print("\n" + "="*60)
print(" FINAL GEPA OPTIMIZED PROMPT ")
print("="*60)
print(OPTIMIZED_PROMPT)
print("="*60)

# FULL EVALUATION FUNCTION
def run_full_evaluation_p5(prompt_template, results_file,
                            label="", use_few_shot=False,
                            use_reasoning=False):
    if os.path.exists(results_file):
        print(f" Loading saved {label} results...")
        pred_df = pd.read_csv(results_file)
    else:
        print(f" Running {label} inference on 160 samples...")
        results = []

        for idx, (_, row) in enumerate(tqdm(
            df_sample.iterrows(), total=len(df_sample), desc=label
        )):
            if use_reasoning:
                prompt = build_pipeline5_prompt(
                    prompt_template, row["text"], use_few_shot
                )
            else:
                prompt = prompt_template + f"""

Sentence: {row['text']}

Return ONLY a JSON object:
{{"political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0}}"""

            raw  = call_llama(prompt, temperature=0.1)
            pred = parse_with_reasoning(raw)

            results.append({
                "pred_political":     pred["political"],
                "pred_racial/ethnic": pred["racial/ethnic"],
                "pred_religious":     pred["religious"],
                "pred_gender/sexual": pred["gender/sexual"],
                "pred_other":         pred["other"],

                "true_political":     int(row["political"]),
                "true_racial/ethnic": int(row["racial/ethnic"]),
                "true_religious":     int(row["religious"]),
                "true_gender/sexual": int(row["gender/sexual"]),
                "true_other":         int(row["other"]),
            })
            time.sleep(0.3)

            # Partial save every 20
            if (idx + 1) % 20 == 0:
                pd.DataFrame(results).to_csv(
                    results_file + ".partial", index=False
                )
                print(f" Partial save at {idx+1}/160")

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

# EVALUATE BASELINE
print("\n EVALUATING BASELINE PROMPT ON 160 SAMPLES ")
baseline_exact, baseline_f1 = run_full_evaluation_p5(
    BASELINE_PROMPT,
    results_file="baseline_results.csv",
    label="Baseline",
    use_few_shot=False,
    use_reasoning=False
)

# EVALUATE PIPELINE 5 GEPA
print("\n EVALUATING PIPELINE 5 GEPA ON 160 SAMPLES ")
if os.path.exists("p5_gepa_results.csv"):
    os.remove("p5_gepa_results.csv")
gepa_exact, gepa_f1 = run_full_evaluation_p5(
    OPTIMIZED_PROMPT,
    results_file="p5_gepa_results.csv",
    label="Pipeline 5 GEPA",
    use_few_shot=True,
    use_reasoning=True
)

# FINAL COMPARISON
PIPELINE1_BASELINE_F1 = 0.3975

print("\n" + "="*60)
print(" FINAL COMPARISON (sklearn Macro-F1 on 160 samples)")
print("="*60)
print(f"{'Metric':<30} {'Baseline (P1)':>14} {'P5 GEPA':>12}")
print("-"*60)
print(f"{'Exact-match Accuracy':<30} {'N/A':>14} {gepa_exact*100:>11.2f}%")
print(f"{'Macro-F1 (%)':<30} {PIPELINE1_BASELINE_F1*100:>13.2f}% {gepa_f1*100:>11.2f}%")
print("="*60)

improvement = (gepa_f1 - PIPELINE1_BASELINE_F1) * 100
if improvement > 0:
    print(f" Pipeline 5 improved Macro-F1 by +{improvement:.2f}% over baseline")
else:
    print(f" No improvement: {improvement:.2f}%")

# SAVE EVERYTHING
with open("pipeline5_optimized_prompt.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("PIPELINE 5 — GEPA + CoT + SELF-CORRECTION\n")
    f.write("Task Model      : meta-llama/llama-3.3-70b-instruct\n")
    f.write("Reflection Model: anthropic/claude-3-haiku\n")
    f.write("GEPA Iterations : 5\n")
    f.write(f"Few-Shot        : {len(FEW_SHOT_WITH_REASONING)} with reasoning\n")
    f.write("="*60 + "\n\n")
    f.write("BASELINE PROMPT:\n")
    f.write("-"*40 + "\n")
    f.write(BASELINE_PROMPT)
    f.write("\n\n")
    f.write("PIPELINE 5 STARTING PROMPT:\n")
    f.write("-"*40 + "\n")
    f.write(PIPELINE5_SYSTEM_PROMPT)
    f.write("\n\n")
    f.write("GEPA OPTIMIZED PROMPT:\n")
    f.write("-"*40 + "\n")
    f.write(OPTIMIZED_PROMPT)
    f.write("\n\n")
    f.write("="*60 + "\n")
    f.write("GEPA HISTORY\n")
    f.write("="*60 + "\n")
    for h in gepa_history:
        f.write(f"Iteration {h['iteration']}: {h['score']*100:.2f}%\n")
    f.write("\n")
    f.write("="*60 + "\n")
    f.write("FINAL PERFORMANCE\n")
    f.write("="*60 + "\n")
    f.write(f"Baseline Macro-F1  : {PIPELINE1_BASELINE_F1*100:.2f}%\n")
    f.write(f"Pipeline 5 Macro-F1: {gepa_f1*100:.2f}%\n")
    f.write(f"Improvement       : {improvement:+.2f}%\n")

print("Saved to: pipeline5_optimized_prompt.txt")
