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
{
        "text": "Democrats are destroying this country with their socialist agenda.",
        "reasoning": "The sentence directly attacks a political party (Democrats) using emotionally charged language ('destroying') and labels their ideology negatively ('socialist agenda'). It expresses strong partisan bias designed to create political division. No racial, religious, gender, or other group is targeted.",
        "self_correction": "✅ Classification is correct. Political=1 because it attacks a political party with hostile language. No corrections needed.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
{
        "text": "Immigrants are stealing jobs from real Americans.",
        "reasoning": "This sentence promotes a harmful stereotype that immigrants take jobs from citizens using dehumanizing framing ('stealing'). It targets an ethnic/national group with hostile language. Initially I considered political=1 but self-correction shows this is primarily racial/ethnic — it attacks a group based on national origin, not a political party.",
        "self_correction": "⚠️ Corrected political from 1 to 0 because the sentence targets immigrants as an ethnic/national group, not a political party. racial_ethnic=1 is correct.",
        "political": 0, "racial_ethnic": 1, "religious": 0, "gender_sexual": 0, "other": 0
    },
 {
         "text": "Muslims are a threat to our Christian values and way of life.",
        "reasoning": "This sentence frames an entire religious group (Muslims) as a threat to another religious group (Christians). It promotes religious intolerance and division by portraying one faith as inherently dangerous to another. This is clear religious polarization.",
        "self_correction": "✅ Classification is correct. Religious=1 because it attacks Muslims and frames them as threatening. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 1, "gender_sexual": 0, "other": 0
    },
{
        "text": "Trans people are mentally ill and should not be near children.",
        "reasoning": "This sentence makes a medically false and dehumanizing claim about transgender people and implies they are dangerous to children. This is severe gender/sexual polarization through harmful stereotyping that denies the humanity of a group.",
        "self_correction": "✅ Classification is correct. gender_sexual=1 because it attacks transgender people with false medical claims. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 1, "other": 0
    },
 {
        "text": "Boomers ruined the economy and now expect young people to be grateful.",
        "reasoning": "This sentence expresses strong generational hatred toward Baby Boomers, blaming an entire age group for economic problems. This is age-based polarization that does not fit into political, racial, religious, or gender categories.",
        "self_correction": "✅ Classification is correct. other=1 because it targets an age group (Boomers) with blame and hatred. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 1
    },
{
        "text": "The radical left is pushing their LGBT agenda on our kids in schools.",
        "reasoning": "This sentence combines political polarization ('radical left') with gender/sexual polarization (framing LGBTQ+ content as a dangerous 'agenda' targeting children). Both categories apply simultaneously. Self-correction confirms multi-label.",
        "self_correction": "⚠️ Initially classified as political=1 only. Corrected to also include gender_sexual=1 because the sentence frames LGBTQ+ people as dangerous to children — this is both political AND gender/sexual polarization.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 1, "other": 0
    },
{
        "text": "These foreign invaders are being protected by the liberal government.",
        "reasoning": "This sentence uses dehumanizing language for immigrants ('foreign invaders') combined with a political attack ('liberal government'). Both racial/ethnic and political polarization are present simultaneously.",
        "self_correction": "⚠️ Initially classified as racial_ethnic=1 only. Corrected to also include political=1 because the sentence attacks the liberal government. Multi-label: political=1 AND racial_ethnic=1.",
        "political": 1, "racial_ethnic": 1, "religious": 0, "gender_sexual": 0, "other": 0
    },
 {
        "text": "Jews control the media and the banks.",
        "reasoning": "This sentence promotes a well-known antisemitic conspiracy theory targeting Jewish people. It constitutes both racial/ethnic polarization (Jewish as ethnic group) and religious polarization (Jewish as religious group).",
        "self_correction": "⚠️ Initially classified as racial_ethnic=1 only. Corrected to also include religious=1 because Jewish identity encompasses both ethnicity and religion. Multi-label: racial_ethnic=1 AND religious=1.",
        "political": 0, "racial_ethnic": 1, "religious": 1, "gender_sexual": 0, "other": 0
    },
 {
        "text": "The government announced new infrastructure spending plans today.",
        "reasoning": "This is a neutral factual news statement. It mentions the government but expresses no partisan bias, no attacks on any group, and no polarizing language. All categories are 0.",
        "self_correction": "✅ Classification is correct. All zeros because the sentence is a neutral factual report with no polarizing language. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
{
        "text": "The election results will be announced tomorrow morning.",
        "reasoning": "This is a neutral factual statement about an upcoming election announcement. It mentions politics but contains no partisan bias, no attacks, and no polarizing language. Merely mentioning an election is NOT polarizing.",
        "self_correction": "✅ Classification is correct. political=0 because merely reporting on an election announcement is neutral. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
{
        "text": "Putin is a war criminal who should be tried at the ICC.",
        "reasoning": "This sentence expresses a strong political opinion about a world leader using charged language ('war criminal'). It refers to legal accountability for an individual political figure.",
        "self_correction": "✅ Classification is correct. political=1 due to strong charged opinion about a political figure. No racial, religious, gender, or other group targeted.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
 {
        "text": "Women should stay home and raise children instead of working.",
        "reasoning": "This sentence promotes a gender stereotype that limits women's roles to domestic duties and discourages their workforce participation. This is clear gender-based polarization through harmful stereotyping.",
        "self_correction": "✅ Classification is correct. gender_sexual=1 because it restricts women's roles through harmful stereotyping. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 1, "other": 0
    },
{
        "text": "The government is waging a war on Christianity.",
        "reasoning": "This sentence frames government actions as a direct attack on a specific religion (Christianity). It promotes religious victimhood narrative combined with political framing.",
        "self_correction": "⚠️ Initially classified as religious=1 only. Corrected to also include political=1 because the sentence attacks the government as an institution. Multi-label: political=1 AND religious=1.",
        "political": 1, "racial_ethnic": 0, "religious": 1, "gender_sexual": 0, "other": 0
    },
{
        "text": "Southerners are all racist and uneducated.",
        "reasoning": "This sentence makes a broad negative generalization about people from a geographic region (Southerners), attributing negative traits to an entire group. This is regional hatred falling under 'other'.",
        "self_correction": "✅ Classification is correct. other=1 because it targets a regional group with sweeping negative generalizations. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 1
    },
{
        "text": "Black lives matter and anyone who disagrees is a racist.",
        "reasoning": "This sentence starts with a racial equality statement but adds a polarizing accusation against those who disagree. The framing promotes racial division by labeling all disagreement as racism.",
        "self_correction": "✅ Classification is correct. racial_ethnic=1 because the sentence promotes racial division through accusatory framing. No corrections needed.",
        "political": 0, "racial_ethnic": 1, "religious": 0, "gender_sexual": 0, "other": 0
    },
{
        "text": "Atheists are immoral people who cannot be trusted.",
        "reasoning": "This sentence makes a broad negative moral judgment about all atheists. This constitutes religious polarization through discrimination against a non-religious group.",
        "self_correction": "✅ Classification is correct. religious=1 because it attacks atheists as a group defined by religious identity. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 1, "gender_sexual": 0, "other": 0
    },
{
        "text": "Israel is committing genocide against Palestinians.",
        "reasoning": "This sentence makes a strong accusation about a country's actions toward an ethnic group. It combines political framing with racial/ethnic content through highly charged language.",
        "self_correction": "⚠️ Initially classified as political=1 only. Corrected to also include racial_ethnic=1 because Palestinians are an ethnic group being discussed in a highly charged accusatory context. Multi-label confirmed.",
        "political": 1, "racial_ethnic": 1, "religious": 0, "gender_sexual": 0, "other": 0
    },{
        "text": "Gay marriage is an abomination against nature.",
        "reasoning": "This sentence expresses extreme hostility toward same-sex marriage using religiously charged language ('abomination'). It directly attacks LGBTQ+ relationships.",
        "self_correction": "✅ Classification is correct. gender_sexual=1 because it attacks same-sex relationships with dehumanizing language. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 1, "other": 0
    },
 {
        "text": "The rich elites control everything while the working class suffers.",
        "reasoning": "This sentence promotes class-based conflict between rich elites and the working class. It combines political overtones with class-based polarization.",
        "self_correction": "⚠️ Initially classified as political=1 only. Corrected to also include other=1 because the sentence promotes class-based division and hatred between economic groups. Multi-label confirmed.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 1
    },
 {
        "text": "Congress passed a new bill on healthcare funding.",
        "reasoning": "This is a neutral factual statement about legislative activity. It mentions government action but contains no partisan attacks, no group targeting, and no polarizing language.",
        "self_correction": "✅ Classification is correct. All zeros because this is neutral factual reporting. No corrections needed.",
        "political": 0, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 0
    },
{
        "text": "When the rich claim business bail outs it's capitalism, when the poor claim benefits it's socialism.",
        "reasoning": "Initially I considered this political=1 only, but self-correction reveals it also promotes class-based division (rich vs poor), making other=1 as well. The sentence frames economic systems as hypocritically unfair through strong contrast.",
        "self_correction": "⚠️ Corrected to add other=1 because the sentence promotes class-based division between rich and poor. Multi-label: political=1 AND other=1.",
        "political": 1, "racial_ethnic": 0, "religious": 0, "gender_sexual": 0, "other": 1
    },
 {
        "text": "felon musk felon trump IDF supporters",
        "reasoning": "This sentence combines political attacks (Trump, Musk labeled as felons) with reference to IDF (Israeli Defense Forces) which introduces racial/ethnic elements related to the Israeli-Palestinian conflict.",
        "self_correction": "⚠️ Initially classified as political=1 only. Corrected to add racial_ethnic=1 because IDF reference introduces ethnic/national group targeting. Multi-label: political=1 AND racial_ethnic=1.",
        "political": 1, "racial_ethnic": 1, "religious": 0, "gender_sexual": 0, "other": 0
    },
