# advisor_enrichment.py
#
# Medical Symptom Advisor - Medical Knowledge Enrichment Module
#
# This script enriches symptom advice by dynamically connecting user input to external authoritative sources:
# OpenFDA for medication adverse event data and PubMed for recent medical research articles.
#
# Main Functionalities:
# - Maintain master lists of common medications and symptoms.
# - Extract medications and symptoms mentioned in AI-generated advice text.
# - Query OpenFDA API to retrieve adverse reaction reports for medications.
# - Query PubMed API to retrieve recent articles related to extracted symptoms.
#
# Core Components:
# - Text processing for keyword extraction (medications and symptoms).
# - REST API integration with OpenFDA and PubMed.
# - XML parsing to extract PubMed article titles and links.
#
# Libraries Used:
# - requests
# - xml.etree.ElementTree
#
# Author: Satish Swarnkar
# University of Texas at Austin


import requests
import xml.etree.ElementTree as ET

# --- Step 1: Build Master Lists ---
medications_master = [
    "acetaminophen", "ibuprofen", "migraine", "oxygen therapy", "decongestants",
    "antihistamines", "saline spray", "saline rinse", "cough suppressant",
    "expectorant", "fever reducer", "otc pain reliever", "warm compress", 
    "lubricating eye drops", "reflux treatment", "migraine preventive therapy"
]

symptoms_master = [
    "tension headache", "migraine", "cluster headache", "thunderclap headache",
    "rebound headache", "cough", "nasal congestion", "runny nose", "sneezing",
    "postnasal drip", "fever", "chills", "fatigue", "facial pain", "ear fullness",
    "blurred vision", "dizziness", "nausea", "vomiting", "neck stiffness", "confusion",
    "headache after bending", "headache after exertion", "valsalva maneuver headache",
    "stress", "screen fatigue", "posture"
]

# --- Step 2: Extract keywords ---
def extract_keywords(text):
    meds = []
    symptoms = []
    text_lower = text.lower()
    for med in medications_master:
        if med in text_lower:
            meds.append(med)
    for symptom in symptoms_master:
        if symptom in text_lower:
            symptoms.append(symptom)
    return meds, symptoms

# --- Step 3: OpenFDA search ---
def search_openfda_reactions(drug_name):
    url = "https://api.fda.gov/drug/event.json"
    params = {
        "search": f"patient.drug.medicinalproduct:{drug_name}",
        "limit": 3
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, params=params)
    
    reactions = []
    if response.ok:
        results = response.json().get('results', [])
        for event in results:
            reaction = event.get('patient', {}).get('reaction', [{}])[0].get('reactionmeddrapt', None)
            if reaction:
                reactions.append(reaction)
    return reactions

# --- Step 4: PubMed search ---
def search_pubmed_ids(symptom):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": symptom,
        "retmode": "json",
        "retmax": 3
    }
    response = requests.get(url, params=params)
    if response.ok:
        return response.json().get('esearchresult', {}).get('idlist', [])
    return []

def fetch_pubmed_titles(pubmed_ids):
    if not pubmed_ids:
        return []
    
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pubmed_ids),
        "retmode": "xml"
    }
    response = requests.get(fetch_url, params=params)
    
    articles = []
    if response.ok:
        root = ET.fromstring(response.content)
        for article in root.findall(".//PubmedArticle"):
            title = article.findtext(".//ArticleTitle")
            pmid = article.findtext(".//MedlineCitation/PMID")
            if title and pmid:
                articles.append((title, f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"))
    return articles

# --- Step 5: Main pipeline ---
# def medical_advisor_enrichment(advisor_output):
#     meds, symptoms = extract_keywords(advisor_output)
    
#     print("\n🧪 Detected Medications:", meds)
#     print("🩺 Detected Symptoms:", symptoms)
    
#     for med in meds:
#         print(f"\n🔎 OpenFDA Search for '{med}'...")
#         reactions = search_openfda_reactions(med)
#         if reactions:
#             for r in reactions:
#                 print(f"  - Reaction reported: {r}")
#         else:
#             print(f"  No adverse events found for {med}.")

#     for symptom in symptoms:
#         print(f"\n🔎 PubMed Search for '{symptom}'...")
#         pubmed_ids = search_pubmed_ids(symptom)
#         articles = fetch_pubmed_titles(pubmed_ids)
#         if articles:
#             for title, link in articles:
#                 print(f"  - Title: {title}")
#                 print(f"    Link: {link}")
#         else:
#             print(f"  No PubMed articles found for '{symptom}'.")
import requests
import xml.etree.ElementTree as ET
import streamlit as st

# (keep same extract_keywords, search_openfda_reactions, search_pubmed_ids, fetch_pubmed_titles as before)

def medical_advisor_enrichment(advisor_output):
    meds, symptoms = extract_keywords(advisor_output)
    
    st.subheader("🔬 Advisor Enrichment: External Medical Data")
    st.info(f"**Detected Medications:** {', '.join(meds) if meds else 'None'}")
    st.info(f"**Detected Symptoms:** {', '.join(symptoms) if symptoms else 'None'}")
    
    if meds:
        st.subheader("💊 OpenFDA Drug Event Findings")
        for med in meds:
            reactions = search_openfda_reactions(med)
            if reactions:
                st.write(f"**{med.title()} Adverse Reactions:**")
                for r in reactions:
                    st.write(f"- {r}")
            else:
                st.write(f"No serious adverse reactions found for {med}.")

    if symptoms:
        st.subheader("📚 PubMed Research Articles")
        for symptom in symptoms:
            pubmed_ids = search_pubmed_ids(symptom)
            articles = fetch_pubmed_titles(pubmed_ids)
            if articles:
                st.write(f"**Research on {symptom.title()}:**")
                for title, link in articles:
                    st.markdown(f"- [{title}]({link})")
            else:
                st.write(f"No research articles found for {symptom}.")


