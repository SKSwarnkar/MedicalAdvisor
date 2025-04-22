#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Cell 1 - Load OpenAI API key from .env
from dotenv import load_dotenv
import os

import openai
from openai import OpenAI
import re

import json
import streamlit as st


# In[ ]:


#GPT_MODEL="gpt-4"
#GPT_MODEL="gpt-3.5-turbo"
GPT_MODEL = "gpt-4o-mini"


# In[ ]:


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if OPENAI_API_KEY:
    print("✅ API key loaded successfully.")
else:
    print("❌ Failed to load API key. Please check your .env file.")

print("OpenAI library version:", openai.__version__)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID")
)



# In[ ]:


# Load API key and organization ID
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG_ID")

# Create a client
client = openai.Client()

# Define the symptoms that need guidelines
symptoms = [
    "Fever",
]

# Function to generate guideline using GPT-3.5 Turbo (client-based)
def generate_guideline(symptom):
    prompt = f"""
    For the symptom "{symptom}", provide:
    1. Clarifying questions (comma-separated)
    2. Red flag conditions (comma-separated)
    3. Triage options (Emergency, Doctor visit, Monitor at home)
    4. Safe advice for each triage level (what to do or when to seek care)

    Please return a structured response as plain text, separated by '---' for each section. For example:

    Clarifying Questions: [Question 1, Question 2]
    Conditions to Flag: [Condition 1, Condition 2]
    Triage Recommendations: [Emergency: Text, Doctor Visit: Text, Monitor at Home: Text]
    Safe Advice: [Text]

    Avoid any extra formatting or code blocks.
    """

    # Chat-based API call using client
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Correct way to access the content from the response object
    raw_response = response.choices[0].message.content

    # print(f"Raw response for {symptom}: {raw_response}")

    return raw_response

# Generate and save guidelines for "Fever" only
def generate_and_save_guidelines(symptoms):
    symptom_guidelines = []

    for symptom in symptoms:
        # print(f"\n\n\nGenerating guidelines for: {symptom}")
        raw_response = generate_guideline(symptom)

        # Extract information using regex or string operations
        clarifying_questions = re.findall(r'Clarifying Questions:\s*(.*?)\s*---', raw_response, re.DOTALL)
        conditions_to_flag = re.findall(r'Conditions to Flag:\s*(.*?)\s*---', raw_response, re.DOTALL)
        triage_recommendation = re.findall(r'Triage Recommendations:\s*(.*?)\s*---', raw_response, re.DOTALL)
        safe_advice = re.findall(r'Safe Advice:\s*(.*?)$', raw_response, re.DOTALL)

        # Clean and split the extracted data into individual items (if needed)
        clarifying_questions = clarifying_questions[0].split('\n') if clarifying_questions else []
        conditions_to_flag = conditions_to_flag[0].split('\n') if conditions_to_flag else []
        triage_recommendation = triage_recommendation[0].split('\n') if triage_recommendation else []
        safe_advice = safe_advice[0].split('\n') if safe_advice else []

        # You can now structure this data as needed
        guideline = {
            "symptom": symptom,
            "clarifying_questions": clarifying_questions,
            "conditions_to_flag": conditions_to_flag,
            "triage_recommendation": triage_recommendation,
            "safe_advice": safe_advice
        }

        symptom_guidelines.append(guideline)

    # Save to a text file or CSV
    with open("symptom_guidelines_fever.txt", "w") as f:
        f.write(str(symptom_guidelines))

# Run the function for "Fever" only
generate_and_save_guidelines(symptoms)


# In[ ]:


# Load the symptom guidelines JSON file
# with open('../data/combined_symptom_database.json', 'r') as file:
#     symptom_guidelines = json.load(file)
# # Print the loaded guidelines
# # print("Loaded symptom guidelines:")
# for guideline in symptom_guidelines:
#     print(guideline)


# In[ ]:


def get_symptom_guidelines(symptom):
    """
    Given a symptom, fetch its corresponding data (questions, conditions, etc.) from the symptom database.
    """
    symptom = symptom.lower()
    if symptom in symptom_guidelines:
        return symptom_guidelines[symptom]
    else:
        return None  # If symptom not found
    print(f"Guidelines for {symptom}:")
    print(f"Clarifying Questions: {guideline['clarifying_questions']}")
    print(f"Conditions to Flag: {guideline['conditions_to_flag']}")
    print(f"Triage Recommendations: {guideline['triage_recommendation']}")
    print(f"Safe Advice: {guideline['safe_advice']}")
# Example usage


# In[ ]:


symptom = "headache"  # Example input from the user
symptom_data = get_symptom_guidelines(symptom)

if symptom_data:
    prompt = f"""
    For the symptom "{symptom}", generate the following:
    1. Clarifying questions to ask the patient.
    2. Red flag conditions that need to be flagged.
    3. Triage options (Emergency, Doctor visit, Monitor at home).
    4. Safe advice for each triage level (what to do or when to seek care).

    Clarifying Questions: {symptom_data['clarifying_questions']}
    Conditions to Flag: {symptom_data['conditions_to_flag']}
    Triage Recommendations: {symptom_data['triage_recommendations']}
    Safe Advice: {symptom_data['safe_advice']}
    """

print("Prompt for OpenAI:")
print(prompt)


# In[ ]:


def generate_guideline_prompt(symptom):
    """
    Given a symptom, generate a structured GPT-4 prompt using the symptom's details.
    """
    symptom_data = get_symptom_guidelines(symptom)

    if symptom_data is None:
        return f"Sorry, we don't have guidelines for {symptom}."

    # Construct the prompt using the structured data
    prompt = f"""
    For the symptom "{symptom}", generate the following:
    1. Clarifying questions to ask the patient.
    2. Red flag conditions that need to be flagged.
    3. Triage options (Emergency, Doctor visit, Monitor at home).
    4. Safe advice for each triage level (what to do or when to seek care).

    Clarifying Questions: {symptom_data['clarifying_questions']}
    Conditions to Flag: {symptom_data['conditions_to_flag']}
    Triage Recommendations: {symptom_data['triage_recommendations']}
    Safe Advice: {symptom_data['safe_advice']}
    """
    return prompt


def display_result(symptom):
    prompt = generate_guideline_prompt(symptom)
    print(f"Generated Prompt: {prompt}")  # Debugging line
    if "Sorry" in prompt:
        st.write(prompt)
    else:
        gpt_response = get_gpt_response(prompt)
        print(f"GPT Response: {gpt_response}")  # Debugging line
        if gpt_response:
            st.write(gpt_response)
        else:
            st.write("No response generated.")


# import streamlit as st
# # Streamlit UI setup
# st.title("Medical Symptom Advisor")
# symptom_input = st.text_input("Enter a symptom:", "")


# if symptom_input:
#     guidelines = generate_guidelines(symptom_input)
#     st.write(guidelines)

