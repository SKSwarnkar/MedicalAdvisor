import openai
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

GPT_MODEL = "gpt-4o-mini"

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID")
)


SYMPTOM_DB= "data/combined_symptom_database.json"

# Load symptom guidelines from a JSON file (ensure the file is in the same directory)
with open(SYMPTOM_DB, 'r') as file:
    symptom_guidelines = json.load(file)
    

def get_symptom_guidelines(symptom):
    """
    Given a symptom, fetch its corresponding data (questions, conditions, etc.) from the symptom database.
    """
    symptom = symptom.lower()
    if symptom in symptom_guidelines:
        return symptom_guidelines[symptom]
    else:
        return None  # If symptom not found

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

def get_gpt_response(prompt):
    """
    Get GPT-4 response for the given prompt.
    """

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    # return response['choices'][0]['message']['content']
    # Correct way to access the content from the response object
    raw_response = response.choices[0].message.content

    # print(f"Raw response for {symptom}: {raw_response}")

    return raw_response
